use crate::decode::Decoder;
use crate::model::CorpipeRuntime;
use crate::{
    AnalysisResult, AnalyzerConfig, CoreferenceConfig, MentionSpan, ResolvedMention, Token,
    UdpipeDocument, UdpipeParser,
};
use anyhow::{Context, Result};
use candle_core::Tensor;
use std::path::Path;
use tokenizers::{AddedToken, Tokenizer};

/// A loaded CorPipe analyzer that can be reused across many texts.
pub struct CorpipeAnalyzer {
    parser: UdpipeParser,
    coreference: CoreferenceAnalyzer,
}

/// Loaded tokenizer and neural model that enrich an existing UDPipe document.
pub struct CoreferenceAnalyzer {
    tokenizer: AnalyzerTokenizer,
    runtime: CorpipeRuntime,
    decoder: Decoder,
}

impl CorpipeAnalyzer {
    /// Load UDPipe, tokenizer, and neural model weights into memory.
    pub fn load(config: AnalyzerConfig) -> Result<Self> {
        Ok(Self {
            parser: UdpipeParser::load(&config.udpipe_model)?,
            coreference: CoreferenceAnalyzer::load(CoreferenceConfig::from(&config))?,
        })
    }

    /// Load an analyzer for a single request and analyze the supplied text.
    pub fn analyze_text(config: AnalyzerConfig, text: &str) -> Result<AnalysisResult> {
        Self::load(config)?.analyze(text)
    }

    /// Parse text once and expose the reusable UDPipe document.
    pub fn parse_udpipe(&self, text: &str) -> Result<UdpipeDocument> {
        self.parser.parse(text)
    }

    /// Run CorPipe over previously produced UDPipe material without parsing again.
    pub fn analyze_udpipe(&self, document: &UdpipeDocument) -> Result<AnalysisResult> {
        self.coreference.analyze(document)
    }

    /// Analyze raw text and return structured tokens, mentions, and entity links.
    pub fn analyze(&self, text: &str) -> Result<AnalysisResult> {
        let document = self.parse_udpipe(text)?;
        self.analyze_udpipe(&document)
    }
}

impl CoreferenceAnalyzer {
    /// Load only tokenizer and CorPipe neural model assets.
    pub fn load(config: CoreferenceConfig) -> Result<Self> {
        let decoder = Decoder::load(&config.model_dir)?;
        let tokenizer = AnalyzerTokenizer::load(&config.tokenizer_json)?;
        let runtime = CorpipeRuntime::load(&config.model_dir, decoder.tag_count())?;
        Ok(Self {
            tokenizer,
            runtime,
            decoder,
        })
    }

    /// Enrich an immutable UDPipe document with mentions and coreference clusters.
    pub fn analyze(&self, document: &UdpipeDocument) -> Result<AnalysisResult> {
        document.validate()?;

        if document.tokens.is_empty() {
            return Ok(AnalysisResult {
                text: document.text.clone(),
                tokens: Vec::new(),
                predicted_tags: Vec::new(),
                mentions: Vec::new(),
                resolved_mentions: Vec::new(),
            });
        }

        let model_input = self.tokenizer.build_input(&document.tokens)?;
        let token_count = document.tokens.len();
        let analysis = self
            .runtime
            .with_context(|| self.run_runtime_phase(&model_input, token_count))?;

        let mut annotated_tokens = document.tokens.clone();
        self.decoder
            .annotate_tokens(&mut annotated_tokens, &analysis.resolved_mentions);

        Ok(AnalysisResult {
            text: document.text.clone(),
            tokens: annotated_tokens,
            predicted_tags: self.decoder.predicted_tags(&analysis.predicted_tag_ids),
            mentions: analysis.mentions,
            resolved_mentions: analysis.resolved_mentions,
        })
    }

    fn run_runtime_phase(
        &self,
        model_input: &ModelInput,
        token_count: usize,
    ) -> Result<CoreferenceOutput> {
        let embeddings = self.runtime.encode_input(&model_input.input_ids)?;
        let word_embeddings = self
            .runtime
            .gather_word_embeddings(&embeddings, &model_input.word_indices)?;
        let tag_logits = self.runtime.tag_logits(&word_embeddings)?;
        let logits_matrix = tag_logits.squeeze(0)?.to_vec2::<f32>()?;
        Self::validate_tag_logits(&logits_matrix, token_count, self.decoder.tag_count())?;
        let valid_mask = vec![true; token_count];
        let predicted_tag_ids = self.decoder.decode(&logits_matrix, &valid_mask);
        let mentions = self.decoder.mentions(&predicted_tag_ids);
        let resolved_mentions =
            self.resolve_mentions(&embeddings, &model_input.word_indices, &mentions)?;

        Ok(CoreferenceOutput {
            predicted_tag_ids,
            mentions,
            resolved_mentions,
        })
    }

    fn resolve_mentions(
        &self,
        embeddings: &Tensor,
        word_indices: &[usize],
        mentions: &[MentionSpan],
    ) -> Result<Vec<ResolvedMention>> {
        if mentions.is_empty() {
            return Ok(Vec::new());
        }

        let subword_mentions: Vec<(usize, usize)> = mentions
            .iter()
            .map(|mention| {
                (
                    word_indices[mention.start],
                    word_indices[mention.end + 1] - 1,
                )
            })
            .collect();

        let scores = self
            .runtime
            .antecedent_scores(embeddings, &subword_mentions, &subword_mentions)?
            .squeeze(0)?
            .to_vec2::<f32>()?;
        Self::validate_antecedent_scores(&scores, mentions.len())?;

        Ok(self.decoder.resolve_antecedents(mentions, &scores))
    }

    fn validate_tag_logits(
        logits: &[Vec<f32>],
        token_count: usize,
        tag_count: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            logits.len() == token_count,
            "CorPipe produced {} word logits for {token_count} UDPipe tokens",
            logits.len()
        );
        for (word_index, scores) in logits.iter().enumerate() {
            anyhow::ensure!(
                scores.len() == tag_count,
                "CorPipe word {word_index} produced {} tag logits; expected {tag_count}",
                scores.len()
            );
            anyhow::ensure!(
                scores.iter().all(|score| score.is_finite()),
                "CorPipe word {word_index} produced a non-finite tag score"
            );
        }
        Ok(())
    }

    fn validate_antecedent_scores(scores: &[Vec<f32>], mention_count: usize) -> Result<()> {
        anyhow::ensure!(
            scores.len() == mention_count,
            "CorPipe produced {} antecedent rows for {mention_count} mentions",
            scores.len()
        );
        for (mention_index, row) in scores.iter().enumerate() {
            anyhow::ensure!(
                row.len() > mention_index,
                "CorPipe antecedent row {mention_index} has {} entries; expected at least {}",
                row.len(),
                mention_index + 1
            );
            anyhow::ensure!(
                row[..=mention_index].iter().all(|score| score.is_finite()),
                "CorPipe antecedent row {mention_index} contains a non-finite candidate score"
            );
        }
        Ok(())
    }
}

struct AnalyzerTokenizer {
    tokenizer: Tokenizer,
    special_tokens: SpecialTokenIds,
}

impl AnalyzerTokenizer {
    fn load(tokenizer_json: &Path) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_json).map_err(|error| {
            anyhow::anyhow!(
                "failed to load tokenizer {}: {error}",
                tokenizer_json.display()
            )
        })?;

        tokenizer
            .add_special_tokens(vec![
                AddedToken::from("[TOKEN_EMPTY]", true),
                AddedToken::from("[TOKEN_CLS]", true),
            ])
            .map_err(|error| anyhow::anyhow!("failed to add special tokens: {error}"))?;

        let special_tokens = SpecialTokenIds {
            cls: tokenizer
                .token_to_id("[TOKEN_CLS]")
                .context("missing [TOKEN_CLS] after adding special token")?,
            sep: tokenizer
                .token_to_id("</s>")
                .context("missing </s> token")?,
        };

        Ok(Self {
            tokenizer,
            special_tokens,
        })
    }

    fn build_input(&self, tokens: &[Token]) -> Result<ModelInput> {
        let token_texts: Vec<&str> = tokens.iter().map(|token| token.form.as_str()).collect();
        let encodings = self
            .tokenizer
            .encode_batch_fast(token_texts, false)
            .map_err(|error| anyhow::anyhow!("failed to tokenize token batch: {error}"))?;

        let total_subwords: usize = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .sum();
        let mut subwords = Vec::with_capacity(total_subwords);
        let mut word_indices = Vec::with_capacity(tokens.len() + 1);

        for (token, encoding) in tokens.iter().zip(encodings.iter()) {
            word_indices.push(subwords.len());
            let ids = encoding.get_ids();
            anyhow::ensure!(
                !ids.is_empty(),
                "tokenizer produced no ids for {:?}",
                token.form
            );

            subwords.extend_from_slice(ids);
        }

        word_indices.push(subwords.len());

        let mut input_ids = Vec::with_capacity(subwords.len() + 4);
        input_ids.push(self.special_tokens.cls);
        input_ids.push(self.special_tokens.sep);
        input_ids.extend_from_slice(&subwords);
        input_ids.push(self.special_tokens.sep);
        input_ids.push(self.special_tokens.sep);

        Ok(ModelInput {
            input_ids,
            word_indices: word_indices.into_iter().map(|index| index + 2).collect(),
        })
    }
}

struct ModelInput {
    input_ids: Vec<u32>,
    word_indices: Vec<usize>,
}

struct CoreferenceOutput {
    predicted_tag_ids: Vec<usize>,
    mentions: Vec<MentionSpan>,
    resolved_mentions: Vec<ResolvedMention>,
}

struct SpecialTokenIds {
    cls: u32,
    sep: u32,
}

#[cfg(test)]
mod tests {
    use super::CoreferenceAnalyzer;

    #[test]
    fn neural_output_validation_rejects_shape_drift_and_non_finite_scores() {
        CoreferenceAnalyzer::validate_tag_logits(&[vec![0.0, 1.0]], 1, 2)
            .expect("valid tag logits");
        assert!(CoreferenceAnalyzer::validate_tag_logits(&[], 1, 2).is_err());
        assert!(CoreferenceAnalyzer::validate_tag_logits(&[vec![0.0]], 1, 2).is_err());
        assert!(CoreferenceAnalyzer::validate_tag_logits(&[vec![0.0, f32::NAN]], 1, 2).is_err());

        CoreferenceAnalyzer::validate_antecedent_scores(&[vec![0.0], vec![0.5, 0.6]], 2)
            .expect("valid antecedent scores");
        assert!(CoreferenceAnalyzer::validate_antecedent_scores(&[vec![0.0]], 2).is_err());
        assert!(CoreferenceAnalyzer::validate_antecedent_scores(&[vec![]], 1).is_err());
        assert!(
            CoreferenceAnalyzer::validate_antecedent_scores(&[vec![f32::INFINITY]], 1).is_err()
        );
    }
}
