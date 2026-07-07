use crate::decode::Decoder;
use crate::model::CorpipeRuntime;
use crate::{AnalysisResult, AnalyzerConfig, MentionSpan, ResolvedMention, Token};
use anyhow::{Context, Result};
use candle_core::Tensor;
use std::path::Path;
use tokenizers::{AddedToken, Tokenizer};
use udpipe_rs::Model;

/// A loaded CorPipe analyzer that can be reused across many texts.
pub struct CorpipeAnalyzer {
    parser_model: Model,
    tokenizer: AnalyzerTokenizer,
    runtime: CorpipeRuntime,
    decoder: Decoder,
}

impl CorpipeAnalyzer {
    /// Load UDPipe, tokenizer, and neural model weights into memory.
    pub fn load(config: AnalyzerConfig) -> Result<Self> {
        let decoder = Decoder::load(&config.model_dir)?;
        let parser_model = Model::load(&config.udpipe_model).with_context(|| {
            format!(
                "failed to load UDPipe model {}",
                config.udpipe_model.display()
            )
        })?;
        let tokenizer = AnalyzerTokenizer::load(&config.tokenizer_json)?;
        let runtime = CorpipeRuntime::load(&config.model_dir, decoder.tag_count())?;

        Ok(Self {
            parser_model,
            tokenizer,
            runtime,
            decoder,
        })
    }

    /// Analyze raw text and return structured tokens, mentions, and entity links.
    pub fn analyze(&self, text: &str) -> Result<AnalysisResult> {
        let words = self
            .parser_model
            .parse(text)
            .with_context(|| "failed to parse text with UDPipe")?;
        let mut tokens: Vec<Token> = words.iter().map(Token::from_word).collect();

        if tokens.is_empty() {
            return Ok(AnalysisResult {
                text: text.to_string(),
                tokens,
                predicted_tags: Vec::new(),
                mentions: Vec::new(),
                resolved_mentions: Vec::new(),
            });
        }

        let model_input = self.tokenizer.build_input(&tokens)?;
        let embeddings = self.runtime.encode_input(&model_input.input_ids)?;
        let word_embeddings = self
            .runtime
            .gather_word_embeddings(&embeddings, &model_input.word_indices)?;
        let tag_logits = self.runtime.tag_logits(&word_embeddings)?;
        let logits_matrix = tag_logits.squeeze(0)?.to_vec2::<f32>()?;
        let valid_mask = vec![true; tokens.len()];
        let predicted_tag_ids = self.decoder.decode(&logits_matrix, &valid_mask);
        let mentions = self.decoder.mentions(&predicted_tag_ids);
        let resolved_mentions =
            self.resolve_mentions(&embeddings, &model_input.word_indices, &mentions)?;

        self.decoder
            .annotate_tokens(&mut tokens, &resolved_mentions);

        Ok(AnalysisResult {
            text: text.to_string(),
            tokens,
            predicted_tags: self.decoder.predicted_tags(&predicted_tag_ids),
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

        Ok(self.decoder.resolve_antecedents(mentions, &scores))
    }
}

/// Convenience wrapper for one-shot analysis when you do not need a reusable instance.
pub fn analyze_text(config: AnalyzerConfig, text: &str) -> Result<AnalysisResult> {
    CorpipeAnalyzer::load(config)?.analyze(text)
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
        let mut subwords = Vec::new();
        let mut word_indices = Vec::with_capacity(tokens.len() + 1);

        for token in tokens {
            word_indices.push(subwords.len());

            let encoding = self
                .tokenizer
                .encode(token.form.as_str(), false)
                .map_err(|error| anyhow::anyhow!("failed to tokenize {:?}: {error}", token.form))?;

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

struct SpecialTokenIds {
    cls: u32,
    sep: u32,
}
