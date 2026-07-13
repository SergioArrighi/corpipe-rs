use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Files needed to load a CorPipe analyzer instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    pub model_dir: PathBuf,
    pub udpipe_model: PathBuf,
    pub tokenizer_json: PathBuf,
}

/// Files needed to load only the CorPipe neural coreference stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreferenceConfig {
    pub model_dir: PathBuf,
    pub tokenizer_json: PathBuf,
}

impl From<&AnalyzerConfig> for CoreferenceConfig {
    fn from(config: &AnalyzerConfig) -> Self {
        Self {
            model_dir: config.model_dir.clone(),
            tokenizer_json: config.tokenizer_json.clone(),
        }
    }
}

/// Immutable UDPipe output that can be exposed before coreference inference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UdpipeDocument {
    pub text: String,
    pub tokens: Vec<Token>,
}

impl UdpipeDocument {
    pub fn new(text: impl Into<String>, tokens: Vec<Token>) -> Self {
        Self {
            text: text.into(),
            tokens,
        }
    }

    /// Normalize words produced by `udpipe-rs` into a reusable CorPipe input.
    pub fn from_words(text: impl Into<String>, words: &[udpipe_rs::Word]) -> Self {
        Self::new(text, words.iter().map(Token::from_word).collect())
    }

    /// Validate the stable token identities required by CorPipe and CorefUD output.
    pub fn validate(&self) -> anyhow::Result<()> {
        let mut cursor = 0;
        let mut expected_sentence = 0;
        while cursor < self.tokens.len() {
            let sentence_index = self.tokens[cursor].sentence_index;
            anyhow::ensure!(
                sentence_index == expected_sentence,
                "expected sentence index {expected_sentence}, found {sentence_index}"
            );

            let sentence_start = cursor;
            while cursor < self.tokens.len() && self.tokens[cursor].sentence_index == sentence_index
            {
                let expected_id = i32::try_from(cursor - sentence_start + 1).map_err(|_| {
                    anyhow::anyhow!("sentence {sentence_index} has too many tokens")
                })?;
                let token = &self.tokens[cursor];
                anyhow::ensure!(
                    token.id == expected_id,
                    "sentence {sentence_index} expected token id {expected_id}, found {}",
                    token.id
                );
                anyhow::ensure!(
                    !token.form.is_empty(),
                    "sentence {sentence_index} token {} has an empty form",
                    token.id
                );
                anyhow::ensure!(
                    token.head >= 0,
                    "sentence {sentence_index} token {} has negative head {}",
                    token.id,
                    token.head
                );
                anyhow::ensure!(
                    token.head != token.id,
                    "sentence {sentence_index} token {} is its own head",
                    token.id
                );
                cursor += 1;
            }

            let sentence_token_count = i32::try_from(cursor - sentence_start)
                .map_err(|_| anyhow::anyhow!("sentence {sentence_index} has too many tokens"))?;
            for token in &self.tokens[sentence_start..cursor] {
                anyhow::ensure!(
                    token.head <= sentence_token_count,
                    "sentence {sentence_index} token {} references missing head {}",
                    token.id,
                    token.head
                );
            }
            expected_sentence += 1;
        }
        Ok(())
    }
}

/// Structured analysis result returned by the library API.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub text: String,
    pub tokens: Vec<Token>,
    pub predicted_tags: Vec<String>,
    pub mentions: Vec<MentionSpan>,
    pub resolved_mentions: Vec<ResolvedMention>,
}

/// A parsed token before it is rendered into CONLL-U text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// Zero-based sentence index as emitted by UDPipe.
    pub sentence_index: i32,
    pub id: i32,
    pub form: String,
    pub lemma: String,
    pub upos: String,
    pub xpos: String,
    pub feats: String,
    pub head: i32,
    pub deprel: String,
    pub deps: String,
    pub misc: String,
}

impl Token {
    pub fn from_word(word: &udpipe_rs::Word) -> Self {
        Self {
            sentence_index: word.sentence_id,
            id: word.id,
            form: word.form.clone(),
            lemma: word.lemma.clone(),
            upos: word.upostag.clone(),
            xpos: word.xpostag.clone(),
            feats: Self::normalized_field(&word.feats).to_string(),
            head: word.head,
            deprel: word.deprel.clone(),
            deps: "_".to_string(),
            misc: Self::normalized_field(&word.misc).to_string(),
        }
    }

    pub(crate) fn add_entity_marker(&mut self, marker: &str) {
        if self.misc == "_" || self.misc.is_empty() {
            self.misc = format!("Entity={marker}");
            return;
        }

        let mut parts: Vec<String> = self.misc.split('|').map(str::to_owned).collect();

        for part in &mut parts {
            if let Some(existing) = part.strip_prefix("Entity=") {
                *part = format!("Entity={existing}{marker}");
                self.misc = parts.join("|");
                return;
            }
        }

        parts.insert(0, format!("Entity={marker}"));
        self.misc = parts.join("|");
    }

    pub(crate) fn has_trailing_space(&self) -> bool {
        !self.misc.split('|').any(|part| part == "SpaceAfter=No")
    }

    fn normalized_field(value: &str) -> &str {
        if value.is_empty() { "_" } else { value }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MentionSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedMention {
    pub span: MentionSpan,
    pub entity_id: usize,
}

#[cfg(test)]
mod tests {
    use super::{Token, UdpipeDocument};

    struct UdpipeDocumentTestFixtures;

    impl UdpipeDocumentTestFixtures {
        fn token(id: i32, head: i32) -> Token {
            Token {
                sentence_index: 0,
                id,
                form: format!("t{id}"),
                lemma: format!("t{id}"),
                upos: "NOUN".into(),
                xpos: "NN".into(),
                feats: "_".into(),
                head,
                deprel: if head == 0 { "root" } else { "nmod" }.into(),
                deps: "_".into(),
                misc: "_".into(),
            }
        }
    }

    #[test]
    fn udpipe_document_accepts_well_formed_token_identities() {
        let document = UdpipeDocument::new(
            "t1 t2",
            vec![
                UdpipeDocumentTestFixtures::token(1, 0),
                UdpipeDocumentTestFixtures::token(2, 1),
            ],
        );
        document.validate().expect("valid UD document");
        let encoded = serde_json::to_string(&document).expect("serialize UD document");
        let decoded: UdpipeDocument =
            serde_json::from_str(&encoded).expect("deserialize UD document");
        assert_eq!(decoded, document);
    }

    #[test]
    fn udpipe_document_rejects_invalid_ids_and_heads() {
        let duplicate = UdpipeDocument::new(
            "t1 t1",
            vec![
                UdpipeDocumentTestFixtures::token(1, 0),
                UdpipeDocumentTestFixtures::token(1, 0),
            ],
        );
        assert!(duplicate.validate().is_err());

        let dangling = UdpipeDocument::new(
            "t1 t2",
            vec![
                UdpipeDocumentTestFixtures::token(1, 0),
                UdpipeDocumentTestFixtures::token(2, 3),
            ],
        );
        assert!(dangling.validate().is_err());

        let self_head = UdpipeDocument::new("t1", vec![UdpipeDocumentTestFixtures::token(1, 1)]);
        assert!(self_head.validate().is_err());
    }
}
