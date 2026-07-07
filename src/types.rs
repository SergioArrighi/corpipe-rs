use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Files needed to load a CorPipe analyzer instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    pub model_dir: PathBuf,
    pub udpipe_model: PathBuf,
    pub tokenizer_json: PathBuf,
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
    pub(crate) fn from_word(word: &udpipe_rs::Word) -> Self {
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
