use crate::{AnalysisResult, Token};
use std::fmt::Write;

impl AnalysisResult {
    /// Render the structured analysis as CorefUD-style CONLL-U text.
    pub fn to_conllu(&self) -> String {
        ConlluRenderer::new(self).render()
    }
}

struct ConlluRenderer<'a> {
    result: &'a AnalysisResult,
    output: String,
    current_sentence: Option<i32>,
}

impl<'a> ConlluRenderer<'a> {
    fn new(result: &'a AnalysisResult) -> Self {
        Self {
            result,
            output: String::new(),
            current_sentence: None,
        }
    }

    fn render(mut self) -> String {
        self.output.push_str("# newdoc\n");
        self.output
            .push_str("# global.Entity = eid-etype-head-other\n");

        for (index, token) in self.result.tokens.iter().enumerate() {
            if self.current_sentence != Some(token.sentence_index) {
                if self.current_sentence.is_some() {
                    self.output.push('\n');
                }

                self.current_sentence = Some(token.sentence_index);
                let sentence_tokens = self.sentence_tokens(index);
                self.write_sentence_header(token.sentence_index, sentence_tokens);
            }

            self.write_token_line(token);
        }

        if !self.result.tokens.is_empty() {
            self.output.push('\n');
        }

        self.output
    }

    fn sentence_tokens(&self, sentence_start: usize) -> &'a [Token] {
        let sentence_index = self.result.tokens[sentence_start].sentence_index;
        let mut sentence_end = sentence_start;

        while sentence_end < self.result.tokens.len()
            && self.result.tokens[sentence_end].sentence_index == sentence_index
        {
            sentence_end += 1;
        }

        &self.result.tokens[sentence_start..sentence_end]
    }

    fn write_sentence_header(&mut self, sentence_index: i32, tokens: &[Token]) {
        let _ = writeln!(self.output, "# sent_id = {}", sentence_index + 1);
        let _ = writeln!(self.output, "# text = {}", self.sentence_text(tokens));
    }

    fn sentence_text(&self, tokens: &[Token]) -> String {
        let mut sentence = String::new();

        for (index, token) in tokens.iter().enumerate() {
            sentence.push_str(&token.form);

            if index + 1 < tokens.len() && token.has_trailing_space() {
                sentence.push(' ');
            }
        }

        sentence
    }

    fn write_token_line(&mut self, token: &Token) {
        let _ = writeln!(
            self.output,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            token.id,
            token.form,
            token.lemma,
            token.upos,
            token.xpos,
            token.feats,
            token.head,
            token.deprel,
            token.deps,
            token.misc,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{AnalysisResult, MentionSpan, ResolvedMention, Token};

    #[test]
    fn renders_sentence_text_from_tokens() {
        let result = AnalysisResult {
            text: "Hello, world!".to_string(),
            tokens: vec![
                Token {
                    sentence_index: 0,
                    id: 1,
                    form: "Hello".to_string(),
                    lemma: "hello".to_string(),
                    upos: "INTJ".to_string(),
                    xpos: "UH".to_string(),
                    feats: "_".to_string(),
                    head: 0,
                    deprel: "root".to_string(),
                    deps: "_".to_string(),
                    misc: "SpaceAfter=No".to_string(),
                },
                Token {
                    sentence_index: 0,
                    id: 2,
                    form: ",".to_string(),
                    lemma: ",".to_string(),
                    upos: "PUNCT".to_string(),
                    xpos: ",".to_string(),
                    feats: "_".to_string(),
                    head: 1,
                    deprel: "punct".to_string(),
                    deps: "_".to_string(),
                    misc: "_".to_string(),
                },
                Token {
                    sentence_index: 0,
                    id: 3,
                    form: "world".to_string(),
                    lemma: "world".to_string(),
                    upos: "NOUN".to_string(),
                    xpos: "NN".to_string(),
                    feats: "_".to_string(),
                    head: 1,
                    deprel: "obj".to_string(),
                    deps: "_".to_string(),
                    misc: "SpaceAfter=No".to_string(),
                },
                Token {
                    sentence_index: 0,
                    id: 4,
                    form: "!".to_string(),
                    lemma: "!".to_string(),
                    upos: "PUNCT".to_string(),
                    xpos: ".".to_string(),
                    feats: "_".to_string(),
                    head: 1,
                    deprel: "punct".to_string(),
                    deps: "_".to_string(),
                    misc: "_".to_string(),
                },
            ],
            predicted_tags: Vec::new(),
            mentions: vec![MentionSpan { start: 2, end: 2 }],
            resolved_mentions: vec![ResolvedMention {
                span: MentionSpan { start: 2, end: 2 },
                entity_id: 1,
            }],
        };

        let conllu = result.to_conllu();
        assert!(conllu.contains("# text = Hello, world!"));
    }
}
