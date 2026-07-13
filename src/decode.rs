use crate::{MentionSpan, ResolvedMention, Token};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::{fs, path::Path};

const DEFAULT_DEPTH: usize = 5;

#[derive(Debug)]
pub(crate) struct Decoder {
    depth: usize,
    tags: Vec<String>,
    predecessor_states: Vec<Vec<usize>>,
    can_start_in_state: Vec<bool>,
    can_end_to_boundary: Vec<bool>,
}

#[derive(Debug, Deserialize)]
struct Options {
    #[serde(default)]
    depth: Option<usize>,
}

impl Decoder {
    pub(crate) fn load(model_dir: &Path) -> Result<Self> {
        let options_path = model_dir.join("options.json");
        let tags_path = model_dir.join("tags.txt");

        let options_text = fs::read_to_string(&options_path)
            .with_context(|| format!("failed to read {}", options_path.display()))?;
        let options: Options = serde_json::from_str(&options_text)
            .with_context(|| format!("failed to parse {}", options_path.display()))?;

        let tags_text = fs::read_to_string(&tags_path)
            .with_context(|| format!("failed to read {}", tags_path.display()))?;
        let tags: Vec<String> = tags_text.lines().map(str::to_owned).collect();

        anyhow::ensure!(!tags.is_empty(), "tags.txt is empty");
        anyhow::ensure!(
            tags[0].is_empty(),
            "expected first tag to be empty boundary tag"
        );

        let depth = options.depth.unwrap_or(DEFAULT_DEPTH);
        anyhow::ensure!(depth > 0, "decoder depth must be greater than zero");

        Ok(Self {
            depth,
            predecessor_states: Self::build_predecessor_states(&tags, depth),
            can_start_in_state: Self::build_start_states(&tags, depth),
            can_end_to_boundary: Self::build_end_states(&tags, depth),
            tags,
        })
    }

    pub(crate) fn tag_count(&self) -> usize {
        self.tags.len()
    }

    pub(crate) fn decode(&self, logits: &[Vec<f32>], valid_mask: &[bool]) -> Vec<usize> {
        let num_words = logits.len();
        if num_words == 0 {
            return Vec::new();
        }

        let num_tags = self.tags.len();
        let num_states = num_tags * self.depth;
        let neg_inf = -1.0e9_f32;

        assert_eq!(valid_mask.len(), num_words);
        assert_eq!(self.predecessor_states.len(), num_states);

        let mut alpha = vec![vec![neg_inf; num_states]; num_words];
        let mut beta = vec![vec![0usize; num_states]; num_words];

        for t in 0..num_words {
            for state in 0..num_states {
                let tag = state % num_tags;
                let mut score = logits[t][tag];

                if !valid_mask[t] && state >= 1 {
                    score = neg_inf;
                }

                if t == 0 && !self.can_start_in_state[state] {
                    score = neg_inf;
                }

                if t == num_words - 1 && !self.can_end_to_boundary[state] {
                    score = neg_inf;
                }

                if t == 0 {
                    alpha[t][state] = score;
                    continue;
                }

                let mut best_prev = 0usize;
                let mut best_score = neg_inf;

                for &prev in &self.predecessor_states[state] {
                    let candidate = alpha[t - 1][prev];
                    if candidate > best_score {
                        best_score = candidate;
                        best_prev = prev;
                    }
                }

                alpha[t][state] = score + best_score;
                beta[t][state] = best_prev;
            }
        }

        let mut states = vec![0usize; num_words];
        states[num_words - 1] = alpha[num_words - 1]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        for t in (0..num_words - 1).rev() {
            states[t] = beta[t + 1][states[t + 1]];
        }

        states.into_iter().map(|state| state % num_tags).collect()
    }

    pub(crate) fn predicted_tags(&self, predicted_tag_ids: &[usize]) -> Vec<String> {
        predicted_tag_ids
            .iter()
            .map(|&tag_id| self.tags[tag_id].clone())
            .collect()
    }

    pub(crate) fn mentions(&self, predicted_tag_ids: &[usize]) -> Vec<MentionSpan> {
        let mut mentions = Vec::new();
        let mut stack = Vec::new();

        for (index, tag_id) in predicted_tag_ids.iter().copied().enumerate() {
            let tag = &self.tags[tag_id % self.tags.len()];
            if tag.is_empty() {
                continue;
            }

            for command in tag.split(',') {
                if command == "PUSH" {
                    stack.push(index);
                } else if let Some(rest) = command.strip_prefix("POP:") {
                    if stack.is_empty() {
                        continue;
                    }

                    let requested = rest.parse::<usize>().unwrap_or(1);
                    let from_top = if requested <= stack.len() {
                        requested
                    } else {
                        1
                    };
                    let start = stack.remove(stack.len() - from_top);
                    mentions.push(MentionSpan { start, end: index });
                } else if !command.is_empty() {
                    panic!("unknown tag command: {command}");
                }
            }
        }

        while let Some(start) = stack.pop() {
            mentions.push(MentionSpan {
                start,
                end: predicted_tag_ids.len() - 1,
            });
        }

        mentions.sort_by_key(|mention| (mention.start, std::cmp::Reverse(mention.end)));
        mentions.dedup();
        mentions
    }

    pub(crate) fn resolve_antecedents(
        &self,
        mentions: &[MentionSpan],
        scores: &[Vec<f32>],
    ) -> Vec<ResolvedMention> {
        assert_eq!(mentions.len(), scores.len());

        let mut next_entity_id = 1usize;
        let mut resolved: Vec<ResolvedMention> = Vec::with_capacity(mentions.len());

        for (index, mention) in mentions.iter().enumerate() {
            assert!(
                scores[index].len() > index,
                "score row {index} must have at least {} entries",
                index + 1,
            );

            let mut best_antecedent = 0usize;
            let mut best_score = f32::NEG_INFINITY;

            for (candidate, &candidate_score) in scores[index].iter().take(index + 1).enumerate() {
                if candidate_score > best_score {
                    best_score = candidate_score;
                    best_antecedent = candidate;
                }
            }

            let entity_id = if best_antecedent == index {
                let entity_id = next_entity_id;
                next_entity_id += 1;
                entity_id
            } else {
                resolved[best_antecedent].entity_id
            };

            resolved.push(ResolvedMention {
                span: mention.clone(),
                entity_id,
            });
        }

        resolved
    }

    pub(crate) fn annotate_tokens(&self, tokens: &mut [Token], resolved: &[ResolvedMention]) {
        for mention in resolved {
            let entity = format!("c{}", mention.entity_id);

            if mention.span.start == mention.span.end {
                if let Some(token) = tokens.get_mut(mention.span.start) {
                    token.add_entity_marker(&format!("({entity}--1)"));
                }
                continue;
            }

            let span_len = mention.span.end - mention.span.start + 1;

            if let Some(token) = tokens.get_mut(mention.span.start) {
                token.add_entity_marker(&format!("({entity}--{span_len}"));
            }

            if let Some(token) = tokens.get_mut(mention.span.end) {
                token.add_entity_marker(&format!("{entity})"));
            }
        }
    }

    fn build_predecessor_states(tags: &[String], depth: usize) -> Vec<Vec<usize>> {
        let state_count = tags.len() * depth;
        let states_by_depth = Self::states_by_depth(tags.len(), depth);
        let mut predecessors = vec![Vec::new(); state_count];

        for previous_state in 0..state_count {
            let previous_depth = previous_state / tags.len();
            let previous_tag = &tags[previous_state % tags.len()];
            let next_depth = Self::apply_tag_to_depth(previous_depth as isize, previous_tag);

            if let Ok(next_depth) = usize::try_from(next_depth)
                && let Some(states) = states_by_depth.get(next_depth)
            {
                for &next_state in states {
                    predecessors[next_state].push(previous_state);
                }
            }
        }

        predecessors
    }

    fn build_start_states(tags: &[String], depth: usize) -> Vec<bool> {
        let state_count = tags.len() * depth;
        (0..state_count)
            .map(|state| state / tags.len() == 0)
            .collect()
    }

    fn build_end_states(tags: &[String], depth: usize) -> Vec<bool> {
        let state_count = tags.len() * depth;
        (0..state_count)
            .map(|state| {
                let state_depth = state / tags.len();
                let state_tag = &tags[state % tags.len()];
                Self::apply_tag_to_depth(state_depth as isize, state_tag) == 0
            })
            .collect()
    }

    fn states_by_depth(num_tags: usize, depth: usize) -> Vec<Vec<usize>> {
        let mut states_by_depth = vec![Vec::with_capacity(num_tags); depth];
        for state in 0..(num_tags * depth) {
            states_by_depth[state / num_tags].push(state);
        }
        states_by_depth
    }

    fn apply_tag_to_depth(mut depth: isize, tag: &str) -> isize {
        if tag.is_empty() {
            return depth;
        }

        for command in tag.split(',') {
            if command == "PUSH" && depth >= 0 {
                depth += 1;
            } else if !command.is_empty() {
                depth -= 1;
            }
        }

        depth
    }
}

#[cfg(test)]
mod tests {
    use super::Decoder;
    use crate::{MentionSpan, ResolvedMention, Token};

    struct DecoderTestFixtures;

    impl DecoderTestFixtures {
        fn token() -> Token {
            Token {
                sentence_index: 0,
                id: 1,
                form: "Alice".to_string(),
                lemma: "Alice".to_string(),
                upos: "PROPN".to_string(),
                xpos: "NNP".to_string(),
                feats: "_".to_string(),
                head: 0,
                deprel: "root".to_string(),
                deps: "_".to_string(),
                misc: "_".to_string(),
            }
        }

        fn decoder(tags: Vec<String>, depth: usize) -> Decoder {
            Decoder {
                predecessor_states: Decoder::build_predecessor_states(&tags, depth),
                can_start_in_state: Decoder::build_start_states(&tags, depth),
                can_end_to_boundary: Decoder::build_end_states(&tags, depth),
                depth,
                tags,
            }
        }
    }

    #[test]
    fn decoder_recovers_synthetic_mention_sequence() {
        let decoder = DecoderTestFixtures::decoder(
            vec!["".to_string(), "PUSH".to_string(), "POP:1".to_string()],
            5,
        );

        let mut logits = vec![vec![-5.0_f32; decoder.tag_count()]; 4];
        logits[0][0] = 5.0;
        logits[1][1] = 5.0;
        logits[2][0] = 5.0;
        logits[3][2] = 5.0;

        let predicted = decoder.decode(&logits, &[true, true, true, true]);
        let mentions = decoder.mentions(&predicted);

        assert_eq!(predicted, vec![0, 1, 0, 2]);
        assert_eq!(mentions, vec![MentionSpan { start: 1, end: 3 }]);
    }

    #[test]
    fn antecedent_resolution_reuses_prior_entity() {
        let decoder = DecoderTestFixtures::decoder(vec!["".to_string()], 5);
        let mentions = vec![
            MentionSpan { start: 0, end: 0 },
            MentionSpan { start: 2, end: 2 },
            MentionSpan { start: 5, end: 5 },
        ];
        let scores = vec![vec![10.0], vec![0.0, 10.0], vec![20.0, 1.0, 5.0]];

        let resolved = decoder.resolve_antecedents(&mentions, &scores);

        assert_eq!(
            resolved,
            vec![
                ResolvedMention {
                    span: MentionSpan { start: 0, end: 0 },
                    entity_id: 1,
                },
                ResolvedMention {
                    span: MentionSpan { start: 2, end: 2 },
                    entity_id: 2,
                },
                ResolvedMention {
                    span: MentionSpan { start: 5, end: 5 },
                    entity_id: 1,
                },
            ]
        );
    }

    #[test]
    fn entity_markers_are_written_in_place() {
        let decoder = DecoderTestFixtures::decoder(vec!["".to_string()], 5);
        let mut tokens = vec![
            DecoderTestFixtures::token(),
            DecoderTestFixtures::token(),
            DecoderTestFixtures::token(),
        ];
        let resolved = vec![
            ResolvedMention {
                span: MentionSpan { start: 0, end: 0 },
                entity_id: 1,
            },
            ResolvedMention {
                span: MentionSpan { start: 1, end: 2 },
                entity_id: 2,
            },
        ];

        decoder.annotate_tokens(&mut tokens, &resolved);

        assert_eq!(tokens[0].misc, "Entity=(c1--1)");
        assert_eq!(tokens[1].misc, "Entity=(c2--2");
        assert_eq!(tokens[2].misc, "Entity=c2)");
    }
}
