use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Module, VarBuilder};
use clap::{Parser, Subcommand};
use safetensors::SafeTensors;
use serde::Deserialize;
use std::{fs, path::PathBuf};
use tokenizers::{AddedToken, Tokenizer};

#[derive(Parser, Debug)]
#[command(name = "corpipe-rs")]
#[command(about = "Rust CorPipe experiment")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    InspectModel {
        /// Directory containing options.json, tags.txt, model.safetensors
        model_dir: PathBuf,
    },
    DecodeInfo {
        /// Directory containing options.json, tags.txt, model.safetensors
        model_dir: PathBuf,
    },
    DecodeSynthetic {
        model_dir: PathBuf,
    },
    ResolveSynthetic,
    SyntheticPipeline {
        model_dir: PathBuf,
    },
    ParseText {
        #[arg(long)]
        udpipe_model: PathBuf,

        text: String,
    },
    PredictTextMock {
        #[arg(long)]
        udpipe_model: PathBuf,

        text: String,
    },
    TokenizeText {
        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    HeadsSmoke {
        model_dir: PathBuf,
    },
    AntecedentSmoke {
        model_dir: PathBuf,
    },
    PredictTextFakeEncoder {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    EmbedSmoke {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    RmsNormSmoke {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    FfSmoke {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    AttentionSmoke {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    BlockSmoke {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    EncoderSmoke {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
    PredictTextRealEncoder {
        #[arg(long)]
        model_dir: PathBuf,

        #[arg(long)]
        udpipe_model: PathBuf,

        #[arg(long)]
        tokenizer_json: PathBuf,

        text: String,
    },
}

#[derive(Debug, Deserialize)]
struct Options {
    #[serde(default)]
    encoder: Option<String>,

    #[serde(default)]
    segment: Option<usize>,

    #[serde(default)]
    right: Option<usize>,

    #[serde(default)]
    depth: Option<usize>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::InspectModel { model_dir } => inspect_model(model_dir),
        Command::DecodeInfo { model_dir } => decode_info(model_dir),
        Command::DecodeSynthetic { model_dir } => decode_synthetic(model_dir),
        Command::ResolveSynthetic => resolve_synthetic(),
        Command::SyntheticPipeline { model_dir } => synthetic_pipeline(model_dir),
        Command::ParseText { udpipe_model, text } => parse_text(udpipe_model, text),
        Command::PredictTextMock { udpipe_model, text } => predict_text_mock(udpipe_model, text),
        Command::TokenizeText {
            udpipe_model,
            tokenizer_json,
            text,
        } => tokenize_text(udpipe_model, tokenizer_json, text),
        Command::HeadsSmoke { model_dir } => heads_smoke(model_dir),
        Command::AntecedentSmoke { model_dir } => antecedent_smoke(model_dir),
        Command::PredictTextFakeEncoder {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => predict_text_fake_encoder(model_dir, udpipe_model, tokenizer_json, text),
        Command::EmbedSmoke {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => embed_smoke(model_dir, udpipe_model, tokenizer_json, text),
        Command::RmsNormSmoke {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => rmsnorm_smoke(model_dir, udpipe_model, tokenizer_json, text),
        Command::FfSmoke {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => ff_smoke(model_dir, udpipe_model, tokenizer_json, text),
        Command::AttentionSmoke {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => attention_smoke(model_dir, udpipe_model, tokenizer_json, text),
        Command::BlockSmoke {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => block_smoke(model_dir, udpipe_model, tokenizer_json, text),
        Command::EncoderSmoke {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => encoder_smoke(model_dir, udpipe_model, tokenizer_json, text),
        Command::PredictTextRealEncoder {
            model_dir,
            udpipe_model,
            tokenizer_json,
            text,
        } => predict_text_real_encoder(model_dir, udpipe_model, tokenizer_json, text),
    }
}

fn inspect_model(model_dir: PathBuf) -> Result<()> {
    let weights_path = model_dir.join("model.safetensors");
    let (options, tags) = load_options_and_tags(&model_dir)?;

    let weights_bytes = fs::read(&weights_path)
        .with_context(|| format!("failed to read {}", weights_path.display()))?;

    let tensors = SafeTensors::deserialize(&weights_bytes)
        .with_context(|| format!("failed to parse {}", weights_path.display()))?;

    println!("model_dir: {}", model_dir.display());
    println!("encoder: {:?}", options.encoder);
    println!("segment: {:?}", options.segment);
    println!("right: {:?}", options.right);
    println!("depth: {:?}", options.depth);
    println!("tags: {}", tags.len());
    println!("tensors: {}", tensors.names().len());

    // rest of inspect_model stays the same...
    Ok(())
}

fn load_options_and_tags(model_dir: &std::path::Path) -> Result<(Options, Vec<String>)> {
    let options_path = model_dir.join("options.json");
    let tags_path = model_dir.join("tags.txt");

    let options_text = fs::read_to_string(&options_path)
        .with_context(|| format!("failed to read {}", options_path.display()))?;
    let options: Options = serde_json::from_str(&options_text)
        .with_context(|| format!("failed to parse {}", options_path.display()))?;

    let tags_text = fs::read_to_string(&tags_path)
        .with_context(|| format!("failed to read {}", tags_path.display()))?;
    let tags: Vec<String> = tags_text.lines().map(|s| s.to_string()).collect();

    Ok((options, tags))
}

fn apply_tag_to_depth(mut depth: isize, tag: &str) -> isize {
    if tag.is_empty() {
        return depth;
    }

    for command in tag.split(',') {
        if command == "PUSH" && depth >= 0 {
            depth += 1;
        } else if !command.is_empty() {
            // This covers POP:n and also PUSH when already invalid.
            // It intentionally mirrors the Python expression:
            // i_depth += 1 if command == "PUSH" and i_depth >= 0 else -1
            depth -= 1;
        }
    }

    depth
}

fn allowed_tag_transitions(tags: &[String], depth: usize) -> Vec<Vec<bool>> {
    let n = tags.len() * depth;
    let mut allowed = vec![vec![false; n]; n];

    for i in 0..n {
        let i_depth = i / tags.len();
        let i_tag = &tags[i % tags.len()];

        let after_i = apply_tag_to_depth(i_depth as isize, i_tag);

        for j in 0..n {
            let j_depth = j / tags.len();

            allowed[i][j] = after_i == j_depth as isize;
        }
    }

    allowed
}

fn decode_info(model_dir: PathBuf) -> Result<()> {
    let (options, tags) = load_options_and_tags(&model_dir)?;
    let depth = options.depth.unwrap_or(5);

    anyhow::ensure!(!tags.is_empty(), "tags.txt is empty");
    anyhow::ensure!(tags[0].is_empty(), "expected first tag to be empty boundary tag");

    let allowed = allowed_tag_transitions(&tags, depth);
    let states = tags.len() * depth;

    println!("tags: {}", tags.len());
    println!("depth: {}", depth);
    println!("states: {}", states);

    let allowed_count: usize = allowed
        .iter()
        .map(|row| row.iter().filter(|&&x| x).count())
        .sum();

    println!("allowed transitions: {}", allowed_count);
    println!("total transitions: {}", states * states);
    println!(
        "allowed ratio: {:.4}",
        allowed_count as f64 / (states * states) as f64
    );

    println!();
    println!("first 30 state transitions from boundary state 0:");
    for j in 0..states.min(30) {
        if allowed[0][j] {
            println!(
                "  0 -> {:>3}  depth={} tag={:?}",
                j,
                j / tags.len(),
                tags[j % tags.len()]
            );
        }
    }

    println!();
    println!("states that can transition to boundary state 0:");
    let mut shown = 0;
    for i in 0..states {
        if allowed[i][0] {
            println!(
                "  {:>3} -> 0  depth={} tag={:?}",
                i,
                i / tags.len(),
                tags[i % tags.len()]
            );
            shown += 1;
            if shown >= 30 {
                break;
            }
        }
    }

    Ok(())
}

fn decode_logits(
    logits: &[Vec<f32>],      // [num_words][num_tags]
    valid_mask: &[bool],      // [num_words]
    allowed: &[Vec<bool>],    // [num_states][num_states]
    num_tags: usize,
    depth: usize,
) -> Vec<usize> {
    let num_words = logits.len();
    let num_states = num_tags * depth;
    let neg_inf = -1.0e9_f32;

    assert_eq!(valid_mask.len(), num_words);
    assert_eq!(allowed.len(), num_states);

    let mut alpha = vec![vec![neg_inf; num_states]; num_words];
    let mut beta = vec![vec![0usize; num_states]; num_words];

    for t in 0..num_words {
        for state in 0..num_states {
            let tag = state % num_tags;

            let mut score = logits[t][tag];

            // Python forces tag 0 for padding positions.
            if !valid_mask[t] && state >= 1 {
                score = neg_inf;
            }

            // Python first-position boundary condition:
            // logits[:, 0, allowed[0, :] == -inf] = -1e9
            if t == 0 && !allowed[0][state] {
                score = neg_inf;
            }

            // Python last-position boundary condition:
            // logits[:, -1, allowed[:, 0] == -inf] = -1e9
            if t == num_words - 1 && !allowed[state][0] {
                score = neg_inf;
            }

            if t == 0 {
                alpha[t][state] = score;
            } else {
                let mut best_prev = 0usize;
                let mut best_score = neg_inf;

                for prev in 0..num_states {
                    if allowed[prev][state] {
                        let candidate = alpha[t - 1][prev];
                        if candidate > best_score {
                            best_score = candidate;
                            best_prev = prev;
                        }
                    }
                }

                alpha[t][state] = score + best_score;
                beta[t][state] = best_prev;
            }
        }
    }

    let mut states = vec![0usize; num_words];

    states[num_words - 1] = alpha[num_words - 1]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    for t in (0..num_words - 1).rev() {
        states[t] = beta[t + 1][states[t + 1]];
    }

    states.into_iter().map(|state| state % num_tags).collect()
}

fn tag_index(tags: &[String], wanted: &str) -> Result<usize> {
    tags.iter()
        .position(|tag| tag == wanted)
        .with_context(|| format!("tag not found: {wanted:?}"))
}

fn decode_synthetic(model_dir: PathBuf) -> Result<()> {
    let (options, tags) = load_options_and_tags(&model_dir)?;
    let depth = options.depth.unwrap_or(5);
    let allowed = allowed_tag_transitions(&tags, depth);

    let empty = tag_index(&tags, "")?;
    let push = tag_index(&tags, "PUSH")?;
    let pop1 = tag_index(&tags, "POP:1")?;

    // Four-word fake sentence.
    // We want the decoder to select:
    //   word 0: ""
    //   word 1: "PUSH"
    //   word 2: ""
    //   word 3: "POP:1"
    //
    // That corresponds to one mention spanning words 1..3.
    let mut logits = vec![vec![-5.0_f32; tags.len()]; 4];

    logits[0][empty] = 5.0;
    logits[1][push] = 5.0;
    logits[2][empty] = 5.0;
    logits[3][pop1] = 5.0;

    let valid_mask = vec![true, true, true, true];

    let predicted = decode_logits(&logits, &valid_mask, &allowed, tags.len(), depth);

    println!("predicted tag ids: {:?}", predicted);
    for (i, tag_id) in predicted.iter().enumerate() {
        println!("  word {i}: {:?}", tags[*tag_id]);
    }

    let mentions = tags_to_mentions(&predicted, &tags);

    println!("mentions:");
    for mention in mentions {
        println!("  {}..{}", mention.start, mention.end);
    }

    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MentionSpan {
    start: usize,
    end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedMention {
    span: MentionSpan,
    entity_id: usize,
}

fn tags_to_mentions(predicted_tag_ids: &[usize], tags: &[String]) -> Vec<MentionSpan> {
    let mut mentions: Vec<MentionSpan> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();

    for (i, tag_id) in predicted_tag_ids.iter().copied().enumerate() {
        let tag = &tags[tag_id % tags.len()];

        if tag.is_empty() {
            continue;
        }

        for command in tag.split(',') {
            if command == "PUSH" {
                stack.push(i);
            } else if let Some(rest) = command.strip_prefix("POP:") {
                if !stack.is_empty() {
                    let requested = rest.parse::<usize>().unwrap_or(1);
                    let from_top = if requested <= stack.len() { requested } else { 1 };
                    let j = stack.len() - from_top;
                    let start = stack.remove(j);
                    mentions.push(MentionSpan { start, end: i });
                }
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

    mentions.sort_by_key(|m| (m.start, std::cmp::Reverse(m.end)));
    mentions.dedup();

    mentions
}

fn resolve_antecedents_synthetic(
    mentions: &[MentionSpan],
    scores: &[Vec<f32>],
) -> Vec<ResolvedMention> {
    assert_eq!(mentions.len(), scores.len());

    let mut next_entity_id = 1usize;
    let mut resolved: Vec<ResolvedMention> = Vec::new();

    for i in 0..mentions.len() {
        // CorPipe only allows antecedents up to and including self.
        assert!(
            scores[i].len() >= i + 1,
            "score row {i} must have at least {} entries",
            i + 1
        );

        let mut best_j = 0usize;
        let mut best_score = f32::NEG_INFINITY;

        for j in 0..=i {
            if scores[i][j] > best_score {
                best_score = scores[i][j];
                best_j = j;
            }
        }

        let entity_id = if best_j == i {
            let entity_id = next_entity_id;
            next_entity_id += 1;
            entity_id
        } else {
            resolved[best_j].entity_id
        };

        resolved.push(ResolvedMention {
            span: mentions[i].clone(),
            entity_id,
        });
    }

    resolved
}

fn resolve_synthetic() -> Result<()> {
    // Imagine these are mention spans from one document:
    //
    // 0: Alice
    // 1: Mary
    // 2: she
    //
    // We want:
    //   Alice -> new entity c1
    //   Mary  -> new entity c2
    //   she   -> links back to Alice/c1

    let mentions = vec![
        MentionSpan { start: 0, end: 0 },
        MentionSpan { start: 2, end: 2 },
        MentionSpan { start: 5, end: 5 },
    ];

    // Row i contains scores for antecedents 0..=i.
    // The self-column i means "start a new entity".
    let scores = vec![
        vec![10.0],              // Alice chooses self -> c1
        vec![0.0, 10.0],         // Mary chooses self -> c2
        vec![20.0, 1.0, 5.0],    // she chooses Alice -> c1
    ];

    let resolved = resolve_antecedents_synthetic(&mentions, &scores);

    for (i, mention) in resolved.iter().enumerate() {
        println!(
            "mention {i}: {}..{} -> c{}",
            mention.span.start,
            mention.span.end,
            mention.entity_id
        );
    }

    Ok(())
}

fn synthetic_pipeline(model_dir: PathBuf) -> Result<()> {
    let (options, tags) = load_options_and_tags(&model_dir)?;
    let depth = options.depth.unwrap_or(5);
    let allowed = allowed_tag_transitions(&tags, depth);

    let empty = tag_index(&tags, "")?;
    let push = tag_index(&tags, "PUSH")?;
    let pop1 = tag_index(&tags, "POP:1")?;

    // Fake sentence:
    //   0 Alice
    //   1 asked
    //   2 Mary
    //   3 whether
    //   4 she
    //   5 agreed
    //
    // We will force three single-token mentions:
    //   Alice: PUSH,POP:1 at word 0
    //   Mary:  PUSH,POP:1 at word 2
    //   she:   PUSH,POP:1 at word 4
    //
    // The tag "PUSH,POP:1" exists in this model's tags.txt.
    let single_token_mention = tag_index(&tags, "PUSH,POP:1")?;

    let words = ["Alice", "asked", "Mary", "whether", "she", "agreed"];

    let mut logits = vec![vec![-5.0_f32; tags.len()]; words.len()];

    logits[0][single_token_mention] = 10.0;
    logits[1][empty] = 10.0;
    logits[2][single_token_mention] = 10.0;
    logits[3][empty] = 10.0;
    logits[4][single_token_mention] = 10.0;
    logits[5][empty] = 10.0;

    // Keep the older PUSH/POP variables used, so the compiler does not warn if warnings are strict.
    let _ = (push, pop1);

    let valid_mask = vec![true; words.len()];
    let predicted = decode_logits(&logits, &valid_mask, &allowed, tags.len(), depth);

    println!("words:");
    for (i, word) in words.iter().enumerate() {
        println!("  {i}: {word}");
    }

    println!();
    println!("predicted tags:");
    for (i, tag_id) in predicted.iter().enumerate() {
        println!("  {:>2} {:<10} {:?}", i, words[i], tags[*tag_id]);
    }

    let mentions = tags_to_mentions(&predicted, &tags);

    println!();
    println!("mentions:");
    for (i, mention) in mentions.iter().enumerate() {
        println!(
            "  mention {i}: {}..{} = {}",
            mention.start,
            mention.end,
            words[mention.start..=mention.end].join(" ")
        );
    }

    // Expected mention order after sorting:
    //   0 Alice
    //   1 Mary
    //   2 she
    //
    // Fake antecedent scores:
    //   Alice chooses self -> c1
    //   Mary chooses self  -> c2
    //   she chooses Alice  -> c1
    let scores = vec![
        vec![10.0],
        vec![0.0, 10.0],
        vec![20.0, 1.0, 5.0],
    ];

    let resolved = resolve_antecedents_synthetic(&mentions, &scores);

    println!();
    println!("resolved entities:");
    for (i, mention) in resolved.iter().enumerate() {
        println!(
            "  mention {i}: {}..{} {:<10} -> c{}",
            mention.span.start,
            mention.span.end,
            words[mention.span.start..=mention.span.end].join(" "),
            mention.entity_id
        );
    }

    Ok(())
}

fn blank(s: &str) -> &str {
    if s.is_empty() {
        "_"
    } else {
        s
    }
}

fn parse_text(udpipe_model: PathBuf, text: String) -> Result<()> {
    let model = udpipe_rs::Model::load(&udpipe_model)
        .with_context(|| format!("failed to load UDPipe model {}", udpipe_model.display()))?;

    let words = model
        .parse(&text)
        .with_context(|| "failed to parse text with UDPipe")?;

    let tokens: Vec<ConlluToken> = words.iter().map(ConlluToken::from).collect();

    print_conllu_tokens(&tokens, &text);

    Ok(())
}

#[derive(Debug, Clone)]
struct ConlluToken {
    sent_id: i32,
    id: i32,
    form: String,
    lemma: String,
    upos: String,
    xpos: String,
    feats: String,
    head: i32,
    deprel: String,
    deps: String,
    misc: String,
}

impl From<&udpipe_rs::Word> for ConlluToken {
    fn from(word: &udpipe_rs::Word) -> Self {
        Self {
            sent_id: word.sentence_id,
            id: word.id,
            form: word.form.clone(),
            lemma: word.lemma.clone(),
            upos: word.upostag.clone(),
            xpos: word.xpostag.clone(),
            feats: blank(&word.feats).to_string(),
            head: word.head,
            deprel: word.deprel.clone(),
            deps: "_".to_string(),
            misc: blank(&word.misc).to_string(),
        }
    }
}

fn print_conllu_tokens(tokens: &[ConlluToken], original_text: &str) {
    println!("# newdoc");

    let mut current_sentence: Option<i32> = None;

    for token in tokens {
        if current_sentence != Some(token.sent_id) {
            if current_sentence.is_some() {
                println!();
            }

            current_sentence = Some(token.sent_id);
            println!("# sent_id = {}", token.sent_id + 1);
            println!("# text = {}", original_text);
        }

        println!(
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

    println!();
}

fn add_misc_attr(misc: &mut String, key: &str, value: &str) {
    if misc == "_" || misc.is_empty() {
        *misc = format!("{key}={value}");
    } else {
        misc.push('|');
        misc.push_str(key);
        misc.push('=');
        misc.push_str(value);
    }
}

fn add_entity_marker(misc: &mut String, marker: &str) {
    if misc.is_empty() || misc == "_" {
        *misc = format!("Entity={marker}");
        return;
    }

    let mut parts: Vec<String> = misc.split('|').map(|s| s.to_string()).collect();

    // If Entity= already exists, append the marker to that same Entity value.
    for part in &mut parts {
        if let Some(existing) = part.strip_prefix("Entity=") {
            *part = format!("Entity={existing}{marker}");
            *misc = parts.join("|");
            return;
        }
    }

    // No Entity= yet. Put Entity first, before SpaceAfter etc.,
    parts.insert(0, format!("Entity={marker}"));
    *misc = parts.join("|");
}

fn predict_text_mock(udpipe_model: PathBuf, text: String) -> Result<()> {
    let model = udpipe_rs::Model::load(&udpipe_model)
        .with_context(|| format!("failed to load UDPipe model {}", udpipe_model.display()))?;

    let words = model
        .parse(&text)
        .with_context(|| "failed to parse text with UDPipe")?;

    let mut tokens: Vec<ConlluToken> = words.iter().map(ConlluToken::from).collect();

    // Mock resolver:
    // Alice -> c1
    // Mary  -> c2
    // she   -> c1
    for token in &mut tokens {
        match token.form.as_str() {
            "Alice" => add_misc_attr(&mut token.misc, "Entity", "(c1--1)"),
            "Mary" => add_misc_attr(&mut token.misc, "Entity", "(c2--1)"),
            "she" | "She" => add_misc_attr(&mut token.misc, "Entity", "(c1--1)"),
            _ => {}
        }
    }

    println!("# newdoc");
    println!("# global.Entity = eid-etype-head-other");

    print_conllu_tokens_without_newdoc(&tokens, &text);

    Ok(())
}

fn print_conllu_tokens_without_newdoc(tokens: &[ConlluToken], original_text: &str) {
    let mut current_sentence: Option<i32> = None;

    for token in tokens {
        if current_sentence != Some(token.sent_id) {
            if current_sentence.is_some() {
                println!();
            }

            current_sentence = Some(token.sent_id);
            println!("# sent_id = {}", token.sent_id + 1);
            println!("# text = {}", original_text);
        }

        println!(
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

    println!();
}

fn tokenize_text(
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let model = udpipe_rs::Model::load(&udpipe_model)
        .with_context(|| format!("failed to load UDPipe model {}", udpipe_model.display()))?;

    let words = model
        .parse(&text)
        .with_context(|| "failed to parse text with UDPipe")?;

    let tokens: Vec<ConlluToken> = words.iter().map(ConlluToken::from).collect();

    let mut tokenizer = Tokenizer::from_file(&tokenizer_json)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tokenizer_json.display()))?;

    tokenizer
        .add_special_tokens(vec![
            AddedToken::from("[TOKEN_EMPTY]", true),
            AddedToken::from("[TOKEN_CLS]", true),
        ])
        .map_err(|e| anyhow::anyhow!("failed to add special tokens: {e}"))?;

    let cls_id = tokenizer
        .token_to_id("[TOKEN_CLS]")
        .context("missing [TOKEN_CLS] after adding special token")?;

    let sep_id = tokenizer
        .token_to_id("</s>")
        .context("missing </s> token")?;

    let mut subwords: Vec<u32> = Vec::new();
    let mut word_indices: Vec<usize> = Vec::new();

    for token in &tokens {
        word_indices.push(subwords.len());

        let encoding = tokenizer
            .encode(token.form.as_str(), false)
            .map_err(|e| anyhow::anyhow!("failed to tokenize {:?}: {e}", token.form))?;

        let ids = encoding.get_ids();

        anyhow::ensure!(
            !ids.is_empty(),
            "tokenizer produced no ids for {:?}",
            token.form
        );

        println!("{:<12} -> {:?}", token.form, ids);

        subwords.extend_from_slice(ids);
    }

    word_indices.push(subwords.len());

    let model_word_indices: Vec<usize> = word_indices.iter().map(|i| i + 2).collect();

    let mut model_input: Vec<u32> = Vec::new();
    model_input.push(cls_id);
    model_input.push(sep_id);
    model_input.extend_from_slice(&subwords);
    model_input.push(sep_id);
    model_input.push(sep_id);

    println!();
    println!("subwords len: {}", subwords.len());
    println!("subwords: {:?}", subwords);

    println!();
    println!("word_indices raw:   {:?}", word_indices);
    println!("word_indices model: {:?}", model_word_indices);

    println!();
    println!("special ids:");
    println!("  [TOKEN_CLS] = {}", cls_id);
    println!("  </s>        = {}", sep_id);

    println!();
    println!("model_input len: {}", model_input.len());
    println!("model_input: {:?}", model_input);

    Ok(())
}

fn heads_smoke(model_dir: PathBuf) -> Result<()> {
    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let heads = CorpipeHeads::load(vb)?;

    let fake_words = Tensor::zeros((1usize, 7usize, 768usize), DType::F32, &device)
        .with_context(|| "failed to create fake embeddings")?;

    let tag_logits = heads.tag_logits(&fake_words)?;

    println!("fake_words shape: {:?}", fake_words.shape());
    println!("tag_logits shape: {:?}", tag_logits.shape());

    let logits_vec = tag_logits
        .get(0)
        .with_context(|| "failed to select batch 0")?
        .get(0)
        .with_context(|| "failed to select word 0")?
        .to_vec1::<f32>()
        .with_context(|| "failed to convert logits to vec")?;

    println!();
    println!("first word logits len: {}", logits_vec.len());
    println!("first 10 logits:");
    for (i, value) in logits_vec.iter().take(10).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn gather_mention_embeddings(
    embeddings: &Tensor,          // [batch, seq, hidden]
    mentions: &[(usize, usize)],  // [(start, end)]
) -> Result<Tensor> {
    let mut rows = Vec::new();

    for &(start, end) in mentions {
        let start_emb = embeddings
            .get(0)?
            .get(start)?
            .flatten_all()?;

        let end_emb = embeddings
            .get(0)?
            .get(end)?
            .flatten_all()?;

        let row = Tensor::cat(&[start_emb, end_emb], 0)?;
        rows.push(row);
    }

    let refs: Vec<&Tensor> = rows.iter().collect();
    let stacked = Tensor::stack(&refs, 0)?;

    // Add batch dimension: [1, num_mentions, 2 * hidden]
    Ok(stacked.unsqueeze(0)?)
}

fn antecedent_smoke(model_dir: PathBuf) -> Result<()> {
    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let heads = CorpipeHeads::load(vb)?;

    let embeddings = Tensor::zeros((1usize, 12usize, 768usize), DType::F32, &device)?;

    let all_mentions = vec![(2usize, 2usize), (4usize, 4usize), (6usize, 6usize)];
    let current_mentions = vec![(4usize, 4usize), (6usize, 6usize)];

    let scores = heads.antecedent_scores(&embeddings, &all_mentions, &current_mentions)?;

    println!("embeddings shape: {:?}", embeddings.shape());
    println!("scores shape:     {:?}", scores.shape());

    println!();
    println!("scores:");
    let scores_vec = scores.squeeze(0)?.to_vec2::<f32>()?;
    for row in scores_vec {
        println!("  {:?}", row);
    }

    Ok(())
}

struct CorpipeHeads {
    dense_hidden_tags: candle_nn::Linear,
    dense_tags: candle_nn::Linear,

    dense_hidden_q: candle_nn::Linear,
    dense_hidden_k: candle_nn::Linear,
    dense_q: candle_nn::Linear,
    dense_k: candle_nn::Linear,
}

impl CorpipeHeads {
    fn load(vb: VarBuilder) -> Result<Self> {
        let dense_hidden_tags = candle_nn::linear(
            768,
            3072,
            vb.pp("_dense_hidden_tags"),
        )
        .with_context(|| "failed to load _dense_hidden_tags")?;

        let dense_tags = candle_nn::linear(
            3072,
            56,
            vb.pp("_dense_tags"),
        )
        .with_context(|| "failed to load _dense_tags")?;

        let dense_hidden_q = candle_nn::linear(
            1536,
            3072,
            vb.pp("_dense_hidden_q"),
        )
        .with_context(|| "failed to load _dense_hidden_q")?;

        let dense_hidden_k = candle_nn::linear(
            1536,
            3072,
            vb.pp("_dense_hidden_k"),
        )
        .with_context(|| "failed to load _dense_hidden_k")?;

        let dense_q = candle_nn::linear_no_bias(
            3072,
            768,
            vb.pp("_dense_q"),
        )
        .with_context(|| "failed to load _dense_q")?;

        let dense_k = candle_nn::linear_no_bias(
            3072,
            768,
            vb.pp("_dense_k"),
        )
        .with_context(|| "failed to load _dense_k")?;

        Ok(Self {
            dense_hidden_tags,
            dense_tags,
            dense_hidden_q,
            dense_hidden_k,
            dense_q,
            dense_k,
        })
    }

    fn tag_logits(&self, word_embeddings: &Tensor) -> Result<Tensor> {
        let hidden = self
            .dense_hidden_tags
            .forward(word_embeddings)
            .with_context(|| "dense_hidden_tags forward failed")?
            .relu()
            .with_context(|| "tag hidden relu failed")?;

        self.dense_tags
            .forward(&hidden)
            .with_context(|| "dense_tags forward failed")
    }

    fn mention_embeddings(
        embeddings: &Tensor,
        mentions: &[(usize, usize)],
    ) -> Result<Tensor> {
        gather_mention_embeddings(embeddings, mentions)
    }

    fn antecedent_scores(
        &self,
        embeddings: &Tensor,
        all_mentions: &[(usize, usize)],
        current_mentions: &[(usize, usize)],
    ) -> Result<Tensor> {
        let all_emb = Self::mention_embeddings(embeddings, all_mentions)?;
        let current_emb = Self::mention_embeddings(embeddings, current_mentions)?;

        let keys = self
            .dense_k
            .forward(
                &self
                    .dense_hidden_k
                    .forward(&all_emb)
                    .with_context(|| "dense_hidden_k forward failed")?
                    .relu()
                    .with_context(|| "key hidden relu failed")?,
            )
            .with_context(|| "dense_k forward failed")?;

        let queries = self
            .dense_q
            .forward(
                &self
                    .dense_hidden_q
                    .forward(&current_emb)
                    .with_context(|| "dense_hidden_q forward failed")?
                    .relu()
                    .with_context(|| "query hidden relu failed")?,
            )
            .with_context(|| "dense_q forward failed")?;

        let keys_t = keys.transpose(1, 2)?;
        let raw_scores = queries.matmul(&keys_t)?;
        let scores = (raw_scores / 768f64.sqrt())?;

        Ok(scores)
    }
}

#[derive(Debug)]
struct TokenizedInput {
    tokens: Vec<ConlluToken>,
    model_input: Vec<u32>,
    model_word_indices: Vec<usize>,
}

fn build_tokenized_input(
    udpipe_model: &PathBuf,
    tokenizer_json: &PathBuf,
    text: &str,
) -> Result<TokenizedInput> {
    let model = udpipe_rs::Model::load(udpipe_model)
        .with_context(|| format!("failed to load UDPipe model {}", udpipe_model.display()))?;

    let words = model
        .parse(text)
        .with_context(|| "failed to parse text with UDPipe")?;

    let tokens: Vec<ConlluToken> = words.iter().map(ConlluToken::from).collect();

    let mut tokenizer = Tokenizer::from_file(tokenizer_json)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", tokenizer_json.display()))?;

    tokenizer
        .add_special_tokens(vec![
            AddedToken::from("[TOKEN_EMPTY]", true),
            AddedToken::from("[TOKEN_CLS]", true),
        ])
        .map_err(|e| anyhow::anyhow!("failed to add special tokens: {e}"))?;

    let cls_id = tokenizer
        .token_to_id("[TOKEN_CLS]")
        .context("missing [TOKEN_CLS] after adding special token")?;

    let sep_id = tokenizer
        .token_to_id("</s>")
        .context("missing </s> token")?;

    let mut subwords: Vec<u32> = Vec::new();
    let mut word_indices: Vec<usize> = Vec::new();

    for token in &tokens {
        word_indices.push(subwords.len());

        let encoding = tokenizer
            .encode(token.form.as_str(), false)
            .map_err(|e| anyhow::anyhow!("failed to tokenize {:?}: {e}", token.form))?;

        let ids = encoding.get_ids();

        anyhow::ensure!(
            !ids.is_empty(),
            "tokenizer produced no ids for {:?}",
            token.form
        );

        subwords.extend_from_slice(ids);
    }

    word_indices.push(subwords.len());

    let model_word_indices: Vec<usize> = word_indices.iter().map(|i| i + 2).collect();

    let mut model_input: Vec<u32> = Vec::new();
    model_input.push(cls_id);
    model_input.push(sep_id);
    model_input.extend_from_slice(&subwords);
    model_input.push(sep_id);
    model_input.push(sep_id);

    Ok(TokenizedInput {
        tokens,
        model_input,
        model_word_indices,
    })
}

fn gather_word_embeddings(
    embeddings: &Tensor,          // [1, seq, hidden]
    word_indices: &[usize],       // includes final boundary
) -> Result<Tensor> {
    let mut rows = Vec::new();

    for &idx in &word_indices[..word_indices.len() - 1] {
        let row = embeddings
            .get(0)?
            .get(idx)?
            .flatten_all()?;

        rows.push(row);
    }

    let refs: Vec<&Tensor> = rows.iter().collect();
    let stacked = Tensor::stack(&refs, 0)?;

    Ok(stacked.unsqueeze(0)?)
}

fn predict_text_fake_encoder(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let (options, tags) = load_options_and_tags(&model_dir)?;
    let depth = options.depth.unwrap_or(5);
    let allowed = allowed_tag_transitions(&tags, depth);

    let mut input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let heads = CorpipeHeads::load(vb)?;

    // Fake encoder embeddings with correct shape:
    // [batch=1, seq_len=model_input.len(), hidden=768]
    let embeddings = Tensor::zeros(
        (1usize, input.model_input.len(), 768usize),
        DType::F32,
        &device,
    )?;

    let word_embeddings = gather_word_embeddings(&embeddings, &input.model_word_indices)?;
    let tag_logits = heads.tag_logits(&word_embeddings)?;

    let logits_matrix = tag_logits
        .squeeze(0)?
        .to_vec2::<f32>()?;

    let valid_mask = vec![true; input.tokens.len()];
    let predicted_tag_ids = decode_logits(
        &logits_matrix,
        &valid_mask,
        &allowed,
        tags.len(),
        depth,
    );

    let mentions = tags_to_mentions(&predicted_tag_ids, &tags);

    // For fake encoder mode, antecedent scores are not meaningful.
    // Make every mention choose itself, so every mention becomes a new entity.
    let mut scores = Vec::new();
    for i in 0..mentions.len() {
        let mut row = vec![-10.0_f32; i + 1];
        row[i] = 10.0;
        scores.push(row);
    }

    let resolved = resolve_antecedents_synthetic(&mentions, &scores);

    for mention in &resolved {
        let value = format!("(c{}--1)", mention.entity_id);

        // This first version only writes single-token mentions neatly.
        // Multi-token CorefUD spans need open/close handling, which we can add next.
        if mention.span.start == mention.span.end {
            if let Some(token) = input.tokens.get_mut(mention.span.start) {
                add_misc_attr(&mut token.misc, "Entity", &value);
            }
        }
    }

    println!("# newdoc");
    println!("# global.Entity = eid-etype-head-other");
    print_conllu_tokens_without_newdoc(&input.tokens, &text);

    eprintln!("debug:");
    eprintln!("  model_input len: {}", input.model_input.len());
    eprintln!("  tokens: {}", input.tokens.len());
    eprintln!("  predicted tags:");
    for (i, tag_id) in predicted_tag_ids.iter().enumerate() {
        eprintln!("    {:>2} {:<12} {:?}", i, input.tokens[i].form, tags[*tag_id]);
    }
    eprintln!("  mentions: {:?}", mentions);

    Ok(())
}

fn load_embedding_weight(vb: VarBuilder) -> Result<Tensor> {
    vb.get(
        (256302usize, 768usize),
        "_encoder.encoder.embed_tokens.weight",
    )
    .with_context(|| "failed to load _encoder.encoder.embed_tokens.weight")
}

fn embedding_lookup(weight: &Tensor, input_ids: &[u32]) -> Result<Tensor> {
    let device = weight.device();

    let ids: Vec<u32> = input_ids.to_vec();

    let ids_tensor = Tensor::from_vec(
        ids,
        (input_ids.len(),),
        device,
    )
    .with_context(|| "failed to create input id tensor")?;

    let embeddings = weight
        .index_select(&ids_tensor, 0)
        .with_context(|| "embedding index_select failed")?;

    // [seq, hidden] -> [1, seq, hidden]
    embeddings.unsqueeze(0).with_context(|| "embedding unsqueeze failed")
}

fn embed_smoke(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let embed_weight = load_embedding_weight(vb)?;

    let embeddings = embedding_lookup(&embed_weight, &input.model_input)?;

    println!("model_input len: {}", input.model_input.len());
    println!("model_input: {:?}", input.model_input);
    println!("embedding weight shape: {:?}", embed_weight.shape());
    println!("embeddings shape:       {:?}", embeddings.shape());

    let first_vec = embeddings
        .get(0)?
        .get(0)?
        .to_vec1::<f32>()?;

    println!();
    println!("first embedding vector len: {}", first_vec.len());
    println!("first 8 values:");
    for (i, value) in first_vec.iter().take(8).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    // UMT5/T5 RMSNorm:
    // x * rsqrt(mean(x^2, dim=-1, keepdim=true) + eps) * weight

    let x_dtype = x.dtype();

    let variance = x
        .sqr()
        .with_context(|| "rms_norm sqr failed")?
        .mean_keepdim(candle_core::D::Minus1)
        .with_context(|| "rms_norm mean failed")?;

    let denom = (variance + eps)?
        .sqrt()
        .with_context(|| "rms_norm sqrt failed")?
        .broadcast_as(x.shape())
        .with_context(|| "rms_norm denom broadcast failed")?;

    let normalized = (x / &denom)?
        .to_dtype(x_dtype)
        .with_context(|| "rms_norm cast failed")?;

    normalized
        .broadcast_mul(weight)
        .with_context(|| "rms_norm weight multiply failed")
}

fn rmsnorm_smoke(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let embed_weight = load_embedding_weight(vb.clone())?;
    let embeddings = embedding_lookup(&embed_weight, &input.model_input)?;

    let norm_weight = vb
        .get(
            (768usize,),
            "_encoder.encoder.block.0.layer.0.layer_norm.weight",
        )
        .with_context(|| "failed to load block 0 attention norm weight")?;

    let normed = rms_norm(&embeddings, &norm_weight, 1e-6)?;

    println!("embeddings shape: {:?}", embeddings.shape());
    println!("norm weight shape: {:?}", norm_weight.shape());
    println!("normed shape:     {:?}", normed.shape());

    let first_vec = normed
        .get(0)?
        .get(0)?
        .to_vec1::<f32>()?;

    println!();
    println!("first normed vector len: {}", first_vec.len());
    println!("first 8 values:");
    for (i, value) in first_vec.iter().take(8).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn gelu(x: &Tensor) -> Result<Tensor> {
    // Approximate GELU used by many transformer implementations:
    // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();

    let x3 = x
        .sqr()
        .with_context(|| "gelu sqr failed")?
        .mul(x)
        .with_context(|| "gelu x^3 failed")?;

    let inner = ((x + (x3 * 0.044715)?)? * c)
        .with_context(|| "gelu inner scale failed")?;

    let tanh = inner.tanh().with_context(|| "gelu tanh failed")?;

    let one_plus = (tanh + 1.0)?;
    let half_x = (x * 0.5)?;

    half_x
        .mul(&one_plus)
        .with_context(|| "gelu final multiply failed")
}

fn linear_no_bias_forward(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // x:      [..., in_dim]
    // weight: [out_dim, in_dim]
    // output: [..., out_dim]

    let x_shape = x.dims().to_vec();
    anyhow::ensure!(
        !x_shape.is_empty(),
        "linear_no_bias_forward expects at least 1 dimension"
    );

    let in_dim = *x_shape.last().unwrap();
    let weight_shape = weight.dims();
    anyhow::ensure!(
        weight_shape.len() == 2,
        "weight must be rank 2, got {:?}",
        weight_shape
    );
    anyhow::ensure!(
        weight_shape[1] == in_dim,
        "input dim mismatch: x last dim={}, weight={:?}",
        in_dim,
        weight_shape
    );

    let leading: usize = x_shape[..x_shape.len() - 1].iter().product();
    let out_dim = weight_shape[0];

    let x_2d = x
        .reshape((leading, in_dim))
        .with_context(|| "linear_no_bias reshape to 2d failed")?;

    let y_2d = x_2d
        .matmul(&weight.t()?)
        .with_context(|| "linear_no_bias matmul failed")?;

    let mut out_shape = x_shape;
    *out_shape.last_mut().unwrap() = out_dim;

    y_2d
        .reshape(out_shape)
        .with_context(|| "linear_no_bias reshape output failed")
}

fn t5_gated_ff(
    x: &Tensor,
    layer_norm_weight: &Tensor,
    wi_0: &Tensor,
    wi_1: &Tensor,
    wo: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let normed = rms_norm(x, layer_norm_weight, eps)?;

    let gate = linear_no_bias_forward(&normed, wi_0)?;
    let gate = gelu(&gate)?;

    let value = linear_no_bias_forward(&normed, wi_1)?;

    let hidden = gate
        .mul(&value)
        .with_context(|| "ff gate/value multiply failed")?;

    let ff = linear_no_bias_forward(&hidden, wo)?;

    (x + ff).with_context(|| "ff residual add failed")
}

fn ff_smoke(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let embed_weight = load_embedding_weight(vb.clone())?;
    let embeddings = embedding_lookup(&embed_weight, &input.model_input)?;

    let prefix = "_encoder.encoder.block.0.layer.1";

    let ln = vb
        .get((768usize,), &format!("{prefix}.layer_norm.weight"))
        .with_context(|| "failed to load ff layer norm")?;

    let wi_0 = vb
        .get((2048usize, 768usize), &format!("{prefix}.DenseReluDense.wi_0.weight"))
        .with_context(|| "failed to load wi_0")?;

    let wi_1 = vb
        .get((2048usize, 768usize), &format!("{prefix}.DenseReluDense.wi_1.weight"))
        .with_context(|| "failed to load wi_1")?;

    let wo = vb
        .get((768usize, 2048usize), &format!("{prefix}.DenseReluDense.wo.weight"))
        .with_context(|| "failed to load wo")?;

    let output = t5_gated_ff(&embeddings, &ln, &wi_0, &wi_1, &wo, 1e-6)?;

    println!("input embeddings shape: {:?}", embeddings.shape());
    println!("ln shape:               {:?}", ln.shape());
    println!("wi_0 shape:             {:?}", wi_0.shape());
    println!("wi_1 shape:             {:?}", wi_1.shape());
    println!("wo shape:               {:?}", wo.shape());
    println!("ff output shape:        {:?}", output.shape());

    let first_vec = output
        .get(0)?
        .get(0)?
        .to_vec1::<f32>()?;

    println!();
    println!("first ff output vector len: {}", first_vec.len());
    println!("first 8 values:");
    for (i, value) in first_vec.iter().take(8).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    candle_nn::ops::softmax(x, candle_core::D::Minus1)
        .with_context(|| "softmax failed")
}

fn relative_position_bucket(
    relative_position: isize,
    num_buckets: usize,
    max_distance: usize,
) -> usize {
    let relative_position = -relative_position;

    let num_buckets_half = num_buckets / 2;

    let mut bucket = 0usize;
    if relative_position > 0 {
        bucket += num_buckets_half;
    }

    let n = relative_position.abs() as usize;
    let max_exact = num_buckets_half / 2;

    if n < max_exact {
        bucket + n
    } else {
        let n_float = n as f64;
        let max_exact_float = max_exact as f64;
        let max_distance_float = max_distance as f64;

        let val = max_exact
            + ((n_float / max_exact_float).ln()
                / (max_distance_float / max_exact_float).ln()
                * ((num_buckets_half - max_exact) as f64)) as usize;

        bucket + val.min(num_buckets_half - 1)
    }
}

fn relative_attention_bias(
    bias_weight: &Tensor, // [num_buckets, num_heads]
    seq_len: usize,
    num_buckets: usize,
    max_distance: usize,
) -> Result<Tensor> {
    let device = bias_weight.device();

    let mut bucket_ids = Vec::with_capacity(seq_len * seq_len);

    for query_pos in 0..seq_len {
        for key_pos in 0..seq_len {
            let rel = query_pos as isize - key_pos as isize;
            let bucket = relative_position_bucket(rel, num_buckets, max_distance);
            bucket_ids.push(bucket as u32);
        }
    }

    let bucket_tensor = Tensor::from_vec(bucket_ids, (seq_len * seq_len,), device)
        .with_context(|| "failed to create relative bucket tensor")?;

    let values = bias_weight
        .index_select(&bucket_tensor, 0)
        .with_context(|| "relative bias index_select failed")?;

    // [seq*seq, heads] -> [seq, seq, heads] -> [heads, seq, seq] -> [1, heads, seq, seq]
    values
        .reshape((seq_len, seq_len, 12usize))
        .with_context(|| "relative bias reshape failed")?
        .permute((2, 0, 1))
        .with_context(|| "relative bias permute failed")?
        .unsqueeze(0)
        .with_context(|| "relative bias unsqueeze failed")
}

fn split_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    // [batch, seq, hidden] -> [batch, heads, seq, head_dim]
    let dims = x.dims();
    anyhow::ensure!(dims.len() == 3, "split_heads expected rank 3, got {:?}", dims);

    let batch = dims[0];
    let seq = dims[1];

    x.reshape((batch, seq, num_heads, head_dim))
        .with_context(|| "split_heads reshape failed")?
        .permute((0, 2, 1, 3))
        .with_context(|| "split_heads permute failed")
}

fn merge_heads(x: &Tensor) -> Result<Tensor> {
    // [batch, heads, seq, head_dim] -> [batch, seq, hidden]
    let dims = x.dims();
    anyhow::ensure!(dims.len() == 4, "merge_heads expected rank 4, got {:?}", dims);

    let batch = dims[0];
    let heads = dims[1];
    let seq = dims[2];
    let head_dim = dims[3];

    x.permute((0, 2, 1, 3))
        .with_context(|| "merge_heads permute failed")?
        .reshape((batch, seq, heads * head_dim))
        .with_context(|| "merge_heads reshape failed")
}

fn t5_self_attention_block(
    x: &Tensor,
    vb: VarBuilder,
    layer: usize,
) -> Result<Tensor> {
    let prefix = format!("_encoder.encoder.block.{layer}.layer.0");

    let ln = vb
        .get((768usize,), &format!("{prefix}.layer_norm.weight"))
        .with_context(|| format!("failed to load attention layer norm for layer {layer}"))?;

    let q_w = vb
        .get((768usize, 768usize), &format!("{prefix}.SelfAttention.q.weight"))
        .with_context(|| format!("failed to load q weight for layer {layer}"))?;

    let k_w = vb
        .get((768usize, 768usize), &format!("{prefix}.SelfAttention.k.weight"))
        .with_context(|| format!("failed to load k weight for layer {layer}"))?;

    let v_w = vb
        .get((768usize, 768usize), &format!("{prefix}.SelfAttention.v.weight"))
        .with_context(|| format!("failed to load v weight for layer {layer}"))?;

    let o_w = vb
        .get((768usize, 768usize), &format!("{prefix}.SelfAttention.o.weight"))
        .with_context(|| format!("failed to load o weight for layer {layer}"))?;

    let bias_w = vb
        .get(
            (32usize, 12usize),
            &format!("{prefix}.SelfAttention.relative_attention_bias.weight"),
        )
        .with_context(|| format!("failed to load relative attention bias for layer {layer}"))?;

    let normed = rms_norm(x, &ln, 1e-6)?;

    let q = linear_no_bias_forward(&normed, &q_w)?;
    let k = linear_no_bias_forward(&normed, &k_w)?;
    let v = linear_no_bias_forward(&normed, &v_w)?;

    let q = split_heads(&q, 12, 64)?;
    let k = split_heads(&k, 12, 64)?;
    let v = split_heads(&v, 12, 64)?;

    let k_t = k.transpose(2, 3)?;
    let scores = q.matmul(&k_t)?;

    let seq_len = x.dims()[1];
    let rel_bias = relative_attention_bias(&bias_w, seq_len, 32, 128)?;

    let scores = scores.broadcast_add(&rel_bias)?;
    let probs = softmax_last_dim(&scores)?;

    let context = probs.matmul(&v)?;
    let context = merge_heads(&context)?;

    let projected = linear_no_bias_forward(&context, &o_w)?;

    (x + projected).with_context(|| format!("attention residual add failed for layer {layer}"))
}

fn attention_smoke(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let embed_weight = load_embedding_weight(vb.clone())?;
    let embeddings = embedding_lookup(&embed_weight, &input.model_input)?;

    let output = t5_self_attention_block(&embeddings, vb.clone(), 0)?;

    println!("input embeddings shape: {:?}", embeddings.shape());
    println!("attention output shape: {:?}", output.shape());

    let first_vec = output
        .get(0)?
        .get(0)?
        .to_vec1::<f32>()?;

    println!();
    println!("first attention output vector len: {}", first_vec.len());
    println!("first 8 values:");
    for (i, value) in first_vec.iter().take(8).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn t5_gated_ff_block(
    x: &Tensor,
    vb: VarBuilder,
    layer: usize,
) -> Result<Tensor> {
    let prefix = format!("_encoder.encoder.block.{layer}.layer.1");

    let ln = vb
        .get((768usize,), &format!("{prefix}.layer_norm.weight"))
        .with_context(|| format!("failed to load ff layer norm for layer {layer}"))?;

    let wi_0 = vb
        .get((2048usize, 768usize), &format!("{prefix}.DenseReluDense.wi_0.weight"))
        .with_context(|| format!("failed to load wi_0 for layer {layer}"))?;

    let wi_1 = vb
        .get((2048usize, 768usize), &format!("{prefix}.DenseReluDense.wi_1.weight"))
        .with_context(|| format!("failed to load wi_1 for layer {layer}"))?;

    let wo = vb
        .get((768usize, 2048usize), &format!("{prefix}.DenseReluDense.wo.weight"))
        .with_context(|| format!("failed to load wo for layer {layer}"))?;

    t5_gated_ff(x, &ln, &wi_0, &wi_1, &wo, 1e-6)
}

fn t5_encoder_block(
    x: &Tensor,
    vb: VarBuilder,
    layer: usize,
) -> Result<Tensor> {
    let x = t5_self_attention_block(x, vb.clone(), layer)
        .with_context(|| format!("attention block failed at layer {layer}"))?;

    let x = t5_gated_ff_block(&x, vb, layer)
        .with_context(|| format!("feed-forward block failed at layer {layer}"))?;

    Ok(x)
}

fn block_smoke(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let embed_weight = load_embedding_weight(vb.clone())?;
    let embeddings = embedding_lookup(&embed_weight, &input.model_input)?;

    let output = t5_encoder_block(&embeddings, vb.clone(), 0)?;

    println!("input embeddings shape: {:?}", embeddings.shape());
    println!("block 0 output shape:   {:?}", output.shape());

    let first_vec = output
        .get(0)?
        .get(0)?
        .to_vec1::<f32>()?;

    println!();
    println!("first block output vector len: {}", first_vec.len());
    println!("first 8 values:");
    for (i, value) in first_vec.iter().take(8).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn t5_encoder_forward(
    input_ids: &[u32],
    vb: VarBuilder,
) -> Result<Tensor> {
    let embed_weight = load_embedding_weight(vb.clone())?;
    let mut x = embedding_lookup(&embed_weight, input_ids)?;

    for layer in 0..12 {
        x = t5_encoder_block(&x, vb.clone(), layer)
            .with_context(|| format!("encoder block {layer} failed"))?;
    }

    let final_ln = vb
        .get(
            (768usize,),
            "_encoder.encoder.final_layer_norm.weight",
        )
        .with_context(|| "failed to load final encoder layer norm")?;

    let x = rms_norm(&x, &final_ln, 1e-6)
        .with_context(|| "final encoder rms norm failed")?;

    Ok(x)
}

fn encoder_smoke(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let output = t5_encoder_forward(&input.model_input, vb)?;

    println!("model_input len:     {}", input.model_input.len());
    println!("encoder output shape: {:?}", output.shape());

    let first_vec = output
        .get(0)?
        .get(0)?
        .to_vec1::<f32>()?;

    println!();
    println!("first encoder output vector len: {}", first_vec.len());
    println!("first 8 values:");
    for (i, value) in first_vec.iter().take(8).enumerate() {
        println!("  {:>2}: {:.6}", i, value);
    }

    Ok(())
}

fn write_entity_mentions(tokens: &mut [ConlluToken], resolved: &[ResolvedMention]) {
    for mention in resolved {
        let entity = format!("c{}", mention.entity_id);

        if mention.span.start == mention.span.end {
            if let Some(token) = tokens.get_mut(mention.span.start) {
                add_entity_marker(&mut token.misc, &format!("({entity}--1)"));
            }
        } else {
            let span_len = mention.span.end - mention.span.start + 1;

            if let Some(token) = tokens.get_mut(mention.span.start) {
                add_entity_marker(&mut token.misc, &format!("({entity}--{span_len}"));
            }

            if let Some(token) = tokens.get_mut(mention.span.end) {
                add_entity_marker(&mut token.misc, &format!("{entity})"));
            }
        }
    }
}

fn predict_text_real_encoder(
    model_dir: PathBuf,
    udpipe_model: PathBuf,
    tokenizer_json: PathBuf,
    text: String,
) -> Result<()> {
    let (options, tags) = load_options_and_tags(&model_dir)?;
    let depth = options.depth.unwrap_or(5);
    let allowed = allowed_tag_transitions(&tags, depth);

    let mut input = build_tokenized_input(&udpipe_model, &tokenizer_json, &text)?;

    let weights_path = model_dir.join("model.safetensors");
    let device = Device::Cpu;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[weights_path.clone()],
            DType::F32,
            &device,
        )
    }
    .with_context(|| format!("failed to load {}", weights_path.display()))?;

    let heads = CorpipeHeads::load(vb.clone())?;

    let embeddings = t5_encoder_forward(&input.model_input, vb.clone())
        .with_context(|| "UMT5 encoder forward failed")?;

    let word_embeddings = gather_word_embeddings(&embeddings, &input.model_word_indices)?;
    let tag_logits = heads.tag_logits(&word_embeddings)?;

    let logits_matrix = tag_logits
        .squeeze(0)?
        .to_vec2::<f32>()?;

    let valid_mask = vec![true; input.tokens.len()];
    let predicted_tag_ids = decode_logits(
        &logits_matrix,
        &valid_mask,
        &allowed,
        tags.len(),
        depth,
    );

    let mentions = tags_to_mentions(&predicted_tag_ids, &tags);

    // Convert word-span mentions to subword-span mentions for antecedent scoring.
    let mut mention_subword_spans = Vec::new();
    for mention in &mentions {
        let start = input.model_word_indices[mention.start];
        let end = input.model_word_indices[mention.end + 1] - 1;
        mention_subword_spans.push((start, end));
    }

    let resolved = if mentions.is_empty() {
        Vec::new()
    } else {
        let scores_tensor = heads.antecedent_scores(
            &embeddings,
            &mention_subword_spans,
            &mention_subword_spans,
        )?;

        let scores = scores_tensor
            .squeeze(0)?
            .to_vec2::<f32>()?;

        eprintln!("  antecedent scores:");
        for (i, row) in scores.iter().enumerate() {
            eprintln!("    mention {i}: {:?}", row);
        }
        resolve_antecedents_synthetic(&mentions, &scores)
    };

    write_entity_mentions(&mut input.tokens, &resolved);

    println!("# newdoc");
    println!("# global.Entity = eid-etype-head-other");
    print_conllu_tokens_without_newdoc(&input.tokens, &text);

    eprintln!("debug:");
    eprintln!("  model_input len: {}", input.model_input.len());
    eprintln!("  tokens: {}", input.tokens.len());
    eprintln!("  mentions: {:?}", mentions);
    eprintln!("  resolved: {:?}", resolved);

    Ok(())
}