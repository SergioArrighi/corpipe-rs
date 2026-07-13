# corpipe-rs

> [!WARNING]
> This repository is an unofficial Rust adaptation of [`ufal/crac2025-corpipe`](https://github.com/ufal/crac2025-corpipe), the CorPipe 25 system by **Milan Straka** at **UFAL / Charles University**.
> Credit for the original method, paper, Python implementation, and released models belongs to the upstream project and author.
> This is **not** the original upstream repository, and it should be cited and described as a downstream Rust port/adaptation.

`corpipe-rs` is a Rust library plus a small CLI for running the current CorPipe-based plain-text inference path locally.
It loads a CorPipe model directory, parses raw text with UDPipe, tokenizes it for the encoder, runs the neural pipeline, returns a structured analysis result, and can render CorefUD-style CONLL-U output.

## Scope

This crate currently focuses on inference.

- It provides a reusable library API for analyzing raw text.
- It provides a thin CLI equivalent of the current `predict-text-real-encoder` command.
- It does not attempt to replace the full upstream training, evaluation, or experiment-management workflow from `ufal/crac2025-corpipe`.

## System Requirements

Practical runtime requirements for the current CPU inference path:

- 64-bit Linux or another 64-bit environment supported by the Rust dependencies
- CPU execution only; no GPU is required
- about **2.52 GiB peak RAM** per inference process with the current measured setup
- at least **4 GiB free RAM** recommended for comfortable single-process use
- about **1.14 GiB** of model assets for the measured configuration, plus small UDPipe and tokenizer files

The `2.52 GiB` figure was measured on the release CLI with:

- `corpipe25-base/model.safetensors` at `1.194 GB` on disk
- `umt5-xl-tokenizer/tokenizer.json` at `16.1 MiB`
- `english-gum-ud-2.5-191206.udpipe` at `8.5 MiB`

Actual memory use will vary with the selected model, tokenizer, allocator behavior, and host paging state, but this is a good working estimate for the current setup.

## Requirements

You need external model assets that are not bundled in this repository:

- a CorPipe model directory containing `model.safetensors`, `options.json`, and `tags.txt`
- a UDPipe model file
- a tokenizer JSON file compatible with the selected encoder

## Library Usage

```rust
use corpipe_rs::{AnalyzerConfig, CorpipeAnalyzer};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let analyzer = CorpipeAnalyzer::load(AnalyzerConfig {
        model_dir: PathBuf::from("/path/to/corpipe-model"),
        udpipe_model: PathBuf::from("/path/to/english.udpipe"),
        tokenizer_json: PathBuf::from("/path/to/tokenizer.json"),
    })?;

    let result = analyzer.analyze(
        "Alice asked Mary to rerun GUM dev after the ParCor holdout, but she still wants another audit note.",
    )?;

    println!("{:#?}", result.resolved_mentions);
    println!("{}", result.to_conllu());
    Ok(())
}
```

The library returns an `AnalysisResult` with:

- the original input text
- parsed tokens
- predicted mention tags
- extracted mention spans
- resolved mentions with entity ids

### Supplying UDPipe data as input

The library API can now run the neural CorPipe stage over an existing `UdpipeDocument`. This is useful when an application already runs UDPipe, needs to persist or publish the dependency parse before coreference, or wants several consumers to share one parse. The CLI continues to accept raw text; pre-parsed UDPipe input is currently a library API.

There are three ways to obtain the document.

#### Parse once with the complete analyzer

The complete analyzer can expose its UDPipe result before coreference. `analyze_udpipe` consumes that exact document and does not parse the text again:

```rust
let ud_document = analyzer.parse_udpipe("Mark is part of the Milan soccer team.")?;

// Publish, persist, or inspect `ud_document` here.

let result = analyzer.analyze_udpipe(&ud_document)?;
```

#### Supply words from `udpipe-rs`

Applications that own their UDPipe model can normalize `udpipe_rs::Word` values directly:

```rust
use corpipe_rs::UdpipeDocument;

let words = udpipe_model.parse(text)?;
let ud_document = UdpipeDocument::from_words(text, &words);
```

`Token::from_word` is also public when an application needs to control document assembly itself.

#### Construct or deserialize a document

`UdpipeDocument` and `Token` implement Serde serialization and deserialization, so a parse can cross a process or storage boundary before CorPipe consumes it:

```rust
use corpipe_rs::UdpipeDocument;

let ud_document: UdpipeDocument = serde_json::from_str(&stored_udpipe_json)?;
ud_document.validate()?;
```

The serialized shape is:

```json
{
  "text": "Mark trains.",
  "tokens": [
    {
      "sentence_index": 0,
      "id": 1,
      "form": "Mark",
      "lemma": "Mark",
      "upos": "PROPN",
      "xpos": "NNP",
      "feats": "_",
      "head": 2,
      "deprel": "nsubj",
      "deps": "_",
      "misc": "_"
    },
    {
      "sentence_index": 0,
      "id": 2,
      "form": "trains",
      "lemma": "train",
      "upos": "VERB",
      "xpos": "VBZ",
      "feats": "_",
      "head": 0,
      "deprel": "root",
      "deps": "_",
      "misc": "SpaceAfter=No"
    },
    {
      "sentence_index": 0,
      "id": 3,
      "form": ".",
      "lemma": ".",
      "upos": "PUNCT",
      "xpos": ".",
      "feats": "_",
      "head": 2,
      "deprel": "punct",
      "deps": "_",
      "misc": "_"
    }
  ]
}
```

Before inference, validation requires:

- zero-based, consecutive `sentence_index` values
- one-based, consecutive token `id` values within each sentence
- non-empty token forms
- dependency heads that are `0` for a root or identify a token in the same sentence
- no negative, missing, or self-referential dependency heads

Empty documents are valid and produce an empty `AnalysisResult`.

### Loading only the CorPipe stage

When UDPipe is managed elsewhere, load `CoreferenceAnalyzer` with no UDPipe model path:

```rust
use corpipe_rs::{CoreferenceAnalyzer, CoreferenceConfig, UdpipeDocument};
use std::path::PathBuf;

let coreference = CoreferenceAnalyzer::load(CoreferenceConfig {
    model_dir: PathBuf::from("/path/to/corpipe-model"),
    tokenizer_json: PathBuf::from("/path/to/tokenizer.json"),
})?;
let ud_document: UdpipeDocument = obtain_udpipe_document();
let result = coreference.analyze(&ud_document)?;
```

`CoreferenceAnalyzer::analyze` borrows the input document. It clones the tokens for the result and writes CorefUD `Entity` markers only to those result tokens, leaving the caller's UDPipe document unchanged.

### Current implementation architecture

The current 0.2.x implementation separates parsing from neural coreference:

1. `UdpipeParser` owns the `udpipe-rs` model and produces a reusable `UdpipeDocument`.
2. `CoreferenceAnalyzer` validates the document, tokenizes its forms, runs the Candle encoder and CorPipe neural heads, validates neural output shapes and finite scores, decodes mention tags, and resolves antecedents.
3. The decoder annotates a cloned token list with CorefUD entity markers and returns `AnalysisResult`.
4. `CorpipeAnalyzer` composes both stages. Its `analyze(text)` method is equivalent to `parse_udpipe(text)` followed by `analyze_udpipe(document)`.

All pipeline behavior is owned by these analyzer and document types rather than module-level helper functions. For single-use callers, `CorpipeAnalyzer::analyze_text(config, text)` loads and runs the composed analyzer in one call; reuse a loaded analyzer for multiple documents to avoid repeatedly loading model assets.

## CLI Usage

Run the current inference command like this:

```sh
cargo run -- predict-text-real-encoder \
  --model-dir /path/to/corpipe-model \
  --udpipe-model /path/to/english.udpipe \
  --tokenizer-json /path/to/tokenizer.json \
  "Alice asked Mary to rerun GUM dev after the ParCor holdout, but she still wants another audit note."
```

The CLI writes CorefUD-style CONLL-U to stdout.

## Project Layout

- `src/lib.rs`: public crate surface
- `src/analyzer.rs`: high-level text analysis pipeline
- `src/udpipe.rs`: reusable UDPipe parser stage
- `src/types.rs`: serializable input and output contracts
- `src/model.rs`: encoder and neural head runtime
- `src/decode.rs`: tag decoding, mention extraction, and antecedent resolution
- `src/render.rs`: CONLL-U rendering
- `src/main.rs`: thin CLI wrapper

## Converting the CorPipe checkpoint for Candle

The original CorPipe 25 release stores its model weights as a PyTorch checkpoint, `model.pt`. For the Rust/Candle implementation, this checkpoint needs to be converted to `safetensors`, since Candle can load `safetensors` directly but does not load arbitrary PyTorch pickle checkpoints. We converted the CorPipe base model, `ufal/corpipe25-corefud1.3-base-251101`, by loading `model.pt` with PyTorch on CPU, extracting the state dictionary, cloning tensors to remove shared-storage aliases, and saving the result as `model.safetensors`. The original checkpoint contains tied embedding tensors, `_encoder.shared.weight` and `_encoder.encoder.embed_tokens.weight`; because these share memory, `safetensors` refuses to save both as-is. We keep `_encoder.encoder.embed_tokens.weight`, which is the name used by the Rust/Candle loader, and drop `_encoder.shared.weight` before saving.

Use:
```python
python3 scripts/convert_pt_to_safetensors.py /path/to/model.pt /path/to/model.safetensors
```

## Tokenizer setup

The CorPipe checkpoint contains the model weights, `options.json`, and `tags.txt`, but the tokenizer must be downloaded separately. Although the base checkpoint uses `google/umt5-base` as the encoder, the original CorPipe Python implementation uses the tokenizer from `google/umt5-xl` for all UMT5-based models. To reproduce the Python preprocessing exactly, the Rust implementation must therefore use the `google/umt5-xl` tokenizer, not the `google/umt5-base` tokenizer.

Download the tokenizer files with:

```sh
mkdir -p /<path>/umt5-xl-tokenizer

huggingface-cli download google/umt5-xl \
  tokenizer.json tokenizer_config.json special_tokens_map.json spiece.model \
  --local-dir /<path>/umt5-xl-tokenizer
```

## Attribution

This work is based on the upstream CorPipe 25 repository:

- Upstream repository: [`ufal/crac2025-corpipe`](https://github.com/ufal/crac2025-corpipe)
- Upstream author listed in `AUTHORS`: **Milan Straka**
- Upstream paper: *CorPipe at CRAC 2025: Evaluating Multilingual Encoders for Multilingual Coreference Resolution*

If you use this repository in research or derivative work, credit the upstream project and paper in addition to this Rust adaptation.

## Licensing

This repository's source code is distributed under the **Mozilla Public License 2.0** to match the upstream source repository license.
See [LICENSE](LICENSE).

Important license note:

- the upstream **source repository** is MPL-2.0
- the upstream README states that released **pretrained model weights** are under **CC BY-NC-SA 4.0**
- this repository does not bundle those weights, and model assets may have separate license terms from the code

If you distribute this project or bundle model artifacts with it, review the applicable upstream licenses carefully.
