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
- `src/model.rs`: encoder and neural head runtime
- `src/decode.rs`: tag decoding, mention extraction, and antecedent resolution
- `src/render.rs`: CONLL-U rendering
- `src/main.rs`: thin CLI wrapper

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
