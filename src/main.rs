use anyhow::Result;
use clap::{Parser, Subcommand};
use corpipe_rs::{AnalyzerConfig, CorpipeAnalyzer};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "corpipe-rs",
    about = "Run CorPipe coreference analysis and emit CorefUD-style CONLL-U"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    PredictTextRealEncoder(AnalyzeArgs),
}

#[derive(Parser, Debug)]
struct AnalyzeArgs {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long)]
    tokenizer_json: PathBuf,

    #[arg(long)]
    udpipe_model: PathBuf,

    text: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let args = match cli.command {
        Command::PredictTextRealEncoder(args) => args,
    };

    let analyzer = CorpipeAnalyzer::load(AnalyzerConfig {
        model_dir: args.model_dir,
        udpipe_model: args.udpipe_model,
        tokenizer_json: args.tokenizer_json,
    })?;

    let analysis = analyzer.analyze(&args.text)?;
    print!("{}", analysis.to_conllu());

    Ok(())
}
