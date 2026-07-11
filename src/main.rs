use anyhow::Result;
use clap::{Parser, Subcommand};
use hotbatch_server::ServeArgs;

#[derive(Debug, Parser)]
#[command(
    name = "hotbatch",
    about = "OpenAI-compatible continuous batching LLM server"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve(ServeArgs),
    Bench(hotbatch_bench::BenchArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    hotbatch_server::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => hotbatch_server::serve(args).await,
        Command::Bench(args) => hotbatch_bench::run(args).await,
    }
}
