use anyhow::Result;
use clap::{Parser, Subcommand};
use hotbatch_server::ServeArgs;

#[derive(Debug, Parser)]
#[command(name = "hotbatch-server")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve(ServeArgs),
}

#[tokio::main]
async fn main() -> Result<()> {
    hotbatch_server::init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => hotbatch_server::serve(args).await,
    }
}
