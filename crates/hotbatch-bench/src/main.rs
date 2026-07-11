use anyhow::Result;
use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "hotbatch-bench")]
struct Cli {
    #[command(flatten)]
    args: hotbatch_bench::BenchArgs,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    hotbatch_bench::run(cli.args).await
}
