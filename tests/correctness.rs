mod common;

use anyhow::{Context, Result};
use hotbatch_server::ServeMode;
use std::process::Command;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fixed_prompt_matches_candle_reference_path() -> Result<()> {
    let prompt = "The capital of France is";
    let max_tokens = 64;
    let seed = 42;
    let reference = reference_generate(prompt, max_tokens, seed).await?;

    let server = common::spawn(ServeMode::Continuous, max_tokens, 1_000).await?;
    let base_url = format!("http://{}", server.addr);
    let served = common::non_stream_completion(&base_url, prompt, max_tokens, seed).await?;
    assert_eq!(served, reference);

    server.stop().await
}

async fn reference_generate(prompt: &str, max_tokens: usize, seed: u64) -> Result<String> {
    let model_name = std::env::var("HOTBATCH_TEST_MODEL").unwrap_or_else(|_| "gpt2".to_string());
    let output = Command::new(assert_cmd::cargo::cargo_bin("candle-gpt2-reference"))
        .args([
            "--model",
            &model_name,
            "--prompt",
            prompt,
            "--max-tokens",
            &max_tokens.to_string(),
            "--seed",
            &seed.to_string(),
        ])
        .output()
        .context("running candle reference subprocess")?;
    assert!(
        output.status.success(),
        "reference subprocess failed: status={:?}\nstderr={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).context("reference stdout was not utf-8")
}
