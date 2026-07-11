mod common;

use anyhow::Result;
use hotbatch_server::ServeMode;
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Barrier;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn temperature_zero_is_byte_deterministic() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 24).await?;
    let base_url = format!("http://{}", server.addr);

    let mut outputs = Vec::new();
    for _ in 0..10 {
        outputs.push(
            common::non_stream_completion(&base_url, "The capital of France is", 16, 42).await?,
        );
    }
    for output in outputs.iter().skip(1) {
        assert_eq!(output, &outputs[0]);
    }

    server.stop().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn seeded_sampling_is_stable_across_batch_membership() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 24).await?;
    let base_url = format!("http://{}", server.addr);
    let prompt = "Seeded sampling should not depend on neighboring requests";
    let baseline = sampled_completion(&base_url, prompt, 16, 0.8, 0.9, 7).await?;

    let barrier = Arc::new(Barrier::new(4));
    let mut tasks = Vec::new();
    for (task_prompt, seed) in [
        (prompt, 7),
        ("Unrelated request one", 11),
        ("Unrelated request two", 13),
    ] {
        let barrier = barrier.clone();
        let base_url = base_url.clone();
        let task_prompt = task_prompt.to_string();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            sampled_completion(&base_url, &task_prompt, 16, 0.8, 0.9, seed).await
        }));
    }
    barrier.wait().await;
    let batched_target = tasks.remove(0).await??;
    for task in tasks {
        task.await??;
    }

    assert_eq!(batched_target, baseline);
    server.stop().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn greedy_output_matches_naive_and_continuous_modes() -> Result<()> {
    let prompt = "The same model path should agree across serving modes";
    let continuous = common::spawn(ServeMode::Continuous, 16).await?;
    let continuous_url = format!("http://{}", continuous.addr);
    let continuous_output = common::non_stream_completion(&continuous_url, prompt, 12, 99).await?;
    continuous.stop().await?;

    let naive = common::spawn(ServeMode::Naive, 16).await?;
    let naive_url = format!("http://{}", naive.addr);
    let naive_output = common::non_stream_completion(&naive_url, prompt, 12, 99).await?;
    naive.stop().await?;

    assert_eq!(naive_output, continuous_output);
    Ok(())
}

async fn sampled_completion(
    base_url: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
) -> Result<String> {
    let value: serde_json::Value = Client::new()
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    Ok(value["choices"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string())
}
