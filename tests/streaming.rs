mod common;

use anyhow::{Context, Result};
use hotbatch_server::ServeMode;
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sse_end_to_end() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 8, 1_000).await?;
    let base_url = format!("http://{}", server.addr);

    let frames = common::collect_sse(&base_url, "Once upon a time", 6, 42).await?;
    assert!(frames
        .iter()
        .any(|frame| frame.contains("\"object\":\"text_completion\"")));
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));

    server.stop().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn staggered_requests_share_decode_forward_passes() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 48, 8_000).await?;
    let base_url = format!("http://{}", server.addr);
    let client = Client::new();

    let first_url = base_url.clone();
    let first = tokio::spawn(async move {
        common::collect_sse(&first_url, "Request one starts first", 40, 1).await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
    let second = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": "gpt2",
            "prompt": "Request two arrives later",
            "max_tokens": 24,
            "stream": true,
            "temperature": 0,
            "seed": 2
        }))
        .send()
        .await
        .context("sending second request")?
        .error_for_status()
        .context("second request status")?;
    let second_frames = common::parse_sse(second).await?;
    let first_frames = first.await.context("first stream join")??;

    assert_eq!(first_frames.last().map(String::as_str), Some("[DONE]"));
    assert_eq!(second_frames.last().map(String::as_str), Some("[DONE]"));
    let metrics = common::metrics(&base_url).await?;
    assert!(
        common::metric_value(&metrics, "hotbatch_shared_decode_steps_total") > 0.0,
        "expected staggered requests to overlap in a decode batch\n{metrics}"
    );

    server.stop().await
}
