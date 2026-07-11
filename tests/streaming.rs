mod common;

use anyhow::{Context, Result};
use hotbatch_server::ServeMode;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Barrier;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sse_end_to_end() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 8).await?;
    let base_url = format!("http://{}", server.addr);

    let prompt = "Once upon a time";
    let frames = common::collect_sse(&base_url, prompt, 6, 42).await?;
    assert!(frames
        .iter()
        .any(|frame| frame.contains("\"object\":\"text_completion\"")));
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));
    let mut streamed_text = String::new();
    for frame in frames.iter().filter(|frame| frame.as_str() != "[DONE]") {
        let payload: Value = serde_json::from_str(frame)?;
        if let Some(text) = payload["choices"][0]["text"].as_str() {
            streamed_text.push_str(text);
        }
    }
    let collected = common::non_stream_completion(&base_url, prompt, 6, 42).await?;
    assert_eq!(streamed_text, collected);

    server.stop().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_share_decode_forward_passes() -> Result<()> {
    const REQUEST_COUNT: usize = 4;

    let server = common::spawn(ServeMode::Continuous, 96).await?;
    let base_url = format!("http://{}", server.addr);
    let barrier = Arc::new(Barrier::new(REQUEST_COUNT + 1));
    let mut requests = Vec::with_capacity(REQUEST_COUNT);
    for index in 0..REQUEST_COUNT {
        let barrier = barrier.clone();
        let base_url = base_url.clone();
        requests.push(tokio::spawn(async move {
            barrier.wait().await;
            common::collect_sse(
                &base_url,
                &format!("Concurrent decode request {index}"),
                80,
                index as u64,
            )
            .await
        }));
    }
    barrier.wait().await;
    for request in requests {
        let frames = request.await.context("stream join")??;
        assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));
    }

    let metrics = common::metrics(&base_url).await?;
    assert!(
        common::metric_value(&metrics, "hotbatch_shared_decode_steps_total") > 0.0,
        "expected concurrent requests to overlap in a decode batch\n{metrics}"
    );

    server.stop().await
}
