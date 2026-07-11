mod common;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use hotbatch_server::{ServeArgs, ServeMode};
use reqwest::Client;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Barrier;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn connected_consumers_can_receive_more_than_channel_capacity() -> Result<()> {
    for mode in [ServeMode::Continuous, ServeMode::Naive] {
        let server = common::spawn(mode, 200).await?;
        let base_url = format!("http://{}", server.addr);
        let response = Client::new()
            .post(format!("{base_url}/v1/completions"))
            .json(&json!({
                "prompt": "Keep delivering tokens to this connected consumer",
                "max_tokens": 160,
                "temperature": 0,
                "seed": 123
            }))
            .send()
            .await
            .context("sending long connected request")?
            .error_for_status()
            .context("long connected request status")?
            .json::<serde_json::Value>()
            .await
            .context("long connected response JSON")?;
        assert_eq!(response["choices"][0]["finish_reason"], "length");
        assert_eq!(response["usage"]["completion_tokens"], 160);
        server.stop().await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_disconnect_evicts_sequence_within_one_step() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 200).await?;
    let base_url = format!("http://{}", server.addr);
    let response = Client::new()
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "prompt": "Cancel this stream after a few tokens",
            "max_tokens": 160,
            "stream": true,
            "temperature": 0,
            "seed": 123
        }))
        .send()
        .await
        .context("sending cancellable request")?
        .error_for_status()
        .context("cancellable status")?;

    let mut stream = response.bytes_stream();
    let mut chunks = 0_usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading cancellable stream")?;
        if String::from_utf8_lossy(&chunk).contains("data:") {
            chunks += 1;
        }
        if chunks >= 5 {
            break;
        }
    }
    let before = common::metrics(&base_url).await?;
    let before_steps = common::metric_value(&before, "hotbatch_scheduler_steps_total");
    drop(stream);

    common::wait_for_metric_at_least(&base_url, "hotbatch_cancelled_sequences_total", 1.0).await?;
    let after = common::metrics(&base_url).await?;
    let after_steps = common::metric_value(&after, "hotbatch_scheduler_steps_total");
    assert!(
        after_steps - before_steps <= 3.0,
        "cancellation should be observed promptly; steps before={before_steps}, after={after_steps}\n{after}"
    );
    assert_eq!(
        common::metric_value(&after, "hotbatch_running_sequences"),
        0.0
    );

    server.stop().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_resolves_running_and_queued_requests() -> Result<()> {
    const REQUEST_COUNT: usize = 12;
    const QUEUED_TARGET: f64 = 4.0;

    let server = hotbatch_server::spawn_server(ServeArgs {
        host: "127.0.0.1".to_string(),
        port: 0,
        mode: ServeMode::Continuous,
        model: "tiny-gpt2".to_string(),
        device: "cpu".to_string(),
        max_running_seqs: 1,
        max_queue_depth: 16,
        max_seq_len: 256,
        max_new_tokens: 200,
    })
    .await?;
    let base_url = format!("http://{}", server.addr);
    let endpoint = format!("{base_url}/v1/completions");
    let client = Client::new();
    let barrier = Arc::new(Barrier::new(REQUEST_COUNT + 1));
    let mut requests = Vec::with_capacity(REQUEST_COUNT);
    for index in 0..REQUEST_COUNT {
        let barrier = barrier.clone();
        let client = client.clone();
        let endpoint = endpoint.clone();
        requests.push(tokio::spawn(async move {
            barrier.wait().await;
            client
                .post(endpoint)
                .json(&json!({
                    "prompt": format!("Keep the running slot occupied while request {index} waits"),
                    "max_tokens": 160,
                    "temperature": 0
                }))
                .send()
                .await
        }));
    }
    barrier.wait().await;
    wait_for_queue_depth(&base_url, QUEUED_TARGET).await?;

    tokio::time::timeout(Duration::from_secs(5), server.stop())
        .await
        .context("server shutdown timed out")??;

    let mut shutdown_errors = 0_usize;
    for request in requests {
        let response = tokio::time::timeout(Duration::from_secs(2), request)
            .await
            .context("request task timed out after shutdown")?
            .context("request task join failed")?;
        if let Ok(response) = response {
            shutdown_errors +=
                usize::from(response.status() == reqwest::StatusCode::INTERNAL_SERVER_ERROR);
        }
    }
    assert!(
        shutdown_errors > QUEUED_TARGET as usize,
        "the running request and queued requests should receive shutdown errors; got {shutdown_errors}"
    );
    Ok(())
}

async fn wait_for_queue_depth(base_url: &str, target: f64) -> Result<()> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let body = common::metrics(base_url).await?;
        if common::metric_value(&body, "hotbatch_queue_depth") >= target {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("queue depth did not reach {target}; last metrics:\n{body}");
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
}
