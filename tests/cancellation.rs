mod common;

use anyhow::{Context, Result};
use futures_util::StreamExt;
use hotbatch_server::ServeMode;
use reqwest::Client;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn client_disconnect_evicts_sequence_within_one_step() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 200, 10_000).await?;
    let base_url = format!("http://{}", server.addr);
    let response = Client::new()
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": "gpt2",
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
