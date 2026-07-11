#![allow(dead_code)]

use anyhow::{anyhow, Context, Result};
use futures_util::StreamExt;
use hotbatch_server::{RunningServer, ServeArgs, ServeMode};
use reqwest::Client;
use serde_json::json;
use std::net::Ipv4Addr;
use std::time::Duration;

static MODEL_LOAD_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

pub async fn spawn(
    mode: ServeMode,
    max_tokens: usize,
    forward_base_us: u64,
) -> Result<RunningServer> {
    let _guard = MODEL_LOAD_LOCK.lock().await;
    let host =
        std::env::var("HOTBATCH_TEST_HOST").unwrap_or_else(|_| Ipv4Addr::LOCALHOST.to_string());
    hotbatch_server::spawn_server(ServeArgs {
        host,
        port: 0,
        mode,
        model: std::env::var("HOTBATCH_TEST_MODEL").unwrap_or_else(|_| "gpt2".to_string()),
        device: "cpu".to_string(),
        max_running_seqs: 8,
        max_queue_depth: 128,
        max_seq_len: 256,
        max_new_tokens: max_tokens,
        forward_base_us,
        forward_per_seq_us: 50,
    })
    .await
}

pub async fn non_stream_completion(
    base_url: &str,
    prompt: &str,
    max_tokens: usize,
    seed: u64,
) -> Result<String> {
    let value: serde_json::Value = Client::new()
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": "gpt2",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed
        }))
        .send()
        .await
        .context("sending completion")?
        .error_for_status()
        .context("completion status")?
        .json()
        .await
        .context("completion JSON")?;
    value
        .get("choices")
        .and_then(|choices| choices.get(0))
        .and_then(|choice| choice.get("text"))
        .and_then(|text| text.as_str())
        .map(ToString::to_string)
        .ok_or_else(|| anyhow!("missing completion text: {value}"))
}

pub async fn collect_sse(
    base_url: &str,
    prompt: &str,
    max_tokens: usize,
    seed: u64,
) -> Result<Vec<String>> {
    let response = Client::new()
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": "gpt2",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": true,
            "temperature": 0,
            "seed": seed
        }))
        .send()
        .await
        .context("sending streaming completion")?
        .error_for_status()
        .context("stream status")?;
    parse_sse(response).await
}

pub async fn parse_sse(response: reqwest::Response) -> Result<Vec<String>> {
    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut frames = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("reading SSE chunk")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(frame_end) = buffer.find("\n\n") {
            let frame = buffer[..frame_end].to_string();
            buffer = buffer[frame_end + 2..].to_string();
            for line in frame.lines() {
                if let Some(data) = line.strip_prefix("data:") {
                    let data = data.trim().to_string();
                    frames.push(data.clone());
                    if data == "[DONE]" {
                        return Ok(frames);
                    }
                }
            }
        }
    }
    Ok(frames)
}

pub async fn metrics(base_url: &str) -> Result<String> {
    Client::new()
        .get(format!("{base_url}/metrics"))
        .send()
        .await
        .context("requesting metrics")?
        .error_for_status()
        .context("metrics status")?
        .text()
        .await
        .context("metrics body")
}

pub fn metric_value(metrics: &str, name: &str) -> f64 {
    for line in metrics.lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(metric_name) = parts.next() else {
            continue;
        };
        if metric_name != name {
            continue;
        }
        let Some(value) = parts.next() else {
            continue;
        };
        if let Ok(parsed) = value.parse::<f64>() {
            return parsed;
        }
    }
    0.0
}

pub async fn wait_for_metric_at_least(base_url: &str, name: &str, target: f64) -> Result<f64> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let body = metrics(base_url).await?;
        let value = metric_value(&body, name);
        if value >= target {
            return Ok(value);
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(anyhow!(
                "metric {name} did not reach {target}; last value={value}\n{body}"
            ));
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
}
