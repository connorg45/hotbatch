mod common;

use anyhow::{Context, Result};
use hotbatch_server::ServeMode;
use reqwest::Client;
use serde_json::{json, Value};

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn continuous_length_stop_and_sse_done_are_reported() -> Result<()> {
    assert_finish_reasons(ServeMode::Continuous).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn naive_length_stop_and_sse_done_are_reported() -> Result<()> {
    assert_finish_reasons(ServeMode::Naive).await
}

async fn assert_finish_reasons(mode: ServeMode) -> Result<()> {
    let server = common::spawn(mode, 64).await?;
    let base_url = format!("http://{}", server.addr);
    let client = Client::new();
    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let model = models["data"][0]["id"].as_str().context("model id")?;
    let prompt = "The capital of France is";

    let first: Value = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "seed": 42
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(first["choices"][0]["finish_reason"], "length");
    let first_token = first["choices"][0]["text"]
        .as_str()
        .filter(|text| !text.is_empty())
        .context("first generated token must decode to text")?;

    let stopped: Value = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "seed": 42,
            "stop": first_token
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(stopped["choices"][0]["finish_reason"], "stop");
    assert_eq!(stopped["choices"][0]["text"], "");
    common::wait_for_metric_at_most(&base_url, "hotbatch_running_sequences", 0.0).await?;

    let split_at = first_token
        .char_indices()
        .last()
        .map(|(index, _)| index)
        .filter(|index| *index > 0)
        .context("first generated token must contain multiple characters")?;
    let inside_token_stop = &first_token[split_at..];
    let visible_prefix = &first_token[..split_at];
    let metrics_before = common::metrics(&base_url).await?;
    let tokens_before = common::metric_value(&metrics_before, "hotbatch_tokens_generated_total");
    let cancelled_before =
        common::metric_value(&metrics_before, "hotbatch_cancelled_sequences_total");
    let stopped_inside_token: Value = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 64,
            "temperature": 0,
            "seed": 42,
            "stop": inside_token_stop
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(stopped_inside_token["choices"][0]["finish_reason"], "stop");
    assert_eq!(stopped_inside_token["choices"][0]["text"], visible_prefix);
    common::wait_for_metric_at_most(&base_url, "hotbatch_running_sequences", 0.0).await?;
    let metrics_after = common::metrics(&base_url).await?;
    let generated =
        common::metric_value(&metrics_after, "hotbatch_tokens_generated_total") - tokens_before;
    assert!(
        generated <= 4.0,
        "a first-token stop should halt model work promptly; generated {generated} tokens\n{metrics_after}"
    );
    assert_eq!(
        common::metric_value(&metrics_after, "hotbatch_cancelled_sequences_total"),
        cancelled_before,
        "a matched stop is not a client disconnect"
    );

    let stopped_stream = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0,
            "seed": 42,
            "stop": inside_token_stop,
            "stream": true
        }))
        .send()
        .await?
        .error_for_status()?;
    let stopped_frames = common::parse_sse(stopped_stream).await?;
    assert_eq!(stopped_frames.last().map(String::as_str), Some("[DONE]"));
    let mut stopped_text = String::new();
    let mut stopped_reason = None;
    for frame in stopped_frames
        .iter()
        .filter(|frame| frame.as_str() != "[DONE]")
    {
        let payload: Value = serde_json::from_str(frame)?;
        if let Some(text) = payload["choices"][0]["text"].as_str() {
            stopped_text.push_str(text);
        }
        if let Some(reason) = payload["choices"][0]["finish_reason"].as_str() {
            stopped_reason = Some(reason.to_string());
        }
    }
    assert_eq!(stopped_text, visible_prefix);
    assert_eq!(stopped_reason.as_deref(), Some("stop"));

    let response = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "max_tokens": 2,
            "temperature": 0,
            "seed": 42,
            "stream": true
        }))
        .send()
        .await?
        .error_for_status()?;
    let frames = common::parse_sse(response).await?;
    assert_eq!(frames.last().map(String::as_str), Some("[DONE]"));
    let terminal: Value = serde_json::from_str(
        frames
            .iter()
            .rev()
            .find(|frame| frame.as_str() != "[DONE]")
            .context("terminal SSE frame")?,
    )?;
    assert_eq!(terminal["choices"][0]["finish_reason"], "length");

    server.stop().await
}
