mod common;

use anyhow::{Context, Result};
use hotbatch_server::{ServeArgs, ServeMode};
use reqwest::{Client, Response, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;

static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

async fn spawn_naive(max_queue_depth: usize) -> Result<hotbatch_server::RunningServer> {
    hotbatch_server::spawn_server(ServeArgs {
        host: "127.0.0.1".to_string(),
        port: 0,
        mode: ServeMode::Naive,
        model: "gpt2".to_string(),
        device: "cpu".to_string(),
        max_running_seqs: 1,
        max_queue_depth,
        max_seq_len: 256,
        max_new_tokens: 64,
    })
    .await
}

fn request_task(
    client: Client,
    endpoint: String,
    request_number: usize,
) -> tokio::task::JoinHandle<reqwest::Result<Response>> {
    tokio::spawn(async move {
        client
            .post(endpoint)
            .json(&json!({
                "prompt": format!(
                    "Keep the naive model occupied while request {request_number} waits"
                ),
                "max_tokens": 64,
                "temperature": 0,
                "seed": request_number
            }))
            .send()
            .await
    })
}

async fn task_response(
    task: tokio::task::JoinHandle<reqwest::Result<Response>>,
) -> Result<Response> {
    tokio::time::timeout(Duration::from_secs(3), task)
        .await
        .context("request did not resolve")?
        .context("request task failed")?
        .context("request transport failed")
}

async fn assert_shutdown_error(response: Response) -> Result<()> {
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value = response.json().await.context("shutdown error JSON")?;
    assert_eq!(body["error"]["type"], "server_error");
    assert_eq!(body["error"]["code"], "generation_error");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn naive_shutdown_resolves_running_and_model_lock_waiters() -> Result<()> {
    let _guard = TEST_LOCK.lock().await;
    const WAITER_COUNT: usize = 4;

    let server = spawn_naive(WAITER_COUNT).await?;
    let base_url = format!("http://{}", server.addr);
    let endpoint = format!("{base_url}/v1/completions");
    let client = Client::new();

    let running = request_task(client.clone(), endpoint.clone(), 0);
    common::wait_for_metric_at_least(&base_url, "hotbatch_running_sequences", 1.0).await?;

    let mut waiters = Vec::with_capacity(WAITER_COUNT);
    for request_number in 1..=WAITER_COUNT {
        waiters.push(request_task(
            client.clone(),
            endpoint.clone(),
            request_number,
        ));
    }
    common::wait_for_metric_at_least(&base_url, "hotbatch_queue_depth", WAITER_COUNT as f64)
        .await?;

    tokio::time::timeout(Duration::from_secs(5), server.stop())
        .await
        .context("naive server shutdown timed out")??;

    assert_shutdown_error(task_response(running).await?).await?;
    for waiter in waiters {
        assert_shutdown_error(task_response(waiter).await?).await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn naive_queue_overflow_returns_openai_429() -> Result<()> {
    let _guard = TEST_LOCK.lock().await;
    let server = spawn_naive(1).await?;
    let base_url = format!("http://{}", server.addr);
    let endpoint = format!("{base_url}/v1/completions");
    let client = Client::new();

    let running = request_task(client.clone(), endpoint.clone(), 0);
    common::wait_for_metric_at_least(&base_url, "hotbatch_running_sequences", 1.0).await?;
    let waiting = request_task(client.clone(), endpoint.clone(), 1);
    common::wait_for_metric_at_least(&base_url, "hotbatch_queue_depth", 1.0).await?;

    let overflow = client
        .post(&endpoint)
        .json(&json!({
            "prompt": "This request exceeds the configured naive queue",
            "max_tokens": 1,
            "temperature": 0
        }))
        .send()
        .await
        .context("sending overflow request")?;
    assert_eq!(overflow.status(), StatusCode::TOO_MANY_REQUESTS);
    let body: Value = overflow.json().await.context("queue-full error JSON")?;
    assert_eq!(body["error"]["type"], "rate_limit_error");
    assert_eq!(body["error"]["param"], Value::Null);
    assert_eq!(body["error"]["code"], "queue_full");

    tokio::time::timeout(Duration::from_secs(5), server.stop())
        .await
        .context("naive server shutdown timed out")??;
    assert_shutdown_error(task_response(running).await?).await?;
    assert_shutdown_error(task_response(waiting).await?).await?;
    Ok(())
}
