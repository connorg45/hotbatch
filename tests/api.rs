mod common;

use anyhow::{Context, Result};
use hotbatch_server::ServeMode;
use reqwest::{Client, Response, StatusCode};
use serde_json::{json, Value};

async fn assert_api_error(
    response: Response,
    status: StatusCode,
    param: Option<&str>,
    code: &str,
) -> Result<Value> {
    assert_eq!(response.status(), status);
    assert_eq!(
        response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("application/json")
    );
    let body: Value = response.json().await.context("error response JSON")?;
    let error = body.get("error").context("error envelope")?;
    assert!(error.get("message").is_some_and(Value::is_string));
    assert!(error.get("type").is_some_and(Value::is_string));
    assert_eq!(error.get("param").and_then(Value::as_str), param);
    assert_eq!(error.get("code").and_then(Value::as_str), Some(code));
    Ok(body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_models_completion_and_chat_schemas() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 8).await?;
    let base_url = format!("http://{}", server.addr);
    let client = Client::new();

    let health = client.get(format!("{base_url}/healthz")).send().await?;
    assert_eq!(health.status(), StatusCode::OK);
    assert_eq!(health.text().await?, "ok");

    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(models.get("object").and_then(Value::as_str), Some("list"));
    let model = models["data"][0]["id"]
        .as_str()
        .context("model id")?
        .to_string();
    assert_eq!(models["data"][0]["object"], "model");

    let completion: Value = client
        .post(format!("{base_url}/v1/completions"))
        .json(&json!({
            "model": model,
            "prompt": ["The capital of France is"],
            "max_tokens": 1,
            "temperature": 0
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(completion["object"], "text_completion");
    assert_eq!(completion["model"], model);
    assert!(completion["id"]
        .as_str()
        .is_some_and(|id| id.starts_with("cmpl-")));
    assert!(completion["choices"][0]["text"].is_string());
    assert_eq!(completion["choices"][0]["finish_reason"], "length");
    assert_eq!(completion["usage"]["completion_tokens"], 1);
    assert_eq!(
        completion["usage"]["total_tokens"].as_u64(),
        completion["usage"]["prompt_tokens"]
            .as_u64()
            .map(|tokens| tokens + 1)
    );

    let chat: Value = client
        .post(format!("{base_url}/v1/chat/completions"))
        .json(&json!({
            "model": model,
            "messages": [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "Say hello."}
            ],
            "max_tokens": 1,
            "temperature": 0
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    assert_eq!(chat["object"], "chat.completion");
    assert_eq!(chat["model"], model);
    assert_eq!(chat["choices"][0]["message"]["role"], "assistant");
    assert!(chat["choices"][0]["message"]["content"].is_string());
    assert_eq!(chat["choices"][0]["finish_reason"], "length");

    server.stop().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_and_invalid_requests_use_openai_errors() -> Result<()> {
    let server = common::spawn(ServeMode::Continuous, 8).await?;
    let base_url = format!("http://{}", server.addr);
    let client = Client::new();
    let endpoint = format!("{base_url}/v1/completions");

    let malformed = client
        .post(&endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body("{\"prompt\":")
        .send()
        .await?;
    assert_api_error(malformed, StatusCode::BAD_REQUEST, None, "invalid_json").await?;

    let missing_content_type = client.post(&endpoint).body("{}").send().await?;
    assert_api_error(
        missing_content_type,
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        None,
        "invalid_json",
    )
    .await?;

    let oversized = format!("{{\"prompt\":\"{}\"}}", "x".repeat(2 * 1024 * 1024));
    let oversized = client
        .post(&endpoint)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(oversized)
        .send()
        .await?;
    assert_api_error(
        oversized,
        StatusCode::PAYLOAD_TOO_LARGE,
        None,
        "invalid_json",
    )
    .await?;

    let invalid_cases = [
        (json!({"prompt": ""}), "prompt", "invalid_prompt"),
        (json!({"prompt": []}), "prompt", "invalid_prompt"),
        (
            json!({"prompt": ["one", "two"]}),
            "prompt",
            "multiple_prompts_not_supported",
        ),
        (
            json!({"prompt": "hello", "max_tokens": 0}),
            "max_tokens",
            "invalid_max_tokens",
        ),
        (
            json!({"prompt": "hello", "max_tokens": 9}),
            "max_tokens",
            "max_tokens_exceeded",
        ),
        (
            json!({"prompt": "hello", "temperature": -0.1}),
            "temperature",
            "invalid_temperature",
        ),
        (
            json!({"prompt": "hello", "temperature": 2.1}),
            "temperature",
            "invalid_temperature",
        ),
        (
            json!({"prompt": "hello", "top_p": 0}),
            "top_p",
            "invalid_top_p",
        ),
        (
            json!({"prompt": "hello", "top_p": 1.1}),
            "top_p",
            "invalid_top_p",
        ),
        (
            json!({"prompt": "hello", "top_k": 0}),
            "top_k",
            "invalid_top_k",
        ),
        (
            json!({"prompt": "hello", "stop": ""}),
            "stop",
            "invalid_stop",
        ),
        (
            json!({"prompt": "hello", "stop": []}),
            "stop",
            "invalid_stop",
        ),
        (
            json!({"prompt": "hello", "stop": ["1", "2", "3", "4", "5"]}),
            "stop",
            "too_many_stop_sequences",
        ),
    ];

    for (body, param, code) in invalid_cases {
        let response = client.post(&endpoint).json(&body).send().await?;
        assert_api_error(response, StatusCode::BAD_REQUEST, Some(param), code).await?;
    }

    let missing_model = client
        .post(&endpoint)
        .json(&json!({"model": "not-loaded", "prompt": "hello"}))
        .send()
        .await?;
    assert_api_error(
        missing_model,
        StatusCode::NOT_FOUND,
        Some("model"),
        "model_not_found",
    )
    .await?;

    let sequence_limit = client
        .post(&endpoint)
        .json(&json!({
            "prompt": " x".repeat(300),
            "max_tokens": 1
        }))
        .send()
        .await?;
    assert_api_error(
        sequence_limit,
        StatusCode::BAD_REQUEST,
        Some("max_tokens"),
        "context_length_exceeded",
    )
    .await?;

    let chat_endpoint = format!("{base_url}/v1/chat/completions");
    let invalid_messages = [
        (json!({"messages": []}), "invalid_messages"),
        (
            json!({"messages": [{"role": "tool", "content": "hello"}]}),
            "invalid_role",
        ),
        (
            json!({"messages": [{"role": "user", "content": "  "}]}),
            "invalid_message_content",
        ),
    ];
    for (body, code) in invalid_messages {
        let response = client.post(&chat_endpoint).json(&body).send().await?;
        assert_api_error(response, StatusCode::BAD_REQUEST, Some("messages"), code).await?;
    }

    server.stop().await
}
