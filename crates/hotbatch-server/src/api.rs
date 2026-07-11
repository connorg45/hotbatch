use crate::sse::{openai_stream, StreamKind};
use crate::{AppState, Engine};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use hotbatch_core::{GenerationHandle, GenerationRequest, SamplerConfig, StreamItem};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: PromptField,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop: Option<StopField>,
    pub stream: Option<bool>,
    pub seed: Option<u64>,
    pub priority: Option<u8>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum PromptField {
    String(String),
    Strings(Vec<String>),
}

impl PromptField {
    fn into_prompt(self) -> String {
        match self {
            Self::String(prompt) => prompt,
            Self::Strings(prompts) => prompts.join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopField {
    String(String),
    Strings(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::String(stop) => vec![stop],
            Self::Strings(stops) => stops,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop: Option<StopField>,
    pub stream: Option<bool>,
    pub seed: Option<u64>,
    pub priority: Option<u8>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub async fn healthz(State(state): State<AppState>) -> Response {
    if state.alive.load(Ordering::SeqCst) {
        "ok".into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "scheduler stopped").into_response()
    }
}

pub async fn models(State(state): State<AppState>) -> Response {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": 0,
            "owned_by": "hotbatch"
        }]
    }))
    .into_response()
}

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let _requested_model = req.model.as_deref();
    let prompt = req.prompt.into_prompt();
    let max_tokens = req.max_tokens.unwrap_or(16).min(512);
    let stream = req.stream.unwrap_or(false);
    match submit_prompt(
        &state,
        SubmitParams {
            prompt,
            max_tokens,
            channel_capacity: if stream {
                32
            } else {
                max_tokens.saturating_add(4)
            },
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k,
            stops: req.stop.map(StopField::into_vec).unwrap_or_default(),
            seed: req.seed.unwrap_or(0),
            priority: req.priority.unwrap_or(5),
        },
    )
    .await
    {
        Ok((handle, prompt_tokens)) => {
            if stream {
                stream_response(
                    format!("cmpl-{}", handle.id),
                    handle,
                    state.tokenizer.clone(),
                    StreamKind::Completion,
                )
            } else {
                collect_completion(format!("cmpl-{}", handle.id), handle, state, prompt_tokens)
                    .await
            }
        }
        Err(response) => response,
    }
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let _requested_model = req.model.as_deref();
    let messages: Vec<hotbatch_core::model::ChatMessage> = req
        .messages
        .iter()
        .map(|message| hotbatch_core::model::ChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
        })
        .collect();
    let prompt = state.tokenizer.chat_template(&messages);
    let max_tokens = req.max_tokens.unwrap_or(16).min(512);
    let stream = req.stream.unwrap_or(false);
    match submit_prompt(
        &state,
        SubmitParams {
            prompt,
            max_tokens,
            channel_capacity: if stream {
                32
            } else {
                max_tokens.saturating_add(4)
            },
            temperature: req.temperature.unwrap_or(0.0),
            top_p: req.top_p.unwrap_or(1.0),
            top_k: req.top_k,
            stops: req.stop.map(StopField::into_vec).unwrap_or_default(),
            seed: req.seed.unwrap_or(0),
            priority: req.priority.unwrap_or(5),
        },
    )
    .await
    {
        Ok((handle, prompt_tokens)) => {
            if stream {
                stream_response(
                    format!("chatcmpl-{}", handle.id),
                    handle,
                    state.tokenizer.clone(),
                    StreamKind::Chat,
                )
            } else {
                collect_chat_completion(
                    format!("chatcmpl-{}", handle.id),
                    handle,
                    state,
                    prompt_tokens,
                )
                .await
            }
        }
        Err(response) => response,
    }
}

struct SubmitParams {
    prompt: String,
    max_tokens: usize,
    channel_capacity: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    stops: Vec<String>,
    seed: u64,
    priority: u8,
}

async fn submit_prompt(
    state: &AppState,
    params: SubmitParams,
) -> Result<(GenerationHandle, usize), Response> {
    let prompt_tokens = state.tokenizer.encode(&params.prompt).map_err(|err| {
        error_response(StatusCode::BAD_REQUEST, &format!("invalid prompt: {err}"))
    })?;
    let stop_sequences = state
        .tokenizer
        .stop_sequences(&params.stops)
        .map_err(|err| error_response(StatusCode::BAD_REQUEST, &format!("invalid stop: {err}")))?;
    let (sender, receiver) = mpsc::channel(params.channel_capacity.max(32));
    let id = Uuid::new_v4();
    let req = GenerationRequest {
        id,
        prompt_hash: hash_tokens(&prompt_tokens),
        prompt_tokens: prompt_tokens.clone(),
        sampler_config: SamplerConfig {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            stop_sequences,
            max_new_tokens: params.max_tokens.max(1),
            eos_token: state.tokenizer.eos_token(),
            seed: params.seed,
        },
        sender,
        priority: params.priority,
        created_at: Instant::now(),
    };

    let handle = match &state.engine {
        Engine::Continuous { queue } => {
            queue.submit(req).map_err(|_| {
                error_response(StatusCode::TOO_MANY_REQUESTS, "request queue is full")
            })?;
            GenerationHandle { id, receiver }
        }
        Engine::Naive(engine) => engine
            .submit(req, receiver)
            .map_err(|err| error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()))?,
    };

    Ok((handle, prompt_tokens.len()))
}

fn stream_response(
    id: String,
    handle: GenerationHandle,
    tokenizer: hotbatch_core::TokenizerBundle,
    kind: StreamKind,
) -> Response {
    Sse::new(openai_stream(id, handle.receiver, tokenizer, kind))
        .keep_alive(KeepAlive::default())
        .into_response()
}

async fn collect_completion(
    id: String,
    mut handle: GenerationHandle,
    state: AppState,
    prompt_tokens: usize,
) -> Response {
    let mut tokens = Vec::new();
    while let Some(item) = handle.receiver.recv().await {
        match item {
            StreamItem::Token(token) => tokens.push(token),
            StreamItem::Done => break,
        }
    }
    let text = match state.tokenizer.decode(&tokens) {
        Ok(text) => text,
        Err(err) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    };
    Json(json!({
        "id": id,
        "object": "text_completion",
        "model": state.model_name,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens.len(),
            "total_tokens": prompt_tokens + tokens.len()
        }
    }))
    .into_response()
}

async fn collect_chat_completion(
    id: String,
    mut handle: GenerationHandle,
    state: AppState,
    prompt_tokens: usize,
) -> Response {
    let mut tokens = Vec::new();
    while let Some(item) = handle.receiver.recv().await {
        match item {
            StreamItem::Token(token) => tokens.push(token),
            StreamItem::Done => break,
        }
    }
    let text = match state.tokenizer.decode(&tokens) {
        Ok(text) => text,
        Err(err) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err.to_string()),
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "model": state.model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": tokens.len(),
            "total_tokens": prompt_tokens + tokens.len()
        }
    }))
    .into_response()
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        Json(json!({
            "error": {
                "message": message,
                "type": "hotbatch_error",
                "code": status.as_u16()
            }
        })),
    )
        .into_response()
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        hash ^= *token as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
