use crate::naive::NaiveSubmitError;
use crate::sse::{openai_stream, FilterUpdate, StreamKind, TextOutputFilter};
use crate::{AppState, Engine};
use axum::extract::rejection::JsonRejection;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use hotbatch_core::model::normalize_model_name;
use hotbatch_core::{FinishReason, GenerationHandle, GenerationRequest, SamplerConfig, StreamItem};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const DEFAULT_MAX_TOKENS: usize = 16;
const MAX_STOP_SEQUENCES: usize = 4;
const RESPONSE_CHANNEL_CAPACITY: usize = 32;

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
    fn into_single_prompt(self) -> Result<String, ApiError> {
        match self {
            Self::String(prompt) => Ok(prompt),
            Self::Strings(mut prompts) if prompts.len() == 1 => Ok(prompts.remove(0)),
            Self::Strings(prompts) if prompts.is_empty() => Err(ApiError::invalid(
                "prompt",
                "prompt array must contain exactly one nonblank string",
                "invalid_prompt",
            )),
            Self::Strings(_) => Err(ApiError::invalid(
                "prompt",
                "multiple prompts are not supported; provide a single prompt",
                "multiple_prompts_not_supported",
            )),
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

#[derive(Debug, Serialize)]
pub struct ApiError {
    #[serde(skip)]
    status: StatusCode,
    pub error: ApiErrorDetail,
}

#[derive(Debug, Serialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub param: Option<String>,
    pub code: &'static str,
}

impl ApiError {
    fn new(
        status: StatusCode,
        message: impl Into<String>,
        kind: &'static str,
        param: Option<&str>,
        code: &'static str,
    ) -> Self {
        Self {
            status,
            error: ApiErrorDetail {
                message: message.into(),
                kind,
                param: param.map(ToOwned::to_owned),
                code,
            },
        }
    }

    fn invalid(param: &'static str, message: impl Into<String>, code: &'static str) -> Self {
        Self::new(
            StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            Some(param),
            code,
        )
    }

    fn model_not_found(model: &str) -> Self {
        Self::new(
            StatusCode::NOT_FOUND,
            format!("The model '{model}' does not exist or is not loaded"),
            "invalid_request_error",
            Some("model"),
            "model_not_found",
        )
    }

    fn queue_full() -> Self {
        Self::new(
            StatusCode::TOO_MANY_REQUESTS,
            "request queue is full; retry later",
            "rate_limit_error",
            None,
            "queue_full",
        )
    }

    fn unavailable() -> Self {
        Self::new(
            StatusCode::SERVICE_UNAVAILABLE,
            "generation scheduler is unavailable",
            "server_error",
            None,
            "scheduler_unavailable",
        )
    }

    pub(crate) fn generation(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
            "server_error",
            None,
            "generation_error",
        )
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            message,
            "server_error",
            None,
            "internal_error",
        )
    }

    fn malformed_json(rejection: JsonRejection) -> Self {
        let status = match rejection.status() {
            StatusCode::PAYLOAD_TOO_LARGE => StatusCode::PAYLOAD_TOO_LARGE,
            StatusCode::UNSUPPORTED_MEDIA_TYPE => StatusCode::UNSUPPORTED_MEDIA_TYPE,
            _ => StatusCode::BAD_REQUEST,
        };
        Self::new(
            status,
            format!("Invalid JSON request: {}", rejection.body_text()),
            "invalid_request_error",
            None,
            "invalid_json",
        )
    }

    pub(crate) fn into_json_string(self) -> String {
        serde_json::to_string(&self).unwrap_or_else(|_| {
            "{\"error\":{\"message\":\"generation failed\",\"type\":\"server_error\",\"param\":null,\"code\":\"internal_error\"}}".to_string()
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self)).into_response()
    }
}

pub async fn healthz(State(state): State<AppState>) -> Response {
    if state.alive.load(Ordering::SeqCst) {
        "ok".into_response()
    } else {
        ApiError::unavailable().into_response()
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
    payload: Result<Json<CompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return ApiError::malformed_json(rejection).into_response(),
    };

    let prompt = match req.prompt.into_single_prompt() {
        Ok(prompt) if !prompt.trim().is_empty() => prompt,
        Ok(_) => {
            return ApiError::invalid(
                "prompt",
                "prompt must be a nonblank string",
                "invalid_prompt",
            )
            .into_response()
        }
        Err(error) => return error.into_response(),
    };
    let params = match validate_options(
        &state,
        req.model.as_deref(),
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.stop,
        req.seed,
        req.priority,
    ) {
        Ok(params) => params,
        Err(error) => return error.into_response(),
    };
    let stream = req.stream.unwrap_or(false);

    match submit_prompt(&state, prompt, params).await {
        Ok((handle, _, stops)) if stream => stream_response(
            format!("cmpl-{}", handle.id),
            handle,
            state.tokenizer.clone(),
            state.model_name.clone(),
            StreamKind::Completion,
            stops,
        ),
        Ok((handle, prompt_tokens, stops)) => {
            collect_completion(
                format!("cmpl-{}", handle.id),
                handle,
                state,
                prompt_tokens,
                stops,
            )
            .await
        }
        Err(error) => error.into_response(),
    }
}

pub async fn chat_completions(
    State(state): State<AppState>,
    payload: Result<Json<ChatCompletionRequest>, JsonRejection>,
) -> Response {
    let Json(req) = match payload {
        Ok(payload) => payload,
        Err(rejection) => return ApiError::malformed_json(rejection).into_response(),
    };

    if req.messages.is_empty() {
        return ApiError::invalid(
            "messages",
            "messages must contain at least one message",
            "invalid_messages",
        )
        .into_response();
    }
    for (index, message) in req.messages.iter().enumerate() {
        if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
            return ApiError::invalid(
                "messages",
                format!("messages[{index}].role must be one of system, user, or assistant"),
                "invalid_role",
            )
            .into_response();
        }
        if message.content.trim().is_empty() {
            return ApiError::invalid(
                "messages",
                format!("messages[{index}].content must be nonblank"),
                "invalid_message_content",
            )
            .into_response();
        }
    }

    let params = match validate_options(
        &state,
        req.model.as_deref(),
        req.max_tokens,
        req.temperature,
        req.top_p,
        req.top_k,
        req.stop,
        req.seed,
        req.priority,
    ) {
        Ok(params) => params,
        Err(error) => return error.into_response(),
    };
    let messages: Vec<hotbatch_core::model::ChatMessage> = req
        .messages
        .into_iter()
        .map(|message| hotbatch_core::model::ChatMessage {
            role: message.role,
            content: message.content,
        })
        .collect();
    let prompt = state.tokenizer.chat_template(&messages);
    let stream = req.stream.unwrap_or(false);

    match submit_prompt(&state, prompt, params).await {
        Ok((handle, _, stops)) if stream => stream_response(
            format!("chatcmpl-{}", handle.id),
            handle,
            state.tokenizer.clone(),
            state.model_name.clone(),
            StreamKind::Chat,
            stops,
        ),
        Ok((handle, prompt_tokens, stops)) => {
            collect_chat_completion(
                format!("chatcmpl-{}", handle.id),
                handle,
                state,
                prompt_tokens,
                stops,
            )
            .await
        }
        Err(error) => error.into_response(),
    }
}

struct SubmitParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: Option<usize>,
    stops: Vec<String>,
    seed: u64,
    priority: u8,
}

#[allow(clippy::too_many_arguments)]
fn validate_options(
    state: &AppState,
    requested_model: Option<&str>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop: Option<StopField>,
    seed: Option<u64>,
    priority: Option<u8>,
) -> Result<SubmitParams, ApiError> {
    if let Some(model) = requested_model {
        if normalize_model_name(model).ok() != Some(state.model_name.as_str()) {
            return Err(ApiError::model_not_found(model));
        }
    }

    let max_tokens = max_tokens.unwrap_or(DEFAULT_MAX_TOKENS.min(state.max_new_tokens));
    if max_tokens == 0 {
        return Err(ApiError::invalid(
            "max_tokens",
            "max_tokens must be greater than zero",
            "invalid_max_tokens",
        ));
    }
    if max_tokens > state.max_new_tokens {
        return Err(ApiError::invalid(
            "max_tokens",
            format!(
                "max_tokens cannot exceed the configured limit of {}",
                state.max_new_tokens
            ),
            "max_tokens_exceeded",
        ));
    }

    let temperature = temperature.unwrap_or(0.0);
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::invalid(
            "temperature",
            "temperature must be finite and between 0 and 2",
            "invalid_temperature",
        ));
    }

    let top_p = top_p.unwrap_or(1.0);
    if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
        return Err(ApiError::invalid(
            "top_p",
            "top_p must be finite, greater than 0, and at most 1",
            "invalid_top_p",
        ));
    }

    if top_k == Some(0) {
        return Err(ApiError::invalid(
            "top_k",
            "top_k must be greater than zero",
            "invalid_top_k",
        ));
    }

    let stops = match stop {
        Some(stop) => {
            let stops = stop.into_vec();
            if stops.is_empty() {
                return Err(ApiError::invalid(
                    "stop",
                    "stop must contain at least one sequence when provided",
                    "invalid_stop",
                ));
            }
            stops
        }
        None => Vec::new(),
    };
    if stops.len() > MAX_STOP_SEQUENCES {
        return Err(ApiError::invalid(
            "stop",
            format!("stop supports at most {MAX_STOP_SEQUENCES} sequences"),
            "too_many_stop_sequences",
        ));
    }
    if stops.iter().any(|stop| stop.trim().is_empty()) {
        return Err(ApiError::invalid(
            "stop",
            "stop sequences must be nonblank strings",
            "invalid_stop",
        ));
    }

    Ok(SubmitParams {
        max_tokens,
        temperature,
        top_p,
        top_k,
        stops,
        seed: seed.unwrap_or(0),
        priority: priority.unwrap_or(5),
    })
}

async fn submit_prompt(
    state: &AppState,
    prompt: String,
    params: SubmitParams,
) -> Result<(GenerationHandle, usize, Vec<String>), ApiError> {
    if !state.alive.load(Ordering::SeqCst) {
        return Err(ApiError::unavailable());
    }

    let prompt_tokens = state.tokenizer.encode(&prompt).map_err(|error| {
        ApiError::invalid(
            "prompt",
            format!("prompt could not be tokenized: {error}"),
            "invalid_prompt",
        )
    })?;
    if prompt_tokens.is_empty() {
        return Err(ApiError::invalid(
            "prompt",
            "prompt must produce at least one token",
            "invalid_prompt",
        ));
    }
    if prompt_tokens.len().saturating_add(params.max_tokens) > state.max_seq_len {
        return Err(ApiError::invalid(
            "max_tokens",
            format!(
                "prompt has {} tokens and max_tokens is {}; their sum exceeds the {}-token sequence limit",
                prompt_tokens.len(),
                params.max_tokens,
                state.max_seq_len
            ),
            "context_length_exceeded",
        ));
    }

    // A small bounded channel limits per-request buffering. The scheduler uses
    // nonblocking sends and evicts a request if its consumer cannot keep up.
    // Textual stop matching happens while the receiver drains this channel, so
    // every sampled token remains visible even when a stop falls inside a BPE
    // token or overlaps another stop sequence.
    let output_capacity = params
        .max_tokens
        .checked_add(1)
        .ok_or_else(|| ApiError::internal("response channel capacity overflow"))?;
    let channel_capacity = output_capacity.clamp(2, RESPONSE_CHANNEL_CAPACITY);
    let (sender, receiver) = mpsc::channel(channel_capacity);
    let id = Uuid::new_v4();
    let response_done = CancellationToken::new();
    let request = GenerationRequest {
        id,
        prompt_hash: hash_tokens(&prompt_tokens),
        prompt_tokens: prompt_tokens.clone(),
        sampler_config: SamplerConfig {
            temperature: params.temperature,
            top_p: params.top_p,
            top_k: params.top_k,
            stop_sequences: Vec::new(),
            max_new_tokens: params.max_tokens,
            eos_token: state.tokenizer.eos_token(),
            seed: params.seed,
        },
        sender,
        priority: params.priority,
        created_at: Instant::now(),
        response_done: response_done.clone(),
    };

    if !state.alive.load(Ordering::SeqCst) {
        return Err(ApiError::unavailable());
    }
    let handle = match &state.engine {
        Engine::Continuous { queue } => {
            queue.submit(request).map_err(|_| {
                if queue.is_closed() {
                    ApiError::unavailable()
                } else {
                    ApiError::queue_full()
                }
            })?;
            GenerationHandle {
                id,
                receiver,
                response_done,
            }
        }
        Engine::Naive(engine) => engine
            .submit(request, receiver)
            .map_err(|error| match error {
                NaiveSubmitError::QueueFull => ApiError::queue_full(),
                NaiveSubmitError::Unavailable => ApiError::unavailable(),
            })?,
    };

    Ok((handle, prompt_tokens.len(), params.stops))
}

fn stream_response(
    id: String,
    handle: GenerationHandle,
    tokenizer: hotbatch_core::TokenizerBundle,
    model: String,
    kind: StreamKind,
    stops: Vec<String>,
) -> Response {
    let GenerationHandle {
        receiver,
        response_done,
        ..
    } = handle;
    Sse::new(openai_stream(
        id,
        receiver,
        tokenizer,
        model,
        kind,
        stops,
        response_done,
    ))
    .keep_alive(KeepAlive::default())
    .into_response()
}

struct CollectedGeneration {
    text: String,
    completion_tokens: usize,
    reason: FinishReason,
}

async fn collect_generation(
    mut handle: GenerationHandle,
    tokenizer: &hotbatch_core::TokenizerBundle,
    stops: Vec<String>,
) -> Result<CollectedGeneration, ApiError> {
    let mut tokens = Vec::new();
    let mut text = String::new();
    let mut output = TextOutputFilter::new(stops);
    loop {
        match handle.receiver.recv().await {
            Some(StreamItem::Token(token)) => {
                tokens.push(token);
                let decoded = tokenizer.decode(&tokens).map_err(|error| {
                    ApiError::generation(format!("generated tokens could not be decoded: {error}"))
                })?;
                let stable = decoded.trim_end_matches('\u{fffd}');
                match output.push(stable).map_err(ApiError::generation)? {
                    FilterUpdate::Continue(fragment) => text.push_str(&fragment),
                    FilterUpdate::Stop(fragment) => {
                        text.push_str(&fragment);
                        handle.stop_generation();
                        return Ok(CollectedGeneration {
                            text,
                            completion_tokens: tokens.len(),
                            reason: FinishReason::Stop,
                        });
                    }
                }
            }
            Some(StreamItem::Finished(reason)) => {
                let decoded = tokenizer.decode(&tokens).map_err(|error| {
                    ApiError::generation(format!("generated tokens could not be decoded: {error}"))
                })?;
                let remaining = output.finish(&decoded).map_err(ApiError::generation)?;
                text.push_str(&remaining);
                return Ok(CollectedGeneration {
                    text,
                    completion_tokens: tokens.len(),
                    reason,
                });
            }
            Some(StreamItem::Error(message)) => {
                return Err(ApiError::generation(format!(
                    "generation failed: {message}"
                )))
            }
            None => {
                return Err(ApiError::generation(
                    "generation ended before a terminal event",
                ))
            }
        }
    }
}

async fn collect_completion(
    id: String,
    handle: GenerationHandle,
    state: AppState,
    prompt_tokens: usize,
    stops: Vec<String>,
) -> Response {
    let generation = match collect_generation(handle, &state.tokenizer, stops).await {
        Ok(result) => result,
        Err(error) => return error.into_response(),
    };
    Json(json!({
        "id": id,
        "object": "text_completion",
        "model": state.model_name,
        "choices": [{
            "index": 0,
            "text": generation.text,
            "finish_reason": finish_reason_label(generation.reason)
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": prompt_tokens + generation.completion_tokens
        }
    }))
    .into_response()
}

async fn collect_chat_completion(
    id: String,
    handle: GenerationHandle,
    state: AppState,
    prompt_tokens: usize,
    stops: Vec<String>,
) -> Response {
    let generation = match collect_generation(handle, &state.tokenizer, stops).await {
        Ok(result) => result,
        Err(error) => return error.into_response(),
    };
    Json(json!({
        "id": id,
        "object": "chat.completion",
        "model": state.model_name,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generation.text
            },
            "finish_reason": finish_reason_label(generation.reason)
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": prompt_tokens + generation.completion_tokens
        }
    }))
    .into_response()
}

pub(crate) fn finish_reason_label(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Length => "length",
        FinishReason::Stop => "stop",
    }
}

fn hash_tokens(tokens: &[u32]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for token in tokens {
        hash ^= *token as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}
