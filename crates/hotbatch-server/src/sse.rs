use crate::api::{finish_reason_label, ApiError};
use axum::response::sse::Event;
use futures_util::Stream;
use hotbatch_core::{StreamItem, TokenizerBundle};
use serde_json::json;
use std::convert::Infallible;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Copy, Clone)]
pub enum StreamKind {
    Completion,
    Chat,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FilterUpdate {
    Continue(String),
    Stop(String),
}

/// Converts incrementally decoded model output into text that is safe to expose.
/// Trailing text that could still become a stop sequence is held until it either
/// matches or becomes ordinary output.
pub(crate) struct TextOutputFilter {
    stops: Vec<String>,
    emitted: String,
}

impl TextOutputFilter {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|stop| !stop.is_empty()).collect(),
            emitted: String::new(),
        }
    }

    pub(crate) fn push(&mut self, decoded: &str) -> Result<FilterUpdate, &'static str> {
        if !decoded.starts_with(&self.emitted) {
            return Err("incremental token decoding changed already emitted text");
        }

        let earliest_stop = self
            .stops
            .iter()
            .filter_map(|stop| decoded.find(stop))
            .min();
        if let Some(stop_at) = earliest_stop {
            if stop_at < self.emitted.len() {
                return Err("a stop sequence appeared in already emitted text");
            }
            let text = decoded[self.emitted.len()..stop_at].to_string();
            self.emitted.push_str(&text);
            return Ok(FilterUpdate::Stop(text));
        }

        let held_bytes = self
            .stops
            .iter()
            .map(|stop| {
                stop.char_indices()
                    .skip(1)
                    .map(|(end, _)| end)
                    .chain(std::iter::once(stop.len()))
                    .filter(|end| decoded.ends_with(&stop[..*end]))
                    .max()
                    .unwrap_or(0)
            })
            .max()
            .unwrap_or(0);
        let safe_end = decoded.len().saturating_sub(held_bytes);
        if safe_end < self.emitted.len() || !decoded.is_char_boundary(safe_end) {
            return Err("stop-sequence buffering crossed already emitted text");
        }
        let text = decoded[self.emitted.len()..safe_end].to_string();
        self.emitted.push_str(&text);
        Ok(FilterUpdate::Continue(text))
    }

    pub(crate) fn finish(&mut self, decoded: &str) -> Result<String, &'static str> {
        let Some(text) = decoded.strip_prefix(&self.emitted) else {
            return Err("incremental token decoding changed already emitted text");
        };
        let text = text.to_string();
        self.emitted.push_str(&text);
        Ok(text)
    }
}

pub fn openai_stream(
    id: String,
    mut receiver: mpsc::Receiver<StreamItem>,
    tokenizer: TokenizerBundle,
    model: String,
    kind: StreamKind,
    stops: Vec<String>,
    response_done: CancellationToken,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut tokens = Vec::new();
        let mut output = TextOutputFilter::new(stops);
        while let Some(item) = receiver.recv().await {
            match item {
                StreamItem::Token(token) => {
                    tokens.push(token);
                    let decoded = match tokenizer.decode(&tokens) {
                        Ok(decoded) => decoded,
                        Err(error) => {
                            let payload = ApiError::generation(format!(
                                "generated tokens could not be decoded: {error}"
                            ))
                            .into_json_string();
                            yield Ok(Event::default().event("error").data(payload));
                            return;
                        }
                    };
                    // Byte-level GPT-2 tokens can end in an incomplete UTF-8
                    // sequence. Hold a trailing replacement character until a
                    // later token completes it, but still emit one SSE frame per
                    // model token for timing and accounting.
                    let stable = decoded.trim_end_matches('\u{fffd}');
                    let update = match output.push(stable) {
                        Ok(update) => update,
                        Err(message) => {
                            let payload = ApiError::generation(message).into_json_string();
                            yield Ok(Event::default().event("error").data(payload));
                            return;
                        }
                    };
                    let (text, matched_stop) = match update {
                        FilterUpdate::Continue(text) => (text, false),
                        FilterUpdate::Stop(text) => (text, true),
                    };
                    let payload = match kind {
                        StreamKind::Completion => json!({
                            "id": id,
                            "object": "text_completion",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": text,
                                "finish_reason": null
                            }]
                        }),
                        StreamKind::Chat => json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": text },
                                "finish_reason": null
                            }]
                        }),
                    };
                    if matched_stop {
                        response_done.cancel();
                    }
                    yield Ok(Event::default().data(payload.to_string()));
                    if matched_stop {
                        let payload = match kind {
                            StreamKind::Completion => json!({
                                "id": id,
                                "object": "text_completion",
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "text": "",
                                    "finish_reason": finish_reason_label(hotbatch_core::FinishReason::Stop)
                                }]
                            }),
                            StreamKind::Chat => json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason_label(hotbatch_core::FinishReason::Stop)
                                }]
                            }),
                        };
                        yield Ok(Event::default().data(payload.to_string()));
                        yield Ok(Event::default().data("[DONE]"));
                        return;
                    }
                }
                StreamItem::Finished(reason) => {
                    let decoded = match tokenizer.decode(&tokens) {
                        Ok(decoded) => decoded,
                        Err(error) => {
                            let payload = ApiError::generation(format!(
                                "generated tokens could not be decoded: {error}"
                            ))
                            .into_json_string();
                            yield Ok(Event::default().event("error").data(payload));
                            return;
                        }
                    };
                    let remaining = match output.finish(&decoded) {
                        Ok(remaining) => remaining,
                        Err(message) => {
                            let payload = ApiError::generation(message).into_json_string();
                            yield Ok(Event::default().event("error").data(payload));
                            return;
                        }
                    };
                    let payload = match kind {
                        StreamKind::Completion => json!({
                            "id": id,
                            "object": "text_completion",
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "text": remaining,
                                "finish_reason": finish_reason_label(reason)
                            }]
                        }),
                        StreamKind::Chat => {
                            let delta = if remaining.is_empty() {
                                json!({})
                            } else {
                                json!({ "content": remaining })
                            };
                            json!({
                                "id": id,
                                "object": "chat.completion.chunk",
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": delta,
                                    "finish_reason": finish_reason_label(reason)
                                }]
                            })
                        }
                    };
                    yield Ok(Event::default().data(payload.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
                StreamItem::Error(message) => {
                    let payload = ApiError::generation(format!("generation failed: {message}"))
                        .into_json_string();
                    yield Ok(Event::default().event("error").data(payload));
                    return;
                }
            }
        }

        let payload = ApiError::generation("generation ended before a terminal event")
            .into_json_string();
        yield Ok(Event::default().event("error").data(payload));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_cross_chunk_prefix_and_omits_match() {
        let mut filter = TextOutputFilter::new(vec!["testing".to_string()]);
        assert_eq!(
            filter.push("a test"),
            Ok(FilterUpdate::Continue("a ".to_string()))
        );
        assert_eq!(
            filter.push("a testing"),
            Ok(FilterUpdate::Stop(String::new()))
        );
    }

    #[test]
    fn stops_inside_a_single_decoded_token() {
        let mut filter = TextOutputFilter::new(vec!["ing".to_string()]);
        assert_eq!(
            filter.push(" testing"),
            Ok(FilterUpdate::Stop(" test".to_string()))
        );
    }

    #[test]
    fn chooses_earliest_overlapping_textual_stop() {
        let mut filter = TextOutputFilter::new(vec!["bc end".to_string(), " end".to_string()]);
        assert_eq!(
            filter.push("abc"),
            Ok(FilterUpdate::Continue("a".to_string()))
        );
        assert_eq!(
            filter.push("abc end"),
            Ok(FilterUpdate::Stop(String::new()))
        );
    }

    #[test]
    fn flushes_unmatched_prefix_at_natural_finish() {
        let mut filter = TextOutputFilter::new(vec!["stop".to_string()]);
        assert_eq!(
            filter.push("do st"),
            Ok(FilterUpdate::Continue("do ".to_string()))
        );
        assert_eq!(filter.finish("do st"), Ok("st".to_string()));
    }
}
