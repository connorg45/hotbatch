use axum::response::sse::Event;
use futures_util::Stream;
use hotbatch_core::{StreamItem, TokenizerBundle};
use serde_json::json;
use std::convert::Infallible;
use tokio::sync::mpsc;

#[derive(Debug, Copy, Clone)]
pub enum StreamKind {
    Completion,
    Chat,
}

pub fn openai_stream(
    id: String,
    mut receiver: mpsc::Receiver<StreamItem>,
    tokenizer: TokenizerBundle,
    kind: StreamKind,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        while let Some(item) = receiver.recv().await {
            match item {
                StreamItem::Token(token) => {
                    let text = tokenizer.token_text(token);
                    let payload = match kind {
                        StreamKind::Completion => json!({
                            "id": id,
                            "object": "text_completion",
                            "choices": [{
                                "index": 0,
                                "text": text,
                                "finish_reason": null
                            }]
                        }),
                        StreamKind::Chat => json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": { "content": text },
                                "finish_reason": null
                            }]
                        }),
                    };
                    yield Ok(Event::default().data(payload.to_string()));
                }
                StreamItem::Done => {
                    let payload = match kind {
                        StreamKind::Completion => json!({
                            "id": id,
                            "object": "text_completion",
                            "choices": [{
                                "index": 0,
                                "text": "",
                                "finish_reason": "stop"
                            }]
                        }),
                        StreamKind::Chat => json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }]
                        }),
                    };
                    yield Ok(Event::default().data(payload.to_string()));
                    yield Ok(Event::default().data("[DONE]"));
                    break;
                }
            }
        }
    }
}
