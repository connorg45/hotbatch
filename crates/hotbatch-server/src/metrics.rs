use crate::api::ApiError;
use crate::AppState;
use axum::extract::State;
use axum::http::header;
use axum::response::{IntoResponse, Response};

pub async fn metrics(State(state): State<AppState>) -> Response {
    match state.metrics.gather() {
        Ok(body) => (
            [(
                header::CONTENT_TYPE,
                "text/plain; version=0.0.4; charset=utf-8",
            )],
            body,
        )
            .into_response(),
        Err(err) => ApiError::internal(format!("failed to gather metrics: {err}")).into_response(),
    }
}
