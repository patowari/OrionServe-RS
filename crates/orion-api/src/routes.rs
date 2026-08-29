//! HTTP handlers and the router.
//!
//! # Streaming
//!
//! Streaming responses use Server-Sent Events, matching OpenAI's wire format:
//! each chunk is a `data:` line carrying JSON, and the stream ends with a
//! literal `data: [DONE]`. Clients written against OpenAI work unchanged.
//!
//! # Cancellation
//!
//! When a streaming client disconnects, the SSE stream is dropped, which drops
//! the receiver the engine writes to. The engine notices on its next send and
//! cancels the request. This is why a disconnected client stops consuming GPU
//! time rather than generating to `max_tokens` for nobody.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use orion_core::{EngineError, FinishReason, ModelMetadata};
use orion_runtime::{EngineHandle, StreamEvent};
use orion_tokenizer::{ChatTemplate, IncrementalDecoder, StopSequenceMatcher, Tokenizer};
use tokio::sync::mpsc;

use crate::types::*;

/// Shared state every handler sees.
#[derive(Clone)]
pub struct AppState {
    pub engine: EngineHandle,
    pub tokenizer: Arc<Tokenizer>,
    pub metadata: Arc<ModelMetadata>,
    pub template: ChatTemplate,
    /// `max_tokens` applied when a request does not specify one.
    pub default_max_tokens: usize,
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AppState")
            .field("model", &self.metadata.name)
            .field("template", &self.template)
            .finish()
    }
}

/// An error that knows its own HTTP status.
///
/// Centralizing the mapping here is the reason [`EngineError`] carries
/// `is_client_error` and `is_retryable`: no handler decides a status code by
/// inspecting a message.
#[derive(Debug)]
pub struct ApiError(EngineError);

impl From<EngineError> for ApiError {
    fn from(e: EngineError) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match &self.0 {
            EngineError::InvalidRequest(_) | EngineError::ContextLengthExceeded { .. } => {
                StatusCode::BAD_REQUEST
            }
            EngineError::QueueFull { .. } => StatusCode::TOO_MANY_REQUESTS,
            EngineError::CacheExhausted { .. } => StatusCode::SERVICE_UNAVAILABLE,
            EngineError::Timeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            EngineError::Cancelled => StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            EngineError::EngineShutdown => StatusCode::SERVICE_UNAVAILABLE,
            EngineError::Tokenizer(_) => StatusCode::BAD_REQUEST,
            EngineError::Backend { .. } | EngineError::Internal(_) => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        };

        // Internal faults are logged in full; the client sees an opaque body.
        if matches!(
            self.0,
            EngineError::Internal(_) | EngineError::Backend { .. }
        ) {
            tracing::error!(error = %self.0, "request failed");
        }

        (status, Json(ErrorResponse::from_engine(&self.0))).into_response()
    }
}

/// Builds the router.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state)
}

/// Liveness: the process is up.
///
/// Deliberately does not touch the engine. A liveness probe that fails when the
/// engine is merely busy would have Kubernetes restart a server that is working
/// perfectly well under load.
async fn health() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "ok"})))
}

/// Readiness: the engine is alive and able to accept work.
async fn ready(State(state): State<AppState>) -> Response {
    match state.engine.stats().await {
        Ok(stats) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ready",
                "running": stats.running,
                "waiting": stats.waiting,
                "kv_cache_utilization": stats.cache_utilization(),
            })),
        )
            .into_response(),
        Err(e) => ApiError(e).into_response(),
    }
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: state.metadata.name.clone(),
            object: "model",
            created: unix_now(),
            owned_by: "orionserve",
        }],
    })
}

/// Renders chat messages and encodes them to tokens.
fn encode_chat(state: &AppState, messages: &[ChatMessage]) -> Result<Vec<u32>, EngineError> {
    if messages.is_empty() {
        return Err(EngineError::InvalidRequest(
            "`messages` must not be empty".into(),
        ));
    }
    let rendered: Vec<(String, String)> = messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone().unwrap_or_default()))
        .collect();
    let prompt = state.template.render(&rendered);
    // The template already inserts the model's special tokens, so the
    // tokenizer must not add its own on top.
    state.tokenizer.encode(&prompt, false)
}

/// Rejects features the engine does not implement, rather than ignoring them.
fn check_unsupported(n: Option<usize>) -> Result<(), EngineError> {
    if let Some(n) = n {
        if n != 1 {
            return Err(EngineError::InvalidRequest(format!(
                "`n` must be 1: multiple completions per request are not supported (got {n})"
            )));
        }
    }
    Ok(())
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    check_unsupported(req.n)?;

    let prompt = encode_chat(&state, &req.messages)?;
    let params = build_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.max_tokens.or(req.max_completion_tokens),
        req.min_tokens,
        req.repetition_penalty,
        req.presence_penalty.or(req.frequency_penalty),
        req.seed,
        req.stop,
        state.default_max_tokens,
    )?;

    // A generous buffer: the engine uses try_send and treats a full channel as
    // a disconnected client, so it must be large enough to absorb a burst while
    // the HTTP task is scheduled.
    let (tx, rx) = mpsc::channel(1024);
    let stop = StopSequenceMatcher::new(params.stop_sequences.clone());

    state
        .engine
        .generate(prompt.clone(), params, tx)
        .await
        .map_err(ApiError)?;

    if req.stream {
        Ok(chat_stream(state, req.model, prompt.len(), rx, stop).into_response())
    } else {
        let body = collect_chat(&state, req.model, prompt.len(), rx, stop).await?;
        Ok(Json(body).into_response())
    }
}

/// Drains a stream into a single non-streaming chat response.
async fn collect_chat(
    state: &AppState,
    model: String,
    prompt_tokens: usize,
    mut rx: mpsc::Receiver<StreamEvent>,
    mut stop: StopSequenceMatcher,
) -> Result<ChatCompletionResponse, ApiError> {
    let mut decoder = IncrementalDecoder::new(true);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish = FinishReason::Length;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token { token, .. } => {
                let piece = decoder.push(&state.tokenizer, token).map_err(ApiError)?;
                text.push_str(&piece);
                completion_tokens += 1;

                if stop.push(&piece).is_some() {
                    finish = FinishReason::Stop;
                    break;
                }
            }
            StreamEvent::Done {
                reason,
                completion_tokens: n,
                ..
            } => {
                text.push_str(&decoder.finish(&state.tokenizer).map_err(ApiError)?);
                completion_tokens = n.max(completion_tokens);
                finish = reason;
                break;
            }
            StreamEvent::Error { code, message } => {
                return Err(ApiError(EngineError::Internal(format!(
                    "{code}: {message}"
                ))));
            }
        }
    }

    // Text after a stop string is not shown: the caller asked to stop there.
    let text = stop.truncate_at_stop(&text).to_string();

    Ok(ChatCompletionResponse {
        id: new_id("chatcmpl"),
        object: "chat.completion",
        created: unix_now(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(text),
            },
            finish_reason: finish_reason_str(&finish),
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
    })
}

/// Streams a chat response as Server-Sent Events.
fn chat_stream(
    state: AppState,
    model: String,
    prompt_tokens: usize,
    mut rx: mpsc::Receiver<StreamEvent>,
    mut stop: StopSequenceMatcher,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = new_id("chatcmpl");
    let created = unix_now();

    let stream = async_stream::stream! {
        let mut decoder = IncrementalDecoder::new(true);
        let mut completion_tokens = 0usize;

        // OpenAI's first chunk carries the role and no content.
        let first = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta { role: Some("assistant".into()), content: None },
                finish_reason: None,
            }],
            usage: None,
        };
        if let Ok(json) = serde_json::to_string(&first) {
            yield Ok(Event::default().data(json));
        }

        let mut finish = FinishReason::Length;

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token { token, .. } => {
                    let piece = match decoder.push(&state.tokenizer, token) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::warn!(error = %e, "decode failed mid-stream");
                            break;
                        }
                    };
                    completion_tokens += 1;

                    // A stop string ends the stream, and the matched text is
                    // not sent.
                    if stop.push(&piece).is_some() {
                        let visible = stop.truncate_at_stop(&piece);
                        if !visible.is_empty() {
                            let chunk = content_chunk(&id, created, &model, visible);
                            if let Ok(json) = serde_json::to_string(&chunk) {
                                yield Ok(Event::default().data(json));
                            }
                        }
                        finish = FinishReason::Stop;
                        break;
                    }

                    // An empty piece means the token completed no character
                    // yet. Sending an empty delta would be noise.
                    if piece.is_empty() {
                        continue;
                    }

                    let chunk = content_chunk(&id, created, &model, &piece);
                    if let Ok(json) = serde_json::to_string(&chunk) {
                        yield Ok(Event::default().data(json));
                    }
                }
                StreamEvent::Done { reason, completion_tokens: n, .. } => {
                    // Flush anything the decoder was holding back.
                    if let Ok(tail) = decoder.finish(&state.tokenizer) {
                        if !tail.is_empty() {
                            let chunk = content_chunk(&id, created, &model, &tail);
                            if let Ok(json) = serde_json::to_string(&chunk) {
                                yield Ok(Event::default().data(json));
                            }
                        }
                    }
                    completion_tokens = n.max(completion_tokens);
                    finish = reason;
                    break;
                }
                StreamEvent::Error { code, message } => {
                    tracing::error!(code, message, "generation failed mid-stream");
                    finish = FinishReason::Error(code.to_string());
                    break;
                }
            }
        }

        // Terminal chunk: finish reason plus usage.
        let last = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta::default(),
                finish_reason: Some(finish_reason_str(&finish)),
            }],
            usage: Some(Usage::new(prompt_tokens, completion_tokens)),
        };
        if let Ok(json) = serde_json::to_string(&last) {
            yield Ok(Event::default().data(json));
        }

        // OpenAI's sentinel. Clients look for exactly this.
        yield Ok(Event::default().data("[DONE]"));
    };

    Sse::new(stream)
}

fn content_chunk(id: &str, created: u64, model: &str, content: &str) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: id.to_string(),
        object: "chat.completion.chunk",
        created,
        model: model.to_string(),
        choices: vec![ChatChunkChoice {
            index: 0,
            delta: ChatDelta {
                role: None,
                content: Some(content.to_string()),
            },
            finish_reason: None,
        }],
        usage: None,
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    check_unsupported(req.n)?;

    let prompt_text = req.prompt.single()?;
    let prompt = state.tokenizer.encode(prompt_text, true)?;
    let params = build_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.max_tokens,
        None,
        req.repetition_penalty,
        None,
        req.seed,
        req.stop,
        state.default_max_tokens,
    )?;

    let (tx, mut rx) = mpsc::channel(1024);
    let mut stop = StopSequenceMatcher::new(params.stop_sequences.clone());
    let prompt_tokens = prompt.len();

    state
        .engine
        .generate(prompt, params, tx)
        .await
        .map_err(ApiError)?;

    // Streaming text completions follow the same SSE shape; only the payload
    // differs. Non-streaming is handled inline here.
    let mut decoder = IncrementalDecoder::new(true);
    let mut text = String::new();
    let mut completion_tokens = 0usize;
    let mut finish = FinishReason::Length;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token { token, .. } => {
                let piece = decoder.push(&state.tokenizer, token).map_err(ApiError)?;
                text.push_str(&piece);
                completion_tokens += 1;
                if stop.push(&piece).is_some() {
                    finish = FinishReason::Stop;
                    break;
                }
            }
            StreamEvent::Done {
                reason,
                completion_tokens: n,
                ..
            } => {
                text.push_str(&decoder.finish(&state.tokenizer).map_err(ApiError)?);
                completion_tokens = n.max(completion_tokens);
                finish = reason;
                break;
            }
            StreamEvent::Error { code, message } => {
                return Err(ApiError(EngineError::Internal(format!(
                    "{code}: {message}"
                ))));
            }
        }
    }

    let mut out = stop.truncate_at_stop(&text).to_string();
    if req.echo {
        out = format!("{prompt_text}{out}");
    }

    Ok(Json(CompletionResponse {
        id: new_id("cmpl"),
        object: "text_completion",
        created: unix_now(),
        model: req.model,
        choices: vec![CompletionChoice {
            index: 0,
            text: out,
            finish_reason: finish_reason_str(&finish),
            logprobs: None,
        }],
        usage: Usage::new(prompt_tokens, completion_tokens),
    })
    .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_codes_follow_the_error_classification() {
        let cases: Vec<(EngineError, StatusCode)> = vec![
            (
                EngineError::InvalidRequest("bad".into()),
                StatusCode::BAD_REQUEST,
            ),
            (
                EngineError::ContextLengthExceeded {
                    prompt_tokens: 1,
                    max_tokens: 1,
                    context_len: 1,
                },
                StatusCode::BAD_REQUEST,
            ),
            (
                EngineError::QueueFull {
                    queued: 1,
                    limit: 1,
                },
                StatusCode::TOO_MANY_REQUESTS,
            ),
            (
                EngineError::CacheExhausted {
                    needed: 1,
                    available: 0,
                },
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                EngineError::Timeout {
                    elapsed_ms: 1,
                    state: "decoding",
                },
                StatusCode::GATEWAY_TIMEOUT,
            ),
            (EngineError::EngineShutdown, StatusCode::SERVICE_UNAVAILABLE),
            (
                EngineError::Internal("secret".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
        ];

        for (err, expected) in cases {
            let got = ApiError(err).into_response().status();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn multiple_completions_are_refused_explicitly() {
        // Better an honest 400 than silently returning one completion for a
        // request that asked for four.
        assert!(check_unsupported(Some(4)).is_err());
        assert!(check_unsupported(Some(1)).is_ok());
        assert!(check_unsupported(None).is_ok());
    }

    #[test]
    fn a_content_chunk_carries_only_the_delta() {
        let c = content_chunk("id", 0, "m", "hello");
        let v = serde_json::to_value(&c).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "hello");
        assert!(v["choices"][0]["delta"].get("role").is_none());
        assert_eq!(v["object"], "chat.completion.chunk");
    }
}
