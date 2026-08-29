//! OpenAI-compatible request and response types.
//!
//! Field names and shapes match the OpenAI API so existing clients work
//! unchanged. Where OpenAI accepts something this engine does not implement,
//! the field is parsed and ignored rather than rejected — a client sending
//! `user` or `logit_bias` should not get a 400 for a field that only affects
//! behaviour it is not relying on.

use orion_core::{EngineError, FinishReason, SamplingParams};
use serde::{Deserialize, Serialize};

/// One message in a chat conversation.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    /// `null` content is legal in OpenAI's schema for assistant tool calls.
    #[serde(default)]
    pub content: Option<String>,
}

/// `POST /v1/chat/completions`
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,

    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    /// Newer OpenAI name for `max_tokens`.
    #[serde(default)]
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub min_tokens: Option<usize>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub n: Option<usize>,

    /// Accepted and ignored: identifies the end user for OpenAI's abuse
    /// tooling, which has no analogue here.
    #[serde(default)]
    pub user: Option<String>,
}

/// `stop` is `string | [string]`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopField {
    Single(String),
    Multiple(Vec<String>),
}

impl StopField {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopField::Single(s) => vec![s],
            StopField::Multiple(v) => v,
        }
    }
}

/// `POST /v1/completions`
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptField,

    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub echo: bool,
    #[serde(default)]
    pub n: Option<usize>,
}

/// `prompt` is `string | [string]`. Token-array prompts are not supported.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PromptField {
    Single(String),
    Multiple(Vec<String>),
}

impl PromptField {
    /// The single prompt this request carries.
    ///
    /// Batched prompts are rejected rather than silently truncated: returning
    /// one completion for a request that asked for several would be a wrong
    /// answer, not a degraded one.
    pub fn single(&self) -> Result<&str, EngineError> {
        match self {
            PromptField::Single(s) => Ok(s),
            PromptField::Multiple(v) if v.len() == 1 => Ok(&v[0]),
            PromptField::Multiple(v) => Err(EngineError::InvalidRequest(format!(
                "batched prompts are not supported: got {} prompts, expected 1",
                v.len()
            ))),
        }
    }
}

/// Builds engine sampling parameters from the OpenAI-shaped fields.
///
/// Applies the engine's own defaults for anything unset, then validates. The
/// error names the offending field, so a client gets something actionable.
#[allow(clippy::too_many_arguments)]
pub fn build_params(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    max_tokens: Option<usize>,
    min_tokens: Option<usize>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    seed: Option<u64>,
    stop: Option<StopField>,
    default_max_tokens: usize,
) -> Result<SamplingParams, EngineError> {
    // OpenAI's presence_penalty is additive on logits; this engine implements
    // the multiplicative CTRL repetition penalty. Rather than silently apply
    // the wrong one, an explicit repetition_penalty wins and presence_penalty
    // is mapped onto it only as a rough fallback.
    let penalty = repetition_penalty.or_else(|| presence_penalty.map(|p| 1.0 + p.max(0.0)));

    let params = SamplingParams {
        temperature: temperature.unwrap_or(1.0),
        top_p: top_p.unwrap_or(1.0),
        top_k: top_k.unwrap_or(0),
        repetition_penalty: penalty.unwrap_or(1.0),
        max_tokens: max_tokens.unwrap_or(default_max_tokens),
        min_tokens: min_tokens.unwrap_or(0),
        seed,
        stop_token_ids: Vec::new(),
        stop_sequences: stop.map(StopField::into_vec).unwrap_or_default(),
        logprobs: None,
        echo: false,
    };
    params.validate()?;
    Ok(params)
}

/// Token accounting returned with every completion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl Usage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// One choice in a non-streaming chat response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// `POST /v1/chat/completions` response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

/// One choice in a streaming chunk.
#[derive(Debug, Clone, Serialize)]
pub struct ChatChunkChoice {
    pub index: usize,
    pub delta: ChatDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// The incremental part of a streaming chunk.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// One `data:` frame of a streaming chat response.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// One choice in a text completion response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionChoice {
    pub index: usize,
    pub text: String,
    pub finish_reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// `POST /v1/completions` response.
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Usage,
}

/// An error body, in OpenAI's envelope shape.
#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl ErrorResponse {
    /// Builds a client-facing error from an engine error.
    ///
    /// Internal errors are deliberately opaque: their message names block ids,
    /// sequence ids and invariants, none of which a client should see.
    pub fn from_engine(e: &EngineError) -> Self {
        let message = match e {
            EngineError::Internal(_) => "internal server error".to_string(),
            other => other.to_string(),
        };
        let r#type = if e.is_client_error() {
            "invalid_request_error"
        } else if e.is_retryable() {
            "server_overloaded"
        } else {
            "server_error"
        };
        Self {
            error: ErrorDetail {
                message,
                r#type: r#type.to_string(),
                code: e.code().to_string(),
                param: None,
            },
        }
    }
}

/// Entry in `GET /v1/models`.
#[derive(Debug, Clone, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
}

/// `GET /v1/models` response.
#[derive(Debug, Clone, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

/// Maps a finish reason to the OpenAI vocabulary.
pub fn finish_reason_str(reason: &FinishReason) -> String {
    reason.as_api_str().to_string()
}

/// Seconds since the Unix epoch, for the `created` field.
pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Generates a response id with the conventional prefix.
pub fn new_id(prefix: &str) -> String {
    format!("{prefix}-{}", uuid::Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_minimal_chat_request_parses() {
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.messages.len(), 1);
        assert!(!req.stream, "stream defaults to false");
        assert!(req.temperature.is_none());
    }

    #[test]
    fn unknown_and_unsupported_fields_do_not_cause_a_rejection() {
        // A client sending fields this engine ignores should still be served.
        let json = r#"{
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "user": "someone",
            "logit_bias": {"1": 2},
            "future_openai_field": true
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.user.as_deref(), Some("someone"));
    }

    #[test]
    fn null_message_content_is_accepted() {
        let json = r#"{
            "model": "t",
            "messages": [{"role": "assistant", "content": null}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.messages[0].content.is_none());
    }

    #[test]
    fn stop_accepts_a_string_or_a_list() {
        let single: StopField = serde_json::from_str(r#""END""#).unwrap();
        assert_eq!(single.into_vec(), vec!["END"]);

        let many: StopField = serde_json::from_str(r#"["A","B"]"#).unwrap();
        assert_eq!(many.into_vec(), vec!["A", "B"]);
    }

    #[test]
    fn a_single_prompt_is_accepted_in_either_form() {
        let s = PromptField::Single("hello".into());
        assert_eq!(s.single().unwrap(), "hello");

        let m = PromptField::Multiple(vec!["hello".into()]);
        assert_eq!(m.single().unwrap(), "hello");
    }

    #[test]
    fn batched_prompts_are_rejected_rather_than_truncated() {
        // Answering one of three prompts would be a wrong answer, not a
        // degraded one.
        let m = PromptField::Multiple(vec!["a".into(), "b".into(), "c".into()]);
        let err = m.single().unwrap_err();
        assert!(matches!(err, EngineError::InvalidRequest(_)));
        assert!(err.to_string().contains('3'), "should name the count");
    }

    #[test]
    fn params_default_when_unspecified() {
        let p = build_params(None, None, None, None, None, None, None, None, None, 128).unwrap();
        assert_eq!(p.temperature, 1.0);
        assert_eq!(p.top_p, 1.0);
        assert_eq!(p.max_tokens, 128, "server default applies");
        assert!(p.seed.is_none());
    }

    #[test]
    fn explicit_params_override_defaults() {
        let p = build_params(
            Some(0.5),
            Some(0.9),
            Some(40),
            Some(64),
            Some(4),
            Some(1.2),
            None,
            Some(7),
            Some(StopField::Single("STOP".into())),
            128,
        )
        .unwrap();

        assert_eq!(p.temperature, 0.5);
        assert_eq!(p.top_k, 40);
        assert_eq!(p.max_tokens, 64);
        assert_eq!(p.min_tokens, 4);
        assert_eq!(p.repetition_penalty, 1.2);
        assert_eq!(p.seed, Some(7));
        assert_eq!(p.stop_sequences, vec!["STOP"]);
    }

    #[test]
    fn an_explicit_repetition_penalty_beats_presence_penalty() {
        // The two are different formulations; the explicit one must win rather
        // than being silently overwritten.
        let p = build_params(
            None,
            None,
            None,
            None,
            None,
            Some(1.5),
            Some(0.8),
            None,
            None,
            16,
        )
        .unwrap();
        assert_eq!(p.repetition_penalty, 1.5);
    }

    #[test]
    fn presence_penalty_maps_onto_repetition_penalty_when_alone() {
        let p = build_params(
            None,
            None,
            None,
            None,
            None,
            None,
            Some(0.5),
            None,
            None,
            16,
        )
        .unwrap();
        assert_eq!(p.repetition_penalty, 1.5);
    }

    #[test]
    fn invalid_params_are_rejected_with_a_named_field() {
        let err = build_params(
            Some(-1.0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            16,
        )
        .unwrap_err();
        assert!(matches!(err, EngineError::InvalidRequest(_)));
        assert!(err.to_string().contains("temperature"));
    }

    #[test]
    fn usage_totals_are_consistent() {
        let u = Usage::new(10, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn internal_errors_are_opaque_to_clients() {
        // The message names block ids and invariants; none of that should leak.
        let e = EngineError::Internal("block 42 desynced from sequence 7".into());
        let body = ErrorResponse::from_engine(&e);

        assert_eq!(body.error.message, "internal server error");
        assert!(!body.error.message.contains("block"));
        assert_eq!(body.error.code, "internal_error");
        assert_eq!(body.error.r#type, "server_error");
    }

    #[test]
    fn client_errors_keep_their_detail() {
        let e = EngineError::ContextLengthExceeded {
            prompt_tokens: 5000,
            max_tokens: 100,
            context_len: 4096,
        };
        let body = ErrorResponse::from_engine(&e);
        assert!(body.error.message.contains("5000"), "detail is useful here");
        assert_eq!(body.error.r#type, "invalid_request_error");
    }

    #[test]
    fn overload_errors_are_typed_for_retry() {
        let e = EngineError::QueueFull {
            queued: 100,
            limit: 100,
        };
        assert_eq!(
            ErrorResponse::from_engine(&e).error.r#type,
            "server_overloaded"
        );
    }

    #[test]
    fn finish_reasons_use_the_openai_vocabulary() {
        assert_eq!(finish_reason_str(&FinishReason::Stop), "stop");
        assert_eq!(finish_reason_str(&FinishReason::Length), "length");
        assert_eq!(
            finish_reason_str(&FinishReason::StopSequence("X".into())),
            "stop",
            "internal distinctions collapse to what clients expect"
        );
    }

    #[test]
    fn responses_serialize_in_the_expected_shape() {
        let resp = ChatCompletionResponse {
            id: new_id("chatcmpl"),
            object: "chat.completion",
            created: unix_now(),
            model: "test".into(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: Some("hi".into()),
                },
                finish_reason: "stop".into(),
            }],
            usage: Usage::new(3, 1),
        };

        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["content"], "hi");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 4);
        assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    }

    #[test]
    fn streaming_chunks_omit_absent_fields() {
        // OpenAI clients expect a delta with only what changed.
        let chunk = ChatCompletionChunk {
            id: "x".into(),
            object: "chat.completion.chunk",
            created: 0,
            model: "test".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some("tok".into()),
                },
                finish_reason: None,
            }],
            usage: None,
        };

        let v = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "tok");
        assert!(
            v["choices"][0]["delta"].get("role").is_none(),
            "absent role must be omitted, not null"
        );
        assert!(v["choices"][0].get("finish_reason").is_none());
        assert!(v.get("usage").is_none());
    }

    #[test]
    fn ids_are_unique() {
        let a = new_id("chatcmpl");
        let b = new_id("chatcmpl");
        assert_ne!(a, b);
    }
}
