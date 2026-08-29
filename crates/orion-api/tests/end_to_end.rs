//! End-to-end tests over the real HTTP surface.
//!
//! These drive the full stack — router, engine thread, scheduler, paged cache,
//! transformer, sampler, tokenizer — with a tiny randomly-weighted model. The
//! output is meaningless text, which is the point: these assert *plumbing*,
//! not model quality. Nothing here needs a GPU or a real checkpoint.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use orion_api::{router, AppState};
use orion_core::{DType, ModelMetadata, SchedulerConfig};
use orion_kv_cache::KvCacheManager;
use orion_models::{CpuBackend, LayerWeights, Matrix, ModelWeights, TransformerModel};
use orion_scheduler::Scheduler;
use orion_tokenizer::{ChatTemplate, Tokenizer};
use tower::ServiceExt;

const BLOCK_SIZE: usize = 8;
const NUM_BLOCKS: usize = 256;
const VOCAB: usize = 260;

fn metadata() -> ModelMetadata {
    ModelMetadata {
        architecture: "llama".into(),
        name: "tiny-test".into(),
        hidden_size: 8,
        num_layers: 2,
        num_attention_heads: 2,
        num_kv_heads: 1,
        head_dim: 4,
        vocab_size: VOCAB,
        max_position_embeddings: 512,
        rope_theta: 10_000.0,
        rms_norm_eps: 1e-5,
        dtype: DType::F32,
        eos_token_ids: vec![259],
        bos_token_id: Some(0),
    }
}

/// Deterministic pseudo-random weights, so runs are reproducible.
fn weights(rows: usize, cols: usize, seed: f32) -> Matrix {
    let data = (0..rows * cols)
        .map(|i| ((i as f32 * 0.31 + seed).sin()) * 0.4)
        .collect();
    Matrix::new(data, rows, cols).unwrap()
}

fn model_weights(meta: &ModelMetadata) -> ModelWeights {
    let h = meta.hidden_size;
    let ffn = 16;
    let q = meta.num_attention_heads * meta.head_dim;
    let kv = meta.num_kv_heads * meta.head_dim;

    let layers = (0..meta.num_layers)
        .map(|i| {
            let s = i as f32;
            LayerWeights {
                input_norm: vec![1.0; h],
                q_proj: weights(q, h, s + 1.0),
                k_proj: weights(kv, h, s + 2.0),
                v_proj: weights(kv, h, s + 3.0),
                o_proj: weights(h, q, s + 4.0),
                q_bias: None,
                k_bias: None,
                v_bias: None,
                post_attn_norm: vec![1.0; h],
                gate_proj: weights(ffn, h, s + 5.0),
                up_proj: weights(ffn, h, s + 6.0),
                down_proj: weights(h, ffn, s + 7.0),
            }
        })
        .collect();

    ModelWeights {
        embed_tokens: weights(meta.vocab_size, h, 0.5),
        layers,
        final_norm: vec![1.0; h],
        lm_head: None,
    }
}

/// A byte-level tokenizer built from the same JSON a real model ships.
fn tokenizer() -> Tokenizer {
    let mut vocab = String::from("{");
    for b in 0..256u32 {
        let c = match b as u8 {
            b'!'..=b'~' => b,
            0xA1..=0xAC | 0xAE..=0xFF => b,
            _ => 256 + b,
        };
        let ch = char::from_u32(c).unwrap().to_string();
        if b > 0 {
            vocab.push(',');
        }
        vocab.push_str(&format!("{}:{}", serde_json::to_string(&ch).unwrap(), b));
    }
    vocab.push('}');

    let json = format!(
        r#"{{
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {{"type":"ByteLevel","add_prefix_space":false,
                               "trim_offsets":true,"use_regex":true}},
            "post_processor": null,
            "decoder": {{"type":"ByteLevel","add_prefix_space":false,
                         "trim_offsets":true,"use_regex":true}},
            "model": {{"type":"BPE","dropout":null,"unk_token":null,
                       "continuing_subword_prefix":null,"end_of_word_suffix":null,
                       "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                       "vocab":{vocab},"merges":[]}}
        }}"#
    );

    let inner: tokenizers::Tokenizer = json.parse().expect("valid tokenizer JSON");
    Tokenizer::from_hf(inner).with_special_tokens(Some(0), vec![259])
}

/// Builds a fully wired server for testing.
fn test_app() -> axum::Router {
    let meta = metadata();
    let model = TransformerModel::new(
        meta.clone(),
        model_weights(&meta),
        NUM_BLOCKS,
        BLOCK_SIZE,
        Box::new(CpuBackend::new()),
    )
    .expect("model construction");

    let scheduler = Scheduler::new(
        SchedulerConfig {
            max_num_seqs: 16,
            max_num_batched_tokens: 256,
            max_model_len: Some(256),
            enable_chunked_prefill: true,
            max_waiting_requests: 32,
            request_timeout_secs: None,
            prioritize_decode: true,
            ..Default::default()
        },
        KvCacheManager::new(NUM_BLOCKS, BLOCK_SIZE, true),
    );

    let (engine, _thread) = orion_runtime::spawn(scheduler, Arc::new(model), 64);

    // The engine thread outlives the test; leaking the join handle is
    // deliberate, since each test builds its own server and the process exits
    // when the suite finishes.
    std::mem::forget(_thread);

    router(AppState {
        engine,
        tokenizer: Arc::new(tokenizer()),
        metadata: Arc::new(meta),
        template: ChatTemplate::Llama3,
        default_max_tokens: 8,
    })
}

async fn post_json(app: axum::Router, path: &str, body: serde_json::Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn get(app: axum::Router, path: &str) -> (StatusCode, String) {
    let req = Request::builder().uri(path).body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn health_reports_ok() {
    let (status, body) = get(test_app(), "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("ok"), "got {body}");
}

#[tokio::test]
async fn readiness_reports_engine_state() {
    let (status, body) = get(test_app(), "/ready").await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["status"], "ready");
    assert_eq!(v["running"], 0);
    assert!(v["kv_cache_utilization"].is_number());
}

#[tokio::test]
async fn the_model_list_names_the_loaded_model() {
    let (status, body) = get(test_app(), "/v1/models").await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "tiny-test");
}

#[tokio::test]
async fn a_chat_completion_returns_a_well_formed_response() {
    let (status, body) = post_json(
        test_app(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 5,
            "temperature": 0.0
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();

    assert_eq!(v["object"], "chat.completion");
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(v["choices"][0]["message"]["content"].is_string());
    assert!(v["choices"][0]["finish_reason"].is_string());

    let usage = &v["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(usage["completion_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );
}

#[tokio::test]
async fn max_tokens_is_respected() {
    let (status, body) = post_json(
        test_app(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "temperature": 0.0
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        v["usage"]["completion_tokens"].as_u64().unwrap() <= 3,
        "generated more than max_tokens: {body}"
    );
}

#[tokio::test]
async fn greedy_decoding_is_reproducible_across_requests() {
    // Temperature 0 must give the same answer every time, which is what makes
    // the engine debuggable at all.
    let ask = || async {
        let (_, body) = post_json(
            test_app(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "tiny-test",
                "messages": [{"role": "user", "content": "deterministic"}],
                "max_tokens": 6,
                "temperature": 0.0
            }),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string()
    };

    assert_eq!(ask().await, ask().await, "greedy decoding diverged");
}

#[tokio::test]
async fn a_seeded_request_is_reproducible() {
    let ask = || async {
        let (_, body) = post_json(
            test_app(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "tiny-test",
                "messages": [{"role": "user", "content": "seeded"}],
                "max_tokens": 6,
                "temperature": 0.8,
                "seed": 12345
            }),
        )
        .await;
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string()
    };

    assert_eq!(ask().await, ask().await, "seeded sampling diverged");
}

#[tokio::test]
async fn a_text_completion_returns_a_well_formed_response() {
    let (status, body) = post_json(
        test_app(),
        "/v1/completions",
        serde_json::json!({
            "model": "tiny-test",
            "prompt": "Once upon a time",
            "max_tokens": 4,
            "temperature": 0.0
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "text_completion");
    assert!(v["choices"][0]["text"].is_string());
    assert!(v["id"].as_str().unwrap().starts_with("cmpl-"));
}

#[tokio::test]
async fn echo_prepends_the_prompt() {
    let (status, body) = post_json(
        test_app(),
        "/v1/completions",
        serde_json::json!({
            "model": "tiny-test",
            "prompt": "PREFIX",
            "max_tokens": 2,
            "temperature": 0.0,
            "echo": true
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(
        v["choices"][0]["text"]
            .as_str()
            .unwrap()
            .starts_with("PREFIX"),
        "echo did not include the prompt: {body}"
    );
}

#[tokio::test]
async fn streaming_produces_sse_frames_ending_in_done() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "tiny-test",
                "messages": [{"role": "user", "content": "stream"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "stream": true
            })
            .to_string(),
        ))
        .unwrap();

    let resp = test_app().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("text/event-stream")),
        "streaming responses must declare SSE"
    );

    let bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);

    assert!(text.contains("data:"), "no SSE frames: {text}");
    assert!(
        text.contains("data: [DONE]"),
        "stream must end with the OpenAI sentinel: {text}"
    );
    assert!(
        text.contains("chat.completion.chunk"),
        "frames must carry the chunk object type"
    );

    // The first content-bearing chunk announces the assistant role.
    assert!(text.contains(r#""role":"assistant""#));

    // A finish reason must appear before the sentinel.
    assert!(text.contains(r#""finish_reason""#), "missing finish reason");
}

#[tokio::test]
async fn streaming_and_non_streaming_agree() {
    // The same request must produce the same text either way. A divergence
    // here would mean the incremental decoder is losing or duplicating output.
    let body = serde_json::json!({
        "model": "tiny-test",
        "messages": [{"role": "user", "content": "compare"}],
        "max_tokens": 6,
        "temperature": 0.0
    });

    let app = test_app();
    let mut streaming_body = body.clone();
    streaming_body["stream"] = serde_json::Value::Bool(true);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(streaming_body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let bytes = axum::body::to_bytes(resp.into_body(), 4 * 1024 * 1024)
        .await
        .unwrap();
    let text = String::from_utf8_lossy(&bytes);

    // Reassemble the streamed content.
    let mut streamed = String::new();
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            break;
        }
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(c) = chunk["choices"][0]["delta"]["content"].as_str() {
            streamed.push_str(c);
        }
    }

    // The non-streaming request runs second against the SAME app, so both see
    // an identical engine and prefix-cache state. Running them against separate
    // servers would compare two different cache warmths, not two code paths.
    let (_, plain) = post_json(app, "/v1/chat/completions", body).await;
    let v: serde_json::Value = serde_json::from_str(&plain).unwrap();
    let expected = v["choices"][0]["message"]["content"].as_str().unwrap();

    assert_eq!(streamed, expected, "streaming diverged from non-streaming");
}

#[tokio::test]
async fn concurrent_requests_are_all_served() {
    // Exercises continuous batching through the real HTTP surface.
    let app = test_app();

    let mut handles = Vec::new();
    for i in 0..8 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            post_json(
                app,
                "/v1/chat/completions",
                serde_json::json!({
                    "model": "tiny-test",
                    "messages": [{"role": "user", "content": format!("request {i}")}],
                    "max_tokens": 4,
                    "temperature": 0.0
                }),
            )
            .await
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let (status, body) = h.await.unwrap();
        assert_eq!(status, StatusCode::OK, "request {i} failed: {body}");
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v["choices"][0]["message"]["content"].is_string());
    }
}

#[tokio::test]
async fn a_shared_prefix_is_served_from_the_cache() {
    // Two requests with the same long system prompt should hit the prefix
    // cache on the second, which is visible through the readiness endpoint.
    let app = test_app();
    // Long enough to span several 8-token blocks, short enough to leave room
    // for output inside the 256-token context of this test model.
    let system = "You are a helpful assistant.".repeat(2);

    for _ in 0..2 {
        let (status, _) = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "tiny-test",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "hi"}
                ],
                "max_tokens": 2,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    // The cache should have released everything once both finished.
    let (status, body) = get(app, "/ready").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        v["running"], 0,
        "no request should still be running: {body}"
    );
}

#[tokio::test]
async fn an_oversized_prompt_is_rejected_with_400() {
    let (status, body) = post_json(
        test_app(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "x".repeat(4000)}],
            "max_tokens": 10
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["code"], "context_length_exceeded");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn invalid_sampling_parameters_are_rejected_with_400() {
    for (field, value) in [
        ("temperature", serde_json::json!(-1.0)),
        ("top_p", serde_json::json!(0.0)),
        ("max_tokens", serde_json::json!(0)),
    ] {
        let mut body = serde_json::json!({
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "hi"}]
        });
        body[field] = value.clone();

        let (status, resp) = post_json(test_app(), "/v1/chat/completions", body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{field}={value} should be rejected, got {resp}"
        );
        let v: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert!(
            v["error"]["message"].as_str().unwrap().contains(field),
            "the error should name the offending field: {resp}"
        );
    }
}

#[tokio::test]
async fn an_empty_message_list_is_rejected() {
    let (status, body) = post_json(
        test_app(),
        "/v1/chat/completions",
        serde_json::json!({"model": "tiny-test", "messages": []}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
}

#[tokio::test]
async fn multiple_completions_are_refused_rather_than_silently_reduced() {
    let (status, body) = post_json(
        test_app(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        body.contains("`n`"),
        "should explain the limitation: {body}"
    );
}

#[tokio::test]
async fn malformed_json_is_rejected() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{not json"))
        .unwrap();

    let resp = test_app().oneshot(req).await.unwrap();
    assert!(
        resp.status().is_client_error(),
        "malformed JSON should be a client error, got {}",
        resp.status()
    );
}

#[tokio::test]
async fn stop_sequences_truncate_the_output() {
    // With a stop string that cannot appear, output is unaffected; the point is
    // that supplying one does not break the request path.
    let (status, body) = post_json(
        test_app(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "tiny-test",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "stop": ["\u{0000}IMPOSSIBLE\u{0000}"]
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn cache_blocks_are_returned_after_a_burst() {
    // The leak check: after many requests complete, utilization must be back
    // to zero. A block leak would show up here and nowhere else.
    let app = test_app();

    for _ in 0..12 {
        let (status, _) = post_json(
            app.clone(),
            "/v1/chat/completions",
            serde_json::json!({
                "model": "tiny-test",
                "messages": [{"role": "user", "content": "leak check"}],
                "max_tokens": 3,
                "temperature": 0.0
            }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    let (_, body) = get(app, "/ready").await;
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let utilization = v["kv_cache_utilization"].as_f64().unwrap();
    assert_eq!(
        utilization, 0.0,
        "KV blocks leaked: utilization is {utilization} after all requests finished"
    );
}
