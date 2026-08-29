//! Hugging Face `config.json` parsing.
//!
//! HF configs are not uniform. The same field appears under different names
//! across architectures and across versions of the same architecture, and some
//! fields are optional with architecture-specific defaults. This module
//! normalizes all of that into [`ModelMetadata`], which is the only shape the
//! rest of the engine ever sees.

use std::path::Path;

use orion_core::{DType, ModelError, ModelMetadata, TokenId};
use serde::Deserialize;

/// The subset of `config.json` OrionServe understands.
///
/// Unknown fields are ignored rather than rejected: HF configs carry a great
/// deal of training-time metadata that is irrelevant to inference, and failing
/// on it would reject nearly every real checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct HfConfig {
    #[serde(default)]
    pub architectures: Vec<String>,

    #[serde(default)]
    pub model_type: Option<String>,

    pub hidden_size: usize,

    #[serde(alias = "n_layer", alias = "num_layers")]
    pub num_hidden_layers: usize,

    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,

    /// Absent on multi-head models, where it equals `num_attention_heads`.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// Usually `hidden_size / num_attention_heads`, but some architectures
    /// decouple them, so it is read when present.
    #[serde(default)]
    pub head_dim: Option<usize>,

    pub vocab_size: usize,

    #[serde(default)]
    pub intermediate_size: Option<usize>,

    #[serde(alias = "n_positions", alias = "max_sequence_length", default)]
    pub max_position_embeddings: Option<usize>,

    #[serde(default)]
    pub rope_theta: Option<f32>,

    #[serde(alias = "layer_norm_eps", alias = "rms_norm_epsilon", default)]
    pub rms_norm_eps: Option<f32>,

    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Either a single id or a list; chat models often define several.
    #[serde(default)]
    pub eos_token_id: Option<EosField>,

    #[serde(default)]
    pub bos_token_id: Option<TokenId>,

    /// Whether the LM head shares weights with the token embedding.
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

/// `eos_token_id` is `int | [int]` depending on the model.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EosField {
    Single(TokenId),
    Multiple(Vec<TokenId>),
}

impl EosField {
    fn into_vec(self) -> Vec<TokenId> {
        match self {
            EosField::Single(t) => vec![t],
            EosField::Multiple(v) => v,
        }
    }
}

/// Architectures this engine can execute.
///
/// A closed set on purpose: silently accepting an unknown architecture and
/// running Llama layers over its weights would produce fluent nonsense rather
/// than an error, which is far worse than refusing to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// Llama and its close relatives: RMSNorm, RoPE, GQA, SwiGLU.
    Llama,
    /// Qwen2. Same layer structure as Llama but with attention QKV biases.
    Qwen2,
}

impl Architecture {
    /// Identifies the architecture from the config's `architectures` list or
    /// its `model_type`.
    pub fn detect(config: &HfConfig) -> Result<Self, ModelError> {
        for arch in &config.architectures {
            match arch.as_str() {
                "LlamaForCausalLM" | "MistralForCausalLM" => return Ok(Architecture::Llama),
                "Qwen2ForCausalLM" => return Ok(Architecture::Qwen2),
                _ => {}
            }
        }
        match config.model_type.as_deref() {
            Some("llama") | Some("mistral") => Ok(Architecture::Llama),
            Some("qwen2") => Ok(Architecture::Qwen2),
            other => Err(ModelError::UnsupportedArchitecture(
                config
                    .architectures
                    .first()
                    .cloned()
                    .or_else(|| other.map(str::to_string))
                    .unwrap_or_else(|| "<unspecified>".into()),
            )),
        }
    }

    /// Whether attention projections carry bias terms.
    ///
    /// Llama has none; Qwen2 has them on Q, K and V but not on the output
    /// projection. Getting this wrong produces subtly wrong logits rather than
    /// a load failure, so it is derived from the architecture rather than
    /// guessed from which tensors happen to be present.
    pub fn has_qkv_bias(self) -> bool {
        match self {
            Architecture::Llama => false,
            Architecture::Qwen2 => true,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen2 => "qwen2",
        }
    }
}

/// Parses a `torch_dtype` string.
fn parse_dtype(s: Option<&str>) -> Result<DType, ModelError> {
    match s {
        None => Ok(DType::F32),
        Some("float32") | Some("float") => Ok(DType::F32),
        Some("float16") | Some("half") => Ok(DType::F16),
        Some("bfloat16") => Ok(DType::BF16),
        Some(other) => Err(ModelError::UnsupportedDtype(other.to_string())),
    }
}

impl HfConfig {
    /// Reads and parses `config.json` from a model directory.
    pub fn load(dir: &Path) -> Result<Self, ModelError> {
        let path = dir.join("config.json");
        if !path.exists() {
            return Err(ModelError::MissingFile(path));
        }
        let text = std::fs::read_to_string(&path).map_err(|source| ModelError::Io {
            path: path.clone(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|e| ModelError::Malformed {
            file: "config.json".into(),
            reason: e.to_string(),
        })
    }

    /// Normalizes into the engine's [`ModelMetadata`], filling defaults and
    /// checking internal consistency.
    pub fn to_metadata(&self, name: String) -> Result<ModelMetadata, ModelError> {
        let arch = Architecture::detect(self)?;

        // GQA: absent num_key_value_heads means multi-head attention.
        let num_kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        let head_dim = self
            .head_dim
            .unwrap_or_else(|| self.hidden_size / self.num_attention_heads.max(1));

        let malformed = |reason: String| ModelError::Malformed {
            file: "config.json".into(),
            reason,
        };

        if self.num_attention_heads == 0 || self.num_hidden_layers == 0 {
            return Err(malformed(
                "num_attention_heads and num_hidden_layers must be non-zero".into(),
            ));
        }
        if num_kv_heads == 0 {
            return Err(malformed("num_key_value_heads must be non-zero".into()));
        }
        if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(malformed(format!(
                "num_attention_heads ({}) must be a multiple of num_key_value_heads ({num_kv_heads})",
                self.num_attention_heads
            )));
        }
        if head_dim == 0 {
            return Err(malformed("head_dim resolved to zero".into()));
        }
        if self.vocab_size == 0 {
            return Err(malformed("vocab_size must be non-zero".into()));
        }

        Ok(ModelMetadata {
            architecture: arch.as_str().to_string(),
            name,
            hidden_size: self.hidden_size,
            num_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size: self.vocab_size,
            max_position_embeddings: self.max_position_embeddings.unwrap_or(2048),
            rope_theta: self.rope_theta.unwrap_or(10_000.0),
            rms_norm_eps: self.rms_norm_eps.unwrap_or(1e-5),
            dtype: parse_dtype(self.torch_dtype.as_deref())?,
            eos_token_ids: self
                .eos_token_id
                .clone()
                .map(EosField::into_vec)
                .unwrap_or_default(),
            bos_token_id: self.bos_token_id,
        })
    }

    /// Feed-forward inner dimension, defaulting to the Llama convention when
    /// the config omits it.
    pub fn ffn_dim(&self) -> usize {
        self.intermediate_size.unwrap_or(4 * self.hidden_size)
    }

    /// Whether the LM head reuses the embedding matrix.
    pub fn ties_embeddings(&self) -> bool {
        self.tie_word_embeddings.unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama_json() -> &'static str {
        r#"{
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "rope_theta": 500000.0,
            "rms_norm_eps": 1e-05,
            "torch_dtype": "bfloat16",
            "eos_token_id": [128001, 128009],
            "bos_token_id": 128000
        }"#
    }

    #[test]
    fn parses_a_realistic_llama_config() {
        let c: HfConfig = serde_json::from_str(llama_json()).unwrap();
        let m = c.to_metadata("llama-3-8b".into()).unwrap();

        assert_eq!(m.architecture, "llama");
        assert_eq!(m.num_layers, 32);
        assert_eq!(m.num_kv_heads, 8);
        assert_eq!(m.head_dim, 128, "derived from hidden_size / heads");
        assert_eq!(m.dtype, DType::BF16);
        assert_eq!(m.eos_token_ids, vec![128001, 128009]);
        assert_eq!(m.bos_token_id, Some(128000));
        assert!(m.uses_gqa());
        assert_eq!(m.gqa_group_size(), 4);
    }

    #[test]
    fn a_single_eos_id_is_accepted_as_well_as_a_list() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100,
            "eos_token_id": 2
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.to_metadata("t".into()).unwrap().eos_token_ids, vec![2]);
    }

    #[test]
    fn absent_kv_heads_means_multi_head_attention() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        let m = c.to_metadata("t".into()).unwrap();
        assert_eq!(m.num_kv_heads, 4);
        assert!(!m.uses_gqa());
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // Real configs carry a lot of training metadata.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100,
            "transformers_version": "4.40.0",
            "use_cache": true,
            "some_future_field": {"nested": [1,2,3]}
        }"#;
        assert!(serde_json::from_str::<HfConfig>(json).is_ok());
    }

    #[test]
    fn alternate_field_names_are_accepted() {
        // GPT-style naming.
        let json = r#"{
            "model_type": "llama",
            "hidden_size": 64, "n_layer": 2, "n_head": 4,
            "vocab_size": 100, "n_positions": 1024
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        let m = c.to_metadata("t".into()).unwrap();
        assert_eq!(m.num_layers, 2);
        assert_eq!(m.num_attention_heads, 4);
        assert_eq!(m.max_position_embeddings, 1024);
    }

    #[test]
    fn an_unsupported_architecture_is_refused() {
        let json = r#"{
            "architectures": ["BertForMaskedLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        let err = c.to_metadata("t".into()).unwrap_err();
        assert!(matches!(err, ModelError::UnsupportedArchitecture(a) if a == "BertForMaskedLM"));
    }

    #[test]
    fn qwen2_is_detected_and_declares_qkv_bias() {
        let json = r#"{
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        let arch = Architecture::detect(&c).unwrap();
        assert_eq!(arch, Architecture::Qwen2);
        assert!(arch.has_qkv_bias());
        assert!(!Architecture::Llama.has_qkv_bias());
    }

    #[test]
    fn inconsistent_head_counts_are_rejected() {
        // 7 query heads cannot be grouped over 2 KV heads.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 7, "num_key_value_heads": 2,
            "vocab_size": 100
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(
            c.to_metadata("t".into()).unwrap_err(),
            ModelError::Malformed { .. }
        ));
    }

    #[test]
    fn zero_valued_dimensions_are_rejected() {
        for (field, json) in [
            (
                "layers",
                r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,"num_hidden_layers":0,"num_attention_heads":4,"vocab_size":100}"#,
            ),
            (
                "vocab",
                r#"{"architectures":["LlamaForCausalLM"],"hidden_size":64,"num_hidden_layers":2,"num_attention_heads":4,"vocab_size":0}"#,
            ),
        ] {
            let c: HfConfig = serde_json::from_str(json).unwrap();
            assert!(c.to_metadata("t".into()).is_err(), "accepted zero {field}");
        }
    }

    #[test]
    fn an_unsupported_dtype_is_refused() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100,
            "torch_dtype": "float8_e4m3fn"
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        assert!(matches!(
            c.to_metadata("t".into()).unwrap_err(),
            ModelError::UnsupportedDtype(_)
        ));
    }

    #[test]
    fn explicit_head_dim_overrides_the_derived_value() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100,
            "head_dim": 32
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        // 64/4 would be 16; the explicit value wins.
        assert_eq!(c.to_metadata("t".into()).unwrap().head_dim, 32);
    }

    #[test]
    fn ffn_dim_falls_back_to_the_llama_convention() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "vocab_size": 100
        }"#;
        let c: HfConfig = serde_json::from_str(json).unwrap();
        assert_eq!(c.ffn_dim(), 256);
        assert!(!c.ties_embeddings());
    }

    #[test]
    fn a_missing_config_file_names_the_path() {
        let err = HfConfig::load(Path::new("/nonexistent/model")).unwrap_err();
        assert!(matches!(err, ModelError::MissingFile(_)));
    }
}
