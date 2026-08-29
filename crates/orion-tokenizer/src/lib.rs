//! Tokenizer integration and incremental streaming decode.
//!
//! Wraps the Hugging Face `tokenizers` crate and adds the piece it does not
//! provide: correct *incremental* decoding for token streaming.
//!
//! # Why streaming decode is not trivial
//!
//! The obvious approach — decode each new token and emit the result — is
//! wrong, for two reasons that both produce visible corruption:
//!
//! 1. **Multi-byte characters span tokens.** A single emoji or CJK character
//!    may be split across two or three BPE tokens. Decoding each in isolation
//!    yields replacement characters where the bytes are incomplete.
//!
//! 2. **Byte-level BPE is context-sensitive.** Decoding `[a, b]` together does
//!    not always equal `decode([a]) + decode([b])`, because the detokenizer
//!    strips prefix markers based on surrounding context.
//!
//! [`IncrementalDecoder`] solves both by decoding a sliding window and emitting
//! only the newly stable suffix.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

use std::path::Path;

use orion_core::{EngineError, ModelError, TokenId};
use tokenizers::Tokenizer as HfTokenizer;

/// A loaded tokenizer.
#[derive(Debug)]
pub struct Tokenizer {
    inner: HfTokenizer,
    bos_token_id: Option<TokenId>,
    eos_token_ids: Vec<TokenId>,
}

impl Tokenizer {
    /// Loads `tokenizer.json` from a model directory.
    pub fn from_directory(dir: &Path) -> Result<Self, ModelError> {
        let path = dir.join("tokenizer.json");
        if !path.exists() {
            return Err(ModelError::MissingFile(path));
        }
        let inner = HfTokenizer::from_file(&path).map_err(|e| ModelError::Malformed {
            file: "tokenizer.json".into(),
            reason: e.to_string(),
        })?;
        Ok(Self {
            inner,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
        })
    }

    /// Builds from an already-constructed HF tokenizer, for tests.
    pub fn from_hf(inner: HfTokenizer) -> Self {
        Self {
            inner,
            bos_token_id: None,
            eos_token_ids: Vec::new(),
        }
    }

    /// Records the special token ids taken from the model config.
    ///
    /// These come from `config.json` rather than from the tokenizer, because
    /// the two occasionally disagree and the model's own view is what governs
    /// generation.
    pub fn with_special_tokens(mut self, bos: Option<TokenId>, eos: Vec<TokenId>) -> Self {
        self.bos_token_id = bos;
        self.eos_token_ids = eos;
        self
    }

    pub fn bos_token_id(&self) -> Option<TokenId> {
        self.bos_token_id
    }

    pub fn eos_token_ids(&self) -> &[TokenId] {
        &self.eos_token_ids
    }

    /// Vocabulary size including added tokens.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Encodes text to token ids.
    ///
    /// `add_special_tokens` controls whether the tokenizer's own template
    /// (typically a BOS) is applied.
    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<TokenId>, EngineError> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| EngineError::Tokenizer(e.to_string()))
    }

    /// Decodes token ids back to text.
    pub fn decode(
        &self,
        ids: &[TokenId],
        skip_special_tokens: bool,
    ) -> Result<String, EngineError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| EngineError::Tokenizer(e.to_string()))
    }

    /// Whether a token id terminates generation.
    pub fn is_eos(&self, token: TokenId) -> bool {
        self.eos_token_ids.contains(&token)
    }
}

/// Emits text incrementally as tokens arrive, without corrupting multi-byte
/// characters or context-sensitive detokenization.
///
/// # How it works
///
/// The decoder decodes a *window* of recent tokens from a fixed anchor, and
/// emits whatever extends past the byte count of that window it has already
/// sent. Because the anchor is fixed between re-anchors, a partial character
/// simply produces more text on the next decode, and the extra bytes are the
/// delta — held-back bytes reappear on their own rather than needing separate
/// bookkeeping.
///
/// The window bounds the cost: decoding the whole history each step would be
/// O(n²) over a long generation, whereas a bounded window is O(window) per step
/// while still giving the detokenizer enough left context to make the same
/// decisions it would on the full sequence.
#[derive(Debug)]
pub struct IncrementalDecoder {
    /// Every token seen, needed because detokenization is context-sensitive.
    tokens: Vec<TokenId>,
    /// Bytes already handed to the caller.
    emitted: String,
    /// Tokens of left context the decoder re-examines each step.
    window: usize,
    /// First token of the current decode window.
    window_start: usize,
    /// Bytes of the current window's decode already emitted.
    window_emitted: usize,
    skip_special_tokens: bool,
}

/// Left context wide enough to cover any realistic multi-token grapheme plus
/// the detokenizer's own lookbehind, while keeping per-step cost constant.
const DEFAULT_WINDOW: usize = 16;

impl IncrementalDecoder {
    pub fn new(skip_special_tokens: bool) -> Self {
        Self {
            tokens: Vec::new(),
            emitted: String::new(),
            window: DEFAULT_WINDOW,
            window_start: 0,
            window_emitted: 0,
            skip_special_tokens,
        }
    }

    /// Sets the left-context window. Mainly for tests.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = window.max(1);
        self
    }

    /// Everything emitted so far.
    pub fn text(&self) -> &str {
        &self.emitted
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Adds a token and returns whatever text became stable.
    ///
    /// Returns an empty string when the token completes no new character —
    /// mid-way through a multi-byte sequence, for instance. Callers should
    /// treat that as "nothing to send yet" rather than as end of stream; the
    /// held-back bytes are emitted once the character completes.
    ///
    /// # Method
    ///
    /// The window is decoded from a fixed start point, and the result is
    /// compared against how much of *that same window* has already been
    /// emitted. Emitting `window_text[window_emitted..]` is what makes held-back
    /// bytes reappear automatically: when a partial character completes, the
    /// re-decode of the same window simply produces more text, and the extra is
    /// exactly the delta.
    ///
    /// An earlier version compared successive decodes of shrinking windows and
    /// dropped anything that was not a clean extension. That silently discarded
    /// every multi-byte character, because the incomplete decode ends in
    /// `U+FFFD` and the completed one does not share it as a prefix.
    pub fn push(&mut self, tokenizer: &Tokenizer, token: TokenId) -> Result<String, EngineError> {
        self.tokens.push(token);

        // Re-anchor the window when it has grown past twice its target, so each
        // step stays O(window) rather than O(total tokens).
        //
        // The re-anchor must preserve the meaning of `window_emitted`, which
        // counts bytes of the *current* window's decode. Decoding the discarded
        // prefix gives exactly how many bytes leave the window, so the offset is
        // rebased rather than reset. Resetting it to zero would re-emit the
        // retained tail, which is what produced doubled text ("helllo wworlld").
        if self.tokens.len() - self.window_start > self.window * 2 {
            let new_start = self.tokens.len() - self.window;
            let dropped = tokenizer.decode(
                &self.tokens[self.window_start..new_start],
                self.skip_special_tokens,
            )?;
            // Only re-anchor when the discarded prefix decodes cleanly. A
            // partial character straddling the new boundary would make the byte
            // count meaningless, so the window simply stays put for now.
            if !dropped.ends_with('\u{FFFD}') && self.window_emitted >= dropped.len() {
                self.window_emitted -= dropped.len();
                self.window_start = new_start;
            }
        }

        let window_text =
            tokenizer.decode(&self.tokens[self.window_start..], self.skip_special_tokens)?;

        // Nothing new decoded yet: the character is still incomplete.
        if window_text.len() <= self.window_emitted {
            return Ok(String::new());
        }

        let mut delta = &window_text[self.window_emitted..];

        // Hold back a *trailing* replacement character: it means the newest
        // bytes do not yet form a complete character. The bytes are not lost —
        // `window_emitted` is not advanced past them, so once the character
        // completes the next decode of this same window yields it in full and
        // it is emitted then.
        //
        // Only the trailing one is trimmed. A replacement character with text
        // after it is already settled: the decoder has moved past those bytes
        // and no future token will complete them, so holding it back would stall
        // the stream permanently rather than delay it.
        if delta.ends_with('\u{FFFD}') {
            delta = delta.trim_end_matches('\u{FFFD}');
        }
        if delta.is_empty() {
            return Ok(String::new());
        }

        let out = delta.to_string();
        self.window_emitted += out.len();
        self.emitted.push_str(&out);
        Ok(out)
    }

    /// Flushes any text held back, at end of generation.
    ///
    /// Called once when a sequence finishes. Whatever the tokenizer produces
    /// for the full token sequence is authoritative, so this emits everything
    /// past the point the streamed output and the full decode agree.
    ///
    /// Keying off the **common prefix** rather than requiring `full` to start
    /// with everything already emitted is what makes held-back bytes recoverable
    /// at the end. A character that was still incomplete when the last token
    /// arrived was deliberately withheld, so the streamed text is a *shorter*
    /// but not always a *prefix-equal* string — an exact `starts_with` check
    /// silently dropped the tail in exactly that case.
    pub fn finish(&mut self, tokenizer: &Tokenizer) -> Result<String, EngineError> {
        let full = tokenizer.decode(&self.tokens, self.skip_special_tokens)?;

        // Longest common prefix, on a character boundary.
        let common = full
            .char_indices()
            .zip(self.emitted.char_indices())
            .take_while(|((_, a), (_, b))| a == b)
            .map(|((i, c), _)| i + c.len_utf8())
            .last()
            .unwrap_or(0);

        if common >= full.len() {
            return Ok(String::new());
        }

        let tail = full[common..].to_string();
        self.emitted = full;
        self.window_emitted += tail.len();
        Ok(tail)
    }
}

/// Renders chat messages into a prompt string.
///
/// # Why templates are not read from the tokenizer
///
/// HF stores chat templates as Jinja2, which needs a full template engine to
/// evaluate. Rather than take that dependency and its attack surface — these
/// templates would be evaluated on operator-supplied model files — this renders
/// the two formats the supported architectures actually use.
///
/// A model whose template differs will produce a subtly wrong prompt, so the
/// format is selected explicitly from the architecture rather than guessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 3: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`
    Llama3,
    /// ChatML, used by Qwen: `<|im_start|>role\ncontent<|im_end|>`
    ChatMl,
    /// No template: messages are concatenated with newlines. For base models.
    Plain,
}

impl ChatTemplate {
    /// Picks a template for an architecture string from `ModelMetadata`.
    pub fn for_architecture(arch: &str) -> Self {
        match arch {
            "qwen2" => ChatTemplate::ChatMl,
            "llama" => ChatTemplate::Llama3,
            _ => ChatTemplate::Plain,
        }
    }

    /// Renders messages, appending the opening of an assistant turn so the
    /// model continues rather than starting a new turn of its own.
    pub fn render(self, messages: &[(String, String)]) -> String {
        let mut out = String::new();
        match self {
            ChatTemplate::Llama3 => {
                for (role, content) in messages {
                    out.push_str(&format!(
                        "<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
                    ));
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatTemplate::ChatMl => {
                for (role, content) in messages {
                    out.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
                }
                out.push_str("<|im_start|>assistant\n");
            }
            ChatTemplate::Plain => {
                for (_, content) in messages {
                    out.push_str(content);
                    out.push('\n');
                }
            }
        }
        out
    }
}

/// Tracks whether any caller-supplied stop string has appeared in the output.
///
/// Matching is over the accumulated text rather than over tokens, because a
/// stop string need not align with token boundaries — "END" may arrive as
/// `["E", "ND"]` and a token-level check would miss it entirely.
#[derive(Debug, Clone)]
pub struct StopSequenceMatcher {
    sequences: Vec<String>,
    /// Longest stop string, bounding how much tail text must be retained.
    max_len: usize,
    buffer: String,
}

impl StopSequenceMatcher {
    pub fn new(sequences: Vec<String>) -> Self {
        let max_len = sequences.iter().map(|s| s.len()).max().unwrap_or(0);
        Self {
            sequences,
            max_len,
            buffer: String::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Feeds new text; returns the stop string that matched, if any.
    pub fn push(&mut self, text: &str) -> Option<String> {
        if self.sequences.is_empty() || text.is_empty() {
            return None;
        }
        self.buffer.push_str(text);

        let found = self
            .sequences
            .iter()
            .find(|s| self.buffer.contains(s.as_str()))
            .cloned();

        // Retain only enough tail to catch a stop string straddling this
        // boundary and the next; without the bound the buffer would grow with
        // the whole generation.
        if self.buffer.len() > self.max_len * 2 {
            let keep_from = self.buffer.len() - self.max_len;
            // Do not split a UTF-8 character.
            let boundary = (0..=keep_from)
                .rev()
                .find(|&i| self.buffer.is_char_boundary(i))
                .unwrap_or(0);
            self.buffer = self.buffer[boundary..].to_string();
        }
        found
    }

    /// Truncates text at the first stop string, which should not be shown to
    /// the caller.
    pub fn truncate_at_stop<'a>(&self, text: &'a str) -> &'a str {
        let mut cut = text.len();
        for s in &self.sequences {
            if let Some(pos) = text.find(s.as_str()) {
                cut = cut.min(pos);
            }
        }
        &text[..cut]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A tiny byte-level BPE tokenizer, built from the same JSON format a real
    /// `tokenizer.json` uses, so tests exercise the production load path.
    ///
    /// Byte-level with no merges means every byte is its own token, which is
    /// exactly what is needed to exercise the multi-byte incremental-decode
    /// case without shipping a real vocabulary.
    fn test_tokenizer() -> Tokenizer {
        // Byte-level BPE maps each of the 256 bytes to a printable placeholder
        // character; this is the standard GPT-2 byte encoder alphabet.
        let mut vocab = String::from("{");
        for b in 0..256u32 {
            let ch = byte_to_unicode(b as u8);
            if b > 0 {
                vocab.push(',');
            }
            vocab.push_str(&format!("{}:{}", serde_json::to_string(&ch).unwrap(), b));
        }
        vocab.push('}');

        let json = format!(
            r#"{{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false,
                                   "trim_offsets": true, "use_regex": true}},
                "post_processor": null,
                "decoder": {{"type": "ByteLevel", "add_prefix_space": false,
                             "trim_offsets": true, "use_regex": true}},
                "model": {{
                    "type": "BPE", "dropout": null, "unk_token": null,
                    "continuing_subword_prefix": null, "end_of_word_suffix": null,
                    "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                    "vocab": {vocab}, "merges": []
                }}
            }}"#
        );

        let inner: HfTokenizer = json.parse().expect("test tokenizer JSON is valid");
        Tokenizer::from_hf(inner)
    }

    /// GPT-2's byte-to-unicode mapping, which byte-level BPE vocabularies use
    /// so that every byte has a printable representation.
    fn byte_to_unicode(b: u8) -> String {
        let c = match b {
            b'!'..=b'~' => b as u32,
            0xA1..=0xAC | 0xAE..=0xFF => b as u32,
            _ => 256 + b as u32,
        };
        char::from_u32(c).unwrap().to_string()
    }

    #[test]
    fn encode_and_decode_round_trip() {
        let t = test_tokenizer();
        let ids = t.encode("hello world", false).unwrap();
        assert!(!ids.is_empty());
        let text = t.decode(&ids, true).unwrap();
        assert!(text.contains("hello"), "got {text:?}");
    }

    #[test]
    fn special_token_ids_come_from_the_model_config() {
        let t = test_tokenizer().with_special_tokens(Some(1), vec![2, 3]);
        assert_eq!(t.bos_token_id(), Some(1));
        assert_eq!(t.eos_token_ids(), &[2, 3]);
        assert!(t.is_eos(2));
        assert!(t.is_eos(3));
        assert!(!t.is_eos(1));
    }

    #[test]
    fn incremental_decoding_reproduces_the_full_text() {
        // The core property: streaming must yield exactly what one batch decode
        // would have produced.
        let t = test_tokenizer();
        let ids = t.encode("hello world testing", false).unwrap();
        let expected = t.decode(&ids, true).unwrap();

        let mut dec = IncrementalDecoder::new(true);
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&dec.push(&t, id).unwrap());
        }
        streamed.push_str(&dec.finish(&t).unwrap());

        assert_eq!(streamed, expected, "streaming diverged from batch decode");
        assert_eq!(dec.text(), expected);
        assert_eq!(dec.num_tokens(), ids.len());
    }

    #[test]
    fn multi_byte_text_streams_identically_to_a_batch_decode() {
        // The contract that matters: whatever the tokenizer produces for the
        // whole sequence, streaming must reproduce byte for byte. Multi-byte
        // characters are exactly where a naive per-token decode diverges.
        let t = test_tokenizer();
        let ids = t.encode("日本語 café 🚀", false).unwrap();
        let expected = t.decode(&ids, true).unwrap();

        let mut dec = IncrementalDecoder::new(true);
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&dec.push(&t, id).unwrap());
        }
        streamed.push_str(&dec.finish(&t).unwrap());

        assert_eq!(streamed, expected, "streaming diverged from batch decode");
    }

    #[test]
    fn no_chunk_ever_ends_mid_character() {
        // Every chunk handed to a client must be valid text on its own. A chunk
        // ending in a replacement character renders as a broken glyph in the
        // middle of a stream, which is what the hold-back logic prevents.
        let t = test_tokenizer();
        let ids = t.encode("日本語 café 🚀", false).unwrap();

        let mut dec = IncrementalDecoder::new(true);
        for &id in &ids {
            let chunk = dec.push(&t, id).unwrap();
            assert!(
                !chunk.ends_with('\u{FFFD}'),
                "emitted a chunk ending mid-character: {chunk:?}"
            );
        }
    }

    #[test]
    fn the_first_token_emits_immediately() {
        let t = test_tokenizer();
        let mut dec = IncrementalDecoder::new(true);
        let out = dec.push(&t, 1).unwrap();
        assert!(!out.is_empty(), "the first token should produce text");
    }

    #[test]
    fn a_bounded_window_still_matches_a_full_decode() {
        // A small window must not change the emitted text, only the cost.
        let t = test_tokenizer();
        let ids = t.encode("hello world testing hello world", false).unwrap();
        let expected = t.decode(&ids, true).unwrap();

        let mut dec = IncrementalDecoder::new(true).with_window(2);
        let mut streamed = String::new();
        for &id in &ids {
            streamed.push_str(&dec.push(&t, id).unwrap());
        }
        streamed.push_str(&dec.finish(&t).unwrap());
        assert_eq!(streamed, expected);
    }

    #[test]
    fn finish_is_idempotent_once_everything_is_emitted() {
        let t = test_tokenizer();
        let mut dec = IncrementalDecoder::new(true);
        dec.push(&t, 1).unwrap();
        dec.finish(&t).unwrap();
        assert_eq!(dec.finish(&t).unwrap(), "", "nothing left to flush");
    }

    #[test]
    fn llama3_template_renders_headers_and_opens_the_assistant_turn() {
        let msgs = vec![
            ("system".to_string(), "Be brief.".to_string()),
            ("user".to_string(), "Hi".to_string()),
        ];
        let out = ChatTemplate::Llama3.render(&msgs);

        assert!(out.contains("<|start_header_id|>system<|end_header_id|>"));
        assert!(out.contains("Be brief.<|eot_id|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(
            out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"),
            "must open the assistant turn so the model continues"
        );
    }

    #[test]
    fn chatml_template_renders_im_markers() {
        let msgs = vec![("user".to_string(), "Hi".to_string())];
        let out = ChatTemplate::ChatMl.render(&msgs);
        assert!(out.contains("<|im_start|>user\nHi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn plain_template_just_concatenates() {
        let msgs = vec![
            ("user".to_string(), "one".to_string()),
            ("assistant".to_string(), "two".to_string()),
        ];
        assert_eq!(ChatTemplate::Plain.render(&msgs), "one\ntwo\n");
    }

    #[test]
    fn templates_are_selected_by_architecture() {
        assert_eq!(
            ChatTemplate::for_architecture("llama"),
            ChatTemplate::Llama3
        );
        assert_eq!(
            ChatTemplate::for_architecture("qwen2"),
            ChatTemplate::ChatMl
        );
        assert_eq!(
            ChatTemplate::for_architecture("something-else"),
            ChatTemplate::Plain
        );
    }

    #[test]
    fn stop_matcher_finds_a_sequence_split_across_pushes() {
        // The reason matching is over text, not tokens: "STOP" may arrive in
        // pieces that no token-level check would catch.
        let mut m = StopSequenceMatcher::new(vec!["STOP".to_string()]);
        assert_eq!(m.push("hello ST"), None);
        assert_eq!(m.push("OP now"), Some("STOP".to_string()));
    }

    #[test]
    fn stop_matcher_ignores_text_without_a_match() {
        let mut m = StopSequenceMatcher::new(vec!["END".to_string()]);
        for chunk in ["hello ", "world ", "again"] {
            assert_eq!(m.push(chunk), None);
        }
    }

    #[test]
    fn an_empty_matcher_never_fires() {
        let mut m = StopSequenceMatcher::new(vec![]);
        assert!(m.is_empty());
        assert_eq!(m.push("anything at all"), None);
    }

    #[test]
    fn stop_matcher_buffer_stays_bounded() {
        // Without truncation the buffer would grow with the whole generation.
        let mut m = StopSequenceMatcher::new(vec!["XY".to_string()]);
        for _ in 0..1000 {
            m.push("aaaaaaaaaa");
        }
        assert!(m.buffer.len() < 100, "buffer grew to {}", m.buffer.len());
        // It must still detect a match after truncation.
        assert_eq!(m.push("XY"), Some("XY".to_string()));
    }

    #[test]
    fn stop_matcher_truncation_respects_utf8_boundaries() {
        let mut m = StopSequenceMatcher::new(vec!["ZZ".to_string()]);
        for _ in 0..100 {
            // Multi-byte characters; a naive byte slice would panic.
            m.push("日本語テキスト");
        }
        assert_eq!(m.push("ZZ"), Some("ZZ".to_string()));
    }

    #[test]
    fn truncate_at_stop_cuts_at_the_earliest_match() {
        let m = StopSequenceMatcher::new(vec!["END".to_string(), "STOP".to_string()]);
        assert_eq!(m.truncate_at_stop("hello STOP world END"), "hello ");
        assert_eq!(m.truncate_at_stop("no match here"), "no match here");
    }

    #[test]
    fn a_missing_tokenizer_file_names_the_path() {
        let err = Tokenizer::from_directory(Path::new("/nonexistent")).unwrap_err();
        assert!(matches!(err, ModelError::MissingFile(_)));
    }
}
