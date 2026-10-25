//! Version-pinned Rust tokenizer adapter. The trained model's special tokens
//! and pre-tokenizer rules are loaded from its official tokenizer.json.
//! No generated model code or foreign inference runtime is executed.

use crate::{Error, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use tokenizers::Tokenizer;

mod bounds;
pub use bounds::{TokenizerPlan, constructor_tree_bytes};

const CHAT_TEMPLATE_BYTES: usize = "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n".len();
pub const APPROVED_TOKENIZER_BYTES: u64 = 12_807_982;
const APPROVED_TOKENIZER_SHA256: &str =
    "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42";
const APPROVED_SPLIT_REGEX: &str = r#"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#;
// Keep the exact approved expression referenced in non-test builds as well as
// descriptor tests, without reparsing 12.8 MiB of JSON in production.
const _: &str = APPROVED_SPLIT_REGEX;
/// NFC can expand one source scalar's UTF-8 spelling by at most 3x; used for
/// the prompt-ID and decoded-text content bounds, not as a heap allowance.
const NFC_MAX_UTF8_EXPANSION: usize = 3;

/// Source-derived serving text bounds. Prompt IDs are at most three per
/// rendered UTF-8 byte; decoded text is at most three UTF-8 bytes per decoded
/// vocabulary byte. The bound is a capacity envelope, not an authorization
/// or numerical proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServingBound {
    max_decoded_token_bytes: usize,
    prompt_prefix_bytes: usize,
}
impl ServingBound {
    fn approved() -> Self {
        Self {
            // Approved source: max vocabulary entry 256 bytes, added 20.
            max_decoded_token_bytes: 256,
            // ByteLevel with add_prefix_space=false adds no prompt prefix.
            prompt_prefix_bytes: 0,
        }
    }
    pub fn max_decoded_token_bytes(self) -> usize {
        self.max_decoded_token_bytes
    }
    pub fn prompt_prefix_bytes(self) -> usize {
        self.prompt_prefix_bytes
    }
}

/// Retained/constructor plan for the approved profile.
///
/// Computable before any source byte is parsed, compiled or bootstrapped, so
/// admission can price the tokenizer ahead of the work it charges. `P` is the
/// process-global pool-stack count of the pinned `regex-automata 0.4.18`
/// (`meta::Config::get_pool_capacity`), the same cached value the production
/// `meta::Regex` uses.
/// One `encode` call's scratch peak for `rendered` bytes; `None` above the
/// priced prompt cap, where admission must fail rather than guess.
pub fn prompt_peak(rendered: u64) -> Option<u64> {
    bounds::prompt_peak(rendered)
}

/// Admitted-input serving peak without the serialized VM; the actual source
/// `Retained` owner prepays that VM row once. `None` above the priced cap.
pub fn admitted_prompt_peak(rendered: u64) -> Option<u64> {
    bounds::admitted_prompt_peak(rendered)
}

/// One `decode` call's scratch peak over `tokens` output IDs.
pub fn decode_peak(tokens: u64) -> Option<u64> {
    bounds::decode_peak(tokens)
}

pub fn approved_plan() -> Result<TokenizerPlan> {
    bounds::plan(regex_automata::meta::Config::new().get_pool_capacity())
}

/// Initialise the pinned crate's process-global ByteLevel statics (`RE`
/// program, `LazyBYTES_CHAR`, `CHAR_BYTES`) exactly once.
///
/// `byte_level.rs` builds them lazily and never drops them, so the compiler
/// scratch of this call is the only time their storage is charged
/// (`Bootstrap::approve`). The approved profile compiles `RE` even with
/// `use_regex=false` and never searches it, but the statics are real storage.
pub fn prime_byte_level() -> Result<()> {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let level = ByteLevel::new(false, false, false);
    let mut text = tokenizers::PreTokenizedString::from(String::from("a"));
    tokenizers::PreTokenizer::pre_tokenize(&level, &mut text)
        .map_err(|e| Error::Malformed(format!("byte-level bootstrap: {e}")))?;
    tokenizers::Decoder::decode_chain(&level, vec![String::from("a")])
        .map_err(|e| Error::Malformed(format!("byte-level bootstrap: {e}")))?;
    Ok(())
}

pub struct QwenTokenizer {
    /// Serializes every adapter call that enters the tokenizer's regex/BPE
    /// machinery (NFC + exact Split + ByteLevel + BPE + Encoding). Approved
    /// source stays exact; special IDs and the source bound sit outside.
    ///
    /// This mutex proves caller cardinality `C = 1`: each fancy-regex delegate
    /// pool then retains at most `P + 1` cache objects (one owner cache plus
    /// one stack cache), never an unbounded population. `P` and the retained
    /// byte counts are derived by [`QwenTokenizer::serving_inventory`] from the
    /// compiled literal descriptor and the pinned crate layouts.
    inner: Mutex<Tokenizer>,
    eos: u32,
    begin: u32,
    end: u32,
    bound: ServingBound,
    /// The approved serving profile disables BPE memoization in the cold
    /// constructor; `serving_inventory` is only valid for that profile.
    memoization_off: bool,
}

impl QwenTokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        if std::fs::metadata(path)?.len() > 64 * 1024 * 1024 {
            return Err(Error::Limit("tokenizer.json exceeds 64 MiB".into()));
        }
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Production seam for already-inspected byte-owned source. The immutable
    /// file size and SHA-256 are checked before construction, so a same-shaped
    /// file is not silently treated as the qualified source.
    pub fn from_approved_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(Error::Limit("tokenizer.json exceeds 64 MiB".into()));
        }
        if !approved_source(bytes) {
            return Err(Error::Malformed(format!(
                "tokenizer.json is not the approved source ({APPROVED_TOKENIZER_BYTES} bytes, SHA256 {APPROVED_TOKENIZER_SHA256})"
            )));
        }
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|e| Error::Malformed(format!("tokenizer.json: {e}")))?;
        Self::from_tokenizer(inner, ServingBound::approved(), true)
    }

    /// Parse unverified bounded bytes for tests and exploratory adapters. It
    /// derives a source descriptor but cannot establish approved source
    /// identity; serving preparation uses `from_approved_bytes`. The approved
    /// byte identity selects the bounded BPE cache policy; other sources keep
    /// memoization on and cannot use `serving_inventory`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() > 64 * 1024 * 1024 {
            return Err(Error::Limit("tokenizer.json exceeds 64 MiB".into()));
        }
        let bound = source_bound(bytes)
            .map_err(|error| Error::Malformed(format!("tokenizer.json: {error}")))?;
        let inner = Tokenizer::from_bytes(bytes)
            .map_err(|e| Error::Malformed(format!("tokenizer.json: {e}")))?;
        let serving = approved_source(bytes);
        Self::from_tokenizer(inner, bound, serving)
    }

    fn from_tokenizer(mut inner: Tokenizer, bound: ServingBound, serving: bool) -> Result<Self> {
        // BPE cache repair through public API only (pinned 0.22.2 exposes
        // `Tokenizer::{get_model, with_model}` and `ModelWrapper::resize_cache`,
        // but no `get_model_mut`). `BPE::clone` builds a fresh cache, and
        // `with_model` drops the old model; the clone is a cold constructor
        // cost priced by `TokenizerPlan::bpe_constructor_peak_bytes`.
        if serving {
            let mut model = inner.get_model().clone();
            model.resize_cache(0);
            inner.with_model(model);
        }
        let begin = inner
            .token_to_id("<|im_start|>")
            .ok_or_else(|| Error::Malformed("Qwen tokenizer missing im_start".into()))?;
        let end = inner
            .token_to_id("<|im_end|>")
            .ok_or_else(|| Error::Malformed("Qwen tokenizer missing im_end".into()))?;
        let eos = end;
        if begin == end {
            return Err(Error::Malformed(
                "Qwen template start/end token collision".into(),
            ));
        }
        Ok(Self {
            inner: Mutex::new(inner),
            eos,
            begin,
            end,
            bound,
            memoization_off: serving,
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, Tokenizer>> {
        self.inner
            .lock()
            .map_err(|_| Error::Malformed("tokenizer lock poisoned; source owner retained".into()))
    }

    pub fn prompt_tokens(&self, message: &str, chat: bool) -> Result<Vec<u32>> {
        if message.len() > 1024 * 1024 {
            return Err(Error::Limit("prompt input exceeds 1 MiB".into()));
        }
        let rendered = if chat {
            format!("<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n")
        } else {
            message.to_owned()
        };
        // A single encode lock; do not lock prompt_tokens separately.
        self.encode(&rendered)
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let inner = self.lock()?;
        let encoding = inner
            .encode(text, false)
            .map_err(|e| Error::Malformed(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        let inner = self.lock()?;
        inner
            .decode(tokens, skip_special)
            .map_err(|e| Error::Malformed(format!("tokenizer decode: {e}")))
    }

    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// Retained source-owner inventory for the approved profile.
    ///
    /// With the private `Mutex` the caller cardinality is `C = 1`, so each
    /// Split delegate's cache pool holds at most `P + 1` objects (owner plus
    /// one stack cache); `P` is the process-global pool-stack count read from
    /// the pinned `regex-automata 0.4.18 meta::Config::get_pool_capacity` (the
    /// same cached value the production `meta::Regex` uses), never a fresh
    /// parallelism sample. Counts come from the compiled literal descriptor and
    /// the pinned collection layouts in [`bounds`].
    pub fn serving_inventory(&self) -> Result<TokenizerPlan> {
        if !self.memoization_off {
            return Err(Error::Malformed(
                "serving inventory requires the approved profile with bounded BPE cache".into(),
            ));
        }
        // The parsed source's measured envelope must match the approved one
        // before the plan that prices it is published.
        if self.bound != ServingBound::approved() {
            return Err(Error::Malformed(
                "parsed tokenizer envelope disagrees with the approved inventory".into(),
            ));
        }
        approved_plan()
    }

    /// One `decode` call's scratch peak for `tokens` output IDs.
    pub fn decode_scratch_bound(&self, tokens: usize) -> usize {
        usize::try_from(bounds::decode_peak(tokens as u64).unwrap_or(u64::MAX))
            .unwrap_or(usize::MAX)
    }

    pub fn serving_bound(&self) -> Result<ServingBound> {
        Ok(self.bound)
    }

    /// Longest decoded vocabulary/added-token entry, tripled for lossy UTF-8
    /// expansion after ByteLevel's per-token reverse mapping. Approved:
    /// 256 x 3 = 768 bytes per decoded token.
    pub fn max_token_bytes(&self) -> Result<usize> {
        Ok(self
            .bound
            .max_decoded_token_bytes
            .saturating_mul(NFC_MAX_UTF8_EXPANSION))
    }

    /// Prompt IDs: NFC-normalized bytes N <= 3B, ByteLevel emits one mapped
    /// scalar per normalized byte, and no-fallback/no-affix BPE only merges.
    /// Added tokens bypass normalization but their bytes are included in B.
    pub fn prompt_token_bound(&self, message_len: usize, chat: bool) -> usize {
        self.rendered_bound(message_len, chat)
            .saturating_mul(NFC_MAX_UTF8_EXPANSION)
    }

    /// One `encode` scratch peak for `B` rendered bytes.
    ///
    /// Named phase sum in [`bounds::prompt_peak`] (NFC, Split, ByteLevel, BPE,
    /// Encoding sidecars and the fancy-regex VM), with the content bounds
    /// `E <= 6B`, `K <= 2B`, `T <= 3B`, `S <= 3B` used as lengths, not as a
    /// per-byte heap allowance. Overflow saturates; admission then fails the
    /// host ledger instead of silently accepting the request.
    pub fn prompt_scratch_bound(&self, message_len: usize, chat: bool) -> usize {
        let text = self.rendered_bound(message_len, chat);
        usize::try_from(bounds::prompt_peak(text as u64).unwrap_or(u64::MAX)).unwrap_or(usize::MAX)
    }

    /// One serving request's admitted-input scratch peak for `B` rendered
    /// bytes, excluding the serialized VM row.
    ///
    /// Callers must hold this actual tokenizer's `Retained` owner, whose
    /// checked `G + V` scratch lease prepays the VM once per tokenizer (shared
    /// by aliases). Requires the approved profile with BPE memoization off and
    /// the exact approved envelope; no unapproved/fallback profile and no
    /// subtraction or zero fallback. Above the original prompt cap this
    /// refuses instead of guessing.
    pub fn admitted_prompt_scratch_bound(&self, message_len: usize, chat: bool) -> Result<u64> {
        if !self.memoization_off || self.bound != ServingBound::approved() {
            return Err(Error::Malformed(
                "admitted prompt bound requires the approved profile with bounded BPE cache".into(),
            ));
        }
        let text = u64::try_from(self.rendered_bound(message_len, chat))
            .map_err(|_| Error::Limit("rendered prompt length".into()))?;
        bounds::admitted_prompt_peak(text)
            .ok_or_else(|| Error::Limit("prompt exceeds the priced serving cap".into()))
    }

    /// B = message bytes plus the exact rendered chat framing.
    fn rendered_bound(&self, message_len: usize, chat: bool) -> usize {
        message_len.saturating_add(if chat { CHAT_TEMPLATE_BYTES } else { 0 })
    }

    pub fn special_ids(&self) -> (u32, u32) {
        (self.begin, self.end)
    }
}

fn approved_source(bytes: &[u8]) -> bool {
    bytes.len() as u64 == APPROVED_TOKENIZER_BYTES
        && format!("{:x}", Sha256::digest(bytes)) == APPROVED_TOKENIZER_SHA256
}

/// Source descriptor for unverified test/exploratory bytes. It never grants
/// approved-source identity; production preparation SHA-gates first.
fn source_bound(bytes: &[u8]) -> std::result::Result<ServingBound, String> {
    let root: Value = serde_json::from_slice(bytes).map_err(|error| format!("parse: {error}"))?;
    if let Some(normalizer) = root.get("normalizer") {
        no_expanding_normalizer(normalizer)?;
    }
    let model = root
        .get("model")
        .ok_or_else(|| "missing tokenizer model".to_string())?;
    let mut max_decoded_token_bytes = 0usize;
    match model.get("vocab") {
        Some(Value::Object(vocab)) => {
            for token in vocab.keys() {
                max_decoded_token_bytes = max_decoded_token_bytes.max(token.len());
            }
        }
        Some(Value::Array(vocab)) => {
            for entry in vocab {
                if let Some(token) = entry.as_array().and_then(|values| values.first())
                    && let Some(token) = token.as_str()
                {
                    max_decoded_token_bytes = max_decoded_token_bytes.max(token.len());
                }
            }
        }
        _ => return Err("tokenizer model vocab must be an object or array".into()),
    }
    if let Some(added) = root.get("added_tokens").and_then(Value::as_array) {
        for token in added {
            if let Some(content) = token.get("content").and_then(Value::as_str) {
                max_decoded_token_bytes = max_decoded_token_bytes.max(content.len());
            }
        }
    }
    if max_decoded_token_bytes == 0 {
        return Err("empty tokenizer vocabulary".into());
    }
    let prompt_prefix_bytes = root
        .get("pre_tokenizer")
        .map(text_prefix_bound)
        .unwrap_or(0);
    Ok(ServingBound {
        max_decoded_token_bytes,
        prompt_prefix_bytes,
    })
}

fn no_expanding_normalizer(value: &Value) -> std::result::Result<(), String> {
    match value.get("type").and_then(Value::as_str) {
        Some("Replace") | Some("Prepend") => {
            Err("expanding normalizer has no source-derived text bound".into())
        }
        Some("Sequence") => {
            if let Some(parts) = value.get("normalizers").and_then(Value::as_array) {
                for part in parts {
                    no_expanding_normalizer(part)?;
                }
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// UTF-8 bytes a pre-tokenizer can add before Split; ByteLevel's space is one
/// byte, or four when it is the mapped prefix scalar. Other pre-tokenizers add
/// no source-independent prefix.
fn text_prefix_bound(value: &Value) -> usize {
    match value.get("type").and_then(Value::as_str) {
        Some("ByteLevel")
            if value.get("add_prefix_space").and_then(Value::as_bool) == Some(true) =>
        {
            4
        }
        Some("Sequence") => value
            .get("pretokenizers")
            .and_then(Value::as_array)
            .map(|parts| {
                parts
                    .iter()
                    .map(text_prefix_bound)
                    .fold(0, usize::saturating_add)
            })
            .unwrap_or(0),
        _ => 0,
    }
}

/// Exact approved descriptor, used by tests to prove option-level recognition
/// independently of the file hash.
#[cfg(test)]
fn exact_approved_profile(root: &Value) -> std::result::Result<(), String> {
    let normalizer = root
        .get("normalizer")
        .ok_or_else(|| "normalizer must be present".to_string())?;
    object_with_exact_keys(normalizer, &["type"])?;
    if normalizer.get("type").and_then(Value::as_str) != Some("NFC") {
        return Err("normalizer must be exact NFC".into());
    }
    if root.get("padding").is_some_and(|value| !value.is_null())
        || root.get("truncation").is_some_and(|value| !value.is_null())
    {
        return Err("padding/truncation must be absent for the prompt bound".into());
    }
    let pre = root
        .get("pre_tokenizer")
        .ok_or_else(|| "pre-tokenizer missing".to_string())?;
    object_with_exact_keys(pre, &["type", "pretokenizers"])?;
    if pre.get("type").and_then(Value::as_str) != Some("Sequence") {
        return Err("pre-tokenizer must be the approved Sequence".into());
    }
    let parts = pre
        .get("pretokenizers")
        .and_then(Value::as_array)
        .ok_or_else(|| "pre-tokenizer sequence missing entries".to_string())?;
    if parts.len() != 2 {
        return Err("approved pre-tokenizer has exactly Split + ByteLevel".into());
    }
    let split = &parts[0];
    object_with_exact_keys(split, &["type", "pattern", "behavior", "invert"])?;
    if split.get("type").and_then(Value::as_str) != Some("Split")
        || split
            .get("pattern")
            .and_then(|pattern| pattern.get("Regex"))
            .and_then(Value::as_str)
            != Some(APPROVED_SPLIT_REGEX)
        || split.get("behavior").and_then(Value::as_str) != Some("Isolated")
        || split.get("invert").and_then(Value::as_bool) != Some(false)
    {
        return Err("pre-tokenizer Split differs from the approved expression/options".into());
    }
    approved_byte_level(&parts[1], "pre-tokenizer ByteLevel")?;
    let post = root
        .get("post_processor")
        .ok_or_else(|| "post-processor missing".to_string())?;
    approved_byte_level(post, "post-processor ByteLevel")?;
    let decoder = root
        .get("decoder")
        .ok_or_else(|| "decoder missing".to_string())?;
    approved_byte_level(decoder, "decoder ByteLevel")?;

    let model = root
        .get("model")
        .ok_or_else(|| "missing tokenizer model".to_string())?;
    object_with_exact_keys(
        model,
        &[
            "type",
            "dropout",
            "unk_token",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "fuse_unk",
            "byte_fallback",
            "ignore_merges",
            "vocab",
            "merges",
        ],
    )?;
    if model.get("type").and_then(Value::as_str) != Some("BPE")
        || !model.get("dropout").is_some_and(Value::is_null)
        || !model.get("unk_token").is_some_and(Value::is_null)
        || model
            .get("continuing_subword_prefix")
            .and_then(Value::as_str)
            != Some("")
        || model.get("end_of_word_suffix").and_then(Value::as_str) != Some("")
        || model.get("fuse_unk").and_then(Value::as_bool) != Some(false)
        || model.get("byte_fallback").and_then(Value::as_bool) != Some(false)
        || model.get("ignore_merges").and_then(Value::as_bool) != Some(false)
        || !model.get("merges").is_some_and(Value::is_array)
    {
        return Err("BPE options differ from the approved no-affix profile".into());
    }
    let added = root
        .get("added_tokens")
        .and_then(Value::as_array)
        .ok_or_else(|| "added tokens missing".to_string())?;
    for token in added {
        let object = token
            .as_object()
            .ok_or_else(|| "added token must be an object".to_string())?;
        if !object.get("special").is_some_and(Value::is_boolean)
            || object.get("normalized").and_then(Value::as_bool) != Some(false)
            || ["single_word", "lstrip", "rstrip"].iter().any(|key| {
                object
                    .get(*key)
                    .is_some_and(|value| value.as_bool() != Some(false))
            })
        {
            return Err("added token normalization/word options differ from approved".into());
        }
    }
    Ok(())
}

#[cfg(test)]
fn approved_byte_level(value: &Value, label: &str) -> std::result::Result<(), String> {
    object_with_exact_keys(
        value,
        &["type", "add_prefix_space", "trim_offsets", "use_regex"],
    )?;
    if value.get("type").and_then(Value::as_str) != Some("ByteLevel")
        || value.get("add_prefix_space").and_then(Value::as_bool) != Some(false)
        || value.get("trim_offsets").and_then(Value::as_bool) != Some(false)
        || value.get("use_regex").and_then(Value::as_bool) != Some(false)
    {
        return Err(format!("{label} differs from approved options"));
    }
    Ok(())
}

#[cfg(test)]
fn object_with_exact_keys(value: &Value, expected: &[&str]) -> std::result::Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| "expected object".to_string())?;
    if object.len() != expected.len() || expected.iter().any(|key| !object.contains_key(*key)) {
        return Err("object keys differ from approved profile".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(value: serde_json::Value) -> std::result::Result<(), String> {
        exact_approved_profile(&value)
    }

    fn bound(value: serde_json::Value) -> std::result::Result<ServingBound, String> {
        source_bound(&serde_json::to_vec(&value).unwrap())
    }

    fn base() -> serde_json::Value {
        serde_json::json!({
            "normalizer":{"type":"NFC"},
            "padding":null, "truncation":null,
            "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "pre_tokenizer":{"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":APPROVED_SPLIT_REGEX},"behavior":"Isolated","invert":false},
                {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}
            ]},
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"BPE","dropout":null,"unk_token":null,
                "continuing_subword_prefix":"","end_of_word_suffix":"",
                "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                "vocab":{"a":0,"longer-token":1},"merges":[]},
            "added_tokens":[{"id":2,"content":"<|added|>","single_word":false,
                "lstrip":false,"rstrip":false,"normalized":false,"special":true}]
        })
    }

    /// Approved-source cross-check; skips when the pinned file is absent.
    ///
    /// `ZG_APPROVED_TOKENIZER` names the approved `tokenizer.json` (12,807,982
    /// bytes, SHA-256 5f9e4d49...cb42).
    #[test]
    fn approved_source_plan_prompt_and_full_prefix_decode_are_consistent() {
        let Some(path) = std::env::var_os("ZG_APPROVED_TOKENIZER") else {
            return;
        };
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len(), 12_807_982);
        let tokenizer = QwenTokenizer::from_approved_bytes(&bytes).unwrap();
        // The measured source envelope must agree with the priced plan.
        assert_eq!(
            tokenizer.serving_inventory().unwrap(),
            approved_plan().unwrap()
        );
        let text = "Hyv\u{e4}\u{e4} p\u{e4}iv\u{e4}\u{e4} \u{2014} 0.8B \u{fc}n\u{ef}c\u{f8}d\u{e9} \u{2713} \u{1f600}";
        // Chat framing adds exactly the approved template bytes.
        let framed = format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(framed.len(), text.len() + CHAT_TEMPLATE_BYTES);
        let chat = tokenizer.prompt_tokens(text, true).unwrap();
        assert_eq!(chat, tokenizer.prompt_tokens(&framed, false).unwrap());
        // The presenter decodes the whole committed prefix each commit.
        let full = tokenizer.decode(&chat, false).unwrap();
        assert!(full.starts_with("<|im_start|>user\n"));
        assert!(full.len() <= tokenizer.prompt_scratch_bound(text.len(), true));
        let mut ids = Vec::with_capacity(512);
        while ids.len() < 512 {
            ids.extend_from_slice(&chat);
        }
        ids.truncate(512);
        let prefix = tokenizer.decode(&ids, false).unwrap();
        assert!(prefix.len() <= tokenizer.decode_scratch_bound(512));
        assert!(tokenizer.decode_scratch_bound(512) > 0);
        // Bounded input: the adapter refuses oversized prompts, not the engine.
        assert!(
            tokenizer
                .prompt_tokens(&"a".repeat(1024 * 1024 + 1), false)
                .is_err()
        );
    }

    #[test]
    fn stored_47_fixture_prompt_lengths_are_measured_from_the_approved_source() {
        // `ZG_SHORT_HF_PROMPTS` is the existing fixed original oracle JSON;
        // `ZG_APPROVED_TOKENIZER` is the exact 12,807,982-byte source. Both
        // are optional so ordinary CPU tests do not require model payloads.
        let (Some(prompts), Some(tokenizer_path)) = (
            std::env::var_os("ZG_SHORT_HF_PROMPTS"),
            std::env::var_os("ZG_APPROVED_TOKENIZER"),
        ) else {
            return;
        };
        let public: Value = serde_json::from_slice(&std::fs::read(prompts).unwrap()).unwrap();
        let tokenizer =
            QwenTokenizer::from_approved_bytes(&std::fs::read(tokenizer_path).unwrap()).unwrap();
        for rows in [47usize, 57, 62, 67, 72] {
            let ids: Vec<u32> = public[rows.to_string()]["prompt_token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
                .collect();
            assert_eq!(ids.len(), rows);
            // Measure the actual rendered/decoded byte length; never infer it
            // from the token count or assume the 512-byte priced envelope.
            let text = tokenizer.decode(&ids, false).unwrap();
            let bytes = u64::try_from(text.len()).unwrap();
            assert!(
                bytes <= (rows as u64).saturating_mul(tokenizer.max_token_bytes().unwrap() as u64)
            );
            assert!(prompt_peak(bytes).is_some());
            // The original tokenizer must reproduce the stored source prompt
            // exactly after its own decode.
            assert_eq!(tokenizer.encode(&text).unwrap(), ids, "row {rows}");
        }
    }

    #[test]
    fn source_bound_uses_actual_vocab_and_approved_nfc_profile() {
        assert!(profile(base()).is_ok());
        let bound = bound(base()).unwrap();
        assert_eq!(bound.max_decoded_token_bytes(), "longer-token".len());
        assert_eq!(bound.prompt_prefix_bytes(), 0);
    }

    #[test]
    fn expanding_or_nonapproved_source_configuration_is_refused() {
        let mut normalizer = base();
        normalizer["normalizer"] =
            serde_json::json!({"type":"Replace","pattern":{"String":"x"},"content":"xx"});
        assert!(bound(normalizer).is_err());
        let mut nfc = base();
        nfc["normalizer"] = serde_json::json!({"type":"NFKC"});
        assert!(profile(nfc).is_err());
        let mut regex = base();
        regex["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = serde_json::json!("\\s+");
        assert!(profile(regex).is_err());
        let mut decoder = base();
        decoder["decoder"] =
            serde_json::json!({"type":"Replace","pattern":{"String":"x"},"content":"xx"});
        assert!(profile(decoder).is_err());
        let mut post = base();
        post["post_processor"] = serde_json::json!({"type":"TemplateProcessing","single":[],"pair":[],"special_tokens":{}});
        assert!(profile(post).is_err());
        let mut padding = base();
        padding["padding"] = serde_json::json!({"strategy":"Fixed","length":8});
        assert!(profile(padding).is_err());
        let mut fallback = base();
        fallback["model"]["byte_fallback"] = serde_json::json!(true);
        assert!(profile(fallback).is_err());
        let mut normalized = base();
        normalized["added_tokens"][0]["normalized"] = serde_json::json!(true);
        assert!(profile(normalized).is_err());
    }

    #[test]
    fn approved_sha_and_decoded_bounds_are_exact() {
        assert_eq!(ServingBound::approved().max_decoded_token_bytes(), 256);
        assert_eq!(
            ServingBound::approved()
                .max_decoded_token_bytes
                .saturating_mul(NFC_MAX_UTF8_EXPANSION),
            768
        );
        assert_eq!(APPROVED_TOKENIZER_BYTES, 12_807_982);
    }

    #[test]
    fn approved_plan_uses_the_actual_pool_capacity_and_names_vm() {
        let expected = regex_automata::meta::Config::new()
            .get_pool_capacity()
            .max(1) as u64;
        let plan = approved_plan().unwrap();
        assert_eq!(plan.pool_stacks, expected);
        assert_eq!(plan.serialized_vm_bytes, 224_000_144);
        // Runtime labels run under the full inherited 0..15 mask. Assert the
        // real production P16 there instead of overriding the derived value.
        if std::env::var_os("ZG_EXPECT_FULL_RUNTIME_MASK").is_some() {
            assert_eq!(std::thread::available_parallelism().unwrap().get(), 16);
            assert_eq!(expected, 16);
            assert_eq!(plan.pool_stacks, 16);
        }
        println!(
            "actual approved P={} R={} G={} V={}",
            plan.pool_stacks,
            plan.retained_bytes,
            plan.request_cache_growth_bytes,
            plan.serialized_vm_bytes
        );
    }
}
