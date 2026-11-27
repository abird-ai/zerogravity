//! BERT encoder semantics and sentence-transformer module closure.
use crate::{Error, Result};
use serde::Deserialize;

const MAX_CONFIG: usize = 64 * 1024;
#[derive(Clone, Debug, PartialEq)]
pub struct BertSentenceConfig {
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) heads: usize,
    pub(crate) layers: usize,
    pub(crate) vocabulary: usize,
    pub(crate) positions: usize,
    pub(crate) types: usize,
    pub(crate) max_sequence: usize,
    pub(crate) epsilon: f32,
    pub(crate) pad_id: u32,
}
#[derive(Deserialize)]
struct Encoder {
    model_type: String,
    architectures: Vec<String>,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: usize,
    layer_norm_eps: f32,
    pad_token_id: u32,
    position_embedding_type: String,
    #[serde(default)]
    is_decoder: bool,
    #[serde(default)]
    add_cross_attention: bool,
    #[serde(default)]
    chunk_size_feed_forward: usize,
    #[serde(default)]
    position_embedding_offset: usize,
}
#[derive(Deserialize)]
struct Sentence {
    max_seq_length: usize,
    do_lower_case: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Module {
    idx: usize,
    name: String,
    path: String,
    #[serde(rename = "type")]
    kind: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pooling {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
    #[serde(default)]
    pooling_mode_weightedmean_tokens: bool,
    #[serde(default)]
    pooling_mode_lasttoken: bool,
}

fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    if bytes.is_empty() || bytes.len() > MAX_CONFIG {
        return Err(Error::Limit("BERT config byte limit".into()));
    }
    // Struct deserialization rejects duplicate semantic fields.
    serde_json::from_slice(bytes).map_err(|e| Error::Malformed(format!("BERT config: {e}")))
}
impl BertSentenceConfig {
    /// First bounded profile: absolute-position, bidirectional dense BERT,
    /// exact-erf GELU, mean pooling then L2 normalize. Geometry comes from
    /// source config, not one hard-coded model size or a fixture whitelist.
    pub fn parse(encoder: &[u8], sentence: &[u8], modules: &[u8], pooling: &[u8]) -> Result<Self> {
        let e: Encoder = parse(encoder)?;
        let s: Sentence = parse(sentence)?;
        let m: Vec<Module> = parse(modules)?;
        let p: Pooling = parse(pooling)?;
        let expected = [
            ("0", "", "sentence_transformers.models.Transformer"),
            ("1", "1_Pooling", "sentence_transformers.models.Pooling"),
            ("2", "2_Normalize", "sentence_transformers.models.Normalize"),
        ];
        if e.model_type != "bert"
            || e.architectures != ["BertModel"]
            || e.hidden_act != "gelu"
            || e.position_embedding_type != "absolute"
            || e.is_decoder
            || e.add_cross_attention
            || e.chunk_size_feed_forward != 0
            || e.position_embedding_offset != 0
            || s.do_lower_case
            || m.len() != expected.len()
            || !m
                .iter()
                .zip(expected)
                .enumerate()
                .all(|(idx, (m, (name, path, kind)))| {
                    m.idx == idx && m.name == name && m.path == path && m.kind == kind
                })
            || p.word_embedding_dimension != e.hidden_size
            || !p.pooling_mode_mean_tokens
            || p.pooling_mode_cls_token
            || p.pooling_mode_max_tokens
            || p.pooling_mode_mean_sqrt_len_tokens
            || p.pooling_mode_weightedmean_tokens
            || p.pooling_mode_lasttoken
        {
            return Err(Error::Unsupported(
                "not the supported BERT mean-normalized encoder contract".into(),
            ));
        }
        // Explicit bounds for the initial CPU candidate. No silent position
        // truncation or alternate activation/pooling/decoder selection.
        if !(1..=4096).contains(&e.hidden_size)
            || !(1..=16384).contains(&e.intermediate_size)
            || !(1..=128).contains(&e.num_attention_heads)
            || !e.hidden_size.is_multiple_of(e.num_attention_heads)
            || !(1..=64).contains(&e.num_hidden_layers)
            || !(1..=262144).contains(&e.vocab_size)
            || !(1..=8192).contains(&e.max_position_embeddings)
            || !(1..=64).contains(&e.type_vocab_size)
            || !(1..=512).contains(&s.max_seq_length)
            || s.max_seq_length > e.max_position_embeddings
            || e.pad_token_id as usize >= e.vocab_size
            || !e.layer_norm_eps.is_finite()
            || e.layer_norm_eps <= 0.0
        {
            return Err(Error::InvalidInput(
                "BERT encoder geometry outside initial CPU bounds".into(),
            ));
        }
        Ok(Self {
            hidden: e.hidden_size,
            intermediate: e.intermediate_size,
            heads: e.num_attention_heads,
            layers: e.num_hidden_layers,
            vocabulary: e.vocab_size,
            positions: e.max_position_embeddings,
            types: e.type_vocab_size,
            max_sequence: s.max_seq_length,
            epsilon: e.layer_norm_eps,
            pad_id: e.pad_token_id,
        })
    }
    pub fn dimensions(&self) -> usize {
        self.hidden
    }
    pub fn max_sequence(&self) -> usize {
        self.max_sequence
    }
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary
    }
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
}
