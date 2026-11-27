//! Source-bound, admitted F32 CPU sentence encoder. No scheduler or wire
//! owner is introduced here. The original snapshot and resource domain are
//! required; no detached path lookup or guessed model coefficients.
use crate::{BertSentenceConfig, Error, Result};
use std::collections::BTreeSet;
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::{SourceWeights, artifact::TensorInfo, cache::SnapshotLease};
use zerogravity_core::{
    accelerator::{EmbeddingRegion, MatVecRegion, SourceIdentity},
    encoding::{ScalarDType, StoredTensorEncoding},
    resource_arbiter::{
        AllocationClass, AllocationRequest, GlobalResourceLease, TransferAdmission,
    },
};
use zerogravity_cpu::{
    ops::encoder::{bidirectional_attention, check_cancel, gelu, layer_norm, mean_l2_normalize},
    source::{source_reference_embedding, source_reference_projection},
};

mod tokenizer;
use tokenizer::BertTokenizer;

const F32: StoredTensorEncoding = StoredTensorEncoding::Scalar(ScalarDType::F32);
const MAX_SOURCE: u64 = 512 * 1024 * 1024;
const CONFIG_LIMIT: u64 = 64 * 1024;
fn overflow() -> Error {
    Error::Limit("BERT allocation extent overflow".into())
}
fn reserve(
    admission: &TransferAdmission,
    kind: AllocationClass,
    bytes: u64,
) -> Result<GlobalResourceLease> {
    admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest { kind, bytes }],
    )
}

struct Binder<'a> {
    source: &'a SourceWeights,
    used: BTreeSet<String>,
}
impl Binder<'_> {
    fn tensor(
        &mut self,
        name: &str,
        shape: &[usize],
        kind: StoredTensorEncoding,
    ) -> Result<TensorInfo> {
        let info = self.source.tensor(name)?;
        if info.kind != kind
            || !info
                .shape
                .iter()
                .copied()
                .eq(shape.iter().map(|&n| n as u64))
        {
            return Err(Error::Malformed(format!(
                "BERT source tensor {name}: incompatible shape or dtype"
            )));
        }
        let data = self.source.bytes(info)?;
        if kind == F32
            && !data
                .chunks_exact(4)
                .all(|b| f32::from_le_bytes(b.try_into().expect("f32 bytes")).is_finite())
        {
            return Err(Error::Malformed(format!("nonfinite BERT weight {name}")));
        }
        if !self.used.insert(name.to_owned()) {
            return Err(Error::Malformed("duplicate BERT tensor binding".into()));
        }
        Ok(info.clone())
    }

    fn vector(&mut self, name: &str, n: usize) -> Result<Vec<f32>> {
        let info = self.tensor(name, &[n], F32)?;
        zerogravity_artifacts::codecs::dequantize(F32, self.source.bytes(&info)?, n)
    }
    fn linear(&mut self, name: &str, input: usize, output: usize) -> Result<Linear> {
        Ok(Linear {
            weight: self.tensor(&format!("{name}.weight"), &[output, input], F32)?,
            bias: self.vector(&format!("{name}.bias"), output)?,
            input,
            output,
        })
    }
    fn norm(&mut self, name: &str, width: usize) -> Result<Norm> {
        Ok(Norm {
            weight: self.vector(&format!("{name}.weight"), width)?,
            bias: self.vector(&format!("{name}.bias"), width)?,
        })
    }
    fn index_buffer(&mut self, name: &str, positions: usize, position_ids: bool) -> Result<()> {
        if !self.source.contains(name) {
            return Ok(());
        }
        let tensor = self.tensor(name, &[1, positions], StoredTensorEncoding::I64)?;
        // Persisted buffers are part of the original source. Do not ignore
        // nonstandard positions or nonzero token types as harmless extras.
        if !self
            .source
            .bytes(&tensor)?
            .chunks_exact(8)
            .enumerate()
            .all(|(i, b)| {
                i64::from_le_bytes(b.try_into().expect("i64 bytes"))
                    == if position_ids { i as i64 } else { 0 }
            })
        {
            return Err(Error::Unsupported(format!(
                "BERT {name} is not the standard generated index buffer"
            )));
        }
        Ok(())
    }
    fn finish(self) -> Result<()> {
        if self.source.tensor_names().any(|n| !self.used.contains(n)) {
            return Err(Error::Unsupported("unbound BERT source tensors".into()));
        }
        Ok(())
    }
}

struct Linear {
    weight: TensorInfo,
    bias: Vec<f32>,
    input: usize,
    output: usize,
}
impl Linear {
    fn rows(
        &self,
        source: &SourceWeights,
        input: &[f32],
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        if input.is_empty() || !input.len().is_multiple_of(self.input) {
            return Err(Error::InvalidInput("BERT affine input extent".into()));
        }
        let data = source.bytes(&self.weight)?;
        let n = (input.len() / self.input)
            .checked_mul(self.output)
            .ok_or_else(overflow)?;
        let mut result = Vec::with_capacity(n);
        for x in input.chunks_exact(self.input) {
            check_cancel(cancel)?;
            // Original shared source GEMV, HF shape [out,in] supplied
            // explicitly rather than the GGUF dimension-order convenience.
            let row = source_reference_projection(MatVecRegion {
                source_identity: source.source_identity(),
                tensor_name: &self.weight.name,
                kind: F32,
                columns: self.input,
                rows: self.output,
                packed_weights: data,
                activation: x,
            })?;
            result.extend(row.into_iter().zip(&self.bias).map(|(a, b)| a + b));
        }
        if !result.iter().all(|x| x.is_finite()) {
            return Err(Error::Malformed("nonfinite affine output".into()));
        }
        Ok(result)
    }
}
struct Norm {
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Norm {
    fn rows(&self, values: &mut [f32], epsilon: f32) -> Result<()> {
        for row in values.chunks_exact_mut(self.weight.len()) {
            layer_norm(row, &self.weight, &self.bias, epsilon)?;
        }
        Ok(())
    }
}

struct Layer {
    query: Linear,
    key: Linear,
    value: Linear,
    attention: Linear,
    attention_norm: Norm,
    intermediate: Linear,
    output: Linear,
    output_norm: Norm,
}
impl Layer {
    fn bind(b: &mut Binder<'_>, root: &str, i: usize, c: &BertSentenceConfig) -> Result<Self> {
        let p = format!("{root}encoder.layer.{i}");
        let h = c.hidden;
        Ok(Self {
            query: b.linear(&format!("{p}.attention.self.query"), h, h)?,
            key: b.linear(&format!("{p}.attention.self.key"), h, h)?,
            value: b.linear(&format!("{p}.attention.self.value"), h, h)?,
            attention: b.linear(&format!("{p}.attention.output.dense"), h, h)?,
            attention_norm: b.norm(&format!("{p}.attention.output.LayerNorm"), h)?,
            intermediate: b.linear(&format!("{p}.intermediate.dense"), h, c.intermediate)?,
            output: b.linear(&format!("{p}.output.dense"), c.intermediate, h)?,
            output_norm: b.norm(&format!("{p}.output.LayerNorm"), h)?,
        })
    }

    fn forward(
        &self,
        source: &SourceWeights,
        c: &BertSentenceConfig,
        input: Vec<f32>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        let attention = {
            let q = self.query.rows(source, &input, cancel)?;
            let k = self.key.rows(source, &input, cancel)?;
            let v = self.value.rows(source, &input, cancel)?;
            bidirectional_attention(&q, &k, &v, c.hidden, c.heads, cancel)?
        };
        let mut hidden = self.attention.rows(source, &attention, cancel)?;
        drop(attention);
        for (a, b) in hidden.iter_mut().zip(input) {
            *a += b;
        }
        self.attention_norm.rows(&mut hidden, c.epsilon)?;
        let mut intermediate = self.intermediate.rows(source, &hidden, cancel)?;
        for v in &mut intermediate {
            *v = gelu(*v);
        }
        let mut out = self.output.rows(source, &intermediate, cancel)?;
        drop(intermediate);
        for (a, b) in out.iter_mut().zip(hidden) {
            *a += b;
        }
        self.output_norm.rows(&mut out, c.epsilon)?;
        Ok(out)
    }
}

/// Private encoder candidate with exact owned source. No loaded model
/// registration, fake text generation session, or independently owned queue.
pub struct BertSentenceEncoder {
    source: SourceWeights,
    config: BertSentenceConfig,
    tokenizer: BertTokenizer,
    words: TensorInfo,
    positions: TensorInfo,
    types: TensorInfo,
    norm: Norm,
    layers: Vec<Layer>,
    admission: TransferAdmission,
    _parameters: GlobalResourceLease,
}
/// Result allocation retains the physical admission until last owner drops.
pub struct Embedding {
    values: Vec<f32>,
    tokens: usize,
    source: SourceIdentity,
    _output: GlobalResourceLease,
}
impl Embedding {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
}

impl BertSentenceEncoder {
    pub fn load(snapshot: &SnapshotLease, admission: &TransferAdmission) -> Result<Self> {
        // Config/module parsing is cold, bounded and charged before copies.
        let _metadata = reserve(admission, AllocationClass::Scratch, 4 * 1024 * 1024)?;
        let config = BertSentenceConfig::parse(
            &snapshot.read_verified("config.json", CONFIG_LIMIT)?,
            &snapshot.read_verified("sentence_bert_config.json", CONFIG_LIMIT)?,
            &snapshot.read_verified("modules.json", CONFIG_LIMIT)?,
            &snapshot.read_verified("1_Pooling/config.json", CONFIG_LIMIT)?,
        )?;
        // Exact owned source digest and tensor directory. No sharded
        // index manufactured for a single-file checkpoint.
        let tokenizer = BertTokenizer::load(snapshot, &config, admission)?;
        let source = snapshot.single_safetensors("model.safetensors", MAX_SOURCE, admission)?;
        Self::bind(config, tokenizer, source, admission)
    }

    fn bind(
        config: BertSentenceConfig,
        tokenizer: BertTokenizer,
        source: SourceWeights,
        admission: &TransferAdmission,
    ) -> Result<Self> {
        let h = config.hidden;
        // Decoded biases and affine norm coefficients, plus conservative
        // bounded descriptor/name/shape backing. Original matrices are
        // borrowed directly from the retained Safetensors allocation.
        let floats = config
            .layers
            .checked_mul(9 * h + config.intermediate)
            .and_then(|n| n.checked_add(2 * h))
            .ok_or_else(overflow)?;
        let bytes = (floats as u64)
            .checked_mul(4)
            .and_then(|n| n.checked_add((config.layers as u64 + 1) * 64 * 1024))
            .ok_or_else(overflow)?;
        let parameters = reserve(admission, AllocationClass::ImmutableWeights, bytes)?;
        let mut b = Binder {
            source: &source,
            used: BTreeSet::new(),
        };
        let root = if source.contains("embeddings.word_embeddings.weight") {
            ""
        } else if source.contains("bert.embeddings.word_embeddings.weight") {
            "bert."
        } else {
            return Err(Error::Malformed("missing BERT word embeddings".into()));
        };
        let prefix = format!("{root}embeddings");
        let words = b.tensor(
            &format!("{prefix}.word_embeddings.weight"),
            &[config.vocabulary, h],
            F32,
        )?;
        let positions = b.tensor(
            &format!("{prefix}.position_embeddings.weight"),
            &[config.positions, h],
            F32,
        )?;
        let types = b.tensor(
            &format!("{prefix}.token_type_embeddings.weight"),
            &[config.types, h],
            F32,
        )?;
        let norm = b.norm(&format!("{prefix}.LayerNorm"), h)?;
        b.index_buffer(&format!("{prefix}.position_ids"), config.positions, true)?;
        b.index_buffer(&format!("{prefix}.token_type_ids"), config.positions, false)?;
        let layers = (0..config.layers)
            .map(|i| Layer::bind(&mut b, root, i, &config))
            .collect::<Result<Vec<_>>>()?;
        // The original optional BERT dense CLS pooler is not used by the
        // SentenceTransformer graph. Check its complete pair if present.
        let pool = format!("{root}pooler.dense");
        if source.contains(&format!("{pool}.weight")) || source.contains(&format!("{pool}.bias")) {
            b.tensor(&format!("{pool}.weight"), &[h, h], F32)?;
            b.tensor(&format!("{pool}.bias"), &[h], F32)?;
        }
        b.finish()?;
        Ok(Self {
            source,
            config,
            tokenizer,
            words,
            positions,
            types,
            norm,
            layers,
            admission: admission.clone(),
            _parameters: parameters,
        })
    }

    pub fn config(&self) -> &BertSentenceConfig {
        &self.config
    }
    pub fn source(&self) -> SourceIdentity {
        self.source.source_identity()
    }

    fn lookup(&self, tensor: &TensorInfo, id: u32) -> Result<Vec<f32>> {
        source_reference_embedding(EmbeddingRegion {
            source_identity: self.source(),
            tensor_name: &tensor.name,
            kind: F32,
            packed_weights: self.source.bytes(tensor)?,
            token: id,
            width: self.config.hidden,
            vocabulary: tensor.shape[0] as usize,
        })
    }

    /// Model-owned original WordPiece input, including CLS/SEP and exact
    /// encoded token usage. Refuse overlong input rather than truncate.
    pub fn embed_text(&self, text: &str, cancel: Option<&AtomicBool>) -> Result<Embedding> {
        check_cancel(cancel)?;
        let (tokens, _charge) = self.tokenizer.encode(text, &self.config, &self.admission)?;
        self.embed_tokens(&tokens, cancel)
    }

    /// Same actual prepared domain must cover scheduling and model scratch.
    pub fn validate_admission(&self, admission: &TransferAdmission) -> Result<()> {
        self._parameters
            .bytes_in_domain(&admission.arbiter, &admission.resource)?;
        Ok(())
    }

    /// One independent unpadded token sequence. Every supplied token is a
    /// real attended position. The direct result retains its own credit.
    pub fn embed_tokens(&self, tokens: &[u32], cancel: Option<&AtomicBool>) -> Result<Embedding> {
        self.validate_tokens(tokens, self.config.max_sequence, self.config.hidden)?;
        check_cancel(cancel)?;
        let output = reserve(
            &self.admission,
            AllocationClass::Output,
            self.config.hidden as u64 * 4,
        )?;
        let mut values = vec![0.; self.config.hidden];
        self.forward_tokens(tokens, &mut values, cancel)?;
        Ok(Embedding {
            values,
            tokens: tokens.len(),
            source: self.source(),
            _output: output,
        })
    }

    /// Caller owns admitted output backing; model retains only private
    /// tokenizer and mathematical scratch. Runtime context applies before
    /// forward, including special tokens.
    pub fn embed_text_into(
        &self,
        text: &str,
        max_tokens: usize,
        output: &mut [f32],
        cancel: Option<&AtomicBool>,
    ) -> Result<usize> {
        check_cancel(cancel)?;
        let (tokens, _charge) = self.tokenizer.encode(text, &self.config, &self.admission)?;
        self.embed_tokens_into(&tokens, max_tokens, output, cancel)
    }

    pub fn embed_tokens_into(
        &self,
        tokens: &[u32],
        max_tokens: usize,
        output: &mut [f32],
        cancel: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.validate_tokens(tokens, max_tokens, output.len())?;
        self.forward_tokens(tokens, output, cancel)
    }

    fn validate_tokens(&self, tokens: &[u32], max_tokens: usize, output_len: usize) -> Result<()> {
        let c = &self.config;
        if tokens.is_empty()
            || tokens.len() > c.max_sequence.min(max_tokens)
            || output_len != c.hidden
            || tokens.iter().any(|&id| id as usize >= c.vocabulary)
        {
            return Err(Error::InvalidInput(
                "BERT token sequence extent/vocabulary".into(),
            ));
        }
        Ok(())
    }

    fn forward_tokens(
        &self,
        tokens: &[u32],
        output: &mut [f32],
        cancel: Option<&AtomicBool>,
    ) -> Result<usize> {
        let c = &self.config;
        check_cancel(cancel)?;
        // At most input+Q/K/V+attention+projection overlap, or input+
        // intermediate+output, plus one-row shared GEMV and reduction scratch.
        let per_token = (c.hidden as u64)
            .checked_mul(8)
            .and_then(|n| n.checked_add(2 * c.intermediate as u64 + 2))
            .ok_or_else(overflow)?;
        let scratch = per_token
            .checked_mul(tokens.len() as u64)
            .and_then(|n| n.checked_add(8 * (c.hidden + c.intermediate) as u64))
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or_else(overflow)?;
        let _scratch = reserve(&self.admission, AllocationClass::Scratch, scratch)?;
        let mut hidden = Vec::with_capacity(tokens.len() * c.hidden);
        let token_type = self.lookup(&self.types, 0)?;
        for (position, &id) in tokens.iter().enumerate() {
            check_cancel(cancel)?;
            let mut row = self.lookup(&self.words, id)?;
            let p = self.lookup(&self.positions, position as u32)?;
            for ((v, p), t) in row.iter_mut().zip(p).zip(&token_type) {
                *v = (*v + *t) + p;
            }
            hidden.extend(row);
        }
        drop(token_type);
        self.norm.rows(&mut hidden, c.epsilon)?;
        for layer in &self.layers {
            check_cancel(cancel)?;
            hidden = layer.forward(&self.source, c, hidden, cancel)?;
        }
        let values = mean_l2_normalize(&hidden, c.hidden, 1e-12)?;
        check_cancel(cancel)?;
        output.copy_from_slice(&values);
        Ok(tokens.len())
    }
}

#[cfg(test)]
mod tests;
