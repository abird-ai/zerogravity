//! Shared source-verified Qwen3.5 dense text model and CPU value adapter.
//! The typed forward schedule defines the same math for CPU and resident GPU
//! execution. Physical backends specialize values and kernels; the common
//! session authority remains the sole logical root publisher.

#[cfg(feature = "cpu")]
use crate::accelerator::{
    AttentionQkCompleted, AttentionQkExecution, AttentionQkRegion, AttentionSoftmaxCompleted,
    AttentionSoftmaxExecution, AttentionSoftmaxRegion, AttentionValueRegion, BinaryOperation,
    BinaryRegion, GdnCoefficientCompleted, GdnCoefficientExecution, GdnCoefficientRegion,
    GdnCoefficients, GdnConvExecution, GdnConvRegion, GdnRecurrentExecution, GdnRecurrentRegion,
    GroupedRmsNormRegion, MatVecKernel, RegionExecution, RmsNormRegion, RopeKvCompleted,
    RopeKvExecution, RopeKvRegion, SiluRegion,
};
#[cfg(feature = "cpu")]
use crate::math::{ggml_source_exp, ggml_source_silu as silu};
use crate::model::{Family, LayerKind, ModelGeometry};
#[cfg(feature = "cpu")]
use crate::session_root::{RootAttempt, RootBinding, SessionAuthority};
use zerogravity_artifacts::SourceWeights;
pub use zerogravity_artifacts::{artifact, model};
pub use zerogravity_core::{Error, Result, accelerator, session_root};
#[cfg(feature = "cpu")]
pub use zerogravity_cpu::{math, weights};
pub mod model_catalog;
#[cfg(feature = "text")]
pub mod tokenize;
#[cfg(feature = "cpu")]
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

#[cfg(feature = "diagnostic-cpu-checkpoint")]
mod checkpoint;
#[cfg(feature = "diagnostic-cpu-checkpoint")]
pub use checkpoint::QwenCheckpoint;
mod forward;
pub mod operator_slice;
pub mod original_binding;
pub use forward::run_layer_sequence;
#[cfg(feature = "cpu")]
mod host_math;

/// Opt-in trained-model numerical observation at one token/region boundary.
/// All values are unrounded f32 CPU outputs, without lossy samples. This is
/// reference evidence to compare to an independent source implementation;
/// producing the observations alone does NOT establish such external parity.
pub use zerogravity_core::accelerator::program::PlanObservation as LayerObservation;

#[cfg(feature = "cpu")]
fn observe(
    trace: &mut Option<&mut Vec<LayerObservation>>,
    position: usize,
    layer: usize,
    stage: &'static str,
    values: &[f32],
) {
    if let Some(records) = trace.as_deref_mut() {
        records.push(LayerObservation {
            position,
            layer,
            stage,
            values: values.to_vec(),
        });
    }
}

#[derive(Clone, Debug)]
#[cfg(any(feature = "cpu", feature = "diagnostic-cpu-checkpoint"))]
enum LayerState {
    Delta {
        // Each conv channel stores 3 oldest->newest prior inputs.
        conv: Arc<Vec<f32>>,
        // 16 x 128 x 128 (values row-major; key index contiguous).
        // Reference root snapshots share memory until first mutation.
        state: Arc<Vec<f32>>,
    },
    Full {
        // Append-only immutable token slabs; cloning a prepared root copies
        // small Arc handles, NOT historical KV matrix bytes.
        keys: Vec<Arc<[f32]>>,
        values: Vec<Arc<[f32]>>,
    },
}

#[derive(Clone)]
#[cfg(feature = "cpu")]
pub struct QwenState {
    layers: Vec<LayerState>,
    position: usize,
    binding: RootBinding,
    attempt: Option<RootAttempt>,
}

#[cfg(feature = "cpu")]
impl QwenState {
    pub fn position(&self) -> usize {
        self.position
    }

    /// Explicit diagnostic copy of every numerical state domain. Conv and
    /// recurrence retain native flattened order. A KV domain contains all key
    /// token slabs followed by all value token slabs, each in token-major order.
    /// These values do not alter portable checkpoint encoding or fingerprints.
    pub fn domain_values(&self) -> Vec<(crate::accelerator::program::StateSlot, Vec<f32>)> {
        state_domain_values(&self.layers)
    }

    /// Deterministic bit digest for replay diagnostics, not a secure hash.
    pub fn fingerprint(&self) -> u64 {
        state_fingerprint(self.position, &self.layers)
    }
    pub fn approximate_bytes(&self) -> usize {
        state_bytes(&self.layers)
    }
}

#[cfg(any(feature = "cpu", feature = "diagnostic-cpu-checkpoint"))]
fn state_domain_values(
    layers: &[LayerState],
) -> Vec<(crate::accelerator::program::StateSlot, Vec<f32>)> {
    use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
    let mut result = Vec::new();
    for (layer, state) in layers.iter().enumerate() {
        match state {
            LayerState::Delta { conv, state } => {
                result.push((StateSlot::Conv(ConvDomain(layer)), conv.as_ref().clone()));
                result.push((
                    StateSlot::Recurrent(RecurrentDomain(layer)),
                    state.as_ref().clone(),
                ));
            }
            LayerState::Full { keys, values } => result.push((
                StateSlot::Kv(KvDomain(layer)),
                keys.iter()
                    .chain(values)
                    .flat_map(|row| row.iter().copied())
                    .collect(),
            )),
        }
    }
    result
}

#[cfg(any(feature = "cpu", feature = "diagnostic-cpu-checkpoint"))]
fn state_fingerprint(position: usize, layers: &[LayerState]) -> u64 {
    fn mix(h: &mut u64, v: u64) {
        *h = (*h ^ v).wrapping_mul(0x100000001b3);
    }
    let mut h = 0xcbf29ce484222325_u64;
    mix(&mut h, position as u64);
    for layer in layers {
        match layer {
            LayerState::Delta { conv, state } => {
                mix(&mut h, 0xdede);
                for &x in conv.iter().chain(state.iter()) {
                    mix(&mut h, u64::from(x.to_bits()));
                }
            }
            LayerState::Full { keys, values } => {
                mix(&mut h, 0xfafa);
                for row in keys.iter().chain(values) {
                    for &x in row.iter() {
                        mix(&mut h, u64::from(x.to_bits()));
                    }
                }
            }
        }
    }
    h
}

#[cfg(any(feature = "cpu", feature = "diagnostic-cpu-checkpoint"))]
fn state_bytes(layers: &[LayerState]) -> usize {
    layers
        .iter()
        .map(|layer| match layer {
            LayerState::Delta { conv, state } => (conv.len() + state.len()) * 4,
            LayerState::Full { keys, values } => {
                keys.iter().chain(values).map(|v| v.len() * 4).sum()
            }
        })
        .sum()
}

/// One immutable, original-checkpoint-backed model allocation shared by
/// arbitrarily many independent sessions. It owns the original GGUF bytes,
/// parsed tensor directory and small source layer constants ONCE. No task
/// cursor, KV cache, mutable device buffer or session epoch lives here.
/// An explicit original HF metadata binding retains the same source owner but
/// refuses sessions/plans until its numerical operators qualify.
///
/// A physical GPU cache keys its immutable weights to this source identity,
/// so sessions of one loaded model may safely reuse an actual source buffer.
pub struct LoadedQwenModel {
    weights: SourceWeights,
    #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
    original: Option<original_binding::OriginalQwenBinding>,
    geometry: ModelGeometry,
    layers: Vec<CachedLayer>,
    final_norm: Vec<f32>,
    state_specs: Vec<StateSpec>,
    #[cfg(feature = "cpu")]
    initial_layers: std::sync::OnceLock<Vec<LayerState>>,
    eps: f32,
}

/// Immutable numerical state geometry; contains no session allocations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StateSpec {
    Delta { conv: usize, state: usize },
    Full { kv_width: usize },
}

/// Shared CPU/resident portable state admission bound, unchanged from v1.
pub const MAX_QWEN_STATE_BYTES: usize = 1024 * 1024 * 1024;

/// Only the mutable per-request/session frontier and an optional device region
/// specializer. Cloning an Arc<LoadedQwenModel> never clones GGUF bytes.
#[cfg(feature = "cpu")]
pub struct QwenDenseCpu {
    model: Arc<LoadedQwenModel>,
    kernel: Option<Arc<dyn MatVecKernel>>,
    state: QwenState,
    authority: SessionAuthority,
}

#[cfg(feature = "cpu")]
impl Deref for QwenDenseCpu {
    type Target = LoadedQwenModel;

    fn deref(&self) -> &Self::Target {
        &self.model
    }
}

#[derive(Debug)]
struct CachedLayer {
    attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ssm_norm: Option<Vec<f32>>,
    a: Option<Vec<f32>>,
    dt: Option<Vec<f32>>,
    conv_kernel: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    ffn_dim: usize,
}

/// Qwen's source GGUF CPU RMSNorm execution semantics: the input square
/// is rounded to f32 BEFORE ggml's f64 accumulation, then mean and inverse
/// square root are each rounded to f32; each subsequent multiply is f32.
/// This differs at quantization decision boundaries from the mathematically
/// higher-precision `math::rms_norm` diagnostic oracle.
#[cfg(feature = "cpu")]
pub fn source_rms_normalize(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    if x.len() != weight.len() || x.is_empty() || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Malformed(
            "invalid Qwen RMSNorm source geometry".into(),
        ));
    }
    if !x.iter().all(|v| v.is_finite()) {
        return Err(Error::Malformed("nonfinite Qwen RMSNorm input".into()));
    }
    let sum = x.iter().map(|&v| f64::from(v * v)).sum::<f64>();
    let mean = (sum / x.len() as f64) as f32;
    let inv = (mean + eps).sqrt().recip();
    let out = x
        .iter()
        .zip(weight)
        .map(|(&v, &w)| (v * inv) * w)
        .collect::<Vec<_>>();
    if !out.iter().all(|v| v.is_finite()) {
        return Err(Error::Malformed("nonfinite Qwen RMSNorm output".into()));
    }
    Ok(out)
}
/// Exact reference for independently rounded RMS-normalized contiguous heads.
/// This is a repeated source operator, not a new norm reduction that mixes
/// heads: source numerical order and every head's own FP32 boundaries remain.
#[cfg(feature = "cpu")]
pub fn source_grouped_rms_normalize(
    input: &[f32],
    weights: &[f32],
    groups: usize,
    eps: f32,
) -> Result<Vec<f32>> {
    let width = weights.len();
    if width == 0 || groups == 0 || width.checked_mul(groups) != Some(input.len()) {
        return Err(Error::Malformed(
            "invalid grouped source RMS normalization dimensions".into(),
        ));
    }
    let mut output = Vec::with_capacity(input.len());
    for group in input.chunks_exact(width) {
        output.extend(source_rms_normalize(group, weights, eps)?);
    }
    Ok(output)
}

/// Original Qwen GGUF CPU causal GDN convolution and three-frame history
/// update. Exactly the former scalar left-to-right f32 expression in ONE
/// authoritative implementation. The caller owns a tentative StateRoot;
/// no committed session state or GPU Handle is modified by this function.
#[cfg(feature = "cpu")]
pub fn source_gdn_conv_step(
    activation: &[f32],
    weights: &[f32],
    history: &mut [f32],
    raw_output: &mut [f32],
) -> Result<()> {
    let channels = activation.len();
    if channels == 0
        || channels.checked_mul(3) != Some(history.len())
        || channels.checked_mul(4) != Some(weights.len())
        || raw_output.len() != channels
        || !activation
            .iter()
            .chain(weights)
            .chain(history.iter())
            .all(|v| v.is_finite())
    {
        return Err(Error::Malformed(
            "GDN causal convolution source geometry/nonfinite input".into(),
        ));
    }
    for channel in 0..channels {
        let c = &mut history[channel * 3..channel * 3 + 3];
        let w = &weights[channel * 4..channel * 4 + 4];
        // Preserve original scalar FP32 multiplication and sum order.
        let raw = c[0] * w[0] + c[1] * w[1] + c[2] * w[2] + activation[channel] * w[3];
        raw_output[channel] = raw;
        c[0] = c[1];
        c[1] = c[2];
        c[2] = activation[channel];
    }
    if !raw_output.iter().all(|v| v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite GDN convolution source output".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "cpu")]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
#[cfg(feature = "cpu")]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}
/// Original source libm/FP32 schedule; shared as the physical region oracle.
#[cfg(feature = "cpu")]
pub fn source_gdn_coefficients(
    region: GdnCoefficientRegion<'_>,
) -> Result<GdnCoefficientCompleted> {
    region.validate()?;
    let mut result = GdnCoefficientCompleted {
        coefficients: Vec::with_capacity(region.alpha.len()),
        softplus: Vec::with_capacity(region.alpha.len()),
        gate_raw: Vec::with_capacity(region.alpha.len()),
    };
    for h in 0..region.alpha.len() {
        let biased_alpha = region.alpha[h] + region.dt[h];
        let sp = softplus(biased_alpha);
        let gate_raw = sp * region.a[h];
        result.softplus.push(sp);
        result.gate_raw.push(gate_raw);
        result.coefficients.push(GdnCoefficients {
            gain: gate_raw.exp(),
            beta: sigmoid(region.beta[h]),
            scale: region.scale,
        });
    }
    result.validate(region.alpha.len())?;
    Ok(result)
}

fn layer_tensor(i: usize, name: &str) -> String {
    format!("blk.{i}.{name}")
}

impl LoadedQwenModel {
    /// Read and validate one immutable GGUF/config pair. The resulting model
    /// contains NO session-specific state and can be shared across users.
    pub fn open(gguf: &Path, config_json: &Path) -> Result<Arc<Self>> {
        let g = ModelGeometry::from_file(config_json)?;
        Self::validate_direct_geometry(&g)?;
        Self::from_checked_source(SourceWeights::open(gguf)?, g)
    }

    /// Prepare the existing family from one immutable original source owner.
    /// Source construction/identity remains the artifact owner's authority;
    /// this performs exactly the same geometry/tensor checks as path opening.
    pub fn from_source(weights: SourceWeights, config_json: &[u8]) -> Result<Arc<Self>> {
        let geometry = ModelGeometry::from_json(config_json)?;
        Self::validate_direct_geometry(&geometry)?;
        Self::from_checked_source(weights, geometry)
    }

    fn from_checked_source(weights: SourceWeights, g: ModelGeometry) -> Result<Arc<Self>> {
        if weights.architecture() != "qwen35" {
            return Err(Error::Unsupported(
                "Qwen runner needs qwen35 GGUF tensors".into(),
            ));
        }
        let hidden = g.hidden_size as usize;
        if weights.tensor("token_embd.weight")?.shape != [g.hidden_size, g.vocab_size] {
            return Err(Error::Malformed(
                "model embedding differs from official config".into(),
            ));
        }
        let final_norm = weights.vector("output_norm.weight")?;
        if final_norm.len() != hidden {
            return Err(Error::Malformed("output norm width".into()));
        }
        let mut layers = Vec::with_capacity(g.layers.len());
        let mut state_specs = Vec::with_capacity(g.layers.len());
        for (i, kind) in g.layers.iter().enumerate() {
            let a = weights.vector(&layer_tensor(i, "attn_norm.weight"))?;
            let b = weights.vector(&layer_tensor(i, "post_attention_norm.weight"))?;
            if a.len() != hidden || b.len() != hidden {
                return Err(Error::Malformed(format!(
                    "layer {i} residual norms mismatch"
                )));
            }
            let ff_dim = &weights.tensor(&layer_tensor(i, "ffn_gate.weight"))?.shape;
            if ff_dim.len() != 2
                || ff_dim[0] != g.hidden_size
                || weights.tensor(&layer_tensor(i, "ffn_down.weight"))?.shape
                    != [ff_dim[1], g.hidden_size]
                || weights.tensor(&layer_tensor(i, "ffn_up.weight"))?.shape != *ff_dim
            {
                return Err(Error::Malformed(format!(
                    "layer {i} FFN dimension mismatch"
                )));
            }
            let ffn_dim = ff_dim[1] as usize;
            if *kind == LayerKind::LinearAttention {
                let kernel = weights.vector(&layer_tensor(i, "ssm_conv1d.weight"));
                // Conv is rank 2; read F32 matrix bytes by a dedicated accessor.
                let kernel = match kernel {
                    Ok(_) => {
                        return Err(Error::Malformed("unexpected vector-shaped ssm conv".into()));
                    }
                    Err(_) => weights.matrix_f32(&layer_tensor(i, "ssm_conv1d.weight"))?,
                };
                let ssm_norm = weights.vector(&layer_tensor(i, "ssm_norm.weight"))?;
                let a_vec = weights.vector(&layer_tensor(i, "ssm_a"))?;
                let dt = weights.vector(&layer_tensor(i, "ssm_dt.bias"))?;
                let heads = g.recurrent_value_heads.unwrap_or(0) as usize;
                let keys = g.recurrent_key_heads.unwrap_or(0) as usize;
                let head_dim = ssm_norm.len();
                if heads == 0
                    || heads != keys
                    || head_dim == 0
                    || a_vec.len() != heads
                    || dt.len() != heads
                    || weights.tensor(&layer_tensor(i, "attn_qkv.weight"))?.shape
                        != [g.hidden_size, (heads * head_dim * 3) as u64]
                    || weights.tensor(&layer_tensor(i, "attn_gate.weight"))?.shape
                        != [g.hidden_size, (heads * head_dim) as u64]
                    || kernel.len() != 4 * heads * head_dim * 3
                {
                    return Err(Error::Malformed(format!(
                        "layer {i} GDN tensor geometry mismatch"
                    )));
                }
                // Preload only small scalar/norm convolution tensors. Large quantized
                // projections stay in the common source-backed CpuWeights.
                layers.push(CachedLayer {
                    attn_norm: a,
                    ffn_norm: b,
                    ssm_norm: Some(ssm_norm),
                    a: Some(a_vec),
                    dt: Some(dt),
                    conv_kernel: Some(kernel),
                    q_norm: None,
                    k_norm: None,
                    ffn_dim,
                });
                state_specs.push(StateSpec::Delta {
                    conv: 3 * heads * head_dim * 3,
                    state: heads * head_dim * head_dim,
                });
            } else {
                let qn = weights.vector(&layer_tensor(i, "attn_q_norm.weight"))?;
                let kn = weights.vector(&layer_tensor(i, "attn_k_norm.weight"))?;
                if qn.len() != g.head_dim as usize || kn.len() != qn.len() {
                    return Err(Error::Malformed(format!(
                        "layer {i} full-attention norms mismatch"
                    )));
                }
                layers.push(CachedLayer {
                    attn_norm: a,
                    ffn_norm: b,
                    ssm_norm: None,
                    a: None,
                    dt: None,
                    conv_kernel: None,
                    q_norm: Some(qn),
                    k_norm: Some(kn),
                    ffn_dim,
                });
                state_specs.push(StateSpec::Full {
                    kv_width: (g.num_key_value_heads * g.head_dim) as usize,
                });
            }
        }
        Ok(Arc::new(Self {
            weights,
            #[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
            original: None,
            geometry: g,
            layers,
            final_norm,
            state_specs,
            #[cfg(feature = "cpu")]
            initial_layers: std::sync::OnceLock::new(),
            eps: 1e-6,
        }))
    }

    /// Start an independent session without disk I/O, tensor decoding or
    /// copying immutable weights. Recurrent zero-state Arcs are copy-on-write
    /// and full-attention KV slabs begin empty for every session.
    #[cfg(feature = "cpu")]
    pub fn new_session(self: &Arc<Self>) -> Result<QwenDenseCpu> {
        QwenDenseCpu::from_shared(self.clone())
    }

    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.weights.source_identity()
    }
    /// Borrow the source-bound embedding descriptor without decoding a row.
    pub fn embedding_source(&self, token: u32) -> Result<crate::accelerator::EmbeddingRegion<'_>> {
        if self.numerical_profile() != original_binding::QwenNumericalProfile::GgufSourceV1 {
            let tensor = self.matrix_descriptor(original_binding::QwenMatrixRole::Embedding)?;
            let region = crate::accelerator::EmbeddingRegion {
                source_identity: self.source_identity(),
                tensor_name: "model.language_model.embed_tokens.weight",
                kind: tensor.kind,
                packed_weights: self.tensor_bytes(&tensor.name)?,
                token,
                width: tensor.columns,
                vocabulary: tensor.rows,
            };
            region.validate()?;
            return Ok(region);
        }
        self.weights.embedding_region(token, "token_embd.weight")
    }

    /// Immutable source tensor metadata for a prepared region planner. No
    /// GGUF weights, mutable session state or physical resources are copied.
    pub fn tensor_info(&self, name: &str) -> Result<&crate::artifact::TensorInfo> {
        self.weights.tensor(name)
    }

    /// Borrow immutable original packed tensor bytes for a source-bound
    /// prepared program. The borrow cannot outlive this loaded model owner.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        self.weights.tensor_bytes(name)
    }

    pub fn geometry(&self) -> &ModelGeometry {
        &self.geometry
    }

    #[cfg(any(feature = "cpu", feature = "diagnostic-cpu-checkpoint"))]
    fn validate_numerical_state(&self, position: usize, layers: &[LayerState]) -> Result<()> {
        self.require_execution_profile()?;
        if position > self.geometry.max_position_embeddings as usize
            || layers.len() != self.layers.len()
        {
            return Err(Error::State("decoder state exceeds maximum context".into()));
        }
        let kv = (self.geometry.num_key_value_heads * self.geometry.head_dim) as usize;
        for (i, (layer, kind)) in layers.iter().zip(&self.geometry.layers).enumerate() {
            match (layer, kind) {
                (LayerState::Delta { conv, state }, LayerKind::LinearAttention) => {
                    let heads = self.geometry.recurrent_value_heads.unwrap_or(0) as usize;
                    let dim = self.layers[i].ssm_norm.as_ref().unwrap().len();
                    if conv.len() != heads * dim * 9
                        || state.len() != heads * dim * dim
                        || !conv.iter().chain(state.iter()).all(|x| x.is_finite())
                    {
                        return Err(Error::State(format!("invalid recurrent layer domain {i}")));
                    }
                }
                (LayerState::Full { keys, values }, LayerKind::FullAttention) => {
                    if keys.len() != position
                        || values.len() != position
                        || !keys
                            .iter()
                            .chain(values)
                            .all(|v| v.len() == kv && v.iter().all(|x| x.is_finite()))
                    {
                        return Err(Error::State(format!("invalid attention KV domain {i}")));
                    }
                }
                _ => return Err(Error::State(format!("invalid state layer type {i}"))),
            }
        }
        Ok(())
    }

    /// Cheap direct-runner geometry admission, before allocating original
    /// source bytes. This is a subset eligibility check on parsed geometry,
    /// not full config validation. from_source always reparses bounded config
    /// bytes before tensor/source checks; mutable geometry is never trusted.
    pub fn validate_direct_geometry(g: &ModelGeometry) -> Result<()> {
        if g.family != Family::QwenHybrid
            || g.layers.len() != 24
            || !matches!(g.hidden_size, 1024 | 2048)
        {
            return Err(Error::Unsupported(
                "initial direct runner supports Qwen3.5 0.8B/2B dense text only".into(),
            ));
        }
        Ok(())
    }

    pub fn source_bytes(&self) -> usize {
        self.weights.source_len()
    }
}

#[cfg(feature = "cpu")]
impl QwenDenseCpu {
    /// Convenience constructor; prefer LoadedQwenModel::open + new_session
    /// when many requests share the same model.
    pub fn open(gguf: &Path, config_json: &Path) -> Result<Self> {
        LoadedQwenModel::open(gguf, config_json)?.new_session()
    }

    /// Binds only the mutable session frontier. No original GGUF or cached
    /// source-layer bytes are copied when this method is called.
    pub fn from_shared(model: Arc<LoadedQwenModel>) -> Result<Self> {
        model.require_execution_profile()?;
        let authority = SessionAuthority::new(model.source_identity())?;
        let binding = authority.binding();
        let layers = model.initial_layers().to_vec();
        Ok(Self {
            model,
            kernel: None,
            state: QwenState {
                layers,
                position: 0,
                binding,
                attempt: None,
            },
            authority,
        })
    }

    pub fn loaded_model(&self) -> &Arc<LoadedQwenModel> {
        &self.model
    }

    pub fn with_kernel(mut self, kernel: Arc<dyn MatVecKernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Opt-in externally owned accelerator for closed tensor regions.
    /// The same Rust StateRoot remains authoritative; unsupported regions
    /// fall back to the source-verified CPU math, while device errors abort
    /// preparation. No GPU availability is implied by this constructor.
    pub fn open_with_kernel(
        gguf: &Path,
        config_json: &Path,
        kernel: Arc<dyn MatVecKernel>,
    ) -> Result<Self> {
        Ok(Self::open(gguf, config_json)?.with_kernel(kernel))
    }

    fn project(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
        zerogravity_cpu::weights::matvec_with_source(
            &self.weights,
            name,
            input,
            self.kernel.as_deref(),
        )
    }

    /// Source-typed optional norm region. Unsupported is the ONLY legal CPU
    /// fallback; a nonfinite/short real GPU readback aborts the pending root.
    fn normalize_region(&self, name: &str, x: &[f32], weights: &[f32]) -> Result<Vec<f32>> {
        if let Some(kernel) = &self.kernel {
            let region = RmsNormRegion {
                source_identity: self.weights.source_identity(),
                tensor_name: name,
                input: x,
                weights,
                eps: self.eps,
            };
            match kernel.try_rms_norm(region)? {
                RegionExecution::Completed(values) => {
                    if values.len() != x.len() || !values.iter().all(|v| v.is_finite()) {
                        return Err(Error::State(format!(
                            "physical normalization output malformed for {name}"
                        )));
                    }
                    return Ok(values);
                }
                RegionExecution::Unsupported => {}
            }
        }
        source_rms_normalize(x, weights, self.eps)
    }

    /// Source-identical per-head RMS normalization. One optional physical
    /// completed region covers all independent rows sharing immutable gamma;
    /// Unsupported alone selects ordinary native Rust CPU row semantics.
    fn normalize_grouped_region(
        &self,
        name: &str,
        input: &[f32],
        weights: &[f32],
        groups: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let width = weights.len();
        if groups == 0 || width == 0 || width.checked_mul(groups) != Some(input.len()) {
            return Err(Error::Malformed(
                "invalid grouped Qwen RMS head geometry".into(),
            ));
        }
        if let Some(kernel) = &self.kernel {
            let region = GroupedRmsNormRegion {
                source_identity: self.weights.source_identity(),
                tensor_name: name,
                input,
                weights,
                group_width: width,
                groups,
                eps,
            };
            match kernel.try_grouped_rms_norm(region)? {
                RegionExecution::Completed(values) => {
                    if values.len() != input.len() || !values.iter().all(|v| v.is_finite()) {
                        return Err(Error::State(format!(
                            "physical grouped RMS normalization incomplete for {name}"
                        )));
                    }
                    return Ok(values);
                }
                RegionExecution::Unsupported => {}
            }
        }
        source_grouped_rms_normalize(input, weights, groups, eps)
    }

    fn embedding_region(&self, token: u32) -> Result<Vec<f32>> {
        let r = self.weights.embedding_region(token, "token_embd.weight")?;
        if let Some(kernel) = &self.kernel {
            match kernel.try_embedding(r)? {
                RegionExecution::Completed(values) => {
                    r.validate_completed(&values)?;
                    return Ok(values);
                }
                RegionExecution::Unsupported => {}
            }
        }
        crate::weights::source_reference_embedding(r)
    }

    fn binary_region(&self, r: BinaryRegion<'_>) -> Result<Vec<f32>> {
        r.validate()?;
        if let Some(kernel) = &self.kernel {
            match kernel.try_binary(r)? {
                RegionExecution::Completed(values) => {
                    r.validate_completed(&values)?;
                    return Ok(values);
                }
                RegionExecution::Unsupported => {}
            }
        }
        source_binary(r)
    }

    fn attention_value_region(&self, r: AttentionValueRegion<'_>) -> Result<Vec<f32>> {
        r.validate()?;
        match self
            .kernel
            .as_ref()
            .map(|k| k.try_attention_value(r))
            .transpose()?
        {
            Some(RegionExecution::Completed(values)) => {
                r.validate_completed(&values)?;
                Ok(values)
            }
            _ => source_attention_value(r),
        }
    }

    fn attention_softmax_region(
        &self,
        r: AttentionSoftmaxRegion<'_>,
    ) -> Result<AttentionSoftmaxCompleted> {
        r.validate()?;
        let outcome = match &self.kernel {
            Some(k) => k.try_attention_softmax(r)?,
            None => AttentionSoftmaxExecution::Unsupported,
        };
        match outcome {
            AttentionSoftmaxExecution::Unsupported => source_attention_softmax(r),
            AttentionSoftmaxExecution::Completed(ready) => {
                ready.validate(r)?;
                Ok(ready)
            }
        }
    }

    fn attention_qk_region(&self, region: AttentionQkRegion<'_>) -> Result<AttentionQkCompleted> {
        region.validate()?;
        let ready = match &self.kernel {
            Some(k) => k.try_attention_qk(region)?,
            None => AttentionQkExecution::Unsupported,
        };
        match ready {
            AttentionQkExecution::Unsupported => source_attention_qk(region),
            AttentionQkExecution::Completed(values) => {
                values.validate(region)?;
                Ok(values)
            }
        }
    }

    fn rope_kv_region(&self, region: RopeKvRegion<'_>) -> Result<RopeKvCompleted> {
        region.validate()?;
        let ready = match &self.kernel {
            Some(k) => k.try_rope_kv(region)?,
            None => RopeKvExecution::Unsupported,
        };
        match ready {
            RopeKvExecution::Unsupported => source_rope_kv(region),
            RopeKvExecution::Completed(values) => {
                values.validate(region)?;
                Ok(values)
            }
        }
    }

    fn gdn_coefficient_region(
        &self,
        region: GdnCoefficientRegion<'_>,
    ) -> Result<GdnCoefficientCompleted> {
        region.validate()?;
        let ready = match &self.kernel {
            Some(kernel) => kernel.try_gdn_coefficients(region)?,
            None => GdnCoefficientExecution::Unsupported,
        };
        match ready {
            GdnCoefficientExecution::Completed(values) => {
                values.validate(region.alpha.len())?;
                if values
                    .coefficients
                    .iter()
                    .any(|c| c.scale.to_bits() != region.scale.to_bits())
                {
                    return Err(Error::State(
                        "GDN completion changed immutable head scale".into(),
                    ));
                }
                Ok(values)
            }
            GdnCoefficientExecution::Unsupported => source_gdn_coefficients(region),
        }
    }

    fn silu_region(
        &self,
        name: &str,
        input: &[f32],
        multiplier: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let region = SiluRegion {
            source_identity: self.weights.source_identity(),
            operation: name,
            input,
            multiplier,
        };
        region.validate()?;
        if let Some(kernel) = &self.kernel {
            match kernel.try_silu(region)? {
                RegionExecution::Completed(values) => {
                    if values.len() != input.len() || !values.iter().all(|v| v.is_finite()) {
                        return Err(Error::State(format!(
                            "physical SiLU output malformed for {name}"
                        )));
                    }
                    return Ok(values);
                }
                RegionExecution::Unsupported => {}
            }
        }
        Ok(input
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let y = silu(x);
                multiplier.map_or(y, |m| m[i] * y)
            })
            .collect())
    }

    fn gdn_conv_region(
        &self,
        name: &str,
        activation: &[f32],
        weights: &[f32],
        history: &mut [f32],
        raw: &mut [f32],
    ) -> Result<()> {
        if let Some(kernel) = &self.kernel {
            let region = GdnConvRegion {
                source_identity: self.weights.source_identity(),
                tensor_name: name,
                activation,
                weights,
                history,
            };
            match kernel.try_gdn_conv(region)? {
                GdnConvExecution::Completed(ready) => {
                    if ready.raw.len() != activation.len()
                        || ready.history.len() != history.len()
                        || !ready
                            .raw
                            .iter()
                            .chain(ready.history.iter())
                            .all(|v| v.is_finite())
                    {
                        return Err(Error::State(format!(
                            "physical GDN convolution returned incomplete/nonfinite source state for {name}"
                        )));
                    }
                    // Only after real physical completion may the PRIVATE
                    // candidate QwenState receive the proposed new history.
                    raw.copy_from_slice(&ready.raw);
                    history.copy_from_slice(&ready.history);
                    return Ok(());
                }
                GdnConvExecution::Unsupported => {}
            }
        }
        source_gdn_conv_step(activation, weights, history, raw)
    }

    fn gdn_recurrent_region(
        &self,
        region: GdnRecurrentRegion<'_>,
    ) -> Result<GdnRecurrentExecution> {
        region.validate()?;
        let result = match &self.kernel {
            Some(kernel) => kernel.try_gdn_recurrent(region)?,
            None => GdnRecurrentExecution::Unsupported,
        };
        if let GdnRecurrentExecution::Completed(ref ready) = result
            && (ready.output.len() != region.q.len()
                || ready.state.len() != region.state.len()
                || !ready
                    .output
                    .iter()
                    .chain(&ready.state)
                    .all(|v| v.is_finite()))
        {
            return Err(Error::State(format!(
                "incomplete/nonfinite GDN recurrent proposal for {}",
                region.operation
            )));
        }
        Ok(result)
    }

    pub fn position(&self) -> usize {
        self.state.position
    }
    pub fn root_id(&self) -> u64 {
        self.authority.binding().root()
    }
    pub fn snapshot(&self) -> QwenState {
        self.state.clone()
    }
    pub fn state_fingerprint(&self) -> u64 {
        self.state.fingerprint()
    }
    pub fn state_bytes(&self) -> usize {
        self.state.approximate_bytes()
    }

    /// Restoring an earlier published snapshot invalidates all prepared drafts
    /// from the abandoned timeline. A foreign-session snapshot is rejected.
    pub fn restore(&mut self, snapshot: &QwenState) -> Result<()> {
        self.validate_state(snapshot)?;
        // The original direct API also accepts a fully validated same-session
        // prepared state. Preserve that as an explicit timeline replacement,
        // rather than treating its unpublished position as a committed binding.
        let attempt = if snapshot.attempt.is_some() {
            self.authority.begin_validated_replacement(
                u64::try_from(snapshot.position)
                    .map_err(|_| Error::Limit("snapshot position overflow".into()))?,
            )?
        } else {
            self.authority.begin_restore(snapshot.binding)?
        };
        self.publish_state(snapshot.clone(), attempt)
    }

    pub fn reset(&mut self) -> Result<()> {
        let attempt = self.authority.begin_reset()?;
        let mut candidate = self.state.clone();
        for layer in &mut candidate.layers {
            match layer {
                LayerState::Delta { conv, state } => {
                    // Fresh domains preserve all immutable prior snapshots.
                    *conv = Arc::new(vec![0.0; conv.len()]);
                    *state = Arc::new(vec![0.0; state.len()]);
                }
                LayerState::Full { keys, values } => {
                    keys.clear();
                    values.clear();
                }
            }
        }
        candidate.position = 0;
        candidate.attempt = Some(attempt.clone());
        self.publish_state(candidate, attempt)
    }

    /// Complete payload validation precedes the exclusive publication guard.
    /// No fallible operation or await occurs after replacing the visible state.
    fn publish_state(&mut self, mut state: QwenState, attempt: RootAttempt) -> Result<()> {
        let current = self.authority.binding();
        if state.binding.source() != current.source()
            || state.binding.session() != current.session()
        {
            return Err(Error::State("foreign state publication payload".into()));
        }
        // Replacement payloads may originate in a historical snapshot. Bind
        // validation to the current attempt without changing numerical values.
        state.binding = attempt.base();
        state.attempt = Some(attempt.clone());
        self.validate_state(&state)?;
        let transition = self.authority.prepare_publish(attempt)?;
        state.attempt = None;
        self.state = state;
        self.state.binding = transition.commit();
        Ok(())
    }

    /// Shape/domain coverage and nonfinite-state guard at the CPU publication
    /// boundary. GPU event readiness will be a separate mandatory condition.
    fn validate_state(&self, draft: &QwenState) -> Result<()> {
        let binding = self.authority.binding();
        let expected_frontier = draft
            .attempt
            .as_ref()
            .map_or(draft.binding.frontier(), |attempt| attempt.frontier());
        if draft.binding.source() != binding.source()
            || draft.binding.session() != binding.session()
            || draft
                .attempt
                .as_ref()
                .is_some_and(|attempt| attempt.base() != draft.binding)
            || u64::try_from(draft.position).ok() != Some(expected_frontier)
            || draft.layers.len() != self.layers.len()
        {
            return Err(Error::State(
                "foreign session state or incomplete domains".into(),
            ));
        }
        self.model
            .validate_numerical_state(draft.position, &draft.layers)
    }

    /// A candidate has no visible effect until commit succeeds.
    pub fn next_logits(&self, token: u32) -> Result<(Vec<f32>, QwenState)> {
        self.check_admission(1)?;
        let attempt = self.authority.begin_advance(1)?;
        let mut draft = self.state.clone();
        draft.attempt = Some(attempt);
        let logits = self.forward_token(token, &mut draft, true, None)?;
        self.validate_state(&draft)?;
        if !logits.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed("decoder produced nonfinite logits".into()));
        }
        Ok((logits, draft))
    }

    const MAX_CPU_SESSION_BYTES: usize = MAX_QWEN_STATE_BYTES;

    /// Small-model CPU logical admission, before any candidate memory is
    /// allocated. This does not replace the physical ResourceArbiter.
    fn check_admission(&self, tokens: usize) -> Result<()> {
        if tokens == 0 {
            return Err(Error::Malformed(
                "prefill must contain at least one token".into(),
            ));
        }
        let count = self
            .state
            .position
            .checked_add(tokens)
            .ok_or_else(|| Error::Limit("decoder position overflow".into()))?;
        if count > self.geometry.max_position_embeddings as usize {
            return Err(Error::Limit("model context capacity exhausted".into()));
        }
        let kv_width = (self.geometry.num_key_value_heads * self.geometry.head_dim) as usize;
        let full_layers = self
            .geometry
            .layers
            .iter()
            .filter(|x| **x == LayerKind::FullAttention)
            .count();
        let growth = tokens
            .checked_mul(kv_width)
            .and_then(|n| n.checked_mul(full_layers))
            .and_then(|n| n.checked_mul(8))
            .ok_or_else(|| Error::Limit("KV byte estimate overflow".into()))?;
        let expected = self
            .state
            .approximate_bytes()
            .checked_add(growth)
            .ok_or_else(|| Error::Limit("session state byte estimate overflow".into()))?;
        if expected > Self::MAX_CPU_SESSION_BYTES {
            return Err(Error::Limit(
                "CPU session state admission exceeds 1 GiB budget".into(),
            ));
        }
        Ok(())
    }

    /// Publish an arbitrary positive, bounded prefill chunk *atomically*:
    /// one immutable logical state root is installed only if every token and
    /// layer succeeds. Prefill doesn't calculate/output logits except for
    /// the final position. This preserves trained model math while avoiding
    /// unnecessary 248K-vocabulary projections at intermediate positions.
    pub fn prefill_transaction(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.prefill_transaction_inner(tokens, false, true)?.0)
    }

    /// Source-equivalent transactional prefill for intermediate chunks.
    /// Deliberately does not compute final logits: no token from this chunk
    /// is externally selected, and the next chunk continues from the same
    /// immutable published StateRoot without a needless vocabulary GEMV.
    /// This is a normal runtime primitive, not a diagnostic-only shortcut:
    /// state publication, source tensor math and rollback checks are identical.
    pub fn prefill_chunk_without_logits(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill_transaction_inner(tokens, false, false)
            .map(|_| ())
    }

    /// Same atomic prefill transaction, but exposes complete non-rounded f32
    /// activations at the LAST input token's layer boundaries for an optional
    /// independently captured mathematical reference. No other position is
    /// copied, and the prior visible StateRoot is unchanged on computation
    /// failure (as in the non-tracing prefill route).
    pub fn prefill_transaction_with_layer_trace(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Vec<f32>, Vec<LayerObservation>)> {
        self.prefill_transaction_inner(tokens, true, true)
    }

    fn prefill_transaction_inner(
        &mut self,
        tokens: &[u32],
        last_token_trace: bool,
        compute_logits: bool,
    ) -> Result<(Vec<f32>, Vec<LayerObservation>)> {
        self.check_admission(tokens.len())?;
        let attempt = self.authority.begin_advance(
            u64::try_from(tokens.len())
                .map_err(|_| Error::Limit("prefill length overflow".into()))?,
        )?;
        // Fail quickly for invalid token IDs before allocating/copying recurrent
        // and KV state. The embedding still performs authoritative lookup.
        if tokens
            .iter()
            .any(|&id| u64::from(id) >= self.geometry.vocab_size)
        {
            return Err(Error::Malformed(
                "prefill contains out-of-vocabulary token ID".into(),
            ));
        }
        let mut prepared = self.state.clone();
        prepared.attempt = Some(attempt.clone());
        let mut logits = Vec::new();
        let mut observations = Vec::new();
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i + 1 == tokens.len();
            let trace = if is_last && last_token_trace {
                Some(&mut observations)
            } else {
                None
            };
            logits = self.forward_token(token, &mut prepared, is_last && compute_logits, trace)?;
        }
        if compute_logits && (logits.is_empty() || !logits.iter().all(|x| x.is_finite())) {
            return Err(Error::Malformed(
                "prefill did not produce valid final logits".into(),
            ));
        }
        self.publish_state(prepared, attempt)?;
        Ok((logits, observations))
    }

    /// Diagnostic-only private candidate. Does not publish model state.
    /// Each observation records the complete layer vector, not a fingerprint,
    /// allowing an independently captured oracle to compare precise stages.
    /// Bounds: one token, supported Qwen dense text shape, at most ~2 MiB
    /// trace data per small model. Never enable inside regular generation.
    pub fn next_logits_with_layer_trace(
        &self,
        token: u32,
    ) -> Result<(Vec<f32>, QwenState, Vec<LayerObservation>)> {
        self.check_admission(1)?;
        let attempt = self.authority.begin_advance(1)?;
        let mut draft = self.state.clone();
        draft.attempt = Some(attempt);
        let mut records = Vec::with_capacity(self.layers.len() * 7);
        let logits = self.forward_token(token, &mut draft, true, Some(&mut records))?;
        self.validate_state(&draft)?;
        if !logits.iter().all(|v| v.is_finite()) {
            return Err(Error::Malformed("nonfinite traced model logits".into()));
        }
        Ok((logits, draft, records))
    }

    pub fn commit(&mut self, mut state: QwenState) -> Result<()> {
        let attempt = state
            .attempt
            .take()
            .ok_or_else(|| Error::State("CPU decoder state is not a prepared proposal".into()))?;
        if state.binding != self.authority.binding()
            || attempt.base() != state.binding
            || self.state.position.checked_add(1) != Some(state.position)
        {
            return Err(Error::State(
                "stale/foreign CPU decoder root proposal".into(),
            ));
        }
        self.publish_state(state, attempt)
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<f32>> {
        let (logits, state) = self.next_logits(token)?;
        self.commit(state)?;
        Ok(logits)
    }

    fn forward_token(
        &self,
        token: u32,
        state: &mut QwenState,
        need_logits: bool,
        trace: Option<&mut Vec<LayerObservation>>,
    ) -> Result<Vec<f32>> {
        let position = state.position;
        let valid_tokens = position
            .checked_add(1)
            .ok_or_else(|| Error::Limit("decoder position overflow".into()))?;
        let inputs = crate::accelerator::program::ForwardInputs {
            token: token as usize,
            position,
            valid_tokens,
        };
        let mut math = host_math::HostMath::new(self, state, trace);
        match forward::forward_token(self.model.as_ref(), &mut math, inputs, need_logits)? {
            Some(value) => value.into_vec(),
            None => Ok(Vec::new()),
        }
    }
}

#[cfg(feature = "cpu")]
pub fn source_binary(r: BinaryRegion<'_>) -> Result<Vec<f32>> {
    r.validate()?;
    let values: Vec<f32> = r
        .left
        .iter()
        .zip(r.right)
        .map(|(&a, &b)| match r.kind {
            BinaryOperation::Add => a + b,
            BinaryOperation::SigmoidGate => a * sigmoid(b),
        })
        .collect();
    r.validate_completed(&values)?;
    Ok(values)
}

/// Source four-by-eight FMA value matmul. Canonical padding executes +0*+0
/// rather than skipping the FMA: this matters when an accumulated lane is -0.
#[cfg(feature = "cpu")]
pub fn source_attention_value(r: AttentionValueRegion<'_>) -> Result<Vec<f32>> {
    r.validate()?;
    let mut output = vec![0.0; r.query_heads * r.head_width];
    for h in 0..r.query_heads {
        let kv = h / (r.query_heads / r.kv_heads);
        for j in 0..r.head_width {
            let mut sums = [[0f32; 8]; 4];
            for p in 0..r.padded_tokens {
                let (value, probability) = if p < r.values.len() {
                    (
                        r.values[p][kv * r.head_width + j],
                        r.probabilities[h * r.values.len() + p],
                    )
                } else {
                    (0.0, 0.0)
                };
                let lane = p % 32;
                let acc = &mut sums[lane / 8][lane % 8];
                *acc = value.mul_add(probability, *acc);
            }
            let reduced = std::array::from_fn::<f32, 8, _>(|lane| {
                (sums[0][lane] + sums[2][lane]) + (sums[1][lane] + sums[3][lane])
            });
            let tmp = std::array::from_fn::<f32, 4, _>(|lane| reduced[lane] + reduced[lane + 4]);
            output[h * r.head_width + j] = (tmp[0] + tmp[1]) + (tmp[2] + tmp[3]);
        }
    }
    r.validate_completed(&output)?;
    Ok(output)
}

/// Source-equivalent GGML Haswell softmax of a causal attention score row.
/// CPU KV cache/mask has a padded key dimension. Invalid masked positions
/// contribute 0, but the *eight-wide* FP32 exponential lane reduction and
/// FP64 accumulation of each eight-lane sum are observable numerically.
/// This specialization avoids allocating the padding itself.
#[cfg(test)]
#[cfg(feature = "cpu")]
fn softmax_source_padded(scores: &[f32]) -> Result<Vec<f32>> {
    if scores.is_empty() || !scores.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed(
            "invalid full attention softmax scores".into(),
        ));
    }
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut result = Vec::with_capacity(scores.len());
    let mut sum = 0f64;
    for block in scores.chunks(8) {
        let mut exponentials = [0.0f32; 8];
        for (i, &value) in block.iter().enumerate() {
            exponentials[i] = ggml_source_exp(value - max);
        }
        let reduced = [
            exponentials[0] + exponentials[4],
            exponentials[1] + exponentials[5],
            exponentials[2] + exponentials[6],
            exponentials[3] + exponentials[7],
        ];
        let partial = (reduced[0] + reduced[2]) + (reduced[1] + reduced[3]);
        sum += f64::from(partial);
        result.extend_from_slice(&exponentials[..block.len()]);
    }
    if sum == 0.0 || !sum.is_finite() {
        return Err(Error::Malformed(
            "attention softmax normalization invalid".into(),
        ));
    }
    let inv = (1.0f64 / sum) as f32;
    for value in &mut result {
        *value *= inv;
    }
    Ok(result)
}

/// Original source padded softmax, returning complete auditable intermediates.
#[cfg(feature = "cpu")]
pub fn source_attention_softmax(
    r: AttentionSoftmaxRegion<'_>,
) -> Result<AttentionSoftmaxCompleted> {
    r.validate()?;
    let mut ready = AttentionSoftmaxCompleted {
        exponentials: Vec::with_capacity(r.scores.len()),
        probabilities: Vec::with_capacity(r.scores.len()),
        sums: Vec::with_capacity(r.heads),
        reciprocals: Vec::with_capacity(r.heads),
        inverse: Vec::with_capacity(r.heads),
    };
    for row in r.scores.chunks_exact(r.width) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        let begin = ready.exponentials.len();
        for block in row.chunks(8) {
            let mut e = [0.0f32; 8];
            for (i, &x) in block.iter().enumerate() {
                e[i] = ggml_source_exp(x - max);
            }
            let partial = ((e[0] + e[4]) + (e[2] + e[6])) + ((e[1] + e[5]) + (e[3] + e[7]));
            sum += f64::from(partial);
            ready.exponentials.extend_from_slice(&e[..block.len()]);
        }
        let reciprocal = 1.0f64 / sum;
        let inv = reciprocal as f32;
        ready.sums.push(sum);
        ready.reciprocals.push(reciprocal);
        ready.inverse.push(inv);
        ready
            .probabilities
            .extend(ready.exponentials[begin..].iter().map(|e| e * inv));
    }
    ready.validate(r)?;
    Ok(ready)
}

/// GGML Haswell x86 AVX2/FMA numerical dot with the same four-accumulator
/// block schedule as the source CPU implementation. The scalar fallback
/// below 32 elements uses the source's f64 accumulation of f32 products.
/// This is a portable implementation of a qualified *numerical schedule*.
#[cfg(feature = "cpu")]
fn source_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let np = a.len() & !31usize;
    let mut sums = [[0.0f32; 8]; 4];
    for index in (0..np).step_by(32) {
        for (i, lanes) in sums.iter_mut().enumerate() {
            for (lane, sum) in lanes.iter_mut().enumerate() {
                let at = index + i * 8 + lane;
                *sum = a[at].mul_add(b[at], *sum);
            }
        }
    }
    let mut combined = [0.0f32; 8];
    for i in 0..8 {
        combined[i] = (sums[0][i] + sums[2][i]) + (sums[1][i] + sums[3][i]);
    }
    let halves = [
        combined[0] + combined[4],
        combined[1] + combined[5],
        combined[2] + combined[6],
        combined[3] + combined[7],
    ];
    let reduced = (halves[0] + halves[1]) + (halves[2] + halves[3]);
    let tail = a[np..]
        .iter()
        .zip(&b[np..])
        .map(|(&x, &y)| f64::from(x * y))
        .sum::<f64>();
    (f64::from(reduced) + tail) as f32
}

/// Original source attention dot tree and separately rounded source scale.
#[cfg(feature = "cpu")]
pub fn source_attention_qk(region: AttentionQkRegion<'_>) -> Result<AttentionQkCompleted> {
    region.validate()?;
    let mut dots = Vec::with_capacity(region.query_heads * region.keys.len());
    let group = region.query_heads / region.kv_heads;
    for (h, q) in region.query.chunks_exact(region.head_width).enumerate() {
        let start = (h / group) * region.head_width;
        for key in region.keys {
            dots.push(source_dot_f32(q, &key[start..start + region.head_width]));
        }
    }
    let scores = dots.iter().map(|dot| dot * region.scale).collect();
    let ready = AttentionQkCompleted { dots, scores };
    ready.validate(region)?;
    Ok(ready)
}

/// Original single-token source schedule; state is value-major with a
/// contiguous key vector per column. The caller owns this private proposal.
#[cfg(feature = "cpu")]
pub fn source_gdn_recurrent_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    coeff: GdnCoefficients,
    output: &mut [f32],
) -> Result<()> {
    let GdnCoefficients { gain, beta, scale } = coeff;
    let d = q.len();
    if d == 0
        || k.len() != d
        || v.len() != d
        || output.len() != d
        || d.checked_mul(d) != Some(state.len())
        || !gain.is_finite()
        || !beta.is_finite()
        || !scale.is_finite()
        || !q
            .iter()
            .chain(k)
            .chain(v)
            .chain(state.iter())
            .all(|x| x.is_finite())
    {
        return Err(Error::Malformed(
            "invalid scalar GDN recurrence geometry".into(),
        ));
    }
    // Fused, single-token GGML Gated DeltaNet AR kernel:
    // decay current physical state, obtain residual via an F32 SIMD/FMA
    // dot, update state with a vector FMA, read with another SIMD/FMA dot,
    // then apply the 1/sqrt(head-width) scale *after* the dot. This is
    // equivalent algebraically to the unfused graph but the floating-point
    // operation order is part of the chosen source-execution specialization.
    for j in 0..d {
        let col = &mut state[j * d..(j + 1) * d];
        for v in col.iter_mut() {
            *v *= gain;
        }
        let prev = source_dot_f32(col, k);
        let delta = (v[j] - prev) * beta;
        for (s, &ki) in col.iter_mut().zip(k) {
            *s = ki.mul_add(delta, *s);
        }
        output[j] = source_dot_f32(col, q) * scale;
    }
    if !state.iter().all(|x| x.is_finite()) || !output.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite Gated DeltaNet recurrence".into(),
        ));
    }
    Ok(())
}

/// Qwen Gated DeltaNet Q/K L2 normalization uses two GGML source ops:
/// RMSNorm(x, eps / width), then Scale(1 / sqrt(width)). Their independently
/// rounded f32 results differ from the single-operation algebraic shortcut
/// x / sqrt(sum(x^2) + eps), especially as recurrent state accumulates.
/// Retain the source's f32 squared operands and f64 reduction accumulator.
#[cfg(test)]
#[cfg(feature = "cpu")]
fn l2_norm(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let sum = x.iter().map(|v| f64::from(v * v)).sum::<f64>();
    let mean = (sum / x.len() as f64) as f32;
    let inv = 1.0 / (mean + eps / n).sqrt();
    let scale = 1.0 / n.sqrt();
    for v in x.iter_mut() {
        *v = (*v * inv) * scale;
    }
}

/// Source-specialized CPU RoPE for the Qwen3.5 GGML IMROPE kernel.
/// Its frequency uses an iteratively multiplied f32 scale and its sin/cos
/// cache is f32. Higher-precision closed-form phase (rope_text) below is
/// retained as a mathematical diagnostic oracle, not the source-execution
/// representation. The two are mathematically equivalent but not bitwise
/// equivalent; tiny RoPE differences can cross a later Q8 rounding boundary.
#[cfg(test)]
#[cfg(feature = "cpu")]
fn rope_text_source_f32(v: &mut [f32], pos: usize, rotate: usize, theta: f32) -> Result<()> {
    if rotate > v.len() || !rotate.is_multiple_of(2) || pos > i32::MAX as usize {
        return Err(Error::Malformed(
            "source RoPE position or width invalid".into(),
        ));
    }
    let half = rotate / 2;
    let theta_scale = theta.powf(-2.0 / rotate as f32);
    let mut angle = pos as f32;
    for i in 0..half {
        let (sin, cos) = angle.sin_cos();
        let a = v[i];
        let b = v[i + half];
        // GGML source Haswell contracts these two multiply-add expressions;
        // without the FMA rounding point the RoPE output differs by 1 ulp,
        // which can later change Q8 quantization decisions.
        v[i] = a.mul_add(cos, -(b * sin));
        v[i + half] = a.mul_add(sin, b * cos);
        angle *= theta_scale;
    }
    Ok(())
}

/// Source f32 iterative phases and FMA schedule, returning only a new KV slab.
#[cfg(feature = "cpu")]
pub fn source_rope_kv(region: RopeKvRegion<'_>) -> Result<RopeKvCompleted> {
    region.validate()?;
    let mut ready = RopeKvCompleted {
        query: region.query.to_vec(),
        key: region.key.to_vec(),
        value: region.value.to_vec(),
    };
    let half = region.rotate / 2;
    for head in ready
        .query
        .chunks_exact_mut(region.head_width)
        .chain(ready.key.chunks_exact_mut(region.head_width))
    {
        let mut angle = region.position as f32;
        for i in 0..half {
            let (sin, cos) = angle.sin_cos();
            let a = head[i];
            let b = head[i + half];
            head[i] = a.mul_add(cos, -(b * sin));
            head[i + half] = a.mul_add(sin, b * cos);
            angle *= region.theta_scale;
        }
    }
    ready.validate(region)?;
    Ok(ready)
}

/// Plain-text multimodal-RoPE's three axes carry identical token positions.
/// The text-only image/video-free mode rotates only the declared partial head.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "independent high precision RoPE oracle used in tests"
    )
)]
#[cfg(feature = "cpu")]
fn rope_text(v: &mut [f32], pos: usize, rotate: usize, theta: f32) -> Result<()> {
    if rotate > v.len() || !rotate.is_multiple_of(2) {
        return Err(Error::Malformed("partial RoPE width exceeds head".into()));
    }
    // Qwen3.5 text uses GGML IMROPE: rotary pairs are half-rotary-dim
    // apart, NOT consecutive lanes (the latter changes trained position math).
    // Text modality supplies the same position for temporal/height/width axes.
    let half = rotate / 2;
    for i in 0..half {
        // Use f64 for phase calculation to preserve angular precision near
        // large absolute positions. Arithmetic is rounded once to the f32
        // model-state width; no model parameter or RoPE frequency is changed.
        let angle = (pos as f64) / f64::from(theta).powf(2.0 * i as f64 / rotate as f64);
        let (sin, cos) = angle.sin_cos();
        let a = f64::from(v[i]);
        let b = f64::from(v[i + half]);
        v[i] = (a * cos - b * sin) as f32;
        v[i + half] = (a * sin + b * cos) as f32;
    }
    Ok(())
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    #[test]
    fn diagnostic_state_domains_preserve_native_layout_and_zero_signs() {
        use super::*;
        use crate::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
        let authority = SessionAuthority::new(
            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
        )
        .unwrap();
        let state = QwenState {
            layers: vec![
                LayerState::Delta {
                    conv: Arc::new(vec![-0.0, 1.0]),
                    state: Arc::new(vec![2.0, -3.0]),
                },
                LayerState::Full {
                    keys: vec![Arc::from(vec![4.0, 5.0]), Arc::from(vec![6.0, 7.0])],
                    values: vec![Arc::from(vec![8.0, 9.0]), Arc::from(vec![10.0, -0.0])],
                },
            ],
            position: 2,
            binding: authority.binding(),
            attempt: None,
        };
        let before = state.fingerprint();
        let domains = state.domain_values();
        assert_eq!(domains.len(), 3);
        assert_eq!(domains[0].0, StateSlot::Conv(ConvDomain(0)));
        assert_eq!(domains[0].1[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(
            domains[1],
            (StateSlot::Recurrent(RecurrentDomain(0)), vec![2.0, -3.0])
        );
        assert_eq!(domains[2].0, StateSlot::Kv(KvDomain(1)));
        assert_eq!(domains[2].1, vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, -0.0]);
        assert_eq!(domains[2].1[7].to_bits(), (-0.0f32).to_bits());
        assert_eq!(state.fingerprint(), before);
    }

    #[test]
    #[ignore = "requires both original local Qwen models; common root authority integration"]
    fn common_root_authority_preserves_snapshot_draft_and_rewind_semantics() {
        use super::*;
        for name in ["qwen35-0.8b", "qwen35-2b"] {
            let dir =
                Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/models")).join(name);
            let model =
                LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
            let mut session = model.new_session().unwrap();
            let initial = session.snapshot();
            session.prefill_chunk_without_logits(&[11, 353]).unwrap();
            let future = session.snapshot();
            let (_, draft) = session.next_logits(17).unwrap();
            let draft_hash = draft.fingerprint();
            let stable = session.authority.binding();
            assert!(session.prefill_transaction(&[42, u32::MAX]).is_err());
            assert_eq!(session.authority.binding(), stable);
            assert_eq!(session.state_fingerprint(), future.fingerprint());
            session.restore(&initial).unwrap();
            assert_eq!(session.root_id(), 2);
            assert_eq!(session.authority.binding().session_epoch(), 1);
            session.restore(&future).unwrap();
            assert_eq!(session.position(), 2);
            assert_eq!(session.root_id(), 3);
            assert_eq!(session.authority.binding().session_epoch(), 2);
            assert!(session.commit(draft.clone()).is_err());
            // Existing direct CPU restore accepts a valid prepared state even
            // after its old proposal became stale. It starts a new timeline.
            session.restore(&draft).unwrap();
            assert_eq!(session.state_fingerprint(), draft_hash);
            assert_eq!(session.position(), 3);
            assert_eq!(session.root_id(), 4);
            assert_eq!(session.authority.binding().session_epoch(), 3);
            assert_eq!(session.snapshot().domain_values().len(), 42);
            assert!(session.state.attempt.is_none());
            assert_eq!(session.state.binding, session.authority.binding());
            let sibling = model.new_session().unwrap();
            let before = session.authority.binding();
            assert!(session.restore(&sibling.snapshot()).is_err());
            assert_eq!(session.authority.binding(), before);
            session.reset().unwrap();
            assert_eq!(session.root_id(), 5);
            assert_eq!(session.authority.binding().session_epoch(), 4);
            assert_eq!(session.state_fingerprint(), initial.fingerprint());
            assert_eq!(session.authority.binding().frontier(), 0);
            assert_eq!(session.state.binding, session.authority.binding());
        }
    }

    #[test]
    fn binary_regions_preserve_zero_signs_and_validate_proposals() {
        use crate::accelerator::{BinaryOperation, BinaryRegion, SourceIdentity};
        for (a, b, bits) in [
            (-0.0, -0.0, 0x80000000),
            (0.0, -0.0, 0),
            (f32::from_bits(1), -f32::from_bits(1), 0),
        ] {
            let left = [a];
            let right = [b];
            let r = BinaryRegion {
                source_identity: SourceIdentity::unbound(),
                operation: "add",
                kind: BinaryOperation::Add,
                left: &left,
                right: &right,
            };
            assert_eq!(super::source_binary(r).unwrap()[0].to_bits(), bits);
            assert!(r.validate_completed(&[]).is_err());
            assert!(r.validate_completed(&[f32::NAN]).is_err());
        }
        let left = [-f32::from_bits(1), -0.0, 1.0];
        let gate = [-128.0, 0.0, 128.0];
        let r = BinaryRegion {
            source_identity: SourceIdentity::unbound(),
            operation: "gate",
            kind: BinaryOperation::SigmoidGate,
            left: &left,
            right: &gate,
        };
        let out = super::source_binary(r).unwrap();
        assert_eq!(out[0].to_bits(), 0x80000000);
        assert_eq!(out[1].to_bits(), 0x80000000);
        assert_eq!(out[2], 1.0);
        assert!(BinaryRegion { right: &[], ..r }.validate().is_err());
    }

    #[test]
    fn attention_value_padding_preserves_original_zero_sign() {
        use crate::accelerator::{AttentionValueRegion, SourceIdentity};
        use std::sync::Arc;
        for (tokens, expected) in [
            (1usize, 0x80000001u32),
            (31, 0),
            (32, 0),
            (33, 0),
            (255, 0),
            (256, 0x80000000),
            (257, 0),
        ] {
            let values: Vec<Arc<[f32]>> = vec![Arc::from(vec![-f32::from_bits(1)]); tokens];
            let probabilities = vec![1.0 / tokens as f32; tokens];
            let r = AttentionValueRegion {
                source_identity: SourceIdentity::unbound(),
                operation: "fixture",
                probabilities: &probabilities,
                values: &values,
                head_width: 1,
                query_heads: 1,
                kv_heads: 1,
                padded_tokens: tokens.div_ceil(256) * 256,
            };
            let actual = super::source_attention_value(r).unwrap();
            assert_eq!(
                actual[0].to_bits(),
                expected,
                "native padded fixture T={tokens}"
            );
            assert!(r.validate_completed(&[]).is_err());
            assert!(r.validate_completed(&[f32::NAN]).is_err());
            for bad in [
                AttentionValueRegion { values: &[], ..r },
                AttentionValueRegion {
                    padded_tokens: 0,
                    ..r
                },
                AttentionValueRegion {
                    query_heads: usize::MAX,
                    head_width: 2,
                    ..r
                },
                AttentionValueRegion {
                    query_heads: 3,
                    kv_heads: 2,
                    ..r
                },
            ] {
                assert!(bad.validate().is_err());
            }
            assert!(
                AttentionValueRegion {
                    probabilities: &vec![-0.0; tokens],
                    ..r
                }
                .validate()
                .is_err()
            );
        }
    }

    use super::*;
    #[test]
    fn grouped_softmax_matches_original_padded_source_and_rejects_incomplete_normalizers() {
        for width in [1, 7, 8, 9, 31, 32, 33, 255, 256, 257, 4096] {
            let scores: Vec<_> = (0..width * 3)
                .map(|i| match i % 7 {
                    0 => -0.0,
                    1 => -f32::from_bits(1),
                    2 => -f32::MAX,
                    _ => ((i * 37 % 140) as f32) - 104.0,
                })
                .collect();
            let r = AttentionSoftmaxRegion {
                source_identity: crate::accelerator::SourceIdentity::unbound(),
                operation: "softmax",
                scores: &scores,
                heads: 3,
                width,
            };
            let ready = source_attention_softmax(r).unwrap();
            let expected: Vec<_> = scores
                .chunks_exact(width)
                .flat_map(|row| softmax_source_padded(row).unwrap())
                .collect();
            assert!(
                ready
                    .probabilities
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        let r = AttentionSoftmaxRegion {
            source_identity: crate::accelerator::SourceIdentity::unbound(),
            operation: "softmax",
            scores: &[0.0, 0.0],
            heads: 1,
            width: 2,
        };
        for bad in [
            AttentionSoftmaxRegion {
                width: usize::MAX,
                ..r
            },
            AttentionSoftmaxRegion { heads: 0, ..r },
            AttentionSoftmaxRegion {
                scores: &[f32::NAN, 0.0],
                ..r
            },
        ] {
            assert!(bad.validate().is_err());
        }
        let mut ready = source_attention_softmax(r).unwrap();
        ready.inverse[0] = 0.25;
        assert!(ready.validate(r).is_err());
        ready.inverse[0] = 0.5;
        ready.reciprocals[0] = 0.25;
        assert!(ready.validate(r).is_err());
        ready.reciprocals[0] = 0.5;
        ready.probabilities.pop();
        assert!(ready.validate(r).is_err());
    }

    #[test]
    fn grouped_rope_proposal_matches_original_scalar_schedule_and_preserves_tail() {
        for width in [64, 128, 256] {
            for pos in [0, 1, 119, 120, 128, 8192, 16384, 32768] {
                let q: Vec<_> = (0..width * 2)
                    .map(|i| ((i * 17 % 107) as f32 - 53.) / 31.)
                    .collect();
                let k: Vec<_> = q[..width].to_vec();
                let v = vec![-0.0; width];
                let region = RopeKvRegion {
                    source_identity: crate::accelerator::SourceIdentity::unbound(),
                    operation: "rope",
                    query: &q,
                    key: &k,
                    value: &v,
                    head_width: width,
                    query_heads: 2,
                    kv_heads: 1,
                    rotate: 64,
                    position: pos,
                    theta_scale: 10_000_000f32.powf(-2. / 64.),
                };
                let ready = source_rope_kv(region).unwrap();
                let mut expected = q.clone();
                expected.extend(&k);
                for head in expected.chunks_exact_mut(width) {
                    rope_text_source_f32(head, pos, 64, 10_000_000.).unwrap();
                }
                assert!(
                    ready
                        .query
                        .iter()
                        .chain(&ready.key)
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                assert!(
                    ready
                        .value
                        .iter()
                        .all(|x| x.to_bits() == (-0.0f32).to_bits())
                );
            }
        }
    }

    #[test]
    fn gdn_conv_source_updates_three_frame_history_only_after_valid_inputs() {
        let channels = 32usize;
        let qkv = (0..channels)
            .map(|i| ((i * 13) as f32 - 199.0) / 117.0)
            .collect::<Vec<_>>();
        let coefficients = (0..channels * 4)
            .map(|i| ((i * 37 % 71) as f32 - 35.0) / 509.0)
            .collect::<Vec<_>>();
        let mut history = (0..channels * 3)
            .map(|i| ((i * 19 % 83) as f32 - 41.0) / 239.0)
            .collect::<Vec<_>>();
        let original = history.clone();
        let mut raw = vec![0f32; channels];
        source_gdn_conv_step(&qkv, &coefficients, &mut history, &mut raw).unwrap();
        for channel in 0..channels {
            let old = &original[channel * 3..channel * 3 + 3];
            let w = &coefficients[channel * 4..channel * 4 + 4];
            let expected = old[0] * w[0] + old[1] * w[1] + old[2] * w[2] + qkv[channel] * w[3];
            assert_eq!(raw[channel].to_bits(), expected.to_bits());
            assert_eq!(
                &history[channel * 3..channel * 3 + 3],
                &[old[1], old[2], qkv[channel]]
            );
        }
        let good = history.clone();
        let mut broken = vec![1.0f32; channels];
        let mut nan = qkv.clone();
        nan[0] = f32::NAN;
        assert!(source_gdn_conv_step(&nan, &coefficients, &mut history, &mut broken).is_err());
        assert_eq!(good, history, "invalid input mutated private GDN state");
        assert_eq!(broken, vec![1.0f32; channels]);
        assert!(source_gdn_conv_step(&qkv, &coefficients[..4], &mut history, &mut broken).is_err());
        assert!(source_gdn_conv_step(&qkv, &coefficients, &mut history[..7], &mut broken).is_err());
    }

    #[test]
    fn grouped_rms_heads_match_independent_source_bits_and_reject_bad_shapes() {
        let coefficients = (0..128)
            .map(|i| 0.25 + (i as f32 / 341.0))
            .collect::<Vec<_>>();
        let activations = (0..(128 * 17))
            .map(|i| ((i * 37 % 499) as f32 - 245.0) / 193.0)
            .collect::<Vec<_>>();
        let grouped = source_grouped_rms_normalize(&activations, &coefficients, 17, 1e-6).unwrap();
        let original = activations
            .chunks_exact(128)
            .flat_map(|head| source_rms_normalize(head, &coefficients, 1e-6).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(grouped.len(), original.len());
        assert!(
            grouped
                .iter()
                .zip(&original)
                .all(|(g, c)| g.to_bits() == c.to_bits())
        );
        for bad_groups in [0, 1, 16, 18, usize::MAX] {
            assert!(
                source_grouped_rms_normalize(&activations, &coefficients, bad_groups, 1e-6)
                    .is_err()
            );
        }
        assert!(source_grouped_rms_normalize(&[], &[], 1, 1e-6).is_err());
        assert!(source_grouped_rms_normalize(&activations, &coefficients, 17, f32::NAN).is_err());
        let mut contaminated = activations;
        contaminated[127] = f32::INFINITY;
        assert!(source_grouped_rms_normalize(&contaminated, &coefficients, 17, 1e-6).is_err());
    }

    #[test]
    fn grouped_l2_matches_original_two_stage_source_bits() {
        for width in [2, 8, 64, 128, 256] {
            for eps in [1e-6, 1e-4] {
                let input = (0..width * 3)
                    .map(|i| match i % 11 {
                        0 => -0.0,
                        1 => 0.0,
                        _ => ((i * 71 % 201) as f32 - 100.0) / 37.0,
                    })
                    .collect::<Vec<_>>();
                let mut original = input.clone();
                for head in original.chunks_exact_mut(width) {
                    l2_norm(head, eps);
                }
                let scales = vec![1.0 / (width as f32).sqrt(); width];
                let grouped =
                    source_grouped_rms_normalize(&input, &scales, 3, eps / width as f32).unwrap();
                assert!(
                    original
                        .iter()
                        .zip(grouped)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    #[test]
    fn scalar_gdn_and_rope_math() {
        let mut q = [3., 4.];
        l2_norm(&mut q, 1e-6);
        assert!((q[0] - 0.6).abs() < 1e-5);
        assert!((q[1] - 0.8).abs() < 1e-5);
        let mut r = [1., 0., 2., 0.];
        rope_text(&mut r, 0, 4, 10_000_000.).unwrap();
        assert_eq!(r, [1., 0., 2., 0.]);
        assert!((sigmoid(0.) - 0.5).abs() < 1e-6);
        assert!((softplus(0.) - (2f32).ln()).abs() < 1e-6);
    }

    #[test]
    fn qwen35_interleaved_text_rope_pairs_half_head() {
        // The Qwen IMROPE text-mode pair (0, 2) must rotate; using the
        // conventional adjacent pair (0, 1) caused real-model divergences.
        let mut value = [1.0f32, 0.0, 0.0, 0.0];
        rope_text(&mut value, 1, 4, 10_000_000.0).unwrap();
        let (sin, cos) = 1.0f32.sin_cos();
        assert!((value[0] - cos).abs() < 1.0e-6);
        assert!((value[2] - sin).abs() < 1.0e-6);
        assert_eq!(value[1], 0.0);
        assert_eq!(value[3], 0.0);
    }

    #[test]
    fn recurrence_32768_token_independent_f64_oracle_and_chunk_parity() {
        const DIM: usize = 8;
        const TOKENS: usize = 32_768;
        let coeff = GdnCoefficients {
            gain: 0.965,
            beta: 0.125,
            scale: 0.35355338,
        };
        let mut direct = vec![0.0f32; DIM * DIM];
        let mut segmented = vec![0.0f32; DIM * DIM];
        let mut reference = vec![0.0f64; DIM * DIM];
        for t in 0..TOKENS {
            let key = std::array::from_fn::<_, DIM, _>(|i| {
                (((t * 13 + i * 29) % 73) as f32 - 36.0) / 650.0
            });
            let query = std::array::from_fn::<_, DIM, _>(|i| {
                (((t * 17 + i * 5) % 31) as f32 - 15.0) / 19.0
            });
            let value = std::array::from_fn::<_, DIM, _>(|i| {
                (((t * 19 + i * 11) % 43) as f32 - 21.0) / 97.0
            });
            let mut out = [0.0f32; DIM];
            let mut second = [0.0f32; DIM];
            source_gdn_recurrent_step(&mut direct, &query, &key, &value, coeff, &mut out).unwrap();
            // This alternate segmentation models splitting a prefill stream
            // into 127-token chunks while preserving causal recurrence.
            source_gdn_recurrent_step(&mut segmented, &query, &key, &value, coeff, &mut second)
                .unwrap();
            assert_eq!(direct, segmented, "chunk boundary at {t} changed state");
            assert_eq!(out, second);
            if t.is_multiple_of(127) {
                let checkpoint = segmented.clone();
                assert_eq!(checkpoint, direct);
            }
            for j in 0..DIM {
                let prev: f64 = (0..DIM)
                    .map(|i| reference[j * DIM + i] * f64::from(key[i]))
                    .sum();
                let delta =
                    (f64::from(value[j]) - f64::from(coeff.gain) * prev) * f64::from(coeff.beta);
                for i in 0..DIM {
                    reference[j * DIM + i] =
                        f64::from(coeff.gain) * reference[j * DIM + i] + f64::from(key[i]) * delta;
                }
                let expected: f64 = (0..DIM)
                    .map(|i| reference[j * DIM + i] * f64::from(query[i]))
                    .sum::<f64>()
                    * f64::from(coeff.scale);
                assert!(
                    (f64::from(out[j]) - expected).abs() < 1e-5,
                    "token {t} head output drift"
                );
            }
            if t.is_multiple_of(4096) {
                assert!(direct.iter().all(|v| v.is_finite()));
            }
        }
    }

    #[test]
    fn state_snapshot_is_copy_on_write_and_kv_history_is_shared() {
        // 32K real KV token slabs would require a large model and long test;
        // this tests the actual immutable storage mechanism with a small
        // source-independent slice, not full-model long-context correctness.
        let original = LayerState::Full {
            keys: vec![Arc::from(vec![1.0f32; 1024]), Arc::from(vec![2.0f32; 1024])],
            values: vec![Arc::from(vec![3.0f32; 1024]), Arc::from(vec![4.0f32; 1024])],
        };
        let mut proposal = original.clone();
        match (&original, &mut proposal) {
            (
                LayerState::Full {
                    keys: a,
                    values: av,
                },
                LayerState::Full {
                    keys: b,
                    values: bv,
                },
            ) => {
                assert!(Arc::ptr_eq(&a[0], &b[0]));
                assert!(Arc::ptr_eq(&av[1], &bv[1]));
                b.push(Arc::from(vec![5.0; 1024]));
                bv.push(Arc::from(vec![6.0; 1024]));
                assert_eq!(a.len(), 2, "prepared KV append mutated published root");
                assert_eq!(b.len(), 3);
                assert_eq!(a[0][0], 1.0);
            }
            _ => panic!("unexpected KV storage"),
        }

        let original = LayerState::Delta {
            conv: Arc::new(vec![2.0, 3.0]),
            state: Arc::new(vec![4.0, 5.0]),
        };
        let mut proposal = original.clone();
        match (&original, &mut proposal) {
            (
                LayerState::Delta { conv: a, state: ar },
                LayerState::Delta { conv: b, state: br },
            ) => {
                assert!(Arc::ptr_eq(a, b));
                assert!(Arc::ptr_eq(ar, br));
                Arc::make_mut(b)[0] = 9.0;
                Arc::make_mut(br)[0] = 8.0;
                assert_eq!(
                    a[0], 2.0,
                    "prepared recurrence mutation leaked to published root"
                );
                assert_eq!(ar[0], 4.0);
                assert!(!Arc::ptr_eq(a, b));
                assert!(!Arc::ptr_eq(ar, br));
            }
            _ => panic!("unexpected recurrence storage"),
        }
    }

    #[test]
    fn immutable_kv_slab_32768_context_root_clone_is_payload_zero_copy() {
        let slabs = 32_768usize;
        let first: Arc<[f32]> = Arc::from(vec![1.0f32; 64]);
        let mut original = LayerState::Full {
            keys: vec![first.clone(); slabs],
            values: vec![first.clone(); slabs],
        };
        let mut proposal = original.clone();
        match (&mut original, &mut proposal) {
            (
                LayerState::Full {
                    keys: old,
                    values: old_v,
                },
                LayerState::Full {
                    keys: new,
                    values: new_v,
                },
            ) => {
                assert_eq!(old.len(), slabs);
                assert!(Arc::ptr_eq(&old[0], &new[0]));
                assert!(Arc::ptr_eq(&old[slabs - 1], &new[slabs - 1]));
                assert!(Arc::ptr_eq(&old_v[17], &new_v[17]));
                new.push(Arc::from(vec![2.0f32; 64]));
                new_v.push(Arc::from(vec![3.0f32; 64]));
                assert_eq!(old.len(), slabs);
                assert_eq!(new.len(), slabs + 1);
                assert_eq!(old[slabs - 1][0], 1.0);
                assert_eq!(new[slabs][0], 2.0);
                assert_eq!(Arc::strong_count(&first), 4 * slabs + 1);
            }
            _ => panic!("expected long KV regions"),
        }
    }

    #[test]
    fn nonzero_rope_positions_up_to_declared_context_capacity() {
        for pos in [0usize, 1, 1023, 4095, 8191, 16383, 32767, 65535, 262143] {
            let theta = 10_000_000.0f32;
            let mut actual = (0..256)
                .map(|i| ((i * 13 + 3) % 43) as f32 / 19.0 - 1.0)
                .collect::<Vec<_>>();
            let original = actual.clone();
            rope_text(&mut actual, pos, 64, theta).unwrap();
            for i in 0..32 {
                let angle = (pos as f64) / (f64::from(theta)).powf(2.0 * i as f64 / 64.0);
                let (sin, cos) = angle.sin_cos();
                let expected = f64::from(original[i]) * cos - f64::from(original[i + 32]) * sin;
                assert!(
                    (f64::from(actual[i]) - expected).abs() < 2.0e-6,
                    "large-context RoPE position {pos} channel {i}"
                );
            }
            assert_eq!(&actual[64..], &original[64..]);
            assert!(rope_text(&mut actual, 1, 300, theta).is_err());
        }
    }
}

impl zerogravity_core::source::TensorSource for LoadedQwenModel {
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source_identity()
    }
    fn tensor(&self, name: &str) -> Result<zerogravity_core::source::SourceTensorRef<'_>> {
        zerogravity_core::source::TensorSource::tensor(&self.weights, name)
    }
}
impl LoadedQwenModel {
    pub fn state_specs(&self) -> &[StateSpec] {
        &self.state_specs
    }
    /// Zero-template storage actually initialized for CPU sessions.
    pub fn initialized_cpu_state_bytes(&self) -> usize {
        #[cfg(feature = "cpu")]
        {
            self.initial_layers
                .get()
                .map_or(0, |layers| state_bytes(layers))
        }
        #[cfg(not(feature = "cpu"))]
        {
            0
        }
    }
    #[cfg(feature = "cpu")]
    fn initial_layers(&self) -> &[LayerState] {
        self.initial_layers.get_or_init(|| {
            self.state_specs
                .iter()
                .map(|spec| match spec {
                    StateSpec::Delta { conv, state } => LayerState::Delta {
                        conv: Arc::new(vec![0.0; *conv]),
                        state: Arc::new(vec![0.0; *state]),
                    },
                    StateSpec::Full { .. } => LayerState::Full {
                        keys: vec![],
                        values: vec![],
                    },
                })
                .collect()
        })
    }
}
