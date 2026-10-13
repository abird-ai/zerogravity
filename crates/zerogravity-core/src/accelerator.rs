//! Device-agnostic, source-typed matvec specialization boundary.
//!
//! First optional region seam between independently verified Qwen3.5 Rust
//! CPU model execution and future native CUDA/ROCm/Vulkan DeviceBackend.
//! No GPU runtime dependency, asynchronous scheduler, or device residency
//! is claimed here.
//!
//! A kernel accepts immutable original source-GGUF matrix bytes and the
//! current f32 activation. Only physically completed, host-visible results
//! may be published. Unsupported permits deliberate CPU fallback; device
//! errors and invalid output must fail the entire prepared state.

use crate::quant::QuantKind;
use crate::{Error, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_SOURCE_ALLOCATION: AtomicU64 = AtomicU64::new(1);

pub mod program;

/// Process-unique identity for ONE immutable loaded source model allocation.
///
/// A model reopen (even with the same pathname/shape) obtains a new identity.
/// This prevents an accelerator from reusing a GPU weight lease after a
/// source artifact is replaced or its previous CPU allocation was released.
/// ID 0 is exclusively reserved for manually constructed diagnostic regions;
/// persistent device caches must reject that unbound identity.
///
/// This is a local residency identity, not a cryptographic model checksum or
/// a distributed content-addressed artifact identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceIdentity(u64);

impl SourceIdentity {
    pub const fn unbound() -> Self {
        Self(0)
    }

    pub const fn is_bound(self) -> bool {
        self.0 != 0
    }

    /// Issue a fresh identity for a newly owned immutable source allocation.
    /// This never accepts a caller-provided numeric identity or wraps.
    pub fn for_new_allocation() -> Result<Self> {
        NEXT_SOURCE_ALLOCATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map(Self)
            .map_err(|_| Error::Limit("process source residency identity exhausted".into()))
    }

    #[cfg(test)]
    pub(crate) const fn from_issued_id(id: u64) -> Self {
        Self(id)
    }
}

/// A source-validated matrix-vector region. Packed weights preserve GGUF
/// block scales and quant codes, never an idealized F32 dequant expansion.
#[derive(Clone, Copy)]
pub struct MatVecRegion<'a> {
    /// The unique identity of an immutable loaded GGUF source allocation.
    /// A backend must include this in every resident-cache key, together
    /// with tensor name, quant codec and matrix geometry.
    pub source_identity: SourceIdentity,
    pub tensor_name: &'a str,
    pub kind: QuantKind,
    pub columns: usize,
    pub rows: usize,
    pub packed_weights: &'a [u8],
    pub activation: &'a [f32],
}

/// Immutable source-trained RMS normalization: weights are predecoded,
/// source-matched F32 norm coefficients; they are not a GGUF matrix or a
/// mutable session. The exact f32-square -> f64-sum -> f32-RMS scale
/// contract is kept by the existing Rust Qwen reference function.
#[derive(Clone, Copy)]
pub struct RmsNormRegion<'a> {
    pub source_identity: SourceIdentity,
    pub tensor_name: &'a str,
    pub input: &'a [f32],
    pub weights: &'a [f32],
    pub eps: f32,
}

/// Source RMS normalization of independently normalized, contiguous heads.
/// Immutable F32 coefficients (source weights or explicitly source-derived
/// scale constants) are *shared* across heads; each head receives
/// its own independently rounded squared-sum/mean/inverse. One grouped
/// physical readback gates the entire region, not one fake Ready per head.
#[derive(Clone, Copy)]
pub struct GroupedRmsNormRegion<'a> {
    pub source_identity: SourceIdentity,
    pub tensor_name: &'a str,
    pub input: &'a [f32],
    pub weights: &'a [f32],
    pub group_width: usize,
    pub groups: usize,
    pub eps: f32,
}

/// A private causal GDN convolution step over source F32 parameters and
/// COPY-ON-WRITE tentative three-frame history. The provider never mutates
/// the input session StateRoot. A Completed result includes BOTH the raw
/// pre-SiLU convolution and the updated history, only after actual readback.
#[derive(Clone, Copy)]
pub struct GdnConvRegion<'a> {
    pub source_identity: SourceIdentity,
    pub tensor_name: &'a str,
    pub activation: &'a [f32],
    pub weights: &'a [f32],
    pub history: &'a [f32],
}

#[derive(Debug)]
pub struct GdnConvCompleted {
    pub raw: Vec<f32>,
    pub history: Vec<f32>,
}

#[derive(Debug)]
pub enum GdnConvExecution {
    Unsupported,
    Completed(GdnConvCompleted),
}

/// Original single-token GDN coefficients, rounded on the source host.
#[derive(Clone, Copy, Debug)]
pub struct GdnCoefficients {
    pub gain: f32,
    pub beta: f32,
    pub scale: f32,
}

/// Per-head source GDN nonlinear coefficients, before any recurrent state write.
/// `scale` is the immutable source-derived inverse sqrt of head width.
#[derive(Clone, Copy)]
pub struct GdnCoefficientRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub alpha: &'a [f32],
    pub beta: &'a [f32],
    pub dt: &'a [f32],
    pub a: &'a [f32],
    pub scale: f32,
}

impl GdnCoefficientRegion<'_> {
    pub fn validate(self) -> Result<()> {
        let n = self.alpha.len();
        if n == 0
            || self.beta.len() != n
            || self.dt.len() != n
            || self.a.len() != n
            || !self
                .alpha
                .iter()
                .chain(self.beta)
                .chain(self.dt)
                .chain(self.a)
                .all(|x| x.is_finite())
            || !self.scale.is_finite()
            || self.scale <= 0.0
        {
            return Err(Error::Malformed(
                "invalid source GDN coefficient inputs".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GdnCoefficientCompleted {
    pub coefficients: Vec<GdnCoefficients>,
    pub softplus: Vec<f32>,
    pub gate_raw: Vec<f32>,
}

impl GdnCoefficientCompleted {
    pub fn validate(&self, heads: usize) -> Result<()> {
        if self.coefficients.len() != heads
            || self.softplus.len() != heads
            || self.gate_raw.len() != heads
            || !self
                .softplus
                .iter()
                .chain(&self.gate_raw)
                .all(|x| x.is_finite())
            || !self
                .coefficients
                .iter()
                .all(|c| c.gain.is_finite() && c.beta.is_finite() && c.scale.is_finite())
        {
            return Err(Error::State(
                "incomplete/nonfinite source GDN coefficients".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum GdnCoefficientExecution {
    Unsupported,
    Completed(GdnCoefficientCompleted),
}

/// Source text IMRoPE and ONE proposed current-token KV slab. Prefix state
/// stays private to the caller; providers cannot replace or mutate it.
#[derive(Clone, Copy)]
pub struct RopeKvRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub head_width: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub rotate: usize,
    pub position: usize,
    pub theta_scale: f32,
}
impl RopeKvRegion<'_> {
    pub fn validate(self) -> Result<()> {
        if self.head_width == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.query_heads.checked_mul(self.head_width) != Some(self.query.len())
            || self.kv_heads.checked_mul(self.head_width) != Some(self.key.len())
            || self.value.len() != self.key.len()
            || self.rotate == 0
            || self.rotate > self.head_width
            || !self.rotate.is_multiple_of(2)
            || self.position > i32::MAX as usize
            || !self.theta_scale.is_finite()
            || self.theta_scale <= 0.0
            || self.theta_scale > 1.0
            || !self
                .query
                .iter()
                .chain(self.key)
                .chain(self.value)
                .all(|x| x.is_finite())
        {
            return Err(Error::Malformed(
                "invalid source RoPE/KV geometry or FP32 input".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct RopeKvCompleted {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
}
impl RopeKvCompleted {
    pub fn validate(&self, region: RopeKvRegion<'_>) -> Result<()> {
        region.validate()?;
        if self.query.len() != region.query.len()
            || self.key.len() != region.key.len()
            || self.value.len() != region.value.len()
            || !self
                .query
                .iter()
                .chain(&self.key)
                .chain(&self.value)
                .all(|x| x.is_finite())
            || self
                .value
                .iter()
                .zip(region.value)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            || self
                .query
                .chunks_exact(region.head_width)
                .zip(region.query.chunks_exact(region.head_width))
                .chain(
                    self.key
                        .chunks_exact(region.head_width)
                        .zip(region.key.chunks_exact(region.head_width)),
                )
                .any(|(a, b)| {
                    a[region.rotate..]
                        .iter()
                        .zip(&b[region.rotate..])
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                })
        {
            return Err(Error::State(
                "incomplete/nonfinite RoPE proposal or modified source KV/tail".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum RopeKvExecution {
    Unsupported,
    Completed(RopeKvCompleted),
}

/// Causal QK scores over immutable session-owned key slabs. No prefix writer.
#[derive(Clone, Copy)]
pub struct AttentionQkRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub query: &'a [f32],
    pub keys: &'a [Arc<[f32]>],
    pub head_width: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub scale: f32,
}
impl AttentionQkRegion<'_> {
    pub fn validate(self) -> Result<()> {
        if self.head_width == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.query_heads.checked_mul(self.head_width) != Some(self.query.len())
            || self.kv_heads.checked_mul(self.head_width).is_none()
            || self.keys.is_empty()
            || self.query_heads.checked_mul(self.keys.len()).is_none()
            || !self.scale.is_finite()
            || self.scale <= 0.0
            || !self.query.iter().all(|x| x.is_finite())
            || self.keys.iter().any(|k| {
                self.kv_heads.checked_mul(self.head_width) != Some(k.len())
                    || !k.iter().all(|x| x.is_finite())
            })
        {
            return Err(Error::Malformed(
                "invalid source attention QK region".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct AttentionQkCompleted {
    /// Head-major, valid causal positions only. Padding is not model state.
    pub dots: Vec<f32>,
    pub scores: Vec<f32>,
}
impl AttentionQkCompleted {
    pub fn validate(&self, region: AttentionQkRegion<'_>) -> Result<()> {
        region.validate()?;
        let n = region.query_heads * region.keys.len();
        if self.dots.len() != n
            || self.scores.len() != n
            || self.dots.iter().zip(&self.scores).any(|(&dot, &score)| {
                !dot.is_finite()
                    || !score.is_finite()
                    || (dot * region.scale).to_bits() != score.to_bits()
            })
        {
            return Err(Error::State(
                "incomplete/nonfinite QK proposal or wrong source scale".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum AttentionQkExecution {
    Unsupported,
    Completed(AttentionQkCompleted),
}

/// Valid causal scores only; the source's padded eight-lane tail is implicit.
#[derive(Clone, Copy)]
pub struct AttentionSoftmaxRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub scores: &'a [f32],
    pub heads: usize,
    pub width: usize,
}
impl AttentionSoftmaxRegion<'_> {
    pub fn validate(self) -> Result<()> {
        if self.heads == 0
            || self.width == 0
            || self.heads.checked_mul(self.width) != Some(self.scores.len())
            || !self.scores.iter().all(|x| x.is_finite())
        {
            return Err(Error::Malformed(
                "invalid source attention softmax region".into(),
            ));
        }
        Ok(())
    }
}
#[derive(Debug)]
pub struct AttentionSoftmaxCompleted {
    pub exponentials: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub sums: Vec<f64>,
    pub reciprocals: Vec<f64>,
    pub inverse: Vec<f32>,
}
impl AttentionSoftmaxCompleted {
    pub fn validate(&self, region: AttentionSoftmaxRegion<'_>) -> Result<()> {
        region.validate()?;
        let n = region.scores.len();
        if self.exponentials.len() != n
            || self.probabilities.len() != n
            || self.sums.len() != region.heads
            || self.reciprocals.len() != region.heads
            || self.inverse.len() != region.heads
        {
            return Err(Error::State("incomplete source softmax proposal".into()));
        }
        for h in 0..region.heads {
            let sum = self.sums[h];
            let reciprocal = self.reciprocals[h];
            let inv = self.inverse[h];
            if !sum.is_finite()
                || sum < 1.0
                || !reciprocal.is_finite()
                || reciprocal.to_bits() != (1.0f64 / sum).to_bits()
                || inv.to_bits() != (reciprocal as f32).to_bits()
                || inv <= 0.0
            {
                return Err(Error::State(
                    "invalid source softmax two-round normalizer".into(),
                ));
            }
            for i in h * region.width..(h + 1) * region.width {
                let exp = self.exponentials[i];
                let prob = self.probabilities[i];
                if !exp.is_finite()
                    || exp.is_sign_negative()
                    || prob.is_sign_negative()
                    || !(0.0..=1.0).contains(&exp)
                    || !prob.is_finite()
                    || !(0.0..=1.0).contains(&prob)
                    || prob.to_bits() != (exp * inv).to_bits()
                {
                    return Err(Error::State("invalid source softmax probability".into()));
                }
            }
        }
        Ok(())
    }
}
#[derive(Debug)]
pub enum AttentionSoftmaxExecution {
    Unsupported,
    Completed(AttentionSoftmaxCompleted),
}

/// Source value-cache matmul with an explicit canonical +0 padded tail.
/// Only valid immutable slabs are owned by the session. Padding is execution
/// geometry, and must participate in the source FMA schedule (including zeros).
#[derive(Clone, Copy)]
pub struct AttentionValueRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub probabilities: &'a [f32],
    pub values: &'a [Arc<[f32]>],
    pub head_width: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub padded_tokens: usize,
}
impl AttentionValueRegion<'_> {
    pub fn validate(self) -> Result<()> {
        if self.head_width == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || self.query_heads.checked_mul(self.head_width).is_none()
            || self.values.is_empty()
            || self.query_heads.checked_mul(self.values.len()) != Some(self.probabilities.len())
            || self.padded_tokens < self.values.len()
            || !self.padded_tokens.is_multiple_of(32)
            || self
                .probabilities
                .iter()
                .any(|x| !x.is_finite() || x.is_sign_negative() || *x > 1.0)
            || self.values.iter().any(|v| {
                self.kv_heads.checked_mul(self.head_width) != Some(v.len())
                    || v.iter().any(|x| !x.is_finite())
            })
        {
            return Err(Error::Malformed(
                "invalid source attention value region".into(),
            ));
        }
        Ok(())
    }
    pub fn validate_completed(self, values: &[f32]) -> Result<()> {
        self.validate()?;
        if values.len() != self.query_heads * self.head_width
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(Error::State(
                "incomplete/nonfinite source attention value proposal".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOperation {
    Add,
    /// `left * (1 / (1 + expf(-right)))`, with separate FP32 rounding.
    SigmoidGate,
}

#[derive(Clone, Copy)]
pub struct BinaryRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub kind: BinaryOperation,
    pub left: &'a [f32],
    pub right: &'a [f32],
}
impl BinaryRegion<'_> {
    pub fn validate(self) -> Result<()> {
        if self.left.is_empty()
            || self.left.len() != self.right.len()
            || self.left.iter().chain(self.right).any(|x| !x.is_finite())
        {
            return Err(Error::Malformed("invalid source binary region".into()));
        }
        Ok(())
    }
    pub fn validate_completed(self, values: &[f32]) -> Result<()> {
        self.validate()?;
        if values.len() != self.left.len() || values.iter().any(|x| !x.is_finite()) {
            return Err(Error::State(
                "incomplete/nonfinite source binary proposal".into(),
            ));
        }
        Ok(())
    }
}

/// Original immutable compressed embedding tensor; runtime token selects a row.
#[derive(Clone, Copy)]
pub struct EmbeddingRegion<'a> {
    pub source_identity: SourceIdentity,
    pub tensor_name: &'a str,
    pub kind: QuantKind,
    pub packed_weights: &'a [u8],
    pub token: u32,
    pub width: usize,
    pub vocabulary: usize,
}
impl EmbeddingRegion<'_> {
    pub fn validate(self) -> Result<()> {
        let row = usize::try_from(self.kind.byte_len(self.width as u64)?)
            .map_err(|_| Error::Limit("source embedding row bytes overflow".into()))?;
        if self.width == 0
            || self.vocabulary == 0
            || self.tensor_name.is_empty()
            || self.token as usize >= self.vocabulary
            || row.checked_mul(self.vocabulary) != Some(self.packed_weights.len())
        {
            return Err(Error::Malformed(
                "invalid original source embedding region".into(),
            ));
        }
        Ok(())
    }
    pub fn validate_completed(self, values: &[f32]) -> Result<()> {
        self.validate()?;
        if values.len() != self.width || values.iter().any(|x| !x.is_finite()) {
            return Err(Error::State(
                "incomplete/nonfinite source embedding proposal".into(),
            ));
        }
        Ok(())
    }
}

/// One layer's independent heads. State is column-major within each head;
/// providers only read previous private state and return a complete proposal.
/// `operation` identifies the source layer, not a resident parameter tensor.
#[derive(Clone, Copy)]
pub struct GdnRecurrentRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub width: usize,
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub coefficients: &'a [GdnCoefficients],
    pub state: &'a [f32],
}

impl GdnRecurrentRegion<'_> {
    /// Validate the entire read set before allocation, submission or mutation.
    pub fn validate(self) -> Result<()> {
        let heads = self.coefficients.len();
        let values = heads.checked_mul(self.width);
        let state = values.and_then(|n| n.checked_mul(self.width));
        if self.width == 0
            || heads == 0
            || values != Some(self.q.len())
            || values != Some(self.k.len())
            || values != Some(self.v.len())
            || state != Some(self.state.len())
            || !self
                .q
                .iter()
                .chain(self.k)
                .chain(self.v)
                .chain(self.state)
                .all(|x| x.is_finite())
            || !self
                .coefficients
                .iter()
                .all(|c| c.gain.is_finite() && c.beta.is_finite() && c.scale.is_finite())
        {
            return Err(Error::Malformed(
                "invalid source GDN recurrent geometry or FP32 inputs".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct GdnRecurrentCompleted {
    pub output: Vec<f32>,
    pub state: Vec<f32>,
}

#[derive(Debug)]
pub enum GdnRecurrentExecution {
    Unsupported,
    Completed(GdnRecurrentCompleted),
}

/// Original GGML Haswell SiLU polynomial, optionally followed by a separately
/// rounded elementwise multiplication. No native exponential approximation.
#[derive(Clone, Copy)]
pub struct SiluRegion<'a> {
    pub source_identity: SourceIdentity,
    pub operation: &'a str,
    pub input: &'a [f32],
    pub multiplier: Option<&'a [f32]>,
}

impl SiluRegion<'_> {
    pub fn validate(self) -> Result<()> {
        if self.input.is_empty()
            || !self.input.iter().all(|x| x.is_finite())
            || self
                .multiplier
                .is_some_and(|m| m.len() != self.input.len() || !m.iter().all(|x| x.is_finite()))
        {
            return Err(Error::Malformed("invalid source SiLU inputs".into()));
        }
        Ok(())
    }
}

/// Completed implies the device execution and readback are confirmed.
#[derive(Debug)]
pub enum RegionExecution {
    Unsupported,
    Completed(Vec<f32>),
}

/// A region provider cannot publish/mutate the caller's Qwen StateRoot.
/// Existing matvec-only implementations retain source compatibility.
/// Unknown optional regions return Unsupported and use the authoritative
/// Rust CPU operator. A device error never permits implicit CPU fallback.
pub trait MatVecKernel: Send + Sync {
    fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution>;

    fn try_rms_norm(&self, _region: RmsNormRegion<'_>) -> Result<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }

    fn try_grouped_rms_norm(&self, _region: GroupedRmsNormRegion<'_>) -> Result<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }

    fn try_gdn_conv(&self, _region: GdnConvRegion<'_>) -> Result<GdnConvExecution> {
        Ok(GdnConvExecution::Unsupported)
    }
    fn try_gdn_recurrent(&self, _region: GdnRecurrentRegion<'_>) -> Result<GdnRecurrentExecution> {
        Ok(GdnRecurrentExecution::Unsupported)
    }

    fn try_silu(&self, _region: SiluRegion<'_>) -> Result<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }

    fn try_gdn_coefficients(
        &self,
        _region: GdnCoefficientRegion<'_>,
    ) -> Result<GdnCoefficientExecution> {
        Ok(GdnCoefficientExecution::Unsupported)
    }
    fn try_rope_kv(&self, _region: RopeKvRegion<'_>) -> Result<RopeKvExecution> {
        Ok(RopeKvExecution::Unsupported)
    }
    fn try_attention_qk(&self, _region: AttentionQkRegion<'_>) -> Result<AttentionQkExecution> {
        Ok(AttentionQkExecution::Unsupported)
    }
    fn try_attention_softmax(
        &self,
        _region: AttentionSoftmaxRegion<'_>,
    ) -> Result<AttentionSoftmaxExecution> {
        Ok(AttentionSoftmaxExecution::Unsupported)
    }
    fn try_attention_value(&self, _region: AttentionValueRegion<'_>) -> Result<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
    fn try_binary(&self, _region: BinaryRegion<'_>) -> Result<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
    fn try_embedding(&self, _region: EmbeddingRegion<'_>) -> Result<RegionExecution> {
        Ok(RegionExecution::Unsupported)
    }
}

/// Validate completed shape/finite output before using it in state.
pub fn validate_completed(region: MatVecRegion<'_>, values: &[f32]) -> Result<()> {
    if values.len() != region.rows || !values.iter().all(|v| v.is_finite()) {
        return Err(Error::State(format!(
            "accelerator returned invalid or incomplete output for '{}': expected {} finite f32 values, got {}",
            region.tensor_name,
            region.rows,
            values.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_optional_norm_is_explicit_unsupported_not_fake_device_completion() {
        struct MatVecOnly;
        impl MatVecKernel for MatVecOnly {
            fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
                Ok(RegionExecution::Unsupported)
            }
        }
        let weights = [1.0_f32, 2.0];
        let region = RmsNormRegion {
            source_identity: SourceIdentity::unbound(),
            tensor_name: "source.test.norm",
            input: &[0.5, -0.25],
            weights: &weights,
            eps: 1e-6,
        };
        assert!(matches!(
            MatVecOnly.try_rms_norm(region).unwrap(),
            RegionExecution::Unsupported
        ));
        assert!(matches!(
            MatVecOnly
                .try_grouped_rms_norm(GroupedRmsNormRegion {
                    source_identity: SourceIdentity::unbound(),
                    tensor_name: "source.test.grouped-norm",
                    input: &[0.5, -0.25, 0.1, 0.2],
                    weights: &weights,
                    group_width: 2,
                    groups: 2,
                    eps: 1e-6,
                })
                .unwrap(),
            RegionExecution::Unsupported
        ));
        assert!(matches!(
            MatVecOnly
                .try_gdn_conv(GdnConvRegion {
                    source_identity: SourceIdentity::unbound(),
                    tensor_name: "blk.0.ssm_conv1d.weight",
                    activation: &[0.1],
                    weights: &[0.1, 0.2, 0.3, 0.4],
                    history: &[0.5, 0.6, 0.7],
                })
                .unwrap(),
            GdnConvExecution::Unsupported
        ));
    }

    #[test]
    fn recurrent_region_rejects_overflow_nonfinite_and_incomplete_read_sets() {
        let coefficients = [GdnCoefficients {
            gain: 0.9,
            beta: 0.4,
            scale: 0.5,
        }];
        let region = GdnRecurrentRegion {
            source_identity: SourceIdentity::unbound(),
            operation: "blk.0.gdn_recurrent",
            width: 2,
            q: &[1.0, 2.0],
            k: &[0.25, -0.5],
            v: &[0.1, 0.2],
            coefficients: &coefficients,
            state: &[0.0; 4],
        };
        assert!(region.validate().is_ok());
        for invalid in [
            GdnRecurrentRegion {
                width: usize::MAX,
                ..region
            },
            GdnRecurrentRegion {
                state: &[0.0; 3],
                ..region
            },
            GdnRecurrentRegion {
                q: &[1.0, f32::NAN],
                ..region
            },
            GdnRecurrentRegion {
                state: &[0.0, 0.0, f32::INFINITY, 0.0],
                ..region
            },
            GdnRecurrentRegion {
                coefficients: &[],
                ..region
            },
            GdnRecurrentRegion {
                coefficients: &[GdnCoefficients {
                    gain: 1.0,
                    beta: f32::NAN,
                    scale: 0.5,
                }],
                ..region
            },
        ] {
            assert!(invalid.validate().is_err());
        }
    }

    #[test]
    fn completed_region_rejects_shape_nan_and_infinity() {
        let weight = [0u8; 18];
        let activations = [0.0f32; 32];
        let region = MatVecRegion {
            source_identity: SourceIdentity::unbound(),
            tensor_name: "blk.0.attn_gate.weight",
            kind: QuantKind::Q4_0,
            columns: 32,
            rows: 1,
            packed_weights: &weight,
            activation: &activations,
        };
        assert!(validate_completed(region, &[0.0]).is_ok());
        assert!(validate_completed(region, &[]).is_err());
        assert!(validate_completed(region, &[1.0, 2.0]).is_err());
        assert!(validate_completed(region, &[f32::NAN]).is_err());
        assert!(validate_completed(region, &[f32::INFINITY]).is_err());
    }
}

#[cfg(test)]
mod silu_tests {
    use super::*;
    #[test]
    fn silu_validation_and_explicit_unsupported() {
        struct MatrixOnly;
        impl MatVecKernel for MatrixOnly {
            fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
                Ok(RegionExecution::Unsupported)
            }
        }
        let r = SiluRegion {
            source_identity: SourceIdentity::unbound(),
            operation: "silu",
            input: &[0.0, -0.0, -3.0],
            multiplier: None,
        };
        r.validate().unwrap();
        assert!(matches!(
            MatrixOnly.try_silu(r).unwrap(),
            RegionExecution::Unsupported
        ));
        assert!(
            SiluRegion {
                multiplier: Some(&[1.0]),
                ..r
            }
            .validate()
            .is_err()
        );
        assert!(
            SiluRegion {
                input: &[f32::NAN],
                ..r
            }
            .validate()
            .is_err()
        );
        assert!(SiluRegion { input: &[], ..r }.validate().is_err());
    }
}

#[cfg(test)]
mod coefficient_tests {
    use super::*;
    #[test]
    fn coefficient_shape_values_and_optional_unsupported() {
        struct MatrixOnly;
        impl MatVecKernel for MatrixOnly {
            fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
                Ok(RegionExecution::Unsupported)
            }
        }
        let r = GdnCoefficientRegion {
            source_identity: SourceIdentity::unbound(),
            operation: "gdn.coefficients",
            alpha: &[0., 1.],
            beta: &[1., 0.],
            dt: &[-3., -4.],
            a: &[-1., -2.],
            scale: 0.5,
        };
        r.validate().unwrap();
        assert!(matches!(
            MatrixOnly.try_gdn_coefficients(r).unwrap(),
            GdnCoefficientExecution::Unsupported
        ));
        for bad in [
            GdnCoefficientRegion { dt: &[], ..r },
            GdnCoefficientRegion {
                alpha: &[f32::NAN, 1.],
                ..r
            },
            GdnCoefficientRegion { scale: 0., ..r },
        ] {
            assert!(bad.validate().is_err());
        }
    }
}

#[cfg(test)]
mod rope_tests {
    use super::*;
    #[test]
    fn rope_proposal_validation_rejects_bad_geometry_and_immutable_copy_changes() {
        struct MatrixOnly;
        impl MatVecKernel for MatrixOnly {
            fn try_execute(&self, _: MatVecRegion<'_>) -> Result<RegionExecution> {
                Ok(RegionExecution::Unsupported)
            }
        }
        let values = [-0.0f32, 0., 1., 2.];
        let r = RopeKvRegion {
            source_identity: SourceIdentity::unbound(),
            operation: "rope",
            query: &values,
            key: &values,
            value: &values,
            head_width: 4,
            query_heads: 1,
            kv_heads: 1,
            rotate: 2,
            position: 1,
            theta_scale: 0.5,
        };
        r.validate().unwrap();
        assert!(matches!(
            MatrixOnly.try_rope_kv(r).unwrap(),
            RopeKvExecution::Unsupported
        ));
        for bad in [
            RopeKvRegion { rotate: 5, ..r },
            RopeKvRegion { kv_heads: 0, ..r },
            RopeKvRegion {
                head_width: usize::MAX,
                ..r
            },
            RopeKvRegion {
                position: usize::MAX,
                ..r
            },
        ] {
            assert!(bad.validate().is_err());
        }
        let mut ready = RopeKvCompleted {
            query: values.to_vec(),
            key: values.to_vec(),
            value: values.to_vec(),
        };
        ready.validate(r).unwrap();
        ready.value[0] = 0.;
        assert!(ready.validate(r).is_err());
        ready.value = values.to_vec();
        ready.key[2] = 3.;
        assert!(ready.validate(r).is_err());
    }
}

#[cfg(test)]
mod attention_tests {
    use super::*;
    #[test]
    fn qk_validates_complete_shape_and_original_scale_bits() {
        let query = [1.0f32; 64];
        let keys = [Arc::<[f32]>::from(vec![2.0; 32])];
        let r = AttentionQkRegion {
            source_identity: SourceIdentity::unbound(),
            operation: "qk",
            query: &query,
            keys: &keys,
            head_width: 32,
            query_heads: 2,
            kv_heads: 1,
            scale: 0.5,
        };
        r.validate().unwrap();
        for bad in [
            AttentionQkRegion { keys: &[], ..r },
            AttentionQkRegion {
                head_width: usize::MAX,
                ..r
            },
            AttentionQkRegion { kv_heads: 0, ..r },
            AttentionQkRegion {
                scale: f32::NAN,
                ..r
            },
        ] {
            assert!(bad.validate().is_err());
        }
        let mut ready = AttentionQkCompleted {
            dots: vec![-0.0, 2.0],
            scores: vec![-0.0, 1.0],
        };
        ready.validate(r).unwrap();
        ready.scores[0] = 0.0;
        assert!(ready.validate(r).is_err());
        ready.scores[0] = -0.0;
        ready.scores[1] = 0.5;
        assert!(ready.validate(r).is_err());
        ready.scores[1] = 1.0;
        ready.dots.pop();
        assert!(ready.validate(r).is_err());
    }
}
