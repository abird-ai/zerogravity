//! Source text recording only; no backend, completion or learned permission.
use super::*;
use crate::session_root::{RootAttempt, RootBinding};

pub const BF16_TEXT_CAPACITY: usize = 64;
pub const BF16_TEXT_VOCABULARY: usize = 262144;

/// Actual original logical call, including masked prefill contraction extent.
#[derive(Clone, Debug)]
pub struct Bf16TextCall {
    geometry: TextCallGeometry,
    attempt: RootAttempt,
    ids: Arc<[u32]>,
}
impl Bf16TextCall {
    /// Bind metadata only. Finite numerical/backend eligibility stays with the
    /// recorded profile and its source-owned lowerer.
    pub fn bind(geometry: TextCallGeometry, attempt: RootAttempt, ids: Arc<[u32]>) -> Result<Self> {
        geometry.validate_call(&attempt, &ids)?;
        if attempt.base().frontier() != 0 && ids.len() != 1 {
            return Err(Error::Unsupported(
                "BF16 text candidate cached call requires one row".into(),
            ));
        }
        Ok(Self {
            geometry,
            attempt,
            ids,
        })
    }
    pub fn geometry(&self) -> TextCallGeometry {
        self.geometry
    }
    pub fn require_current(&self, current: RootBinding) -> Result<()> {
        if current != self.attempt.base() {
            return Err(Error::State("BF16 text foreign/stale original call".into()));
        }
        Ok(())
    }
    pub fn attempt(&self) -> &RootAttempt {
        &self.attempt
    }
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn rows(&self) -> usize {
        self.ids.len()
    }
    pub fn row(&self, index: usize) -> Result<Bf16TextRow<'_>> {
        if index >= self.ids.len() {
            return Err(Error::State("BF16 text row outside actual call".into()));
        }
        Ok(Bf16TextRow { call: self, index })
    }
}
/// A borrowed actual row; neither a completed producer nor a published root.
pub struct Bf16TextRow<'a> {
    call: &'a Bf16TextCall,
    index: usize,
}
impl Bf16TextRow<'_> {
    pub fn token(&self) -> u32 {
        self.call.ids[self.index]
    }
    pub fn position(&self) -> usize {
        self.call.attempt.base().frontier() as usize + self.index
    }
    pub fn valid_tokens(&self) -> usize {
        self.position() + 1
    }
    pub fn needs_logits(&self) -> bool {
        self.index + 1 == self.call.rows()
    }
    pub fn pv_geometry(&self, width: usize) -> Result<Bf16TextPvGeometry> {
        if width == 0
            || width
                .checked_mul(self.call.attempt.frontier() as usize)
                .is_none()
        {
            return Err(Error::Unsupported(
                "BF16 text PV positive checked width".into(),
            ));
        }
        Ok(Bf16TextPvGeometry {
            query_rows: self.call.rows(),
            contraction: self.call.attempt.frontier() as usize,
            active: self.valid_tokens(),
            width,
        })
    }
    /// Explicit actual call M even though this candidate uses the source CPU
    /// per-output full-K contract. This is not an original GPU GEMM selection.
    pub fn projection_geometry(&self, matrix: &SourceTensor) -> Result<Bf16TextProjectionGeometry> {
        if matrix.source != self.call.attempt.base().source()
            || matrix.kind != QuantKind::Bf16
            || matrix.axes != MatrixAxes::OutputInput
            || matrix.rows == 0
            || matrix.rows > self.call.geometry.vocabulary()
            || matrix.columns == 0
        {
            return Err(Error::State(
                "BF16 text actual projection call/source extent".into(),
            ));
        }
        matrix.packed_bytes()?;
        Ok(Bf16TextProjectionGeometry {
            query_rows: self.call.rows(),
            input_columns: matrix.columns,
            output_rows: matrix.rows,
            contract: ProjectionContract::TorchCpuBf16Avx2,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16TextProjectionGeometry {
    pub query_rows: usize,
    pub input_columns: usize,
    pub output_rows: usize,
    pub contract: ProjectionContract,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16TextPvGeometry {
    pub query_rows: usize,
    pub contraction: usize,
    pub active: usize,
    pub width: usize,
}

/// Numeric zero ties, lowest original ID; source max bits remain unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bf16TextGreedy;
impl Bf16TextGreedy {
    pub fn prefer(candidate: (u32, u32), prior: (u32, u32)) -> Result<bool> {
        let (id, bits) = candidate;
        let (old_id, old_bits) = prior;
        let value = f32::from_bits(bits);
        let old = f32::from_bits(old_bits);
        if id as usize >= BF16_TEXT_VOCABULARY
            || old_id as usize >= BF16_TEXT_VOCABULARY
            || !value.is_finite()
            || !old.is_finite()
            || bits & 0xffff != 0
            || old_bits & 0xffff != 0
        {
            return Err(Error::Malformed(
                "BF16 text complete finite source argmax word".into(),
            ));
        }
        let key = |bits: u32| {
            if bits & 0x7fff_ffff == 0 {
                0x8000_0000
            } else if bits & 0x8000_0000 != 0 {
                !bits
            } else {
                bits ^ 0x8000_0000
            }
        };
        Ok(key(bits) > key(old_bits) || key(bits) == key(old_bits) && id < old_id)
    }
}

pub(super) fn validate_builder(
    capacity: usize,
    required: &BTreeSet<StateSlot>,
    slice: bool,
) -> Result<()> {
    if slice
        || capacity > BF16_TEXT_CAPACITY
        || required
            .iter()
            .copied()
            .ne((0..15).map(|i| StateSlot::Kv(KvDomain(i))))
    {
        return Err(Error::Unsupported(
            "BF16 text complete15-domain bounded plan".into(),
        ));
    }
    Ok(())
}
pub(super) fn validate_constant(vector: &SourceVector<'_>) -> Result<()> {
    if !matches!(vector.values.len(), 256 | 512 | 1536)
        || vector
            .values
            .iter()
            .any(|v| !v.is_finite() || v.to_bits() & 0xffff != 0)
    {
        return Err(Error::Unsupported(
            "BF16 text direct source gamma width/precision".into(),
        ));
    }
    Ok(())
}
pub(super) fn precision(
    builder: &PlanBuilder,
    operation: &PlanOperation,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    use ActivationPrecision::{Bf16WidenedF32 as Bf16, F32};
    if inputs.iter().any(|v| v.precision != Bf16) {
        return Err(Error::State("BF16 text foreign arithmetic input".into()));
    }
    match operation {
        PlanOperation::Embed { tensor, token } => {
            typed::validate_bf16_matrix(tensor)?;
            if token.0 != ParameterSlot::Token || !inputs.is_empty() {
                return Err(Error::State("BF16 text actual embedding parameter".into()));
            }
            Ok(Bf16)
        }
        PlanOperation::Project { tensor } => {
            typed::validate_bf16_matrix(tensor)?;
            if inputs.len() != 1
                || inputs[0].elements != tensor.columns
                || !matches!(tensor.columns, 256 | 1536 | 2048 | 4096 | 6144 | 12288)
                || tensor.rows > BF16_TEXT_VOCABULARY
            {
                return Err(Error::Unsupported(
                    "BF16 text full source projection extent".into(),
                ));
            }
            Ok(Bf16)
        }
        PlanOperation::Rms { gamma, eps, groups } => {
            let g = builder
                .constants
                .get(gamma.0 as usize)
                .ok_or_else(|| Error::State("BF16 text source gamma absent".into()))?;
            if inputs.len() != 1
                || g.source != builder.source
                || eps.to_bits() != 1e-6f32.to_bits()
                || !matches!(
                    (g.values.len(), *groups),
                    (256 | 512 | 1536, 1) | (256 | 512, 8) | (256, 35)
                )
                || g.values.len().checked_mul(*groups) != Some(inputs[0].elements)
            {
                return Err(Error::Unsupported(
                    "BF16 text direct grouped RMS contract".into(),
                ));
            }
            Ok(Bf16)
        }
        PlanOperation::TextElementwise(operation) => {
            operation.validate(inputs.len())?;
            if inputs.is_empty() || inputs.iter().any(|v| v.shape != inputs[0].shape) {
                return Err(Error::State("BF16 text elementwise exact shapes".into()));
            }
            Ok(Bf16)
        }
        PlanOperation::TextRotary { geometry, .. } => {
            if inputs.len() != 1
                || !matches!(
                    (geometry.head_width, geometry.rotated_pairs),
                    (256, 128) | (512, 64)
                )
                || geometry.theta.to_bits()
                    != if geometry.head_width == 256 {
                        10000.0f32.to_bits()
                    } else {
                        1000000.0f32.to_bits()
                    }
            {
                return Err(Error::Unsupported("BF16 text source rotary extent".into()));
            }
            geometry.validate(inputs[0].elements)?;
            Ok(Bf16)
        }
        PlanOperation::TextWindowQk {
            geometry, window, ..
        } => {
            if geometry.query_heads != 8
                || geometry.kv_heads != 1
                || !matches!((geometry.width, window), (256, Some(512)) | (512, None))
            {
                return Err(Error::Unsupported(
                    "BF16 text unit-scale grouped attention".into(),
                ));
            }
            Ok(Bf16)
        }
        PlanOperation::AttentionSoftmax { heads: 8, .. } => Ok(Bf16),
        PlanOperation::AttentionValue { geometry, .. } => {
            if geometry.query_heads != 8
                || geometry.kv_heads != 1
                || !matches!(geometry.width, 256 | 512)
            {
                return Err(Error::Unsupported("BF16 text PV grouped attention".into()));
            }
            Ok(Bf16)
        }
        PlanOperation::View(_) | PlanOperation::Contiguous => inputs
            .first()
            .map(|v| v.precision)
            .ok_or_else(|| Error::State("BF16 text missing view input".into())),
        PlanOperation::AppendKv
        | PlanOperation::KvPrefixGuard { .. }
        | PlanOperation::ObserveState { .. }
        | PlanOperation::Observe { .. }
        | PlanOperation::CheckFinite { .. } => Ok(F32),
        _ => Err(Error::Unsupported(
            "operation has no BF16 text candidate contract".into(),
        )),
    }
}

#[cfg(test)]
mod tests;
