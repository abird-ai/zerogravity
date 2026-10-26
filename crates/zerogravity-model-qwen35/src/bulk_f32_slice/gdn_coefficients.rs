//! Closed original 2B layer0 bulk projections → existing GGUF coefficients.
//! Observed teachers or retained GGUF prefix; no model/state producer permit.
use super::*;
use crate::accelerator::{GdnCoefficientCompleted, GdnCoefficientRegion, SourceIdentity};

/// The projection schedule changes; the existing GGUF scalar consumer does not.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BulkGdnCoefficientContract {
    HaswellTinyBlasF32GgufCoefficientsV1,
}

struct Pair {
    alpha: BulkF32Slice,
    beta: BulkF32Slice,
    operation: String,
    _metadata: GlobalResourceLease,
}

/// One issuer for both original raw learned matrices and normalized teachers.
pub struct BulkGdnCoefficientSlice {
    pair: Arc<Pair>,
}

impl BulkGdnCoefficientSlice {
    #[allow(clippy::too_many_arguments)] // Source, temporal and physical scope.
    pub fn new(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        layer: usize,
        normalized: ExternalBulkF32<'_>,
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        if layer != 0
            || model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || model.geometry().hidden_size != 2048
            || model.geometry().recurrent_value_heads != Some(16)
            || model.geometry().layers.first() != Some(&crate::config::LayerKind::LinearAttention)
            || model.layers[0].ssm_norm.as_ref().map(Vec::len) != Some(128)
            || model.layers[0].dt.as_ref().map(Vec::len) != Some(16)
            || model.layers[0].a.as_ref().map(Vec::len) != Some(16)
            || normalized.layout != BulkF32Layout::dense([2048, ids.len(), 1, 1])?
            || admission.resource.backend != "host"
        {
            return Err(Error::Unsupported(
                "closed original 2B layer0 bulk coefficient geometry".into(),
            ));
        }
        // Retained through the last output clone, before the span/name/Arc.
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let span = TeacherSpan::new(
            model.source_identity(),
            base,
            capacity,
            model.geometry().vocab_size,
            ids,
        )?;
        let alpha = BulkF32Slice::bind_for_span(
            model.clone(),
            span.clone(),
            Some(QwenMatrixRole::GdnAlpha(layer)),
            None,
            ExternalBulkF32 {
                name: normalized.name,
                layout: normalized.layout,
                backing: normalized.backing,
            },
            admission.clone(),
            cancelled,
        )?;
        let beta = BulkF32Slice::bind_for_span(
            model,
            span,
            Some(QwenMatrixRole::GdnBeta(layer)),
            None,
            normalized,
            admission,
            cancelled,
        )?;
        if alpha.matrix().left().span() != beta.matrix().left().span()
            || alpha.matrix().output().shape() != [16, ids.len(), 1, 1]
            || beta.matrix().output() != alpha.matrix().output()
        {
            return Err(Error::State(
                "paired coefficient producer issuer/shape".into(),
            ));
        }
        not_cancelled(cancelled)?;
        Ok(Self {
            pair: Arc::new(Pair {
                alpha,
                beta,
                operation: crate::layer_tensor(layer, "gdn_coefficients"),
                _metadata: metadata,
            }),
        })
    }

    /// Retain the actual normalized producer; source/IDs/admission cannot be replaced.
    pub fn from_prefix(prefix: GgufPrefixOutput, cancelled: Option<&AtomicBool>) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        let model = prefix.model();
        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || model.geometry().hidden_size != 2048
            || model.geometry().recurrent_value_heads != Some(16)
            || model.geometry().layers.first() != Some(&crate::config::LayerKind::LinearAttention)
            || model.layers[0].ssm_norm.as_ref().map(Vec::len) != Some(128)
            || model.layers[0].dt.as_ref().map(Vec::len) != Some(16)
            || model.layers[0].a.as_ref().map(Vec::len) != Some(16)
            || prefix.span().len() != 64
            || prefix.source() != model.source_identity()
            || prefix.normalized()?.len() != 131072
            || prefix.admission().resource.backend != "host"
        {
            return Err(Error::Unsupported(
                "closed original 2B retained prefix coefficient scope".into(),
            ));
        }
        let admission = prefix.admission();
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let alpha =
            BulkF32Slice::bind_prefix(prefix.clone(), QwenMatrixRole::GdnAlpha(0), cancelled)?;
        let beta = BulkF32Slice::bind_prefix(prefix, QwenMatrixRole::GdnBeta(0), cancelled)?;
        if alpha.matrix().left().span() != beta.matrix().left().span()
            || alpha.matrix().output().shape() != [16, 64, 1, 1]
            || beta.matrix().output() != alpha.matrix().output()
        {
            return Err(Error::State(
                "retained prefix coefficient issuer/shape".into(),
            ));
        }
        not_cancelled(cancelled)?;
        Ok(Self {
            pair: Arc::new(Pair {
                alpha,
                beta,
                operation: crate::layer_tensor(0, "gdn_coefficients"),
                _metadata: metadata,
            }),
        })
    }

    /// Record the unchanged source-bound producer fragments, without a root.
    pub fn record_projections(&self) -> Result<[RecordedOperatorSlice; 2]> {
        Ok([self.pair.alpha.record()?, self.pair.beta.record()?])
    }

    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkGdnCoefficientOutput> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        let count = self.pair.alpha.matrix().left().span().len();
        let admission = &self.pair.alpha.inputs.owner.admission;
        // Per teacher: 16 three-word coefficients, softplus and raw gate,
        // plus the three retained Vec headers. Producer leases stay separate.
        let bytes = count
            .checked_mul(
                16 * 5 * std::mem::size_of::<f32>()
                    + std::mem::size_of::<GdnCoefficientCompleted>(),
            )
            .ok_or_else(|| Error::Limit("bulk coefficient output bytes".into()))?;
        let lease = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: bytes as u64,
            }],
        )?;
        // Both use existing HostMath generic forward fragments. No captured
        // alpha/beta result or legacy HostValue retag enters the consumer.
        let alpha = self.pair.alpha.execute(cancelled)?;
        let beta = self.pair.beta.execute(cancelled)?;
        let model = &self.pair.alpha.inputs.owner.model;
        let dt = model.layers[0].dt.as_deref().unwrap();
        let a = model.layers[0].a.as_deref().unwrap();
        let mut stages = Vec::new();
        stages
            .try_reserve_exact(count)
            .map_err(|_| Error::Limit("bulk coefficient result allocation".into()))?;
        if stages.capacity() != count {
            return Err(Error::Limit("bulk coefficient result capacity".into()));
        }
        // Columns are independent coefficient evaluations, never a model/token
        // loop or recurrent-state owner. [16,N,1,1] is head-fast per column.
        for teacher in 0..count {
            not_cancelled(cancelled)?;
            let range = teacher * 16..(teacher + 1) * 16;
            let completed = crate::source_gdn_coefficients(GdnCoefficientRegion {
                source_identity: model.source_identity(),
                operation: &self.pair.operation,
                alpha: alpha.view(range.clone())?,
                beta: beta.view(range)?,
                dt,
                a,
                scale: 1.0 / 128.0f32.sqrt(),
            })?;
            if completed.coefficients.capacity() != 16
                || completed.softplus.capacity() != 16
                || completed.gate_raw.capacity() != 16
            {
                return Err(Error::Limit("bulk coefficient stage capacity".into()));
            }
            stages.push(completed);
        }
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        not_cancelled(cancelled)?;
        Ok(BulkGdnCoefficientOutput {
            completed: Arc::new(Completed {
                stages,
                alpha,
                beta,
                _lease: lease,
                pair: self.pair.clone(),
            }),
        })
    }
}

struct Completed {
    stages: Vec<GdnCoefficientCompleted>,
    alpha: BulkF32Output,
    beta: BulkF32Output,
    _lease: GlobalResourceLease,
    pair: Arc<Pair>,
}

/// Private immutable outputs; no Vec escape, StateRoot or inference eligibility.
#[derive(Clone)]
pub struct BulkGdnCoefficientOutput {
    completed: Arc<Completed>,
}
impl BulkGdnCoefficientOutput {
    pub(super) fn prefix(&self) -> Option<&GgufPrefixOutput> {
        match &self.completed.pair.alpha.inputs.right {
            RightInput::ProducedPrefix(prefix) => Some(prefix),
            RightInput::External(_) => None,
        }
    }
    pub(super) fn model(&self) -> &Arc<LoadedQwenModel> {
        &self.completed.pair.alpha.inputs.owner.model
    }
    pub(super) fn admission(&self) -> &BulkF32HostAdmission {
        &self.completed.pair.alpha.inputs.owner.admission
    }
    pub fn contract(&self) -> BulkGdnCoefficientContract {
        BulkGdnCoefficientContract::HaswellTinyBlasF32GgufCoefficientsV1
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.completed.alpha.source_identity()
    }
    pub fn span(&self) -> &TeacherSpan {
        self.completed.pair.alpha.matrix().left().span()
    }
    pub fn projected_alpha(&self) -> Result<&[f32]> {
        self.completed.alpha.values()
    }
    pub fn projected_beta(&self) -> Result<&[f32]> {
        self.completed.beta.values()
    }
    pub fn teacher(&self, index: usize) -> Result<&GdnCoefficientCompleted> {
        self.completed
            .stages
            .get(index)
            .ok_or_else(|| Error::Malformed("bulk coefficient teacher index".into()))
    }
}
