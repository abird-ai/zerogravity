//! Original selected Q6_K rows → source GGUF RMS → raw effective gamma.
use super::source_bindings::{PrefixBinding, PrefixContract};
use super::*;
use crate::accelerator::SourceIdentity;

pub(crate) struct GgufPrefixOwner {
    pub(crate) descriptor: PrefixBinding,
    pub(crate) model: Arc<LoadedQwenModel>,
    pub(crate) admission: BulkF32HostAdmission,
    _metadata: GlobalResourceLease,
}
pub struct GgufPrefixSlice {
    pub(crate) owner: Arc<GgufPrefixOwner>,
}
impl GgufPrefixSlice {
    #[allow(clippy::too_many_arguments)] // Immutable source, temporal and physical scope.
    pub fn new(
        model: Arc<LoadedQwenModel>,
        base: usize,
        capacity: usize,
        ids: &[u32],
        admission: BulkF32HostAdmission,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        if model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || model.geometry().hidden_size != 2048
            || model.geometry().vocab_size != 248320
            || model.geometry().layers.first() != Some(&crate::config::LayerKind::LinearAttention)
            || model.layers[0].attn_norm.len() != 2048
            || model.layers[0].attn_norm.iter().any(|v| !v.is_finite())
            || base != 0
            || ids.len() != 64
            || admission.resource.backend != "host"
        {
            return Err(Error::Unsupported(
                "closed original 2B first64 GGUF prefix scope".into(),
            ));
        }
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
        let descriptor = PrefixBinding::new(
            span,
            model.as_ref(),
            PrefixContract::First64GgmlF32V1,
            model.eps,
        )?;
        not_cancelled(cancelled)?;
        Ok(Self {
            owner: Arc::new(GgufPrefixOwner {
                descriptor,
                model,
                admission,
                _metadata: metadata,
            }),
        })
    }
    pub fn descriptor(&self) -> &PrefixBinding {
        &self.owner.descriptor
    }
    pub fn record(&self) -> Result<RecordedOperatorSlice> {
        let mut builder = PlanBuilder::new_operator_slice(
            self.descriptor().source(),
            ProgramNumericalProfile::GgufSourceV1,
        )?;
        let (_, normalized) = crate::forward::gguf_prefix_slice(&mut builder, self.descriptor())?;
        builder.finish_operator_slice(normalized)
    }
    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<GgufPrefixOutput> {
        self.execute_with_provider(None, cancelled)
    }
    pub fn execute_with_provider(
        &self,
        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<GgufPrefixOutput> {
        not_cancelled(cancelled)?;
        if kernel.is_some() {
            return Err(Error::Unsupported(
                "unaware provider cannot execute GGUF prefix contract".into(),
            ));
        }
        let mut math = crate::host_math::HostMath::for_gguf_prefix_slice(self, cancelled);
        let (embedding, normalized) =
            crate::forward::gguf_prefix_slice(&mut math, self.descriptor())?;
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        not_cancelled(cancelled)?;
        Ok(GgufPrefixOutput {
            embedding,
            normalized,
            owner: self.owner.clone(),
        })
    }
}

#[derive(Clone)]
pub struct GgufPrefixOutput {
    embedding: crate::host_math::HostValue,
    normalized: crate::host_math::HostValue,
    pub(crate) owner: Arc<GgufPrefixOwner>,
}
impl GgufPrefixOutput {
    pub fn embedding(&self) -> Result<&[f32]> {
        self.embedding.as_slice()
    }
    pub fn unweighted(&self) -> Result<&[f32]> {
        self.normalized.gguf_prefix_unweighted(&self.owner)
    }
    pub fn normalized(&self) -> Result<&[f32]> {
        self.normalized.as_slice()
    }
    pub fn span(&self) -> &TeacherSpan {
        self.owner.descriptor.span()
    }
    pub fn source(&self) -> SourceIdentity {
        self.owner.descriptor.source()
    }
    pub fn descriptor(&self) -> &PrefixBinding {
        &self.owner.descriptor
    }
    pub(crate) fn same_producer(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner) && self.normalized.same_backing(&other.normalized)
    }
    pub(crate) fn model(&self) -> &Arc<LoadedQwenModel> {
        &self.owner.model
    }
    pub(crate) fn admission(&self) -> &BulkF32HostAdmission {
        &self.owner.admission
    }
}
