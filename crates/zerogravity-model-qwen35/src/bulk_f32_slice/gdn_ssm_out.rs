//! Computed bulk GDN gate → exact original Q8_0/OpenBLAS ssm_out cut.
use super::source_bindings::{GdnOutputBinding, GdnOutputContract};
use super::*;

pub(crate) struct ZenSsmOutOwner {
    pub(crate) descriptor: GdnOutputBinding,
    pub(crate) gated: BulkGdnGatedOutput,
    pub(crate) model: Arc<LoadedQwenModel>,
    pub(crate) admission: BulkF32HostAdmission,
    _metadata: GlobalResourceLease,
}
pub struct BulkGdnSsmOutSlice {
    pub(crate) owner: Arc<ZenSsmOutOwner>,
}
impl BulkGdnSsmOutSlice {
    /// Only the actual retained producer supplies source, issuer and admission.
    /// No caller-provided matrix/activation, model route or state publication.
    pub fn new(gated: BulkGdnGatedOutput, cancelled: Option<&AtomicBool>) -> Result<Self> {
        not_cancelled(cancelled)?;
        let coefficients = gated.recurrent().coefficients();
        let model = coefficients.model().clone();
        if coefficients.span().len() != 64
            || model.numerical_profile() != QwenNumericalProfile::GgufSourceV1
            || model.source_identity() != coefficients.source_identity()
            || gated.gated().len() != 131072
            || gated.gated().iter().any(|v| !v.is_finite())
        {
            return Err(Error::Unsupported(
                "closed Zen ssm_out produced gate/source/span".into(),
            ));
        }
        let admission = coefficients.admission().clone();
        let metadata = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: META,
            }],
        )?;
        let descriptor = GdnOutputBinding::new(
            coefficients.span().clone(),
            model.as_ref(),
            "blk.0.ssm_out.weight",
            GdnOutputContract::OpenBlas033Zen16F32V1,
        )?;
        not_cancelled(cancelled)?;
        Ok(Self {
            owner: Arc::new(ZenSsmOutOwner {
                descriptor,
                gated,
                model,
                admission,
                _metadata: metadata,
            }),
        })
    }
    pub fn descriptor(&self) -> &GdnOutputBinding {
        &self.owner.descriptor
    }
    pub fn record(&self) -> Result<RecordedOperatorSlice> {
        let mut builder = PlanBuilder::new_operator_slice(
            self.descriptor().source(),
            ProgramNumericalProfile::GgufSourceV1,
        )?;
        let output = crate::forward::zen_ssm_out_slice(&mut builder, self.descriptor())?;
        builder.finish_operator_slice(output)
    }
    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<BulkGdnSsmOutOutput> {
        self.execute_with_provider(None, cancelled)
    }
    pub fn execute_with_provider(
        &self,
        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<BulkGdnSsmOutOutput> {
        not_cancelled(cancelled)?;
        if kernel.is_some() {
            return Err(Error::Unsupported(
                "unaware provider cannot execute Zen ssm_out contract".into(),
            ));
        }
        let mut math = crate::host_math::HostMath::for_zen_ssm_out_slice(self, cancelled);
        let value = crate::forward::zen_ssm_out_slice(&mut math, self.descriptor())?;
        not_cancelled(cancelled)?;
        Ok(BulkGdnSsmOutOutput {
            value,
            owner: self.owner.clone(),
        })
    }
}

/// Borrowed output retains the primitive ticket and every actual producer.
#[derive(Clone)]
pub struct BulkGdnSsmOutOutput {
    value: crate::host_math::HostValue,
    owner: Arc<ZenSsmOutOwner>,
}
impl BulkGdnSsmOutOutput {
    pub fn values(&self) -> Result<&[f32]> {
        self.value.as_slice()
    }
    pub fn descriptor(&self) -> &GdnOutputBinding {
        &self.owner.descriptor
    }
    pub fn producer(&self) -> &BulkGdnGatedOutput {
        &self.owner.gated
    }
}
