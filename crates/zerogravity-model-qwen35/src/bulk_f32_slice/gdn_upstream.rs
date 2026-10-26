//! Actual retained normalized prefix → original QKV/Z and source Q/K/V.
use super::source_bindings::{GdnUpstreamBinding, GdnUpstreamContract};
use super::*;

pub(crate) struct GgufGdnUpstreamOwner {
    pub(crate) descriptor: GdnUpstreamBinding,
    pub(crate) prefix: GgufPrefixOutput,
    _metadata: GlobalResourceLease,
}
pub struct GgufGdnUpstreamSlice {
    pub(crate) owner: Arc<GgufGdnUpstreamOwner>,
}
impl GgufGdnUpstreamSlice {
    pub fn new(prefix: GgufPrefixOutput, cancelled: Option<&AtomicBool>) -> Result<Self> {
        not_cancelled(cancelled)?;
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        if prefix.span().base() != 0
            || prefix.span().len() != 64
            || prefix.normalized()?.len() != 131072
            || prefix.model().geometry().recurrent_key_heads != Some(16)
            || prefix.model().geometry().recurrent_value_heads != Some(16)
            || prefix.model().layers[0].conv_kernel.as_ref().map(Vec::len) != Some(24576)
        {
            return Err(Error::Unsupported(
                "closed original 2B first64 upstream scope".into(),
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
        let descriptor = GdnUpstreamBinding::new(
            prefix.descriptor().clone(),
            prefix.model().as_ref(),
            GdnUpstreamContract::First64Q4RepackF32V1,
        )?;
        not_cancelled(cancelled)?;
        Ok(Self {
            owner: Arc::new(GgufGdnUpstreamOwner {
                descriptor,
                prefix,
                _metadata: metadata,
            }),
        })
    }
    pub fn descriptor(&self) -> &GdnUpstreamBinding {
        &self.owner.descriptor
    }
    pub fn record(&self) -> Result<RecordedOperatorSlice> {
        let mut builder = PlanBuilder::new_operator_slice(
            self.descriptor().source(),
            ProgramNumericalProfile::GgufSourceV1,
        )?;
        let values = crate::forward::gguf_gdn_upstream_slice(&mut builder, self.descriptor())?;
        builder.finish_operator_slice(values.operands)
    }
    pub fn execute(&self, cancelled: Option<&AtomicBool>) -> Result<GgufGdnUpstreamOutput> {
        self.execute_with_provider(None, cancelled)
    }
    pub fn execute_with_provider(
        &self,
        kernel: Option<&dyn crate::accelerator::MatVecKernel>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<GgufGdnUpstreamOutput> {
        not_cancelled(cancelled)?;
        if kernel.is_some() {
            return Err(Error::Unsupported(
                "unaware provider cannot execute closed GGUF upstream".into(),
            ));
        }
        let mut math = crate::host_math::HostMath::for_gguf_gdn_upstream_slice(self, cancelled);
        let values = crate::forward::gguf_gdn_upstream_slice(&mut math, self.descriptor())?;
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        not_cancelled(cancelled)?;
        Ok(GgufGdnUpstreamOutput {
            qkv: values.qkv,
            z: values.z,
            convolution: values.convolution,
            operands: values.operands,
            owner: self.owner.clone(),
        })
    }
}
#[derive(Clone)]
pub struct GgufGdnUpstreamOutput {
    qkv: crate::host_math::HostValue,
    z: crate::host_math::HostValue,
    convolution: crate::host_math::HostValue,
    operands: crate::host_math::HostValue,
    pub(crate) owner: Arc<GgufGdnUpstreamOwner>,
}
impl GgufGdnUpstreamOutput {
    pub fn descriptor(&self) -> &GdnUpstreamBinding {
        &self.owner.descriptor
    }
    pub fn prefix(&self) -> &GgufPrefixOutput {
        &self.owner.prefix
    }
    pub fn qkv(&self) -> Result<&[f32]> {
        self.qkv.as_slice()
    }
    pub fn z(&self) -> Result<&[f32]> {
        self.z.as_slice()
    }
    pub fn convolution_raw(&self) -> Result<&[f32]> {
        self.convolution.gguf_gdn_conv_raw(&self.owner)
    }
    pub fn convolution_activated(&self) -> Result<&[f32]> {
        self.convolution.as_slice()
    }
    pub fn final_convolution_history(&self) -> Result<&[f32]> {
        self.convolution.gguf_gdn_conv_history(&self.owner)
    }
    pub fn query(&self) -> Result<&[f32]> {
        Ok(&self.operands.as_slice()?[..131072])
    }
    pub fn key(&self) -> Result<&[f32]> {
        Ok(&self.operands.as_slice()?[131072..262144])
    }
    pub fn value(&self) -> Result<&[f32]> {
        Ok(&self.operands.as_slice()?[262144..])
    }
    pub(crate) fn same_producer(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.owner, &other.owner) && self.operands.same_backing(&other.operands)
    }
}
