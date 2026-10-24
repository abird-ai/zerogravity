//! Retained producer binding under the existing HostMath adapter.
use super::bulk_f32::BulkBacking;
use super::*;
use crate::accelerator::program::ZenSsmOutDescriptor;
use crate::operator_slice::bulk_f32::{BulkGdnSsmOutSlice, not_cancelled};
use std::sync::atomic::AtomicBool;
use zerogravity_cpu::math::haswell_bulk_f32::{BulkF32Admission, verify_environment};
use zerogravity_cpu::math::zen_blas_q8_f32::{ZenQ8F32Request, execute};

pub(super) struct ZenSsmOutCall<'a> {
    slice: &'a BulkGdnSsmOutSlice,
    cancelled: Option<&'a AtomicBool>,
}
impl<'engine, 'state, 'trace> HostMath<'engine, 'state, 'trace> {
    pub(crate) fn for_zen_ssm_out_slice(
        slice: &'engine BulkGdnSsmOutSlice,
        cancelled: Option<&'engine AtomicBool>,
    ) -> Self {
        let mut math = Self::for_operator_slice(slice.owner.model.as_ref(), None);
        math.zen_ssm_out_call = Some(ZenSsmOutCall { slice, cancelled });
        math
    }
    fn zen_call(&self) -> Result<&ZenSsmOutCall<'engine>> {
        if self.engine.is_some()
            || self.state.is_some()
            || self.region_kernel.is_some()
            || self.bulk_call.is_some()
            || self.gguf_gdn_upstream_call.is_some()
            || self.gguf_prefix_call.is_some()
            || self.source.program_numerical_profile() != ProgramNumericalProfile::GgufSourceV1
        {
            return Err(Error::Unsupported(
                "Zen ssm_out private same-engine host slice only".into(),
            ));
        }
        self.zen_ssm_out_call
            .as_ref()
            .ok_or_else(|| Error::Unsupported("Zen ssm_out has no retained-gated issuer".into()))
    }
    pub(super) fn import_zen_ssm_out(&self, descriptor: &ZenSsmOutDescriptor) -> Result<HostValue> {
        let call = self.zen_call()?;
        not_cancelled(call.cancelled)?;
        verify_environment()?;
        let owner = &call.slice.owner;
        if descriptor != &owner.descriptor
            || descriptor.source() != self.source.source_identity()
            || descriptor.span() != owner.gated.recurrent().coefficients().span()
        {
            return Err(Error::State(
                "Zen ssm_out exact bound producer/source/issuer".into(),
            ));
        }
        HostValue::from_bulk(
            BulkBacking::ZenInput {
                owner: owner.clone(),
            },
            descriptor.layout().backing_words(),
            descriptor.source(),
        )
    }
    pub(super) fn execute_zen_ssm_out(
        &self,
        descriptor: &ZenSsmOutDescriptor,
        input: &HostValue,
    ) -> Result<HostValue> {
        let call = self.zen_call()?;
        not_cancelled(call.cancelled)?;
        verify_environment()?;
        let owner = &call.slice.owner;
        if descriptor != &owner.descriptor
            || descriptor.source() != self.source.source_identity()
            || descriptor.span() != owner.gated.recurrent().coefficients().span()
            || input.source != Some(descriptor.source())
            || input.profile != Some(ProgramNumericalProfile::GgufSourceV1)
            || input.precision != ActivationPrecision::F32
            || input.elements != descriptor.layout().backing_words()
            || input.view != TensorView::vector(0, input.elements)
            || !matches!(input.data.bulk.as_ref(), Some(BulkBacking::ZenInput { owner: actual }) if Arc::ptr_eq(actual, owner))
        {
            return Err(Error::State(
                "Zen ssm_out foreign/observed/retagged/partial producer".into(),
            ));
        }
        descriptor.tensor().validate_source(owner.model.as_ref())?;
        let weights = owner.model.tensor_bytes(&descriptor.tensor().name)?;
        let completed = execute(
            ZenQ8F32Request {
                source: descriptor.source(),
                weights,
                input: input.as_slice()?,
            },
            BulkF32Admission {
                arbiter: &owner.admission.arbiter,
                resource: &owner.admission.resource,
                owner: &owner.admission.owner,
            },
            call.cancelled,
        )?;
        verify_environment()?;
        not_cancelled(call.cancelled)?;
        if completed.source() != descriptor.source()
            || completed.shape() != descriptor.layout().shape()
            || completed.values().len() != descriptor.layout().backing_words()
        {
            return Err(Error::State(
                "Zen ssm_out primitive source/output geometry".into(),
            ));
        }
        HostValue::from_bulk(
            BulkBacking::ZenOutput {
                completed,
                _owner: owner.clone(),
            },
            descriptor.layout().backing_words(),
            descriptor.source(),
        )
    }
}
