//! Four actual final source rows; no independent batch/model/root authority.
use super::super::*;
use super::values::CohortValues;
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, short_hf::ShortHfCallInput};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
pub(super) struct CohortHeadInput {
    inputs: Box<[Arc<ShortHfCallInput>]>,
    values: Arc<CohortValues>,
    source: ResidentLease,
    selected: ResidentLease,
    status: ResidentLease,
    source_stride: usize,
    source_offset: usize,
    width_words: usize,
    submitted: AtomicBool,
}
impl CohortHeadInput {
    pub(super) fn prepare(
        inputs: &[Arc<ShortHfCallInput>],
        values: Arc<CohortValues>,
        status: ResidentLease,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        crate::vllm_cuda_qualification::cohort_native::require_input_shape(inputs)?;
        init.require_service(values.service())?;
        if inputs.len() != 4
            || status.bytes() != 4
            || !Arc::ptr_eq(status.service(), values.service())
        {
            return Err(Error::State(
                "B4 final original row population/status".into(),
            ));
        }
        let logical = inputs[0].rows();
        let hidden = inputs[0].call().model_profile().hidden_size();
        let normalized = inputs[0].record().normalized();
        for input in inputs {
            if !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.rows() != logical
                || input.record().normalized() != normalized
                || input.workload().prompt_rows() != 47
            {
                return Err(Error::State(
                    "B4 final normalized source producer/span".into(),
                ));
            }
        }
        let source = values.lease(normalized)?.clone();
        if source.bytes() != 4 * logical * hidden * 2 {
            return Err(Error::State(
                "B4 exact final normalized physical source".into(),
            ));
        }
        let selected = init.allocate_empty(4 * hidden * 2, BufferRole::Activation)?;
        let result = Arc::new(Self {
            inputs: inputs.to_vec().into_boxed_slice(),
            values,
            source,
            selected,
            status,
            source_stride: logical * (hidden / 2),
            source_offset: (logical - 1) * (hidden / 2),
            width_words: hidden / 2,
            submitted: AtomicBool::new(false),
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn lease(&self) -> &ResidentLease {
        &self.selected
    }
    pub(super) fn enqueue(&self, node: usize, permits: &[PrefixPhaseEnqueue<'_>]) -> Result<usize> {
        if node != 253 || permits.len() != 4 {
            return Err(Error::State(
                "B4 actual final normalizer source node/permits".into(),
            ));
        }
        self.values.require_inputs(&self.inputs)?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, self.values.service())?;
        }
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State(
                "B4 original final row selection already consumed".into(),
            ));
        }
        unsafe {
            crate::program_kernels::copy_strided_words::launch_unchecked(
                self.values.service().client(),
                grid(4 * self.width_words),
                threads(),
                arg(&self.source.handle(), self.source.bytes() / 4),
                arg(&self.selected.handle(), 4 * self.width_words),
                arg(&self.status.handle(), 1),
                self.source_offset as u32,
                0,
                self.source_stride as u32,
                self.width_words as u32,
                4,
                self.width_words,
                false,
            );
        }
        Ok(1)
    }
    pub(super) fn accesses(&self) -> [(ResidentLease, DeviceAccess); 3] {
        [
            (self.source.clone(), DeviceAccess::Read),
            (self.selected.clone(), DeviceAccess::Write),
            (self.status.clone(), DeviceAccess::Write),
        ]
    }
}
