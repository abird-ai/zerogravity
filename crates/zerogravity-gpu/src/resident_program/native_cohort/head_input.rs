//! Four actual final source rows; no independent batch/model/root authority.
use super::super::*;
use super::values::CohortValues;
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, text::TextCallInput};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
fn require_last_row(rows: usize, selected: Option<usize>) -> Result<usize> {
    if rows == 0 || selected != Some(rows - 1) {
        return Err(Error::State(
            "B4 captured head requires actual final selected row".into(),
        ));
    }
    Ok(rows - 1)
}
pub(super) struct CohortHeadInput {
    inputs: Box<[Arc<TextCallInput>]>,
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
        inputs: &[Arc<TextCallInput>],
        values: Arc<CohortValues>,
        status: ResidentLease,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        crate::cuda::native::cohort_native::require_input_shape(inputs)?;
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
        let hidden = inputs[0].scope().native_capture().hidden_width();
        let selected_row =
            require_last_row(logical, inputs[0].record().selection().selected_row())?;
        let normalized = inputs[0]
            .record()
            .selection()
            .normalized()
            .ok_or_else(|| Error::State("B4 final normalized selection absent".into()))?;
        for input in inputs {
            if !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.rows() != logical
                || input.record().selection().normalized() != Some(normalized)
                || input.record().selection().selected_row() != Some(logical - 1)
                || input.prompt_tokens() != 47
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
            source_offset: selected_row * (hidden / 2),
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_cannot_retarget_captured_last_row_gather() {
        for rows in [1, 47] {
            assert_eq!(require_last_row(rows, Some(rows - 1)).unwrap(), rows - 1);
            assert!(require_last_row(rows, None).is_err());
            assert!(require_last_row(rows, Some(rows)).is_err());
        }
        assert!(require_last_row(47, Some(0)).is_err());
        assert!(require_last_row(0, None).is_err());
        assert!(require_last_row(0, Some(usize::MAX)).is_err());
    }
}
