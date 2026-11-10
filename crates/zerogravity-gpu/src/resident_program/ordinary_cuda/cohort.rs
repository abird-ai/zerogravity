//! Aligned independent-session decode batching over the ordinary CUDA body.
use super::{attention, body::OrdinaryCudaBody};
use crate::{
    cuda::ordinary_text::Step,
    resident_leases::{ResidentLease, ResidentSubmission as PhysicalSubmission},
    resident_program::{StateBinding, StateBuffers},
};
use cubecl::server::Handle;
use std::collections::BTreeMap;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, PlanOperation, SourceCall, SourceCallMode, SourceOperatorKind,
        SourceRowSelection, SourceRows, StateVersion, TextCallGeometry,
    },
    device_events::DeviceAccess,
};

pub(crate) struct OrdinaryCudaDecodeCohort {
    pub(crate) logits: ResidentLease,
    pub(crate) status: Handle,
    pub(crate) proposals: Vec<Vec<(StateVersion, StateBuffers)>>,
    pub(crate) dispatches: usize,
    pub(crate) fused_dispatches: usize,
}

impl OrdinaryCudaBody {
    fn cohort_value_handle(
        &self,
        request: &ForwardPlan,
        id: zerogravity_core::accelerator::program::ValueId,
        per_row: usize,
        rows: usize,
    ) -> Result<Handle> {
        self.value_handle_at(
            request,
            id,
            per_row
                .checked_mul(rows)
                .ok_or_else(|| Error::Limit("ordinary CUDA cohort value extent".into()))?,
            0,
        )
    }

    /// One physical traversal for aligned cached-decode rows. Stateless work is
    /// row-batched; stateful operators retain each session's private root/state.
    pub(crate) async fn execute_decode_cohort(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        logits: ResidentLease,
    ) -> Result<OrdinaryCudaDecodeCohort> {
        if requests.len() != 4 || states.len() != requests.len() {
            return Err(Error::Unsupported(
                "ordinary CUDA decode cohort requires four rows".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| self.validate_request(request))
            .collect::<Result<Vec<_>>>()?;
        let first = calls[0];
        if first.mode() != SourceCallMode::Decode
            || first.rows().len() != 1
            || calls.iter().any(|call| {
                call.mode() != SourceCallMode::Decode
                    || call.rows().len() != 1
                    || call.rows().base() != first.rows().base()
                    || call.rows().capacity() != first.rows().capacity()
                    || call.rows().vocabulary() != first.rows().vocabulary()
                    || call.source() != first.source()
            })
        {
            return Err(Error::Unsupported(
                "ordinary CUDA decode cohort requires aligned cached rows".into(),
            ));
        }
        let geometry = TextCallGeometry::new(
            first.source(),
            first.rows().vocabulary(),
            first.rows().capacity(),
        )?;
        let ids = calls
            .iter()
            .map(|call| call.rows().ids()[0])
            .collect::<Vec<_>>();
        let batch_call = SourceCall::new(
            SourceRows::new(geometry, first.rows().base(), ids.into())?,
            SourceCallMode::Prefill,
        )?;
        let state_maps = calls
            .iter()
            .zip(states)
            .map(|(call, state)| self.validate_states(call, state))
            .collect::<Result<Vec<_>>>()?;
        flight.require_accesses(
            self.accesses(crate::resident_program::ResidentOutput::Logits)
                .iter()
                .map(|(lease, access)| (lease, *access)),
        )?;
        flight.require_accesses([(&logits, DeviceAccess::Write)])?;
        for row in states {
            for binding in row {
                for buffer in binding.prior.buffers() {
                    flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Read)])?;
                }
                for buffer in binding.candidate.buffers() {
                    flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Write)])?;
                }
            }
        }
        flight.retain_completion_owner(self.cold().clone())?;
        flight.retain_completion_owner(self.gemm().clone())?;

        let client = self.cold().service().client();
        self.clear_and_bind_call(client, &batch_call, false)?;
        let batch_step = Step::new(requests.len(), first.rows().base(), first.rows().capacity())?;
        let single_step = Step::new(1, first.rows().base(), first.rows().capacity())?;
        let cold = self.cold().plan();
        let mut dispatches = 0usize;
        let mut fused_dispatches = 0usize;
        let mut gate_alias = BTreeMap::<u32, u32>::new();

        for (node_index, node) in cold.nodes().iter().enumerate() {
            let PlanOperation::SourceOperatorTemplate {
                descriptor: template,
                ..
            } = &node.operation
            else {
                return Err(Error::State(
                    "ordinary CUDA cohort contains non-source template".into(),
                ));
            };
            if template.row_selection() == SourceRowSelection::LastActive {
                if !matches!(template.kind(), SourceOperatorKind::Projection { .. })
                    || node.inputs.len() != 1
                    || node.outputs.len() != 1
                {
                    return Err(Error::State(
                        "ordinary CUDA cohort last-active operator differs".into(),
                    ));
                }
                let input = self.cold().value_lease(node.inputs[0].0 as usize)?;
                let launched = self
                    .cold()
                    .projection(node_index)?
                    .enqueue_decode_rows(
                        requests.len(),
                        self.gemm(),
                        flight,
                        input,
                        &logits,
                        self.cold().source_page(),
                    )
                    .await?;
                dispatches += launched;
                fused_dispatches += launched;
                dispatches += attention::certify(
                    client,
                    &logits.handle(),
                    &self.cold().status().handle(),
                    requests.len() * first.rows().vocabulary(),
                )?;
                continue;
            }
            let stateful = !matches!(
                template.state_access(),
                zerogravity_core::accelerator::program::SourceStateAccess::None
            );
            if stateful {
                for row in 0..requests.len() {
                    let descriptor = template.bind_runtime_call(calls[row])?;
                    let (launched, fused) = self
                        .enqueue_node(
                            flight,
                            cold,
                            node_index,
                            node,
                            &descriptor,
                            &state_maps[row],
                            single_step,
                            row,
                            false,
                            None,
                            &mut gate_alias,
                        )
                        .await?;
                    dispatches += launched;
                    fused_dispatches += fused;
                }
            } else {
                let descriptor = template.bind_runtime_call(&batch_call)?;
                let repeated_position =
                    matches!(template.kind(), SourceOperatorKind::QkRmsRotaryGated(_));
                let (launched, fused) = self
                    .enqueue_node(
                        flight,
                        cold,
                        node_index,
                        node,
                        &descriptor,
                        &state_maps[0],
                        batch_step,
                        0,
                        repeated_position,
                        None,
                        &mut gate_alias,
                    )
                    .await?;
                dispatches += launched;
                fused_dispatches += fused;
            }
        }
        if let Some(page) = self.cold().source_page() {
            page.finish()?;
        }
        if !gate_alias.is_empty() {
            return Err(Error::State(
                "ordinary CUDA cohort attention gate alias not consumed".into(),
            ));
        }
        let expected = cold.proposed_domains();
        let proposals = states
            .iter()
            .map(|row| {
                expected
                    .iter()
                    .map(|version| {
                        let state = row
                            .iter()
                            .find(|state| state.slot == version.slot)
                            .ok_or_else(|| {
                                Error::State("ordinary CUDA cohort proposal absent".into())
                            })?;
                        Ok((*version, state.candidate.clone()))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(OrdinaryCudaDecodeCohort {
            logits,
            status: self.cold().status().handle(),
            proposals,
            dispatches,
            fused_dispatches,
        })
    }
}
