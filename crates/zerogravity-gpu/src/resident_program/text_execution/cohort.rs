//! Independent cached-decode rows sharing one text-execution activation traversal.
//! Projections use their existing multi-row primitive; stateful operators retain
//! each session's private state and position.
use super::{
    cold::{Cold, view},
    layout,
    math::{arg, dim, grid},
    *,
};
use crate::{
    resident_leases::{ResidentLease, ResidentSubmission as Flight},
    resident_program::{StateBinding, bf16_projection::Bf16ProjectionBindings},
};
use cubecl::{bytes::Bytes, server::Handle};
use zerogravity_core::{
    accelerator::program::{InvocationPhase, PlanNode, SourceOperator, SourceOperatorKind},
    device_events::DeviceAccess,
};

pub(crate) struct DecodeCohortSubmission {
    pub status: Handle,
    pub dispatches: usize,
    pub fused_dispatches: usize,
}

impl Cold {
    pub(crate) fn decode_cohort_projection_bytes(&self, width: usize) -> Result<usize> {
        if !matches!(width, 2 | 4 | 8) {
            return Err(Error::Unsupported(
                "text-execution decode cohort requires two, four or eight rows".into(),
            ));
        }
        self.layout
            .projection_scratch_words
            .checked_mul(width)
            .and_then(|words| words.checked_mul(4))
            .ok_or_else(|| Error::Limit("text-execution decode cohort projection scratch".into()))
    }

    pub(crate) async fn execute_decode_cohort(
        self: &Arc<Self>,
        flight: &mut Flight<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        projection: [&ResidentLease; 3],
        logits: &ResidentLease,
    ) -> Result<DecodeCohortSubmission> {
        let width = requests.len();
        if !matches!(width, 2 | 4 | 8) || states.len() != width {
            return Err(Error::Unsupported(
                "text-execution decode cohort requires two, four or eight rows".into(),
            ));
        }
        let calls = requests
            .iter()
            .map(|request| self.validate_request(request))
            .collect::<Result<Vec<_>>>()?;
        let base = calls[0].tokens().base();
        if calls.iter().any(|call| {
            call.phase() != InvocationPhase::Decode
                || call.tokens().len() != 1
                || call.tokens().base() != base
        }) {
            return Err(Error::Unsupported(
                "text-execution decode cohort requires aligned cached rows".into(),
            ));
        }
        let maps = calls
            .iter()
            .zip(states)
            .map(|(call, states)| self.states(call, states))
            .collect::<Result<Vec<_>>>()?;

        // Each row was validated against the resident workspace above. Also prove
        // that roots from different rows do not alias one another.
        let mut roots = Vec::new();
        for row in states {
            for binding in row {
                for buffers in [&binding.prior, &binding.candidate] {
                    for buffer in buffers.buffers() {
                        roots.push(buffer.original_lease()?.clone());
                    }
                }
            }
        }
        let workspace = self
            .accesses
            .iter()
            .map(|(lease, _)| lease.clone())
            .collect::<Vec<_>>();
        if !self
            .service
            .pairwise_disjoint_extension(&workspace, &roots)?
        {
            return Err(Error::State(
                "text-execution decode cohort physical roots overlap".into(),
            ));
        }

        flight.require_accesses(self.accesses.iter().map(|(lease, access)| (lease, *access)))?;
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
        flight.require_accesses([
            (projection[0], DeviceAccess::Write),
            (projection[1], DeviceAccess::Write),
            (projection[2], DeviceAccess::Write),
            (logits, DeviceAccess::Write),
        ])?;
        flight.retain_completion_owner(self.clone())?;
        self.native
            .require_live(self.service.client(), self.source.source_identity())?;
        self.service
            .client()
            .write(&self.status_view()?, Bytes::from_bytes_vec(vec![0; 4]));

        let nodes = requests[0].nodes();
        if requests
            .iter()
            .any(|request| request.nodes().len() != nodes.len())
        {
            return Err(Error::State(
                "text-execution decode cohort node coverage differs".into(),
            ));
        }
        let mut dispatches = 0usize;
        let mut fused_dispatches = 0usize;
        for (index, node) in nodes.iter().enumerate() {
            let descriptors = requests
                .iter()
                .map(|request| {
                    let peer = &request.nodes()[index];
                    if peer.inputs != node.inputs || peer.outputs != node.outputs {
                        return Err(Error::State(
                            "text-execution decode cohort topology differs".into(),
                        ));
                    }
                    let PlanOperation::SourceOperator { descriptor, .. } = &peer.operation else {
                        return Err(Error::State(
                            "text-execution decode cohort materialization differs".into(),
                        ));
                    };
                    Ok(descriptor)
                })
                .collect::<Result<Vec<_>>>()?;
            let first = descriptors[0];
            if matches!(first.kind(), SourceOperatorKind::Projection { .. }) {
                let (actual, fused) = self
                    .projection_decode_cohort(flight, node, first, width, projection, logits)
                    .await?;
                dispatches = dispatches
                    .checked_add(actual)
                    .ok_or_else(|| Error::Limit("decode cohort dispatch count".into()))?;
                fused_dispatches = fused_dispatches
                    .checked_add(fused)
                    .ok_or_else(|| Error::Limit("decode cohort fused count".into()))?;
                continue;
            }

            for row in 0..width {
                let descriptor = descriptors[row];
                let input_lengths = descriptor.input_elements()?;
                let output_lengths = descriptor.output_elements()?;
                if node.inputs.len() != input_lengths.len()
                    || node.outputs.len() != output_lengths.len()
                {
                    return Err(Error::State(
                        "text-execution decode cohort operand count".into(),
                    ));
                }
                let inputs = node
                    .inputs
                    .iter()
                    .zip(&input_lengths)
                    .map(|(id, elements)| {
                        let all = self.value(
                            *id,
                            elements
                                .checked_mul(width)
                                .ok_or_else(|| Error::Limit("decode cohort input extent".into()))?,
                        )?;
                        super::math::subview(
                            &all,
                            row.checked_mul(*elements)
                                .and_then(|n| n.checked_mul(4))
                                .ok_or_else(|| Error::Limit("decode cohort input offset".into()))?,
                            elements
                                .checked_mul(4)
                                .ok_or_else(|| Error::Limit("decode cohort input bytes".into()))?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let outputs = node
                    .outputs
                    .iter()
                    .zip(&output_lengths)
                    .map(|(id, elements)| {
                        let all = self.value(
                            *id,
                            elements.checked_mul(width).ok_or_else(|| {
                                Error::Limit("decode cohort output extent".into())
                            })?,
                        )?;
                        super::math::subview(
                            &all,
                            row.checked_mul(*elements)
                                .and_then(|n| n.checked_mul(4))
                                .ok_or_else(|| {
                                    Error::Limit("decode cohort output offset".into())
                                })?,
                            elements
                                .checked_mul(4)
                                .ok_or_else(|| Error::Limit("decode cohort output bytes".into()))?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                dispatches = dispatches
                    .checked_add(
                        self.node_views(
                            flight, node, descriptor, &maps[row], 1, base, &inputs, &outputs,
                        )
                        .await?,
                    )
                    .ok_or_else(|| Error::Limit("decode cohort dispatch count".into()))?;
            }
        }
        self.page.finish()?;
        if fused_dispatches == 0 {
            return Err(Error::State(
                "text-execution decode cohort produced no shared projection".into(),
            ));
        }
        Ok(DecodeCohortSubmission {
            status: self.status.handle(),
            dispatches,
            fused_dispatches,
        })
    }

    async fn projection_decode_cohort(
        &self,
        flight: &mut Flight<'_>,
        node: &PlanNode,
        descriptor: &SourceOperator,
        width: usize,
        projection: [&ResidentLease; 3],
        logits: &ResidentLease,
    ) -> Result<(usize, usize)> {
        let SourceOperatorKind::Projection { selected_row } = descriptor.kind() else {
            unreachable!("validated projection")
        };
        let input_lengths = descriptor.input_elements()?;
        let output_lengths = descriptor.output_elements()?;
        let [input_elements] = input_lengths.as_slice() else {
            return Err(Error::State("decode cohort projection input arity".into()));
        };
        let [output_elements] = output_lengths.as_slice() else {
            return Err(Error::State("decode cohort projection output arity".into()));
        };
        let columns = descriptor.matrices()[0].columns;
        if *input_elements != columns {
            return Err(Error::State(
                "decode cohort projection one-row input geometry".into(),
            ));
        }
        let input = self.value(
            node.inputs[0],
            input_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort projection input".into()))?,
        )?;
        let combined = descriptor
            .matrices()
            .iter()
            .map(|matrix| matrix.rows)
            .sum::<usize>();
        let expected_output = output_elements
            .checked_mul(width)
            .ok_or_else(|| Error::Limit("decode cohort projection output".into()))?;
        if expected_output != width * combined {
            return Err(Error::State(
                "decode cohort projection output geometry".into(),
            ));
        }
        let output = if selected_row.is_some() {
            if node.outputs.len() != 1 || Some(node.outputs[0]) != requests_logits(node, self) {
                return Err(Error::State(
                    "decode cohort selected projection is not logits".into(),
                ));
            }
            view(logits, 0, expected_output * 4)?
        } else {
            self.value(node.outputs[0], expected_output)?
        };
        let output_lease = if selected_row.is_some() {
            logits
        } else {
            self.value_lease(node.outputs[0])?
        };
        let client = self.service.client();
        let status = self.status_view()?;
        let mut offset = 0usize;
        let mut dispatches = 0usize;
        let mut fused = 0usize;
        for matrix in descriptor.matrices() {
            let temporary = view(projection[0], 0, width * matrix.rows * 4)?;
            dispatches = dispatches
                .checked_add(
                    self.operands[&matrix.name]
                        .enqueue_rows(
                            flight,
                            &self.page,
                            Bf16ProjectionBindings {
                                input: &input,
                                input_lease: self.value_lease(node.inputs[0])?,
                                output: &temporary,
                                output_lease: projection[0],
                                packed: &self.packed,
                                accumulation: projection[1],
                                flags: projection[2],
                                status: &self.status,
                                zero: &self.zero,
                            },
                            width,
                            None,
                        )
                        .await?,
                )
                .ok_or_else(|| Error::Limit("decode cohort projection dispatches".into()))?;
            unsafe {
                super::kernels::columns::launch_unchecked(
                    client,
                    grid(width * matrix.rows),
                    dim(),
                    arg(&temporary, width * matrix.rows),
                    arg(&output, expected_output),
                    arg(&status, 1),
                    width as u32,
                    0u32,
                    offset as u32,
                    matrix.rows,
                    matrix.rows,
                    combined,
                );
            }
            flight.require_accesses([(output_lease, DeviceAccess::Write)])?;
            dispatches += 1;
            fused += 1;
            offset += matrix.rows;
        }
        Ok((dispatches, fused))
    }
}

fn requests_logits(
    node: &PlanNode,
    cold: &Cold,
) -> Option<zerogravity_core::accelerator::program::ValueId> {
    cold.layout
        .plan
        .logits()
        .filter(|id| node.outputs.as_slice() == [*id])
}
