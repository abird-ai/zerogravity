//! Independent cached-decode rows sharing one text-execution activation traversal.
//! Projections use their existing multi-row primitive; stateful operators retain
//! each session's private state and position.
use super::{
    cold::{Cold, view},
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
            if descriptors[1..].iter().any(|descriptor| {
                first.kind() != descriptor.kind()
                    || first.numerical_contract() != descriptor.numerical_contract()
                    || first.matrices() != descriptor.matrices()
                    || first.parameters() != descriptor.parameters()
            }) {
                return Err(Error::State(
                    "text-execution decode cohort operator semantics differ".into(),
                ));
            }
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
            if matches!(first.kind(), SourceOperatorKind::CausalConvSplit { .. }) {
                dispatches = dispatches
                    .checked_add(self.convolution_decode_cohort(node, first, &maps, width, base)?)
                    .ok_or_else(|| Error::Limit("decode cohort dispatch count".into()))?;
                continue;
            }
            if matches!(first.kind(), SourceOperatorKind::GatedDelta { .. }) {
                dispatches = dispatches
                    .checked_add(self.gated_delta_decode_cohort(node, first, &maps, width)?)
                    .ok_or_else(|| Error::Limit("decode cohort dispatch count".into()))?;
                continue;
            }
            if matches!(
                first.kind(),
                SourceOperatorKind::Rms(_)
                    | SourceOperatorKind::ResidualRms(_)
                    | SourceOperatorKind::QkRmsRotaryGated(_)
                    | SourceOperatorKind::Sigmoid { .. }
                    | SourceOperatorKind::Multiply { .. }
                    | SourceOperatorKind::SiluAndMul { .. }
            ) {
                let input_lengths = first.input_elements()?;
                let output_lengths = first.output_elements()?;
                if node.inputs.len() != input_lengths.len()
                    || node.outputs.len() != output_lengths.len()
                {
                    return Err(Error::State(
                        "text-execution decode cohort row-wise operand count".into(),
                    ));
                }
                let inputs = node
                    .inputs
                    .iter()
                    .zip(&input_lengths)
                    .map(|(id, elements)| {
                        self.value(
                            *id,
                            elements.checked_mul(width).ok_or_else(|| {
                                Error::Limit("decode cohort row-wise input extent".into())
                            })?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let outputs = node
                    .outputs
                    .iter()
                    .zip(&output_lengths)
                    .map(|(id, elements)| {
                        self.value(
                            *id,
                            elements.checked_mul(width).ok_or_else(|| {
                                Error::Limit("decode cohort row-wise output extent".into())
                            })?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                dispatches = dispatches
                    .checked_add(
                        self.node_views(
                            flight, node, first, &maps[0], width, base, &inputs, &outputs,
                        )
                        .await?,
                    )
                    .ok_or_else(|| Error::Limit("decode cohort dispatch count".into()))?;
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

    fn convolution_decode_cohort(
        &self,
        node: &PlanNode,
        descriptor: &SourceOperator,
        states: &[BTreeMap<StateSlot, &StateBinding>],
        width: usize,
        base: usize,
    ) -> Result<usize> {
        let SourceOperatorKind::CausalConvSplit {
            domain,
            geometry: g,
        } = descriptor.kind()
        else {
            unreachable!("validated convolution")
        };
        let input_lengths = descriptor.input_elements()?;
        let output_lengths = descriptor.output_elements()?;
        let [packed_elements, coefficient_elements] = input_lengths.as_slice() else {
            return Err(Error::State("decode cohort convolution input arity".into()));
        };
        let [activated_elements, gate_elements] = output_lengths.as_slice() else {
            return Err(Error::State(
                "decode cohort convolution output arity".into(),
            ));
        };
        if node.inputs.len() != 2
            || node.outputs.len() != 2
            || *packed_elements != g.qkv_channels + g.z_channels
            || *coefficient_elements != g.coefficient_channels
            || *activated_elements != g.qkv_channels
            || *gate_elements != g.z_channels
            || descriptor.parameters().len() != 1
        {
            return Err(Error::State(
                "decode cohort convolution one-row geometry".into(),
            ));
        }
        let packed = self.value(
            node.inputs[0],
            packed_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort convolution input".into()))?,
        )?;
        let activated = self.value(
            node.outputs[0],
            activated_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort convolution activated".into()))?,
        )?;
        let gate = self.value(
            node.outputs[1],
            gate_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort convolution gate".into()))?,
        )?;
        let weights = self.parameter(&descriptor.parameters()[0])?;
        let state_words = g
            .qkv_channels
            .checked_mul(4)
            .ok_or_else(|| Error::Limit("decode cohort convolution state".into()))?;
        let mut prior = Vec::with_capacity(width);
        let mut candidate = Vec::with_capacity(width);
        for map in states {
            let binding = map
                .get(&StateSlot::Conv(domain))
                .copied()
                .ok_or_else(|| Error::State("decode cohort convolution state absent".into()))?;
            let p = binding.prior.handles();
            let n = binding.candidate.handles();
            if p.len() != 1 || n.len() != 1 {
                return Err(Error::State(
                    "decode cohort convolution vector state".into(),
                ));
            }
            prior.push(p[0].clone());
            candidate.push(n[0].clone());
        }
        if prior.len() != width || candidate.len() != width {
            return Err(Error::State(
                "decode cohort convolution state coverage".into(),
            ));
        }
        let p = |row: usize| prior.get(row).unwrap_or(&prior[0]);
        let n = |row: usize| candidate.get(row).unwrap_or(&candidate[0]);
        let status = self.status_view()?;
        let client = self.service.client();
        unsafe {
            super::kernels::convolution_cohort::launch_unchecked(
                client,
                grid(width * g.qkv_channels),
                dim(),
                arg(&packed, width * packed_elements),
                arg(p(0), state_words),
                arg(p(1), state_words),
                arg(p(2), state_words),
                arg(p(3), state_words),
                arg(p(4), state_words),
                arg(p(5), state_words),
                arg(p(6), state_words),
                arg(p(7), state_words),
                arg(&weights, g.qkv_channels * 4),
                arg(&activated, width * activated_elements),
                arg(&gate, width * gate_elements),
                arg(n(0), state_words),
                arg(n(1), state_words),
                arg(n(2), state_words),
                arg(n(3), state_words),
                arg(n(4), state_words),
                arg(n(5), state_words),
                arg(n(6), state_words),
                arg(n(7), state_words),
                arg(&status, 1),
                u32::from(base != 0),
                width,
                g.qkv_channels,
                g.z_channels,
            );
        }
        Ok(1)
    }

    fn gated_delta_decode_cohort(
        &self,
        node: &PlanNode,
        descriptor: &SourceOperator,
        states: &[BTreeMap<StateSlot, &StateBinding>],
        width: usize,
    ) -> Result<usize> {
        let SourceOperatorKind::GatedDelta { domain, geometry } = descriptor.kind() else {
            unreachable!("validated recurrent GDN")
        };
        let input_lengths = descriptor.input_elements()?;
        let output_lengths = descriptor.output_elements()?;
        let [activated_elements, gate_elements, ba_elements] = input_lengths.as_slice() else {
            return Err(Error::State("decode cohort recurrent input arity".into()));
        };
        let [output_elements] = output_lengths.as_slice() else {
            return Err(Error::State("decode cohort recurrent output arity".into()));
        };
        let recurrent_width = geometry.value_heads * 128;
        let channels = (2 * geometry.key_heads + geometry.value_heads) * 128;
        if node.inputs.len() != 3
            || node.outputs.len() != 1
            || *activated_elements != channels
            || *gate_elements != recurrent_width
            || *ba_elements != geometry.value_heads * 2
            || *output_elements != recurrent_width
        {
            return Err(Error::State(
                "decode cohort recurrent one-row geometry".into(),
            ));
        }
        let activated = self.value(
            node.inputs[0],
            activated_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort recurrent activated".into()))?,
        )?;
        let gate = self.value(
            node.inputs[1],
            gate_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort recurrent gate".into()))?,
        )?;
        let ba = self.value(
            node.inputs[2],
            ba_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort recurrent coefficients".into()))?,
        )?;
        let output = self.value(
            node.outputs[0],
            output_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort recurrent output".into()))?,
        )?;
        let mut prior = Vec::with_capacity(width);
        let mut candidate = Vec::with_capacity(width);
        for map in states {
            let binding = map
                .get(&StateSlot::Recurrent(domain))
                .copied()
                .ok_or_else(|| Error::State("decode cohort recurrent state absent".into()))?;
            let p = binding.prior.handles();
            let n = binding.candidate.handles();
            if p.len() != 1 || n.len() != 1 {
                return Err(Error::State("decode cohort recurrent vector state".into()));
            }
            prior.push(p[0].clone());
            candidate.push(n[0].clone());
        }
        self.gdn_decode_cohort(
            &activated,
            &gate,
            &ba,
            &prior,
            &candidate,
            &output,
            descriptor.parameters(),
            geometry,
            width,
        )
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
