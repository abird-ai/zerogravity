//! Independent invocations sharing one text-execution activation traversal.
//! Projections use their existing multi-row primitive; stateful operators retain
//! each session's private state and position.
use super::{
    cold::{Cold, view},
    math::{arg, dim, grid},
    *,
};
use crate::{
    resident_leases::{ResidentLease, ResidentSubmission as Flight},
    resident_program::{
        StateBinding,
        bf16_projection::{Bf16ProjectionBindings, Bf16ProjectionStorage, PreparedBf16Projection},
    },
};
use cubecl::{bytes::Bytes, server::Handle};
use zerogravity_core::{
    accelerator::program::{InvocationPhase, PlanNode, SourceOperator, SourceOperatorKind},
    device_events::{BufferRole, DeviceAccess},
};

pub(crate) struct CohortSubmission {
    pub status: Handle,
    pub dispatches: usize,
    pub fused_dispatches: usize,
}

/// Transient activation/scratch extensions of the original cold arena.
/// Reuse slots that already fit the flattened independent invocations.
pub(crate) struct CohortBuffers {
    slots: Vec<ResidentLease>,
    packed: ResidentLease,
    projection: [ResidentLease; 3],
    selected: Option<ResidentLease>,
    pub(crate) logits: Option<ResidentLease>,
    extra: Vec<ResidentLease>,
}
impl CohortBuffers {
    pub(crate) fn accesses(&self) -> impl Iterator<Item = (ResidentLease, DeviceAccess)> + '_ {
        self.extra
            .iter()
            .cloned()
            .map(|lease| (lease, DeviceAccess::Write))
    }
    fn value(
        &self,
        cold: &Cold,
        id: zerogravity_core::accelerator::program::ValueId,
        words: usize,
    ) -> Result<Handle> {
        view(cold.value_lease_in(&self.slots, id)?, 0, bytes(words)?)
    }
}

fn bytes(words: usize) -> Result<usize> {
    words
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("text cohort word extent".into()))
}

/// Resident weights can revisit a bounded token tile without a source reload.
/// Keep streamed weight traversal across the whole cohort.
fn projection_token_tile(operand: &PreparedBf16Projection, active: usize, native: bool) -> usize {
    match operand.storage() {
        Bf16ProjectionStorage::Resident if !native => active.min(128),
        _ => active,
    }
}

impl Cold {
    fn cohort_calls<'a>(
        &self,
        requests: &'a [ForwardPlan],
    ) -> Result<Vec<&'a zerogravity_core::accelerator::program::TextInvocation>> {
        if !matches!(requests.len(), 2 | 4 | 8) {
            return Err(Error::Unsupported("text cohort width".into()));
        }
        let calls = requests
            .iter()
            .map(|r| self.validate_request(r))
            .collect::<Result<Vec<_>>>()?;
        let first = calls[0];
        if calls.iter().any(|call| {
            call.phase() != first.phase()
                || call.tokens().len() != first.tokens().len()
                || call.tokens().base() != first.tokens().base()
        }) {
            return Err(Error::Unsupported(
                "text cohort requires equal aligned invocations".into(),
            ));
        }
        Ok(calls)
    }

    pub(crate) fn cohort_buffers(
        &self,
        requests: &[ForwardPlan],
        emit_logits: bool,
    ) -> Result<Option<CohortBuffers>> {
        let calls = self.cohort_calls(requests)?;
        let width = calls.len();
        let rows = calls[0].tokens().len();
        let nodes = if emit_logits {
            requests[0].nodes()
        } else {
            requests[0].state_nodes()
        };
        let mut slots = vec![0usize; self.slots.len()];
        let mut projection = 0usize;
        let mut packed = 0usize;
        let mut selected = 0usize;
        for node in nodes {
            let PlanOperation::SourceOperator { descriptor: d, .. } = &node.operation else {
                return Err(Error::State("text cohort materialization absent".into()));
            };
            for (ids, words) in [
                (&node.inputs, d.input_elements()?),
                (&node.outputs, d.output_elements()?),
            ] {
                if ids.len() != words.len() {
                    return Err(Error::State("text cohort operand count".into()));
                }
                for (&id, words) in ids.iter().zip(words) {
                    let i = id.0 as usize;
                    if words > self.layout.plan.values()[i].capacity_elements()? {
                        return Err(Error::State("text cohort logical value extent".into()));
                    }
                    // Terminal logits are a separate readback owner.
                    if Some(id) != self.layout.plan.logits() {
                        let slot = self.layout.bindings[i].slot;
                        slots[slot] = slots[slot].max(bytes(words * width)?);
                    }
                }
            }
            if let SourceOperatorKind::Projection { selected_row } = d.kind() {
                let active = if selected_row.is_some() {
                    width
                } else {
                    width * rows
                };
                for m in d.matrices() {
                    let operand = &self.operands[&m.name];
                    let tile =
                        projection_token_tile(operand, active, self.rocblas_workspace.is_some());
                    let words = tile
                        .checked_mul(m.rows)
                        .ok_or_else(|| Error::Limit("text cohort projection extent".into()))?;
                    if tile * operand.max_launch_rows()
                        > self.service.client().properties().hardware.max_cube_count.0 as usize
                    {
                        return Ok(None);
                    }
                    projection = projection.max(bytes(words)?);
                    packed = packed.max(bytes(tile * m.columns)? / 2);
                    if selected_row.is_some() && rows > 1 {
                        selected = selected.max(bytes(active * m.columns)?);
                    }
                }
            }
        }
        if projection == 0 || packed == 0 {
            return Err(Error::State("text cohort projection absent".into()));
        }
        let logits_bytes = if emit_logits {
            bytes(width * self.layout.vocabulary)?
        } else {
            0
        };
        let extra_bytes = layout::sum(
            slots
                .iter()
                .zip(&self.slots)
                .map(|(&n, lease)| if n > lease.bytes() { n } else { 0 })
                .chain([
                    if packed > self.packed.bytes() {
                        packed
                    } else {
                        0
                    },
                    if projection > self.scratch[layout::PROJECT].bytes() {
                        projection
                    } else {
                        0
                    },
                    if projection > self.accumulation.bytes() {
                        projection
                    } else {
                        0
                    },
                    if projection > self.flags.bytes() {
                        projection
                    } else {
                        0
                    },
                    selected,
                    logits_bytes,
                ]),
        )?;
        if extra_bytes > self.service.remaining_capacity()? {
            return Ok(None);
        }
        let mut extra = Vec::new();
        let mut allocate = |n, role| {
            let lease = self.service.allocate_uninitialized(n, role)?;
            extra.push(lease.clone());
            Ok::<_, Error>(lease)
        };
        let mut extend = |lease: &ResidentLease, n, role| {
            if n <= lease.bytes() {
                Ok(lease.clone())
            } else {
                allocate(n, role)
            }
        };
        let slots = slots
            .into_iter()
            .zip(&self.slots)
            .map(|(n, lease)| extend(lease, n, BufferRole::Activation))
            .collect::<Result<Vec<_>>>()?;
        let packed = extend(&self.packed, packed, BufferRole::Scratch)?;
        let projection = [
            extend(
                &self.scratch[layout::PROJECT],
                projection,
                BufferRole::Activation,
            )?,
            extend(&self.accumulation, projection, BufferRole::Scratch)?,
            extend(&self.flags, projection, BufferRole::Scratch)?,
        ];
        let selected = (selected != 0)
            .then(|| allocate(selected, BufferRole::Activation))
            .transpose()?;
        let logits = (logits_bytes != 0)
            .then(|| allocate(logits_bytes, BufferRole::Output))
            .transpose()?;
        Ok(Some(CohortBuffers {
            slots,
            packed,
            projection,
            selected,
            logits,
            extra,
        }))
    }

    pub(crate) async fn execute_cohort(
        self: &Arc<Self>,
        flight: &mut Flight<'_>,
        requests: &[ForwardPlan],
        states: &[Vec<StateBinding>],
        buffers: &CohortBuffers,
        device_tokens: Option<&[Handle]>,
    ) -> Result<CohortSubmission> {
        let calls = self.cohort_calls(requests)?;
        let width = calls.len();
        if states.len() != width {
            return Err(Error::State("text cohort state count".into()));
        }
        let base = calls[0].tokens().base();
        let rows = calls[0].tokens().len();
        let decode = calls[0].phase() == InvocationPhase::Decode;
        if device_tokens.is_some_and(|tokens| !decode || tokens.len() != width || rows != 1) {
            return Err(Error::State("HIP cohort device-token extent/phase".into()));
        }
        let emit_logits = buffers.logits.is_some();
        let maps = calls
            .iter()
            .zip(states)
            .map(|(call, states)| self.states(call, states))
            .collect::<Result<Vec<_>>>()?;

        // Each row was validated against the resident workspace above. Also prove
        // that roots from different rows do not alias one another.
        let mut roots = buffers.extra.clone();
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
                "text-execution cohort physical roots overlap".into(),
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
        flight.require_accesses(
            buffers
                .extra
                .iter()
                .map(|lease| (lease, DeviceAccess::Write)),
        )?;
        flight.retain_completion_owner(self.clone())?;
        self.native
            .require_live(self.service.client(), self.source.source_identity())?;
        self.service
            .client()
            .write(&self.status_view()?, Bytes::from_bytes_vec(vec![0; 4]));

        let node_lists = requests
            .iter()
            .map(|request| {
                if emit_logits {
                    request.nodes()
                } else {
                    request.state_nodes()
                }
            })
            .collect::<Vec<_>>();
        let nodes = node_lists[0];
        if node_lists.iter().any(|peer| peer.len() != nodes.len()) {
            return Err(Error::State(
                "text-execution cohort node coverage differs".into(),
            ));
        }
        let mut dispatches = 0usize;
        let mut fused_dispatches = 0usize;
        for (index, node) in nodes.iter().enumerate() {
            let descriptors = node_lists
                .iter()
                .map(|nodes| {
                    let peer = &nodes[index];
                    if peer.inputs != node.inputs || peer.outputs != node.outputs {
                        return Err(Error::State(
                            "text-execution cohort topology differs".into(),
                        ));
                    }
                    let PlanOperation::SourceOperator { descriptor, .. } = &peer.operation else {
                        return Err(Error::State(
                            "text-execution cohort materialization differs".into(),
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
                    "text-execution cohort operator semantics differ".into(),
                ));
            }
            if matches!(first.kind(), SourceOperatorKind::Projection { .. }) {
                let (actual, fused) = self
                    .projection_cohort(flight, node, first, width, buffers)
                    .await?;
                dispatches = dispatches
                    .checked_add(actual)
                    .ok_or_else(|| Error::Limit("text cohort dispatch count".into()))?;
                fused_dispatches = fused_dispatches
                    .checked_add(fused)
                    .ok_or_else(|| Error::Limit("text cohort fused count".into()))?;
                continue;
            }
            if decode && matches!(first.kind(), SourceOperatorKind::CausalConvSplit { .. }) {
                dispatches = dispatches
                    .checked_add(
                        self.convolution_decode_cohort(node, first, &maps, width, base, buffers)?,
                    )
                    .ok_or_else(|| Error::Limit("text cohort dispatch count".into()))?;
                continue;
            }
            let rowwise = match first.kind() {
                SourceOperatorKind::Rms(_)
                | SourceOperatorKind::ResidualRms(_)
                | SourceOperatorKind::Sigmoid { .. }
                | SourceOperatorKind::Multiply { .. }
                | SourceOperatorKind::SiluAndMul { .. } => true,
                // Rotary positions repeat per prompt, not across the flattened
                // token axis. Only the existing aligned decode path shares it.
                SourceOperatorKind::QkRmsRotaryGated(_) => decode,
                _ => false,
            };
            if rowwise {
                let input_lengths = first.input_elements()?;
                let output_lengths = first.output_elements()?;
                if node.inputs.len() != input_lengths.len()
                    || node.outputs.len() != output_lengths.len()
                {
                    return Err(Error::State(
                        "text-execution cohort row-wise operand count".into(),
                    ));
                }
                let inputs = node
                    .inputs
                    .iter()
                    .zip(&input_lengths)
                    .map(|(id, elements)| {
                        buffers.value(
                            self,
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
                        buffers.value(
                            self,
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
                            flight,
                            node,
                            first,
                            &maps[0],
                            width * rows,
                            base,
                            &buffers.slots,
                            &inputs,
                            &outputs,
                            None,
                        )
                        .await?,
                    )
                    .ok_or_else(|| Error::Limit("text cohort dispatch count".into()))?;
                continue;
            }

            for row in 0..width {
                let descriptor = descriptors[row];
                let input_lengths = descriptor.input_elements()?;
                let output_lengths = descriptor.output_elements()?;
                if node.inputs.len() != input_lengths.len()
                    || node.outputs.len() != output_lengths.len()
                {
                    return Err(Error::State("text-execution cohort operand count".into()));
                }
                let inputs = node
                    .inputs
                    .iter()
                    .zip(&input_lengths)
                    .map(|(id, elements)| {
                        let all = buffers.value(
                            self,
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
                        let all = buffers.value(
                            self,
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
                            flight,
                            node,
                            descriptor,
                            &maps[row],
                            rows,
                            base,
                            &buffers.slots,
                            &inputs,
                            &outputs,
                            device_tokens.map(|tokens| &tokens[row]),
                        )
                        .await?,
                    )
                    .ok_or_else(|| Error::Limit("text cohort dispatch count".into()))?;
            }
        }
        self.page.finish()?;
        if fused_dispatches == 0 {
            return Err(Error::State(
                "text-execution cohort produced no shared projection".into(),
            ));
        }
        Ok(CohortSubmission {
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
        buffers: &CohortBuffers,
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
        let packed = buffers.value(
            self,
            node.inputs[0],
            packed_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort convolution input".into()))?,
        )?;
        let activated = buffers.value(
            self,
            node.outputs[0],
            activated_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("decode cohort convolution activated".into()))?,
        )?;
        let gate = buffers.value(
            self,
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

    async fn projection_cohort(
        &self,
        flight: &mut Flight<'_>,
        node: &PlanNode,
        descriptor: &SourceOperator,
        width: usize,
        buffers: &CohortBuffers,
    ) -> Result<(usize, usize)> {
        let SourceOperatorKind::Projection { selected_row } = descriptor.kind() else {
            unreachable!("validated projection")
        };
        let rows = descriptor.invocation().tokens().len();
        let columns = descriptor.matrices()[0].columns;
        let combined = descriptor.matrices().iter().map(|m| m.rows).sum::<usize>();
        if node.inputs.len() != 1
            || node.outputs.len() != 1
            || descriptor.input_elements()? != [rows * columns]
            || descriptor.output_elements()?
                != [if selected_row.is_some() {
                    combined
                } else {
                    rows * combined
                }]
        {
            return Err(Error::State("text cohort projection geometry".into()));
        }
        let active = if selected_row.is_some() {
            width
        } else {
            width * rows
        };
        let input_words = width * rows * columns;
        let input = buffers.value(self, node.inputs[0], input_words)?;
        let input_lease = self.value_lease_in(&buffers.slots, node.inputs[0])?;
        let status = self.status_view()?;
        let client = self.service.client();
        let mut dispatches = 0usize;
        let (input, input_lease) = if let Some(row) = selected_row.filter(|_| rows > 1) {
            let lease = buffers
                .selected
                .as_ref()
                .ok_or_else(|| Error::State("text cohort selected input absent".into()))?;
            let target = view(lease, 0, bytes(active * columns)?)?;
            crate::program_kernels::checked_copy_words(width, columns, rows * columns, columns)?;
            unsafe {
                crate::program_kernels::copy_strided_words::launch_unchecked(
                    client,
                    grid(width * columns),
                    dim(),
                    arg(&input, input_words),
                    arg(&target, active * columns),
                    arg(&status, 1),
                    (row * columns) as u32,
                    0u32,
                    (rows * columns) as u32,
                    columns as u32,
                    width,
                    columns,
                    false,
                );
            }
            dispatches += 1;
            (target, lease)
        } else {
            (input, input_lease)
        };
        let expected_output = active * combined;
        let output_lease = if selected_row.is_some() {
            if Some(node.outputs[0]) != self.layout.plan.logits() {
                return Err(Error::State(
                    "text cohort selected projection is not logits".into(),
                ));
            }
            buffers
                .logits
                .as_ref()
                .ok_or_else(|| Error::State("text cohort logits absent".into()))?
        } else {
            self.value_lease_in(&buffers.slots, node.outputs[0])?
        };
        let output = view(output_lease, 0, bytes(expected_output)?)?;
        let mut offset = 0usize;
        let mut fused = 0usize;
        for matrix in descriptor.matrices() {
            let operand = &self.operands[&matrix.name];
            let tile = projection_token_tile(operand, active, self.rocblas_workspace.is_some());
            // A single complete matrix already has the final row stride.
            // Keep scratch for combined matrices, aliases and the head readback.
            let direct = selected_row.is_none()
                && matrix.rows == combined
                && input_lease.id() != output_lease.id();
            for first in (0..active).step_by(tile) {
                let count = tile.min(active - first);
                let elements = count * matrix.rows;
                let input =
                    super::math::subview(&input, bytes(first * columns)?, bytes(count * columns)?)?;
                let output = super::math::subview(
                    &output,
                    bytes(first * combined)?,
                    bytes(count * combined)?,
                )?;
                let (destination, destination_lease) = if direct {
                    (output.clone(), output_lease)
                } else {
                    (
                        view(&buffers.projection[0], 0, bytes(elements)?)?,
                        &buffers.projection[0],
                    )
                };
                dispatches = dispatches
                    .checked_add(
                        self.enqueue_projection(
                            flight,
                            operand,
                            Bf16ProjectionBindings {
                                input: &input,
                                input_lease,
                                output: &destination,
                                output_lease: destination_lease,
                                packed: &buffers.packed,
                                accumulation: &buffers.projection[1],
                                flags: &buffers.projection[2],
                                status: &self.status,
                                zero: &self.zero,
                            },
                            count,
                        )
                        .await?,
                    )
                    .ok_or_else(|| Error::Limit("text cohort projection dispatches".into()))?;
                if !direct {
                    flight.require_accesses([(output_lease, DeviceAccess::Write)])?;
                    unsafe {
                        super::kernels::columns::launch_unchecked(
                            client,
                            grid(elements),
                            dim(),
                            arg(&destination, elements),
                            arg(&output, count * combined),
                            arg(&status, 1),
                            count as u32,
                            0u32,
                            offset as u32,
                            matrix.rows,
                            matrix.rows,
                            combined,
                        );
                    }
                    dispatches += 1;
                }
            }
            fused += 1;
            offset += matrix.rows;
        }
        Ok((dispatches, fused))
    }
}
