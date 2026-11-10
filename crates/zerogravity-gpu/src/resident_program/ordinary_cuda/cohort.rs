//! Aligned independent-session decode batching over the ordinary CUDA body.
use super::{attention, body::OrdinaryCudaBody, exact_view};
use crate::{
    cuda::{
        gdn::{
            CONV_STATE_BYTES, CachedConvBatch, CachedDecodeBatch, CachedGatedBatch, GdnBuffer,
            GdnPhase, GdnRuntimeBuffer, GdnRuntimeCacheSlot,
            state_transport::{
                CONV_ELEMENTS, RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
            },
        },
        ordinary_text::Step,
    },
    resident_leases::{ResidentLease, ResidentSubmission as PhysicalSubmission},
    resident_program::{ResidentBufferLayout, ResidentPhysicalType, StateBinding, StateBuffers},
};
use cubecl::bytes::Bytes;
use cubecl::server::Handle;
use std::collections::BTreeMap;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, PlanOperation, SourceCall, SourceCallMode, SourceOperatorKind,
        SourceRowSelection, SourceRows, StateSlot, StateVersion, TextCallGeometry,
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
    fn exact_handle_view(
        handle: &Handle,
        offset: usize,
        bytes: usize,
        role: &str,
    ) -> Result<Handle> {
        let available = usize::try_from(handle.size_in_used())
            .map_err(|_| Error::Limit(format!("{role} handle extent")))?;
        let end = offset
            .checked_add(bytes)
            .filter(|end| *end <= available)
            .ok_or_else(|| Error::State(format!("{role} handle range")))?;
        Ok(handle
            .clone()
            .offset_start(offset as u64)
            .offset_end((available - end) as u64))
    }

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

    fn enqueue_gdn_convolution_cohort(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &zerogravity_core::accelerator::program::PlanNode,
        descriptor: &zerogravity_core::accelerator::program::SourceOperator,
        states: &[&StateBinding],
        step: Step,
    ) -> Result<(usize, usize)> {
        let ([packed_qkvz, _packed_ba], [conv_output, z]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN conv SSA arity".into(),
            ));
        };
        if states.len() != 4 || step.rows != 4 || !self.cold().layout().groups().direct() {
            return Err(Error::Unsupported(
                "ordinary CUDA direct four-row GDN conv cohort".into(),
            ));
        }
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let [conv_weights] = self.cold().node_parameters(node_index)? else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN conv parameters".into(),
            ));
        };
        let state_layout = ResidentBufferLayout::new(ResidentPhysicalType::F32, CONV_ELEMENTS)?;
        let service = self.cold().service();
        let client = service.client();
        let cache = self.cold().gdn_internal(GdnRuntimeBuffer::CacheArena)?;
        let status = self.cold().status().handle();
        let mut dispatches = 0usize;

        for (row, state) in states.iter().enumerate() {
            let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
                (&state.prior, &state.candidate)
            else {
                return Err(Error::State(
                    "ordinary CUDA cohort GDN conv state layout".into(),
                ));
            };
            prior.require_layout(state_layout)?;
            candidate.require_layout(state_layout)?;
            let slot = GdnRuntimeCacheSlot::new(row + 1)?;
            let native = plane_handle(cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
            StateTransport::prepare(
                service,
                StateCopy::ConvIn,
                prior.handle.clone(),
                native,
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }

        let full_packed = self.cohort_value_handle(request, *packed_qkvz, inputs[0], step.rows)?;
        let batch_bytes = inputs[0]
            .checked_mul(step.rows)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary CUDA cohort GDN scratch bytes".into()))?;
        let output_scratch = exact_view(
            self.cold().gdn_internal(GdnRuntimeBuffer::HChunks)?,
            0,
            batch_bytes,
            "ordinary GDN batch conv output scratch",
        )?;
        let full_conv = self.cohort_value_handle(request, *conv_output, outputs[0], step.rows)?;
        let full_z = self.cohort_value_handle(request, *z, outputs[1], step.rows)?;
        let indices_lease = self.cold().gdn_internal(GdnRuntimeBuffer::CacheIndices)?;
        let indices = exact_view(indices_lease, 0, 16, "ordinary GDN batch cache indices")?;
        let index_bytes = (1_i32..=4).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
        client.write(&indices, Bytes::from_bytes_vec(index_bytes));
        CachedConvBatch::prepare(4)?.launch(
            client,
            &full_packed,
            &output_scratch,
            &conv_weights.lease().handle(),
            &cache.handle(),
            &indices,
        )?;
        dispatches += 1;

        for (row, state) in states.iter().enumerate() {
            let packed = self.value_handle_at(request, *packed_qkvz, inputs[0], row)?;
            let conv = self.value_handle_at(request, *conv_output, outputs[0], row)?;
            let z_out = self.value_handle_at(request, *z, outputs[1], row)?;
            let conv_source = Self::exact_handle_view(
                &output_scratch,
                row * inputs[0] * 2,
                outputs[0] * 2,
                "ordinary CUDA cohort conv prefix",
            )?;
            let z_source = Self::exact_handle_view(
                &packed,
                outputs[0] * 2,
                outputs[1] * 2,
                "ordinary CUDA cohort z suffix",
            )?;
            dispatches +=
                attention::copy_bf16_prefix(client, &conv_source, &conv, &status, outputs[0])?;
            dispatches +=
                attention::copy_bf16_prefix(client, &z_source, &z_out, &status, outputs[1])?;
            let StateBuffers::Vector(candidate) = &state.candidate else {
                unreachable!();
            };
            let slot = GdnRuntimeCacheSlot::new(row + 1)?;
            let native = plane_handle(cache, slot.convolution_offset(), CONV_STATE_BYTES)?;
            StateTransport::prepare(
                service,
                StateCopy::ConvOut,
                native,
                candidate.handle.clone(),
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }
        dispatches += attention::certify(client, &full_conv, &status, outputs[0] * step.rows)?;
        dispatches += attention::certify(client, &full_z, &status, outputs[1] * step.rows)?;
        Ok((dispatches, 0))
    }

    fn enqueue_gdn_recurrent_cohort(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &zerogravity_core::accelerator::program::PlanNode,
        descriptor: &zerogravity_core::accelerator::program::SourceOperator,
        states: &[&StateBinding],
        step: Step,
    ) -> Result<(usize, usize)> {
        let ([_conv_output, z, packed_ba], [output]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN recurrent SSA arity".into(),
            ));
        };
        if states.len() != 4 || step.rows != 4 || !self.cold().layout().groups().direct() {
            return Err(Error::Unsupported(
                "ordinary CUDA direct four-row GDN recurrent cohort".into(),
            ));
        }
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let [a_log, dt_bias, gamma] = self.cold().node_parameters(node_index)? else {
            return Err(Error::State(
                "ordinary CUDA cohort GDN recurrent parameters".into(),
            ));
        };
        let recurrent_layout =
            ResidentBufferLayout::new(ResidentPhysicalType::F32, RECURRENT_ELEMENTS)?;
        let service = self.cold().service();
        let client = service.client();
        let cache = self.cold().gdn_internal(GdnRuntimeBuffer::CacheArena)?;
        let status = self.cold().status().handle();
        let mut dispatches = 0usize;

        for (row, state) in states.iter().enumerate() {
            let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
                (&state.prior, &state.candidate)
            else {
                return Err(Error::State(
                    "ordinary CUDA cohort GDN recurrent state layout".into(),
                ));
            };
            prior.require_layout(recurrent_layout)?;
            candidate.require_layout(recurrent_layout)?;
            let slot = GdnRuntimeCacheSlot::new(row + 1)?;
            let native = plane_handle(cache, slot.recurrent_offset(), RECURRENT_ELEMENTS * 4)?;
            StateTransport::prepare(
                service,
                StateCopy::RecurrentIn,
                prior.handle.clone(),
                native,
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }

        let rows = step.rows;
        let packed_row_bytes = GdnBuffer::PackedQkvz.bytes(GdnPhase::CachedDecode);
        let conv_scratch_bytes = packed_row_bytes
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch decode conv scratch".into()))?;
        let core_bytes = GdnBuffer::Core
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch decode core scratch".into()))?;
        let rstd_bytes = GdnBuffer::Rstd
            .bytes(GdnPhase::CachedDecode)
            .checked_mul(rows)
            .ok_or_else(|| Error::Limit("ordinary GDN batch gated rstd scratch".into()))?;
        let scratch = self.cold().gdn_internal(GdnRuntimeBuffer::HChunks)?;
        let conv_scratch = exact_view(
            scratch,
            0,
            conv_scratch_bytes,
            "ordinary GDN batch decode conv scratch",
        )?;
        let core_scratch = exact_view(
            scratch,
            conv_scratch_bytes,
            core_bytes,
            "ordinary GDN batch decode core scratch",
        )?;
        let rstd_offset = conv_scratch_bytes
            .checked_add(core_bytes)
            .ok_or_else(|| Error::Limit("ordinary GDN batch rstd offset".into()))?;
        let rstd_scratch = exact_view(
            scratch,
            rstd_offset,
            rstd_bytes,
            "ordinary GDN batch gated rstd scratch",
        )?;
        let full_ba = self.cohort_value_handle(request, *packed_ba, inputs[2], rows)?;
        let ba_total = inputs[2]
            .checked_mul(rows)
            .and_then(|words| words.checked_mul(2))
            .ok_or_else(|| Error::Limit("ordinary GDN batch BA bytes".into()))?;
        let ba_view = ba_total
            .checked_sub(32)
            .ok_or_else(|| Error::Limit("ordinary GDN batch BA view".into()))?;
        let a = Self::exact_handle_view(&full_ba, 32, ba_view, "ordinary GDN batch A")?;
        let beta = Self::exact_handle_view(&full_ba, 0, ba_view, "ordinary GDN batch beta")?;
        let full_z = self.cohort_value_handle(request, *z, inputs[1], rows)?;
        let full_output = self.cohort_value_handle(request, *output, outputs[0], rows)?;
        let recurrent_cache = exact_view(
            cache,
            CONV_STATE_BYTES,
            cache
                .bytes()
                .checked_sub(CONV_STATE_BYTES)
                .ok_or_else(|| Error::Limit("ordinary GDN recurrent cache view".into()))?,
            "ordinary GDN batch recurrent cache",
        )?;
        let indices_lease = self.cold().gdn_internal(GdnRuntimeBuffer::CacheIndices)?;
        let indices = exact_view(indices_lease, 0, 16, "ordinary GDN batch cache indices")?;
        let index_bytes = (1_i32..=4).flat_map(i32::to_le_bytes).collect::<Vec<_>>();
        client.write(&indices, Bytes::from_bytes_vec(index_bytes));

        CachedDecodeBatch::prepare(4)?.launch(
            client,
            &conv_scratch,
            &a,
            &beta,
            &a_log.lease().handle(),
            &dt_bias.lease().handle(),
            &core_scratch,
            &recurrent_cache,
            &indices,
        )?;
        dispatches += 1;
        CachedGatedBatch::prepare(4)?.launch(
            client,
            &core_scratch,
            &full_output,
            &gamma.lease().handle(),
            &full_z,
            &rstd_scratch,
        )?;
        dispatches += 1;
        dispatches += attention::certify(client, &full_output, &status, outputs[0] * rows)?;

        for (row, state) in states.iter().enumerate() {
            let StateBuffers::Vector(candidate) = &state.candidate else {
                unreachable!();
            };
            let slot = GdnRuntimeCacheSlot::new(row + 1)?;
            let native = plane_handle(cache, slot.recurrent_offset(), RECURRENT_ELEMENTS * 4)?;
            StateTransport::prepare(
                service,
                StateCopy::RecurrentOut,
                native,
                candidate.handle.clone(),
                self.cold().status(),
            )?
            .launch(service)?;
            dispatches += 1;
        }
        Ok((dispatches, 0))
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
                if let SourceOperatorKind::CausalConvSplit { domain, .. } = template.kind()
                    && self.cold().layout().groups().direct()
                {
                    let descriptor = template.bind_runtime_call(calls[0])?;
                    let bound = state_maps
                        .iter()
                        .map(|states| {
                            states
                                .get(&StateSlot::Conv(domain))
                                .copied()
                                .ok_or_else(|| {
                                    Error::State("ordinary CUDA cohort conv state absent".into())
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (launched, fused) = self.enqueue_gdn_convolution_cohort(
                        cold,
                        node_index,
                        node,
                        &descriptor,
                        &bound,
                        batch_step,
                    )?;
                    dispatches += launched;
                    fused_dispatches += fused;
                    continue;
                }
                if let SourceOperatorKind::GatedDelta { domain, .. } = template.kind()
                    && self.cold().layout().groups().direct()
                {
                    let descriptor = template.bind_runtime_call(calls[0])?;
                    let bound = state_maps
                        .iter()
                        .map(|states| {
                            states
                                .get(&StateSlot::Recurrent(domain))
                                .copied()
                                .ok_or_else(|| {
                                    Error::State(
                                        "ordinary CUDA cohort recurrent state absent".into(),
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (launched, fused) = self.enqueue_gdn_recurrent_cohort(
                        cold,
                        node_index,
                        node,
                        &descriptor,
                        &bound,
                        batch_step,
                    )?;
                    dispatches += launched;
                    fused_dispatches += fused;
                    continue;
                }
                if let SourceOperatorKind::FlashAttention { domain, .. } = template.kind() {
                    let ([query], [output]) = (node.inputs.as_slice(), node.outputs.as_slice())
                    else {
                        return Err(Error::State(
                            "ordinary CUDA cohort attention SSA arity".into(),
                        ));
                    };
                    let descriptor = template.bind_runtime_call(calls[0])?;
                    let inputs = descriptor.input_elements()?;
                    let outputs = descriptor.output_elements()?;
                    let query =
                        self.cohort_value_handle(cold, *query, inputs[0], batch_step.rows)?;
                    let output =
                        self.cohort_value_handle(cold, *output, outputs[0], batch_step.rows)?;
                    let kv = state_maps
                        .iter()
                        .map(|states| {
                            let state =
                                states.get(&StateSlot::Kv(domain)).copied().ok_or_else(|| {
                                    Error::State(
                                        "ordinary CUDA cohort attention state absent".into(),
                                    )
                                })?;
                            let StateBuffers::Kv { keys, values } = &state.candidate else {
                                return Err(Error::State(
                                    "ordinary CUDA cohort attention KV layout".into(),
                                ));
                            };
                            Ok((keys, values))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    dispatches += self.cold().attention().enqueue_cohort(
                        client,
                        &descriptor,
                        &query,
                        &kv,
                        &output,
                        self.cold().status(),
                    )?;
                    continue;
                }
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
