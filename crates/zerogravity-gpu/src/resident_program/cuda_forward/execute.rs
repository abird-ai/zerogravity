//! Request-local CUDA forward execution under the existing physical flight.
//! No queue/root/completion authority is created here.

use super::{CudaForwardCohortValues, attention, body::CudaForwardBody, exact_view, layout};
use crate::{
    cuda::{
        gdn::{
            GdnRuntimeBuffer, GdnRuntimeCacheSlot, GdnRuntimePartition, GdnRuntimePlan,
            state_transport::{
                CONV_ELEMENTS, RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
            },
        },
        text_forward::{
            Access, BufferRole as ForwardRole, Definition, LaunchBinding, RowExtent, Step,
        },
    },
    resident_leases::{ResidentLease, ResidentSubmission as PhysicalSubmission},
    resident_program::{
        DeviceBuffer, ResidentBufferLayout, ResidentOutput, ResidentPhysicalType,
        ResidentSubmission as ProgramSubmission, StateBinding, StateBindingMode, StateBuffers,
        StateLayout, StateRequirement, cohort_rows::CohortRowSpans,
    },
};
use cubecl::{bytes::Bytes, server::Handle};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, InvocationPhase, PlanNode, PlanOperation, SourceOperator, SourceOperatorKind,
        SourceStateAccess, StateSlot, TextInvocation, ValueId,
    },
    device_events::{BufferRole, DeviceAccess},
};

fn require_not_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(Error::State("CUDA forward request cancelled".into()));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct CohortValues<'a> {
    arena: &'a CudaForwardCohortValues,
    spans: &'a CohortRowSpans,
}

fn require_in_place_kv_alias(state: &StateBinding) -> Result<()> {
    if state.mode != StateBindingMode::KvInPlace {
        return Ok(());
    }
    let (
        StateBuffers::Kv {
            keys: prior_keys,
            values: prior_values,
        },
        StateBuffers::Kv {
            keys: candidate_keys,
            values: candidate_values,
        },
    ) = (&state.prior, &state.candidate)
    else {
        return Err(Error::State(
            "CUDA forward in-place KV binding requires paired KV buffers".into(),
        ));
    };
    for (prior, candidate) in [
        (prior_keys, candidate_keys),
        (prior_values, candidate_values),
    ] {
        let prior = prior.original_lease()?;
        let candidate = candidate.original_lease()?;
        if prior.id() != candidate.id()
            || prior.allocation_generation() != candidate.allocation_generation()
            || prior.bytes() != candidate.bytes()
        {
            return Err(Error::State(
                "CUDA forward in-place KV prior/candidate identity differs".into(),
            ));
        }
    }
    Ok(())
}

fn prepare_kv_candidate(
    client: &cubecl::client::Client,
    status: &Handle,
    state: &StateBinding,
    step: Step,
    kv_width: usize,
) -> Result<usize> {
    require_in_place_kv_alias(state)?;
    let StateBuffers::Kv {
        keys: prior_keys,
        values: prior_values,
    } = &state.prior
    else {
        return Err(Error::State("CUDA forward prior KV layout".into()));
    };
    let StateBuffers::Kv {
        keys: candidate_keys,
        values: candidate_values,
    } = &state.candidate
    else {
        return Err(Error::State("CUDA forward candidate KV layout".into()));
    };
    if state.mode == StateBindingMode::KvInPlace {
        return Ok(0);
    }

    let state_elements = step
        .capacity
        .checked_mul(kv_width)
        .ok_or_else(|| Error::Limit("CUDA forward KV state extent".into()))?;
    let mut launches = 0;
    launches += attention::zero_bf16(client, &candidate_keys.handle, state_elements)?;
    launches += attention::zero_bf16(client, &candidate_values.handle, state_elements)?;
    if step.base != 0 {
        let elements = step
            .base
            .checked_mul(kv_width)
            .ok_or_else(|| Error::Limit("CUDA forward KV prefix element count".into()))?;
        for (prior, candidate, role) in [
            (prior_keys, candidate_keys, "key"),
            (prior_values, candidate_values, "value"),
        ] {
            let source = exact_view(
                prior.original_lease()?,
                0,
                elements * 2,
                &format!("forward prior {role} prefix"),
            )?;
            let destination = exact_view(
                candidate.original_lease()?,
                0,
                elements * 2,
                &format!("forward candidate {role} prefix"),
            )?;
            launches +=
                attention::copy_bf16_prefix(client, &source, &destination, status, elements)?;
        }
    }
    Ok(launches)
}

impl CudaForwardBody {
    #[allow(clippy::too_many_arguments)] // Physical readout/token/cancellation bindings stay explicit.
    pub(in crate::resident_program) async fn execute(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        request: &ForwardPlan,
        states: &[StateBinding],
        output: ResidentOutput,
        compact_greedy: bool,
        device_token: Option<&Handle>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ProgramSubmission> {
        require_not_cancelled(cancelled)?;
        let call = self.validate_request(request)?;
        let state_map = self.validate_states(call, states)?;
        flight.require_accesses(
            self.accesses(output)
                .iter()
                .map(|(lease, access)| (lease, *access)),
        )?;
        for binding in states {
            for buffer in binding.prior.buffers() {
                flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Read)])?;
            }
            for buffer in binding.candidate.buffers() {
                flight.require_accesses([(buffer.original_lease()?, DeviceAccess::Write)])?;
            }
        }
        flight.retain_completion_owner(self.cold().clone())?;
        flight.retain_completion_owner(self.gemm().clone())?;
        // Once a streamed call starts, finish its original flight; cancellation
        // discards the completed candidate rather than abandoning a page owner.
        let cancelled = if self.waits_on_cancel() {
            None
        } else {
            cancelled
        };

        let client = self.cold().service().client();
        if let Some(token) = device_token
            && (call.tokens().len() != 1
                || token.service != client.service_id()
                || token.size_in_used() != 4)
        {
            return Err(Error::State(
                "CUDA forward device-token binding differs".into(),
            ));
        }
        self.clear_and_bind_call(client, call, device_token.is_some())?;
        let step = Step::new(
            call.tokens().len(),
            call.tokens().base(),
            call.tokens().capacity(),
        )?;
        let mut dispatches = 0usize;
        let mut fused_dispatches = 0usize;
        let mut gate_alias = BTreeMap::<u32, u32>::new();

        let nodes = match output {
            ResidentOutput::StateOnly => request.state_nodes(),
            ResidentOutput::Logits => request.nodes(),
        };
        for (node_index, node) in nodes.iter().enumerate() {
            require_not_cancelled(cancelled)?;
            let PlanOperation::SourceOperator {
                descriptor,
                state: declared_state,
            } = &node.operation
            else {
                return Err(Error::State(
                    "CUDA forward request contains non-source operation".into(),
                ));
            };
            if descriptor.invocation() != call {
                return Err(Error::State(
                    "CUDA forward source node call differs from request".into(),
                ));
            }
            let state_ok = match descriptor.state_access() {
                SourceStateAccess::None => declared_state.is_none() && node.state.is_none(),
                SourceStateAccess::Write(slot) => {
                    declared_state.is_none()
                        && node.state.is_some_and(|effect| {
                            effect.read.slot == slot
                                && effect.write.slot == slot
                                && effect.write.ordinal == effect.read.ordinal + 1
                        })
                }
                SourceStateAccess::Read(slot) => {
                    node.state.is_none()
                        && declared_state
                            .is_some_and(|version| version.slot == slot && version.ordinal == 1)
                }
            };
            if !state_ok {
                return Err(Error::State(
                    "CUDA forward source node state/version differs".into(),
                ));
            }
            let (launched, fused) = self
                .enqueue_node(
                    flight,
                    request,
                    node_index,
                    node,
                    descriptor,
                    &state_map,
                    step,
                    0,
                    None,
                    device_token,
                    &mut gate_alias,
                )
                .await?;
            dispatches = dispatches
                .checked_add(launched)
                .ok_or_else(|| Error::Limit("CUDA forward dispatch count".into()))?;
            fused_dispatches = fused_dispatches
                .checked_add(fused)
                .ok_or_else(|| Error::Limit("CUDA forward fused dispatch count".into()))?;
        }
        if let Some(page) = self.cold().source_page() {
            page.finish()?;
        }
        require_not_cancelled(cancelled)?;
        if !gate_alias.is_empty() {
            return Err(Error::State(
                "CUDA forward attention gate alias not consumed".into(),
            ));
        }

        let logits = if output == ResidentOutput::Logits {
            let id = request
                .logits()
                .ok_or_else(|| Error::State("CUDA forward logits value absent".into()))?;
            if compact_greedy {
                let physical = ResidentBufferLayout::new(
                    ResidentPhysicalType::Bf16,
                    self.cold().layout().geometry().vocabulary,
                )?;
                Some(DeviceBuffer::from_lease(
                    self.cold().value_lease(id.0 as usize)?.clone(),
                    physical,
                    BufferRole::Activation,
                    DeviceAccess::Read,
                )?)
            } else {
                let source =
                    self.value_handle(request, id, self.cold().layout().geometry().vocabulary)?;
                let target = self.cold().logits_f32().handle();
                dispatches += attention::widen(
                    client,
                    &source,
                    &target,
                    &self.cold().status().handle(),
                    self.cold().layout().geometry().vocabulary,
                )?;
                Some(DeviceBuffer::from_lease(
                    self.cold().logits_f32().clone(),
                    ResidentBufferLayout::new(
                        ResidentPhysicalType::F32,
                        self.cold().layout().geometry().vocabulary,
                    )?,
                    BufferRole::Output,
                    DeviceAccess::Write,
                )?)
            }
        } else {
            None
        };

        require_not_cancelled(cancelled)?;
        let proposals = request
            .proposed_domains()
            .iter()
            .map(|version| {
                let binding = state_map.get(&version.slot).ok_or_else(|| {
                    Error::State("CUDA forward proposed state binding absent".into())
                })?;
                Ok((*version, binding.candidate.clone()))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ProgramSubmission {
            logits,
            proposals,
            #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
            cohort_proposals: None,
            status: self.cold().status().handle(),
            quant_status: Vec::new(),
            embedding_status: Vec::new(),
            dispatches,
            fused_dispatches,
        })
    }

    pub(in crate::resident_program) fn validate_request<'a>(
        &self,
        request: &'a ForwardPlan,
    ) -> Result<&'a TextInvocation> {
        let cold = self.cold().plan();
        if !request.shares_semantics(cold)
            || request.nodes().len() != cold.nodes().len()
            || request.values().len() != cold.values().len()
            || request.required_domains() != cold.required_domains()
            || request.proposed_domains() != cold.proposed_domains()
        {
            return Err(Error::State(
                "CUDA forward request does not share cold plan semantics".into(),
            ));
        }
        for (actual, template) in request.nodes().iter().zip(cold.nodes()) {
            if !actual.shares_topology(template) {
                return Err(Error::State(
                    "CUDA forward request topology differs from cold plan".into(),
                ));
            }
            if !matches!(actual.operation, PlanOperation::SourceOperator { .. }) {
                return Err(Error::State(
                    "CUDA forward request was not materialized".into(),
                ));
            }
        }
        let call = request
            .text_invocation()
            .ok_or_else(|| Error::State("CUDA forward request text invocation absent".into()))?;
        if call.source() != cold.source()
            || call.tokens().capacity() != cold.prefix_capacity()
            || call.tokens().vocabulary() != self.cold().layout().geometry().vocabulary
            || !(1..=layout::MAX_ROWS).contains(&call.tokens().len())
            || call.tokens().frontier() > cold.prefix_capacity()
            || (call.phase() == InvocationPhase::Decode
                && (call.tokens().base() == 0 || call.tokens().len() != 1))
        {
            return Err(Error::State(
                "CUDA forward request call/source/frontier differs".into(),
            ));
        }
        Ok(call)
    }

    pub(super) fn validate_states<'a>(
        &self,
        call: &TextInvocation,
        states: &'a [StateBinding],
    ) -> Result<BTreeMap<StateSlot, &'a StateBinding>> {
        if states.len() != self.cold().layout().states().len() {
            return Err(Error::State(
                "CUDA forward state binding coverage differs".into(),
            ));
        }
        let cold_ids = self
            .owned_accesses()
            .iter()
            .map(|(lease, _)| lease.id())
            .collect::<BTreeSet<_>>();
        let mut seen = BTreeSet::new();
        let mut map = BTreeMap::new();
        for (requirement, binding) in self.cold().layout().states().iter().zip(states) {
            if binding.slot != requirement.slot || binding.prior_tokens != call.tokens().base() {
                return Err(Error::State(
                    "CUDA forward state slot/frontier differs".into(),
                ));
            }
            let prior = self.state_leases(&binding.prior, requirement)?;
            let candidate = self.state_leases(&binding.candidate, requirement)?;
            if prior.len() != candidate.len() {
                return Err(Error::State(
                    "CUDA forward state prior/candidate plane count differs".into(),
                ));
            }
            let validate = |lease: &ResidentLease| {
                Arc::ptr_eq(lease.service(), self.cold().service())
                    && lease.source_identity() == self.cold().plan().source()
                    && lease.service_generation() == self.cold().service().generation()
                    && lease.role() == BufferRole::RecurrentState
                    && !cold_ids.contains(&lease.id())
            };
            if binding.mode == StateBindingMode::KvInPlace {
                require_in_place_kv_alias(binding)?;
                if !matches!(requirement.layout, StateLayout::Kv { .. })
                    || prior.len() != 2
                    || prior.iter().any(|lease| {
                        !validate(lease) || lease.is_initialized() || !seen.insert(lease.id())
                    })
                {
                    return Err(Error::State(
                        "CUDA forward in-place KV staging ownership differs".into(),
                    ));
                }
            } else {
                for lease in prior.iter().chain(&candidate) {
                    if !validate(lease) || !seen.insert(lease.id()) {
                        return Err(Error::State(
                            "CUDA forward state physical owner aliases/differs".into(),
                        ));
                    }
                }
                if prior.iter().any(|lease| !lease.is_initialized()) {
                    return Err(Error::State(
                        "CUDA forward prior state is not physically initialized".into(),
                    ));
                }
            }
            map.insert(binding.slot, binding);
        }
        Ok(map)
    }

    fn state_leases<'a>(
        &self,
        buffers: &'a StateBuffers,
        requirement: &StateRequirement,
    ) -> Result<Vec<&'a ResidentLease>> {
        let (planes, physical) = requirement.plane_layout()?;
        match (&requirement.layout, buffers) {
            (StateLayout::Vector { .. }, StateBuffers::Vector(value)) if planes == 1 => {
                value.require_layout(physical)?;
                Ok(vec![value.original_lease()?])
            }
            (StateLayout::Kv { .. }, StateBuffers::Kv { keys, values }) if planes == 2 => {
                keys.require_layout(physical)?;
                values.require_layout(physical)?;
                Ok(vec![keys.original_lease()?, values.original_lease()?])
            }
            _ => Err(Error::State(
                "CUDA forward typed state buffer/layout differs".into(),
            )),
        }
    }

    pub(super) fn clear_and_bind_call(
        &self,
        client: &cubecl::client::Client,
        call: &TextInvocation,
        device_token: bool,
    ) -> Result<()> {
        client.write(
            &self.cold().status().handle(),
            Bytes::from_bytes_vec(vec![0u8; 4]),
        );
        // Streamed embedding uses the authenticated host row directly. A
        // committed device token can likewise feed resident embedding without
        // copying its ID back to this scratch buffer.
        if device_token || self.cold().embedding().lease().is_none() {
            return Ok(());
        }
        let ids = call
            .tokens()
            .ids()
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>();
        let tokens = exact_view(
            self.cold().token_ids(),
            0,
            ids.len(),
            "forward request token IDs",
        )?;
        client.write(&tokens, Bytes::from_bytes_vec(ids));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn enqueue_node(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        states: &BTreeMap<StateSlot, &StateBinding>,
        step: Step,
        workspace_row: usize,
        row_positions: Option<&[usize]>,
        device_token: Option<&Handle>,
        gate_alias: &mut BTreeMap<u32, u32>,
    ) -> Result<(usize, usize)> {
        self.enqueue_node_inner(
            flight,
            request,
            node_index,
            node,
            descriptor,
            states,
            step,
            workspace_row,
            row_positions,
            device_token,
            gate_alias,
            None,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn enqueue_node_with_values(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        states: &BTreeMap<StateSlot, &StateBinding>,
        step: Step,
        workspace_row: usize,
        gate_alias: &mut BTreeMap<u32, u32>,
        values: &CudaForwardCohortValues,
        spans: &CohortRowSpans,
    ) -> Result<(usize, usize)> {
        self.enqueue_node_inner(
            flight,
            request,
            node_index,
            node,
            descriptor,
            states,
            step,
            workspace_row,
            None,
            None,
            gate_alias,
            Some(CohortValues {
                arena: values,
                spans,
            }),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn enqueue_node_inner(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        states: &BTreeMap<StateSlot, &StateBinding>,
        step: Step,
        workspace_row: usize,
        row_positions: Option<&[usize]>,
        device_token: Option<&Handle>,
        gate_alias: &mut BTreeMap<u32, u32>,
        values: Option<CohortValues<'_>>,
    ) -> Result<(usize, usize)> {
        use SourceOperatorKind::*;
        let client = self.cold().service().client();
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        if inputs.len() != node.inputs.len() || outputs.len() != node.outputs.len() {
            return Err(Error::State(
                "CUDA forward source operator SSA arity differs".into(),
            ));
        }
        match descriptor.kind() {
            EmbeddingRows => {
                let [output] = node.outputs.as_slice() else {
                    unreachable!()
                };
                if self.cold().embedding().lease().is_none() {
                    let page = self.cold().source_page().ok_or_else(|| {
                        Error::State("CUDA streamed embedding page absent".into())
                    })?;
                    let [matrix] = self.cold().embedding().matrices() else {
                        return Err(Error::State("CUDA streamed embedding source arity".into()));
                    };
                    let target = self.value_lease_in(values, *output)?;
                    let width = self.cold().layout().geometry().hidden_width;
                    let lane_base = if let Some(values) = values {
                        let rows = descriptor.invocation().tokens().len();
                        if outputs[0] % rows != 0 {
                            return Err(Error::State(
                                "streamed cohort embedding row geometry".into(),
                            ));
                        }
                        values
                            .spans
                            .element_span(workspace_row, outputs[0] / rows)?
                            .start
                            .checked_mul(2)
                            .ok_or_else(|| Error::Limit("streamed embedding lane offset".into()))?
                    } else {
                        workspace_row
                            .checked_mul(outputs[0])
                            .and_then(|elements| elements.checked_mul(2))
                            .ok_or_else(|| Error::Limit("streamed embedding lane offset".into()))?
                    };
                    for (row, &token) in descriptor.invocation().tokens().ids().iter().enumerate() {
                        let token = token as usize;
                        let destination = exact_view(
                            target,
                            lane_base + row * width * 2,
                            width * 2,
                            "streamed embedding row",
                        )?;
                        page.with_rows(flight, matrix, token..token + 1, None, |view| {
                            attention::copy_bf16_prefix(
                                client,
                                view.handle(),
                                &destination,
                                &self.cold().status().handle(),
                                width,
                            )?;
                            Ok(vec![target.clone()])
                        })
                        .await?;
                    }
                    return Ok((step.rows, 0));
                }
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                let out = self.value_handle_extent_in(
                    values,
                    request,
                    *output,
                    outputs[0],
                    workspace_row,
                    row_extent,
                )?;
                let tokens = if let Some(values) = values {
                    if device_token.is_some() {
                        return Err(Error::State(
                            "CUDA forward cohort embedding cannot bind a device token".into(),
                        ));
                    }
                    let token_bytes = descriptor
                        .invocation()
                        .tokens()
                        .ids()
                        .iter()
                        .flat_map(|id| id.to_ne_bytes())
                        .collect::<Vec<_>>();
                    let active = values.arena.token_handle(workspace_row, step.rows)?;
                    client.write(&active, Bytes::from_bytes_vec(token_bytes));
                    values.arena.token_handle_bounded(
                        workspace_row,
                        step.rows,
                        row_extent.bound(),
                    )?
                } else {
                    match device_token {
                        Some(token) => {
                            if step.rows != 1 || token.size_in_used() != 4 {
                                return Err(Error::State(
                                    "forward embedding device-token extent".into(),
                                ));
                            }
                            token.clone()
                        }
                        None => exact_view(
                            self.cold().token_ids(),
                            0,
                            step.rows * 4,
                            "forward embedding token IDs",
                        )?,
                    }
                };
                self.enqueue_primitive(
                    self.cold().program().embedding_bounded(row_extent)?,
                    vec![
                        LaunchBinding::new(ForwardRole::TokenIds, Access::Read, tokens),
                        LaunchBinding::new(
                            ForwardRole::EmbeddingTable,
                            Access::Read,
                            self.cold()
                                .embedding()
                                .lease()
                                .ok_or_else(|| {
                                    Error::State("CUDA resident embedding absent".into())
                                })?
                                .handle(),
                        ),
                        LaunchBinding::new(ForwardRole::Output, Access::Write, out),
                    ],
                )?;
                Ok((1, 0))
            }
            Projection { selected_row } => {
                if values.is_some() {
                    return Err(Error::State(
                        "CUDA forward cohort projection must use shared GEMM lowering".into(),
                    ));
                }
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("CUDA forward projection SSA arity".into()));
                };
                let projection = self.cold().projection(node_index)?;
                let input_lease = self.cold().value_lease(input.0 as usize)?;
                let output_lease = self.cold().value_lease(output.0 as usize)?;
                let launches = projection
                    .enqueue(
                        descriptor,
                        self.gemm(),
                        flight,
                        input_lease,
                        output_lease,
                        self.cold().status(),
                        self.cold().source_page(),
                    )
                    .await?;
                Ok((
                    launches,
                    usize::from(selected_row.is_none() && step.rows > 1),
                ))
            }
            Rms(geometry) => {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("CUDA forward RMS SSA arity".into()));
                };
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                let parameters = self.cold().node_parameters(node_index)?;
                let [gamma] = parameters else {
                    return Err(Error::State("CUDA forward RMS gamma arity".into()));
                };
                self.enqueue_primitive(
                    self.cold()
                        .program()
                        .rms_bounded(row_extent, geometry.epsilon)?,
                    vec![
                        LaunchBinding::new(
                            ForwardRole::Input,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *input,
                                inputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::RawGamma,
                            Access::Read,
                            gamma.lease().handle(),
                        ),
                        LaunchBinding::new(
                            ForwardRole::Output,
                            Access::Write,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *output,
                                outputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            ResidualRms(geometry) => {
                let ([input, residual], [output, carried]) =
                    (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("CUDA forward residual RMS SSA arity".into()));
                };
                let residual_lease = self.value_lease_in(values, *residual)?;
                let carried_lease = self.value_lease_in(values, *carried)?;
                if residual_lease.id() != carried_lease.id()
                    || residual_lease.allocation_generation()
                        != carried_lease.allocation_generation()
                {
                    return Err(Error::State(
                        "CUDA forward residual RMS carried owner differs".into(),
                    ));
                }
                let parameters = self.cold().node_parameters(node_index)?;
                let [gamma] = parameters else {
                    return Err(Error::State("CUDA forward residual RMS gamma arity".into()));
                };
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                self.enqueue_primitive(
                    self.cold()
                        .program()
                        .residual_rms_bounded(row_extent, geometry.epsilon)?,
                    vec![
                        LaunchBinding::new(
                            ForwardRole::Input,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *input,
                                inputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::Residual,
                            Access::ReadWrite,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *residual,
                                inputs[1],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::RawGamma,
                            Access::Read,
                            gamma.lease().handle(),
                        ),
                        LaunchBinding::new(
                            ForwardRole::Output,
                            Access::Write,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *output,
                                outputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            QkRmsRotaryGated(geometry) => {
                let ([packed], [query, key, gate]) =
                    (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("CUDA forward QK SSA arity".into()));
                };
                let parameters = self.cold().node_parameters(node_index)?;
                let [query_gamma, key_gamma] = parameters else {
                    return Err(Error::State("CUDA forward QK gamma arity".into()));
                };
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                let packed_bounded = self.value_handle_extent_in(
                    values,
                    request,
                    *packed,
                    inputs[0],
                    workspace_row,
                    row_extent,
                )?;
                let query_bounded = self.value_handle_extent_in(
                    values,
                    request,
                    *query,
                    outputs[0],
                    workspace_row,
                    row_extent,
                )?;
                let key_bounded = self.value_handle_extent_in(
                    values,
                    request,
                    *key,
                    outputs[1],
                    workspace_row,
                    row_extent,
                )?;

                let fused_prefill = row_positions.is_none()
                    && matches!(
                        descriptor.invocation().phase(),
                        zerogravity_core::accelerator::program::InvocationPhase::Prefill
                    );
                let launches = if fused_prefill {
                    self.enqueue_primitive(
                        self.cold().program().query_key_head_rms_rotary_stateless(
                            row_extent.into(),
                            step.base,
                            geometry.epsilon,
                        )?,
                        vec![
                            LaunchBinding::new(
                                ForwardRole::PackedAttention,
                                Access::Read,
                                packed_bounded,
                            ),
                            LaunchBinding::new(
                                ForwardRole::QueryRawGamma,
                                Access::Read,
                                query_gamma.lease().handle(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::KeyRawGamma,
                                Access::Read,
                                key_gamma.lease().handle(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::InverseFrequency,
                                Access::Read,
                                self.cold().inverse_frequency().handle(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::RotaryTables,
                                Access::Read,
                                self.cold().rotary().tables().handle(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::QueryNormalized,
                                Access::Write,
                                query_bounded,
                            ),
                            LaunchBinding::new(
                                ForwardRole::KeyNormalized,
                                Access::Write,
                                key_bounded,
                            ),
                        ],
                    )?;
                    1
                } else {
                    self.enqueue_primitive(
                        self.cold()
                            .program()
                            .query_head_rms_bounded(row_extent, geometry.epsilon)?,
                        vec![
                            LaunchBinding::new(
                                ForwardRole::PackedAttention,
                                Access::Read,
                                packed_bounded.clone(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::QueryRawGamma,
                                Access::Read,
                                query_gamma.lease().handle(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::QueryNormalized,
                                Access::Write,
                                query_bounded,
                            ),
                        ],
                    )?;
                    self.enqueue_primitive(
                        self.cold()
                            .program()
                            .key_head_rms_bounded(row_extent, geometry.epsilon)?,
                        vec![
                            LaunchBinding::new(
                                ForwardRole::PackedAttention,
                                Access::Read,
                                packed_bounded,
                            ),
                            LaunchBinding::new(
                                ForwardRole::KeyRawGamma,
                                Access::Read,
                                key_gamma.lease().handle(),
                            ),
                            LaunchBinding::new(
                                ForwardRole::KeyNormalized,
                                Access::Write,
                                key_bounded,
                            ),
                        ],
                    )?;
                    let rotary_launches = if let Some(positions) = row_positions {
                        if workspace_row != 0 || positions.len() != step.rows {
                            return Err(Error::State(
                                "CUDA forward explicit rotary position coverage".into(),
                            ));
                        }
                        let query_elements = outputs[0]
                            .checked_div(step.rows)
                            .ok_or_else(|| Error::State("CUDA forward query row extent".into()))?;
                        let key_elements = outputs[1]
                            .checked_div(step.rows)
                            .ok_or_else(|| Error::State("CUDA forward key row extent".into()))?;
                        let mut launches = 0;
                        for (row, &position) in positions.iter().enumerate() {
                            launches += self.cold().rotary().enqueue(
                                &self.value_handle_in(
                                    values,
                                    request,
                                    *query,
                                    query_elements,
                                    row,
                                )?,
                                &self.value_handle_in(values, request, *key, key_elements, row)?,
                                self.cold().inverse_frequency(),
                                self.cold().status(),
                                position,
                                1,
                            )?;
                        }
                        launches
                    } else {
                        let query_handle = self.value_handle_in(
                            values,
                            request,
                            *query,
                            outputs[0],
                            workspace_row,
                        )?;
                        let key_handle =
                            self.value_handle_in(values, request, *key, outputs[1], workspace_row)?;
                        self.cold().rotary().enqueue(
                            &query_handle,
                            &key_handle,
                            self.cold().inverse_frequency(),
                            self.cold().status(),
                            step.base,
                            step.rows,
                        )?
                    };
                    2 + rotary_launches
                };

                if gate_alias.insert(gate.0, packed.0).is_some() {
                    return Err(Error::State("CUDA forward QK gate alias duplicated".into()));
                }
                Ok((launches, 0))
            }
            KvAppend { domain, .. } => {
                let ([key, packed], []) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("CUDA forward KV append SSA arity".into()));
                };
                let state = self.state(states, StateSlot::Kv(domain))?;
                let StateBuffers::Kv {
                    keys: candidate_keys,
                    values: candidate_values,
                } = &state.candidate
                else {
                    return Err(Error::State("CUDA forward candidate KV layout".into()));
                };
                let mut launches = prepare_kv_candidate(
                    client,
                    &self.cold().status().handle(),
                    state,
                    step,
                    self.cold().layout().geometry().kv_width()?,
                )?;
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                self.enqueue_primitive(
                    self.cold().program().kv_append_bounded(step, row_extent)?,
                    vec![
                        LaunchBinding::new(
                            ForwardRole::Key,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *key,
                                inputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::PackedAttention,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *packed,
                                inputs[1],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::KeyCache,
                            Access::ReadWrite,
                            candidate_keys.handle.clone(),
                        ),
                        LaunchBinding::new(
                            ForwardRole::ValueCache,
                            Access::ReadWrite,
                            candidate_values.handle.clone(),
                        ),
                    ],
                )?;
                launches += 1;
                Ok((launches, 0))
            }
            Attention { domain, .. } => {
                let ([query], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("CUDA forward attention SSA arity".into()));
                };
                let state = self.state(states, StateSlot::Kv(domain))?;
                let StateBuffers::Kv {
                    keys,
                    values: cached_values,
                } = &state.candidate
                else {
                    return Err(Error::State("CUDA forward attention KV layout".into()));
                };
                let query =
                    self.value_handle_in(values, request, *query, inputs[0], workspace_row)?;
                let output_handle =
                    self.value_handle_in(values, request, *output, outputs[0], workspace_row)?;
                let launches = self.cold().attention().enqueue(
                    client,
                    descriptor,
                    &query,
                    keys,
                    cached_values,
                    &output_handle,
                    self.cold().status(),
                )?;
                Ok((launches, 0))
            }
            Sigmoid { .. } => {
                let ([gate], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("CUDA forward sigmoid SSA arity".into()));
                };
                let packed = gate_alias.remove(&gate.0).map(ValueId).ok_or_else(|| {
                    Error::State("CUDA forward sigmoid gate producer alias absent".into())
                })?;
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                self.enqueue_primitive(
                    self.cold().program().sigmoid_bounded(row_extent)?,
                    vec![
                        LaunchBinding::new(
                            ForwardRole::PackedAttention,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                packed,
                                step.rows
                                    * 2
                                    * (self.cold().layout().geometry().query_width()?
                                        + self.cold().layout().geometry().kv_width()?),
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::Output,
                            Access::Write,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *output,
                                outputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            Multiply { .. } => {
                let ([input, gate], [output]) = (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("CUDA forward multiply SSA arity".into()));
                };
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                self.enqueue_primitive(
                    self.cold().program().multiply_bounded(row_extent)?,
                    vec![
                        LaunchBinding::new(
                            ForwardRole::Input,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *input,
                                inputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::Gate,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *gate,
                                inputs[1],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::Output,
                            Access::Write,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *output,
                                outputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            SiluAndMul { .. } => {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("CUDA forward SiLU SSA arity".into()));
                };
                let row_extent = self.row_extent_in(values, request, step.rows)?;
                self.enqueue_primitive(
                    self.cold().program().silu_up_bounded(row_extent)?,
                    vec![
                        LaunchBinding::new(
                            ForwardRole::PackedGateUp,
                            Access::Read,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *input,
                                inputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                        LaunchBinding::new(
                            ForwardRole::Output,
                            Access::Write,
                            self.value_handle_extent_in(
                                values,
                                request,
                                *output,
                                outputs[0],
                                workspace_row,
                                row_extent,
                            )?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            CausalConvSplit { domain, .. } => self.enqueue_gdn_convolution(
                request,
                node_index,
                node,
                descriptor,
                self.state(states, StateSlot::Conv(domain))?,
                step,
                workspace_row,
                values,
            ),
            GatedDelta { domain, .. } => self.enqueue_gdn_recurrent(
                request,
                node_index,
                node,
                descriptor,
                self.state(states, StateSlot::Recurrent(domain))?,
                step,
                workspace_row,
                values,
            ),
            RetainedRows { .. }
            | CausalConvSiluZero { .. }
            | GroupedQkNormalize(_)
            | StateCarry { .. } => Err(Error::Unsupported(
                "CUDA forward request contains unsupported source operator".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)] // Keep row/state/source bindings explicit.
    fn enqueue_gdn_convolution(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        state: &StateBinding,
        step: Step,
        workspace_row: usize,
        values: Option<CohortValues<'_>>,
    ) -> Result<(usize, usize)> {
        if !self.cold().layout().groups().direct() {
            if values.is_some() {
                return Err(Error::Unsupported(
                    "CUDA forward cohort values require direct GDN geometry".into(),
                ));
            }
            return self.enqueue_grouped_convolution(
                request,
                node_index,
                node,
                descriptor,
                state,
                step,
                workspace_row,
            );
        }
        let ([packed_qkvz, packed_ba], [conv_output, z]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "CUDA forward GDN convolution SSA arity".into(),
            ));
        };
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let parameters = self.cold().node_parameters(node_index)?;
        let [conv_weights] = parameters else {
            return Err(Error::State(
                "CUDA forward GDN convolution parameter arity".into(),
            ));
        };
        let StateBuffers::Vector(prior) = &state.prior else {
            return Err(Error::State(
                "CUDA forward GDN prior conv state layout".into(),
            ));
        };
        let StateBuffers::Vector(candidate) = &state.candidate else {
            return Err(Error::State(
                "CUDA forward GDN candidate conv state layout".into(),
            ));
        };
        prior.require_layout(ResidentBufferLayout::new(
            ResidentPhysicalType::F32,
            CONV_ELEMENTS,
        )?)?;
        candidate.require_layout(ResidentBufferLayout::new(
            ResidentPhysicalType::F32,
            CONV_ELEMENTS,
        )?)?;

        let runtime = self.gdn_plan(step, descriptor.invocation().phase())?;
        self.write_gdn_controls(&runtime)?;
        let cache = self.cold().gdn_internal(GdnRuntimeBuffer::CacheArena)?;
        let view = runtime.convolution_state();
        let native = plane_handle(cache, view.offset(), view.bytes())?;
        StateTransport::prepare(
            self.cold().service(),
            StateCopy::ConvIn,
            prior.handle.clone(),
            native.clone(),
            self.cold().status(),
        )?
        .launch(self.cold().service())?;

        let row_extent =
            self.gdn_row_extent(request, descriptor.invocation().phase(), step.rows)?;
        let mut handles = self.gdn_internal_handles()?;
        handles.insert(
            GdnRuntimeBuffer::PackedQkvz,
            self.value_handle_extent_in(
                values,
                request,
                *packed_qkvz,
                inputs[0],
                workspace_row,
                row_extent,
            )?,
        );
        handles.insert(
            GdnRuntimeBuffer::PackedBa,
            self.value_handle_extent_in(
                values,
                request,
                *packed_ba,
                inputs[1],
                workspace_row,
                row_extent,
            )?,
        );
        let conv_handle =
            self.value_handle_in(values, request, *conv_output, outputs[0], workspace_row)?;
        let z_handle = self.value_handle_in(values, request, *z, outputs[1], workspace_row)?;
        handles.insert(
            GdnRuntimeBuffer::ConvOutput,
            self.value_handle_extent_in(
                values,
                request,
                *conv_output,
                outputs[0],
                workspace_row,
                row_extent,
            )?,
        );
        handles.insert(
            GdnRuntimeBuffer::ZCompact,
            self.value_handle_extent_in(
                values,
                request,
                *z,
                outputs[1],
                workspace_row,
                row_extent,
            )?,
        );
        handles.insert(GdnRuntimeBuffer::ConvWeights, conv_weights.lease().handle());
        let launches = runtime
            .prepare_partition_launches(
                self.cold().service().client(),
                &handles,
                GdnRuntimePartition::Convolution,
            )?
            .launch()?;
        let mut dispatches = 1usize
            .checked_add(launches)
            .ok_or_else(|| Error::Limit("CUDA forward GDN conv dispatch count".into()))?;
        dispatches += attention::certify(
            self.cold().service().client(),
            &conv_handle,
            &self.cold().status().handle(),
            outputs[0],
        )?;
        dispatches += attention::certify(
            self.cold().service().client(),
            &z_handle,
            &self.cold().status().handle(),
            outputs[1],
        )?;
        StateTransport::prepare(
            self.cold().service(),
            StateCopy::ConvOut,
            native,
            candidate.handle.clone(),
            self.cold().status(),
        )?
        .launch(self.cold().service())?;
        dispatches += 1;
        Ok((dispatches, 0))
    }

    #[allow(clippy::too_many_arguments)] // Keep row/state/source bindings explicit.
    fn enqueue_gdn_recurrent(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        state: &StateBinding,
        step: Step,
        workspace_row: usize,
        values: Option<CohortValues<'_>>,
    ) -> Result<(usize, usize)> {
        if !self.cold().layout().groups().direct() {
            if values.is_some() {
                return Err(Error::Unsupported(
                    "CUDA forward cohort values require direct GDN geometry".into(),
                ));
            }
            return self.enqueue_grouped_recurrent(
                request,
                node_index,
                node,
                descriptor,
                state,
                step,
                workspace_row,
            );
        }
        let ([conv_output, z, packed_ba], [output]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State("CUDA forward GDN recurrent SSA arity".into()));
        };
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let parameters = self.cold().node_parameters(node_index)?;
        let [a_log, dt_bias, gamma] = parameters else {
            return Err(Error::State(
                "CUDA forward GDN recurrent parameter arity".into(),
            ));
        };
        let StateBuffers::Vector(prior) = &state.prior else {
            return Err(Error::State(
                "CUDA forward GDN prior recurrent state layout".into(),
            ));
        };
        let StateBuffers::Vector(candidate) = &state.candidate else {
            return Err(Error::State(
                "CUDA forward GDN candidate recurrent state layout".into(),
            ));
        };
        let recurrent_layout =
            ResidentBufferLayout::new(ResidentPhysicalType::F32, RECURRENT_ELEMENTS)?;
        prior.require_layout(recurrent_layout)?;
        candidate.require_layout(recurrent_layout)?;

        let runtime = self.gdn_plan(step, descriptor.invocation().phase())?;
        self.write_gdn_controls(&runtime)?;
        let cache = self.cold().gdn_internal(GdnRuntimeBuffer::CacheArena)?;
        let native = plane_handle(
            cache,
            runtime.slot().recurrent_offset(),
            RECURRENT_ELEMENTS * 4,
        )?;
        StateTransport::prepare(
            self.cold().service(),
            StateCopy::RecurrentIn,
            prior.handle.clone(),
            native.clone(),
            self.cold().status(),
        )?
        .launch(self.cold().service())?;

        let row_extent =
            self.gdn_row_extent(request, descriptor.invocation().phase(), step.rows)?;
        let mut handles = self.gdn_internal_handles()?;
        handles.insert(
            GdnRuntimeBuffer::ConvOutput,
            self.value_handle_extent_in(
                values,
                request,
                *conv_output,
                inputs[0],
                workspace_row,
                row_extent,
            )?,
        );
        handles.insert(
            GdnRuntimeBuffer::ZCompact,
            self.value_handle_extent_in(values, request, *z, inputs[1], workspace_row, row_extent)?,
        );
        handles.insert(
            GdnRuntimeBuffer::PackedBa,
            self.value_handle_extent_in(
                values,
                request,
                *packed_ba,
                inputs[2],
                workspace_row,
                row_extent,
            )?,
        );
        let output_handle =
            self.value_handle_in(values, request, *output, outputs[0], workspace_row)?;
        handles.insert(
            GdnRuntimeBuffer::Gated,
            self.value_handle_extent_in(
                values,
                request,
                *output,
                outputs[0],
                workspace_row,
                row_extent,
            )?,
        );
        handles.insert(GdnRuntimeBuffer::ALog, a_log.lease().handle());
        handles.insert(GdnRuntimeBuffer::DtBias, dt_bias.lease().handle());
        handles.insert(GdnRuntimeBuffer::GatedGamma, gamma.lease().handle());

        let launches = runtime
            .prepare_partition_launches(
                self.cold().service().client(),
                &handles,
                GdnRuntimePartition::Recurrent,
            )?
            .launch()?;
        let mut dispatches = 1usize
            .checked_add(launches)
            .ok_or_else(|| Error::Limit("CUDA forward GDN recurrent dispatch count".into()))?;
        dispatches += attention::certify(
            self.cold().service().client(),
            &output_handle,
            &self.cold().status().handle(),
            outputs[0],
        )?;
        StateTransport::prepare(
            self.cold().service(),
            StateCopy::RecurrentOut,
            native,
            candidate.handle.clone(),
            self.cold().status(),
        )?
        .launch(self.cold().service())?;
        dispatches += 1;
        Ok((dispatches, 0))
    }

    pub(super) fn gdn_plan(&self, step: Step, mode: InvocationPhase) -> Result<GdnRuntimePlan> {
        let slot = GdnRuntimeCacheSlot::new(1)?;
        if self.cold().plan().text_invocation().is_some() {
            return Err(Error::State(
                "CUDA forward cold plan unexpectedly owns a text invocation".into(),
            ));
        }
        let plan = match mode {
            InvocationPhase::Decode => {
                GdnRuntimePlan::cached_decode(layout::gdn_runtime_config()?, slot, step.base)?
            }
            InvocationPhase::Prefill => GdnRuntimePlan::prefill_chunk(
                layout::gdn_runtime_config()?,
                slot,
                step.base,
                step.rows,
                step.base != 0,
            )?,
        };
        if plan.workspace() != self.cold().layout().gdn() {
            return Err(Error::State(
                "CUDA forward GDN request workspace differs from cold layout".into(),
            ));
        }
        Ok(plan)
    }

    pub(super) fn write_gdn_controls(&self, plan: &GdnRuntimePlan) -> Result<()> {
        for control in plan.controls() {
            if self
                .cold()
                .layout()
                .gdn()
                .retained_control_bytes(control.role())
                .is_some()
            {
                if control.retained_offset(self.cold().layout().gdn()) != Some(0) {
                    return Err(Error::State(
                        "CUDA forward GDN control is not retained at its bound workspace offset"
                            .into(),
                    ));
                }
                continue;
            }
            let lease = self.cold().gdn_internal(control.role())?;
            let handle = exact_view(lease, 0, control.bytes().len(), "forward GDN control")?;
            self.cold()
                .service()
                .client()
                .write(&handle, Bytes::from_bytes_vec(control.bytes().to_vec()));
        }
        Ok(())
    }

    pub(super) fn gdn_internal_handles(&self) -> Result<BTreeMap<GdnRuntimeBuffer, Handle>> {
        let mut handles = BTreeMap::new();
        for layout in self.cold().layout().gdn().buffers() {
            if let Ok(lease) = self.cold().gdn_internal(layout.role()) {
                handles.insert(layout.role(), lease.handle());
            }
        }
        Ok(handles)
    }

    fn state<'a>(
        &self,
        states: &'a BTreeMap<StateSlot, &'a StateBinding>,
        slot: StateSlot,
    ) -> Result<&'a StateBinding> {
        states
            .get(&slot)
            .copied()
            .ok_or_else(|| Error::State("CUDA forward state slot binding absent".into()))
    }

    fn value_lease_in<'a>(
        &'a self,
        values: Option<CohortValues<'a>>,
        id: ValueId,
    ) -> Result<&'a ResidentLease> {
        match values {
            Some(values) => values.arena.lease(id),
            None => self.cold().value_lease(id.0 as usize),
        }
    }

    fn value_handle_in(
        &self,
        values: Option<CohortValues<'_>>,
        request: &ForwardPlan,
        id: ValueId,
        elements: usize,
        lane: usize,
    ) -> Result<Handle> {
        let shape = request
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("CUDA forward value ID absent".into()))?;
        if elements == 0 || elements > shape.capacity_elements()? {
            return Err(Error::State(
                "CUDA forward active value extent exceeds static capacity".into(),
            ));
        }
        match values {
            Some(values) => {
                let rows = values.spans.lane_rows(lane)?;
                if !elements.is_multiple_of(rows) {
                    return Err(Error::State(
                        "CUDA forward cohort value is not row-proportional".into(),
                    ));
                }
                values
                    .arena
                    .lane_row_handle(id, elements / rows, values.spans, lane)
            }
            None => self.value_handle_at(request, id, elements, lane),
        }
    }

    fn row_extent_in(
        &self,
        values: Option<CohortValues<'_>>,
        request: &ForwardPlan,
        active_rows: usize,
    ) -> Result<RowExtent> {
        let bound_rows = if values.is_some() {
            request
                .source_plan_geometry()
                .ok_or_else(|| Error::State("CUDA forward cohort source geometry absent".into()))?
                .max_rows()
        } else {
            active_rows
        };
        RowExtent::bounded(active_rows, bound_rows)
    }

    fn gdn_row_extent(
        &self,
        request: &ForwardPlan,
        mode: InvocationPhase,
        active_rows: usize,
    ) -> Result<RowExtent> {
        match mode {
            InvocationPhase::Prefill => {
                let bound_rows = request
                    .source_plan_geometry()
                    .ok_or_else(|| Error::State("CUDA forward GDN source geometry absent".into()))?
                    .max_rows();
                RowExtent::bounded(active_rows, bound_rows)
            }
            InvocationPhase::Decode => RowExtent::exact(active_rows),
        }
    }

    fn value_handle_extent_in(
        &self,
        values: Option<CohortValues<'_>>,
        request: &ForwardPlan,
        id: ValueId,
        active_elements: usize,
        lane: usize,
        rows: RowExtent,
    ) -> Result<Handle> {
        if active_elements == 0 || !active_elements.is_multiple_of(rows.active()) {
            return Err(Error::State(
                "CUDA forward bounded value is not row-proportional".into(),
            ));
        }
        let shape = request
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("CUDA forward bounded value ID absent".into()))?;
        let elements_per_row = active_elements / rows.active();
        let bound_elements = elements_per_row
            .checked_mul(rows.bound())
            .ok_or_else(|| Error::Limit("CUDA forward bounded value extent".into()))?;
        if active_elements > shape.capacity_elements()?
            || bound_elements > shape.capacity_elements()?
        {
            return Err(Error::State(
                "CUDA forward bounded value exceeds static capacity".into(),
            ));
        }
        match values {
            Some(values) => {
                if values.spans.lane_rows(lane)? != rows.active() {
                    return Err(Error::State(
                        "CUDA forward bounded lane active rows differ".into(),
                    ));
                }
                values.arena.lane_row_bounded_handle(
                    id,
                    elements_per_row,
                    values.spans,
                    lane,
                    rows.bound(),
                )
            }
            None if rows.active() == rows.bound() => {
                self.value_handle_at(request, id, active_elements, lane)
            }
            None if lane == 0 => {
                let bytes = bound_elements
                    .checked_mul(2)
                    .ok_or_else(|| Error::Limit("CUDA forward bounded BF16 bytes".into()))?;
                exact_view(
                    self.cold().value_lease(id.0 as usize)?,
                    0,
                    bytes,
                    "forward bounded BF16 value",
                )
            }
            None => Err(Error::State(
                "CUDA forward wider singleton value requires row zero".into(),
            )),
        }
    }

    pub(super) fn value_handle(
        &self,
        request: &ForwardPlan,
        id: ValueId,
        elements: usize,
    ) -> Result<Handle> {
        self.value_handle_at(request, id, elements, 0)
    }

    pub(super) fn value_handle_at(
        &self,
        request: &ForwardPlan,
        id: ValueId,
        elements: usize,
        row: usize,
    ) -> Result<Handle> {
        let shape = request
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("CUDA forward value ID absent".into()))?;
        let start = row
            .checked_mul(elements)
            .ok_or_else(|| Error::Limit("CUDA forward active row offset".into()))?;
        let end = start
            .checked_add(elements)
            .ok_or_else(|| Error::Limit("CUDA forward active row extent".into()))?;
        if elements == 0 || end > shape.capacity_elements()? {
            return Err(Error::State(
                "CUDA forward active row extent exceeds static capacity".into(),
            ));
        }
        let offset = start
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("CUDA forward active BF16 offset".into()))?;
        let bytes = elements
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("CUDA forward active BF16 bytes".into()))?;
        exact_view(
            self.cold().value_lease(id.0 as usize)?,
            offset,
            bytes,
            "forward active BF16 value",
        )
    }

    pub(super) fn enqueue_primitive(
        &self,
        definition: Definition,
        bindings: Vec<LaunchBinding>,
    ) -> Result<()> {
        let client = self.cold().service().client();
        let status = LaunchBinding::new(
            ForwardRole::Status,
            Access::ReadWrite,
            self.cold().status().handle(),
        );
        let prepared = definition.prepare_owned(client, bindings, status)?;
        prepared.enqueue_owned(client, &self.cold().status().handle())
    }
}

#[cfg(all(
    test,
    feature = "cuda-native-qualification",
    feature = "cuda-cublas",
    target_os = "linux"
))]
mod in_place_tests {
    use super::*;
    use crate::cuda::{
        test_support::CudaDiagnostic,
        text_forward::{
            Access as ForwardAccess, BufferRole as ForwardRole, Geometry, LaunchBinding, Program,
        },
    };
    use cubecl::CubeElement;
    use zerogravity_core::{
        accelerator::program::KvDomain,
        device_events::{BufferRole, DeviceAccess},
    };

    fn bf16_bytes(words: &[u16]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    #[ignore = "physical CUDA forward KV in-place staging preserves gathered prefix"]
    fn physical_forward_kv_in_place_staging() -> Result<()> {
        futures_lite::future::block_on(async {
            const CAPACITY: usize = 8;
            const BASE: usize = 3;
            const ROWS: usize = 1;
            const PACKED_WIDTH: usize = 5120;
            const VALUE_OFFSET: usize = 4608;

            let diagnostic = CudaDiagnostic::new("forward-kv-in-place")?;
            let service = diagnostic.service.clone();
            let client = service.client();
            let geometry = Geometry::qwen35_08();
            let kv_width = geometry.kv_width()?;
            let plane_elements = CAPACITY
                .checked_mul(kv_width)
                .ok_or_else(|| Error::Limit("forward in-place KV plane elements".into()))?;
            let plane_bytes = plane_elements
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("forward in-place KV plane bytes".into()))?;
            let physical = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, plane_elements)?;

            let key_lease =
                service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
            let value_lease =
                service.allocate_uninitialized(plane_bytes, BufferRole::RecurrentState)?;
            if key_lease.is_initialized() || value_lease.is_initialized() {
                return Err(Error::State(
                    "forward in-place KV staging was already initialized".into(),
                ));
            }
            let key_buffer = DeviceBuffer::from_lease(
                key_lease.clone(),
                physical,
                BufferRole::RecurrentState,
                DeviceAccess::Write,
            )?;
            let value_buffer = DeviceBuffer::from_lease(
                value_lease.clone(),
                physical,
                BufferRole::RecurrentState,
                DeviceAccess::Write,
            )?;
            let staging = StateBuffers::Kv {
                keys: key_buffer,
                values: value_buffer,
            };
            let state = StateBinding {
                slot: StateSlot::Kv(KvDomain(0)),
                prior: staging.clone(),
                candidate: staging,
                prior_tokens: BASE,
                mode: StateBindingMode::KvInPlace,
            };
            require_in_place_kv_alias(&state)?;

            let mut initial_keys = vec![0u16; plane_elements];
            let mut initial_values = vec![0u16; plane_elements];
            for token in 0..BASE {
                for lane in 0..kv_width {
                    initial_keys[token * kv_width + lane] =
                        ((token * 17 + lane * 3 + 1) % 251) as u16;
                    initial_values[token * kv_width + lane] =
                        ((token * 19 + lane * 5 + 7) % 251) as u16;
                }
            }
            let new_keys = (0..kv_width)
                .map(|lane| ((lane * 11 + 29) % 251) as u16)
                .collect::<Vec<_>>();
            let new_values = (0..kv_width)
                .map(|lane| ((lane * 13 + 37) % 251) as u16)
                .collect::<Vec<_>>();
            let mut packed = vec![0u16; PACKED_WIDTH];
            packed[VALUE_OFFSET..VALUE_OFFSET + kv_width].copy_from_slice(&new_values);

            let accesses = vec![
                (key_lease.clone(), DeviceAccess::Write),
                (value_lease.clone(), DeviceAccess::Write),
            ];
            let flight = service.submit(0, &accesses)?;
            flight.require_accesses([
                (&key_lease, DeviceAccess::Write),
                (&value_lease, DeviceAccess::Write),
            ])?;
            client.write(
                &key_lease.handle(),
                Bytes::from_bytes_vec(bf16_bytes(&initial_keys)),
            );
            client.write(
                &value_lease.handle(),
                Bytes::from_bytes_vec(bf16_bytes(&initial_values)),
            );

            let step = Step::new(ROWS, BASE, CAPACITY)?;
            let status = client.create_from_slice(u32::as_bytes(&[0u32]));
            let preparation_launches =
                prepare_kv_candidate(client, &status, &state, step, kv_width)?;
            if preparation_launches != 0 {
                return Err(Error::State(
                    "forward in-place KV preparation launched zero/copy work".into(),
                ));
            }

            let program = Program::new(geometry)?;
            let key = client.create_from_slice(&bf16_bytes(&new_keys));
            let packed = client.create_from_slice(&bf16_bytes(&packed));
            let prepared = program.kv_append(step)?.prepare_owned(
                client,
                vec![
                    LaunchBinding::new(ForwardRole::Key, ForwardAccess::Read, key),
                    LaunchBinding::new(ForwardRole::PackedAttention, ForwardAccess::Read, packed),
                    LaunchBinding::new(
                        ForwardRole::KeyCache,
                        ForwardAccess::ReadWrite,
                        key_lease.handle(),
                    ),
                    LaunchBinding::new(
                        ForwardRole::ValueCache,
                        ForwardAccess::ReadWrite,
                        value_lease.handle(),
                    ),
                ],
                LaunchBinding::new(
                    ForwardRole::Status,
                    ForwardAccess::ReadWrite,
                    status.clone(),
                ),
            )?;
            prepared.enqueue_owned(client, &status)?;

            let mut expected_keys = initial_keys.clone();
            let mut expected_values = initial_values.clone();
            expected_keys[BASE * kv_width..(BASE + ROWS) * kv_width].copy_from_slice(&new_keys);
            expected_values[BASE * kv_width..(BASE + ROWS) * kv_width].copy_from_slice(&new_values);
            let outputs = vec![key_lease.clone(), value_lease.clone()];
            let readbacks = vec![
                key_lease.readback(0, plane_bytes)?,
                value_lease.readback(0, plane_bytes)?,
            ];
            let (ready, data) = flight
                .complete_with_readback(&outputs, &readbacks, |cohort| {
                    if cohort.len() != 2
                        || &cohort[0][..] != bf16_bytes(&expected_keys).as_slice()
                        || &cohort[1][..] != bf16_bytes(&expected_values).as_slice()
                    {
                        return Err(Error::State(
                            "forward in-place KV staging contents differ".into(),
                        ));
                    }
                    Ok(())
                })
                .await?;
            let status_bytes = client
                .read_one(status)
                .map_err(|error| Error::State(format!("forward in-place KV status: {error}")))?;
            if status_bytes.len() != 4
                || u32::from_le_bytes(status_bytes[..4].try_into().unwrap()) != 0
                || outputs.iter().any(|lease| !lease.is_initialized())
            {
                return Err(Error::State(
                    "forward in-place KV completion/status differs".into(),
                ));
            }
            drop(ready);
            drop(data);
            drop(readbacks);
            drop(outputs);
            drop(state);
            drop(accesses);
            drop(key_lease);
            drop(value_lease);
            if service.resident_bytes()? != 0 {
                return Err(Error::State(
                    "forward in-place KV staging did not retire".into(),
                ));
            }
            println!(
                "P12_FORWARD_KV_INPLACE_PASS base={BASE} rows={ROWS} capacity={CAPACITY} kv_width={kv_width} preparation_launches=0 prefix_preserved=true new_row_exact=true inactive_tail_zero=true unique_write_access=true all_staging_retired=true"
            );
            Ok(())
        })
    }
}
