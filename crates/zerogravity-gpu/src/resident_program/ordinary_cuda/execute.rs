//! Request-local ordinary CUDA execution under the existing physical flight.
//! No queue/root/completion authority is created here.

use super::{attention, body::OrdinaryCudaBody, exact_view, layout};
use crate::{
    cuda::{
        gdn::{
            GdnRuntimeBuffer, GdnRuntimeCacheSlot, GdnRuntimePartition, GdnRuntimePlan,
            state_transport::{
                CONV_ELEMENTS, RECURRENT_ELEMENTS, StateCopy, StateTransport, plane_handle,
            },
        },
        ordinary_text::{Access, BufferRole as OrdinaryRole, Definition, LaunchBinding, Step},
    },
    resident_leases::{ResidentLease, ResidentSubmission as PhysicalSubmission},
    resident_program::{
        DeviceBuffer, ResidentBufferLayout, ResidentOutput, ResidentPhysicalType,
        ResidentSubmission as ProgramSubmission, StateBinding, StateBuffers, StateLayout,
        StateRequirement,
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
        ForwardPlan, PlanNode, PlanOperation, SourceCall, SourceCallMode, SourceOperator,
        SourceOperatorKind, SourceStateAccess, StateSlot, ValueId,
    },
    device_events::{BufferRole, DeviceAccess},
};

fn require_not_cancelled(cancelled: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
        return Err(Error::State("ordinary CUDA request cancelled".into()));
    }
    Ok(())
}

impl OrdinaryCudaBody {
    pub(in crate::resident_program) fn execute(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        request: &ForwardPlan,
        states: &[StateBinding],
        output: ResidentOutput,
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
        flight.retain_completion_owner(self.blas().clone())?;

        let client = self.cold().service().client();
        self.clear_and_bind_call(client, call)?;
        let step = Step::new(
            call.rows().len(),
            call.rows().base(),
            call.rows().capacity(),
        )?;
        let mut dispatches = self.enqueue_positions(client, step)?;
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
                    "ordinary CUDA request contains non-source operation".into(),
                ));
            };
            if descriptor.call() != call {
                return Err(Error::State(
                    "ordinary CUDA source node call differs from request".into(),
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
                    "ordinary CUDA source node state/version differs".into(),
                ));
            }
            let (launched, fused) = self.enqueue_node(
                flight,
                request,
                node_index,
                node,
                descriptor,
                &state_map,
                step,
                &mut gate_alias,
            )?;
            dispatches = dispatches
                .checked_add(launched)
                .ok_or_else(|| Error::Limit("ordinary CUDA dispatch count".into()))?;
            fused_dispatches = fused_dispatches
                .checked_add(fused)
                .ok_or_else(|| Error::Limit("ordinary CUDA fused dispatch count".into()))?;
        }
        require_not_cancelled(cancelled)?;
        if !gate_alias.is_empty() {
            return Err(Error::State(
                "ordinary CUDA attention gate alias not consumed".into(),
            ));
        }

        let logits = if output == ResidentOutput::Logits {
            let id = request
                .logits()
                .ok_or_else(|| Error::State("ordinary CUDA logits value absent".into()))?;
            let source = self.value_handle(request, id, layout::VOCABULARY)?;
            let target = self.cold().logits_f32().handle();
            dispatches += attention::widen(
                client,
                &source,
                &target,
                &self.cold().status().handle(),
                layout::VOCABULARY,
            )?;
            Some(DeviceBuffer::from_lease(
                self.cold().logits_f32().clone(),
                ResidentBufferLayout::new(ResidentPhysicalType::F32, layout::VOCABULARY)?,
                BufferRole::Output,
                DeviceAccess::Write,
            )?)
        } else {
            None
        };

        require_not_cancelled(cancelled)?;
        let proposals = request
            .proposed_domains()
            .iter()
            .map(|version| {
                let binding = state_map.get(&version.slot).ok_or_else(|| {
                    Error::State("ordinary CUDA proposed state binding absent".into())
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
    ) -> Result<&'a SourceCall> {
        let cold = self.cold().plan();
        if !request.shares_semantics(cold)
            || request.nodes().len() != cold.nodes().len()
            || request.values().len() != cold.values().len()
            || request.required_domains() != cold.required_domains()
            || request.proposed_domains() != cold.proposed_domains()
        {
            return Err(Error::State(
                "ordinary CUDA request does not share cold plan semantics".into(),
            ));
        }
        for (actual, template) in request.nodes().iter().zip(cold.nodes()) {
            if !actual.shares_topology(template) {
                return Err(Error::State(
                    "ordinary CUDA request topology differs from cold plan".into(),
                ));
            }
            if !matches!(actual.operation, PlanOperation::SourceOperator { .. }) {
                return Err(Error::State(
                    "ordinary CUDA request was not materialized".into(),
                ));
            }
        }
        let call = request
            .source_call()
            .ok_or_else(|| Error::State("ordinary CUDA request source call absent".into()))?;
        if call.source() != cold.source()
            || call.rows().capacity() != cold.prefix_capacity()
            || call.rows().vocabulary() != layout::VOCABULARY
            || !(1..=layout::MAX_ROWS).contains(&call.rows().len())
            || call.rows().frontier() > cold.prefix_capacity()
            || (call.mode() == SourceCallMode::Decode
                && (call.rows().base() == 0 || call.rows().len() != 1))
        {
            return Err(Error::State(
                "ordinary CUDA request call/source/frontier differs".into(),
            ));
        }
        Ok(call)
    }

    fn validate_states<'a>(
        &self,
        call: &SourceCall,
        states: &'a [StateBinding],
    ) -> Result<BTreeMap<StateSlot, &'a StateBinding>> {
        if states.len() != self.cold().layout().states().len() {
            return Err(Error::State(
                "ordinary CUDA state binding coverage differs".into(),
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
            if binding.slot != requirement.slot || binding.prior_tokens != call.rows().base() {
                return Err(Error::State(
                    "ordinary CUDA state slot/frontier differs".into(),
                ));
            }
            let prior = self.state_leases(&binding.prior, requirement)?;
            let candidate = self.state_leases(&binding.candidate, requirement)?;
            if prior.len() != candidate.len() {
                return Err(Error::State(
                    "ordinary CUDA state prior/candidate plane count differs".into(),
                ));
            }
            for lease in prior.iter().chain(&candidate) {
                if !Arc::ptr_eq(lease.service(), self.cold().service())
                    || lease.source_identity() != self.cold().plan().source()
                    || lease.service_generation() != self.cold().service().generation()
                    || lease.role() != BufferRole::RecurrentState
                    || cold_ids.contains(&lease.id())
                    || !seen.insert(lease.id())
                {
                    return Err(Error::State(
                        "ordinary CUDA state physical owner aliases/differs".into(),
                    ));
                }
            }
            if prior.iter().any(|lease| !lease.is_initialized()) {
                return Err(Error::State(
                    "ordinary CUDA prior state is not physically initialized".into(),
                ));
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
                "ordinary CUDA typed state buffer/layout differs".into(),
            )),
        }
    }

    fn clear_and_bind_call(
        &self,
        client: &cubecl::client::Client,
        call: &SourceCall,
    ) -> Result<()> {
        client.write(
            &self.cold().status().handle(),
            Bytes::from_bytes_vec(vec![0u8; 4]),
        );
        let ids = call
            .rows()
            .ids()
            .iter()
            .flat_map(|id| id.to_le_bytes())
            .collect::<Vec<_>>();
        let tokens = exact_view(
            self.cold().token_ids(),
            0,
            ids.len(),
            "ordinary request token IDs",
        )?;
        client.write(&tokens, Bytes::from_bytes_vec(ids));
        Ok(())
    }

    fn enqueue_positions(&self, client: &cubecl::client::Client, step: Step) -> Result<usize> {
        let positions = exact_view(
            self.cold().positions(),
            0,
            step.rows * 4,
            "ordinary request positions",
        )?;
        self.enqueue_primitive(
            self.cold().program().positions(step)?,
            vec![LaunchBinding::new(
                OrdinaryRole::Positions,
                Access::Write,
                positions,
            )],
        )?;
        let _ = client;
        Ok(1)
    }

    #[allow(clippy::too_many_arguments)]
    fn enqueue_node(
        &self,
        flight: &mut PhysicalSubmission<'_>,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        states: &BTreeMap<StateSlot, &StateBinding>,
        step: Step,
        gate_alias: &mut BTreeMap<u32, u32>,
    ) -> Result<(usize, usize)> {
        use SourceOperatorKind::*;
        let client = self.cold().service().client();
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        if inputs.len() != node.inputs.len() || outputs.len() != node.outputs.len() {
            return Err(Error::State(
                "ordinary CUDA source operator SSA arity differs".into(),
            ));
        }
        match descriptor.kind() {
            EmbeddingRows => {
                let [output] = node.outputs.as_slice() else {
                    unreachable!()
                };
                let out = self.value_handle(request, *output, outputs[0])?;
                let tokens = exact_view(
                    self.cold().token_ids(),
                    0,
                    step.rows * 4,
                    "ordinary embedding token IDs",
                )?;
                self.enqueue_primitive(
                    self.cold().program().embedding(step.rows)?,
                    vec![
                        LaunchBinding::new(OrdinaryRole::TokenIds, Access::Read, tokens),
                        LaunchBinding::new(
                            OrdinaryRole::EmbeddingTable,
                            Access::Read,
                            self.cold().embedding().lease().handle(),
                        ),
                        LaunchBinding::new(OrdinaryRole::Output, Access::Write, out),
                    ],
                )?;
                Ok((1, 0))
            }
            Projection { selected_row } => {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("ordinary CUDA projection SSA arity".into()));
                };
                let projection = self.cold().projection(node_index)?;
                let input_lease = self.cold().value_lease(input.0 as usize)?;
                let output_lease = self.cold().value_lease(output.0 as usize)?;
                projection.enqueue(descriptor, self.blas(), flight, input_lease, output_lease)?;
                let output_handle = self.value_handle(request, *output, outputs[0])?;
                let mut launches = 1usize;
                launches += attention::certify(
                    client,
                    &output_handle,
                    &self.cold().status().handle(),
                    outputs[0],
                )?;
                Ok((
                    launches,
                    usize::from(selected_row.is_none() && step.rows > 1),
                ))
            }
            Rms(geometry) => {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("ordinary CUDA RMS SSA arity".into()));
                };
                let parameters = self.cold().node_parameters(node_index)?;
                let [gamma] = parameters else {
                    return Err(Error::State("ordinary CUDA RMS gamma arity".into()));
                };
                self.enqueue_primitive(
                    self.cold().program().rms(step.rows, geometry.epsilon)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::Input,
                            Access::Read,
                            self.value_handle(request, *input, inputs[0])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::RawGamma,
                            Access::Read,
                            gamma.lease().handle(),
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Output,
                            Access::Write,
                            self.value_handle(request, *output, outputs[0])?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            ResidualRms(geometry) => {
                let ([input, residual], [output, carried]) =
                    (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("ordinary CUDA residual RMS SSA arity".into()));
                };
                let residual_lease = self.cold().value_lease(residual.0 as usize)?;
                let carried_lease = self.cold().value_lease(carried.0 as usize)?;
                if residual_lease.id() != carried_lease.id()
                    || residual_lease.allocation_generation()
                        != carried_lease.allocation_generation()
                {
                    return Err(Error::State(
                        "ordinary CUDA residual RMS carried owner differs".into(),
                    ));
                }
                let parameters = self.cold().node_parameters(node_index)?;
                let [gamma] = parameters else {
                    return Err(Error::State(
                        "ordinary CUDA residual RMS gamma arity".into(),
                    ));
                };
                self.enqueue_primitive(
                    self.cold()
                        .program()
                        .residual_rms(step.rows, geometry.epsilon)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::Input,
                            Access::Read,
                            self.value_handle(request, *input, inputs[0])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Residual,
                            Access::ReadWrite,
                            self.value_handle(request, *residual, inputs[1])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::RawGamma,
                            Access::Read,
                            gamma.lease().handle(),
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Output,
                            Access::Write,
                            self.value_handle(request, *output, outputs[0])?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            QkRmsRotaryGated(geometry) => {
                let ([packed], [query, key, gate]) =
                    (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("ordinary CUDA QK SSA arity".into()));
                };
                let parameters = self.cold().node_parameters(node_index)?;
                let [query_gamma, key_gamma] = parameters else {
                    return Err(Error::State("ordinary CUDA QK gamma arity".into()));
                };
                let packed_handle = self.value_handle(request, *packed, inputs[0])?;
                let query_handle = self.value_handle(request, *query, outputs[0])?;
                let key_handle = self.value_handle(request, *key, outputs[1])?;
                self.enqueue_primitive(
                    self.cold()
                        .program()
                        .query_head_rms(step.rows, geometry.epsilon)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::PackedAttention,
                            Access::Read,
                            packed_handle.clone(),
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::QueryRawGamma,
                            Access::Read,
                            query_gamma.lease().handle(),
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::QueryNormalized,
                            Access::Write,
                            query_handle.clone(),
                        ),
                    ],
                )?;
                self.enqueue_primitive(
                    self.cold()
                        .program()
                        .key_head_rms(step.rows, geometry.epsilon)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::PackedAttention,
                            Access::Read,
                            packed_handle,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::KeyRawGamma,
                            Access::Read,
                            key_gamma.lease().handle(),
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::KeyNormalized,
                            Access::Write,
                            key_handle.clone(),
                        ),
                    ],
                )?;
                let positions = exact_view(
                    self.cold().positions(),
                    0,
                    step.rows * 4,
                    "ordinary RoPE positions",
                )?;
                self.enqueue_primitive(
                    self.cold().program().rope(step)?,
                    vec![
                        LaunchBinding::new(OrdinaryRole::Query, Access::ReadWrite, query_handle),
                        LaunchBinding::new(OrdinaryRole::Key, Access::ReadWrite, key_handle),
                        LaunchBinding::new(OrdinaryRole::Positions, Access::Read, positions),
                        LaunchBinding::new(
                            OrdinaryRole::InverseFrequency,
                            Access::Read,
                            self.cold().inverse_frequency().handle(),
                        ),
                    ],
                )?;
                if gate_alias.insert(gate.0, packed.0).is_some() {
                    return Err(Error::State(
                        "ordinary CUDA QK gate alias duplicated".into(),
                    ));
                }
                Ok((3, 0))
            }
            KvAppend { domain, .. } => {
                let ([key, packed], []) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("ordinary CUDA KV append SSA arity".into()));
                };
                let state = self.state(states, StateSlot::Kv(domain))?;
                let StateBuffers::Kv {
                    keys: prior_keys,
                    values: prior_values,
                } = &state.prior
                else {
                    return Err(Error::State("ordinary CUDA prior KV layout".into()));
                };
                let StateBuffers::Kv {
                    keys: candidate_keys,
                    values: candidate_values,
                } = &state.candidate
                else {
                    return Err(Error::State("ordinary CUDA candidate KV layout".into()));
                };
                let mut launches = 0;
                if step.base != 0 {
                    let elements = step.base.checked_mul(512).ok_or_else(|| {
                        Error::Limit("ordinary CUDA KV prefix element count".into())
                    })?;
                    for (prior, candidate, role) in [
                        (prior_keys, candidate_keys, "key"),
                        (prior_values, candidate_values, "value"),
                    ] {
                        let source = exact_view(
                            prior.original_lease()?,
                            0,
                            elements * 2,
                            &format!("ordinary prior {role} prefix"),
                        )?;
                        let destination = exact_view(
                            candidate.original_lease()?,
                            0,
                            elements * 2,
                            &format!("ordinary candidate {role} prefix"),
                        )?;
                        launches += attention::copy_bf16_prefix(
                            client,
                            &source,
                            &destination,
                            &self.cold().status().handle(),
                            elements,
                        )?;
                    }
                }
                self.enqueue_primitive(
                    self.cold().program().kv_append(step)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::Key,
                            Access::Read,
                            self.value_handle(request, *key, inputs[0])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::PackedAttention,
                            Access::Read,
                            self.value_handle(request, *packed, inputs[1])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::KeyCache,
                            Access::ReadWrite,
                            candidate_keys.handle.clone(),
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::ValueCache,
                            Access::ReadWrite,
                            candidate_values.handle.clone(),
                        ),
                    ],
                )?;
                launches += 1;
                Ok((launches, 0))
            }
            FlashAttention { domain, .. } => {
                let ([query], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("ordinary CUDA attention SSA arity".into()));
                };
                let state = self.state(states, StateSlot::Kv(domain))?;
                let StateBuffers::Kv { keys, values } = &state.candidate else {
                    return Err(Error::State("ordinary CUDA attention KV layout".into()));
                };
                let query = self.value_handle(request, *query, inputs[0])?;
                let output_handle = self.value_handle(request, *output, outputs[0])?;
                let launches = self.cold().attention().enqueue(
                    client,
                    descriptor,
                    &query,
                    keys,
                    values,
                    &output_handle,
                    self.cold().status(),
                )?;
                Ok((launches, 0))
            }
            Sigmoid { .. } => {
                let ([gate], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("ordinary CUDA sigmoid SSA arity".into()));
                };
                let packed = gate_alias.remove(&gate.0).map(ValueId).ok_or_else(|| {
                    Error::State("ordinary CUDA sigmoid gate producer alias absent".into())
                })?;
                self.enqueue_primitive(
                    self.cold().program().sigmoid(step.rows)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::PackedAttention,
                            Access::Read,
                            self.value_handle(request, packed, step.rows * 5120)?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Output,
                            Access::Write,
                            self.value_handle(request, *output, outputs[0])?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            Multiply { .. } => {
                let ([input, gate], [output]) = (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("ordinary CUDA multiply SSA arity".into()));
                };
                self.enqueue_primitive(
                    self.cold().program().multiply(step.rows)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::Input,
                            Access::Read,
                            self.value_handle(request, *input, inputs[0])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Gate,
                            Access::Read,
                            self.value_handle(request, *gate, inputs[1])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Output,
                            Access::Write,
                            self.value_handle(request, *output, outputs[0])?,
                        ),
                    ],
                )?;
                Ok((1, 0))
            }
            SiluAndMul { .. } => {
                let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("ordinary CUDA SiLU SSA arity".into()));
                };
                self.enqueue_primitive(
                    self.cold().program().silu_up(step.rows)?,
                    vec![
                        LaunchBinding::new(
                            OrdinaryRole::PackedGateUp,
                            Access::Read,
                            self.value_handle(request, *input, inputs[0])?,
                        ),
                        LaunchBinding::new(
                            OrdinaryRole::Output,
                            Access::Write,
                            self.value_handle(request, *output, outputs[0])?,
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
            ),
            GatedDelta { domain, .. } => self.enqueue_gdn_recurrent(
                request,
                node_index,
                node,
                descriptor,
                self.state(states, StateSlot::Recurrent(domain))?,
                step,
            ),
            RetainedRows { .. }
            | CausalConvSiluZero { .. }
            | GroupedQkNormalize(_)
            | StateCarry { .. } => Err(Error::Unsupported(
                "ordinary CUDA request contains unsupported source operator".into(),
            )),
        }
    }

    fn enqueue_gdn_convolution(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        state: &StateBinding,
        step: Step,
    ) -> Result<(usize, usize)> {
        let ([packed_qkvz, packed_ba], [conv_output, z]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "ordinary CUDA GDN convolution SSA arity".into(),
            ));
        };
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let parameters = self.cold().node_parameters(node_index)?;
        let [conv_weights] = parameters else {
            return Err(Error::State(
                "ordinary CUDA GDN convolution parameter arity".into(),
            ));
        };
        let StateBuffers::Vector(prior) = &state.prior else {
            return Err(Error::State(
                "ordinary CUDA GDN prior conv state layout".into(),
            ));
        };
        let StateBuffers::Vector(candidate) = &state.candidate else {
            return Err(Error::State(
                "ordinary CUDA GDN candidate conv state layout".into(),
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

        let runtime = self.gdn_plan(step, descriptor.call().mode())?;
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

        let mut handles = self.gdn_internal_handles()?;
        handles.insert(
            GdnRuntimeBuffer::PackedQkvz,
            self.value_handle(request, *packed_qkvz, inputs[0])?,
        );
        handles.insert(
            GdnRuntimeBuffer::PackedBa,
            self.value_handle(request, *packed_ba, inputs[1])?,
        );
        let conv_handle = self.value_handle(request, *conv_output, outputs[0])?;
        let z_handle = self.value_handle(request, *z, outputs[1])?;
        handles.insert(GdnRuntimeBuffer::ConvOutput, conv_handle.clone());
        handles.insert(GdnRuntimeBuffer::ZCompact, z_handle.clone());
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
            .ok_or_else(|| Error::Limit("ordinary CUDA GDN conv dispatch count".into()))?;
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

    fn enqueue_gdn_recurrent(
        &self,
        request: &ForwardPlan,
        node_index: usize,
        node: &PlanNode,
        descriptor: &SourceOperator,
        state: &StateBinding,
        step: Step,
    ) -> Result<(usize, usize)> {
        let ([conv_output, z, packed_ba], [output]) =
            (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State("ordinary CUDA GDN recurrent SSA arity".into()));
        };
        let inputs = descriptor.input_elements()?;
        let outputs = descriptor.output_elements()?;
        let parameters = self.cold().node_parameters(node_index)?;
        let [a_log, dt_bias, gamma] = parameters else {
            return Err(Error::State(
                "ordinary CUDA GDN recurrent parameter arity".into(),
            ));
        };
        let StateBuffers::Vector(prior) = &state.prior else {
            return Err(Error::State(
                "ordinary CUDA GDN prior recurrent state layout".into(),
            ));
        };
        let StateBuffers::Vector(candidate) = &state.candidate else {
            return Err(Error::State(
                "ordinary CUDA GDN candidate recurrent state layout".into(),
            ));
        };
        let recurrent_layout =
            ResidentBufferLayout::new(ResidentPhysicalType::F32, RECURRENT_ELEMENTS)?;
        prior.require_layout(recurrent_layout)?;
        candidate.require_layout(recurrent_layout)?;

        let runtime = self.gdn_plan(step, descriptor.call().mode())?;
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

        let mut handles = self.gdn_internal_handles()?;
        handles.insert(
            GdnRuntimeBuffer::ConvOutput,
            self.value_handle(request, *conv_output, inputs[0])?,
        );
        handles.insert(
            GdnRuntimeBuffer::ZCompact,
            self.value_handle(request, *z, inputs[1])?,
        );
        handles.insert(
            GdnRuntimeBuffer::PackedBa,
            self.value_handle(request, *packed_ba, inputs[2])?,
        );
        let output_handle = self.value_handle(request, *output, outputs[0])?;
        handles.insert(GdnRuntimeBuffer::Gated, output_handle.clone());
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
            .ok_or_else(|| Error::Limit("ordinary CUDA GDN recurrent dispatch count".into()))?;
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

    fn gdn_plan(&self, step: Step, mode: SourceCallMode) -> Result<GdnRuntimePlan> {
        let slot = GdnRuntimeCacheSlot::new(1)?;
        if self.cold().plan().source_call().is_some() {
            return Err(Error::State(
                "ordinary CUDA cold plan unexpectedly owns a source call".into(),
            ));
        }
        let plan = match mode {
            SourceCallMode::Decode => {
                GdnRuntimePlan::cached_decode(layout::gdn_runtime_config()?, slot, step.base)?
            }
            SourceCallMode::Prefill => GdnRuntimePlan::prefill_chunk(
                layout::gdn_runtime_config()?,
                slot,
                step.base,
                step.rows,
                step.base != 0,
            )?,
        };
        if plan.workspace() != self.cold().layout().gdn() {
            return Err(Error::State(
                "ordinary CUDA GDN request workspace differs from cold layout".into(),
            ));
        }
        Ok(plan)
    }

    fn write_gdn_controls(&self, plan: &GdnRuntimePlan) -> Result<()> {
        for control in plan.controls() {
            let lease = self.cold().gdn_internal(control.role())?;
            let handle = exact_view(lease, 0, control.bytes().len(), "ordinary GDN control")?;
            self.cold()
                .service()
                .client()
                .write(&handle, Bytes::from_bytes_vec(control.bytes().to_vec()));
        }
        Ok(())
    }

    fn gdn_internal_handles(&self) -> Result<BTreeMap<GdnRuntimeBuffer, Handle>> {
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
            .ok_or_else(|| Error::State("ordinary CUDA state slot binding absent".into()))
    }

    fn value_handle(&self, request: &ForwardPlan, id: ValueId, elements: usize) -> Result<Handle> {
        let shape = request
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("ordinary CUDA value ID absent".into()))?;
        if elements == 0 || elements > shape.capacity_elements()? {
            return Err(Error::State(
                "ordinary CUDA active value extent exceeds static capacity".into(),
            ));
        }
        let bytes = elements
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("ordinary CUDA active BF16 bytes".into()))?;
        exact_view(
            self.cold().value_lease(id.0 as usize)?,
            0,
            bytes,
            "ordinary active BF16 value",
        )
    }

    fn enqueue_primitive(
        &self,
        definition: Definition,
        bindings: Vec<LaunchBinding>,
    ) -> Result<()> {
        let client = self.cold().service().client();
        let status = LaunchBinding::new(
            OrdinaryRole::Status,
            Access::ReadWrite,
            self.cold().status().handle(),
        );
        let prepared = definition.prepare_owned(client, bindings, status)?;
        prepared.enqueue_owned(client, &self.cold().status().handle())
    }
}
