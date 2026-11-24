//! Original sealed-plan node order, lowered inside one original resident flight.
use super::{
    cold::{Cold, view},
    math::{arg, dim, grid, subview},
    *,
};
use crate::resident_leases::{ResidentLease, ResidentSubmission as Flight};
use crate::resident_program::{
    BufferRange, DeviceBuffer, ResidentOutput, ResidentSubmission, StateBinding,
    bf16_projection::{Bf16ProjectionBindings, Bf16ProjectionStorage},
};
use cubecl::{bytes::Bytes, server::Handle};
use zerogravity_core::{
    accelerator::program::{
        InvocationPhase, PlanNode, SourceOperator, SourceOperatorKind, TextInvocation, ValueId,
    },
    device_events::{BufferRole, DeviceAccess},
};

impl Cold {
    pub(super) fn value_lease(&self, id: ValueId) -> Result<&ResidentLease> {
        let b = self
            .layout
            .bindings
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("text-execution value binding absent".into()))?;
        self.slots
            .get(b.slot)
            .ok_or_else(|| Error::State("text-execution value slot absent".into()))
    }
    pub(super) fn value(&self, id: ValueId, elements: usize) -> Result<Handle> {
        let shape = self
            .layout
            .plan
            .values()
            .get(id.0 as usize)
            .ok_or_else(|| Error::State("text-execution logical value absent".into()))?;
        if elements == 0 || elements > shape.capacity_elements()? {
            return Err(Error::State(
                "text-execution actual value exceeds capacity".into(),
            ));
        }
        view(self.value_lease(id)?, 0, elements * 4)
    }
    pub(in crate::resident_program) fn states<'a>(
        &self,
        call: &TextInvocation,
        states: &'a [StateBinding],
    ) -> Result<BTreeMap<StateSlot, &'a StateBinding>> {
        if states.len() != self.layout.preflight.states.len() {
            return Err(Error::State("text-execution root domain count".into()));
        }
        let mut map = BTreeMap::new();
        let base_len = self.accesses.len();
        let mut physical = self
            .accesses
            .iter()
            .map(|(lease, _)| lease.clone())
            .collect::<Vec<_>>();
        for (b, expected) in states.iter().zip(&self.layout.preflight.states) {
            if b.slot != expected.slot
                || map.insert(b.slot, b).is_some()
                || b.prior_tokens != call.tokens().base()
            {
                return Err(Error::State(
                    "text-execution original domain/frontier differs".into(),
                ));
            }
            let (planes, shape) = expected.plane_layout()?;
            for (buffers, prior) in [(&b.prior, true), (&b.candidate, false)] {
                let actual = buffers.buffers();
                if actual.len() != planes {
                    return Err(Error::State("text-execution root plane count".into()));
                }
                for buffer in actual {
                    buffer.require_layout(shape)?;
                    let lease = buffer.original_lease()?;
                    if !Arc::ptr_eq(lease.service(), &self.service)
                        || lease.source_identity() != self.source.source_identity()
                        || lease.role() != BufferRole::RecurrentState
                        || lease.service_generation() != self.service.generation()
                        || prior && !lease.is_initialized()
                    {
                        return Err(Error::State(
                            "text-execution original root lease differs".into(),
                        ));
                    }
                    let range = BufferRange::new(&buffer.handle)?;
                    range.require_exact_len(shape.bytes()?)?;
                    physical.push(lease.clone());
                }
            }
        }
        let (workspace, roots) = physical.split_at(base_len);
        if !self.service.pairwise_disjoint_extension(workspace, roots)? {
            return Err(Error::State(
                "text-execution independent root/workspace owners overlap".into(),
            ));
        }
        Ok(map)
    }
    pub(in crate::resident_program) async fn execute(
        self: &Arc<Self>,
        flight: &mut Flight<'_>,
        request: &ForwardPlan,
        states: &[StateBinding],
        output: ResidentOutput,
        mut observer: Option<&mut dyn super::super::text_observation::TextObserver>,
    ) -> Result<ResidentSubmission> {
        let call = self.validate_request(request)?;
        let state = self.states(call, states)?;
        flight.require_accesses(self.accesses.iter().map(|(l, a)| (l, *a)))?;
        for b in states {
            for v in b.prior.buffers() {
                flight.require_accesses([(v.original_lease()?, DeviceAccess::Read)])?;
            }
            for v in b.candidate.buffers() {
                flight.require_accesses([(v.original_lease()?, DeviceAccess::Write)])?;
            }
        }
        flight.retain_completion_owner(self.clone())?;
        self.native
            .require_live(self.service.client(), self.source.source_identity())?;
        self.service
            .client()
            .write(&self.status_view()?, Bytes::from_bytes_vec(vec![0; 4]));
        if let Some(observer) = &mut observer {
            self.observe_states(flight, call, states, true, &mut **observer)
                .await?;
        }
        let nodes = match output {
            ResidentOutput::StateOnly => request.state_nodes(),
            ResidentOutput::Logits => request.nodes(),
        };
        let mut dispatches = 0usize;
        // A submitted call is not cut short by cancellation. Its original flight
        // must complete (or quarantine); publication decides completed discard.
        for node in nodes {
            let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
                unreachable!("validated original materialization")
            };
            dispatches = dispatches
                .checked_add(self.node(flight, node, descriptor, &state).await?)
                .ok_or_else(|| Error::Limit("text-execution dispatch count".into()))?;
            if let Some(observer) = &mut observer {
                for (id, words) in node.outputs.iter().zip(descriptor.output_elements()?) {
                    self.observe(
                        flight,
                        call,
                        super::super::text_observation::TextFrameKind::Value(*id),
                        self.value_lease(*id)?,
                        words,
                        &mut **observer,
                    )
                    .await?;
                }
            }
        }
        if let Some(observer) = &mut observer {
            self.observe_states(flight, call, states, false, &mut **observer)
                .await?;
        }
        self.page.finish()?;
        let logits = if output == ResidentOutput::Logits {
            let id = request
                .logits()
                .ok_or_else(|| Error::State("text-execution logits absent".into()))?;
            Some(DeviceBuffer::from_lease(
                self.value_lease(id)?.clone(),
                ResidentBufferLayout::new(ResidentPhysicalType::F32, self.layout.vocabulary)?,
                BufferRole::Activation,
                DeviceAccess::Write,
            )?)
        } else {
            None
        };
        let proposals = request
            .proposed_domains()
            .iter()
            .map(|v| {
                Ok((
                    *v,
                    state
                        .get(&v.slot)
                        .ok_or_else(|| Error::State("text-execution proposed root absent".into()))?
                        .candidate
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ResidentSubmission {
            logits,
            proposals,
            #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
            cohort_proposals: None,
            status: self.status.handle(),
            quant_status: Vec::new(),
            embedding_status: Vec::new(),
            dispatches,
            fused_dispatches: 0,
        })
    }
    async fn node(
        &self,
        flight: &mut Flight<'_>,
        node: &PlanNode,
        d: &SourceOperator,
        states: &BTreeMap<StateSlot, &StateBinding>,
    ) -> Result<usize> {
        let rows = d.invocation().tokens().len();
        let base = d.invocation().tokens().base();
        let input_lengths = d.input_elements()?;
        let output_lengths = d.output_elements()?;
        if node.inputs.len() != input_lengths.len() || node.outputs.len() != output_lengths.len() {
            return Err(Error::State("text-execution node operand count".into()));
        }
        let inputs = node
            .inputs
            .iter()
            .zip(&input_lengths)
            .map(|(id, n)| self.value(*id, *n))
            .collect::<Result<Vec<_>>>()?;
        let outputs = node
            .outputs
            .iter()
            .zip(&output_lengths)
            .map(|(id, n)| self.value(*id, *n))
            .collect::<Result<Vec<_>>>()?;
        self.node_views(flight, node, d, states, rows, base, &inputs, &outputs)
            .await
    }

    pub(super) async fn node_views(
        &self,
        flight: &mut Flight<'_>,
        node: &PlanNode,
        d: &SourceOperator,
        states: &BTreeMap<StateSlot, &StateBinding>,
        rows: usize,
        base: usize,
        inputs: &[Handle],
        outputs: &[Handle],
    ) -> Result<usize> {
        let input_lengths = d.input_elements()?;
        let output_lengths = d.output_elements()?;
        let client = self.service.client();
        let status = self.status_view()?;
        use SourceOperatorKind::*;
        match d.kind() {
            EmbeddingRows => {
                let tensor = &d.matrices()[0];
                let p = &self.operands[&tensor.name];
                let width = tensor.columns;
                for (row, &id) in d.invocation().tokens().ids().iter().enumerate() {
                    let target = subview(&outputs[0], row * width * 4, width * 4)?;
                    let range = id as usize..id as usize + 1;
                    let launch = |weights: &Handle| unsafe {
                        super::super::bf16_projection::widen::launch_unchecked(
                            client,
                            grid(width),
                            dim(),
                            arg(weights, width / 2),
                            arg(&target, width),
                            arg(&status, 1),
                            crate::program_kernels::NONFINITE_VALUE,
                        );
                    };
                    match p.storage() {
                        Bf16ProjectionStorage::Resident => {
                            let resident = p.resident().ok_or_else(|| {
                                Error::State("text-execution embedding not Ready".into())
                            })?;
                            resident.with_rows(
                                flight,
                                &self.source,
                                p.matrix(),
                                range,
                                &mut |actual| {
                                    launch(actual.handle());
                                    Ok(1)
                                },
                            )?;
                        }
                        Bf16ProjectionStorage::Streamed => {
                            self.page
                                .with_rows(flight, p.matrix(), range.clone(), None, |actual| {
                                    if !actual.matrix().same_binding(p.matrix())
                                        || actual.rows() != range
                                        || actual.bytes() != width * 2
                                    {
                                        return Err(Error::State(
                                            "text-execution original embedding row view".into(),
                                        ));
                                    }
                                    launch(actual.handle());
                                    Ok(vec![
                                        self.value_lease(node.outputs[0])?.clone(),
                                        self.status.clone(),
                                    ])
                                })
                                .await?;
                        }
                    }
                }
                Ok(rows)
            }
            Projection { selected_row } => {
                let columns = d.matrices()[0].columns;
                let active = if selected_row.is_some() { 1 } else { rows };
                let input = if let Some(row) = selected_row {
                    subview(&inputs[0], row * columns * 4, columns * 4)?
                } else {
                    inputs[0].clone()
                };
                let combined = d.matrices().iter().map(|m| m.rows).sum::<usize>();
                let input_lease = self.value_lease(node.inputs[0])?;
                let output_lease = self.value_lease(node.outputs[0])?;
                let direct = active == 1 && input_lease.id() != output_lease.id();
                let mut offset = 0;
                let mut count = 0;
                for m in d.matrices() {
                    let temporary;
                    let (destination, destination_lease) = if direct {
                        (subview(&outputs[0], offset * 4, m.rows * 4)?, output_lease)
                    } else {
                        temporary = self.s(layout::PROJECT, active * m.rows)?;
                        (temporary.clone(), &self.scratch[layout::PROJECT])
                    };
                    let bindings = Bf16ProjectionBindings {
                        input: &input,
                        input_lease,
                        output: &destination,
                        output_lease: destination_lease,
                        packed: &self.packed,
                        accumulation: &self.accumulation,
                        flags: &self.flags,
                        status: &self.status,
                        zero: &self.zero,
                    };
                    count += if active == 1 {
                        self.operands[&m.name]
                            .enqueue_rows_certified(flight, &self.page, bindings, active, None)
                            .await?
                    } else {
                        self.operands[&m.name]
                            .enqueue_rows(flight, &self.page, bindings, active, None)
                            .await?
                    };
                    if !direct {
                        unsafe {
                            kernels::columns::launch_unchecked(
                                client,
                                grid(active * m.rows),
                                dim(),
                                arg(&destination, active * m.rows),
                                arg(&outputs[0], active * combined),
                                arg(&status, 1),
                                active as u32,
                                0u32,
                                offset as u32,
                                m.rows,
                                m.rows,
                                combined,
                            );
                        }
                        count += 1;
                    }
                    offset += m.rows;
                }
                Ok(count)
            }
            Rms(g) | ResidualRms(g) => {
                let mut count = 0;
                let input = if matches!(d.kind(), ResidualRms(_)) {
                    unsafe {
                        kernels::residual::launch_unchecked(
                            client,
                            grid(rows * g.width),
                            dim(),
                            arg(&inputs[0], rows * g.width),
                            arg(&inputs[1], rows * g.width),
                            arg(&outputs[1], rows * g.width),
                            arg(&status, 1),
                            (rows * g.width) as u32,
                        );
                    }
                    count += 1;
                    &outputs[1]
                } else {
                    &inputs[0]
                };
                let gamma = self.parameter(&d.parameters()[0])?;
                count += self.rms(
                    input,
                    &gamma,
                    &outputs[0],
                    g.width,
                    rows,
                    crate::source_rms::Gamma::OnePlusRaw,
                )?;
                Ok(count)
            }
            CausalConvSplit {
                domain,
                geometry: g,
            } => {
                let b = states[&StateSlot::Conv(domain)];
                let prior = b.prior.handles();
                let next = b.candidate.handles();
                let weights = self.parameter(&d.parameters()[0])?;
                unsafe {
                    kernels::convolution::launch_unchecked(
                        client,
                        grid(g.qkv_channels),
                        dim(),
                        arg(&inputs[0], input_lengths[0]),
                        arg(&prior[0], g.qkv_channels * 4),
                        arg(&weights, g.qkv_channels * 4),
                        arg(&outputs[0], output_lengths[0]),
                        arg(&outputs[1], output_lengths[1]),
                        arg(&next[0], g.qkv_channels * 4),
                        arg(&status, 1),
                        rows as u32,
                        u32::from(base != 0),
                        g.qkv_channels,
                        g.z_channels,
                    );
                }
                Ok(1)
            }
            GatedDelta { domain, geometry } => {
                let b = states[&StateSlot::Recurrent(domain)];
                self.gdn(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    &b.prior.handles()[0],
                    &b.candidate.handles()[0],
                    &outputs[0],
                    d.parameters(),
                    geometry,
                    rows,
                    d.invocation().phase(),
                )
            }
            QkRmsRotaryGated(g) => {
                let qh = g.attention.query_heads;
                let kh = g.attention.kv_heads;
                let q = qh * 256;
                let k = kh * 256;
                let rawq = self.s(layout::QUERY, rows * q)?;
                let rawk = self.s(layout::KEY, rows * k)?;
                let normq = self.s(layout::CORE, rows * q)?;
                let normk = self.s(layout::VALUE, rows * k)?;
                let gammaq = self.parameter(&d.parameters()[0])?;
                let gammak = self.parameter(&d.parameters()[1])?;
                let freq = self.table(3, 32 * 4)?;
                let table = self.trig_table()?;
                let angles = self.s(layout::ANGLES, 32)?;
                let trig = self.s(layout::TRIG, 64)?;
                unsafe {
                    kernels::split_attention::launch_unchecked(
                        client,
                        grid(rows * q),
                        dim(),
                        arg(&inputs[0], input_lengths[0]),
                        arg(&rawq, rows * q),
                        arg(&rawk, rows * k),
                        arg(&outputs[2], rows * q),
                        rows as u32,
                        qh,
                        kh,
                    );
                }
                let mut count = 1;
                for (raw, norm, gamma, heads) in
                    [(&rawq, &normq, &gammaq, qh), (&rawk, &normk, &gammak, kh)]
                {
                    count += self.rms(
                        raw,
                        gamma,
                        norm,
                        256,
                        rows * heads,
                        crate::source_rms::Gamma::OnePlusRaw,
                    )?;
                }
                let aligned_decode = rows > 1
                    && d.invocation().phase() == InvocationPhase::Decode
                    && d.invocation().tokens().len() == 1;
                if aligned_decode {
                    unsafe {
                        crate::source_rotary::kernels::angles::launch_unchecked(
                            client,
                            grid(32),
                            dim(),
                            arg(&freq, 32),
                            arg(&angles, 32),
                            base as u32,
                            32,
                        );
                        crate::source_trig::cpu::sincos_angles::launch_unchecked(
                            client,
                            grid(32),
                            dim(),
                            arg(&angles, 32),
                            arg(&table, crate::source_trig::cpu::TABLES.len()),
                            arg(&trig, 64),
                            arg(&status, 1),
                            32,
                        );
                    }
                    count += 2;
                    for (norm, target, heads) in
                        [(&normq, &outputs[0], qh), (&normk, &outputs[1], kh)]
                    {
                        let width = heads * 256;
                        unsafe {
                            kernels::columns::launch_unchecked(
                                client,
                                grid(rows * width),
                                dim(),
                                arg(norm, rows * width),
                                arg(target, rows * width),
                                arg(&status, 1),
                                rows as u32,
                                0u32,
                                0u32,
                                width,
                                width,
                                width,
                            );
                            crate::source_rotary::kernels::apply::launch_unchecked(
                                client,
                                grid(rows * heads * 32),
                                dim(),
                                arg(norm, rows * width),
                                arg(&trig, 64),
                                arg(target, rows * width),
                                arg(&status, 1),
                                256,
                                rows * width,
                                64,
                            );
                        }
                        count += 2;
                    }
                    return Ok(count);
                }
                for row in 0..rows {
                    unsafe {
                        crate::source_rotary::kernels::angles::launch_unchecked(
                            client,
                            grid(32),
                            dim(),
                            arg(&freq, 32),
                            arg(&angles, 32),
                            (base + row) as u32,
                            32,
                        );
                        crate::source_trig::cpu::sincos_angles::launch_unchecked(
                            client,
                            grid(32),
                            dim(),
                            arg(&angles, 32),
                            arg(&table, crate::source_trig::cpu::TABLES.len()),
                            arg(&trig, 64),
                            arg(&status, 1),
                            32,
                        );
                    }
                    count += 2;
                    for (norm, target, heads) in
                        [(&normq, &outputs[0], qh), (&normk, &outputs[1], kh)]
                    {
                        let width = heads * 256;
                        let norm = subview(norm, row * width * 4, width * 4)?;
                        let out = subview(target, row * width * 4, width * 4)?;
                        unsafe {
                            kernels::columns::launch_unchecked(
                                client,
                                grid(width),
                                dim(),
                                arg(&norm, width),
                                arg(&out, width),
                                arg(&status, 1),
                                1u32,
                                0u32,
                                0u32,
                                width,
                                width,
                                width,
                            );
                            crate::source_rotary::kernels::apply::launch_unchecked(
                                client,
                                grid(heads * 32),
                                dim(),
                                arg(&norm, width),
                                arg(&trig, 64),
                                arg(&out, width),
                                arg(&status, 1),
                                256,
                                width,
                                64,
                            );
                        }
                        count += 2;
                    }
                }
                Ok(count)
            }
            KvAppend {
                domain,
                key_width,
                packed_value,
            } => {
                let b = states[&StateSlot::Kv(domain)];
                let prior = b.prior.handles();
                let next = b.candidate.handles();
                let cap = d.invocation().tokens().capacity();
                unsafe {
                    kernels::append_kv::launch_unchecked(
                        client,
                        grid(cap * key_width),
                        dim(),
                        arg(&prior[0], cap * key_width),
                        arg(&prior[1], cap * key_width),
                        arg(&inputs[0], rows * key_width),
                        arg(&inputs[1], input_lengths[1]),
                        arg(&next[0], cap * key_width),
                        arg(&next[1], cap * key_width),
                        arg(&status, 1),
                        base as u32,
                        rows as u32,
                        packed_value.offset as u32,
                        key_width,
                        packed_value.row_stride,
                        cap,
                    );
                }
                Ok(1)
            }
            Attention {
                domain,
                geometry: g,
                ..
            } => {
                let b = states[&StateSlot::Kv(domain)];
                let next = b.candidate.handles();
                let tokens = base + rows;
                let kv = g.kv_heads * 256;
                let geometry = crate::source_attention::rows::Geometry::new(
                    g.query_heads,
                    g.kv_heads,
                    d.invocation().tokens().capacity(),
                )?;
                let scratch = geometry
                    .scratch_words()
                    .into_iter()
                    .enumerate()
                    .map(|(i, n)| self.s(layout::ATTENTION.start + i, n))
                    .collect::<Result<Vec<_>>>()?;
                crate::source_attention::rows::launch(
                    client,
                    geometry,
                    base,
                    rows,
                    &inputs[0],
                    &subview(&next[0], 0, tokens * kv * 4)?,
                    &subview(&next[1], 0, tokens * kv * 4)?,
                    &outputs[0],
                    std::array::from_fn(|i| &scratch[i]),
                    &status,
                    &self.zero_view()?,
                )
            }
            Sigmoid { width } => self.sigmoid(&inputs[0], &outputs[0], rows * width),
            Multiply { width } => {
                unsafe {
                    kernels::multiply::launch_unchecked(
                        client,
                        grid(rows * width),
                        dim(),
                        arg(&inputs[0], rows * width),
                        arg(&inputs[1], rows * width),
                        arg(&outputs[0], rows * width),
                        arg(&status, 1),
                        (rows * width) as u32,
                    );
                }
                Ok(1)
            }
            SiluAndMul { width } => {
                unsafe {
                    kernels::silu_mul::launch_unchecked(
                        client,
                        grid(rows * width),
                        dim(),
                        arg(&inputs[0], rows * width * 2),
                        arg(&outputs[0], rows * width),
                        arg(&status, 1),
                        rows as u32,
                        width,
                    );
                }
                Ok(1)
            }
            _ => Err(Error::Unsupported(
                "text-execution unimplemented kernel".into(),
            )),
        }
    }
}
