//! Batch-shaped physical lowering of the original, source-bound ForwardPlan.
//!
//! No model schedule, session authority or Ready constructor lives here. The
//! physical owner admits every allocation, retains original leases before
//! launch, and publishes independent roots only after real completion.

use super::*;
use crate::activation_quant::{self, PARAMETER_WORDS};
use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
use crate::source_float::{round_source, widen_source};

/// One original plan, shared immutable tensors, and a private B-shaped workspace.
pub(crate) struct BatchProgram {
    inner: ResidentProgram,
    rows: usize,
    nodes: Vec<PlanNode>,
    packed: Vec<StateBinding>,
    positions: Handle,
    greedy: [Handle; 3],
    projection_blocks: Vec<Option<PreparedBlocks>>,
    owned: Vec<Handle>,
}

impl BatchProgram {
    /// New allocations only: the original source handles remain shared.
    /// External per-session prior/candidate allocations are reported separately.
    pub fn preflight(original: &ResidentProgram, rows: usize) -> Result<ResidentPreflight> {
        ResidentProgram::require_legacy_contract(&original.plan)?;
        if !matches!(rows, 2 | 4 | 8) {
            return Err(Error::Unsupported(
                "resident native batch rows must be 2/4/8".into(),
            ));
        }
        let mut budget = original.preflight.clone();
        let scale = |n: usize| {
            n.checked_mul(rows)
                .ok_or_else(|| Error::Limit("resident batch allocation overflow".into()))
        };
        // Device indexing uses u32 element offsets throughout the original ABI.
        for shape in original.plan.values() {
            bytes(scale(shape.capacity_elements()?)?, 4)?;
        }
        for state in &budget.states {
            bytes(scale(state.bytes / 4)?, 4)?;
        }
        budget.source_bytes = 0;
        budget.value_bytes = scale(budget.value_bytes)?;
        budget.status_bytes = checked_sum([scale(budget.status_bytes - 4)?, 4])?;
        budget.parameter_bytes = bytes(rows, 8)?;
        budget.prior_state_bytes = scale(budget.prior_state_bytes)?;
        budget.candidate_state_bytes = scale(budget.candidate_state_bytes)?;
        for size in &mut budget.scratch {
            *size = scale(*size)?;
            bytes(*size / 4, 4)?;
        }
        // Packed copies are scratch, never a shared mutable StateRoot. Both
        // complete cohorts are charged in addition to the independent roots.
        budget.workspace_bytes = checked_sum([
            checked_sum(budget.scratch)?,
            budget.prior_state_bytes,
            budget.candidate_state_bytes,
            checked_sum(Self::greedy_sizes(original, rows)?)?,
        ])?;
        Ok(budget)
    }

    fn greedy_sizes(original: &ResidentProgram, rows: usize) -> Result<[usize; 3]> {
        let logits = original
            .plan
            .logits()
            .ok_or_else(|| Error::Unsupported("native batch requires planned logits".into()))?;
        let vocabulary = original.plan.values()[logits.0 as usize].capacity_elements()?;
        crate::resident_batch_greedy::workspace_bytes(vocabulary, rows)
    }

    pub fn prepare(
        client: &Client,
        original: &ResidentProgram,
        rows: usize,
        mut allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
    ) -> Result<Self> {
        let preflight = Self::preflight(original, rows)?;
        ResidentProgram::ensure_capability(client)?;
        if !crate::activation_quant_fused::is_cuda_service(client) {
            return Err(Error::Unsupported(
                "native source batch requires CUDA lowering".into(),
            ));
        }
        let mut owned = Vec::new();
        let mut recorded =
            |role: ResidentBufferRole, n: usize, init: AllocationInit<'_>| -> Result<Handle> {
                let handle = allocate(role, n, init)?;
                owned.push(handle.clone());
                Ok(handle)
            };
        let mut allocator = ProgramAllocator {
            client,
            allocate: &mut recorded,
        };
        let aliases = contiguous_aliases(&original.plan)?;
        let mut values: Vec<DeviceBuffer> = Vec::with_capacity(original.values.len());
        for (shape, alias) in original.plan.values().iter().zip(aliases) {
            let elements = rows * shape.capacity_elements()?;
            values.push(if let Some(original) = alias {
                values[original.0 as usize].clone()
            } else {
                DeviceBuffer::new(
                    allocator.empty(ResidentBufferRole::Value, bytes(elements, 4)?)?,
                    elements,
                )
            });
        }
        let scratch: [Handle; SCRATCH_SLOTS] = (0..SCRATCH_SLOTS)
            .map(|i| allocator.empty(ResidentBufferRole::Workspace, preflight.scratch[i]))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("fixed scratch slot count");
        let greedy: [Handle; 3] = Self::greedy_sizes(original, rows)?
            .into_iter()
            .map(|size| allocator.empty(ResidentBufferRole::Workspace, size))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("fixed greedy workspace count");
        let node_status = original
            .plan
            .nodes()
            .iter()
            .map(|node| {
                let (blocks, outputs) = match &node.operation {
                    PlanOperation::Embed { tensor, .. } => (rows * tensor.columns, 0),
                    PlanOperation::Project { tensor } => original.tensors[&tensor.name]
                        .layout
                        .map_or((0, 0), |l| (rows * l.blocks, rows * l.rows)),
                    _ => (0, 0),
                };
                Ok(NodeStatus {
                    block: if blocks == 0 {
                        None
                    } else {
                        Some(allocator.empty(ResidentBufferRole::Status, bytes(blocks, 4)?)?)
                    },
                    row: if outputs == 0 {
                        None
                    } else {
                        Some(allocator.empty(ResidentBufferRole::Status, bytes(outputs, 4)?)?)
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut packed = Vec::with_capacity(preflight.states.len());
        for requirement in &preflight.states {
            let mut state = || -> Result<StateBuffers> {
                let mut buffer = |elements| -> Result<DeviceBuffer> {
                    Ok(DeviceBuffer::new(
                        allocator.empty(ResidentBufferRole::Workspace, bytes(elements, 4)?)?,
                        elements,
                    ))
                };
                Ok(match requirement.layout {
                    StateLayout::Vector { elements } => {
                        StateBuffers::Vector(buffer(rows * elements)?)
                    }
                    StateLayout::Kv { slab, capacity } => StateBuffers::Kv {
                        keys: buffer(rows * slab * capacity)?,
                        values: buffer(rows * slab * capacity)?,
                    },
                })
            };
            packed.push(StateBinding {
                slot: requirement.slot,
                prior: state()?,
                candidate: state()?,
                prior_tokens: 0,
            });
        }
        let tensors = original
            .tensors
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    ResidentTensor {
                        packed: tensor.packed.clone(),
                        elements: tensor.elements,
                        scales: tensor.scales.clone(),
                        biases: tensor.biases.clone(),
                        layout: tensor.layout,
                    },
                )
            })
            .collect();
        let mut nodes = original.plan.nodes().to_vec();
        for node in &mut nodes {
            match &mut node.operation {
                PlanOperation::Rms { groups, .. }
                | PlanOperation::AttentionSoftmax { heads: groups, .. } => *groups *= rows,
                PlanOperation::Rope { geometry, .. } => {
                    geometry.attention.query_heads *= rows;
                    geometry.attention.kv_heads *= rows;
                }
                PlanOperation::AttentionQk { geometry, .. }
                | PlanOperation::AttentionValue { geometry, .. } => {
                    geometry.query_heads *= rows;
                    geometry.kv_heads *= rows;
                }
                _ => {}
            }
        }
        let positions = allocator.empty(ResidentBufferRole::Parameter, bytes(rows, 4)?)?;
        let mut inner = ResidentProgram {
            _source_owner: Arc::clone(&original._source_owner),
            plan: original.plan.clone(),
            preflight,
            tensors,
            constants: original.constants.clone(),
            values,
            scratch,
            status: allocator.empty(ResidentBufferRole::Status, 4)?,
            node_status,
            // Quantized batch input preparation below binds existing originals.
            projections: Vec::new(),
            token: allocator.empty(ResidentBufferRole::Parameter, bytes(rows, 4)?)?,
            exp_tables: original.exp_tables.clone(),
            trig_tables: original.trig_tables.clone(),
            owned_ranges: OwnedRanges::new(client.service_id()),
        };
        for handle in inner.owned_handles() {
            inner.owned_ranges.insert(BufferRange::new(&handle)?)?;
        }
        for handle in &greedy {
            inner.owned_ranges.insert(BufferRange::new(handle)?)?;
        }
        inner.owned_ranges.insert(BufferRange::new(&positions)?)?;
        for binding in &packed {
            for handle in binding
                .prior
                .handles()
                .into_iter()
                .chain(binding.candidate.handles())
            {
                inner.owned_ranges.insert(BufferRange::new(&handle)?)?;
            }
        }
        let projection_blocks = inner
            .plan
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let PlanOperation::Project { tensor } = &node.operation else {
                    return Ok(None);
                };
                if tensor.kind == QuantKind::F32 {
                    return Ok(None);
                }
                let layout = inner.tensors[&tensor.name]
                    .layout
                    .ok_or_else(|| Error::State("batch projection layout absent".into()))?;
                let flags = &inner.node_status[index];
                PreparedBlocks::prepare_cuda(
                    client,
                    &layout,
                    rows,
                    &BlockBindings {
                        input: &inner.values[node.inputs[0].0 as usize].handle,
                        parameters: &inner.scratch[0],
                        scaled: &inner.scratch[1],
                        codes: &inner.scratch[2],
                        block_status: flags.block.as_ref().ok_or_else(|| {
                            Error::State("batch projection block status absent".into())
                        })?,
                    },
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner,
            rows,
            nodes,
            packed,
            positions,
            greedy,
            projection_blocks,
            owned,
        })
    }

    pub fn budget(&self) -> &ResidentPreflight {
        &self.inner.preflight
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn greedy_workspace(&self) -> &[Handle; 3] {
        &self.greedy
    }

    pub fn logits(&self) -> Option<DeviceBuffer> {
        self.inner
            .plan
            .logits()
            .map(|id| self.inner.values[id.0 as usize].clone())
    }

    pub fn status(&self) -> Handle {
        self.inner.status.clone()
    }

    pub fn owned_handles(&self) -> Vec<Handle> {
        self.owned.clone()
    }

    pub fn execution_handles(&self, output: ResidentOutput) -> Vec<Handle> {
        let mut handles = self.inner.execution_handles(output);
        handles.push(self.positions.clone());
        for binding in &self.packed {
            handles.extend(binding.prior.handles());
            handles.extend(binding.candidate.handles());
        }
        if output == ResidentOutput::Logits {
            handles.extend(self.greedy.iter().cloned());
        }
        handles
    }

    pub fn validate_runtime(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
    ) -> Result<()> {
        if parameters.len() != self.rows || bindings.len() != self.rows {
            return Err(Error::Malformed("resident batch row count changed".into()));
        }
        let mut priors = Vec::new();
        let mut candidates = Vec::new();
        for (parameter, binding) in parameters.iter().zip(bindings) {
            self.inner.validate_runtime(client, *parameter, binding)?;
            for node in self.inner.plan.nodes() {
                if let PlanOperation::Embed { tensor, .. } = &node.operation
                    && parameter.token as usize >= tensor.rows
                {
                    return Err(Error::Malformed(
                        "native batch token exceeds source vocabulary".into(),
                    ));
                }
            }
            for state in binding {
                for handle in state.prior.handles() {
                    priors.push(BufferRange::new(&handle)?);
                }
                for handle in state.candidate.handles() {
                    candidates.push(BufferRange::new(&handle)?);
                }
            }
        }
        for (i, candidate) in candidates.iter().enumerate() {
            if priors
                .iter()
                .chain(candidates[..i].iter())
                .any(|other| candidate.overlaps(other))
            {
                return Err(Error::State(
                    "native batch candidate aliases another row".into(),
                ));
            }
        }
        Ok(())
    }

    /// ONE traversal of the original plan. Arithmetic stages cover every row;
    /// private physical state handles are only gathered/scattered, never shared.
    pub fn launch(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(client, parameters, bindings, None, output)
    }

    /// Trusted owner checks each token's producer RootBinding and retains the
    /// independent original packet leases before this physical preflight.
    pub fn validate_device_tokens(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        tokens: &[DeviceBuffer],
    ) -> Result<()> {
        self.validate_runtime(client, parameters, bindings)?;
        if tokens.len() != self.rows {
            return Err(Error::Malformed("native device-token row count".into()));
        }
        let mut ranges = Vec::with_capacity(tokens.len());
        for token in tokens {
            let range = token.check(client, 1)?;
            if self.inner.owned_ranges.overlaps(&range)
                || ranges
                    .iter()
                    .any(|prior: &BufferRange| range.overlaps(prior))
            {
                return Err(Error::State(
                    "native device-token aliases program/another row".into(),
                ));
            }
            for row in bindings {
                for binding in row {
                    for handle in binding.candidate.handles() {
                        if range.overlaps(&BufferRange::new(&handle)?) {
                            return Err(Error::State(
                                "native device-token aliases proposed state".into(),
                            ));
                        }
                    }
                }
            }
            ranges.push(range);
        }
        Ok(())
    }

    /// One native gather reads the row-bound original device packets. It never
    /// serializes their token IDs through host scalars or uploads a token.
    pub fn launch_device_tokens(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        tokens: &[DeviceBuffer],
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(client, parameters, bindings, Some(tokens), output)
    }

    fn launch_impl(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        device_tokens: Option<&[DeviceBuffer]>,
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        if let Some(tokens) = device_tokens {
            self.validate_device_tokens(client, parameters, bindings, tokens)?;
        } else {
            self.validate_runtime(client, parameters, bindings)?;
        }
        unsafe {
            crate::program_kernels::clear_step_status::launch_unchecked(
                client,
                grid(1),
                threads(),
                arg(&self.inner.status, 1),
            );
            if let Some(tokens) = device_tokens {
                // Inactive read-only arguments reuse row zero; compile-time
                // row bounds prevent them from participating in the gather.
                let packet = |row: usize| &tokens.get(row).unwrap_or(&tokens[0]).handle;
                batch_device_tokens::launch_unchecked(
                    client,
                    grid(self.rows),
                    threads(),
                    arg(packet(0), 1),
                    arg(packet(1), 1),
                    arg(packet(2), 1),
                    arg(packet(3), 1),
                    arg(packet(4), 1),
                    arg(packet(5), 1),
                    arg(packet(6), 1),
                    arg(packet(7), 1),
                    arg(&self.inner.token, self.rows),
                    self.rows,
                );
            } else {
                let mut tokens = [0u32; 8];
                for (token, parameter) in tokens.iter_mut().zip(parameters) {
                    *token = parameter.token;
                }
                batch_tokens::launch_unchecked(
                    client,
                    grid(self.rows),
                    threads(),
                    arg(&self.inner.token, self.rows),
                    tokens[0],
                    tokens[1],
                    tokens[2],
                    tokens[3],
                    tokens[4],
                    tokens[5],
                    tokens[6],
                    tokens[7],
                    self.rows,
                );
            }
            let mut positions = [0u32; 8];
            for (position, parameter) in positions.iter_mut().zip(parameters) {
                *position = parameter.position as u32;
            }
            batch_tokens::launch_unchecked(
                client,
                grid(self.rows),
                threads(),
                arg(&self.positions, self.rows),
                positions[0],
                positions[1],
                positions[2],
                positions[3],
                positions[4],
                positions[5],
                positions[6],
                positions[7],
                self.rows,
            );
        }
        let mut dispatches = 3 + self.transport_state(client, bindings, false);
        // One actual native projection-dot launch per Project. Activation
        // preparation/status stages remain in the total dispatch count only.
        let mut fused_dispatches = 0;
        let count = output.nodes(&self.inner.plan).len();
        for (index, node) in self.nodes[..count].iter().enumerate() {
            dispatches += self.launch_node(client, index, node, parameters)?;
            if matches!(node.operation, PlanOperation::Project { .. }) {
                fused_dispatches += 1;
            }
            if matches!(node.operation, PlanOperation::View(_)) {
                continue;
            }
            for (out_index, id) in node.outputs.iter().enumerate() {
                let value = &self.inner.values[id.0 as usize];
                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
                    && out_index == 1
                {
                    unsafe {
                        crate::resident_bindings::certify_causal_rows::launch_unchecked(
                            client,
                            grid(value.elements),
                            threads(),
                            arg(&value.handle, value.elements),
                            arg(&self.inner.status, 1),
                            arg(&self.positions, self.rows),
                            geometry.query_heads / self.rows,
                            self.rows,
                            self.inner.plan.prefix_capacity(),
                        );
                    }
                } else {
                    dispatches += self.certify(client, &value.handle, value.elements);
                    continue;
                }
                dispatches += 1;
            }
        }
        dispatches += self.transport_state(client, bindings, true);
        let mut proposals = Vec::new();
        for row in bindings {
            for version in self.inner.plan.proposed_domains() {
                let binding = row
                    .iter()
                    .find(|b| b.slot == version.slot)
                    .expect("validated domain");
                for value in binding.candidate.buffers() {
                    dispatches += self.certify(client, &value.handle, value.elements);
                }
                proposals.push((*version, binding.candidate.clone()));
            }
        }
        let mut quant_status = Vec::new();
        let mut embedding_status = Vec::new();
        for (index, node) in self.nodes[..count].iter().enumerate() {
            match &node.operation {
                PlanOperation::Project { tensor } => {
                    if let Some(layout) = self.inner.tensors[&tensor.name].layout {
                        quant_status.push((
                            self.inner.node_status[index]
                                .block
                                .as_ref()
                                .expect("quant status")
                                .clone(),
                            self.rows * layout.blocks,
                            self.inner.node_status[index]
                                .row
                                .as_ref()
                                .expect("quant row status")
                                .clone(),
                            self.rows * layout.rows,
                        ));
                    }
                }
                PlanOperation::Embed { tensor, .. } => embedding_status.push((
                    self.inner.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status")
                        .clone(),
                    self.rows * tensor.columns,
                )),
                _ => {}
            }
        }
        Ok(ResidentSubmission {
            logits: self
                .inner
                .plan
                .logits()
                .filter(|_| output == ResidentOutput::Logits)
                .map(|id| self.inner.values[id.0 as usize].clone()),
            proposals,
            status: self.inner.status.clone(),
            quant_status,
            embedding_status,
            dispatches,
            fused_dispatches,
        })
    }

    fn transport_state(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        scatter: bool,
    ) -> usize {
        let mut count = 0;
        for (row, bindings) in bindings.iter().enumerate() {
            for packed in &self.packed {
                let binding = bindings
                    .iter()
                    .find(|b| b.slot == packed.slot)
                    .expect("validated domain");
                let (private, dense) = if scatter {
                    (&binding.candidate, &packed.candidate)
                } else {
                    (&binding.prior, &packed.prior)
                };
                match (private, dense) {
                    (StateBuffers::Vector(private), StateBuffers::Vector(dense)) => {
                        if scatter {
                            count += self.copy_dense(
                                client,
                                &dense.handle,
                                dense.elements,
                                row * private.elements,
                                &private.handle,
                                private.elements,
                                0,
                                private.elements,
                            );
                        } else {
                            count += self.copy_dense(
                                client,
                                &private.handle,
                                private.elements,
                                0,
                                &dense.handle,
                                dense.elements,
                                row * private.elements,
                                private.elements,
                            );
                        }
                    }
                    (
                        StateBuffers::Kv { keys, values },
                        StateBuffers::Kv {
                            keys: dense_k,
                            values: dense_v,
                        },
                    ) => {
                        let capacity = self.inner.plan.prefix_capacity();
                        let slab = keys.elements / capacity;
                        unsafe {
                            transport_kv::launch_unchecked(
                                client,
                                grid(keys.elements),
                                threads(),
                                arg(&keys.handle, keys.elements),
                                arg(&values.handle, values.elements),
                                arg(&dense_k.handle, dense_k.elements),
                                arg(&dense_v.handle, dense_v.elements),
                                row,
                                slab,
                                capacity,
                                self.rows,
                                scatter,
                            );
                        }
                        count += 1;
                    }
                    _ => unreachable!("validated state layout"),
                }
            }
        }
        count
    }

    fn certify(&self, client: &Client, source: &Handle, elements: usize) -> usize {
        let mut dispatches = 0;
        for offset in (0..elements).step_by(crate::program_kernels::MAX_WORD_OPS) {
            let count = (elements - offset).min(crate::program_kernels::MAX_WORD_OPS);
            unsafe {
                crate::program_kernels::certify_finite_strided::launch_unchecked(
                    client,
                    grid(count),
                    threads(),
                    arg(source, elements),
                    arg(&self.inner.status, 1),
                    offset as u32,
                    count as u32,
                    1,
                    count,
                );
            }
            dispatches += 1;
        }
        dispatches
    }

    #[allow(clippy::too_many_arguments)] // Explicit physical ranges, bounded per dispatch.
    fn copy_dense(
        &self,
        client: &Client,
        source: &Handle,
        source_elements: usize,
        source_offset: usize,
        destination: &Handle,
        destination_elements: usize,
        destination_offset: usize,
        elements: usize,
    ) -> usize {
        let mut dispatches = 0;
        for offset in (0..elements).step_by(crate::program_kernels::MAX_WORD_OPS) {
            let count = (elements - offset).min(crate::program_kernels::MAX_WORD_OPS);
            self.inner.copy::<false>(
                client,
                source,
                source_elements,
                destination,
                destination_elements,
                TensorView::vector(source_offset + offset, count),
                destination_offset + offset,
            );
            dispatches += 1;
        }
        dispatches
    }

    fn kv(&self, version: StateVersion) -> (&DeviceBuffer, &DeviceBuffer) {
        let binding = self
            .packed
            .iter()
            .find(|b| b.slot == version.slot)
            .expect("validated state domain");
        let state = if version.ordinal == 0 {
            &binding.prior
        } else {
            &binding.candidate
        };
        match state {
            StateBuffers::Kv { keys, values } => (keys, values),
            _ => unreachable!("typed KV state"),
        }
    }

    fn launch_node(
        &self,
        client: &Client,
        index: usize,
        node: &PlanNode,
        parameters: &[RuntimeParameters],
    ) -> Result<usize> {
        let p = parameters[0];
        let input = |i: usize| &self.inner.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.inner.values[node.outputs[i].0 as usize];
        let scratch = |i: usize| &self.inner.scratch[i];
        unsafe {
            match &node.operation {
                PlanOperation::TypedEmbed { .. }
                | PlanOperation::TypedProject { .. }
                | PlanOperation::TypedOrdinaryRms { .. } => Err(Error::Unsupported(
                    "resident batch has no qualified typed lowering".into(),
                )),
                PlanOperation::Embed { tensor, .. } => {
                    let t = &self.inner.tensors[&tensor.name];
                    let status = self.inner.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status");
                    let elements = self.rows * tensor.columns;
                    batch_embedding::launch_unchecked(
                        client,
                        grid(elements),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(&self.inner.token, self.rows),
                        arg(&output(0).handle, elements),
                        arg(status, elements),
                        tensor.rows,
                        tensor.columns,
                        self.rows,
                        tensor.kind == QuantKind::Q6K,
                    );
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(elements),
                        threads(),
                        arg(status, elements),
                        arg(&self.inner.status, 1),
                        elements,
                    );
                    Ok(2)
                }
                PlanOperation::Project { tensor } => {
                    self.project(client, index, tensor, input(0), output(0))
                }
                PlanOperation::View(view) => {
                    batch_view::launch_unchecked(
                        client,
                        grid(output(0).elements),
                        threads(),
                        arg(&input(0).handle, input(0).elements),
                        arg(&output(0).handle, output(0).elements),
                        arg(&self.inner.status, 1),
                        input(0).elements / self.rows,
                        view.offset,
                        view.rows,
                        view.width,
                        view.row_stride,
                        self.rows,
                    );
                    Ok(1)
                }
                PlanOperation::Conv { weights } => {
                    let elements = input(0).elements;
                    let effect = node.state.expect("conv effect");
                    let state = self
                        .packed
                        .iter()
                        .find(|b| b.slot == effect.read.slot)
                        .expect("conv state");
                    let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
                        (&state.prior, &state.candidate)
                    else {
                        unreachable!("typed conv state")
                    };
                    batch_conv_products::launch_unchecked(
                        client,
                        grid(elements),
                        threads(),
                        arg(&input(0).handle, elements),
                        arg(
                            &self.inner.constants[weights.0 as usize],
                            elements / self.rows * 4,
                        ),
                        arg(&prior.handle, elements * 3),
                        arg(scratch(0), elements * 4),
                        arg(&candidate.handle, elements * 3),
                        elements / self.rows,
                        self.rows,
                    );
                    for level in 0..3 {
                        crate::gdn_conv::gdn_conv_reduce::launch_unchecked(
                            client,
                            grid(elements),
                            threads(),
                            arg(scratch(0), elements * 4),
                            arg(&output(0).handle, elements),
                            elements,
                            level,
                        );
                    }
                    Ok(4)
                }
                PlanOperation::Coefficients { dt, a, scale } => {
                    let heads = input(0).elements;
                    for i in 0..2 {
                        crate::resident_bindings::certify_interval::launch_unchecked(
                            client,
                            grid(heads),
                            threads(),
                            arg(&input(i).handle, heads),
                            arg(&self.inner.status, 1),
                            -128.0,
                            128.0,
                            heads,
                        );
                    }
                    batch_coefficient_input::launch_unchecked(
                        client,
                        grid(heads),
                        threads(),
                        arg(&input(0).handle, heads),
                        arg(&input(1).handle, heads),
                        arg(&self.inner.constants[dt.0 as usize], heads / self.rows),
                        arg(&self.inner.constants[a.0 as usize], heads / self.rows),
                        arg(scratch(0), heads * 4),
                        heads / self.rows,
                        self.rows,
                    );
                    for stage in 0..8 {
                        crate::gdn_coefficients::coefficient_step::launch_unchecked(
                            client,
                            grid(heads),
                            threads(),
                            arg(scratch(0), heads * 4),
                            arg(
                                &self.inner.exp_tables,
                                crate::source_transcendentals::TABLES.len(),
                            ),
                            arg(scratch(1), heads * 6),
                            arg(scratch(2), heads * 3),
                            arg(scratch(3), heads * 4),
                            heads,
                            stage,
                        );
                    }
                    crate::resident_bindings::coefficient_output::launch_unchecked(
                        client,
                        grid(heads),
                        threads(),
                        arg(scratch(3), heads * 4),
                        arg(&output(0).handle, heads * 3),
                        arg(&output(1).handle, heads),
                        arg(&output(2).handle, heads),
                        *scale,
                        heads,
                    );
                    Ok(12)
                }
                PlanOperation::Rope { geometry, .. } => self.rope(client, node, *geometry),
                PlanOperation::AppendKv => {
                    let effect = node.state.expect("KV append effect");
                    let (previous_k, previous_v) = self.kv(effect.read);
                    let (candidate_k, candidate_v) = self.kv(effect.write);
                    let cap = self.inner.plan.prefix_capacity();
                    let slab = input(0).elements / self.rows;
                    append_kv_rows::launch_unchecked(
                        client,
                        grid(self.rows * slab * cap),
                        threads(),
                        arg(&previous_k.handle, self.rows * slab * cap),
                        arg(&previous_v.handle, self.rows * slab * cap),
                        arg(&input(0).handle, self.rows * slab),
                        arg(&input(1).handle, self.rows * slab),
                        arg(&candidate_k.handle, self.rows * slab * cap),
                        arg(&candidate_v.handle, self.rows * slab * cap),
                        arg(&self.positions, self.rows),
                        slab,
                        cap,
                        self.rows,
                    );
                    Ok(1)
                }
                PlanOperation::AttentionQk {
                    state,
                    geometry,
                    scale,
                    ..
                } => {
                    let (keys, _) = self.kv(*state);
                    let cap = self.inner.plan.prefix_capacity();
                    let query = geometry.query_heads * geometry.width;
                    let key = cap * geometry.kv_heads * geometry.width;
                    let columns = geometry.query_heads * cap;
                    let mut count = self.copy_dense(
                        client,
                        &input(0).handle,
                        query,
                        0,
                        scratch(0),
                        query + key,
                        0,
                        query,
                    );
                    count += self.copy_dense(
                        client,
                        &keys.handle,
                        key,
                        0,
                        scratch(0),
                        query + key,
                        query,
                        key,
                    );
                    for block in 0..geometry.width / 32 {
                        let (previous, lanes) = if block % 2 == 0 {
                            (scratch(2), scratch(1))
                        } else {
                            (scratch(1), scratch(2))
                        };
                        for stage in 0..3 {
                            crate::attention_qk::accumulate::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(scratch(0), query + key),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(scratch(3), columns * 32),
                                arg(scratch(4), columns * 32),
                                geometry.width,
                                cap,
                                geometry.query_heads,
                                geometry.kv_heads,
                                block,
                                stage,
                            );
                            count += 1;
                        }
                    }
                    let (last, other) = if (geometry.width / 32 - 1) % 2 == 0 {
                        (scratch(1), scratch(2))
                    } else {
                        (scratch(2), scratch(1))
                    };
                    count += self
                        .inner
                        .attention_reduce(client, last, other, scratch(5), columns);
                    crate::attention_qk::scale::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(scratch(5), columns * 2),
                        *scale,
                        columns,
                    );
                    count += 1;
                    count += self.copy_dense(
                        client,
                        scratch(5),
                        columns * 2,
                        0,
                        &output(0).handle,
                        columns,
                        0,
                        columns,
                    );
                    crate::resident_bindings::masked_scores_rows::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(scratch(5), columns * 2),
                        arg(&output(1).handle, columns),
                        arg(&self.positions, self.rows),
                        geometry.query_heads / self.rows,
                        self.rows,
                        cap,
                    );
                    Ok(count + 1)
                }
                PlanOperation::AttentionSoftmax { heads, .. } => self.inner.softmax_masked(
                    client,
                    input(0),
                    output(0),
                    *heads,
                    self.inner.plan.prefix_capacity(),
                    CausalMask::Rows {
                        positions: &self.positions,
                        heads_per_row: *heads / self.rows,
                    },
                ),
                PlanOperation::AttentionValue {
                    state,
                    geometry,
                    padding,
                    ..
                } => {
                    let (_, values) = self.kv(*state);
                    let cap = self.inner.plan.prefix_capacity();
                    let probabilities = geometry.query_heads * cap;
                    let value = cap * geometry.kv_heads * geometry.width;
                    let columns = geometry.query_heads * geometry.width;
                    let mut count = self.copy_dense(
                        client,
                        &input(0).handle,
                        probabilities,
                        0,
                        scratch(0),
                        probabilities + value,
                        0,
                        probabilities,
                    );
                    count += self.copy_dense(
                        client,
                        &values.handle,
                        value,
                        0,
                        scratch(0),
                        probabilities + value,
                        probabilities,
                        value,
                    );
                    // Preserve the original source's padded FMA count, including
                    // its signed-zero behavior beyond the active prefix.
                    let blocks = parameters
                        .iter()
                        .map(|parameter| parameter.valid_tokens.div_ceil(*padding) * padding / 32)
                        .max()
                        .expect("validated native rows");
                    for block in 0..blocks {
                        let (previous, lanes) = if block % 2 == 0 {
                            (scratch(2), scratch(1))
                        } else {
                            (scratch(1), scratch(2))
                        };
                        for stage in 0..3 {
                            crate::attention_value::accumulate_rows::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(scratch(0), probabilities + value),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(scratch(3), columns * 32),
                                arg(scratch(4), columns * 32),
                                arg(&self.positions, self.rows),
                                block as u32,
                                geometry.width,
                                cap,
                                geometry.query_heads,
                                geometry.kv_heads,
                                self.rows,
                                *padding,
                                stage,
                            );
                            count += 1;
                        }
                    }
                    let (last, other) = if (blocks - 1) % 2 == 0 {
                        (scratch(1), scratch(2))
                    } else {
                        (scratch(2), scratch(1))
                    };
                    Ok(count
                        + self.inner.attention_reduce(
                            client,
                            last,
                            other,
                            &output(0).handle,
                            columns,
                        ))
                }
                PlanOperation::CheckFinite { .. } => {
                    Ok(self.certify(client, &input(0).handle, input(0).elements))
                }
                // Original source kernels, accumulation order, physical rounding
                // boundaries and attention padding remain the authority.
                _ => self.inner.launch_node(client, index, node, p, &self.packed),
            }
        }
    }

    fn rope(&self, client: &Client, node: &PlanNode, geometry: RopeGeometry) -> Result<usize> {
        let input = |i: usize| &self.inner.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.inner.values[node.outputs[i].0 as usize];
        let scratch = |i: usize| &self.inner.scratch[i];
        let g = geometry.attention;
        let heads = g.query_heads + g.kv_heads;
        let half = geometry.rotate / 2;
        let q = input(0).elements;
        let k = input(1).elements;
        let count = q + k + input(2).elements;
        let mut dispatches = 0;
        for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
            dispatches += self.copy_dense(
                client,
                &input(i).handle,
                input(i).elements,
                0,
                scratch(0),
                count,
                offset,
                input(i).elements,
            );
        }
        unsafe {
            for stage in 0..half {
                crate::rope_kv::angle_step_rows::launch_unchecked(
                    client,
                    grid(self.rows),
                    threads(),
                    arg(scratch(1), self.rows * half),
                    arg(&self.positions, self.rows),
                    geometry.theta_scale,
                    half,
                    self.rows,
                    stage,
                );
                dispatches += 1;
            }
            crate::source_trig::sincos_angles::launch_unchecked(
                client,
                grid(self.rows * half),
                threads(),
                arg(scratch(1), self.rows * half),
                arg(
                    &self.inner.trig_tables,
                    crate::source_trig::TRIG_TABLES.len(),
                ),
                arg(scratch(2), self.rows * half * 2),
                self.rows * half,
            );
            crate::rope_kv::rope_products::launch_unchecked(
                client,
                grid(heads * half),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                g.width,
                half,
                heads,
                g.query_heads / self.rows,
                self.rows,
            );
            crate::rope_kv::rope_sums::launch_unchecked(
                client,
                grid(count),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                arg(scratch(4), heads * half * 2),
                arg(scratch(6), count),
                g.width,
                half,
                heads,
                count,
                g.query_heads / self.rows,
                self.rows,
            );
            crate::rope_kv::rope_subtract::launch_unchecked(
                client,
                grid(heads * half * 2),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                arg(scratch(4), heads * half * 2),
                arg(scratch(5), heads * half * 2),
                g.width,
                half,
                heads,
                g.query_heads / self.rows,
                self.rows,
            );
            crate::rope_kv::rope_apply::launch_unchecked(
                client,
                grid(heads * half * 2),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                arg(scratch(4), heads * half * 2),
                arg(scratch(5), heads * half * 2),
                arg(scratch(6), count),
                g.width,
                half,
                heads,
                g.query_heads / self.rows,
                self.rows,
            );
            dispatches += 5;
        }
        for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
            dispatches += self.copy_dense(
                client,
                scratch(6),
                count,
                offset,
                &output(i).handle,
                output(i).elements,
                0,
                output(i).elements,
            );
        }
        Ok(dispatches)
    }

    fn project(
        &self,
        client: &Client,
        index: usize,
        tensor: &SourceTensor,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<usize> {
        let t = &self.inner.tensors[&tensor.name];
        unsafe {
            if tensor.kind == QuantKind::F32 {
                crate::batch_projection::f32_source_batched_projection::launch_unchecked(
                    client,
                    grid(output.elements),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(&input.handle, input.elements),
                    arg(&output.handle, output.elements),
                    tensor.columns,
                    tensor.rows,
                    self.rows,
                    true,
                );
                return Ok(1);
            }
            let layout = t.layout.expect("source quant layout");
            let blocks = self.rows * layout.blocks;
            let columns = self.rows * layout.columns;
            let rows = self.rows * layout.rows;
            let composite = self.rows * layout.composite_count;
            let params = blocks * PARAMETER_WORDS;
            let offsets = if layout.with_bias { composite } else { 1 };
            let s = |i: usize| &self.inner.scratch[i];
            let flags = &self.inner.node_status[index];
            let block_status = flags.block.as_ref().expect("quant block status");
            let row_status = flags.row.as_ref().expect("quant row status");
            let mut dispatches = self
                .projection_blocks
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::State("native projection cold binding absent".into()))?
                .launch(client)?;
            batch_composites::launch_unchecked(
                client,
                grid(rows),
                threads(),
                arg(&t.scales, layout.composite_count),
                arg(
                    &t.biases,
                    if layout.with_bias {
                        layout.composite_count
                    } else {
                        1
                    },
                ),
                arg(s(0), params),
                arg(s(3), composite),
                arg(s(4), offsets),
                arg(row_status, rows),
                layout.blocks,
                layout.rows,
                self.rows,
                layout.with_bias,
            );
            dispatches += 1;
            for (handle, count) in [(block_status, blocks), (row_status, rows)] {
                crate::resident_bindings::merge_status::launch_unchecked(
                    client,
                    grid(count),
                    threads(),
                    arg(handle, count),
                    arg(&self.inner.status, 1),
                    count,
                );
                dispatches += 1;
            }
            match tensor.kind {
                QuantKind::Q4_0 => crate::batch_quant_projection::q4_0_batched::launch_unchecked(
                    client,
                    grid(rows),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(s(3), composite),
                    arg(s(2), columns),
                    arg(&output.handle, rows),
                    layout.blocks,
                    layout.columns,
                    layout.rows,
                    self.rows,
                    true,
                ),
                QuantKind::Q4_1 => crate::batch_quant_projection::q4_1_batched::launch_unchecked(
                    client,
                    grid(rows),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(s(3), composite),
                    arg(s(4), offsets),
                    arg(s(2), columns),
                    arg(&output.handle, rows),
                    layout.blocks,
                    layout.columns,
                    layout.rows,
                    self.rows,
                    true,
                ),
                QuantKind::Q8_0 => {
                    crate::batch_quant_projection::q8_source_batched_projection::launch_unchecked(
                        client,
                        grid(rows),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(s(3), composite),
                        arg(s(2), columns),
                        arg(&output.handle, rows),
                        layout.blocks,
                        layout.columns,
                        layout.rows,
                        self.rows,
                        true,
                    )
                }
                QuantKind::Q6K => crate::batch_quant_projection::q6_k_batched::launch_unchecked(
                    client,
                    grid(rows),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(s(3), composite),
                    arg(s(2), columns),
                    arg(&output.handle, rows),
                    layout.blocks,
                    layout.columns,
                    layout.rows,
                    self.rows,
                    true,
                ),
                _ => return Err(Error::Unsupported("native resident source codec".into())),
            }
            dispatches += 1;
            Ok(dispatches)
        }
    }
}

#[cube(launch_unchecked)]
fn batch_device_tokens(
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    t4: &[u32],
    t5: &[u32],
    t6: &[u32],
    t7: &[u32],
    output: &mut [u32],
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows {
        let mut token = 0u32;
        if i == 0usize {
            token = t0[0];
        }
        if i == 1usize {
            token = t1[0];
        }
        if i == 2usize {
            token = t2[0];
        }
        if i == 3usize {
            token = t3[0];
        }
        if i == 4usize {
            token = t4[0];
        }
        if i == 5usize {
            token = t5[0];
        }
        if i == 6usize {
            token = t6[0];
        }
        if i == 7usize {
            token = t7[0];
        }
        output[i] = token;
    }
}

#[cube(launch_unchecked)]
fn batch_tokens(
    output: &mut [u32],
    t0: u32,
    t1: u32,
    t2: u32,
    t3: u32,
    t4: u32,
    t5: u32,
    t6: u32,
    t7: u32,
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows {
        let mut value = t0;
        if i == 1usize {
            value = t1;
        }
        if i == 2usize {
            value = t2;
        }
        if i == 3usize {
            value = t3;
        }
        if i == 4usize {
            value = t4;
        }
        if i == 5usize {
            value = t5;
        }
        if i == 6usize {
            value = t6;
        }
        if i == 7usize {
            value = t7;
        }
        output[i] = value;
    }
}

#[cube(launch_unchecked)]
fn batch_view(
    input: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] source_row: usize,
    #[comptime] offset: usize,
    #[comptime] view_rows: usize,
    #[comptime] width: usize,
    #[comptime] stride: usize,
    #[comptime] batch: usize,
) {
    let i = ABSOLUTE_POS;
    let row_elements = view_rows * width;
    if i < batch * row_elements {
        let request = i / row_elements;
        let local = i % row_elements;
        let bits = input[request * source_row + offset + (local / width) * stride + local % width];
        output[i] = bits;
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            status[0].fetch_or(2u32);
        }
    }
}

/// Every row appends at its own position and writes canonical future KV zeros.
#[cube(launch_unchecked)]
fn append_kv_rows(
    previous_k: &[u32],
    previous_v: &[u32],
    k: &[u32],
    v: &[u32],
    candidate_k: &mut [u32],
    candidate_v: &mut [u32],
    positions: &[u32],
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] batch: usize,
) {
    let i = ABSOLUTE_POS;
    if i < capacity * batch * slab {
        let token = i / (batch * slab);
        let row = (i / slab) % batch;
        let position = positions[row] as usize;
        let mut key = 0u32;
        let mut value = 0u32;
        if token < position {
            key = previous_k[i];
            value = previous_v[i];
        } else if token == position {
            key = k[row * slab + i % slab];
            value = v[row * slab + i % slab];
        }
        candidate_k[i] = key;
        candidate_v[i] = value;
    }
}

/// Private [T,H,D] handles ↔ physically packed [T,B,H,D] workspace.
#[cube(launch_unchecked)]
fn transport_kv(
    private_k: &mut [u32],
    private_v: &mut [u32],
    dense_k: &mut [u32],
    dense_v: &mut [u32],
    #[comptime] row: usize,
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] batch: usize,
    #[comptime] scatter: bool,
) {
    let i = ABSOLUTE_POS;
    if i < capacity * slab {
        let dense = ((i / slab) * batch + row) * slab + i % slab;
        if scatter {
            private_k[i] = dense_k[dense];
            private_v[i] = dense_v[dense];
        } else {
            dense_k[dense] = private_k[i];
            dense_v[dense] = private_v[i];
        }
    }
}

#[cube(launch_unchecked)]
fn batch_conv_products(
    activation: &[f32],
    weights: &[f32],
    history: &[f32],
    products: &mut [f32],
    candidate: &mut [f32],
    #[comptime] channels: usize,
    #[comptime] batch: usize,
) {
    let channel = ABSOLUTE_POS;
    if channel < batch * channels {
        let h = channel * 3;
        let w = (channel % channels) * 4;
        let dst = channel * 4;
        let c0 = history[h];
        let c1 = history[h + 1];
        let c2 = history[h + 2];
        let x = activation[channel];
        products[dst] = f32::cast_from(f64::cast_from(c0) * f64::cast_from(weights[w]));
        products[dst + 1] = f32::cast_from(f64::cast_from(c1) * f64::cast_from(weights[w + 1]));
        products[dst + 2] = f32::cast_from(f64::cast_from(c2) * f64::cast_from(weights[w + 2]));
        products[dst + 3] = f32::cast_from(f64::cast_from(x) * f64::cast_from(weights[w + 3]));
        candidate[h] = c1;
        candidate[h + 1] = c2;
        candidate[h + 2] = x;
    }
}

#[cube(launch_unchecked)]
fn batch_coefficient_input(
    alpha: &[u32],
    beta: &[u32],
    dt: &[u32],
    a: &[u32],
    output: &mut [u32],
    #[comptime] heads: usize,
    #[comptime] batch: usize,
) {
    let h = ABSOLUTE_POS;
    if h < batch * heads {
        output[h * 4] = alpha[h];
        output[h * 4 + 1] = beta[h];
        output[h * 4 + 2] = dt[h % heads];
        output[h * 4 + 3] = a[h % heads];
    }
}

#[cube(launch_unchecked)]
fn batch_composites(
    weight_scales: &[f32],
    biases: &[f32],
    parameters: &[f32],
    scales: &mut [f32],
    offsets: &mut [f32],
    status: &mut [u32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] with_bias: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows * batch {
        let request = row / rows;
        let source_row = row % rows;
        let mut flags = 0u32;
        for block in 0..blocks {
            let index = row * blocks + block;
            let source = source_row * blocks + block;
            let parameter = (request * blocks + block) * 4;
            let a = weight_scales[source];
            let b = parameters[parameter + 2];
            let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
            let scale = round_source(widen_source(a) * widen_source(b), sign);
            scales[index] = scale;
            if (u32::reinterpret(scale) & 0x7fffffffu32) >= 0x7f800000u32 {
                flags |= activation_quant::COMPOSITE_NONFINITE;
            }
            if with_bias {
                let a = biases[source];
                let b = parameters[parameter + 3];
                let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
                let offset = round_source(widen_source(a) * widen_source(b), sign);
                offsets[index] = offset;
                if (u32::reinterpret(offset) & 0x7fffffffu32) >= 0x7f800000u32 {
                    flags |= activation_quant::COMPOSITE_NONFINITE;
                }
            }
        }
        status[row] = flags;
    }
}

/// Original embedding equation, with only request/token offsets added.
#[cube(launch_unchecked)]
fn batch_embedding(
    packed: &[u8],
    tokens: &[u32],
    output: &mut [f32],
    status: &mut [u32],
    #[comptime] vocabulary: usize,
    #[comptime] width: usize,
    #[comptime] batch: usize,
    #[comptime] q6_k: bool,
) {
    let at = ABSOLUTE_POS;
    if at < batch * width {
        let request = at / width;
        let i = at % width;
        let mut flags = 0u32;
        let mut decoded = 0.0f32;
        if tokens[request] >= u32::cast_from(vocabulary) {
            flags = crate::embedding::TOKEN_OUT_OF_RANGE;
        } else {
            let row = usize::cast_from(tokens[request]);
            let (scale_bits, code, local_scale) = if q6_k {
                let base = (row * (width / 256) + i / 256) * 210;
                let scale_bits = u32::cast_from(packed[base + 208])
                    | (u32::cast_from(packed[base + 209]) << 8u32);
                let half = (i % 256) / 128;
                let group = (i % 128) / 32;
                let lane = i % 32;
                let low = u32::cast_from(packed[base + half * 64 + (group % 2) * 32 + lane]);
                let high = u32::cast_from(packed[base + 128 + half * 32 + lane]);
                let mut nibble = low & 15u32;
                if group >= 2usize {
                    nibble = low >> 4u32;
                }
                let code =
                    i32::cast_from(nibble | (((high >> u32::cast_from(2 * group)) & 3u32) << 4u32))
                        - 32i32;
                let local_scale = crate::embedding::signed_byte(
                    packed[base + 192 + half * 8 + group * 2 + lane / 16],
                );
                (scale_bits, code, local_scale)
            } else {
                let base = (row * (width / 32) + i / 32) * 34;
                let scale_bits =
                    u32::cast_from(packed[base]) | (u32::cast_from(packed[base + 1]) << 8u32);
                (
                    scale_bits,
                    crate::embedding::signed_byte(packed[base + 2 + i % 32]),
                    1i32,
                )
            };
            if (scale_bits & 0x7c00u32) == 0x7c00u32 {
                flags = crate::embedding::NONFINITE_SCALE;
            } else {
                let d = crate::embedding::half_to_float(scale_bits);
                let scale = f32::cast_from(local_scale);
                let value = f32::cast_from(code);
                let sign =
                    (u32::reinterpret(d) ^ u32::reinterpret(scale) ^ u32::reinterpret(value))
                        & 0x80000000u32;
                decoded = round_source(
                    widen_source(d) * widen_source(scale) * widen_source(value),
                    sign,
                );
                if (u32::reinterpret(decoded) & 0x7f800000u32) == 0x7f800000u32 {
                    flags = crate::embedding::NONFINITE_OUTPUT;
                    decoded = 0.0f32;
                }
            }
        }
        output[at] = decoded;
        status[at] = flags;
    }
}
