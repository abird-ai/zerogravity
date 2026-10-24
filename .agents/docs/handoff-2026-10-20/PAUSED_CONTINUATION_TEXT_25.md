# Paused continuation text — volume 25

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/proposed/crates/zerogravity-gpu/src/resident_program/batch.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 74039; SHA256 `f317c62ae7441b53bbb5a252aab68378db8b5a6413dfe14fe8bf113ab3007f5f`.

````text
//! Batch-shaped physical lowering of the original, source-bound ForwardPlan.
//!
//! No model schedule, session authority or Ready constructor lives here. The
//! physical owner admits every allocation, retains original leases before
//! launch, and publishes independent roots only after real completion.

use super::*;
use crate::activation_quant::{self, PARAMETER_WORDS};
use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
use crate::resident_state_transport::{
    Direction, PreparedCertifiedScatter, PreparedTransport, Row, Shape,
};

enum PreparedStateTransport {
    Opaque(PreparedTransport),
    Certified(PreparedCertifiedScatter),
}
impl PreparedStateTransport {
    fn launch(&self, client: &Client) -> Result<usize> {
        match self {
            Self::Opaque(transport) => transport.launch(client),
            Self::Certified(transport) => transport.launch(client),
        }
    }
}
use crate::source_float::{round_source, widen_source};

fn transport_shape(layout: StateLayout) -> Shape {
    match layout {
        StateLayout::Vector { elements } => Shape::Vector { words: elements },
        StateLayout::Kv { slab, capacity } => Shape::Kv { slab, capacity },
    }
}

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
    #[cfg(all(feature = "qualification", feature = "cuda"))]
    graph_generation: u64,
    #[cfg(all(feature = "qualification", feature = "cuda"))]
    graph_pin: Arc<()>,
}

/// Physical kernel launch sites in the existing node body, not graph enqueues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchBodyReceipt {
    pub dispatches: usize,
    pub fused_dispatches: usize,
}

#[cfg(all(feature = "qualification", feature = "cuda"))]
mod graph;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use graph::{BatchGraphProfile, BatchGraphReplay};
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
mod state_transport_qualification;

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
            crate::resident_state_transport::geometry(transport_shape(state.layout), rows)?;
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
        for state in &preflight.states {
            crate::resident_state_transport::ensure_certified_capability(
                client,
                transport_shape(state.layout),
                rows,
            )?;
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
        #[cfg(feature = "cuda")]
        inner.preflight_silu(client, rows)?;
        Ok(Self {
            inner,
            rows,
            nodes,
            packed,
            positions,
            greedy,
            projection_blocks,
            owned,
            #[cfg(all(feature = "qualification", feature = "cuda"))]
            graph_generation: graph::next_generation()?,
            #[cfg(all(feature = "qualification", feature = "cuda"))]
            graph_pin: Arc::new(()),
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
        #[cfg(feature = "cuda")]
        self.inner.preflight_silu(client, self.rows)?;
        let mut priors = Vec::new();
        let mut candidates = Vec::new();
        for (parameter, binding) in parameters.iter().zip(bindings) {
            self.inner
                .validate_state_runtime(client, *parameter, binding)?;
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
        // Both directions and every domain are checked before the enclosing
        // original physical submission begins. No allocation/upload occurs.
        self.prepare_state_transports(client, bindings, false)?;
        self.prepare_state_transports(client, bindings, true)?;
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
        let prefix_dispatches = self.launch_inputs(client, parameters, bindings, device_tokens)?;
        let body = self.launch_body(client, parameters, output)?;
        self.launch_outputs(client, bindings, output, prefix_dispatches, body)
    }

    fn launch_inputs(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        device_tokens: Option<&[DeviceBuffer]>,
    ) -> Result<usize> {
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
        Ok(3 + self.transport_state(client, bindings, false)?)
    }

    /// Original nodes and their certifications, in unchanged source order.
    pub(crate) fn launch_body(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        output: ResidentOutput,
    ) -> Result<BatchBodyReceipt> {
        let mut dispatches = 0;
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
        Ok(BatchBodyReceipt {
            dispatches,
            fused_dispatches,
        })
    }

    fn launch_outputs(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        output: ResidentOutput,
        prefix_dispatches: usize,
        body: BatchBodyReceipt,
    ) -> Result<ResidentSubmission> {
        let mut dispatches = prefix_dispatches + body.dispatches;
        let fused_dispatches = body.fused_dispatches;
        let count = output.nodes(&self.inner.plan).len();
        dispatches += self.transport_state(client, bindings, true)?;
        let mut proposals = Vec::new();
        for row in bindings {
            for version in self.inner.plan.proposed_domains() {
                let binding = row
                    .iter()
                    .find(|b| b.slot == version.slot)
                    .expect("validated domain");
                // Typed scatter copied and certified EVERY original candidate
                // word in the same launches; keep the exact all-domain proposal.
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

    fn prepare_state_transports(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        scatter: bool,
    ) -> Result<Vec<PreparedStateTransport>> {
        if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
            return Err(Error::State("state transport domain/row coverage".into()));
        }
        let mut prepared = Vec::with_capacity(self.packed.len());
        for packed in &self.packed {
            let dense = if scatter {
                &packed.candidate
            } else {
                &packed.prior
            };
            let private = bindings
                .iter()
                .map(|row| {
                    let binding = row
                        .iter()
                        .find(|binding| binding.slot == packed.slot)
                        .ok_or_else(|| {
                            Error::State("state transport original domain absent".into())
                        })?;
                    Ok(if scatter {
                        &binding.candidate
                    } else {
                        &binding.prior
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            match dense {
                StateBuffers::Vector(dense) => {
                    let words = dense.elements / self.rows;
                    if words.checked_mul(self.rows) != Some(dense.elements) {
                        return Err(Error::State("state transport dense vector geometry".into()));
                    }
                    let rows = private
                        .iter()
                        .map(|value| match value {
                            StateBuffers::Vector(value) if value.elements == words => Ok(Row {
                                first: &value.handle,
                                second: None,
                            }),
                            _ => Err(Error::State("state transport vector row changed".into())),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    prepared.push(if scatter {
                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
                            client,
                            &rows,
                            &dense.handle,
                            None,
                            Shape::Vector { words },
                            &self.inner.status,
                        )?)
                    } else {
                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
                            client,
                            &rows,
                            &dense.handle,
                            None,
                            Shape::Vector { words },
                            Direction::Gather,
                        )?)
                    });
                }
                StateBuffers::Kv { keys, values } => {
                    let capacity = self.inner.plan.prefix_capacity();
                    let stride = self
                        .rows
                        .checked_mul(capacity)
                        .filter(|&n| n != 0)
                        .ok_or_else(|| Error::Limit("state transport KV stride".into()))?;
                    let slab = keys.elements / stride;
                    let words = slab
                        .checked_mul(capacity)
                        .ok_or_else(|| Error::Limit("state transport KV row words".into()))?;
                    if slab.checked_mul(stride) != Some(keys.elements)
                        || keys.elements != values.elements
                    {
                        return Err(Error::State("state transport dense KV geometry".into()));
                    }
                    let rows = private
                        .iter()
                        .map(|value| match value {
                            StateBuffers::Kv { keys, values }
                                if keys.elements == words && values.elements == words =>
                            {
                                Ok(Row {
                                    first: &keys.handle,
                                    second: Some(&values.handle),
                                })
                            }
                            _ => Err(Error::State("state transport KV row changed".into())),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    prepared.push(if scatter {
                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
                            client,
                            &rows,
                            &keys.handle,
                            Some(&values.handle),
                            Shape::Kv { slab, capacity },
                            &self.inner.status,
                        )?)
                    } else {
                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
                            client,
                            &rows,
                            &keys.handle,
                            Some(&values.handle),
                            Shape::Kv { slab, capacity },
                            Direction::Gather,
                        )?)
                    });
                }
            }
        }
        Ok(prepared)
    }

    fn transport_state(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        scatter: bool,
    ) -> Result<usize> {
        self.prepare_state_transports(client, bindings, scatter)?
            .iter()
            .try_fold(0, |count, transport| Ok(count + transport.launch(client)?))
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
                | PlanOperation::TypedOrdinaryRms { .. }
                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
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
                PlanOperation::Silu { .. } => self.inner.launch_silu(client, node, self.rows),
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
        let ordered_angles = crate::rope_angles::PreparedAngles::rows_for_cuda(
            client,
            scratch(1),
            &self.positions,
            geometry.theta_scale,
            half,
            self.rows,
        )?;
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
            dispatches += if let Some(angles) = ordered_angles {
                angles.launch(client)?
            } else {
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
                }
                half
            };
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
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
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

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/proposed/crates/zerogravity-gpu/src/resident_program/silu/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 955; SHA256 `68323ff7eef6ae3f0c80020e98f0642ebfd079418fc8e78066825ed369e6158d`.

````text
use super::*;

#[test]
fn packed_domains_keep_actual_rows_instead_of_flattening_to_b1() -> Result<()> {
    for rows in [2, 4, 8] {
        for width in [1, 31, 32, 33, 65_536] {
            assert_eq!(ordered_width(rows, rows * width)?, Some(width));
        }
    }
    // This packed B8 shape exceeds the B1 width limit, but every genuine row
    // remains in the accepted domain. No padding/request is introduced.
    assert_eq!(ordered_width(8, 8 * 65_536)?, Some(65_536));
    assert_eq!(ordered_width(1, 8 * 65_536)?, None);
    Ok(())
}

#[test]
fn only_known_geometry_uses_original_route_and_partition_changes_refuse() -> Result<()> {
    for rows in [1, 2, 4, 8] {
        assert_eq!(ordered_width(rows, 0)?, None);
        assert_eq!(ordered_width(rows, rows * 65_537)?, None);
    }
    for (rows, words) in [(0, 32), (3, 96), (16, 512), (2, 65), (4, 129), (8, 257)] {
        assert!(ordered_width(rows, words).is_err());
    }
    Ok(())
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/proposed/crates/zerogravity-gpu/src/resident_program/silu.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6001; SHA256 `05aebe1aae9846d4509284793d8e8a634f1be3aa7ce4490f37bda19c832f4918`.

````text
//! Same original SiLU buffers/certificates; no persistent binding metadata.
use super::*;

#[cfg(feature = "cuda")]
fn ordered_width(rows: usize, elements: usize) -> Result<Option<usize>> {
    if !matches!(rows, 1 | 2 | 4 | 8) || !elements.is_multiple_of(rows) {
        return Err(Error::State("resident SiLU row partition changed".into()));
    }
    let width = elements / rows;
    Ok(crate::ordered_silu::supports_geometry(rows, width).then_some(width))
}

impl ResidentProgram {
    #[cfg(feature = "cuda")]
    fn ordered_silu_binding(
        &self,
        client: &Client,
        node: &PlanNode,
        rows: usize,
    ) -> Result<Option<crate::ordered_silu::PreparedOrderedSilu>> {
        if !crate::activation_quant_fused::is_cuda_service(client) {
            return Ok(None);
        }
        if self.plan.numerical_profile()
            != zerogravity_core::accelerator::program::ProgramNumericalProfile::GgufSourceV1
            || self.plan.is_operator_slice()
        {
            return Err(Error::Unsupported(
                "resident SiLU numerical contract".into(),
            ));
        }
        let PlanOperation::Silu { multiplied } = &node.operation else {
            return Err(Error::State("resident SiLU operation changed".into()));
        };
        let input = |index: usize| -> Result<&DeviceBuffer> {
            node.inputs
                .get(index)
                .and_then(|id| self.values.get(id.0 as usize))
                .ok_or_else(|| Error::State("resident SiLU input binding absent".into()))
        };
        let input = input(0)?;
        let output = node
            .outputs
            .first()
            .and_then(|id| self.values.get(id.0 as usize))
            .ok_or_else(|| Error::State("resident SiLU output binding absent".into()))?;
        let multiplier = if *multiplied {
            Some(
                node.inputs
                    .get(1)
                    .and_then(|id| self.values.get(id.0 as usize))
                    .ok_or_else(|| Error::State("resident SiLU multiplier absent".into()))?,
            )
        } else {
            None
        };
        if output.elements != input.elements
            || multiplier.is_some_and(|value| value.elements != input.elements)
        {
            return Err(Error::State("resident SiLU value geometry changed".into()));
        }
        let Some(width) = ordered_width(rows, input.elements)? else {
            // Only the immutable closed geometry selects the original route.
            // Hardware/range/alias errors from prepare always propagate.
            return Ok(None);
        };
        crate::ordered_silu::PreparedOrderedSilu::prepare(
            client,
            rows,
            width,
            crate::ordered_silu::Bindings {
                input: &input.handle,
                multiplier: multiplier.map(|value| &value.handle),
                output: &output.handle,
                trace: None,
            },
        )
        .map(Some)
    }

    /// Full immutable plan, including StateOnly-skipped tail nodes. Successful
    /// prepare uses only stack bindings and original Arc handle clones.
    #[cfg(feature = "cuda")]
    fn preflight_silu(&self, client: &Client, rows: usize) -> Result<()> {
        Self::require_legacy_contract(&self.plan)?;
        for node in self.plan.nodes() {
            if matches!(node.operation, PlanOperation::Silu { .. }) {
                self.ordered_silu_binding(client, node, rows)?;
            }
        }
        Ok(())
    }

    fn launch_silu(&self, client: &Client, node: &PlanNode, rows: usize) -> Result<usize> {
        let PlanOperation::Silu { multiplied } = &node.operation else {
            return Err(Error::State("resident SiLU operation changed".into()));
        };
        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
        let s = |i: usize| &self.scratch[i];
        let n = input(0).elements;
        // Repeat the exact cold binding before this node's first launch. No
        // mutable origin/range/rows, persistent table or Err fallback exists.
        #[cfg(feature = "cuda")]
        let ordered = self.ordered_silu_binding(client, node, rows)?;
        #[cfg(not(feature = "cuda"))]
        let _ = rows;
        // SAFETY: original typed plan/ranges and owning submission retain the
        // same original input/output/status/scratch leases as before factoring.
        unsafe {
            crate::resident_bindings::certify_interval::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(&input(0).handle, n),
                arg(&self.status, 1),
                -256.0,
                256.0,
                n,
            );
        }
        #[cfg(feature = "cuda")]
        if let Some(ordered) = ordered {
            return Ok(1 + ordered.launch(client)?);
        }
        // Original CUDA-outside-geometry and Vulkan body, byte-equivalent
        // modulo indentation to the pre-integration PlanOperation::Silu arm.
        unsafe {
            for stage in 0..12 {
                crate::silu::silu_step::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input(0).handle, n),
                    arg(s(0), n * crate::silu::SCRATCH),
                    arg(&output(0).handle, n),
                    n,
                    stage,
                );
            }
            if *multiplied {
                crate::silu::silu_gate::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input(1).handle, n),
                    arg(&output(0).handle, n),
                    n,
                );
            }
        }
        Ok(13 + usize::from(*multiplied))
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/proposed/crates/zerogravity-gpu/src/resident_program.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 105439; SHA256 `712dafddfd944c7932f71d99e4c750d3faef1044638f6cbbca883c7388558114`.

````text
//! H16 typed ForwardPlan -> resident physical-kernel bindings.
//!
//! This module contains no Qwen layer schedule and no activation mathematics on
//! the host. It does not admit work, fence a device, publish a root, or declare
//! Ready. The enclosing physical owner must admit the complete preflight,
//! retain `handles_for_attempt` BEFORE launch, and retain them after launch
//! until actual completion or quarantine. One mutable program is one execution
//! workspace; reuse is permitted only after that owner's completion fence.

use crate::{
    activation_quant::ActivationLayout,
    device_activation::DeviceActivation,
    host_descriptor::StaticSource,
    source_accumulation::{self, SourceAccumulation},
};
use cubecl::{client::Client, prelude::*, server::Handle};
use cubecl_common::device::ServiceId;
use cubecl_runtime::memory_management::ManagedMemoryId;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
};
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::Error;
use zerogravity_core::Result;
use zerogravity_core::accelerator::{
    MatVecRegion, SourceIdentity,
    program::{
        ForwardPlan, PlanNode, PlanOperation, RopeGeometry, SourceTensor, StateSlot, StateVersion,
        TensorView, ValueId,
    },
};
use zerogravity_core::source::TensorSource;
#[cfg(all(test, feature = "qualification"))]
use zerogravity_model_qwen35::LoadedQwenModel;

const SCRATCH_SLOTS: usize = 12;

mod batch;
pub(crate) use batch::BatchProgram;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use batch::{BatchBodyReceipt, BatchGraphProfile, BatchGraphReplay};

#[cfg(feature = "qualification")]
#[path = "resident_projection_qualification.rs"]
mod projection_qualification;

/// Allocation coordinates only. Original handles and admitted leases remain
/// the lifetime owners; this copyable metadata never pins or promotes memory.
#[derive(Clone, Debug)]
struct BufferRange {
    service: ServiceId,
    memory: ManagedMemoryId,
    bytes: Range<u64>,
}
impl BufferRange {
    fn new(handle: &Handle) -> Result<Self> {
        let start = handle.offset_start.unwrap_or(0);
        let end = handle.size().checked_sub(handle.offset_end.unwrap_or(0));
        let end = end
            .filter(|&end| start < end)
            .ok_or_else(|| Error::State("resident buffer byte range invalid".into()))?;
        Ok(Self {
            service: handle.service,
            memory: handle.clone().binding().memory.id(),
            bytes: start..end,
        })
    }
    fn overlaps(&self, other: &Self) -> bool {
        self.service == other.service
            && self.memory == other.memory
            && self.bytes.start < other.bytes.end
            && other.bytes.start < self.bytes.end
    }
}

/// Exact immutable ranges, prepared once. Distinct slices of a shared
/// allocation remain separate; only identical aliases are deduplicated.
struct OwnedRanges {
    service: ServiceId,
    by_memory: BTreeMap<ManagedMemoryId, Vec<Range<u64>>>,
}
impl OwnedRanges {
    fn new(service: ServiceId) -> Self {
        Self {
            service,
            by_memory: BTreeMap::new(),
        }
    }
    fn insert(&mut self, range: BufferRange) -> Result<()> {
        if range.service != self.service {
            return Err(Error::State("resident owned range service differs".into()));
        }
        let ranges = self.by_memory.entry(range.memory).or_default();
        if !ranges.contains(&range.bytes) {
            ranges.push(range.bytes);
        }
        Ok(())
    }
    fn overlaps(&self, range: &BufferRange) -> bool {
        range.service == self.service
            && self.by_memory.get(&range.memory).is_some_and(|ranges| {
                ranges
                    .iter()
                    .any(|r| r.start < range.bytes.end && range.bytes.start < r.end)
            })
    }
}
fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| Error::Limit("resident program byte sum overflow".into()))
    })
}
fn bytes(elements: usize, width: usize) -> Result<usize> {
    elements
        .checked_mul(width)
        .filter(|_| u32::try_from(elements).is_ok())
        .ok_or_else(|| Error::Limit("resident program buffer index/byte overflow".into()))
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}
fn arg(handle: &Handle, elements: usize) -> BufferArg {
    // SAFETY: all call sites are internal to the checked typed lowering;
    // preflight/state binding validates the physical byte ranges separately.
    unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
}

/// Every View is already physically packed by this lowerer. Contiguous can
/// therefore name that same immutable dense value without allocating/copying.
/// Resolve chains to earlier original values; never alias a range or a state.
fn contiguous_aliases(plan: &ForwardPlan) -> Result<Vec<Option<ValueId>>> {
    let mut aliases = vec![None; plan.values().len()];
    let mut produced = BTreeSet::new();
    for node in plan.nodes() {
        if matches!(node.operation, PlanOperation::Contiguous) {
            let ([input], [output]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                return Err(Error::Malformed("resident contiguous alias arity".into()));
            };
            let shape = |id: ValueId| {
                plan.values()
                    .get(id.0 as usize)
                    .ok_or_else(|| Error::Malformed("resident alias value index".into()))
            };
            if node.state.is_some()
                || input.0 >= output.0
                || !produced.contains(&input.0)
                || shape(*input)? != shape(*output)?
            {
                return Err(Error::State(
                    "resident contiguous alias is not an equal-shape earlier value".into(),
                ));
            }
            aliases[output.0 as usize] = Some(aliases[input.0 as usize].unwrap_or(*input));
        }
        produced.extend(node.outputs.iter().map(|id| id.0));
    }
    Ok(aliases)
}

#[derive(Clone, Debug)]
pub(crate) struct DeviceBuffer {
    pub handle: Handle,
    pub elements: usize,
}
impl DeviceBuffer {
    pub fn new(handle: Handle, elements: usize) -> Self {
        Self { handle, elements }
    }
    fn check(&self, client: &Client, expected: usize) -> Result<BufferRange> {
        let range = BufferRange::new(&self.handle)?;
        if self.elements != expected
            || range.service != client.service_id()
            || range.bytes.end - range.bytes.start < bytes(expected, 4)? as u64
        {
            return Err(Error::State(
                "resident program binding size/service mismatch".into(),
            ));
        }
        Ok(range)
    }
}
#[derive(Clone, Debug)]
pub(crate) enum StateBuffers {
    Vector(DeviceBuffer),
    Kv {
        keys: DeviceBuffer,
        values: DeviceBuffer,
    },
}
impl StateBuffers {
    pub fn buffers(&self) -> Vec<&DeviceBuffer> {
        match self {
            Self::Vector(v) => vec![v],
            Self::Kv { keys, values } => vec![keys, values],
        }
    }
    pub fn handles(&self) -> Vec<Handle> {
        match self {
            Self::Vector(v) => vec![v.handle.clone()],
            Self::Kv { keys, values } => vec![keys.handle.clone(), values.handle.clone()],
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct StateBinding {
    pub slot: StateSlot,
    pub prior: StateBuffers,
    pub candidate: StateBuffers,
    /// Committed prefix length; meaningful for paired KV only.
    pub prior_tokens: usize,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct RuntimeParameters {
    pub token: u32,
    pub position: usize,
    pub valid_tokens: usize,
}
enum CausalMask<'a> {
    Uniform(usize),
    Rows {
        positions: &'a Handle,
        heads_per_row: usize,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentOutput {
    StateOnly,
    Logits,
}
impl ResidentOutput {
    fn nodes(self, plan: &ForwardPlan) -> &[PlanNode] {
        match self {
            Self::StateOnly => plan.state_nodes(),
            Self::Logits => plan.nodes(),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum StateLayout {
    Vector { elements: usize },
    Kv { slab: usize, capacity: usize },
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct StateRequirement {
    pub slot: StateSlot,
    pub layout: StateLayout,
    pub bytes: usize,
}

/// Exact logical allocations, excluding allocator page rounding. Prior and
/// candidate state are external bindings and reported separately to prevent
/// double charging an already-resident committed root.
#[derive(Clone, Debug)]
pub(crate) struct ResidentPreflight {
    pub source: SourceIdentity,
    pub source_bytes: usize,
    pub value_bytes: usize,
    pub workspace_bytes: usize,
    pub status_bytes: usize,
    pub parameter_bytes: usize,
    pub prior_state_bytes: usize,
    pub candidate_state_bytes: usize,
    pub states: Vec<StateRequirement>,
    scratch: [usize; SCRATCH_SLOTS],
}
impl ResidentPreflight {
    pub fn owned_bytes(&self) -> Result<usize> {
        checked_sum([
            self.source_bytes,
            self.value_bytes,
            self.workspace_bytes,
            self.status_bytes,
            self.parameter_bytes,
        ])
    }
    pub fn total_with_state(&self) -> Result<usize> {
        checked_sum([
            self.owned_bytes()?,
            self.prior_state_bytes,
            self.candidate_state_bytes,
        ])
    }
}
struct ResidentTensor {
    packed: Handle,
    elements: usize,
    scales: Handle,
    biases: Handle,
    layout: Option<ActivationLayout>,
}
struct NodeStatus {
    block: Option<Handle>,
    row: Option<Handle>,
}

pub(crate) struct ResidentProgram {
    // The immutable original allocation remains alive for every resident handle.
    _source_owner: Arc<dyn TensorSource>,
    plan: ForwardPlan,
    preflight: ResidentPreflight,
    tensors: BTreeMap<String, ResidentTensor>,
    constants: Vec<Handle>,
    values: Vec<DeviceBuffer>,
    scratch: [Handle; SCRATCH_SLOTS],
    status: Handle,
    node_status: Vec<NodeStatus>,
    // Immutable per-operation views of existing originals. Preparation neither
    // allocates memory nor initializes values/statuses skipped by state-only.
    projections: Vec<Option<Box<DeviceActivation>>>,
    token: Handle,
    exp_tables: Handle,
    trig_tables: Handle,
    owned_ranges: OwnedRanges,
}
#[derive(Clone, Debug)]
pub(crate) struct ResidentSubmission {
    pub logits: Option<DeviceBuffer>,
    pub proposals: Vec<(StateVersion, StateBuffers)>,
    pub status: Handle,
    /// Every dynamic-quant status pair, in plan order. Zero is success.
    pub quant_status: Vec<(Handle, usize, Handle, usize)>,
    pub embedding_status: Vec<(Handle, usize)>,
    pub dispatches: usize,
    /// Conservative subset: actual native multi-row projection launches.
    pub fused_dispatches: usize,
}

impl ResidentProgram {
    fn require_legacy_contract(plan: &ForwardPlan) -> Result<()> {
        use zerogravity_core::accelerator::program::{
            ActivationPrecision, ProgramNumericalProfile,
        };
        if plan.numerical_profile() != ProgramNumericalProfile::GgufSourceV1
            || plan.is_operator_slice()
            || plan
                .precisions()
                .iter()
                .any(|precision| *precision != ActivationPrecision::F32)
            || plan.nodes().iter().any(|node| {
                matches!(
                    node.operation,
                    PlanOperation::TypedEmbed { .. }
                        | PlanOperation::TypedProject { .. }
                        | PlanOperation::TypedOrdinaryRms { .. }
                        | PlanOperation::TypedCausalConvSilu { .. }
                )
            })
        {
            return Err(Error::Unsupported(
                "resident backend has no qualified typed numerical contract".into(),
            ));
        }
        Ok(())
    }
    /// Pure metadata/source validation; no GPU allocation or live math.
    pub fn preflight(model: &dyn TensorSource, plan: &ForwardPlan) -> Result<ResidentPreflight> {
        Self::require_legacy_contract(plan)?;
        if plan.source() != model.source_identity()
            || !plan.source().is_bound()
            || plan.frontier_delta() != 1
            || plan.prefix_capacity() == 0
            || plan.prefix_capacity() > 512
            || !plan.prefix_capacity().is_multiple_of(256)
        {
            return Err(Error::Unsupported(
                "resident plan source/capacity/frontier unsupported".into(),
            ));
        }
        let aliases = contiguous_aliases(plan)?;
        let mut source_bytes = checked_sum(plan.constants().iter().map(|c| c.values.len() * 4))?;
        source_bytes = checked_sum([
            source_bytes,
            crate::source_transcendentals::TABLES.len() * 8,
            crate::source_trig::TRIG_TABLES.len() * 8,
            4,
        ])?; // shared no-bias dummy
        let mut seen = BTreeMap::<String, SourceTensor>::new();
        let mut projected = BTreeSet::new();
        let mut scratch = [4usize; SCRATCH_SLOTS];
        let mut status_bytes = 4usize;
        let mut state_specs = BTreeMap::new();
        for constant in plan.constants() {
            if constant.source != plan.source() || constant.values.iter().any(|x| !x.is_finite()) {
                return Err(Error::Malformed(
                    "resident static vector source/finite mismatch".into(),
                ));
            }
        }
        for node in plan.nodes() {
            use zerogravity_core::accelerator::program::ParameterSlot;
            let parameter = match &node.operation {
                PlanOperation::Embed { token, .. } => Some((token.0, ParameterSlot::Token)),
                PlanOperation::Rope { position, .. }
                | PlanOperation::KvPrefixGuard { position, .. } => {
                    Some((position.0, ParameterSlot::Position))
                }
                PlanOperation::AttentionQk { valid_tokens, .. }
                | PlanOperation::AttentionSoftmax { valid_tokens, .. }
                | PlanOperation::AttentionValue { valid_tokens, .. } => {
                    Some((valid_tokens.0, ParameterSlot::ValidTokens))
                }
                _ => None,
            };
            if parameter.is_some_and(|(actual, expected)| actual != expected) {
                return Err(Error::Unsupported(
                    "resident symbolic parameter differs from source schedule".into(),
                ));
            }
            match &node.operation {
                PlanOperation::Recurrent { width }
                    if *width == 0 || !width.is_multiple_of(32) || *width > 256 =>
                {
                    return Err(Error::Unsupported(
                        "resident recurrence requires source width 32..256 multiple32".into(),
                    ));
                }
                PlanOperation::AttentionQk { geometry, .. }
                | PlanOperation::AttentionValue { geometry, .. }
                    if geometry.width == 0
                        || !geometry.width.is_multiple_of(32)
                        || geometry.width > 512
                        || geometry.query_heads > 64
                        || geometry.kv_heads > 64 =>
                {
                    return Err(Error::Unsupported(
                        "resident attention exceeds source geometry bounds".into(),
                    ));
                }
                PlanOperation::Rope { geometry, .. }
                    if geometry.rotate > 64
                        || geometry.attention.width > 512
                        || geometry.attention.query_heads > 64
                        || geometry.attention.kv_heads > 64 =>
                {
                    return Err(Error::Unsupported(
                        "resident RoPE exceeds source geometry bounds".into(),
                    ));
                }
                PlanOperation::Coefficients { dt, a, .. }
                    if plan.constants()[dt.0 as usize]
                        .values
                        .iter()
                        .any(|x| x.abs() > 128.0)
                        || plan.constants()[a.0 as usize]
                            .values
                            .iter()
                            .any(|x| *x > 0.0 || x.abs() > 1_000_000.0) =>
                {
                    return Err(Error::Unsupported(
                        "resident source GDN dt/A outside attested bounds".into(),
                    ));
                }
                _ => {}
            }
            if let PlanOperation::Embed { tensor, .. } | PlanOperation::Project { tensor } =
                &node.operation
            {
                if tensor.source != plan.source() {
                    return Err(Error::State("resident tensor source mismatch".into()));
                }
                // Operator support is independent of source-storage reuse. A
                // tensor first seen by Project must still satisfy Embed's
                // narrower source codec/geometry contract on every use.
                let embedding = if matches!(node.operation, PlanOperation::Embed { .. }) {
                    Some(crate::embedding::EmbeddingLayout::new(
                        tensor.kind,
                        tensor.rows,
                        tensor.columns,
                    )?)
                } else {
                    None
                };
                tensor.validate_source(model)?;
                if matches!(node.operation, PlanOperation::Project { .. })
                    && projected.insert(tensor.name.clone())
                {
                    let block = if tensor.kind == QuantKind::Q6K {
                        256
                    } else {
                        32
                    };
                    if tensor.kind != QuantKind::F32 {
                        source_bytes = checked_sum([
                            source_bytes,
                            bytes(
                                (tensor.columns / block) * tensor.rows,
                                if tensor.kind == QuantKind::Q4_1 { 8 } else { 4 },
                            )?,
                        ])?;
                    }
                }
                if let Some(prior) = seen.get(&tensor.name) {
                    if prior != tensor {
                        return Err(Error::State(
                            "resident source name reused with different geometry".into(),
                        ));
                    }
                } else {
                    let region = region(model, tensor)?;
                    let source = if let Some(embedding) = embedding {
                        embedding.source_bytes
                    } else {
                        StaticSource::gpu_bytes_for(region)?
                    };
                    source_bytes = checked_sum([source_bytes, source])?;
                    seen.insert(tensor.name.clone(), tensor.clone());
                }
            }
            let n = |id: ValueId| -> Result<usize> {
                plan.values()
                    .get(id.0 as usize)
                    .ok_or_else(|| Error::Malformed("resident plan value index invalid".into()))?
                    .capacity_elements()
            };
            let input = |i: usize| -> Result<usize> {
                n(*node
                    .inputs
                    .get(i)
                    .ok_or_else(|| Error::Malformed("resident plan input missing".into()))?)
            };
            let cap = plan.prefix_capacity();
            if let PlanOperation::AttentionQk {
                state, geometry, ..
            }
            | PlanOperation::AttentionValue {
                state, geometry, ..
            } = &node.operation
            {
                let slab = geometry
                    .kv_heads
                    .checked_mul(geometry.width)
                    .ok_or_else(|| Error::Limit("resident attention KV slab overflow".into()))?;
                if !matches!(
                    state_specs.get(&state.slot),
                    Some(StateLayout::Kv { slab: stored, capacity })
                        if *stored == slab && *capacity == cap
                ) {
                    return Err(Error::State(
                        "resident attention KV slab differs from geometry".into(),
                    ));
                }
            }
            for id in node.inputs.iter().chain(node.outputs.iter()) {
                if n(*id)? > crate::program_kernels::MAX_WORD_OPS {
                    return Err(Error::Unsupported(
                        "resident value exceeds bounded transport domain".into(),
                    ));
                }
            }
            match &node.operation {
                PlanOperation::Rms { groups, .. }
                    if *groups > 4096 || input(0)? / groups > 16_384 =>
                {
                    return Err(Error::Unsupported(
                        "resident RMS exceeds source bounds".into(),
                    ));
                }
                PlanOperation::Silu { .. }
                | PlanOperation::Add
                | PlanOperation::SigmoidGate
                | PlanOperation::Coefficients { .. }
                    if input(0)? > 65_536 =>
                {
                    return Err(Error::Unsupported(
                        "resident elementwise/coefficient width exceeds source bounds".into(),
                    ));
                }
                PlanOperation::Conv { .. } if input(0)? > 16_384 => {
                    return Err(Error::Unsupported(
                        "resident convolution exceeds source bounds".into(),
                    ));
                }
                PlanOperation::Recurrent { width } if input(0)? / width > 64 => {
                    return Err(Error::Unsupported(
                        "resident recurrent head count exceeds source bounds".into(),
                    ));
                }
                PlanOperation::AttentionSoftmax { heads, .. } if *heads > 64 => {
                    return Err(Error::Unsupported(
                        "resident softmax head count exceeds source bounds".into(),
                    ));
                }
                _ => {}
            }
            let sizes: Vec<usize> = match &node.operation {
                PlanOperation::Embed { tensor, .. } => {
                    status_bytes = checked_sum([status_bytes, tensor.columns * 4])?;
                    vec![]
                }
                PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32 => {
                    let l = ActivationLayout::new(tensor.kind, tensor.columns, tensor.rows)?;
                    status_bytes = checked_sum([status_bytes, (l.blocks + l.rows) * 4])?;
                    vec![
                        l.blocks * 16,
                        l.columns * 4,
                        l.columns * 4,
                        l.composite_count * 4,
                        if l.with_bias {
                            l.composite_count * 4
                        } else {
                            4
                        },
                    ]
                }
                PlanOperation::Rms { groups, .. } => vec![input(0)? * 4, groups * 8],
                PlanOperation::Silu { .. } => vec![input(0)? * crate::silu::SCRATCH * 4],
                PlanOperation::SigmoidGate => vec![
                    input(0)? * crate::elementwise::SIGMOID_SCRATCH * 4,
                    input(0)? * 8,
                ],
                PlanOperation::Conv { .. } => {
                    state_specs.insert(
                        node.state
                            .ok_or_else(|| Error::Malformed("conv missing effect".into()))?
                            .read
                            .slot,
                        StateLayout::Vector {
                            elements: input(0)? * 3,
                        },
                    );
                    vec![input(0)? * 16]
                }
                PlanOperation::Coefficients { .. } => vec![
                    input(0)? * 16,
                    input(0)? * 24,
                    input(0)? * 24,
                    input(0)? * 16,
                ],
                PlanOperation::Recurrent { width } => {
                    let v = input(0)?;
                    state_specs.insert(
                        node.state
                            .ok_or_else(|| Error::Malformed("recurrence missing effect".into()))?
                            .read
                            .slot,
                        StateLayout::Vector {
                            elements: v * width,
                        },
                    );
                    vec![v * 12, v * width * 4, v * 4, v * 128, v * 64]
                }
                PlanOperation::Rope { geometry, .. } => {
                    let g = geometry.attention;
                    let half = geometry.rotate / 2;
                    let h = g.query_heads + g.kv_heads;
                    let count = input(0)? + input(1)? + input(2)?;
                    vec![
                        count * 4,
                        half * 4,
                        half * 8,
                        h * half * 8,
                        h * half * 16,
                        h * half * 16,
                        count * 4,
                    ]
                }
                PlanOperation::AppendKv => {
                    let slab = input(0)?;
                    state_specs.insert(
                        node.state
                            .ok_or_else(|| Error::Malformed("KV append missing effect".into()))?
                            .read
                            .slot,
                        StateLayout::Kv {
                            slab,
                            capacity: cap,
                        },
                    );
                    vec![]
                }
                PlanOperation::AttentionQk { geometry: g, .. } => {
                    let c = g.query_heads * cap;
                    vec![
                        (g.query_heads * g.width + cap * g.kv_heads * g.width) * 4,
                        c * 128,
                        c * 128,
                        c * 256,
                        c * 256,
                        c * 8,
                    ]
                }
                PlanOperation::AttentionSoftmax { heads: h, .. } => {
                    let n = h * cap;
                    let groups = cap.div_ceil(8);
                    vec![
                        n * 40,
                        n * 4,
                        n * 8,
                        n * 8,
                        h * 4,
                        h * groups * 16,
                        h * groups * 8,
                        n * 4,
                        h * 8,
                        h * 8,
                        h * 4,
                    ]
                }
                PlanOperation::AttentionValue {
                    geometry: g,
                    padding,
                    ..
                } => {
                    if *padding != 256 {
                        return Err(Error::Unsupported(
                            "resident attention value padding differs from source".into(),
                        ));
                    }
                    let c = g.query_heads * g.width;
                    vec![
                        (g.query_heads * cap + cap * g.kv_heads * g.width) * 4,
                        c * 128,
                        c * 128,
                        c * 256,
                        c * 256,
                    ]
                }
                PlanOperation::View(view) => {
                    let ([source], [destination]) =
                        (node.inputs.as_slice(), node.outputs.as_slice())
                    else {
                        return Err(Error::Malformed("resident View arity".into()));
                    };
                    if node.state.is_some()
                        || view.rows.checked_mul(view.width) != Some(n(*destination)?)
                        || source.0 >= destination.0
                    {
                        return Err(Error::Malformed(
                            "resident View must fully write one dense private value".into(),
                        ));
                    }
                    view.validate(input(0)?)?;
                    crate::program_kernels::checked_copy_words(
                        view.rows,
                        view.width,
                        view.row_stride,
                        view.width,
                    )?;
                    vec![]
                }
                _ => vec![],
            };
            for (slot, size) in sizes.into_iter().enumerate() {
                if u32::try_from(size / 4).is_err() {
                    return Err(Error::Limit(
                        "resident workspace exceeds shader index domain".into(),
                    ));
                }
                scratch[slot] = scratch[slot].max(size);
            }
        }
        if state_specs.len() != plan.required_domains().len()
            || plan
                .required_domains()
                .iter()
                .any(|s| !state_specs.contains_key(s))
        {
            return Err(Error::State(
                "resident plan state requirements incomplete".into(),
            ));
        }
        let mut states = Vec::new();
        for (slot, layout) in state_specs {
            let b = match layout {
                StateLayout::Vector { elements } => bytes(elements, 4)?,
                StateLayout::Kv { slab, capacity } => bytes(slab * capacity, 8)?,
            };
            states.push(StateRequirement {
                slot,
                layout,
                bytes: b,
            });
        }
        let state_bytes = checked_sum(states.iter().map(|s| s.bytes))?;
        let value_bytes = checked_sum(
            plan.values()
                .iter()
                .zip(&aliases)
                .filter(|(_, alias)| alias.is_none())
                .map(|(s, _)| s.capacity_elements().and_then(|n| bytes(n, 4)))
                .collect::<Result<Vec<_>>>()?,
        )?;
        Ok(ResidentPreflight {
            source: plan.source(),
            source_bytes,
            value_bytes,
            workspace_bytes: checked_sum(scratch)?,
            status_bytes,
            parameter_bytes: 4,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states,
            scratch,
        })
    }

    /// Check the complete resident shader profile before physical submission.
    pub fn ensure_capability(client: &Client) -> Result<()> {
        source_accumulation::ensure_capability(client, SourceAccumulation::WideF64)?;
        crate::activation_quant_fused::ensure_selected_capability(client)?;
        for t in [
            cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U64),
            cubecl::ir::ElemType::Int(cubecl::ir::IntKind::I64),
        ] {
            if !client.properties().features.supports_type(t) {
                return Err(Error::Unsupported(
                    "resident source arithmetic requires shader Int64/UInt64".into(),
                ));
            }
        }
        let ty = cubecl::ir::Type::atomic(cubecl::ir::ElemType::UInt(cubecl::ir::UIntKind::U32));
        let usage = client.properties().atomic_type_usage(ty);
        if !usage.contains(cubecl::ir::features::AtomicUsage::LoadStore)
            || !usage.contains(cubecl::ir::features::AtomicUsage::Bitwise)
        {
            return Err(Error::Unsupported(
                "resident certificates require U32 atomic load/store and bitwise operations".into(),
            ));
        }
        Ok(())
    }

    /// Upload each immutable tensor/vector once, and allocate the exact logical
    /// preflight. Caller admits/retains these allocations before any launch.
    pub fn prepare_with_allocator(
        client: &Client,
        model: Arc<dyn TensorSource>,
        plan: ForwardPlan,
        mut allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
    ) -> Result<Self> {
        let preflight = Self::preflight(model.as_ref(), &plan)?;
        Self::ensure_capability(client)?;
        let mut allocation = ProgramAllocator {
            client,
            allocate: &mut allocate,
        };
        let dummy = allocation.upload(ResidentBufferRole::Source, f32::as_bytes(&[0.0]))?;
        let mut tensors = BTreeMap::new();
        let projected: BTreeSet<_> = plan
            .nodes()
            .iter()
            .filter_map(|n| match &n.operation {
                PlanOperation::Project { tensor } => Some(tensor.name.clone()),
                _ => None,
            })
            .collect();
        for node in plan.nodes() {
            let (tensor, embedding) = match &node.operation {
                PlanOperation::Embed { tensor, .. } => (tensor, true),
                PlanOperation::Project { tensor } => (tensor, false),
                _ => continue,
            };
            if tensors.contains_key(&tensor.name) {
                continue;
            }
            let (packed, elements, scales, biases, layout) = if embedding
                && !projected.contains(&tensor.name)
            {
                let b = model.tensor(&tensor.name)?.packed_bytes;
                (
                    allocation.upload(ResidentBufferRole::Source, b)?,
                    b.len(),
                    dummy.clone(),
                    dummy.clone(),
                    None,
                )
            } else {
                let (descriptor, b) = StaticSource::from_region(region(model.as_ref(), tensor)?)?;
                let scales = if descriptor.weight_scales().is_empty() {
                    dummy.clone()
                } else {
                    allocation.upload(
                        ResidentBufferRole::Source,
                        f32::as_bytes(descriptor.weight_scales()),
                    )?
                };
                let biases = if descriptor.biases().is_empty() {
                    dummy.clone()
                } else {
                    allocation.upload(
                        ResidentBufferRole::Source,
                        f32::as_bytes(descriptor.biases()),
                    )?
                };
                (
                    allocation.upload(ResidentBufferRole::Source, &b)?,
                    descriptor.gpu_elements,
                    scales,
                    biases,
                    if tensor.kind == QuantKind::F32 {
                        None
                    } else {
                        Some(ActivationLayout::new(
                            tensor.kind,
                            tensor.columns,
                            tensor.rows,
                        )?)
                    },
                )
            };
            tensors.insert(
                tensor.name.clone(),
                ResidentTensor {
                    packed,
                    elements,
                    scales,
                    biases,
                    layout,
                },
            );
        }
        let constants = plan
            .constants()
            .iter()
            .map(|c| allocation.upload(ResidentBufferRole::Source, f32::as_bytes(&c.values)))
            .collect::<Result<Vec<_>>>()?;
        let aliases = contiguous_aliases(&plan)?;
        let mut values: Vec<DeviceBuffer> = Vec::with_capacity(plan.values().len());
        for (shape, alias) in plan.values().iter().zip(&aliases) {
            let n = shape.capacity_elements()?;
            values.push(if let Some(original) = alias {
                // Clone the complete admitted original, preserving its service,
                // allocation and byte range. No second lease or upload exists.
                values[original.0 as usize].clone()
            } else {
                DeviceBuffer {
                    handle: allocation.empty(ResidentBufferRole::Value, bytes(n, 4)?)?,
                    elements: n,
                }
            });
        }
        let scratch: [Handle; SCRATCH_SLOTS] = (0..SCRATCH_SLOTS)
            .map(|i| allocation.empty(ResidentBufferRole::Workspace, preflight.scratch[i]))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("fixed scratch slot count");
        let node_status = plan
            .nodes()
            .iter()
            .map(|n| {
                Ok(match &n.operation {
                    PlanOperation::Embed { tensor, .. } => NodeStatus {
                        block: Some(
                            allocation.empty(ResidentBufferRole::Status, tensor.columns * 4)?,
                        ),
                        row: None,
                    },
                    PlanOperation::Project { tensor } => {
                        if let Some(l) = tensors[&tensor.name].layout {
                            NodeStatus {
                                block: Some(
                                    allocation.empty(ResidentBufferRole::Status, l.blocks * 4)?,
                                ),
                                row: Some(
                                    allocation.empty(ResidentBufferRole::Status, l.rows * 4)?,
                                ),
                            }
                        } else {
                            NodeStatus {
                                block: None,
                                row: None,
                            }
                        }
                    }
                    _ => NodeStatus {
                        block: None,
                        row: None,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut program = Self {
            _source_owner: model,
            plan,
            preflight,
            tensors,
            constants,
            values,
            scratch,
            node_status,
            projections: Vec::new(),
            status: allocation.empty(ResidentBufferRole::Status, 4)?,
            token: allocation.empty(ResidentBufferRole::Parameter, 4)?,
            exp_tables: allocation.upload(
                ResidentBufferRole::Source,
                f64::as_bytes(&crate::source_transcendentals::TABLES),
            )?,
            trig_tables: allocation.upload(
                ResidentBufferRole::Source,
                f64::as_bytes(&crate::source_trig::TRIG_TABLES),
            )?,
            owned_ranges: OwnedRanges::new(client.service_id()),
        };
        program.projections = program
            .plan
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| match &node.operation {
                PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32 => program
                    .projection_binding(client, index, tensor)
                    .map(|binding| Some(Box::new(binding))),
                _ => Ok(None),
            })
            .collect::<Result<Vec<_>>>()?;
        #[cfg(feature = "cuda")]
        program.preflight_silu(client, 1)?;
        for handle in program.owned_handles() {
            program.owned_ranges.insert(BufferRange::new(&handle)?)?;
        }
        Ok(program)
    }

    /// Cold-only construction from the same original per-operation bindings
    /// previously rebuilt at every quantized Project. All handles already exist
    /// and remain owned by the original admitted program/attempt cohort.
    fn projection_binding(
        &self,
        client: &Client,
        index: usize,
        tensor: &SourceTensor,
    ) -> Result<DeviceActivation> {
        let t = &self.tensors[&tensor.name];
        let layout = t
            .layout
            .ok_or_else(|| Error::State("resident quant projection layout absent".into()))?;
        let flags = &self.node_status[index];
        let input = self.plan.nodes()[index].inputs[0];
        DeviceActivation::from_bound(
            client,
            &layout,
            self.values[input.0 as usize].handle.clone(),
            t.scales.clone(),
            t.biases.clone(),
            [
                self.scratch[0].clone(),
                self.scratch[1].clone(),
                self.scratch[2].clone(),
                self.scratch[3].clone(),
                self.scratch[4].clone(),
                flags.block.as_ref().expect("quant block status").clone(),
                flags.row.as_ref().expect("quant row status").clone(),
            ],
        )
    }
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn budget(&self) -> &ResidentPreflight {
        &self.preflight
    }
    pub fn values(&self) -> &[DeviceBuffer] {
        &self.values
    }
    pub fn source_identity(&self) -> SourceIdentity {
        self.plan.source()
    }
    /// No handles are created after this cohort is retained. Includes all
    /// original source, static, intermediate, status and external state handles.
    pub fn owned_handles(&self) -> Vec<Handle> {
        self.handles_for_attempt(&[])
    }

    /// Shared source/scratch remain pinned; skipped output values and per-node
    /// statuses are absent so their uninitialized storage cannot become Ready.
    /// This is cold binding metadata, not a second allocation or program.
    pub fn execution_handles(&self, output: ResidentOutput) -> Vec<Handle> {
        let nodes = output.nodes(&self.plan);
        let mut handles = vec![
            self.status.clone(),
            self.token.clone(),
            self.exp_tables.clone(),
            self.trig_tables.clone(),
        ];
        for t in self.tensors.values() {
            handles.extend([t.packed.clone(), t.scales.clone(), t.biases.clone()]);
        }
        handles.extend(self.constants.iter().cloned());
        handles.extend(self.scratch.iter().cloned());
        // SSA aliases retain the same original handle. The enclosing owner
        // selects each admitted lease once; all alias producers precede it in
        // this same closed prefix, including during state-only prefill.
        handles.extend(
            nodes
                .iter()
                .flat_map(|n| &n.outputs)
                .map(|id| self.values[id.0 as usize].handle.clone()),
        );
        for s in &self.node_status[..nodes.len()] {
            handles.extend(s.block.iter().cloned());
            handles.extend(s.row.iter().cloned());
        }
        handles
    }

    pub fn handles_for_attempt(&self, bindings: &[StateBinding]) -> Vec<Handle> {
        let mut handles = self.execution_handles(ResidentOutput::Logits);
        for b in bindings {
            handles.extend(b.prior.handles());
            handles.extend(b.candidate.handles());
        }
        handles
    }
    pub fn validate_runtime(
        &self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        self.preflight_silu(client, 1)?;
        self.validate_state_runtime(client, p, bindings)
    }

    fn validate_state_runtime(
        &self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
    ) -> Result<()> {
        if self.owned_ranges.service != client.service_id() {
            return Err(Error::State(
                "resident program used with another physical service".into(),
            ));
        }
        if p.valid_tokens
            != p.position
                .checked_add(1)
                .ok_or_else(|| Error::Limit("resident position overflow".into()))?
            || p.valid_tokens > self.plan.prefix_capacity()
            || p.position > 32768
        {
            return Err(Error::Limit(
                "resident runtime prefix exceeds prepared capacity".into(),
            ));
        }
        let mut seen = BTreeMap::new();
        let mut priors = Vec::new();
        let mut candidates = Vec::new();
        for b in bindings {
            if seen.insert(b.slot, b).is_some() {
                return Err(Error::State("duplicate resident state binding".into()));
            }
        }
        if bindings.len() != self.preflight.states.len() {
            return Err(Error::State("resident state binding count mismatch".into()));
        }
        for r in &self.preflight.states {
            let b = seen
                .get(&r.slot)
                .ok_or_else(|| Error::State("missing resident state domain".into()))?;
            for (state, ranges) in [(&b.prior, &mut priors), (&b.candidate, &mut candidates)] {
                match (r.layout, state) {
                    (StateLayout::Vector { elements }, StateBuffers::Vector(v)) => {
                        ranges.push(v.check(client, elements)?);
                    }
                    (StateLayout::Kv { slab, capacity }, StateBuffers::Kv { keys, values }) => {
                        ranges.push(keys.check(client, slab * capacity)?);
                        ranges.push(values.check(client, slab * capacity)?);
                        if b.prior_tokens != p.position {
                            return Err(Error::State(
                                "resident KV prefix differs from bound position".into(),
                            ));
                        }
                    }
                    _ => {
                        return Err(Error::State(
                            "resident state binding has wrong typed layout".into(),
                        ));
                    }
                }
            }
        }
        // Reject overlapping candidate writes against EVERY committed input,
        // sibling candidate and program-owned allocation, even across domains.
        for (i, c) in candidates.iter().enumerate() {
            if self.owned_ranges.overlaps(c)
                || priors
                    .iter()
                    .chain(candidates[..i].iter())
                    .any(|h| c.overlaps(h))
            {
                return Err(Error::State(
                    "resident private proposal aliases an input/workspace/proposal".into(),
                ));
            }
        }
        Ok(())
    }

    /// Bind metadata, launch GPU-only work, and return UNCOMPLETED handles.
    /// There is no CPU fallback and no readback anywhere in this method.
    pub fn launch(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(client, p, bindings, None, output)
    }

    /// Direct device-token binding (for example the first word of the compact
    /// GPU greedy packet). Caller retains this extra original before launch.
    pub fn launch_device_token(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        token: &DeviceBuffer,
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        token.check(client, 1)?;
        self.launch_impl(client, p, bindings, Some(token), output)
    }
    fn launch_impl(
        &mut self,
        client: &Client,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        token: Option<&DeviceBuffer>,
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.validate_runtime(client, p, bindings)?;
        let mut dispatches = 0;
        unsafe {
            crate::program_kernels::clear_step_status::launch_unchecked(
                client,
                grid(1),
                threads(),
                arg(&self.status, 1),
            );
            if let Some(token) = token {
                self.copy::<false>(
                    client,
                    &token.handle,
                    1,
                    &self.token,
                    1,
                    TensorView::vector(0, 1),
                    0,
                );
            } else {
                crate::resident_bindings::write_token::launch_unchecked(
                    client,
                    grid(1),
                    threads(),
                    arg(&self.token, 1),
                    p.token,
                );
            }
        }
        dispatches += 2;
        let nodes = output.nodes(&self.plan);
        for (index, node) in nodes.iter().enumerate() {
            dispatches += self.launch_node(client, index, node, p, bindings)?;
            // A preflight-validated View fully writes its dense destination and
            // certifies every copied word in the same kernel. All other output
            // certificates (including Contiguous aliases) remain separate.
            if matches!(node.operation, PlanOperation::View(_)) {
                continue;
            }
            for (out_index, id) in node.outputs.iter().enumerate() {
                let value = &self.values[id.0 as usize];
                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
                    && out_index == 1
                {
                    unsafe {
                        crate::resident_bindings::certify_causal::launch_unchecked(
                            client,
                            grid(value.elements),
                            threads(),
                            arg(&value.handle, value.elements),
                            arg(&self.status, 1),
                            p.valid_tokens as u32,
                            geometry.query_heads,
                            self.plan.prefix_capacity(),
                        );
                    }
                } else {
                    self.certify(client, &value.handle, value.elements);
                }
                dispatches += 1;
            }
        }
        let logits = self
            .plan
            .logits()
            .filter(|_| output == ResidentOutput::Logits);
        if let Some(logits) = logits {
            let output = &self.values[logits.0 as usize];
            self.certify(client, &output.handle, output.elements);
            dispatches += 1;
        }
        // Certify all private proposals without moving a state scalar to host.
        for b in bindings {
            for value in b.candidate.buffers() {
                self.certify(client, &value.handle, value.elements);
                dispatches += 1;
            }
        }
        let mut proposals = Vec::new();
        for version in self.plan.proposed_domains() {
            let b = bindings
                .iter()
                .find(|b| b.slot == version.slot)
                .expect("validated state binding");
            proposals.push((*version, b.candidate.clone()));
        }
        let mut quant_status = Vec::new();
        let mut embedding_status = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            match &node.operation {
                PlanOperation::Project { tensor } => {
                    if let Some(l) = self.tensors[&tensor.name].layout {
                        quant_status.push((
                            self.node_status[i]
                                .block
                                .as_ref()
                                .expect("allocated block status")
                                .clone(),
                            l.blocks,
                            self.node_status[i]
                                .row
                                .as_ref()
                                .expect("allocated row status")
                                .clone(),
                            l.rows,
                        ));
                    }
                }
                PlanOperation::Embed { tensor, .. } => embedding_status.push((
                    self.node_status[i]
                        .block
                        .as_ref()
                        .expect("allocated embedding status")
                        .clone(),
                    tensor.columns,
                )),
                _ => {}
            }
        }
        Ok(ResidentSubmission {
            logits: logits.map(|v| self.values[v.0 as usize].clone()),
            proposals,
            status: self.status.clone(),
            quant_status,
            embedding_status,
            dispatches,
            fused_dispatches: 0,
        })
    }
    #[allow(clippy::too_many_arguments)] // Explicit typed source/destination ranges.
    fn copy<const CERTIFY_FINITE: bool>(
        &self,
        client: &Client,
        source: &Handle,
        source_len: usize,
        dest: &Handle,
        dest_len: usize,
        view: TensorView,
        offset: usize,
    ) {
        // Geometry and range checks are resolved by typed plan/preflight; the
        // physical helper also detects bad offsets before touching device RAM.
        unsafe {
            crate::program_kernels::copy_strided_words::launch_unchecked(
                client,
                grid(view.rows * view.width),
                threads(),
                arg(source, source_len),
                arg(dest, dest_len),
                arg(&self.status, 1),
                view.offset as u32,
                offset as u32,
                view.row_stride as u32,
                view.width as u32,
                view.rows,
                view.width,
                CERTIFY_FINITE,
            );
        }
    }
    fn certify(&self, client: &Client, source: &Handle, n: usize) {
        unsafe {
            crate::program_kernels::certify_finite_strided::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(source, n),
                arg(&self.status, 1),
                0,
                n as u32,
                1,
                n,
            );
        }
    }
    fn launch_node(
        &self,
        client: &Client,
        index: usize,
        node: &PlanNode,
        p: RuntimeParameters,
        bindings: &[StateBinding],
    ) -> Result<usize> {
        let input = |i: usize| &self.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.values[node.outputs[i].0 as usize];
        let s = |i: usize| &self.scratch[i];
        let state = |version: StateVersion| -> &StateBuffers {
            let b = bindings
                .iter()
                .find(|b| b.slot == version.slot)
                .expect("validated domain");
            if version.ordinal == 0 {
                &b.prior
            } else {
                &b.candidate
            }
        };
        let vector = |version: StateVersion| -> &DeviceBuffer {
            match state(version) {
                StateBuffers::Vector(v) => v,
                _ => unreachable!("typed vector effect"),
            }
        };
        let kv = |version: StateVersion| -> (&DeviceBuffer, &DeviceBuffer) {
            match state(version) {
                StateBuffers::Kv { keys, values } => (keys, values),
                _ => unreachable!("typed KV effect"),
            }
        };
        let cap = self.plan.prefix_capacity();
        // SAFETY: immutable typed plan, preflight and validated state bindings
        // bound every offset/count. Parent retains the complete original cohort.
        unsafe {
            match &node.operation {
                PlanOperation::TypedEmbed { .. }
                | PlanOperation::TypedProject { .. }
                | PlanOperation::TypedOrdinaryRms { .. }
                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                    "resident typed operation has no qualified lowering".into(),
                )),
                PlanOperation::Embed { tensor, .. } => {
                    let t = &self.tensors[&tensor.name];
                    let status = self.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status");
                    crate::embedding::gather_quantized::launch_unchecked(
                        client,
                        grid(tensor.columns),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(&self.token, 1),
                        arg(&output(0).handle, tensor.columns),
                        arg(status, tensor.columns),
                        tensor.rows,
                        tensor.columns,
                        tensor.kind == QuantKind::Q6K,
                    );
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(tensor.columns),
                        threads(),
                        arg(status, tensor.columns),
                        arg(&self.status, 1),
                        tensor.columns,
                    );
                    Ok(2)
                }
                PlanOperation::Project { tensor } => {
                    let t = &self.tensors[&tensor.name];
                    let out = output(0);
                    if tensor.kind == QuantKind::F32 {
                        crate::mixed_ops::f32_ggml_dot_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&input(0).handle, tensor.columns),
                            arg(&out.handle, tensor.rows),
                            tensor.columns,
                            tensor.rows,
                            true,
                        );
                        return Ok(1);
                    }
                    let l = t.layout.expect("quant layout");
                    let dynamic = self.projections[index]
                        .as_ref()
                        .expect("prepared quant projection binding");
                    let preparation_dispatches = dynamic.launch_prepare(client, &l)?;
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(l.blocks),
                        threads(),
                        arg(&dynamic.block_status, l.blocks),
                        arg(&self.status, 1),
                        l.blocks,
                    );
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(l.rows),
                        threads(),
                        arg(&dynamic.row_status, l.rows),
                        arg(&self.status, 1),
                        l.rows,
                    );
                    match tensor.kind {
                        QuantKind::Q4_0 => crate::q4_q8::source_q4_q8_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        QuantKind::Q4_1 => crate::mixed_ops::q4_1_q8_1_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.offsets, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        QuantKind::Q8_0 => crate::mixed_ops::q8_0_q8_0_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        QuantKind::Q6K => crate::mixed_ops::q6_k_q8_k_gemv::launch_unchecked(
                            client,
                            grid(tensor.rows),
                            threads(),
                            arg(&t.packed, t.elements),
                            arg(&dynamic.scales, l.composite_count),
                            arg(&dynamic.codes, l.columns),
                            arg(&out.handle, tensor.rows),
                            l.blocks,
                            tensor.rows,
                            true,
                        ),
                        _ => {
                            return Err(Error::Unsupported(
                                "resident matrix codec unsupported".into(),
                            ));
                        }
                    }
                    Ok(preparation_dispatches + 3)
                }
                PlanOperation::View(view) => {
                    self.copy::<true>(
                        client,
                        &input(0).handle,
                        input(0).elements,
                        &output(0).handle,
                        output(0).elements,
                        *view,
                        0,
                    );
                    Ok(1)
                }
                PlanOperation::Contiguous => {
                    // Cold binding aliases the entire already-packed input.
                    // launch_impl still certifies this semantic output.
                    Ok(0)
                }
                PlanOperation::Rms { gamma, eps, groups } => {
                    let n = input(0).elements;
                    let width = n / groups;
                    crate::rms_norm::rms_squares::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(s(0), n),
                        n,
                    );
                    crate::rms_norm::rms_inverse::launch_unchecked(
                        client,
                        grid(*groups),
                        threads(),
                        arg(s(0), n),
                        arg(s(1), groups * 2),
                        *eps,
                        width,
                        *groups,
                    );
                    crate::rms_norm::rms_round_inverse::launch_unchecked(
                        client,
                        grid(*groups),
                        threads(),
                        arg(s(1), groups * 2),
                        *groups,
                    );
                    crate::rms_norm::rms_apply::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&self.constants[gamma.0 as usize], width),
                        arg(s(1), groups * 2),
                        arg(&output(0).handle, n),
                        width,
                        *groups,
                    );
                    Ok(4)
                }
                PlanOperation::Silu { .. } => self.launch_silu(client, node, 1),
                PlanOperation::Add => {
                    let n = input(0).elements;
                    crate::elementwise::residual_add::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&input(1).handle, n),
                        arg(&output(0).handle, n),
                        n,
                    );
                    Ok(1)
                }
                PlanOperation::SigmoidGate => {
                    let n = input(0).elements;
                    for stage in 0..crate::elementwise::SIGMOID_STAGES {
                        crate::elementwise::sigmoid_gate_step::launch_unchecked(
                            client,
                            grid(n),
                            threads(),
                            arg(&input(1).handle, n),
                            arg(&input(0).handle, n),
                            arg(
                                &self.exp_tables,
                                crate::source_transcendentals::TABLES.len(),
                            ),
                            arg(s(0), n * crate::elementwise::SIGMOID_SCRATCH),
                            arg(s(1), n),
                            arg(&output(0).handle, n),
                            n,
                            stage,
                        );
                    }
                    Ok(crate::elementwise::SIGMOID_STAGES)
                }
                PlanOperation::Conv { weights } => {
                    let n = input(0).elements;
                    let effect = node.state.expect("conv effect");
                    crate::gdn_conv::gdn_conv_products::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(0).handle, n),
                        arg(&self.constants[weights.0 as usize], n * 4),
                        arg(&vector(effect.read).handle, n * 3),
                        arg(s(0), n * 4),
                        arg(&vector(effect.write).handle, n * 3),
                        n,
                    );
                    for level in 0..3 {
                        crate::gdn_conv::gdn_conv_reduce::launch_unchecked(
                            client,
                            grid(n),
                            threads(),
                            arg(s(0), n * 4),
                            arg(&output(0).handle, n),
                            n,
                            level,
                        );
                    }
                    Ok(4)
                }
                PlanOperation::Coefficients { dt, a, scale } => {
                    let h = input(0).elements;
                    for i in 0..2 {
                        crate::resident_bindings::certify_interval::launch_unchecked(
                            client,
                            grid(h),
                            threads(),
                            arg(&input(i).handle, h),
                            arg(&self.status, 1),
                            -128.0,
                            128.0,
                            h,
                        );
                    }
                    crate::resident_bindings::coefficient_input::launch_unchecked(
                        client,
                        grid(h),
                        threads(),
                        arg(&input(0).handle, h),
                        arg(&input(1).handle, h),
                        arg(&self.constants[dt.0 as usize], h),
                        arg(&self.constants[a.0 as usize], h),
                        arg(s(0), h * 4),
                        h,
                    );
                    for stage in 0..8 {
                        crate::gdn_coefficients::coefficient_step::launch_unchecked(
                            client,
                            grid(h),
                            threads(),
                            arg(s(0), h * 4),
                            arg(
                                &self.exp_tables,
                                crate::source_transcendentals::TABLES.len(),
                            ),
                            arg(s(1), h * 6),
                            arg(s(2), h * 3),
                            arg(s(3), h * 4),
                            h,
                            stage,
                        );
                    }
                    crate::resident_bindings::coefficient_output::launch_unchecked(
                        client,
                        grid(h),
                        threads(),
                        arg(s(3), h * 4),
                        arg(&output(0).handle, h * 3),
                        arg(&output(1).handle, h),
                        arg(&output(2).handle, h),
                        *scale,
                        h,
                    );
                    Ok(12)
                }
                PlanOperation::Recurrent { width } => {
                    let n = input(0).elements;
                    let m = n * width;
                    let effect = node.state.expect("recurrent effect");
                    for i in 0..3 {
                        self.copy::<false>(
                            client,
                            &input(i).handle,
                            n,
                            s(0),
                            n * 3,
                            TensorView::vector(0, n),
                            n * i,
                        );
                    }
                    crate::gdn_recurrent::decay::launch_unchecked(
                        client,
                        grid(m),
                        threads(),
                        arg(&vector(effect.read).handle, m),
                        arg(&input(3).handle, n / width * 3),
                        arg(s(1), m),
                        *width,
                        m,
                    );
                    let mut count = 4;
                    count += self.recurrent_dot(client, s(1), s(0), s(3), s(4), s(2), *width, n, n);
                    crate::gdn_recurrent::residual::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(0), n * 3),
                        arg(s(2), n),
                        n,
                    );
                    crate::gdn_recurrent::beta::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(3).handle, n / width * 3),
                        arg(s(2), n),
                        *width,
                        n,
                    );
                    crate::gdn_recurrent::update::launch_unchecked(
                        client,
                        grid(m),
                        threads(),
                        arg(s(1), m),
                        arg(s(0), n * 3),
                        arg(s(2), n),
                        arg(&vector(effect.write).handle, m),
                        *width,
                        n,
                        m,
                    );
                    count += 3;
                    count += self.recurrent_dot(
                        client,
                        &vector(effect.write).handle,
                        s(0),
                        s(3),
                        s(4),
                        &output(0).handle,
                        *width,
                        n,
                        0,
                    );
                    crate::gdn_recurrent::scale::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input(3).handle, n / width * 3),
                        arg(&output(0).handle, n),
                        *width,
                        n,
                    );
                    Ok(count + 1)
                }
                PlanOperation::KvPrefixGuard { .. } => Ok(0), // exact position checked before first launch
                PlanOperation::Rope { geometry, .. } => {
                    let g = geometry.attention;
                    let h = g.query_heads + g.kv_heads;
                    let half = geometry.rotate / 2;
                    let q = input(0).elements;
                    let k = input(1).elements;
                    let v = input(2).elements;
                    let count = q + k + v;
                    let ordered_angles = crate::rope_angles::PreparedAngles::single_for_cuda(
                        client,
                        s(1),
                        p.position as f32,
                        geometry.theta_scale,
                        half,
                    )?;
                    for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
                        self.copy::<false>(
                            client,
                            &input(i).handle,
                            input(i).elements,
                            s(0),
                            count,
                            TensorView::vector(0, input(i).elements),
                            offset,
                        );
                    }
                    let angle_dispatches = if let Some(angles) = ordered_angles {
                        angles.launch(client)?
                    } else {
                        for stage in 0..half {
                            crate::rope_kv::angle_step::launch_unchecked(
                                client,
                                grid(1),
                                threads(),
                                arg(s(1), half),
                                p.position as f32,
                                geometry.theta_scale,
                                stage,
                            );
                        }
                        half
                    };
                    crate::source_trig::sincos_angles::launch_unchecked(
                        client,
                        grid(half),
                        threads(),
                        arg(s(1), half),
                        arg(&self.trig_tables, crate::source_trig::TRIG_TABLES.len()),
                        arg(s(2), half * 2),
                        half,
                    );
                    crate::rope_kv::rope_products::launch_unchecked(
                        client,
                        grid(h * half),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        g.width,
                        half,
                        h,
                        g.query_heads,
                        1,
                    );
                    crate::rope_kv::rope_sums::launch_unchecked(
                        client,
                        grid(count),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        arg(s(4), h * half * 2),
                        arg(s(6), count),
                        g.width,
                        half,
                        h,
                        count,
                        g.query_heads,
                        1,
                    );
                    crate::rope_kv::rope_subtract::launch_unchecked(
                        client,
                        grid(h * half * 2),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        arg(s(4), h * half * 2),
                        arg(s(5), h * half * 2),
                        g.width,
                        half,
                        h,
                        g.query_heads,
                        1,
                    );
                    crate::rope_kv::rope_apply::launch_unchecked(
                        client,
                        grid(h * half * 2),
                        threads(),
                        arg(s(0), count),
                        arg(s(2), half * 2),
                        arg(s(3), h * half * 2),
                        arg(s(4), h * half * 2),
                        arg(s(5), h * half * 2),
                        arg(s(6), count),
                        g.width,
                        half,
                        h,
                        g.query_heads,
                        1,
                    );
                    for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
                        self.copy::<false>(
                            client,
                            s(6),
                            count,
                            &output(i).handle,
                            output(i).elements,
                            TensorView::vector(offset, output(i).elements),
                            0,
                        );
                    }
                    Ok(angle_dispatches + 11)
                }
                PlanOperation::AppendKv => {
                    let effect = node.state.expect("KV append effect");
                    let (previous_k, previous_v) = kv(effect.read);
                    let (candidate_k, candidate_v) = kv(effect.write);
                    let slab = input(0).elements;
                    crate::resident_bindings::append_kv::launch_unchecked(
                        client,
                        grid(slab * cap),
                        threads(),
                        arg(&previous_k.handle, slab * cap),
                        arg(&previous_v.handle, slab * cap),
                        arg(&input(0).handle, slab),
                        arg(&input(1).handle, slab),
                        arg(&candidate_k.handle, slab * cap),
                        arg(&candidate_v.handle, slab * cap),
                        p.position as u32,
                        slab,
                        cap,
                    );
                    Ok(1)
                }
                PlanOperation::AttentionQk {
                    state: version,
                    geometry: g,
                    scale,
                    ..
                } => {
                    let (keys, _) = kv(*version);
                    let query = g.query_heads * g.width;
                    let key = cap * g.kv_heads * g.width;
                    let columns = g.query_heads * cap;
                    self.copy::<false>(
                        client,
                        &input(0).handle,
                        query,
                        s(0),
                        query + key,
                        TensorView::vector(0, query),
                        0,
                    );
                    self.copy::<false>(
                        client,
                        &keys.handle,
                        key,
                        s(0),
                        query + key,
                        TensorView::vector(0, key),
                        query,
                    );
                    for block in 0..g.width / 32 {
                        let (previous, lanes) = if block % 2 == 0 {
                            (s(2), s(1))
                        } else {
                            (s(1), s(2))
                        };
                        for stage in 0..3 {
                            crate::attention_qk::accumulate::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(s(0), query + key),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(s(3), columns * 32),
                                arg(s(4), columns * 32),
                                g.width,
                                cap,
                                g.query_heads,
                                g.kv_heads,
                                block,
                                stage,
                            );
                        }
                    }
                    let last = if (g.width / 32 - 1) % 2 == 0 {
                        s(1)
                    } else {
                        s(2)
                    };
                    let other = if (g.width / 32 - 1) % 2 == 0 {
                        s(2)
                    } else {
                        s(1)
                    };
                    let mut count = 2 + g.width / 32 * 3;
                    count += self.attention_reduce(client, last, other, s(5), columns);
                    crate::attention_qk::scale::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(s(5), columns * 2),
                        *scale,
                        columns,
                    );
                    self.copy::<false>(
                        client,
                        s(5),
                        columns * 2,
                        &output(0).handle,
                        columns,
                        TensorView::vector(0, columns),
                        0,
                    );
                    crate::resident_bindings::masked_scores::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(s(5), columns * 2),
                        arg(&output(1).handle, columns),
                        p.valid_tokens as u32,
                        g.query_heads,
                        cap,
                    );
                    Ok(count + 3)
                }
                PlanOperation::AttentionSoftmax { heads, .. } => {
                    self.softmax(client, input(0), output(0), *heads, cap, p.valid_tokens)
                }
                PlanOperation::AttentionValue {
                    state: version,
                    geometry: g,
                    padding,
                    ..
                } => {
                    let (_, values) = kv(*version);
                    let probabilities = g.query_heads * cap;
                    let value = cap * g.kv_heads * g.width;
                    let columns = g.query_heads * g.width;
                    self.copy::<false>(
                        client,
                        &input(0).handle,
                        probabilities,
                        s(0),
                        probabilities + value,
                        TensorView::vector(0, probabilities),
                        0,
                    );
                    self.copy::<false>(
                        client,
                        &values.handle,
                        value,
                        s(0),
                        probabilities + value,
                        TensorView::vector(0, value),
                        probabilities,
                    );
                    // Exact source padded length is RUNTIME metadata; extra all-zero
                    // FMA blocks can change -0 to +0 and must never execute.
                    let blocks = p.valid_tokens.div_ceil(*padding) * padding / 32;
                    for block in 0..blocks {
                        let (previous, lanes) = if block % 2 == 0 {
                            (s(2), s(1))
                        } else {
                            (s(1), s(2))
                        };
                        for stage in 0..3 {
                            crate::attention_value::accumulate::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(s(0), probabilities + value),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(s(3), columns * 32),
                                arg(s(4), columns * 32),
                                block as u32,
                                g.width,
                                cap,
                                g.query_heads,
                                g.kv_heads,
                                stage,
                            );
                        }
                    }
                    let last = if (blocks - 1) % 2 == 0 { s(1) } else { s(2) };
                    let other = if (blocks - 1) % 2 == 0 { s(2) } else { s(1) };
                    Ok(2 + blocks * 3
                        + self.attention_reduce(client, last, other, &output(0).handle, columns))
                }
                PlanOperation::CheckFinite { .. } => {
                    self.certify(client, &input(0).handle, input(0).elements);
                    Ok(1)
                }
                PlanOperation::Observe { .. } | PlanOperation::ObserveState { .. } => Ok(0),
            }
        }
    }
    #[allow(clippy::too_many_arguments)] // Exact already-qualified recurrent buffer roles.
    fn recurrent_dot(
        &self,
        client: &Client,
        state: &Handle,
        qkv: &Handle,
        lanes: &Handle,
        reduced: &Handle,
        output: &Handle,
        width: usize,
        values: usize,
        offset: usize,
    ) -> usize {
        unsafe {
            for block in 0..width / 32 {
                crate::gdn_recurrent::dot_lanes::launch_unchecked(
                    client,
                    grid(values * 32),
                    threads(),
                    arg(state, values * width),
                    arg(qkv, values * 3),
                    arg(lanes, values * 32),
                    width,
                    values,
                    offset,
                    block,
                );
            }
            for level in 0..5 {
                let input = if level % 2 == 0 { lanes } else { reduced };
                let destination = if level == 4 {
                    output
                } else if level % 2 == 0 {
                    reduced
                } else {
                    lanes
                };
                let in_len = if level % 2 == 0 {
                    values * 32
                } else {
                    values * 16
                };
                let out_len = if level == 4 {
                    values
                } else if level % 2 == 0 {
                    values * 16
                } else {
                    values * 32
                };
                crate::gdn_recurrent::dot_reduce::launch_unchecked(
                    client,
                    grid(values * (16 >> level)),
                    threads(),
                    arg(input, in_len),
                    arg(destination, out_len),
                    values,
                    level,
                );
            }
        }
        width / 32 + 5
    }
    fn attention_reduce(
        &self,
        client: &Client,
        lanes: &Handle,
        reduced: &Handle,
        output: &Handle,
        columns: usize,
    ) -> usize {
        unsafe {
            for level in 0..5 {
                let input = if level % 2 == 0 { lanes } else { reduced };
                let destination = if level == 4 {
                    output
                } else if level % 2 == 0 {
                    reduced
                } else {
                    lanes
                };
                let in_len = if level % 2 == 0 {
                    columns * 32
                } else {
                    columns * 16
                };
                let out_len = if level == 4 {
                    columns
                } else if level % 2 == 0 {
                    columns * 16
                } else {
                    columns * 32
                };
                crate::attention_qk::reduce::launch_unchecked(
                    client,
                    grid(columns * (16 >> level)),
                    threads(),
                    arg(input, in_len),
                    arg(destination, out_len),
                    columns,
                    level,
                );
            }
        }
        5
    }
    fn softmax(
        &self,
        client: &Client,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        h: usize,
        cap: usize,
        valid: usize,
    ) -> Result<usize> {
        self.softmax_masked(client, input, output, h, cap, CausalMask::Uniform(valid))
    }

    fn softmax_masked(
        &self,
        client: &Client,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        h: usize,
        cap: usize,
        mask: CausalMask<'_>,
    ) -> Result<usize> {
        use crate::attention_softmax as sm;
        let n = h * cap;
        let groups = cap.div_ceil(8);
        let s = |i: usize| &self.scratch[i];
        // The original finite valid prefix is followed by -Inf scores. Maxima
        // and source polynomial stages therefore preserve all valid bits; the
        // padded exponentials become canonical +0 before the original tree.
        unsafe {
            sm::maxima::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(&input.handle, n),
                arg(s(4), h),
                h,
                cap,
            );
            for stage in 0..2 {
                sm::prepare::launch_unchecked(
                    client,
                    grid(n),
                    threads(),
                    arg(&input.handle, n),
                    arg(s(4), h),
                    arg(s(0), n * sm::SCRATCH),
                    n,
                    cap,
                    stage,
                );
            }
            for step in 0..8 {
                if step == 1 || step == 3 {
                    sm::prepare::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(&input.handle, n),
                        arg(s(4), h),
                        arg(s(0), n * sm::SCRATCH),
                        n,
                        cap,
                        if step == 1 { 2 } else { 3 },
                    );
                }
                for stage in 0..3 {
                    sm::polynomial::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(0), n * sm::SCRATCH),
                        arg(s(1), n),
                        arg(s(2), n),
                        arg(s(3), n),
                        n,
                        step,
                        stage,
                    );
                }
            }
            sm::exponentials::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(s(0), n * sm::SCRATCH),
                arg(s(7), n),
                n,
            );
            match mask {
                CausalMask::Uniform(valid) => {
                    crate::resident_bindings::zero_future::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(7), n),
                        valid as u32,
                        h,
                        cap,
                    );
                }
                CausalMask::Rows {
                    positions,
                    heads_per_row,
                } => {
                    crate::resident_bindings::zero_future_rows::launch_unchecked(
                        client,
                        grid(n),
                        threads(),
                        arg(s(7), n),
                        arg(positions, h / heads_per_row),
                        heads_per_row,
                        h / heads_per_row,
                        cap,
                    );
                }
            }
            for level in 0..3 {
                let (source, destination, source_len, destination_len) = if level == 0 {
                    (s(7), s(5), n, h * groups * 4)
                } else if level == 1 {
                    (s(5), s(6), h * groups * 4, h * groups * 2)
                } else {
                    (s(6), s(5), h * groups * 2, h * groups * 4)
                };
                sm::reduce::launch_unchecked(
                    client,
                    grid(h * groups * (4 >> level)),
                    threads(),
                    arg(source, source_len),
                    arg(destination, destination_len),
                    h,
                    cap,
                    level,
                );
            }
            // Source totals are positive. Extra all-+0 groups leave their f64
            // sum unchanged, retaining one fixed capacity shader specialization.
            sm::totals::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(s(5), h * groups * 4),
                arg(s(8), h),
                h,
                groups,
            );
            sm::reciprocal::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(s(8), h),
                arg(s(9), h),
                h,
            );
            sm::inverses::launch_unchecked(
                client,
                grid(h),
                threads(),
                arg(s(9), h),
                arg(s(10), h),
                h,
            );
            sm::probabilities::launch_unchecked(
                client,
                grid(n),
                threads(),
                arg(s(7), n),
                arg(s(10), h),
                arg(&output.handle, n),
                n,
                cap,
            );
        }
        Ok(38)
    }
}

fn region<'a>(model: &'a dyn TensorSource, t: &'a SourceTensor) -> Result<MatVecRegion<'a>> {
    Ok(MatVecRegion {
        source_identity: t.source,
        tensor_name: &t.name,
        kind: t.kind,
        columns: t.columns,
        rows: t.rows,
        packed_weights: model.tensor(&t.name)?.packed_bytes,
        activation: &[],
    })
}
/// Allocation is delegated to the already-armed physical initialization
/// owner. It admits and retains Empty before writing Upload bytes into it.
#[derive(Clone, Copy, Debug)]
pub(crate) enum ResidentBufferRole {
    Source,
    Value,
    Workspace,
    Status,
    Parameter,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum AllocationInit<'a> {
    Empty,
    Upload(&'a [u8]),
}
struct ProgramAllocator<'a, F> {
    client: &'a Client,
    allocate: &'a mut F,
}
impl<F: FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>>
    ProgramAllocator<'_, F>
{
    fn check(&self, handle: Handle, bytes: usize) -> Result<Handle> {
        if handle.service != self.client.service_id() || handle.size_in_used() < bytes as u64 {
            return Err(Error::State(
                "resident allocator returned invalid service/size".into(),
            ));
        }
        Ok(handle)
    }
    fn empty(&mut self, role: ResidentBufferRole, bytes: usize) -> Result<Handle> {
        let handle = (self.allocate)(role, bytes, AllocationInit::Empty)?;
        self.check(handle, bytes)
    }
    fn upload(&mut self, role: ResidentBufferRole, bytes: &[u8]) -> Result<Handle> {
        let handle = (self.allocate)(role, bytes.len(), AllocationInit::Upload(bytes))?;
        self.check(handle, bytes.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "qualification")]
    use std::path::Path;

    #[test]
    fn prepared_ranges_match_independent_byte_coverage_with_gaps_and_aliases() {
        let service = ServiceId::of::<()>(cubecl_common::device::DeviceId::new(0, 0));
        let memory = cubecl_runtime::memory_management::ManagedMemoryHandle::new()
            .binding()
            .id();
        let range = |bytes| BufferRange {
            service,
            memory,
            bytes,
        };
        let mut owned = OwnedRanges::new(service);
        for bytes in [2..4, 8..10, 2..4] {
            owned.insert(range(bytes)).unwrap();
        }
        assert_eq!(owned.by_memory[&memory].len(), 2);
        for start in 0..12 {
            for end in start + 1..=12 {
                // Independent oracle enumerates byte membership, including
                // adjacency and the unowned gap in a single allocation.
                let expected = (start..end).any(|i| [2, 3, 8, 9].contains(&i));
                assert_eq!(owned.overlaps(&range(start..end)), expected);
            }
        }
        let gap = range(4..8);
        assert!(!gap.overlaps(&range(2..4)));
        assert!(!gap.overlaps(&range(8..10)));
        assert!(gap.overlaps(&range(7..9)));
    }

    #[test]
    fn prepared_ranges_keep_service_and_allocation_identity_separate() {
        let device = cubecl_common::device::DeviceId::new(0, 0);
        let service = ServiceId::of::<()>(device);
        let foreign = ServiceId::of::<u8>(device);
        let id = || {
            cubecl_runtime::memory_management::ManagedMemoryHandle::new()
                .binding()
                .id()
        };
        let a = BufferRange {
            service,
            memory: id(),
            bytes: 2..8,
        };
        let b = BufferRange {
            memory: id(),
            ..a.clone()
        };
        let c = BufferRange {
            service: foreign,
            ..a.clone()
        };
        let mut owned = OwnedRanges::new(service);
        owned.insert(a.clone()).unwrap();
        assert!(owned.overlaps(&a));
        assert!(!owned.overlaps(&b));
        assert!(!owned.overlaps(&c));
        assert!(!a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert!(owned.insert(c).is_err());
        assert_eq!(owned.by_memory.len(), 1);
    }

    #[test]
    fn budget_arithmetic_refuses_overflow_and_shader_index_excess() {
        assert_eq!(checked_sum([4, 8, 16]).unwrap(), 28);
        assert!(checked_sum([usize::MAX, 1]).is_err());
        assert!(bytes(usize::MAX, 8).is_err());
        assert_eq!(bytes(262_144, 4).unwrap(), 1_048_576);
    }

    #[test]
    #[cfg(feature = "qualification")]
    #[ignore = "requires both existing trained checkpoints; metadata only, no GPU or token math"]
    fn real_source_preflight_counts_tied_embedding_once_and_all_private_domains() {
        let cache = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/models");
        for name in ["qwen35-0.8b", "qwen35-2b"] {
            let dir = cache.join(name);
            let model =
                LoadedQwenModel::open(&dir.join("model.gguf"), &dir.join("config.json")).unwrap();
            for cap in [256, 512] {
                for logits in [false, true] {
                    let plan = model.forward_plan(cap, logits, false).unwrap();
                    let b = ResidentProgram::preflight(model.as_ref(), &plan).unwrap();
                    let aliases = contiguous_aliases(&plan).unwrap();
                    let alias_count = aliases.iter().flatten().count();
                    let fused_views = plan
                        .nodes()
                        .iter()
                        .filter(|node| matches!(node.operation, PlanOperation::View(_)))
                        .count();
                    assert!(fused_views > 0);
                    let prepared_projections = plan
                        .nodes()
                        .iter()
                        .filter(|node| {
                            matches!(&node.operation, PlanOperation::Project { tensor } if tensor.kind != QuantKind::F32)
                        })
                        .count();
                    assert!(prepared_projections > 0);
                    // Ordinary schedule has two packs per full-attention
                    // layer; complete GDN diagnostics add two per GDN layer.
                    assert!([12, 48].contains(&alias_count));
                    let unaliased: usize = plan
                        .values()
                        .iter()
                        .map(|s| s.capacity_elements().unwrap() * 4)
                        .sum();
                    let saved: usize = aliases
                        .iter()
                        .enumerate()
                        .filter(|(_, alias)| alias.is_some())
                        .map(|(i, alias)| {
                            let original = alias.unwrap().0 as usize;
                            assert!(original < i);
                            assert!(aliases[original].is_none());
                            assert_eq!(plan.values()[i], plan.values()[original]);
                            plan.values()[i].capacity_elements().unwrap() * 4
                        })
                        .sum();
                    assert!(saved > 0);
                    assert_eq!(b.value_bytes, unaliased - saved);
                    assert_eq!(b.states.len(), 42);
                    assert_eq!(b.prior_state_bytes, b.candidate_state_bytes);
                    assert!(b.states.iter().all(|s| s.bytes > 0));
                    let mut tensors = BTreeMap::<String, &SourceTensor>::new();
                    let mut projections = BTreeSet::new();
                    for node in plan.nodes() {
                        match &node.operation {
                            PlanOperation::Embed { tensor, .. } => {
                                tensors.insert(tensor.name.clone(), tensor);
                            }
                            PlanOperation::Project { tensor } => {
                                tensors.insert(tensor.name.clone(), tensor);
                                projections.insert(tensor.name.clone());
                            }
                            _ => {}
                        }
                    }
                    let mut source = 4
                        + crate::source_transcendentals::TABLES.len() * 8
                        + crate::source_trig::TRIG_TABLES.len() * 8;
                    for t in tensors.values() {
                        source += StaticSource::gpu_bytes_for(region(model.as_ref(), t).unwrap())
                            .unwrap();
                        if projections.contains(&t.name) && t.kind != QuantKind::F32 {
                            source += t.rows
                                * (t.columns / if t.kind == QuantKind::Q6K { 256 } else { 32 })
                                * if t.kind == QuantKind::Q4_1 { 8 } else { 4 };
                        }
                    }
                    source += plan
                        .constants()
                        .iter()
                        .map(|c| c.values.len() * 4)
                        .sum::<usize>();
                    assert_eq!(b.source_bytes, source, "{name} cap={cap} logits={logits}");
                    assert_eq!(
                        b.total_with_state().unwrap(),
                        b.owned_bytes().unwrap() + b.prior_state_bytes + b.candidate_state_bytes
                    );
                    println!(
                        "RESIDENT_METADATA_BUDGET model={name} capacity={cap} logits={logits} source={} values={} workspace={} status={} prior={} proposed={} total={} aliases={alias_count} saved_value_bytes={saved} fused_view_certificates={fused_views} prepared_projection_bindings={prepared_projections}",
                        b.source_bytes,
                        b.value_bytes,
                        b.workspace_bytes,
                        b.status_bytes,
                        b.prior_state_bytes,
                        b.candidate_state_bytes,
                        b.total_with_state().unwrap()
                    );
                }
            }
            for cap in [1, 255, 257, 768] {
                let plan = model.forward_plan(cap, true, false).unwrap();
                assert!(ResidentProgram::preflight(model.as_ref(), &plan).is_err());
            }
        }
    }
}

#[cfg(test)]
#[path = "resident_program/typed_refusal_tests.rs"]
mod typed_refusal_tests;

mod silu;

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/composition.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1079; SHA256 `f6584a69561808d2b69456e1267730601aa402be3cb8a9fc726fc66b52a406ea`.

````text
{
  "status": "ROOT_UNACCEPTED_INTEGRATION_FOR_REGRESSION_TESTING",
  "patches": [
    {
      "path": ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/NEW_DELTA.patch",
      "sha256": "b8a2f59f32e3d86b6784c7dc1bda0889143d21f922f37df2c7c751b2fdf623f5"
    },
    {
      "path": ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/ORDERED_GEOMETRY_SEAM.patch",
      "sha256": "775bbe0bb8fbd7022fbd3bca64d85dfa7af8e9cc332d37518bc91618a2d74592"
    },
    {
      "path": ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/ROOT_WIRING.patch",
      "sha256": "b4c44ebf38575efefa7896cc6f7dc56d2620df90ef4e04e39426273389d69658"
    },
    {
      "path": ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/followup-visibility-v2/VISIBILITY_ONLY.patch",
      "sha256": "01ecb17b43fe65c68125ec647f8d052b022178cdc146eb418fd7e6adf708372f"
    }
  ],
  "acceptance_requires": "all physical/compiler/model/serving/fullisolated/routes gates; loaded-SASS remains open",
  "production_accepted": false
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/diagnostic-composition-1.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 314; SHA256 `af9df858ef6f86520f52e64e78e65e32df0f9edc214acf285188b8b39ad51311`.

````text
{
  "status": "UNACCEPTED_DIAGNOSTIC_FIX_FOR_QUALIFICATION",
  "patch": ".agents/runs/serving-and-speculation-ordered-silu-compiler-witness-review-20261020/DUMP_COLLISION_ONLY.patch",
  "sha256": "eeacb5ea3ae18d4ec4eab731bcba934375846533940cb5f3df21c2b8e036f3cd",
  "arithmetic_or_NVRTC_or_queue_changed": false
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/dump-transport-isolated-1/manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 659; SHA256 `7f3082624ad0425ee0aa71e23f3d97efa79f769401f2cf55eccc3727c78a9e43`.

````text
{
  "status": "ROOT_PREPARED_EXACT_HELPER_AND_UNIT_EXTRACTION",
  "original_source_sha256": "8bbf35a1d2e526c76efbe4a9628fed446975f5a3b7c608d9869ec658e30d7362",
  "extracted_source_sha256": "a14eff987ead3a820d95e59d35b93fa7a8a03791860745a6ba85740135344c91",
  "source_assertions": [
    "whole PTX_DUMP_SEQUENCE/write_ptx_dump exact source substring",
    "whole actual unit exact source substring",
    "existing approved cubecl_core StableHasher, no stub"
  ],
  "extern": "/home/pvl/spaces/abird/src/zerogravity/target/debug/deps/libcubecl_core-ea81bc68680492b3.rlib",
  "extern_sha256": "8cc93a79643fc05d4dc149e86bc91e3f96fa047ae8b348e7c84ade859d6e4b09"
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/dump-transport-isolated-1/result.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1282; SHA256 `775566a53426a7f8ae6d722368099d8579e286fbf63c315a72cad3497e6f7f6a`.

````text
{
  "status": "PASS_EXACT_EXTRACTED_FILE_TRANSPORT_UNIT",
  "steps": [
    {
      "stage": "compile",
      "argv": [
        "rustc",
        "--edition=2024",
        "--test",
        ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/dump-transport-isolated-1/source.rs",
        "--extern",
        "cubecl_core=/home/pvl/spaces/abird/src/zerogravity/target/debug/deps/libcubecl_core-ea81bc68680492b3.rlib",
        "-L",
        "dependency=target/debug/deps",
        "-o",
        ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/dump-transport-isolated-1/unit"
      ],
      "exit_code": 0,
      "log_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    },
    {
      "stage": "actual-unit",
      "argv": [
        ".agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/dump-transport-isolated-1/unit",
        "--exact",
        "tests::ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct",
        "--nocapture",
        "--test-threads=1"
      ],
      "exit_code": 0,
      "log_sha256": "c096ea6c24ff4bb43781254f51b72404fb5be9023e4cf1d98802666f68df007a"
    }
  ],
  "CUDA_or_numerical_or_SASS_proof": false
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/dump-transport-isolated-1/source.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3223; SHA256 `a14eff987ead3a820d95e59d35b93fa7a8a03791860745a6ba85740135344c91`.

````text
use cubecl_core::hash::StableHasher;
static PTX_DUMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn write_ptx_dump(
    dir: &std::path::Path,
    kernel_id: &str,
    ptx: &[u8],
) -> std::io::Result<std::path::PathBuf> {
    use std::io::Write;
    use std::sync::atomic::Ordering;

    let hash = StableHasher::hash_one(&kernel_id);
    for _ in 0..64 {
        let sequence = PTX_DUMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let stem = format!("ptx-p{}-s{sequence:016x}-k{hash:032x}", std::process::id());
        let path = dir.join(format!("{stem}.ptx"));
        let mut output = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(output) => output,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        };
        // Reserve both files before writing. Never overwrite an existing id or
        // PTX, even on hash collision, sequence wrap, pid reuse or repeated id.
        let mut id = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path.with_extension("kernel-id.txt"))?;
        id.write_all(kernel_id.as_bytes())?;
        output.write_all(ptx)?;
        return Ok(path);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "CUDA PTX diagnostic path collisions exhausted",
    ))
}


#[cfg(test)]
mod tests {
    #[test]
    fn ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct() {
        struct OwnedDirectory(std::path::PathBuf);
        impl Drop for OwnedDirectory {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let sequence = super::PTX_DUMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "cubecl-ptx-dump-test-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path).unwrap();
        let directory = OwnedDirectory(path);
        // Synthetic transport fixtures, never compiler or numerical witnesses.
        let suffix = "same settings suffix ".repeat(20);
        let first = format!("full-first-id {suffix}");
        let second = format!("full-second-id {suffix}");
        let samples = [
            (first.as_str(), b"synthetic ptx one\n".as_slice()),
            (second.as_str(), b"synthetic ptx two\n".as_slice()),
            (first.as_str(), b"synthetic repeated id\n".as_slice()),
        ];
        let mut paths = Vec::new();
        for (id, ptx) in samples {
            let path = super::write_ptx_dump(&directory.0, id, ptx).unwrap();
            assert!(!paths.contains(&path));
            assert!(path.file_name().unwrap().len() < 255);
            assert_eq!(std::fs::read(&path).unwrap(), ptx);
            assert_eq!(
                std::fs::read_to_string(path.with_extension("kernel-id.txt")).unwrap(),
                id
            );
            paths.push(path);
        }
        assert_eq!(std::fs::read_dir(&directory.0).unwrap().count(), 6);
    }

}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/release-1.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1455; SHA256 `9dc5af07494b2b16c1bef5071e10b3b43e1d4221bbb497dc1a2c679b76b70056`.

````text
{
  "status": "FAIL_PRESERVED_NOT_ACCEPTED",
  "source_freeze_sha256": "30fdcfa2b67df26a82a272e14f80b22c315bae7b048784156adb04f52ccba244",
  "coordinator_sha256": "f5d36cc73f7b015c0da9f9ad2f00da2b1dc4886417b6884d46eba71c3c79cb00",
  "steps": [
    {
      "stage": "cuda-strict",
      "argv": [
        "cargo",
        "clippy",
        "--locked",
        "--offline",
        "-p",
        "zerogravity-gpu",
        "--features",
        "cuda,qualification,hardware-abort-probe",
        "--all-targets",
        "--",
        "-D",
        "warnings"
      ],
      "exit_code": 0,
      "log_sha256": "a1eddf74d1719fa893dacc64adb075e6a8b2fa87c20bdb16e625cf1d4f3d81ed",
      "diagnostic_seconds": 7.171438646997558
    },
    {
      "stage": "dump-transport-unit",
      "argv": [
        "cargo",
        "test",
        "--locked",
        "--offline",
        "-p",
        "cubecl-cuda",
        "--no-default-features",
        "--features",
        "cpp,std",
        "--lib",
        "compute::context::tests::ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct",
        "--",
        "--exact",
        "--nocapture",
        "--test-threads=1"
      ],
      "exit_code": 101,
      "log_sha256": "e8ba25ea0890bf6630709e9f811ccb71944e489015181c521f81c609f9040c2a",
      "diagnostic_seconds": 0.0708014169940725
    }
  ],
  "artifacts": {},
  "production_accepted": false,
  "failure": "AssertionError('dump-transport-unit')"
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/release-2.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5606; SHA256 `033c6eb404cd55537cc337716a9047dce4ff269cb7c32b8982296cb755e25508`.

````text
{
  "status": "PASS_RELEASE_BUILD_REQUIRES_PHYSICAL_OPERATOR_MODEL_SERVING_AND_FULL_GATE",
  "source_freeze_sha256": "30fdcfa2b67df26a82a272e14f80b22c315bae7b048784156adb04f52ccba244",
  "coordinator_sha256": "4b8dce079c9a1c7ebbacca57f5607203012d89ba5b35a0c3a2dd7ba8314034ab",
  "steps": [
    {
      "stage": "cuda-strict",
      "argv": [
        "cargo",
        "clippy",
        "--locked",
        "--offline",
        "-p",
        "zerogravity-gpu",
        "--features",
        "cuda,qualification,hardware-abort-probe",
        "--all-targets",
        "--",
        "-D",
        "warnings"
      ],
      "exit_code": 0,
      "log_sha256": "8950be1e8f4499f41b8e38ef15f09a0e2671be53a43aa5dec4d794320908881d",
      "diagnostic_seconds": 0.40920100497896783
    },
    {
      "stage": "lib-build",
      "argv": [
        "cargo",
        "test",
        "--locked",
        "--offline",
        "--release",
        "-p",
        "zerogravity-gpu",
        "--features",
        "cuda,qualification,hardware-abort-probe",
        "--lib",
        "--no-run",
        "--message-format=json-render-diagnostics"
      ],
      "exit_code": 0,
      "log_sha256": "d25bbf3e19cb772b904ad0640cd1de813ecc86570c60bb004621087ba96b807c",
      "diagnostic_seconds": 218.34422282601008
    },
    {
      "stage": "qualifier-build",
      "argv": [
        "cargo",
        "build",
        "--locked",
        "--offline",
        "--release",
        "-p",
        "zerogravity-gpu-qualify",
        "--features",
        "cuda,qualification,hardware-abort-probe",
        "--message-format=json-render-diagnostics"
      ],
      "exit_code": 0,
      "log_sha256": "427a833122d4b60276f738cd76ae1fdda7fd4e70c95bb2674c4fc5e111091b62",
      "diagnostic_seconds": 210.19529990997398
    },
    {
      "stage": "serving-strict",
      "argv": [
        "cargo",
        "clippy",
        "--locked",
        "--offline",
        "-p",
        "zerogravity-cli",
        "--features",
        "cache,serve,cuda",
        "--all-targets",
        "--",
        "-D",
        "warnings"
      ],
      "exit_code": 0,
      "log_sha256": "8ed55edfa4c7403e2aa63586c6bcafd48902a7926de2414cac3012e6262640cc",
      "diagnostic_seconds": 10.120088619994931
    },
    {
      "stage": "serving-build",
      "argv": [
        "cargo",
        "test",
        "--locked",
        "--offline",
        "--release",
        "-p",
        "zerogravity-cli",
        "--features",
        "cache,serve,cuda",
        "--test",
        "serve_gpu",
        "--bin",
        "zerogravity",
        "--no-run",
        "--message-format=json-render-diagnostics"
      ],
      "exit_code": 0,
      "log_sha256": "1a7105625fc97c100276109f5e37f687fa4004b37a5b80266d3bac2710cf129c",
      "diagnostic_seconds": 499.7780929050059
    }
  ],
  "artifacts": {
    "libtest": {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/frozen-libtest-2",
      "bytes": 36319880,
      "sha256": "f603eb47ef26853e85fd77e7f69b42e86b63eebdebb9b84a07d7601383309bcc",
      "features": [
        "cuda",
        "default",
        "hardware-abort-probe",
        "qualification"
      ],
      "source_executable": "/home/pvl/spaces/abird/src/zerogravity/target/release/deps/zerogravity_gpu-f14454d5d4c300ab"
    },
    "qualifier": {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/frozen-qualifier-2",
      "bytes": 37073968,
      "sha256": "fe9604655aeb92e7f7c0cfc668964cba1e2ade2a608f546e1764ef170dded16a",
      "features": [
        "cuda",
        "default",
        "hardware-abort-probe",
        "qualification"
      ],
      "source_executable": "/home/pvl/spaces/abird/src/zerogravity/target/release/zerogravity-gpu-probe"
    },
    "http": {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/frozen-http-2",
      "bytes": 15708136,
      "sha256": "02c4b4eda6b6eedb8670b2ea68335645f15a0ca326e8be1d07f4466380a44f7c",
      "features": [
        "cache",
        "cpu",
        "cuda",
        "default",
        "model-qwen35",
        "serve",
        "text"
      ],
      "source_executable": "/home/pvl/spaces/abird/src/zerogravity/target/release/deps/serve_gpu-83e0383237ce4b95"
    },
    "worker": {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/frozen-worker-2",
      "bytes": 36754544,
      "sha256": "f1f2071db1e1d9ec6fcef99568047880530a40cdce3cd9cbac8873976c86695c",
      "features": [
        "cache",
        "cpu",
        "cuda",
        "default",
        "model-qwen35",
        "serve",
        "text"
      ],
      "source_executable": "/home/pvl/spaces/abird/src/zerogravity/target/release/deps/zerogravity-0876edd786ddb79e"
    },
    "server": {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/frozen-server-2",
      "bytes": 36619448,
      "sha256": "af0f04902559d13cc817138f973ea517ca8279150e12a22941fc955730e04faa",
      "features": [
        "cache",
        "cpu",
        "cuda",
        "default",
        "model-qwen35",
        "serve",
        "text"
      ],
      "source_executable": "/home/pvl/spaces/abird/src/zerogravity/target/release/zerogravity"
    }
  },
  "production_accepted": false
}

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/root-physical-1.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7991; SHA256 `4311bf93a3aa2410657a739630c16481c39860fc5267c4d5aa70a77215af8442`.

````text
"""ROOT serial original-queue diagnostics; no speed or acceptance claim."""
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()

def load(name, path, digest):
    assert sha(path) == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

owned = load('ordered_integration_owned', ROOT/'.agents/runs/serving-and-speculation-vllm-gguf-matched-bench-20261020/owned_container_v4.py', '53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d')
guard = load('ordered_integration_guard', ROOT/'.agents/runs/serving-and-speculation-state-scatter-certification-20261020/root-serving-cancellation-2.py', '00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128')
freeze = HERE/'source-freeze-3.json'
assert sha(freeze) == '30fdcfa2b67df26a82a272e14f80b22c315bae7b048784156adb04f52ccba244'
manifest = json.loads(freeze.read_text())
build_path = HERE/'release-2.json'
build = json.loads(build_path.read_text())
assert build['status'] == 'PASS_RELEASE_BUILD_REQUIRES_PHYSICAL_OPERATOR_MODEL_SERVING_AND_FULL_GATE'
assert build['source_freeze_sha256'] == sha(freeze)
artifacts = build['artifacts']
lib = artifacts['libtest']['path']
qualifier = artifacts['qualifier']['path']

def verify():
    for row in manifest['files']:
        p = ROOT/row['file']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    for row in artifacts.values():
        p = Path(row['path'])
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    assert subprocess.check_output(['git','hash-object','README.md'],cwd=ROOT,text=True).strip() == 'a7be69c35cf61d14e8810f9cb5a6a43d1397bf27'

def quiet():
    for _ in range(30):
        if not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
            return
        time.sleep(.2)
    raise RuntimeError('GPU not quiet: no next diagnostic')

cases = []
test_args = lambda name: [lib,'--ignored','--exact',name,'--nocapture','--test-threads=1']
for fault in ('','unknown','preexisting','nonfinite','range','known-ready-abandon'):
    cases.append(dict(name='operator-'+(fault or 'normal'), argv=test_args('ordered_silu::qualification::physical_ordered_silu'), fault=fault, test=True))
for model in ('0.8b','2b'):
    source = f'target/models/qwen35-{model}/model.gguf'
    cases.append(dict(name=model+'-b1-full',argv=[qualifier,'resident-qualify-cuda',source,'4096','5120']))
    for rows in (2,4,8):
        for mode in ('resident-fused-cuda','resident-fused-greedy-cuda','resident-cuda-graph-wide-cuda'):
            cases.append(dict(name=f'{model}-b{rows}-{mode}',argv=[qualifier,mode,source,'4096','5120',str(rows)]))
        for mode in ('state','logits','greedy'):
            cases.append(dict(name=f'{model}-b{rows}-graph-unknown-{mode}',argv=[qualifier,'resident-cuda-graph-wide-abort-cuda',source,'4096','5120',str(rows),mode]))
    for mode in ('resident-fused-abort-cuda','resident-fused-greedy-abort-cuda'):
        cases.append(dict(name=model+'-'+mode,argv=[qualifier,mode,source,'4096','5120','2']))
for model in ('08b','2b'):
    for rows in (2,4,8):
        name=f'resident_inference::fused_qualification::prefix_qualification::original_qwen35_{model}_native_prefix_b{rows}'
        cases.append(dict(name=f'{model}-prefix-b{rows}',argv=test_args(name),test=True))
    for kind in ('pressure','abort'):
        name=f'resident_inference::fused_qualification::prefix_qualification::original_qwen35_{model}_native_prefix_{kind}'
        cases.append(dict(name=f'{model}-prefix-{kind}',argv=test_args(name),test=True))
    name=f'resident_inference::admission_qualification::original_qwen35_{model}_native_prelaunch_admission_pressure'
    cases.append(dict(name=f'{model}-admission-pressure',argv=test_args(name),test=True))
assert len(cases) == 60
result = HERE/'physical-1.json'
assert not result.exists()
report = dict(status='RUNNING_NOT_ACCEPTED',source_freeze_sha256=sha(freeze),release_build_sha256=sha(build_path),coordinator_sha256=sha(Path(__file__)),case_count=len(cases),cases=[],production_or_performance_qualified=False)
try:
    with guard.SigtermCancellation() as cancellation:
        for case in cases:
            cancellation.check()
            verify()
            quiet()
            receipt = dict(case, status='UNQUALIFIED')
            env = dict(os.environ,LD_PRELOAD='',LD_AUDIT='',ZG_ORDERED_SILU_FAULT=case.get('fault',''))
            assert not env.get('CUBECL_DEBUG_LOG') and not env.get('CUBECL_DEBUG_OPTION') and not env.get('CUBECL_CUDA_DUMP_PTX')
            cwd = ROOT
            if case['name'] == 'operator-normal':
                cwd = HERE/'compilation-only-cwd-1'
                cwd.mkdir(mode=0o700)
                dump = HERE/'actual-ptx-1'
                dump.mkdir(mode=0o700)
                compiler_log = HERE/'actual-generated-cpp-1.log'
                assert not compiler_log.exists()
                config = cwd/'cubecl.toml'
                config.write_text('[compilation.logger]\nlevel = "full"\nfile = "'+str(compiler_log)+'"\nappend = true\n\n[profiling.logger]\nlevel = "disabled"\n')
                env['CUBECL_CUDA_DUMP_PTX'] = str(dump)
                receipt.update(compilation_config_sha256=sha(config),PTX_directory=str(dump),compiler_log=str(compiler_log),profiling_enabled=False)
            child = log = None
            started = time.monotonic()
            try:
                cancellation.check()
                child = subprocess.Popen(case['argv'],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
                receipt['owned_cli_pid'] = child.pid
                log = owned.Log(child.stdout,HERE/(case['name']+'-physical-1.log'))
                receipt['exit_code'] = cancellation.observe(log,child,1800)
            except BaseException as error:
                receipt['failure'] = repr(error)
            finally:
                errors=[]
                if child is not None:
                    try:
                        owned.cleanup_group(child,receipt)
                    except BaseException as error:
                        errors.append(repr(error))
                if log is not None:
                    try:
                        log.finish()
                    except BaseException as error:
                        errors.append(repr(error))
                    receipt['log'] = dict(pipe_eof=log.eof,closed=log.closed,bytes=log.total,sha256=sha(log.path))
                elif child is not None and child.stdout is not None:
                    child.stdout.close()
                receipt['cleanup_errors'] = errors
                receipt['diagnostic_seconds'] = time.monotonic()-started
                report['cases'].append(receipt)
                result.write_text(json.dumps(report,indent=2)+'\n')
            assert receipt.get('exit_code') == 0 and 'failure' not in receipt and not errors and log.eof and log.closed, receipt
            text=log.path.read_text()
            assert 'PASS' in text and (not case.get('test') or '1 passed; 0 failed; 0 ignored' in text),case['name']
            receipt['status']='PASS_PHYSICAL_CASE_REQUIRES_REVIEW'
            quiet()
            verify()
            result.write_text(json.dumps(report,indent=2)+'\n')
            print('PASS',case['name'],flush=True)
        cancellation.check()
    report['status']='PASS_SIX_OPERATOR_AND54_MODEL_CASES_REQUIRES_COMPILER_SERVING_AND_FULL_GATES'
except BaseException as error:
    report['status']='FAIL_PRESERVED_NOT_ACCEPTED'
    report['failure']=repr(error)
    raise
finally:
    result.write_text(json.dumps(report,indent=2)+'\n')

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/root-release-1.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4729; SHA256 `f5d36cc73f7b015c0da9f9ad2f00da2b1dc4886417b6884d46eba71c3c79cb00`.

````text
"""ROOT sole Cargo coordinator; unaccepted artifacts for physical regression."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((HERE/'source-freeze-1.json').read_text())
manifest = dict(status='ROOT_UNACCEPTED_ORDERED_SILU_INTEGRATION', files=[
    dict(file=r['file'], bytes=(ROOT/r['file']).stat().st_size, sha256=sha(ROOT/r['file'])) for r in prior['files']],
    production_accepted=False)
freeze = HERE/'source-freeze-2.json'
with freeze.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')

def verify():
    for row in manifest['files']:
        p = ROOT/row['file']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    assert subprocess.check_output(['git', 'hash-object', 'README.md'], cwd=ROOT, text=True).strip() == 'a7be69c35cf61d14e8810f9cb5a6a43d1397bf27'

steps = [
    ('cuda-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-gpu', '--features', 'cuda,qualification,hardware-abort-probe', '--all-targets', '--', '-D', 'warnings']),
    ('dump-transport-unit', ['test', '--locked', '--offline', '-p', 'cubecl-cuda', '--no-default-features', '--features', 'cpp,std', '--lib', 'compute::context::tests::ptx_dumps_keep_suffix_collisions_and_repeated_ids_distinct', '--', '--exact', '--nocapture', '--test-threads=1']),
    ('lib-build', ['test', '--locked', '--offline', '--release', '-p', 'zerogravity-gpu', '--features', 'cuda,qualification,hardware-abort-probe', '--lib', '--no-run', '--message-format=json-render-diagnostics']),
    ('qualifier-build', ['build', '--locked', '--offline', '--release', '-p', 'zerogravity-gpu-qualify', '--features', 'cuda,qualification,hardware-abort-probe', '--message-format=json-render-diagnostics']),
    ('serving-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-cli', '--features', 'cache,serve,cuda', '--all-targets', '--', '-D', 'warnings']),
    ('serving-build', ['test', '--locked', '--offline', '--release', '-p', 'zerogravity-cli', '--features', 'cache,serve,cuda', '--test', 'serve_gpu', '--bin', 'zerogravity', '--no-run', '--message-format=json-render-diagnostics']),
]
result = HERE/'release-1.json'
assert not result.exists()
report = dict(status='RUNNING_NOT_ACCEPTED', source_freeze_sha256=sha(freeze), coordinator_sha256=sha(Path(__file__)), steps=[], artifacts={}, production_accepted=False)
try:
    for name, args in steps:
        verify()
        log = HERE/(name+'-release-1.log')
        start = time.monotonic()
        with log.open('xb') as stream:
            code = subprocess.run(['cargo']+args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
        report['steps'].append(dict(stage=name, argv=['cargo']+args, exit_code=code, log_sha256=sha(log), diagnostic_seconds=time.monotonic()-start))
        result.write_text(json.dumps(report, indent=2)+'\n')
        print(name, code, flush=True)
        assert code == 0, name
        verify()
        if not name.endswith('build'):
            continue
        for line in log.read_text().splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get('reason') != 'compiler-artifact' or not value.get('executable'):
                continue
            target = value['target']['name']
            role = {'zerogravity_gpu': 'libtest', 'zerogravity-gpu-probe': 'qualifier', 'serve_gpu': 'http'}.get(target)
            if target == 'zerogravity':
                role = 'worker' if value['profile']['test'] else 'server'
            if role:
                assert role not in report['artifacts']
                source = Path(value['executable'])
                destination = HERE/('frozen-'+role+'-1')
                assert not destination.exists()
                shutil.copyfile(source, destination)
                destination.chmod(0o755)
                report['artifacts'][role] = dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha(destination), features=value['features'], source_executable=str(source))
        result.write_text(json.dumps(report, indent=2)+'\n')
    assert set(report['artifacts']) == {'libtest', 'qualifier', 'worker', 'http', 'server'}
    report['status'] = 'PASS_RELEASE_BUILD_REQUIRES_PHYSICAL_OPERATOR_MODEL_SERVING_AND_FULL_GATE'
except BaseException as error:
    report['status'] = 'FAIL_PRESERVED_NOT_ACCEPTED'
    report['failure'] = repr(error)
    raise
finally:
    result.write_text(json.dumps(report, indent=2)+'\n')

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/root-release-2.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4581; SHA256 `4b8dce079c9a1c7ebbacca57f5607203012d89ba5b35a0c3a2dd7ba8314034ab`.

````text
"""ROOT sole Cargo coordinator; unaccepted artifacts for physical regression."""
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prior = json.loads((HERE/'source-freeze-1.json').read_text())
manifest = dict(status='ROOT_UNACCEPTED_ORDERED_SILU_INTEGRATION', files=[
    dict(file=r['file'], bytes=(ROOT/r['file']).stat().st_size, sha256=sha(ROOT/r['file'])) for r in prior['files']],
    production_accepted=False)
freeze = HERE/'source-freeze-3.json'
with freeze.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')

def verify():
    for row in manifest['files']:
        p = ROOT/row['file']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    assert subprocess.check_output(['git', 'hash-object', 'README.md'], cwd=ROOT, text=True).strip() == 'a7be69c35cf61d14e8810f9cb5a6a43d1397bf27'

assert json.loads((HERE/'dump-transport-isolated-1/result.json').read_text())['status'] == 'PASS_EXACT_EXTRACTED_FILE_TRANSPORT_UNIT'

steps = [
    ('cuda-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-gpu', '--features', 'cuda,qualification,hardware-abort-probe', '--all-targets', '--', '-D', 'warnings']),
    ('lib-build', ['test', '--locked', '--offline', '--release', '-p', 'zerogravity-gpu', '--features', 'cuda,qualification,hardware-abort-probe', '--lib', '--no-run', '--message-format=json-render-diagnostics']),
    ('qualifier-build', ['build', '--locked', '--offline', '--release', '-p', 'zerogravity-gpu-qualify', '--features', 'cuda,qualification,hardware-abort-probe', '--message-format=json-render-diagnostics']),
    ('serving-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-cli', '--features', 'cache,serve,cuda', '--all-targets', '--', '-D', 'warnings']),
    ('serving-build', ['test', '--locked', '--offline', '--release', '-p', 'zerogravity-cli', '--features', 'cache,serve,cuda', '--test', 'serve_gpu', '--bin', 'zerogravity', '--no-run', '--message-format=json-render-diagnostics']),
]
result = HERE/'release-2.json'
assert not result.exists()
report = dict(status='RUNNING_NOT_ACCEPTED', source_freeze_sha256=sha(freeze), coordinator_sha256=sha(Path(__file__)), steps=[], artifacts={}, production_accepted=False)
try:
    for name, args in steps:
        verify()
        log = HERE/(name+'-release-2.log')
        start = time.monotonic()
        with log.open('xb') as stream:
            code = subprocess.run(['cargo']+args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
        report['steps'].append(dict(stage=name, argv=['cargo']+args, exit_code=code, log_sha256=sha(log), diagnostic_seconds=time.monotonic()-start))
        result.write_text(json.dumps(report, indent=2)+'\n')
        print(name, code, flush=True)
        assert code == 0, name
        verify()
        if not name.endswith('build'):
            continue
        for line in log.read_text().splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if value.get('reason') != 'compiler-artifact' or not value.get('executable'):
                continue
            target = value['target']['name']
            role = {'zerogravity_gpu': 'libtest', 'zerogravity-gpu-probe': 'qualifier', 'serve_gpu': 'http'}.get(target)
            if target == 'zerogravity':
                role = 'worker' if value['profile']['test'] else 'server'
            if role:
                assert role not in report['artifacts']
                source = Path(value['executable'])
                destination = HERE/('frozen-'+role+'-2')
                assert not destination.exists()
                shutil.copyfile(source, destination)
                destination.chmod(0o755)
                report['artifacts'][role] = dict(path=str(destination), bytes=destination.stat().st_size, sha256=sha(destination), features=value['features'], source_executable=str(source))
        result.write_text(json.dumps(report, indent=2)+'\n')
    assert set(report['artifacts']) == {'libtest', 'qualifier', 'worker', 'http', 'server'}
    report['status'] = 'PASS_RELEASE_BUILD_REQUIRES_PHYSICAL_OPERATOR_MODEL_SERVING_AND_FULL_GATE'
except BaseException as error:
    report['status'] = 'FAIL_PRESERVED_NOT_ACCEPTED'
    report['failure'] = repr(error)
    raise
finally:
    result.write_text(json.dumps(report, indent=2)+'\n')

````

## Source: .agents/runs/serving-and-speculation-ordered-silu-integration-20261020/root-composition-1/root-strict-1.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3000; SHA256 `d92528fc38a05e489c1d031fced3d100634ba2051ba00cc9b4e8b1717908a73b`.

````text
"""ROOT serial source checks of the unaccepted resident candidate."""
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
previous = ROOT / '.agents/runs/serving-and-speculation-ordered-silu-20261020/root-source-freeze-2.json'
names = [r['file'] for r in json.loads(previous.read_text())['files']]
names += ['crates/zerogravity-gpu/src/resident_program/silu.rs',
          'crates/zerogravity-gpu/src/resident_program/silu/tests.rs']
assert len(set(names)) == 461
manifest = dict(status='ROOT_UNACCEPTED_INTEGRATION_SOURCE', files=[
    dict(file=n, bytes=(ROOT/n).stat().st_size, sha256=sha(ROOT/n)) for n in sorted(names)],
    production_accepted=False)
freeze = HERE / 'source-freeze-1.json'
with freeze.open('x') as stream:
    json.dump(manifest, stream, indent=2)
    stream.write('\n')

def verify():
    for row in manifest['files']:
        p = ROOT/row['file']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], p
    assert subprocess.check_output(['git', 'hash-object', 'README.md'], cwd=ROOT, text=True).strip() == 'a7be69c35cf61d14e8810f9cb5a6a43d1397bf27'

steps = [
    ('cuda-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-gpu', '--features', 'cuda,qualification,hardware-abort-probe', '--all-targets', '--', '-D', 'warnings']),
    ('vulkan-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-gpu', '--features', 'vulkan,qualification', '--all-targets', '--', '-D', 'warnings']),
    ('default-strict', ['clippy', '--locked', '--offline', '-p', 'zerogravity-gpu', '--all-targets', '--', '-D', 'warnings']),
    ('cuda-geometry-units', ['test', '--locked', '--offline', '-p', 'zerogravity-gpu', '--features', 'cuda,qualification', 'resident_program::silu::tests', '--', '--nocapture']),
]
result = HERE / 'strict-1.json'
assert not result.exists()
report = dict(status='RUNNING_NOT_ACCEPTED', source_freeze_sha256=sha(freeze), coordinator_sha256=sha(Path(__file__)), steps=[], production_accepted=False)
try:
    for name, args in steps:
        verify()
        log = HERE / (name+'-1.log')
        start = time.monotonic()
        with log.open('xb') as stream:
            code = subprocess.run(['cargo']+args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT).returncode
        report['steps'].append(dict(stage=name, argv=['cargo']+args, exit_code=code, log_sha256=sha(log), diagnostic_seconds=time.monotonic()-start))
        result.write_text(json.dumps(report, indent=2)+'\n')
        print(name, code, flush=True)
        assert code == 0, name
        verify()
    report['status'] = 'PASS_STRICT_AND_GEOMETRY_REQUIRES_PHYSICAL_INTEGRATION'
except BaseException as error:
    report['status'] = 'FAIL_PRESERVED_NOT_ACCEPTED'
    report['failure'] = repr(error)
    raise
finally:
    result.write_text(json.dumps(report, indent=2)+'\n')

````

