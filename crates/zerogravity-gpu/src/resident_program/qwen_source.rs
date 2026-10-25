//! Original source records lowered by the existing resident node loop.
use super::*;
use crate::resident_leases::{
    ResidentInitialization, ResidentLease, ResidentService,
    frame_checkpoint::ResidentFrameCheckpoint, source_page,
};
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zerogravity_artifacts::source_weights::{ParameterWindow, VerifiedMatrix, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    Bf16TextCall, EmbeddingContract, ProjectionContract, SourceParameter, TorchCpuGdnPhase,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_model_qwen35::original_binding::RecordedWindowToken;
mod layout;
mod math;
mod transport;

pub use layout::QwenSourceResourceInventory;
/// Distinct source-native preparation, never numerical/serving eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QwenSourceReadout {
    TeacherLogits,
    NaturalGreedy,
}
impl QwenSourceReadout {
    pub(crate) fn natural(self) -> bool {
        self == Self::NaturalGreedy
    }
}
pub(crate) fn vocabulary(record: &RecordedWindowToken) -> Result<usize> {
    let plan = record.plan();
    plan.values()[plan
        .logits()
        .ok_or_else(|| Error::State("source full vocabulary absent".into()))?
        .0 as usize]
        .capacity_elements()
}

/// Observation coordinates only; no operand, source or publication authority.
#[derive(Clone, Copy, Debug)]
pub enum QwenSourceFrameKind {
    Value {
        layer: usize,
        stage: &'static str,
    },
    State {
        slot: StateSlot,
        plane: usize,
        prior: bool,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct QwenSourceFrame {
    pub position: usize,
    pub kind: QwenSourceFrameKind,
    pub offset_words: usize,
    pub total_words: usize,
}
/// Completed bounded actual frames only; expected data cannot supply an operand.
pub trait QwenSourceObserver {
    fn completed(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> Result<()>;
}
use super::bf16_projection::{
    Bf16ProjectionBindings, Bf16ProjectionStorage, PreparedBf16Projection,
};
pub(super) trait Execution {
    fn call(&self) -> &Bf16TextCall;
    fn page(
        &self,
        page: &Arc<source_page::ResidentSourcePage>,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        enqueue: &mut dyn FnMut(&source_page::SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()>;
    fn projection(
        &self,
        operand: &PreparedBf16Projection,
        page: &Arc<source_page::ResidentSourcePage>,
        bindings: Bf16ProjectionBindings<'_>,
    ) -> Result<usize>;
    fn observe(
        &self,
        body: &QwenSourceBody,
        layer: usize,
        stage: &'static str,
        lease: &ResidentLease,
        words: usize,
    ) -> Result<()>;
}
#[derive(Clone, Copy)]
struct Epoch {
    value: Option<ValueId>,
    epoch: usize,
}
pub(crate) struct QwenSourceBody {
    // Raw aliases precede the original source-owning managed leases.
    values: Vec<DeviceBuffer>,
    service: Arc<ResidentService>,
    recorded: Arc<RecordedWindowToken>,
    layout: layout::Layout,
    slots: Vec<ResidentLease>,
    parameters: BTreeMap<String, ResidentLease>,
    projections: BTreeMap<String, Arc<PreparedBf16Projection>>,
    scratch: Vec<ResidentLease>,
    tables: Vec<ResidentLease>,
    page: Arc<source_page::ResidentSourcePage>,
    frame: Arc<ResidentFrameCheckpoint>,
    packed: ResidentLease,
    accumulation: ResidentLease,
    projection_flags: ResidentLease,
    status: ResidentLease,
    zero: ResidentLease,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    epochs: Mutex<Vec<Epoch>>,
    next: AtomicUsize,
    _metadata: WindowWorkspace,
    _native: Arc<crate::backend::hip_native_policy::PreparedSourceNative>,
}
struct ParameterUpload {
    _window: ParameterWindow,
    _bytes: Vec<u8>,
    _staging: WindowWorkspace,
}
pub(crate) struct PendingQwenSource {
    body: Arc<QwenSourceBody>,
}
pub(super) fn trimmed(lease: &ResidentLease, offset: usize, bytes: usize) -> Result<Handle> {
    if bytes == 0 || offset.checked_add(bytes).is_none_or(|n| n > lease.bytes()) {
        return Err(Error::State("source exact original logical range".into()));
    }
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - offset - bytes) as u64))
}
impl QwenSourceBody {
    #[cfg(feature = "gemma-bf16-qualification")]
    pub(crate) fn native_identity(
        &self,
    ) -> crate::backend::hip_native_policy::HipSourceSetIdentity {
        self._native.identity(&self.service)
    }
    #[cfg(feature = "gemma-bf16-qualification")]
    pub(crate) fn require_native_view(&self, client: &Client) -> Result<()> {
        self._native
            .require_live(client, self.recorded.plan().source())
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub(crate) fn final_owner_native_probe(
        &self,
    ) -> crate::backend::hip_native_policy::FinalNativeProbe {
        self._native.final_owner_probe()
    }
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    pub(crate) fn arm_source_page_event_probe(
        &self,
        mode: cubecl::hip::SourcePageEventProbeMode,
    ) -> Result<source_page::SourcePageProbeParts> {
        self.page.arm_source_page_event_probe(mode)
    }
    #[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
    pub(crate) fn source_page_event_probe(
        &self,
    ) -> Result<(cubecl::hip::SourcePageEventProbeSnapshot, bool, bool)> {
        self.page.source_page_event_probe()
    }
    pub(crate) fn prepare(
        recorded: Arc<RecordedWindowToken>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        native: Arc<crate::backend::hip_native_policy::PreparedSourceNative>,
    ) -> Result<PendingQwenSource> {
        init.require_service(&service)?;
        native.require_live(service.client(), recorded.plan().source())?;
        let layout = layout::Layout::derive(&recorded)?;
        if service.source_identity() != recorded.plan().source()
            || service.client().is_capturing()
            || layout.owned_bytes > service.remaining_capacity()?
        {
            return Err(Error::Limit(
                "source complete managed/native preflight".into(),
            ));
        }
        let source = recorded.source().source().clone();
        let metadata = source.reserve_workspace(layout::METADATA_BYTES)?;
        let mut projections = BTreeMap::new();
        for node in recorded.plan().nodes() {
            if let PlanOperation::TypedProject {
                tensor,
                contract: ProjectionContract::TorchCpuBf16Avx2,
            } = &node.operation
            {
                if projections.contains_key(&tensor.name) {
                    continue;
                }
                let matrix = source.matrix(&tensor.name, tensor.axes)?;
                let operand = PreparedBf16Projection::new(
                    source.clone(),
                    matrix,
                    Bf16ProjectionStorage::Streamed,
                )?;
                operand.require_capability(&service)?;
                if operand.storage() != Bf16ProjectionStorage::Streamed
                    || operand.source().source_identity() != recorded.plan().source()
                {
                    return Err(Error::State("Qwen explicit streamed source operand".into()));
                }
                let pending = futures_lite::future::block_on(operand.upload(init, None))?;
                operand.bind_initialized(pending)?;
                if operand.resident().is_some() || operand.max_launch_rows() > 2048 {
                    return Err(Error::Unsupported(
                        "Qwen finite streamed projection storage".into(),
                    ));
                }
                projections.insert(tensor.name.clone(), operand);
            }
        }
        let page = match native.source_page_actor(&service) {
            Some(actor) => source_page::ResidentSourcePage::allocate_observed(
                source.clone(),
                service.clone(),
                init,
                Some(actor),
            )?,
            None => {
                source_page::ResidentSourcePage::allocate(source.clone(), service.clone(), init)?
            }
        };
        let frame = ResidentFrameCheckpoint::new(source.clone())?;
        let slots = layout
            .slots
            .iter()
            .map(|n| init.allocate_empty(*n, BufferRole::Activation))
            .collect::<Result<Vec<_>>>()?;
        let mut parameters = BTreeMap::new();
        for (name, p) in &layout.parameters {
            let window = source.read_declared_parameter(p, None)?;
            let bytes = layout::aligned(window.parameter().elements() * 4)?;
            let staging = source.reserve_workspace(3 * bytes)?;
            let mut widened = vec![0u8; bytes];
            match p.kind {
                QuantKind::F32 => widened[..p.bytes].copy_from_slice(window.as_slice()),
                QuantKind::Bf16 => {
                    for (out, input) in widened
                        .chunks_exact_mut(4)
                        .zip(window.as_slice().chunks_exact(2))
                    {
                        out.copy_from_slice(
                            &(u32::from(u16::from_le_bytes(input.try_into().unwrap())) << 16)
                                .to_le_bytes(),
                        );
                    }
                }
                _ => return Err(Error::State("source parameter physical dtype".into())),
            }
            if widened[..window.parameter().elements() * 4]
                .chunks_exact(4)
                .any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite())
            {
                return Err(Error::Malformed("source raw parameter nonfinite".into()));
            }
            let owner = Arc::new(ParameterUpload {
                _window: window,
                _bytes: widened,
                _staging: staging,
            });
            // The original initialization retains the authentic window BEFORE copy.
            init.retain_completion_owner(owner.clone())?;
            parameters.insert(
                name.clone(),
                init.upload_bytes(&owner._bytes, BufferRole::ImmutableWeights)?,
            );
        }
        let scratch = layout
            .math
            .iter()
            .map(|n| init.allocate_empty(*n, BufferRole::Scratch))
            .collect::<Result<Vec<_>>>()?;
        let tables = math::immutable_tables()
            .into_iter()
            .map(|mut bytes| {
                bytes.resize(layout::aligned(bytes.len())?, 0);
                init.upload_bytes(&bytes, BufferRole::ImmutableWeights)
            })
            .collect::<Result<Vec<_>>>()?;
        let packed = init.allocate_empty(
            layout::aligned(layout.max_columns * 2)?,
            BufferRole::Scratch,
        )?;
        let accumulation = init.allocate_empty(layout::aligned(2048 * 4)?, BufferRole::Scratch)?;
        let projection_flags =
            init.allocate_empty(layout::aligned(2048 * 4)?, BufferRole::Scratch)?;
        let status = init.allocate_empty(layout::ALIGNMENT, BufferRole::Scratch)?;
        let zero = init.upload_bytes(&[0; layout::ALIGNMENT], BufferRole::ImmutableWeights)?;
        let mut accesses = Vec::new();
        accesses.extend(
            slots
                .iter()
                .chain(&scratch)
                .cloned()
                .map(|l| (l, DeviceAccess::Write)),
        );
        accesses.extend(
            parameters
                .values()
                .chain(&tables)
                .chain([&zero])
                .cloned()
                .map(|l| (l, DeviceAccess::Read)),
        );
        accesses.extend(
            [
                &packed,
                &accumulation,
                &projection_flags,
                &status,
                page.lease(),
            ]
            .into_iter()
            .cloned()
            .map(|l| (l, DeviceAccess::Write)),
        );
        if layout::sum(accesses.iter().map(|(l, _)| l.bytes()))? != layout.owned_bytes {
            return Err(Error::State(
                "source actual managed owner inventory differs".into(),
            ));
        }
        let values = recorded
            .plan()
            .values()
            .iter()
            .zip(&layout.bindings)
            .map(|(shape, b)| {
                Ok(DeviceBuffer::new(
                    trimmed(&slots[b.slot], 0, b.bytes)?,
                    shape.capacity_elements()?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let body = Arc::new(Self {
            service,
            recorded,
            layout,
            slots,
            parameters,
            projections,
            scratch,
            tables,
            page,
            frame,
            packed,
            accumulation,
            projection_flags,
            status,
            zero,
            values,
            accesses,
            epochs: Mutex::new(Vec::new()),
            next: AtomicUsize::new(0),
            _metadata: metadata,
            _native: native,
        });
        init.retain_completion_owner(body.clone())?;
        Ok(PendingQwenSource { body })
    }
    fn lease(&self, id: ValueId) -> &ResidentLease {
        &self.slots[self.layout.bindings[id.0 as usize].slot]
    }
    pub(super) fn begin(&self, call: &Bf16TextCall, p: RuntimeParameters) -> Result<()> {
        let row = call.row(0)?;
        if call.rows() != 1
            || row.position() != self.recorded.position()
            || row.position() != p.position
            || row.token() != p.token
            || row.valid_tokens() != p.valid_tokens
        {
            return Err(Error::State(
                "source exact original record/call phase".into(),
            ));
        }
        self._native
            .require_live(self.service.client(), self.recorded.plan().source())?;
        self.next.store(0, Ordering::Release);
        *self
            .epochs
            .lock()
            .map_err(|_| Error::State("source epochs poisoned".into()))? = vec![
            Epoch {
                value: None,
                epoch: 0
            };
            self.slots.len()
        ];
        Ok(())
    }
    pub(super) fn validate_runtime(&self, client: &Client, p: RuntimeParameters) -> Result<()> {
        if client.service_id() != self.service.client().service_id()
            || p.position != self.recorded.position()
            || p.valid_tokens != p.position + 1
        {
            return Err(Error::State(
                "source original service/record position".into(),
            ));
        }
        self._native
            .require_live(client, self.recorded.plan().source())?;
        self.service.ensure_usable()
    }
    pub(super) fn finish_node(&self, index: usize, node: &PlanNode) -> Result<()> {
        let mut epochs = self
            .epochs
            .lock()
            .map_err(|_| Error::State("source epochs poisoned".into()))?;
        for id in &node.outputs {
            let b = self.layout.bindings[id.0 as usize];
            epochs[b.slot] = Epoch {
                value: Some(*id),
                epoch: b.epoch,
            };
        }
        self.next
            .compare_exchange(index, index + 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("source repeated/concurrent node".into()))?;
        Ok(())
    }
    pub(super) fn require_finished(&self) -> Result<()> {
        if self.next.load(Ordering::Acquire) != self.recorded.plan().nodes().len() {
            return Err(Error::State(
                "source complete full64 traversal missing".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn enqueue(
        &self,
        index: usize,
        node: &PlanNode,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        execution: &dyn Execution,
    ) -> Result<usize> {
        self.service.ensure_usable()?;
        if self.next.load(Ordering::Acquire) != index
            || !same_node(&self.recorded.plan().nodes()[index], node)
        {
            return Err(Error::State("source immutable SSA operation/order".into()));
        }
        {
            let epochs = self
                .epochs
                .lock()
                .map_err(|_| Error::State("source epochs poisoned".into()))?;
            for id in &node.inputs {
                let b = self.layout.bindings[id.0 as usize];
                if epochs[b.slot].value != Some(*id) || epochs[b.slot].epoch != b.epoch {
                    return Err(Error::State("source stale producer storage epoch".into()));
                }
            }
        }
        let input = node
            .inputs
            .iter()
            .map(|id| &self.values[id.0 as usize])
            .collect::<Vec<_>>();
        let output = node
            .outputs
            .iter()
            .map(|id| &self.values[id.0 as usize])
            .collect::<Vec<_>>();
        let client = self.service.client();
        let status = trimmed(&self.status, 0, 4)?;
        let zero = trimmed(&self.zero, 0, 4)?;
        match &node.operation {
            PlanOperation::TypedEmbed {
                tensor,
                contract: EmbeddingContract::Bf16OutputInput,
                ..
            } => {
                let matrix = self
                    .recorded
                    .source()
                    .source()
                    .matrix(&tensor.name, tensor.axes)?;
                let token = execution.call().row(0)?.token() as usize;
                let out = self.lease(node.outputs[0]).clone();
                execution.page(&self.page, &matrix, token..token + 1, &mut |page| {
                    if !page.matrix().same_binding(&matrix)
                        || page.rows() != (token..token + 1)
                        || page.bytes() != tensor.columns * 2
                    {
                        return Err(Error::State("source selected authentic row".into()));
                    }
                    unsafe {
                        transport::widen::launch_unchecked(
                            client,
                            grid(tensor.columns),
                            threads(),
                            arg(page.handle(), tensor.columns / 2),
                            arg(&output[0].handle, tensor.columns),
                            arg(&status, 1),
                            crate::program_kernels::NONFINITE_VALUE,
                        );
                    }
                    Ok(vec![out.clone()])
                })?;
                Ok(1)
            }
            PlanOperation::TypedProject {
                tensor,
                contract: ProjectionContract::TorchCpuBf16Avx2,
            } => self.project(tensor, node, input[0], output[0], execution),
            PlanOperation::View(view) => {
                self.copy(input[0], output[0], *view);
                Ok(1)
            }
            PlanOperation::Contiguous => {
                self.copy(
                    input[0],
                    output[0],
                    TensorView::vector(0, input[0].elements),
                );
                Ok(1)
            }
            PlanOperation::TypedSourceStateRead { binding, .. } => {
                let actual = state_buffers(bindings, binding.slot(), false)?;
                if actual.len() != output.len() {
                    return Err(Error::State("source read component count".into()));
                }
                for ((from, to), n) in actual.iter().zip(&output).zip(binding.components()) {
                    self.copy(from, to, TensorView::vector(0, *n));
                }
                Ok(output.len())
            }
            PlanOperation::TypedSourceStateWrite { binding } => {
                let actual = state_buffers(bindings, binding.slot(), true)?;
                if actual.len() != input.len() {
                    return Err(Error::State("source write component count".into()));
                }
                for ((from, to), active) in input.iter().zip(&actual).zip(binding.components()) {
                    unsafe {
                        transport::state_copy::launch_unchecked(
                            client,
                            grid(to.elements),
                            threads(),
                            arg(&from.handle, *active),
                            arg(&to.handle, to.elements),
                            arg(&status, 1),
                            *active,
                            to.elements,
                        );
                    }
                }
                Ok(input.len())
            }
            PlanOperation::Observe { layer, stage } => {
                execution.observe(
                    self,
                    *layer,
                    stage,
                    self.lease(node.inputs[0]),
                    input[0].elements,
                )?;
                Ok(0)
            }
            PlanOperation::CheckFinite { .. } => Ok(0),
            operation => {
                let handles = self
                    .scratch
                    .iter()
                    .map(ResidentLease::handle)
                    .collect::<Vec<_>>();
                let scratch = math::Scratch::new(&handles)?;
                let parameter = |p: &SourceParameter| -> Result<Handle> {
                    let expected = self
                        .layout
                        .parameters
                        .get(&p.name)
                        .ok_or_else(|| Error::State("source parameter not bound".into()))?;
                    if expected != p {
                        return Err(Error::State("source raw parameter identity".into()));
                    }
                    trimmed(
                        &self.parameters[&p.name],
                        0,
                        p.shape.iter().product::<u64>() as usize * 4,
                    )
                };
                match operation {
                    PlanOperation::TypedOrdinaryRms {
                        parameter: p,
                        contract,
                    } => {
                        let width = contract.validate(p, input[0].elements)?;
                        math::ordinary_rms(
                            client,
                            &scratch,
                            &input[0].handle,
                            &parameter(p)?,
                            &output[0].handle,
                            &status,
                            &zero,
                            width,
                            input[0].elements / width,
                        )
                    }
                    PlanOperation::TypedTorchGdn { contract, phase } => {
                        if (*phase == TorchCpuGdnPhase::InitialOne) != (p.position == 0) {
                            return Err(Error::State(
                                "source GDN actual initial/cached join".into(),
                            ));
                        }
                        math::gdn(
                            client,
                            &scratch,
                            contract,
                            input
                                .iter()
                                .map(|v| &v.handle)
                                .collect::<Vec<_>>()
                                .try_into()
                                .map_err(|_| Error::State("source six GDN operands".into()))?,
                            [
                                &parameter(&contract.convolution)?,
                                &parameter(&contract.a_log)?,
                                &parameter(&contract.dt_bias)?,
                                &parameter(&contract.norm)?,
                            ],
                            output
                                .iter()
                                .map(|v| &v.handle)
                                .collect::<Vec<_>>()
                                .try_into()
                                .map_err(|_| Error::State("source six GDN producers".into()))?,
                            &self.tables[0].handle(),
                            &self.tables[1].handle(),
                            &status,
                            &zero,
                            *phase,
                        )
                    }
                    PlanOperation::TypedTorchFullConvolution => math::full_convolution(
                        client,
                        &input[0].handle,
                        &input[1].handle,
                        &input[2].handle,
                        &output[0].handle,
                        &status,
                        input[2].elements,
                    ),
                    PlanOperation::TypedTorchAttention { geometry, position } => {
                        if *position != p.position {
                            return Err(Error::State("source attention record position".into()));
                        }
                        math::attention(
                            client,
                            &scratch,
                            &input.iter().map(|v| &v.handle).collect::<Vec<_>>(),
                            output
                                .iter()
                                .map(|v| &v.handle)
                                .collect::<Vec<_>>()
                                .try_into()
                                .map_err(|_| {
                                    Error::State("source nine attention producers".into())
                                })?,
                            &self.tables[3].handle(),
                            &self.tables[2].handle(),
                            &self.tables[1].handle(),
                            &status,
                            &zero,
                            *geometry,
                            *position,
                        )
                    }
                    PlanOperation::Silu { multiplied: true } => math::silu_mul(
                        client,
                        &input[0].handle,
                        &input[1].handle,
                        &output[0].handle,
                        &status,
                        input[0].elements,
                    ),
                    PlanOperation::Add => {
                        unsafe {
                            transport::add::launch_unchecked(
                                client,
                                grid(input[0].elements),
                                threads(),
                                arg(&input[0].handle, input[0].elements),
                                arg(&input[1].handle, input[0].elements),
                                arg(&output[0].handle, input[0].elements),
                                arg(&status, 1),
                                input[0].elements,
                            );
                        }
                        Ok(1)
                    }
                    _ => Err(Error::Unsupported(
                        "operation outside complete full64 source record".into(),
                    )),
                }
            }
        }
    }
    fn copy(&self, input: &DeviceBuffer, output: &DeviceBuffer, view: TensorView) {
        unsafe {
            crate::program_kernels::copy_strided_words::launch_unchecked(
                self.service.client(),
                grid(view.rows * view.width),
                threads(),
                arg(&input.handle, input.elements),
                arg(&output.handle, output.elements),
                arg(&self.status.handle(), 1),
                view.offset as u32,
                0,
                view.row_stride as u32,
                view.width as u32,
                view.rows,
                view.width,
                true,
            );
        }
    }
    fn project(
        &self,
        tensor: &SourceTensor,
        node: &PlanNode,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        execution: &dyn Execution,
    ) -> Result<usize> {
        execution.call().row(0)?.projection_geometry(tensor)?;
        let operand = self
            .projections
            .get(&tensor.name)
            .ok_or_else(|| Error::State("source prepared projection absent".into()))?;
        let matrix = self
            .recorded
            .source()
            .source()
            .matrix(&tensor.name, tensor.axes)?;
        if !operand.matrix().same_binding(&matrix) {
            return Err(Error::State(
                "source prepared projection binding changed".into(),
            ));
        }
        execution.projection(
            operand,
            &self.page,
            Bf16ProjectionBindings {
                input: &input.handle,
                input_lease: self.lease(node.inputs[0]),
                output: &output.handle,
                output_lease: self.lease(node.outputs[0]),
                packed: &self.packed,
                accumulation: &self.accumulation,
                flags: &self.projection_flags,
                status: &self.status,
                zero: &self.zero,
            },
        )
    }
}
fn state_buffers(
    bindings: &[StateBinding],
    slot: StateSlot,
    candidate: bool,
) -> Result<Vec<&DeviceBuffer>> {
    let binding = bindings
        .iter()
        .find(|b| b.slot == slot)
        .ok_or_else(|| Error::State("source original state binding absent".into()))?;
    Ok(if candidate {
        &binding.candidate
    } else {
        &binding.prior
    }
    .buffers())
}
fn same_node(a: &PlanNode, b: &PlanNode) -> bool {
    if a.name != b.name || a.inputs != b.inputs || a.outputs != b.outputs || a.state != b.state {
        return false;
    }
    match (&a.operation, &b.operation) {
        (
            PlanOperation::TypedEmbed {
                tensor: a,
                token: ta,
                contract: ca,
            },
            PlanOperation::TypedEmbed {
                tensor: b,
                token: tb,
                contract: cb,
            },
        ) => a == b && ta == tb && ca == cb,
        (
            PlanOperation::TypedProject {
                tensor: a,
                contract: ca,
            },
            PlanOperation::TypedProject {
                tensor: b,
                contract: cb,
            },
        ) => a == b && ca == cb,
        (
            PlanOperation::TypedOrdinaryRms {
                parameter: a,
                contract: ca,
            },
            PlanOperation::TypedOrdinaryRms {
                parameter: b,
                contract: cb,
            },
        ) => a == b && ca == cb,
        (
            PlanOperation::TypedTorchGdn {
                contract: a,
                phase: pa,
            },
            PlanOperation::TypedTorchGdn {
                contract: b,
                phase: pb,
            },
        ) => a == b && pa == pb,
        (
            PlanOperation::TypedTorchAttention {
                geometry: a,
                position: pa,
            },
            PlanOperation::TypedTorchAttention {
                geometry: b,
                position: pb,
            },
        ) => a == b && pa == pb,
        (
            PlanOperation::TypedSourceStateRead {
                binding: a,
                state: sa,
            },
            PlanOperation::TypedSourceStateRead {
                binding: b,
                state: sb,
            },
        ) => a == b && sa == sb,
        (
            PlanOperation::TypedSourceStateWrite { binding: a },
            PlanOperation::TypedSourceStateWrite { binding: b },
        ) => a == b,
        (PlanOperation::View(a), PlanOperation::View(b)) => {
            a.offset == b.offset
                && a.rows == b.rows
                && a.width == b.width
                && a.row_stride == b.row_stride
        }
        (
            PlanOperation::Observe {
                layer: a,
                stage: sa,
            },
            PlanOperation::Observe {
                layer: b,
                stage: sb,
            },
        ) => a == b && sa == sb,
        (PlanOperation::CheckFinite { layer: a }, PlanOperation::CheckFinite { layer: b }) => {
            a == b
        }
        (PlanOperation::Silu { multiplied: a }, PlanOperation::Silu { multiplied: b }) => a == b,
        (PlanOperation::TypedTorchFullConvolution, PlanOperation::TypedTorchFullConvolution)
        | (PlanOperation::Contiguous, PlanOperation::Contiguous)
        | (PlanOperation::Add, PlanOperation::Add) => true,
        _ => false,
    }
}
impl QwenSourceBody {
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn recorded(&self) -> &Arc<RecordedWindowToken> {
        &self.recorded
    }
    pub(crate) fn bind(self: &Arc<Self>) -> Result<ResidentProgram> {
        if self
            .parameters
            .values()
            .chain(&self.tables)
            .chain([&self.zero])
            .any(|l| !l.is_initialized())
        {
            return Err(Error::State(
                "source immutable initialization incomplete".into(),
            ));
        }
        let mut owned_ranges = OwnedRanges::new(self.service.client().service_id());
        let mut ranges = Vec::<BufferRange>::new();
        for (l, _) in &self.accesses {
            let range = BufferRange::new(&l.handle())?;
            range.require_exact_len(l.bytes())?;
            if ranges.iter().any(|r| r.overlaps(&range)) {
                return Err(Error::State("source physical owners overlap".into()));
            }
            owned_ranges.insert(range.clone())?;
            ranges.push(range);
        }
        let source_bytes = layout::sum(
            self.parameters
                .values()
                .chain(&self.tables)
                .chain([&self.zero])
                .map(ResidentLease::bytes),
        )?;
        let status_bytes = self.status.bytes();
        let preflight = ResidentPreflight {
            source: self.recorded.plan().source(),
            source_bytes,
            value_bytes: self.layout.value_bytes,
            workspace_bytes: self.layout.owned_bytes
                - source_bytes
                - self.layout.value_bytes
                - status_bytes,
            status_bytes,
            parameter_bytes: 0,
            prior_state_bytes: self.layout.root_bytes,
            candidate_state_bytes: self.layout.root_bytes,
            states: self.layout.states.clone(),
            scratch: [0; SCRATCH_SLOTS],
        };
        Ok(ResidentProgram {
            _source_owner: ResidentProgramSource::TypedWindows(
                self.recorded.source().source().clone(),
            ),
            plan: self.recorded.plan().clone(),
            preflight,
            values: self.values.clone(),
            status: self.status.handle(),
            owned_ranges,
            body: ResidentProgramBody::QwenSource(self.clone()),
        })
    }
}
impl PendingQwenSource {
    pub(crate) fn bind(self) -> Result<(ResidentProgram, Arc<QwenSourceBody>)> {
        Ok((self.body.bind()?, self.body))
    }
}
struct ArmedExecution<'a, 'o, 's> {
    flight: std::cell::RefCell<&'a mut crate::resident_leases::ResidentSubmission<'s>>,
    observer: &'a std::cell::RefCell<Option<&'o mut dyn QwenSourceObserver>>,
    call: &'a Bf16TextCall,
    cancelled: Option<&'a std::sync::atomic::AtomicBool>,
}
impl Execution for ArmedExecution<'_, '_, '_> {
    fn call(&self) -> &Bf16TextCall {
        self.call
    }
    fn page(
        &self,
        page: &Arc<source_page::ResidentSourcePage>,
        matrix: &VerifiedMatrix,
        rows: Range<usize>,
        enqueue: &mut dyn FnMut(&source_page::SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()> {
        futures_lite::future::block_on(page.with_rows(
            &mut self.flight.borrow_mut(),
            matrix,
            rows,
            self.cancelled,
            enqueue,
        ))
    }
    fn projection(
        &self,
        operand: &PreparedBf16Projection,
        page: &Arc<source_page::ResidentSourcePage>,
        bindings: Bf16ProjectionBindings<'_>,
    ) -> Result<usize> {
        futures_lite::future::block_on(operand.enqueue(
            &mut self.flight.borrow_mut(),
            page,
            bindings,
            self.cancelled,
        ))
    }
    fn observe(
        &self,
        body: &QwenSourceBody,
        layer: usize,
        stage: &'static str,
        lease: &ResidentLease,
        words: usize,
    ) -> Result<()> {
        let mut observer = self.observer.borrow_mut();
        let Some(observer) = observer.as_deref_mut() else {
            return Ok(());
        };
        body.page.drain()?;
        let position = self.call.row(0)?.position();
        for offset in (0..words).step_by(source_page::PAGE_BYTES / 4) {
            let n = (words - offset).min(source_page::PAGE_BYTES / 4);
            let read = lease.readback(offset * 4, n * 4)?;
            futures_lite::future::block_on(body.frame.compare(
                &mut self.flight.borrow_mut(),
                &read,
                |actual| {
                    observer.completed(
                        QwenSourceFrame {
                            position,
                            kind: QwenSourceFrameKind::Value { layer, stage },
                            offset_words: offset,
                            total_words: words,
                        },
                        actual,
                    )
                },
            ))?;
        }
        Ok(())
    }
}
impl ResidentProgram {
    pub(crate) fn qwen_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        match &self.body {
            ResidentProgramBody::QwenSource(b) => Ok(b.accesses.clone()),
            _ => Err(Error::State(
                "distinct original Qwen source body required".into(),
            )),
        }
    }
    pub(crate) fn is_qwen_source(&self) -> bool {
        matches!(self.body, ResidentProgramBody::QwenSource(_))
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_qwen_source(
        &mut self,
        flight: &mut crate::resident_leases::ResidentSubmission<'_>,
        call: &Bf16TextCall,
        current: zerogravity_core::session_root::RootBinding,
        bindings: &[StateBinding],
        observer: &std::cell::RefCell<Option<&mut dyn QwenSourceObserver>>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ResidentSubmission> {
        let ResidentProgramBody::QwenSource(body) = &self.body else {
            return Err(Error::State("Qwen source body absent".into()));
        };
        let body = body.clone();
        call.require_current(current)?;
        if current.source() != body.recorded.plan().source() {
            return Err(Error::State("source actual call identity".into()));
        }
        if cancelled.is_some_and(|f| f.load(Ordering::Acquire)) {
            return Err(Error::State("source call cancelled".into()));
        }
        flight.require_accesses(body.accesses.iter().map(|(l, a)| (l, *a)))?;
        flight.retain_completion_owner(body.clone())?;
        let actual = call.row(0)?;
        let p = RuntimeParameters {
            token: actual.token(),
            position: actual.position(),
            valid_tokens: actual.valid_tokens(),
        };
        body.begin(call, p)?;
        let execution = ArmedExecution {
            flight: std::cell::RefCell::new(flight),
            observer,
            call,
            cancelled,
        };
        let submission = self.launch_impl(
            body.service.client(),
            CurrentExecution::Single {
                parameters: p,
                states: bindings,
            },
            None,
            ResidentOutput::Logits,
            NativeExecution::new(cancelled).with_qwen(&execution),
        )?;
        // All ordinary windows are queued; Ready may only be published after
        // the original actor proved each retired page/window completed.
        body.page.finish()?;
        Ok(submission)
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn checkpoint_qwen_state(
        &self,
        flight: &mut crate::resident_leases::ResidentSubmission<'_>,
        lease: &ResidentLease,
        position: usize,
        slot: StateSlot,
        plane: usize,
        prior: bool,
        active: usize,
        observer: &mut dyn QwenSourceObserver,
    ) -> Result<()> {
        let ResidentProgramBody::QwenSource(body) = &self.body else {
            return Err(Error::State("Qwen source checkpoint body absent".into()));
        };
        body.page.drain()?;
        let words = if matches!(slot, StateSlot::Kv(_)) {
            active * 1024
        } else {
            lease.bytes() / 4
        };
        for offset in (0..lease.bytes() / 4).step_by(source_page::PAGE_BYTES / 4) {
            let n = (lease.bytes() / 4 - offset).min(source_page::PAGE_BYTES / 4);
            let read = lease.readback(offset * 4, n * 4)?;
            body.frame
                .compare(flight, &read, |bytes| {
                    let included = words.saturating_sub(offset).min(n);
                    if bytes[included * 4..].iter().any(|b| *b != 0) {
                        return Err(Error::State("source canonical unused state tail".into()));
                    }
                    if included != 0 {
                        observer.completed(
                            QwenSourceFrame {
                                position,
                                kind: QwenSourceFrameKind::State { slot, plane, prior },
                                offset_words: offset,
                                total_words: words,
                            },
                            &bytes[..included * 4],
                        )?;
                    }
                    Ok(())
                })
                .await?;
        }
        Ok(())
    }
}

pub(crate) fn catalogue(
    client: &Client,
    records: &[Arc<RecordedWindowToken>; 2],
    readout: QwenSourceReadout,
) -> Result<(
    Vec<Box<dyn cubecl_server::kernel::CubeKernel>>,
    WindowWorkspace,
)> {
    let catalogue = math::prepared::enumerate(client, records, readout)?;
    Ok((catalogue.kernels, catalogue.metadata))
}
pub(crate) fn inventory(
    records: &[Arc<RecordedWindowToken>; 2],
    native: &crate::backend::hip_native_policy::PreparedSourceNative,
    readout: QwenSourceReadout,
) -> Result<QwenSourceResourceInventory> {
    inventory_charges(records, native.host_bytes(), native.device_bytes(), readout)
}
pub(crate) fn preparation_inventory(
    records: &[Arc<RecordedWindowToken>; 2],
    readout: QwenSourceReadout,
) -> Result<QwenSourceResourceInventory> {
    let (host, device) = crate::backend::hip_native_policy::qwen_preparation_upper_bytes();
    inventory_charges(records, host, device, readout)
}
#[cfg(feature = "gemma-bf16-qualification")]
pub(crate) fn scoped_preparation_inventory(
    records: &[Arc<RecordedWindowToken>; 2],
    readout: QwenSourceReadout,
) -> Result<QwenSourceResourceInventory> {
    let (host, device) = crate::backend::hip_native_policy::qwen_preparation_upper_bytes();
    let host = host
        .checked_add(crate::backend::hip_native_policy::scoped_metadata_upper_bytes()?)
        .ok_or_else(|| Error::Limit("Qwen scoped host sum".into()))?;
    inventory_charges(records, host, device, readout)
}

fn inventory_charges(
    records: &[Arc<RecordedWindowToken>; 2],
    native_host: usize,
    native_device: usize,
    readout: QwenSourceReadout,
) -> Result<QwenSourceResourceInventory> {
    let a = layout::Layout::derive(&records[0])?;
    let b = layout::Layout::derive(&records[1])?;
    if a.states
        .iter()
        .zip(&b.states)
        .any(|(a, b)| a.slot != b.slot || a.bytes != b.bytes)
        || a.root_bytes != b.root_bytes
    {
        return Err(Error::State("source paired root inventory changed".into()));
    }
    let six = a
        .root_bytes
        .checked_mul(6)
        .ok_or_else(|| Error::Limit("source six-root sum".into()))?;
    let (greedy_workspace_bytes, feedback_peak_bytes) = if readout.natural() {
        let sampler = crate::resident_greedy::workspace_bytes(vocabulary(&records[0])?)?;
        let workspace = layout::sum(
            sampler
                .map(layout::aligned)
                .into_iter()
                .collect::<Result<Vec<_>>>()?,
        )?;
        // Two sessions +two snapshots +one sole Pending; no token lease escapes.
        (workspace, 5 * layout::aligned(12)?)
    } else {
        (0, 0)
    };
    let owned = layout::sum([
        a.owned_bytes,
        b.owned_bytes,
        greedy_workspace_bytes,
        feedback_peak_bytes,
    ])?;
    let root_initialization_host = a
        .root_bytes
        .checked_mul(3)
        .ok_or_else(|| Error::Limit("source complete root host staging overflow".into()))?;
    let planes = a
        .states
        .iter()
        .map(|r| {
            if matches!(r.layout, StateLayout::Kv { .. }) {
                2
            } else {
                1
            }
        })
        .sum();
    let root_pool_metadata =
        crate::resident_leases::backing_pool::ResidentPool::metadata_bytes(planes, 6)?;
    let source = records[0].source().source();
    // Actual immutable source cap/scan geometry, two record/body owners, two
    // session initialization scopes +one Pending/call, bounded page/frame/control
    // and worst-case raw parameter conversion. Weights remain streamed FDs.
    let parameter_upload = a
        .parameters
        .values()
        .chain(b.parameters.values())
        .map(|p| p.shape.iter().product::<u64>() as usize * 4)
        .max()
        .unwrap_or(0);
    let host_upper = layout::sum([
        source.metadata_reserved_bytes() as usize,
        source.staging_chunk_bytes() * 3,
        4 * 1024 * 1024, // authentic facade control +manifest, source operators below
        4 * 1024 * 1024,
        2 * layout::RECORD_BYTES,
        2 * layout::METADATA_BYTES,
        1024 * 1024, // exact generated catalogue owner
        2 * root_initialization_host,
        root_pool_metadata,
        32 * 1024 * 1024,
        16 * 1024 * 1024, // session, Pending, current call
        2 * (64 * 1024 + 4096),
        source_page::transport_host_bytes(source)?,
        3 * parameter_upload,
        native_host,
        2 * 1024 * 1024 * 1024, // separate persistent context host
    ])?;
    if host_upper as u64 > source.workspace_limit_bytes()? {
        return Err(Error::Limit(
            "source complete original host domain preflight".into(),
        ));
    }
    Ok(QwenSourceResourceInventory {
        value_slot_bytes: layout::sum([a.value_bytes, b.value_bytes])?,
        value_slots: a.slots.len() + b.slots.len(),
        parameter_bytes: layout::sum([a.parameter_bytes, b.parameter_bytes])?,
        math_scratch_bytes: layout::sum(a.math.iter().chain(&b.math).copied())?,
        source_transport_bytes: source_page::PAGE_BYTES
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("Qwen initial and cached source pages".into()))?,
        owned_device_bytes: owned,
        greedy_workspace_bytes,
        feedback_peak_bytes,
        root_payload_bytes: a.root_bytes,
        six_root_payload_bytes: six,
        root_initialization_host_bytes: root_initialization_host,
        root_pool_metadata_bytes: root_pool_metadata,
        native_host_bytes: native_host,
        host_upper_bytes: host_upper,
        native_device_bytes: native_device,
        admitted_host_domain_bytes: usize::try_from(
            records[0].source().source().workspace_limit_bytes()?,
        )
        .map_err(|_| Error::Limit("source host domain usize".into()))?,
        complete_device_bytes: layout::sum([owned, six, native_device])?,
        largest_binding_bytes: a
            .slots
            .iter()
            .chain(&b.slots)
            .chain(a.math.iter())
            .chain(b.math.iter())
            .copied()
            .chain([source_page::PAGE_BYTES])
            .max()
            .unwrap_or(0),
    })
}
