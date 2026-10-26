//! Source-bound BF16 text lowering of the original ForwardPlan; no family traversal.
// Previous names: resident_program::gemma_text, GemmaText*, PendingGemmaText.
use super::*;
use crate::resident_leases::resident_matrix::view::ResidentMatrixView;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::resident_leases::{frame_checkpoint::ResidentFrameCheckpoint, source_page};
use crate::resident_program::text_source::TextSourcePlan;
use std::cell::RefCell;
use std::sync::{
    Mutex,
    atomic::{AtomicUsize, Ordering},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{Bf16TextCall, Bf16TextRow, ValueShape};
use zerogravity_core::accelerator::program::{SourceNorm, TextObservation};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

mod layout;
pub use layout::{TextHostOwners, TextResourceInventory};
mod math;
mod native_owners;
pub use native_owners::TextNativeOwners;
mod storage;
mod transport;
use crate::resident_leases::resident_matrix::{PendingResidentMatrix, ResidentMatrix};
use crate::resident_program::bf16_projection::{Bf16ProjectionBindings, PreparedBf16Projection};
pub use storage::TextStorage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextTraffic {
    pub value_upload_bytes: u64,
    pub resident_weight_bytes: usize,
    pub selected_row_uploads: u64,
    pub selected_row_reuses: u64,
    pub selected_row_avoided_upload_bytes: u64,
    pub producer_readback_bytes: usize,
    pub kv_diagnostic_readback_bytes: usize,
}
pub trait TextObserver {
    fn completed_producer(
        &mut self,
        position: usize,
        observation: &TextObservation,
        actual: &[u8],
    ) -> Result<()>;
    fn completed_kv(
        &mut self,
        prior: bool,
        domain: zerogravity_core::accelerator::program::KvDomain,
        value: bool,
        active: usize,
        actual: &[u8],
    ) -> Result<()>;
}
pub(super) trait Execution {
    fn row(&self) -> usize;
    fn call(&self) -> &Bf16TextCall;
    fn page(
        &self,
        page: &Arc<source_page::ResidentSourcePage>,
        matrix: &zerogravity_artifacts::source_weights::VerifiedMatrix,
        rows: Range<usize>,
        reuse: bool,
        enqueue: &mut dyn FnMut(&source_page::SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()>;
    fn projection(
        &self,
        operand: &PreparedBf16Projection,
        page: &Arc<source_page::ResidentSourcePage>,
        bindings: Bf16ProjectionBindings<'_>,
    ) -> Result<usize>;
    fn observing(&self) -> bool;
    fn resident(
        &self,
        resident: &Arc<ResidentMatrix>,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
        matrix: &zerogravity_artifacts::source_weights::VerifiedMatrix,
        rows: Range<usize>,
        enqueue: &mut dyn FnMut(&ResidentMatrixView) -> Result<usize>,
    ) -> Result<usize>;
    fn observe(
        &self,
        body: &Bf16TextBody,
        observation: &TextObservation,
        read: &crate::resident_leases::ResidentReadback,
    ) -> Result<()>;
}
struct ArmedExecution<'a, 's, 'o> {
    flight: RefCell<&'a mut crate::resident_leases::ResidentSubmission<'s>>,
    observer: &'a RefCell<Option<&'o mut dyn TextObserver>>,
    row: usize,
    call: &'a Bf16TextCall,
    cancelled: Option<&'a std::sync::atomic::AtomicBool>,
}
impl Execution for ArmedExecution<'_, '_, '_> {
    fn row(&self) -> usize {
        self.row
    }
    fn call(&self) -> &Bf16TextCall {
        self.call
    }
    fn page(
        &self,
        page: &Arc<source_page::ResidentSourcePage>,
        matrix: &zerogravity_artifacts::source_weights::VerifiedMatrix,
        rows: Range<usize>,
        reuse: bool,
        enqueue: &mut dyn FnMut(&source_page::SourcePageView<'_>) -> Result<Vec<ResidentLease>>,
    ) -> Result<()> {
        if reuse {
            futures_lite::future::block_on(page.with_cached_row(
                &mut self.flight.borrow_mut(),
                matrix,
                rows,
                self.cancelled,
                enqueue,
            ))
        } else {
            futures_lite::future::block_on(page.with_rows(
                &mut self.flight.borrow_mut(),
                matrix,
                rows,
                self.cancelled,
                enqueue,
            ))
        }
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
    fn observing(&self) -> bool {
        self.observer.borrow().is_some()
    }
    fn resident(
        &self,
        resident: &Arc<ResidentMatrix>,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
        matrix: &zerogravity_artifacts::source_weights::VerifiedMatrix,
        rows: Range<usize>,
        enqueue: &mut dyn FnMut(&ResidentMatrixView) -> Result<usize>,
    ) -> Result<usize> {
        cancelled(self.cancelled)?;
        resident.with_rows(&mut self.flight.borrow_mut(), source, matrix, rows, enqueue)
    }
    fn observe(
        &self,
        body: &Bf16TextBody,
        observation: &TextObservation,
        read: &crate::resident_leases::ResidentReadback,
    ) -> Result<()> {
        let position = self.call.row(self.row)?.position();
        futures_lite::future::block_on(body.frame.compare(
            &mut self.flight.borrow_mut(),
            read,
            |actual| {
                body.producer_readback_bytes
                    .fetch_add(actual.len(), Ordering::AcqRel);
                self.observer
                    .borrow_mut()
                    .as_deref_mut()
                    .ok_or_else(|| {
                        Error::State("BF16 text diagnostic checkpoint without observer".into())
                    })?
                    .completed_producer(position, observation, actual)
            },
        ))
    }
}
#[derive(Clone, Copy)]
struct SlotEpoch {
    value: Option<ValueId>,
    epoch: usize,
}
struct ActualCall {
    base: zerogravity_core::session_root::RootBinding,
    frontier: u64,
    ids: Arc<[u32]>,
}
pub(crate) struct Bf16TextBody {
    values: Vec<DeviceBuffer>,
    service: Arc<ResidentService>,
    recorded: Arc<TextSourcePlan>,
    layout: layout::Layout,
    slots: Vec<ResidentLease>,
    gammas: Vec<ResidentLease>,
    math: Vec<ResidentLease>,
    tables: Vec<ResidentLease>,
    page: Arc<source_page::ResidentSourcePage>,
    selection: storage::Selection,
    frame: Arc<ResidentFrameCheckpoint>,
    compact_causal: ResidentLease,
    packed_input: ResidentLease,
    projection_accumulation: ResidentLease,
    projection_status: ResidentLease,
    status: ResidentLease,
    zero: ResidentLease,
    observations: BTreeMap<usize, TextObservation>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    epochs: Mutex<Vec<SlotEpoch>>,
    next: AtomicUsize,
    row: AtomicUsize,
    current_call: Mutex<Option<ActualCall>>,
    producer_readback_bytes: AtomicUsize,
    kv_diagnostic_readback_bytes: AtomicUsize,
    _metadata: WindowWorkspace,
    _native: Arc<TextNativeOwners>,
}
pub(crate) struct PendingBf16Text {
    body: Arc<Bf16TextBody>,
    matrices: Vec<Option<PendingResidentMatrix>>,
}
fn cancelled(flag: Option<&std::sync::atomic::AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("BF16 text BF16 source call cancelled".into()));
    }
    Ok(())
}
fn trimmed(lease: &ResidentLease, offset: usize, n: usize) -> Result<Handle> {
    if n == 0 || offset.checked_add(n).is_none_or(|n| n > lease.bytes()) {
        return Err(Error::State(
            "BF16 text original logical allocation range".into(),
        ));
    }
    Ok(lease
        .handle()
        .offset_start(offset as u64)
        .offset_end((lease.bytes() - offset - n) as u64))
}
struct ColdNativeCost {
    provider: usize,
    host: usize,
    native_host: usize,
    native_device: usize,
    context_host: usize,
    context_device: usize,
}
#[cfg(all(
    feature = "qualification",
    target_os = "linux",
    any(feature = "hip", feature = "vulkan")
))]
fn cold_native_cost(storage: TextStorage) -> Result<ColdNativeCost> {
    #[cfg(feature = "hip")]
    if !storage.is_vulkan() {
        let cost = TextNativeOwners::profile(storage)?.cost();
        return Ok(ColdNativeCost {
            provider: cost.provider,
            host: cost.host,
            native_host: cost.native_host,
            native_device: cost.native_device,
            context_host: cost.context_host,
            context_device: cost.context_device,
        });
    }
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        target_os = "linux"
    ))]
    {
        let cost = TextNativeOwners::vulkan_profile(storage).cost();
        return Ok(ColdNativeCost {
            provider: cost.provider,
            host: cost.host,
            native_host: cost.native_host,
            native_device: cost.native_device,
            context_host: cost.context_host,
            context_device: cost.context_device,
        });
    }
    #[allow(unreachable_code)]
    Err(Error::Unsupported(
        "BF16 text cold policy has no compiled native backend".into(),
    ))
}
impl Bf16TextBody {
    pub(crate) fn require_capability(client: &Client) -> Result<()> {
        use cubecl::ir::{ElemType, FloatKind, IntKind, UIntKind};
        let properties = client.properties();
        for kind in [
            ElemType::Float(FloatKind::F32),
            ElemType::Float(FloatKind::F64),
            ElemType::UInt(UIntKind::U32),
            ElemType::UInt(UIntKind::U64),
            ElemType::Int(IntKind::I32),
            ElemType::Int(IntKind::I64),
        ] {
            if !properties.features.supports_type(kind) {
                return Err(Error::Unsupported(
                    "BF16 text source arithmetic requires actual F32/F64/U32/U64/I32/I64".into(),
                ));
            }
        }
        let usage =
            properties.atomic_type_usage(cubecl::ir::Type::atomic(ElemType::UInt(UIntKind::U32)));
        if !usage.contains(cubecl::ir::features::AtomicUsage::LoadStore)
            || !usage.contains(cubecl::ir::features::AtomicUsage::Bitwise)
            || properties.hardware.max_bindings < 7
            || properties.hardware.max_units_per_cube < 64
            || properties.hardware.max_cube_dim.0 < 64
            || properties.hardware.max_shared_memory_size < 512
            || properties.hardware.max_cube_count.0 < 8192
        {
            return Err(Error::Unsupported(
                "BF16 text original U32 certificate/workgroup/binding capability".into(),
            ));
        }
        Ok(())
    }
    #[cfg(all(
        feature = "qualification",
        target_os = "linux",
        any(feature = "hip", feature = "vulkan")
    ))]
    pub(crate) fn cold_resource_inventory(
        recorded: &TextSourcePlan,
        host: &TextHostOwners,
        storage: TextStorage,
    ) -> Result<TextResourceInventory> {
        let cost = cold_native_cost(storage)?;
        let layout = layout::Layout::derive_cost(
            recorded,
            &math::scratch_layout(),
            host,
            storage,
            cost.native_host,
            cost.native_device,
            cost.context_host,
            cost.context_device,
        )?;
        if layout.inventory.complete_physical_device_bytes > cost.provider
            || layout.inventory.device_with_six_roots_bytes > cost.provider - cost.context_device
            || layout.inventory.host_upper_bytes > cost.host
        {
            return Err(Error::Limit("BF16 text complete cold source/native/context/root/upload inventory exceeds fixed profile".into()));
        }
        Ok(layout.inventory)
    }
    #[cfg(all(
        feature = "qualification",
        feature = "hip",
        feature = "qwen-source-qualification"
    ))]
    pub(crate) fn scoped_cold_resource_inventory(
        recorded: &TextSourcePlan,
        host: &TextHostOwners,
    ) -> Result<TextResourceInventory> {
        let cost = TextNativeOwners::profile(TextStorage::ResidentProjections)?.cost();
        let native_host = cost
            .native_host
            .checked_add(crate::hip::native::scoped_metadata_upper_bytes()?)
            .ok_or_else(|| Error::Limit("BF16 text scoped host sum".into()))?;
        let layout = layout::Layout::derive_cost(
            recorded,
            &math::scratch_layout(),
            host,
            TextStorage::ResidentProjections,
            native_host,
            cost.native_device,
            cost.context_host,
            cost.context_device,
        )?;
        Ok(layout.inventory)
    }
    pub(crate) fn resource_inventory(
        recorded: &TextSourcePlan,
        host: &TextHostOwners,
        native: &TextNativeOwners,
        storage: TextStorage,
    ) -> Result<TextResourceInventory> {
        Ok(
            layout::Layout::derive(recorded, &math::scratch_layout(), host, native, storage)?
                .inventory,
        )
    }
    pub(crate) async fn prepare(
        recorded: Arc<TextSourcePlan>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        host: &TextHostOwners,
        native: Arc<TextNativeOwners>,
        storage: TextStorage,
        flag: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<PendingBf16Text> {
        init.require_service(&service)?;
        Self::require_capability(service.client())?;
        native.require(service.client(), recorded.plan().source())?;
        let layout =
            layout::Layout::derive(&recorded, &math::scratch_layout(), host, &native, storage)?;
        if service.source_identity() != recorded.plan().source()
            || service.client().is_capturing()
            || layout
                .inventory
                .device_with_six_roots_bytes
                .checked_sub(native.device_bytes())
                .ok_or_else(|| Error::State("BF16 text native inventory exceeds total".into()))?
                > service.remaining_capacity()?
            || layout.inventory.largest_binding_bytes as u64
                > service.client().properties().memory.max_page_size
        {
            return Err(Error::Limit(
                "BF16 text complete original service/source/binding/resource preflight".into(),
            ));
        }
        let selection = storage::Selection::derive(&recorded, storage)?;
        if service.client().properties().hardware.max_cube_count.0
            < selection.projection_rows as u32
        {
            return Err(Error::Unsupported(
                "BF16 text complete independent row grid exceeds device limit".into(),
            ));
        }
        let source = recorded.source().clone();
        let mut matrices = Vec::with_capacity(selection.operands.len());
        for operand in &selection.operands {
            cancelled(flag)?;
            operand.require_capability(&service)?;
            matrices.push(operand.upload(init, flag).await?);
        }
        let metadata =
            source.reserve_workspace(4 * 1024 * 1024 + layout::LOWERER_METADATA_BYTES)?;
        #[cfg(feature = "hip")]
        let actor = native.source_page_actor(&service);
        #[cfg(not(feature = "hip"))]
        let actor: Option<source_page::PageActor> = None;
        let page = match actor {
            #[cfg(feature = "hip")]
            Some(actor) => source_page::ResidentSourcePage::allocate_observed(
                source.clone(),
                service.clone(),
                init,
                Some(actor),
            )?,
            _ => source_page::ResidentSourcePage::allocate(source.clone(), service.clone(), init)?,
        };
        let frame = ResidentFrameCheckpoint::new(source)?;
        let slots = layout
            .slots
            .iter()
            .map(|n| init.allocate_empty(*n, BufferRole::Activation))
            .collect::<Result<Vec<_>>>()?;
        let mut gammas = Vec::new();
        for (constant, n) in recorded.plan().constants().iter().zip(&layout.gammas) {
            let mut bytes = vec![0u8; *n];
            for (word, value) in bytes.chunks_exact_mut(4).zip(constant.values.iter()) {
                word.copy_from_slice(&value.to_le_bytes());
            }
            gammas.push(init.upload_bytes(&bytes, BufferRole::ImmutableWeights)?);
        }
        let math = layout
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
        let compact_causal = init.allocate_empty(2048, BufferRole::Scratch)?;
        let packed_input = init.allocate_empty(12288 * 2, BufferRole::Scratch)?;
        let projection_accumulation =
            init.allocate_empty(selection.projection_rows * 4, BufferRole::Scratch)?;
        let projection_status =
            init.allocate_empty(selection.projection_rows * 4, BufferRole::Scratch)?;
        let status = init.allocate_empty(layout::ALIGNMENT, BufferRole::Scratch)?;
        let zero = init.upload_bytes(&[0u8; layout::ALIGNMENT], BufferRole::ImmutableWeights)?;
        let mut accesses = Vec::new();
        accesses.extend(slots.iter().cloned().map(|l| (l, DeviceAccess::Write)));
        accesses.extend(
            gammas
                .iter()
                .chain(&tables)
                .cloned()
                .map(|l| (l, DeviceAccess::Read)),
        );
        accesses.extend(math.iter().cloned().map(|l| (l, DeviceAccess::Write)));
        for lease in [
            &compact_causal,
            &packed_input,
            &projection_accumulation,
            &projection_status,
            &status,
            page.lease(),
        ] {
            accesses.push((lease.clone(), DeviceAccess::Write));
        }
        accesses.push((zero.clone(), DeviceAccess::Read));
        accesses.extend(
            matrices
                .iter()
                .filter_map(|m| m.as_ref())
                .map(|m| (m.lease().clone(), DeviceAccess::Read)),
        );
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
        let actual = layout::sum(accesses.iter().map(|(l, _)| l.bytes()))?;
        if actual != layout.inventory.owned_device_bytes {
            return Err(Error::State(
                "BF16 text actual allocation inventory differs".into(),
            ));
        }
        let observations = recorded
            .observations()
            .iter()
            .cloned()
            .map(|o| (o.node, o))
            .collect();
        let body = Arc::new(Self {
            service,
            recorded,
            layout,
            slots,
            gammas,
            math,
            tables,
            page,
            selection,
            frame,
            compact_causal,
            packed_input,
            projection_accumulation,
            projection_status,
            status,
            zero,
            values,
            observations,
            accesses,
            epochs: Mutex::new(Vec::new()),
            next: AtomicUsize::new(0),
            row: AtomicUsize::new(usize::MAX),
            current_call: Mutex::new(None),
            producer_readback_bytes: AtomicUsize::new(0),
            kv_diagnostic_readback_bytes: AtomicUsize::new(0),
            _metadata: metadata,
            _native: native,
        });
        init.retain_completion_owner(body.clone())?;
        Ok(PendingBf16Text { body, matrices })
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    fn value_lease(&self, id: ValueId) -> &ResidentLease {
        &self.slots[self.layout.bindings[id.0 as usize].slot]
    }
    pub(super) fn begin(
        &self,
        call: &Bf16TextCall,
        p: RuntimeParameters,
        row: usize,
    ) -> Result<()> {
        self.service.ensure_usable()?;
        let actual = call.row(row)?;
        let mut current = self
            .current_call
            .lock()
            .map_err(|_| Error::State("BF16 text current actual call poisoned".into()))?;
        if actual.position() != p.position
            || actual.valid_tokens() != p.valid_tokens
            || actual.token() != p.token
            || row != 0 && self.row.load(Ordering::Acquire) != row - 1
        {
            return Err(Error::State(
                "BF16 text original current call row/order/token/frontier".into(),
            ));
        }
        if row == 0 {
            if current.is_some()
                && self.next.load(Ordering::Acquire) != self.recorded.plan().nodes().len()
            {
                return Err(Error::State(
                    "BF16 text previous source call incomplete".into(),
                ));
            }
            *current = Some(ActualCall {
                base: call.attempt().base(),
                frontier: call.attempt().frontier(),
                ids: Arc::from(call.ids()),
            });
        } else if !current.as_ref().is_some_and(|actual| {
            actual.base == call.attempt().base()
                && actual.frontier == call.attempt().frontier()
                && actual.ids.as_ref() == call.ids()
        }) || self.next.load(Ordering::Acquire)
            != self.recorded.plan().state_nodes().len()
        {
            return Err(Error::State(
                "BF16 text foreign call IDs or incomplete previous row".into(),
            ));
        }
        self.row.store(row, Ordering::Release);
        self.next.store(0, Ordering::Release);
        *self
            .epochs
            .lock()
            .map_err(|_| Error::State("BF16 text slot epochs poisoned".into()))? = vec![
            SlotEpoch {
                value: None,
                epoch: 0
            };
            self.slots
                .len()
        ];
        Ok(())
    }
    fn require_node(&self, index: usize, node: &PlanNode) -> Result<()> {
        if self.next.load(Ordering::Acquire) != index
            || !same_node(&self.recorded.plan().nodes()[index], node)
        {
            return Err(Error::State(
                "BF16 text actual source SSA node/order".into(),
            ));
        }
        let epochs = self
            .epochs
            .lock()
            .map_err(|_| Error::State("BF16 text slot epochs poisoned".into()))?;
        for id in &node.inputs {
            let b = self.layout.bindings[id.0 as usize];
            if epochs[b.slot].value != Some(*id) || epochs[b.slot].epoch != b.epoch {
                return Err(Error::State(
                    "BF16 text stale producer storage epoch".into(),
                ));
            }
        }
        Ok(())
    }
    pub(super) fn finish_node(&self, index: usize, node: &PlanNode) -> Result<()> {
        let mut epochs = self
            .epochs
            .lock()
            .map_err(|_| Error::State("BF16 text slot epochs poisoned".into()))?;
        for id in &node.outputs {
            let b = self.layout.bindings[id.0 as usize];
            epochs[b.slot] = SlotEpoch {
                value: Some(*id),
                epoch: b.epoch,
            };
        }
        self.next
            .compare_exchange(index, index + 1, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::State("BF16 text concurrent/replayed source node".into()))?;
        Ok(())
    }
    pub(super) fn require_finished(&self, output: ResidentOutput) -> Result<()> {
        if self.next.load(Ordering::Acquire) != output.nodes(self.recorded.plan()).len() {
            return Err(Error::State(
                "BF16 text complete original row traversal missing".into(),
            ));
        }
        Ok(())
    }
    pub(super) fn validate_runtime(&self, client: &Client, p: RuntimeParameters) -> Result<()> {
        if client.service_id() != self.service.client().service_id()
            || p.valid_tokens > 64
            || p.valid_tokens != p.position + 1
        {
            return Err(Error::State(
                "BF16 text actual physical service/capacity".into(),
            ));
        }
        self.service.ensure_usable()
    }
    pub(super) fn enqueue(
        &self,
        index: usize,
        node: &PlanNode,
        p: RuntimeParameters,
        bindings: &[StateBinding],
        execution: &dyn Execution,
        flag: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<usize> {
        cancelled(flag)?;
        self.service.ensure_usable()?;
        self.require_node(index, node)?;
        let row = execution.call().row(execution.row())?;
        let inputs = node
            .inputs
            .iter()
            .map(|id| self.values[id.0 as usize].clone())
            .collect::<Vec<_>>();
        let outputs = node
            .outputs
            .iter()
            .map(|id| self.values[id.0 as usize].clone())
            .collect::<Vec<_>>();
        let client = self.service.client();
        match &node.operation {
            PlanOperation::Embed { tensor, .. } => {
                let matrix = self.recorded.source().matrix(&tensor.name, tensor.axes)?;
                let token = row.token() as usize;
                let out = self.value_lease(node.outputs[0]).clone();
                if let Some(operand) = self.selection.nodes[index]
                    && let Some(resident) = self.selection.operands[operand].resident()
                {
                    return execution.resident(
                        resident,
                        self.selection.operands[operand].source(),
                        &matrix,
                        token..token + 1,
                        &mut |view| {
                            if !view.matrix().same_binding(&matrix)
                                || view.rows() != (token..token + 1)
                                || view.bytes() != tensor.columns * 2
                            {
                                return Err(Error::State(
                                    "BF16 text exact tied resident embedding row".into(),
                                ));
                            }
                            unsafe {
                                transport::widen::launch_unchecked(
                                    client,
                                    grid(tensor.columns),
                                    threads(),
                                    arg(view.handle(), tensor.columns / 2),
                                    arg(&outputs[0].handle, tensor.columns),
                                    arg(&self.status.handle(), 1),
                                    1u32,
                                );
                            }
                            Ok(1)
                        },
                    );
                }
                execution.page(
                    &self.page,
                    &matrix,
                    token..token + 1,
                    self.layout.inventory.resident_matrices > 0,
                    &mut |page| {
                        if !std::ptr::eq(page.matrix(), &matrix)
                            || page.rows() != (token..token + 1)
                            || page.bytes() != tensor.columns * 2
                        {
                            return Err(Error::State(
                                "BF16 text exact selected source row page".into(),
                            ));
                        }
                        unsafe {
                            transport::widen::launch_unchecked(
                                client,
                                grid(tensor.columns),
                                threads(),
                                arg(page.handle(), tensor.columns / 2),
                                arg(&outputs[0].handle, tensor.columns),
                                arg(&self.status.handle(), 1),
                                1u32,
                            );
                        }
                        Ok(vec![out.clone()])
                    },
                )?;
                Ok(1)
            }
            PlanOperation::Project { tensor } => self.project(
                index,
                &row,
                tensor,
                &inputs[0],
                &outputs[0],
                node.outputs[0],
                execution,
            ),
            PlanOperation::View(view) => {
                self.copy::<true>(&inputs[0], &outputs[0], *view);
                Ok(1)
            }
            PlanOperation::Contiguous => {
                self.copy::<true>(
                    &inputs[0],
                    &outputs[0],
                    TensorView::vector(0, inputs[0].elements),
                );
                Ok(1)
            }
            PlanOperation::KvPrefixGuard { .. } => Ok(0),
            PlanOperation::AppendKv => {
                let effect = node
                    .state
                    .ok_or_else(|| Error::State("BF16 text append actual state seal".into()))?;
                let binding = bindings
                    .iter()
                    .find(|b| b.slot == effect.read.slot)
                    .ok_or_else(|| Error::State("BF16 text actual KV owner absent".into()))?;
                let (
                    StateBuffers::Kv {
                        keys: pk,
                        values: pv,
                    },
                    StateBuffers::Kv {
                        keys: ck,
                        values: cv,
                    },
                ) = (&binding.prior, &binding.candidate)
                else {
                    return Err(Error::State("BF16 text original KV binding type".into()));
                };
                let slab = inputs[0].elements;
                unsafe {
                    crate::resident_bindings::append_kv::launch_unchecked(
                        client,
                        grid(slab * 64),
                        threads(),
                        arg(&pk.handle, slab * 64),
                        arg(&pv.handle, slab * 64),
                        arg(&inputs[0].handle, slab),
                        arg(&inputs[1].handle, slab),
                        arg(&ck.handle, slab * 64),
                        arg(&cv.handle, slab * 64),
                        p.position as u32,
                        slab,
                        64,
                    );
                }
                Ok(1)
            }
            PlanOperation::Observe { .. } => {
                self.observe(index, node, &row, execution)?;
                Ok(usize::from(
                    execution.observing()
                        && matches!(
                            self.recorded.plan().values()[node.inputs[0].0 as usize],
                            ValueShape::CausalRows { .. }
                        ),
                ))
            }
            PlanOperation::CheckFinite { .. } => Ok(0), // original loop certifies every produced value
            operation => self.math(index, operation, &inputs, &outputs, bindings, &row),
        }
    }
    fn copy<const FINITE: bool>(
        &self,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        view: TensorView,
    ) {
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
                FINITE,
            );
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn project(
        &self,
        index: usize,
        row: &Bf16TextRow<'_>,
        tensor: &SourceTensor,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
        id: ValueId,
        execution: &dyn Execution,
    ) -> Result<usize> {
        let (matrix, geometry) = self.recorded.projection_for_row(index, row)?;
        if geometry.input_columns != tensor.columns
            || geometry.output_rows != tensor.rows
            || geometry.query_rows != execution.call().rows()
        {
            return Err(Error::State(
                "BF16 text original full-K projection/callM".into(),
            ));
        }
        row.projection_geometry(tensor)?;
        let operand = self.selection.nodes[index]
            .and_then(|i| self.selection.operands.get(i))
            .ok_or_else(|| Error::State("BF16 text prepared original projection absent".into()))?;
        if !operand.matrix().same_binding(matrix) {
            return Err(Error::State(
                "BF16 text prepared projection source member differs".into(),
            ));
        }
        execution.projection(
            operand,
            &self.page,
            Bf16ProjectionBindings {
                input: &input.handle,
                input_lease: self.value_lease(self.recorded.plan().nodes()[index].inputs[0]),
                output: &output.handle,
                output_lease: self.value_lease(id),
                packed: &self.packed_input,
                accumulation: &self.projection_accumulation,
                flags: &self.projection_status,
                status: &self.status,
                zero: &self.zero,
            },
        )
    }

    fn observe(
        &self,
        index: usize,
        node: &PlanNode,
        row: &Bf16TextRow<'_>,
        execution: &dyn Execution,
    ) -> Result<()> {
        if !execution.observing() {
            return Ok(());
        }
        self.page.drain()?;
        let observation = self
            .observations
            .get(&index)
            .ok_or_else(|| Error::State("BF16 text actual source producer frame absent".into()))?;
        let id = node.inputs[0];
        let source = &self.values[id.0 as usize];
        let bytes = observation.bytes(row)?;
        let read = match self.recorded.plan().values()[id.0 as usize] {
            ValueShape::CausalRows { heads, capacity } => {
                let output = DeviceBuffer::new(trimmed(&self.compact_causal, 0, bytes)?, bytes / 4);
                self.copy::<false>(
                    source,
                    &output,
                    TensorView {
                        offset: 0,
                        width: row.valid_tokens(),
                        row_stride: capacity,
                        rows: heads,
                    },
                );
                self.compact_causal.readback(0, bytes)?
            }
            _ => self.value_lease(id).readback(0, bytes)?,
        };
        execution.observe(self, observation, &read)
    }
    fn math(
        &self,
        index: usize,
        operation: &PlanOperation,
        inputs: &[DeviceBuffer],
        outputs: &[DeviceBuffer],
        bindings: &[StateBinding],
        row: &Bf16TextRow<'_>,
    ) -> Result<usize> {
        let client = self.service.client();
        let status = trimmed(&self.status, 0, 4)?;
        let zero = trimmed(&self.zero, 0, 4)?;
        let handles = self
            .math
            .iter()
            .map(ResidentLease::handle)
            .collect::<Vec<_>>();
        let scratch = math::MathScratch::new(&handles)?;
        let tables = self
            .tables
            .iter()
            .map(ResidentLease::handle)
            .collect::<Vec<_>>();
        let kv = |version: StateVersion| -> Result<(&DeviceBuffer, &DeviceBuffer)> {
            let binding = bindings
                .iter()
                .find(|b| b.slot == version.slot)
                .ok_or_else(|| {
                    Error::State("BF16 text source attention original KV domain absent".into())
                })?;
            match if version.ordinal == 0 {
                &binding.prior
            } else {
                &binding.candidate
            } {
                StateBuffers::Kv { keys, values } => Ok((keys, values)),
                _ => Err(Error::State(
                    "BF16 text source attention original paired KV".into(),
                )),
            }
        };
        match operation {
            PlanOperation::Rms { gamma, eps, groups } => {
                if eps.to_bits() != 1e-6f32.to_bits() {
                    return Err(Error::State("BF16 text direct RMS source epsilon".into()));
                }
                let (width, raw) = match self.recorded.normalization_source(index)? {
                    SourceNorm::Direct(parameter) => {
                        let expected = self.recorded.plan().constants()[gamma.0 as usize]
                            .values
                            .len();
                        if parameter.shape != [expected as u64] {
                            return Err(Error::State("BF16 text direct source gamma shape".into()));
                        }
                        (
                            expected,
                            Some(trimmed(&self.gammas[gamma.0 as usize], 0, expected * 4)?),
                        )
                    }
                    SourceNorm::Unit { width } => (*width, None),
                };
                math::rms(
                    client,
                    &scratch,
                    &inputs[0].handle,
                    raw.as_ref(),
                    &outputs[0].handle,
                    &status,
                    &zero,
                    width,
                    *groups,
                )
            }
            PlanOperation::TextElementwise(op) => math::elementwise(
                client,
                &inputs[0].handle,
                inputs.get(1).map(|v| &v.handle),
                &outputs[0].handle,
                &status,
                inputs[0].elements,
                *op,
            ),
            PlanOperation::TextRotary { geometry, .. } => math::rotary(
                client,
                &scratch,
                &tables,
                &inputs[0].handle,
                &outputs[0].handle,
                &status,
                inputs[0].elements,
                row.position(),
                *geometry,
            ),
            PlanOperation::TextWindowQk {
                state,
                geometry,
                window,
                ..
            } => {
                let (keys, _) = kv(*state)?;
                math::qk(
                    client,
                    &inputs[0].handle,
                    &keys.handle,
                    &outputs[0].handle,
                    &outputs[1].handle,
                    &status,
                    &zero,
                    geometry.width,
                    row.valid_tokens(),
                    *window,
                )
            }
            PlanOperation::AttentionSoftmax { heads: 8, .. } => math::softmax(
                client,
                &scratch,
                &inputs[0].handle,
                &outputs[0].handle,
                &status,
                row.valid_tokens(),
            ),
            PlanOperation::AttentionValue {
                state, geometry, ..
            } => {
                let (_, values) = kv(*state)?;
                let actual = row.pv_geometry(geometry.width)?;
                math::pv(
                    client,
                    &inputs[0].handle,
                    &values.handle,
                    &outputs[0].handle,
                    &status,
                    &zero,
                    geometry.width,
                    actual.active,
                    actual.query_rows,
                    actual.contraction,
                )
            }
            _ => Err(Error::Unsupported(
                "operation outside complete source-bound BF16 text text contract".into(),
            )),
        }
    }
}

fn same_node(a: &PlanNode, b: &PlanNode) -> bool {
    if a.name != b.name || a.inputs != b.inputs || a.outputs != b.outputs || a.state != b.state {
        return false;
    }
    let attention =
        |a: &zerogravity_core::accelerator::program::AttentionGeometry,
         b: &zerogravity_core::accelerator::program::AttentionGeometry| {
            a.width == b.width && a.query_heads == b.query_heads && a.kv_heads == b.kv_heads
        };
    match (&a.operation, &b.operation) {
        (
            PlanOperation::Embed {
                tensor: a,
                token: ta,
            },
            PlanOperation::Embed {
                tensor: b,
                token: tb,
            },
        ) => a == b && ta == tb,
        (PlanOperation::Project { tensor: a }, PlanOperation::Project { tensor: b }) => a == b,
        (
            PlanOperation::Rms {
                gamma: a,
                eps: ea,
                groups: ga,
            },
            PlanOperation::Rms {
                gamma: b,
                eps: eb,
                groups: gb,
            },
        ) => a == b && ea.to_bits() == eb.to_bits() && ga == gb,
        (PlanOperation::View(a), PlanOperation::View(b)) => {
            a.offset == b.offset
                && a.width == b.width
                && a.rows == b.rows
                && a.row_stride == b.row_stride
        }
        (PlanOperation::TextElementwise(a), PlanOperation::TextElementwise(b)) => a == b,
        (
            PlanOperation::TextRotary {
                geometry: a,
                position: pa,
            },
            PlanOperation::TextRotary {
                geometry: b,
                position: pb,
            },
        ) => a == b && pa == pb,
        (
            PlanOperation::TextWindowQk {
                state: a,
                geometry: ga,
                valid_tokens: va,
                window: wa,
            },
            PlanOperation::TextWindowQk {
                state: b,
                geometry: gb,
                valid_tokens: vb,
                window: wb,
            },
        ) => a == b && attention(ga, gb) && va == vb && wa == wb,
        (
            PlanOperation::AttentionSoftmax {
                heads: a,
                valid_tokens: va,
            },
            PlanOperation::AttentionSoftmax {
                heads: b,
                valid_tokens: vb,
            },
        ) => a == b && va == vb,
        (
            PlanOperation::AttentionValue {
                state: a,
                geometry: ga,
                valid_tokens: va,
                padding: pa,
            },
            PlanOperation::AttentionValue {
                state: b,
                geometry: gb,
                valid_tokens: vb,
                padding: pb,
            },
        ) => a == b && attention(ga, gb) && va == vb && pa == pb,
        (
            PlanOperation::KvPrefixGuard {
                state: a,
                position: pa,
            },
            PlanOperation::KvPrefixGuard {
                state: b,
                position: pb,
            },
        ) => a == b && pa == pb,
        (PlanOperation::CheckFinite { layer: a }, PlanOperation::CheckFinite { layer: b }) => {
            a == b
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
        (PlanOperation::AppendKv, PlanOperation::AppendKv)
        | (PlanOperation::Contiguous, PlanOperation::Contiguous) => true,
        _ => false,
    }
}

impl PendingBf16Text {
    pub(crate) fn bind(self) -> Result<ResidentProgram> {
        let body = self.body;
        if self.matrices.len() != body.selection.operands.len() {
            return Err(Error::State(
                "BF16 text resident source coverage count".into(),
            ));
        }
        for (operand, pending) in body.selection.operands.iter().zip(self.matrices) {
            operand.bind_initialized(pending)?;
        }
        if body
            .gammas
            .iter()
            .chain(&body.tables)
            .chain(std::iter::once(&body.zero))
            .any(|l| !l.is_initialized())
        {
            return Err(Error::State(
                "BF16 text original immutable source initialization incomplete".into(),
            ));
        }
        let mut owned_ranges = OwnedRanges::new(body.service.client().service_id());
        let mut ranges = Vec::<BufferRange>::new();
        for (lease, _) in &body.accesses {
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(lease.bytes())?;
            if ranges.iter().any(|r| r.overlaps(&range)) {
                return Err(Error::State(
                    "BF16 text original logical allocations overlap".into(),
                ));
            }
            owned_ranges.insert(range.clone())?;
            ranges.push(range);
        }
        let inventory = &body.layout.inventory;
        let source_bytes = layout::sum(
            body.gammas
                .iter()
                .chain(&body.tables)
                .chain(std::iter::once(&body.zero))
                .chain(
                    body.selection
                        .operands
                        .iter()
                        .filter_map(|o| o.resident())
                        .map(|m| m.lease()),
                )
                .map(ResidentLease::bytes),
        )?;
        let status_bytes = body.status.bytes();
        let preflight = ResidentPreflight {
            source: body.recorded.plan().source(),
            source_bytes,
            value_bytes: inventory.value_slot_bytes,
            workspace_bytes: inventory.owned_device_bytes
                - source_bytes
                - inventory.value_slot_bytes
                - status_bytes,
            status_bytes,
            parameter_bytes: 0,
            prior_state_bytes: inventory.single_root_payload_bytes,
            candidate_state_bytes: inventory.single_root_payload_bytes,
            states: body.layout.states.clone(),
            scratch: [0; SCRATCH_SLOTS],
        };
        Ok(ResidentProgram {
            _source_owner: ResidentProgramSource::Windows(body.recorded.source().clone()),
            plan: Arc::clone(body.recorded.shared_plan()),
            preflight,
            values: body.values.clone(),
            status: body.status.handle(),
            owned_ranges,
            body: ResidentProgramBody::Bf16Text(body),
        })
    }
}

impl ResidentProgram {
    pub(crate) fn bf16_text_traffic(&self) -> Result<TextTraffic> {
        let ResidentProgramBody::Bf16Text(body) = &self.body else {
            return Err(Error::State(
                "BF16 text traffic needs its original prepared body".into(),
            ));
        };
        let row_traffic = body.page.row_traffic()?;
        Ok(TextTraffic {
            selected_row_uploads: row_traffic.completed_uploads,
            selected_row_reuses: row_traffic.completed_reuses,
            selected_row_avoided_upload_bytes: row_traffic.avoided_upload_bytes,
            value_upload_bytes: body.service.uploaded_value_bytes(),
            resident_weight_bytes: body.layout.inventory.resident_matrix_bytes,
            producer_readback_bytes: body.producer_readback_bytes.load(Ordering::Acquire),
            kv_diagnostic_readback_bytes: body.kv_diagnostic_readback_bytes.load(Ordering::Acquire),
        })
    }
    pub(crate) fn bf16_text_accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let ResidentProgramBody::Bf16Text(body) = &self.body else {
            return Err(Error::Unsupported(
                "distinct BF16 text source body required".into(),
            ));
        };
        Ok(body.accesses.clone())
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn checkpoint_bf16_text_kv(
        &self,
        flight: &mut crate::resident_leases::ResidentSubmission<'_>,
        lease: &ResidentLease,
        prior: bool,
        domain: zerogravity_core::accelerator::program::KvDomain,
        value: bool,
        active: usize,
        observer: &mut dyn TextObserver,
    ) -> Result<()> {
        let ResidentProgramBody::Bf16Text(body) = &self.body else {
            return Err(Error::Unsupported(
                "distinct BF16 text source body required".into(),
            ));
        };
        let read = lease.readback(0, lease.bytes())?;
        body.frame
            .compare(flight, &read, |bytes| {
                body.kv_diagnostic_readback_bytes
                    .fetch_add(bytes.len(), Ordering::AcqRel);
                let slab = lease.bytes() / 4 / 64;
                let end = active
                    .checked_mul(slab)
                    .and_then(|n| n.checked_mul(4))
                    .filter(|n| *n <= bytes.len())
                    .ok_or_else(|| Error::State("BF16 text actual KV frame extent".into()))?;
                if bytes[end..].iter().any(|b| *b != 0) {
                    return Err(Error::State(
                        "BF16 text canonical unused KV tail is not +0".into(),
                    ));
                }
                observer.completed_kv(prior, domain, value, active, &bytes[..end])
            })
            .await
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_bf16_text(
        &mut self,
        flight: &mut crate::resident_leases::ResidentSubmission<'_>,
        call: &Bf16TextCall,
        current: zerogravity_core::session_root::RootBinding,
        row: usize,
        bindings: &[StateBinding],
        observer: &RefCell<Option<&mut dyn TextObserver>>,
        flag: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ResidentSubmission> {
        let ResidentProgramBody::Bf16Text(body) = &self.body else {
            return Err(Error::Unsupported(
                "distinct BF16 text source-bound body required".into(),
            ));
        };
        let body = body.clone();
        cancelled(flag)?;
        call.require_current(current)?;
        if current.source() != body.recorded.plan().source() {
            return Err(Error::State("BF16 text actual call source differs".into()));
        }
        flight.require_accesses(body.accesses.iter().map(|(l, a)| (l, *a)))?;
        flight.retain_completion_owner(body.clone())?;
        let actual = call.row(row)?;
        let parameters = RuntimeParameters {
            token: actual.token(),
            position: actual.position(),
            valid_tokens: actual.valid_tokens(),
        };
        body.begin(call, parameters, row)?;
        let execution = ArmedExecution {
            flight: RefCell::new(flight),
            observer,
            row,
            call,
            cancelled: flag,
        };
        let submission = self.launch_impl(
            body.service.client(),
            CurrentExecution::Single {
                parameters,
                states: bindings,
            },
            None,
            if actual.needs_logits() {
                ResidentOutput::Logits
            } else {
                ResidentOutput::StateOnly
            },
            NativeExecution::new(flag).with_bf16_text(&execution),
        )?;
        body.page.finish()?;
        Ok(submission)
    }
}

#[cfg(all(
    feature = "qualification",
    any(feature = "hip", feature = "vulkan"),
    feature = "hardware-abort-probe"
))]
impl ResidentProgram {
    pub(crate) fn bf16_text_final_owner_native_probe(
        &self,
    ) -> Result<crate::backend::final_native::FinalNativeProbe> {
        match &self.body {
            ResidentProgramBody::Bf16Text(body) => body._native.final_owner_probe(),
            _ => Err(Error::State(
                "BF16 text original program/native owner differs".into(),
            )),
        }
    }
}

#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    target_os = "linux"
))]
impl ResidentProgram {
    pub(crate) fn bf16_text_vulkan_native_identity(
        &self,
        service: &ResidentService,
    ) -> Result<crate::vulkan::native::VulkanSourceSetIdentity> {
        match &self.body {
            ResidentProgramBody::Bf16Text(body) => body._native.vulkan_identity(service),
            _ => Err(Error::State("BF16 text original Vulkan body absent".into())),
        }
    }
}

#[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
impl ResidentProgram {
    pub(crate) fn bf16_text_native_identity(
        &self,
        service: &ResidentService,
    ) -> Result<crate::hip::native::HipSourceSetIdentity> {
        match &self.body {
            ResidentProgramBody::Bf16Text(body) => body._native.identity(service),
            _ => Err(Error::State("BF16 text original body absent".into())),
        }
    }
}
