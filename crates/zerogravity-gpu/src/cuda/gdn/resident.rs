//! Prepared original-service phase; caller owns the armed physical flight.
use super::*;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{client::Client, server::Handle};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::{AtomicBool, Ordering},
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

mod arithmetic;
use arithmetic::CopyArithmetic;
pub(crate) use arithmetic::GdnArithmetic;

#[cfg(target_os = "linux")]
mod component;
#[cfg(target_os = "linux")]
pub(crate) mod solve_leaf;
#[cfg(target_os = "linux")]
mod state_adapter;
#[cfg(target_os = "linux")]
pub(crate) use component::{ComponentConvEnqueued, ComponentGdnAdapter};
#[cfg(target_os = "linux")]
pub(crate) use state_adapter::GdnStateAdapter;
#[cfg(target_os = "linux")]
mod model;
#[cfg(target_os = "linux")]
mod split;
#[cfg(target_os = "linux")]
pub(crate) use model::{ModelGdnAdapter, ModelGdnConvEnqueued, ModelGdnTransportStorage};
#[cfg(target_os = "linux")]
pub(crate) use split::GdnConvEnqueued;

/// Actual computed GEMM outputs, original source parameters and prior cache.
/// No intermediate override or expected-tensor input exists in this interface.
pub(crate) struct GdnTypedOutputs {
    pub activated: ResidentLease,
    pub z: ResidentLease,
    pub gated: ResidentLease,
}
pub(crate) struct GdnInputs {
    pub qkvz: ResidentLease,
    pub ba: ResidentLease,
    pub cache: ResidentLease,
    pub conv_weights: ResidentLease,
    pub a_log: ResidentLease,
    pub dt_bias: ResidentLease,
    pub gamma: ResidentLease,
}
#[derive(Clone)]
struct Stage {
    spec: LaunchSpec,
    launch: Option<crate::cuda::kernel::PreparedPinnedLaunch>,
    stage: GdnStage,
}
/// All allocations/control uploads finish before native provider activation.
/// Retain this Arc in ROOT's CompletionOwner before the first enqueue.
pub(crate) struct GdnPrepared {
    arithmetic: Arc<GdnArithmetic>,
    stages: Arc<[Stage]>,
    packing: Arc<[Copy]>,
    inverse_zero: Option<Copy>,
    initial: Option<Copy>,
    scatter: Option<Copy>,
    buffers: Arc<BTreeMap<GdnBuffer, ResidentLease>>,
    service: Arc<ResidentService>,
    phase: GdnPhase,
    slot: GdnCacheSlot,
    typed_outputs: bool,
    // Original generated source/graph admission outlives kernels and handles.
    code: ComponentCode,
}
enum ComponentCode {
    Default,
    #[cfg(target_os = "linux")]
    Text(Arc<super::text::TextGdnCode>),
    #[cfg(target_os = "linux")]
    Qwen2b(Arc<crate::cuda::native::qwen2b_component::ComponentGdnBinding>),
    #[cfg(target_os = "linux")]
    FixedPrompt(Arc<crate::cuda::native::qwen2b_component::ComponentGdnBinding>),
}
impl ComponentCode {
    fn arithmetic_family(&self) -> super::arithmetic::GdnFamilyIdentity {
        match self {
            Self::Default => super::arithmetic::GdnFamilyIdentity::original08(),
            #[cfg(target_os = "linux")]
            Self::Text(binding) => binding.family(),
            #[cfg(target_os = "linux")]
            Self::Qwen2b(binding) | Self::FixedPrompt(binding) => *binding.arithmetic_family(),
        }
    }
    fn require(&self, service: &Arc<ResidentService>, phase: GdnPhase) -> Result<()> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Text(binding) => binding.require(service, phase),
            Self::Default => {
                let _ = (service, phase);
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Self::Qwen2b(binding) | Self::FixedPrompt(binding) => {
                binding.require_source(service)?;
                if !matches!(
                    (self, binding.values().recipe().kind()),
                    (
                        Self::Qwen2b(_),
                        crate::resident_program::qwen2b_component::ComponentKind::Qwen2b
                    ) | (
                        Self::FixedPrompt(_),
                        crate::resident_program::qwen2b_component::ComponentKind::FixedPrompt(_)
                    )
                ) {
                    return Err(Error::State(
                        "component code discriminator differs from sealed source recipe".into(),
                    ));
                }
                if binding.phase()? != phase {
                    return Err(Error::State("source2B exact code phase".into()));
                }
                Ok(())
            }
        }
    }
    fn kernel(&self, phase: GdnPhase, stage: GdnStage, typed: bool) -> Result<LaunchSpec> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Text(binding) if typed => binding.kernel(phase, stage),
            #[cfg(target_os = "linux")]
            Self::Text(_) => Err(Error::Unsupported(
                "text requires actual typed GDN outputs".into(),
            )),
            Self::Default if typed => kernel_with_typed_outputs(phase, stage),
            Self::Default => kernel(phase, stage),
            #[cfg(target_os = "linux")]
            Self::FixedPrompt(binding) if typed => {
                let crate::resident_program::qwen2b_component::ComponentKind::FixedPrompt(
                    arithmetic,
                ) = binding.values().recipe().kind()
                else {
                    return Err(Error::State(
                        "fixed_prompt actual source-code discriminator differs".into(),
                    ));
                };
                arithmetic.require_owned()?;
                if phase != GdnPhase::FixedPrompt {
                    return Err(Error::Unsupported("fixed_prompt code phase only".into()));
                }
                kernel_with_typed_outputs(phase, stage)
            }
            #[cfg(target_os = "linux")]
            Self::FixedPrompt(_) => Err(Error::State(
                "fixed_prompt requires genuine typed outputs".into(),
            )),
            #[cfg(target_os = "linux")]
            Self::Qwen2b(_) if typed => kernel_qwen2b_component(phase, stage),
            #[cfg(target_os = "linux")]
            Self::Qwen2b(_) => Err(Error::State(
                "source2B code requires genuine typed outputs".into(),
            )),
        }
    }
    fn shares_private_history(&self) -> bool {
        #[cfg(target_os = "linux")]
        if matches!(self, Self::Text(_)) {
            return true;
        }
        false
    }
    fn workspace_control(&self, role: GdnBuffer, bytes: &[u8]) -> Result<Option<ResidentLease>> {
        #[cfg(target_os = "linux")]
        if let Self::Text(code) = self {
            return code.workspace().control(role, bytes).map(Some);
        }
        let _ = (role, bytes);
        Ok(None)
    }
    fn workspace_lease(&self, role: GdnBuffer) -> Result<Option<ResidentLease>> {
        #[cfg(target_os = "linux")]
        if let Self::Text(code) = self {
            return code.workspace().optional_private(role);
        }
        let _ = role;
        Ok(None)
    }
    fn retain(&self, init: &mut crate::resident_leases::ResidentInitialization<'_>) -> Result<()> {
        match self {
            #[cfg(target_os = "linux")]
            Self::Text(binding) => init.retain_completion_owner(binding.clone()),
            Self::Default => {
                let _ = init;
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Self::Qwen2b(binding) | Self::FixedPrompt(binding) => {
                init.retain_completion_owner(binding.clone())
            }
        }
    }
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("closed vLLM GDN phase cancelled".into()));
    }
    Ok(())
}
fn is_read_only(role: GdnBuffer) -> bool {
    matches!(
        role,
        GdnBuffer::ConvWeights
            | GdnBuffer::ALog
            | GdnBuffer::DtBias
            | GdnBuffer::GatedGamma
            | GdnBuffer::CacheIndices
            | GdnBuffer::HasInitial
            | GdnBuffer::QueryStarts
            | GdnBuffer::ConvBatch
            | GdnBuffer::ConvChunkOffsets
            | GdnBuffer::ChunkIndices
            | GdnBuffer::ChunkOffsets
    )
}
fn validate_lease(
    service: &Arc<ResidentService>,
    role: GdnBuffer,
    phase: GdnPhase,
    lease: &ResidentLease,
) -> Result<()> {
    if !Arc::ptr_eq(service, lease.service())
        || lease.source_identity() != service.source_identity()
        || lease.service_generation() != service.generation()
        || lease.bytes() != role.bytes(phase)
        || lease.handle().service != service.client().service_id()
        || is_read_only(role) && !lease.is_initialized()
    {
        return Err(Error::State(
            "closed vLLM actual source/service/lease geometry".into(),
        ));
    }
    Ok(())
}
fn handle(lease: &ResidentLease, spec: BindingSpec) -> Result<Handle> {
    let end = spec
        .offset
        .checked_add(spec.physical_bytes)
        .filter(|&end| end <= lease.bytes())
        .ok_or_else(|| Error::State("closed vLLM resident binding span".into()))?;
    Ok(lease
        .handle()
        .offset_start(spec.offset as u64)
        .offset_end((lease.bytes() - end) as u64))
}
impl GdnPrepared {
    #[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
    pub(crate) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        let mut count = 0;
        for stage in self.stages.iter() {
            stage.spec.kernel.prepare_code(queue)?;
            count += 1;
        }
        for copy in self
            .packing
            .iter()
            .chain(self.inverse_zero.iter())
            .chain(self.initial.iter())
            .chain(self.scatter.iter())
        {
            copy.launch
                .as_ref()
                .ok_or_else(|| Error::State("cold GDN copy definition absent".into()))?
                .prepare_code(queue)?;
            count += 1;
        }
        Ok(count)
    }
    pub(crate) async fn prepare(
        service: Arc<ResidentService>,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        inputs: GdnInputs,
    ) -> Result<Arc<Self>> {
        Self::prepare_with_cancel(service, phase, slot, inputs, None).await
    }
    pub(crate) async fn prepare_with_cancel(
        service: Arc<ResidentService>,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        inputs: GdnInputs,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_impl(
            service,
            phase,
            slot,
            inputs,
            None,
            ComponentCode::Default,
            cancelled,
        )
        .await
    }
    pub(crate) async fn prepare_initial_outputs(
        service: Arc<ResidentService>,
        phase: GdnPhase,
        inputs: GdnInputs,
        outputs: GdnTypedOutputs,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if !matches!(phase, GdnPhase::InitialOne | GdnPhase::Initial64) {
            return Err(Error::Unsupported(
                "typed GDN initial phases only; cached SSA needs separate binding".into(),
            ));
        }
        Self::prepare_impl(
            service,
            phase,
            if phase == GdnPhase::InitialOne {
                GdnCacheSlot::One
            } else {
                GdnCacheSlot::Four
            },
            inputs,
            Some(outputs),
            ComponentCode::Default,
            cancelled,
        )
        .await
    }
    /// Original canonical-state transaction caller supplies genuine public
    /// producer outputs and a phase/slot seal; this grants no root authority.
    pub(crate) async fn prepare_attempt_outputs(
        service: Arc<ResidentService>,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        inputs: GdnInputs,
        outputs: GdnTypedOutputs,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        slot.validate(phase)?;
        Self::prepare_impl(
            service,
            phase,
            slot,
            inputs,
            Some(outputs),
            ComponentCode::Default,
            cancelled,
        )
        .await
    }
    async fn prepare_impl(
        service: Arc<ResidentService>,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        inputs: GdnInputs,
        outputs: Option<GdnTypedOutputs>,
        code: ComponentCode,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        service.ensure_usable()?;
        code.require(&service, phase)?;
        service.pinned_cuda_queue()?;
        let typed_outputs = outputs.is_some();
        let control: Vec<(GdnBuffer, Arc<[u8]>)> = metadata(phase, slot)?
            .into_iter()
            .map(|(role, bytes)| (role, Arc::from(bytes)))
            .collect();
        let arithmetic_controls = Arc::from(control.clone());
        let mut buffers = BTreeMap::from([
            (GdnBuffer::PackedQkvz, inputs.qkvz),
            (GdnBuffer::PackedBa, inputs.ba),
            (GdnBuffer::CacheArena, inputs.cache),
            (GdnBuffer::ConvWeights, inputs.conv_weights),
            (GdnBuffer::ALog, inputs.a_log),
            (GdnBuffer::DtBias, inputs.dt_bias),
            (GdnBuffer::GatedGamma, inputs.gamma),
        ]);
        let mut ids = BTreeSet::new();
        if let Some(outputs) = outputs {
            for (role, lease) in [
                (GdnBuffer::ConvOutput, outputs.activated),
                (GdnBuffer::ZCompact, outputs.z),
                (GdnBuffer::Gated, outputs.gated),
            ] {
                validate_lease(&service, role, phase, &lease)?;
                if buffers.values().any(|prior| prior.id() == lease.id()) {
                    return Err(Error::State(
                        "typed GDN output aliases input/source/sibling".into(),
                    ));
                }
                buffers.insert(role, lease);
            }
        }
        for (&role, lease) in &buffers {
            validate_lease(&service, role, phase, lease)?;
            if !ids.insert(lease.id()) {
                return Err(Error::State(
                    "closed vLLM independent source/producer leases alias".into(),
                ));
            }
        }
        // Compile definitions/handles are prepared while allocations remain
        // ordinary resident storage, before the native allocation pool starts.
        let mut specs = Vec::new();
        for &stage in phase.stages() {
            let spec = code.kernel(phase, stage, typed_outputs)?;
            specs.push((stage, spec));
        }
        let mut required = BTreeSet::new();
        for (_, spec) in &specs {
            for binding in &spec.bindings {
                required.insert(binding.role);
            }
        }
        not_cancelled(cancelled)?;
        let mut initialization = service.begin_initialization(0)?;
        code.retain(&mut initialization)?;
        for (role, value) in control {
            not_cancelled(cancelled)?;
            if !required.contains(&role) {
                return Err(Error::State("closed vLLM unexpected metadata role".into()));
            }
            let lease = if let Some(lease) = code.workspace_control(role, &value)? {
                lease
            } else {
                initialization.upload_bytes(&value, BufferRole::ImmutableWeights)?
            };
            buffers.insert(role, lease);
        }
        for role in required {
            if let std::collections::btree_map::Entry::Vacant(entry) = buffers.entry(role) {
                not_cancelled(cancelled)?;
                let role_class = if matches!(role, GdnBuffer::InitialState | GdnBuffer::FinalState)
                {
                    BufferRole::RecurrentState
                } else {
                    BufferRole::Scratch
                };
                entry.insert(if let Some(lease) = code.workspace_lease(role)? {
                    lease
                } else {
                    initialization.allocate_empty(role.bytes(phase), role_class)?
                });
            }
        }
        let mut stages = Vec::new();
        for (stage, spec) in specs {
            let handles = spec
                .bindings
                .iter()
                .map(|&binding| {
                    handle(
                        buffers.get(&binding.role).ok_or_else(|| {
                            Error::State("closed vLLM prepared role absent".into())
                        })?,
                        binding,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let launch = spec.kernel.prepare(service.client(), spec.grid, handles)?;
            stages.push(Stage {
                spec,
                launch: Some(launch),
                stage,
            });
        }
        let mut packing = Vec::new();
        if phase.tokens() > 1 {
            for kind in [CopyKind::A, CopyKind::B, CopyKind::Z] {
                packing.push(Copy::prepare(&service, phase, slot, kind, &buffers)?);
            }
        }
        if typed_outputs && phase.tokens() == 1 {
            if phase == GdnPhase::CachedDecode {
                packing.push(Copy::prepare(
                    &service,
                    phase,
                    slot,
                    CopyKind::QkvPrefix,
                    &buffers,
                )?);
            }
            packing.push(Copy::prepare(&service, phase, slot, CopyKind::Z, &buffers)?);
        }
        let (initial, scatter) = if phase == GdnPhase::CachedDecode {
            (None, None)
        } else {
            let kind = if phase.has_initial_state() {
                CopyKind::Gather
            } else {
                CopyKind::Zero
            };
            (
                Some(Copy::prepare(&service, phase, slot, kind, &buffers)?),
                Some(Copy::prepare(
                    &service,
                    phase,
                    slot,
                    CopyKind::Scatter,
                    &buffers,
                )?),
            )
        };
        // Original solve_tril uses zeros_like before its lower-block stores.
        let inverse_zero = if phase == GdnPhase::CachedDecode {
            None
        } else {
            Some(Copy::prepare(
                &service,
                phase,
                slot,
                CopyKind::ZeroInverse,
                &buffers,
            )?)
        };
        not_cancelled(cancelled)?;
        let arithmetic = Arc::new(GdnArithmetic::new(
            phase,
            slot,
            typed_outputs,
            arithmetic_controls,
            &stages,
            &packing,
            [inverse_zero.as_ref(), initial.as_ref(), scatter.as_ref()],
        )?);
        code.arithmetic_family().require_same(arithmetic.family())?;
        // Actual preparation completion retires its event pins, not storage.
        drop(initialization.complete_sync().await?);
        not_cancelled(cancelled)?;
        Ok(Arc::new(Self {
            arithmetic,
            stages: stages.into(),
            packing: packing.into(),
            inverse_zero,
            initial,
            scatter,
            buffers: Arc::new(buffers),
            service,
            phase,
            slot,
            typed_outputs,
            code,
        }))
    }
    pub(crate) fn arithmetic(&self) -> &Arc<GdnArithmetic> {
        &self.arithmetic
    }
    pub(crate) fn lease(&self, role: GdnBuffer) -> Result<&ResidentLease> {
        #[cfg(target_os = "linux")]
        if self.code.shares_private_history() && !super::text::retained_history(role) {
            return Err(Error::Unsupported(
                "pooled GDN private intermediate is not a retained per-layer history".into(),
            ));
        }
        self.bound_lease(role)
    }
    fn bound_lease(&self, role: GdnBuffer) -> Result<&ResidentLease> {
        self.buffers
            .get(&role)
            .ok_or_else(|| Error::State("closed vLLM phase has no requested role".into()))
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.buffers
            .iter()
            .map(|(&role, lease)| {
                (
                    lease.clone(),
                    if is_read_only(role) {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    },
                )
            })
            .collect()
    }
    /// No allocation, upload, physical Ready or root publication. Any error
    /// after enqueue must drop ROOT's armed flight into existing quarantine.
    pub(crate) fn launch(&self, cancelled: Option<&AtomicBool>) -> Result<usize> {
        if self.typed_outputs {
            return Err(Error::Unsupported(
                "typed public GDN outputs require original-root split nodes".into(),
            ));
        }
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        if self.phase.has_initial_state()
            && !self.bound_lease(GdnBuffer::CacheArena)?.is_initialized()
        {
            return Err(Error::State(
                "closed vLLM prior cache is not physically Ready".into(),
            ));
        }
        let count = self.launch_partition(StagePartition::Convolution, cancelled)?
            + self.launch_partition(StagePartition::Recurrent, cancelled)?;
        not_cancelled(cancelled)?;
        let expected = match self.phase {
            GdnPhase::InitialOne => 12,
            GdnPhase::CachedDecode => 3,
            _ => 15,
        };
        if count != expected {
            return Err(Error::State("closed vLLM phase dispatch count".into()));
        }
        Ok(count)
    }
    fn launch_partition(
        &self,
        partition: StagePartition,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.service.ensure_usable()?;
        self.code.require(&self.service, self.phase)?;
        let mut count = 0;
        if partition == StagePartition::Convolution {
            for copy in self.packing.iter() {
                not_cancelled(cancelled)?;
                copy.launch(self.service.client())?;
                count += 1;
            }
        }
        for stage in self
            .stages
            .iter()
            .filter(|stage| partition.contains(stage.stage))
        {
            not_cancelled(cancelled)?;
            if stage.stage == GdnStage::Solve {
                self.inverse_zero
                    .as_ref()
                    .ok_or_else(|| Error::State("prefill inverse preparation absent".into()))?
                    .launch(self.service.client())?;
                count += 1;
            }
            if stage.stage == GdnStage::H {
                self.initial
                    .as_ref()
                    .ok_or_else(|| Error::State("prefill state preparation absent".into()))?
                    .launch(self.service.client())?;
                count += 1;
            }
            stage
                .launch
                .as_ref()
                .ok_or_else(|| Error::State("GDN metadata has no prepared launch".into()))?
                .launch()?;
            count += 1;
            if stage.stage == GdnStage::H {
                self.scatter
                    .as_ref()
                    .ok_or_else(|| Error::State("prefill state scatter absent".into()))?
                    .launch(self.service.client())?;
                count += 1;
            }
        }
        not_cancelled(cancelled)?;
        let extra = if partition == StagePartition::Convolution && self.typed_outputs {
            match self.phase {
                GdnPhase::InitialOne => 1,
                GdnPhase::CachedDecode => 2,
                _ => 0,
            }
        } else {
            0
        };
        let expected = partition.dispatches(self.phase) + extra;
        if count != expected {
            return Err(Error::State("closed GDN partition dispatch count".into()));
        }
        Ok(count)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StagePartition {
    Convolution,
    Recurrent,
}
impl StagePartition {
    fn contains(self, stage: GdnStage) -> bool {
        match self {
            Self::Convolution => stage == GdnStage::Conv,
            Self::Recurrent => stage != GdnStage::Conv,
        }
    }
    fn dispatches(self, phase: GdnPhase) -> usize {
        let conv = if phase.tokens() > 1 { 4 } else { 1 };
        match self {
            Self::Convolution => conv,
            Self::Recurrent => match phase {
                GdnPhase::InitialOne => 12 - conv,
                GdnPhase::CachedDecode => 3 - conv,
                _ => 15 - conv,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CopyKind {
    QkvPrefix,
    A,
    B,
    Z,
    Gather,
    Scatter,
    Zero,
    ZeroInverse,
}
#[derive(Clone)]
struct Copy {
    arithmetic: CopyArithmetic,
    launch: Option<crate::cuda::kernel::PreparedPinnedLaunch>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CopyGeometry {
    source: Option<GdnBuffer>,
    destination: GdnBuffer,
    rows: usize,
    words: usize,
    source_stride: usize,
    destination_stride: usize,
    source_offset: usize,
    destination_offset: usize,
}
fn copy_geometry(phase: GdnPhase, slot: GdnCacheSlot, kind: CopyKind) -> Result<CopyGeometry> {
    use GdnBuffer::*;
    slot.validate(phase)?;
    let value = match kind {
        CopyKind::A | CopyKind::B if phase.tokens() > 1 => CopyGeometry {
            source: Some(PackedBa),
            destination: if matches!(kind, CopyKind::A) {
                AContiguous
            } else {
                BContiguous
            },
            rows: phase.tokens(),
            words: 8,
            source_stride: 16,
            destination_stride: 8,
            source_offset: if matches!(kind, CopyKind::A) { 8 } else { 0 },
            destination_offset: 0,
        },
        CopyKind::QkvPrefix if phase == GdnPhase::CachedDecode => CopyGeometry {
            source: Some(PackedQkvz),
            destination: ConvOutput,
            rows: 1,
            words: 3072,
            source_stride: 4096,
            destination_stride: 3072,
            source_offset: 0,
            destination_offset: 0,
        },
        CopyKind::Z => CopyGeometry {
            source: Some(PackedQkvz),
            destination: ZCompact,
            rows: phase.tokens(),
            words: 1024,
            source_stride: 4096,
            destination_stride: 1024,
            source_offset: 3072,
            destination_offset: 0,
        },
        CopyKind::Gather if phase == GdnPhase::ContinuedTwo => CopyGeometry {
            source: Some(CacheArena),
            destination: InitialState,
            rows: 1,
            words: RECURRENT_STATE_BYTES / 4,
            source_stride: RECURRENT_STATE_BYTES / 4,
            destination_stride: RECURRENT_STATE_BYTES / 4,
            source_offset: slot.recurrent_offset() / 4,
            destination_offset: 0,
        },
        CopyKind::Scatter if phase != GdnPhase::CachedDecode => CopyGeometry {
            source: Some(FinalState),
            destination: CacheArena,
            rows: 1,
            words: RECURRENT_STATE_BYTES / 4,
            source_stride: RECURRENT_STATE_BYTES / 4,
            destination_stride: RECURRENT_STATE_BYTES / 4,
            source_offset: 0,
            destination_offset: slot.recurrent_offset() / 4,
        },
        CopyKind::Zero
            if matches!(
                phase,
                GdnPhase::InitialOne
                    | GdnPhase::Initial64
                    | GdnPhase::FixedPrompt
                    | GdnPhase::TextPrefill(_)
            ) =>
        {
            CopyGeometry {
                source: None,
                destination: InitialState,
                rows: 1,
                words: RECURRENT_STATE_BYTES / 4,
                source_stride: 0,
                destination_stride: RECURRENT_STATE_BYTES / 4,
                source_offset: 0,
                destination_offset: 0,
            }
        }
        CopyKind::ZeroInverse if phase != GdnPhase::CachedDecode => CopyGeometry {
            source: None,
            destination: Inverse,
            rows: 1,
            words: Inverse.bytes(phase) / 4,
            source_stride: 0,
            destination_stride: Inverse.bytes(phase) / 4,
            source_offset: 0,
            destination_offset: 0,
        },
        _ => return Err(Error::Unsupported("closed vLLM copy phase".into())),
    };
    let fits = |role: GdnBuffer, offset: usize, stride: usize| -> Result<()> {
        let end = (value.rows - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(offset))
            .and_then(|n| n.checked_add(value.words));
        if end.is_none_or(|words| words > role.bytes(phase) / 4) {
            return Err(Error::State("closed vLLM copy span".into()));
        }
        Ok(())
    };
    if let Some(source) = value.source {
        fits(source, value.source_offset, value.source_stride)?;
    }
    fits(
        value.destination,
        value.destination_offset,
        value.destination_stride,
    )?;
    Ok(value)
}
impl Copy {
    fn prepare(
        service: &Arc<ResidentService>,
        phase: GdnPhase,
        slot: GdnCacheSlot,
        kind: CopyKind,
        buffers: &BTreeMap<GdnBuffer, ResidentLease>,
    ) -> Result<Self> {
        let g = copy_geometry(phase, slot, kind)?;
        let entry = match kind {
            CopyKind::QkvPrefix => "zg_vllm_copy_qkv_prefix",
            CopyKind::A => "zg_vllm_copy_a",
            CopyKind::B => "zg_vllm_copy_b",
            CopyKind::Z => "zg_vllm_copy_z",
            CopyKind::Gather => "zg_vllm_gather_h0",
            CopyKind::Scatter => "zg_vllm_scatter_ht",
            CopyKind::Zero => "zg_vllm_zero_h0",
            CopyKind::ZeroInverse => "zg_vllm_zero_inverse",
        };
        let get = |role| {
            buffers
                .get(&role)
                .map(ResidentLease::handle)
                .ok_or_else(|| Error::State("closed copy original role absent".into()))
        };
        let mut handles = Vec::new();
        let mut bindings = Vec::new();
        if let Some(source) = g.source {
            handles.push(get(source)?);
            bindings.push((source.bytes(phase), BufferIOAttr::ReadOnly));
        }
        handles.push(get(g.destination)?);
        bindings.push((g.destination.bytes(phase), BufferIOAttr::WriteOnly));
        let (parameters, statement) = if g.source.is_some() {
            (
                "const unsigned int* source, unsigned int* destination",
                format!(
                    "unsigned int row = index / {}u; unsigned int col = index % {}u; destination[{} + row * {} + col] = source[{} + row * {} + col];",
                    g.words,
                    g.words,
                    g.destination_offset,
                    g.destination_stride,
                    g.source_offset,
                    g.source_stride
                ),
            )
        } else {
            (
                "unsigned int* destination",
                "destination[index] = 0u;".into(),
            )
        };
        let total = g.rows * g.words;
        let source = format!(
            "// cubecl.cuda.dynamic_shared_bytes=0\n#if __CUDA_ARCH__ != 860\n#error Closed SM86 copy\n#endif\nextern \"C\" __global__ __launch_bounds__(128) void {entry}({parameters}) {{ unsigned int index = blockIdx.x * blockDim.x + threadIdx.x; if (index < {total}u) {{ {statement} }} }}\n"
        );
        let source: Arc<str> = source.into();
        let bindings: Arc<[(usize, BufferIOAttr)]> = bindings.into();
        let kernel = PinnedKernel::new(entry, source.clone(), 128, bindings.clone())?;
        let grid = [(total as u32).div_ceil(128), 1, 1];
        let arithmetic = CopyArithmetic {
            kind,
            geometry: g,
            source,
            entry,
            grid,
            bindings,
        };
        let launch = kernel.prepare(service.client(), grid, handles)?;
        Ok(Self {
            arithmetic,
            launch: Some(launch),
        })
    }
    fn launch(&self, client: &Client) -> Result<()> {
        self.launch
            .as_ref()
            .ok_or_else(|| Error::State("GDN copy metadata has no prepared launch".into()))?
            .launch_on(client)
    }
}

#[cfg(test)]
mod tests;

#[cfg(target_os = "linux")]
impl GdnPrepared {
    /// Fresh source/current-record seal only; identical checked kernels,
    /// handles, copies and arithmetic instance keep their original owners.
    pub(crate) fn bind_text_outputs(
        &self,
        code: Arc<super::text::TextGdnCode>,
    ) -> Result<Arc<Self>> {
        let ComponentCode::Text(old) = &self.code else {
            return Err(Error::Unsupported(
                "text code cannot retag a legacy actor".into(),
            ));
        };
        if Arc::ptr_eq(old.scope(), code.scope())
            || Arc::ptr_eq(old.values(), code.values())
            || old.geometry() != code.geometry()
            || old.layer() != code.layer()
        {
            return Err(Error::State("text actual fresh code/geometry/layer".into()));
        }
        let cache = self.bound_lease(GdnBuffer::CacheArena)?.clone();
        code.require_inputs(&self.service, &code.inputs(cache), &code.outputs())?;
        for (role, actual) in self.buffers.iter() {
            let expected = if *role == GdnBuffer::Rstd {
                actual.clone()
            } else if let Some(index) = [
                GdnBuffer::PackedQkvz,
                GdnBuffer::PackedBa,
                GdnBuffer::ConvOutput,
                GdnBuffer::ZCompact,
                GdnBuffer::Gated,
            ]
            .iter()
            .position(|r| r == role)
            {
                code.operands()[index].clone()
            } else if let Some(index) = [
                GdnBuffer::ConvWeights,
                GdnBuffer::ALog,
                GdnBuffer::DtBias,
                GdnBuffer::GatedGamma,
            ]
            .iter()
            .position(|r| r == role)
            {
                code.parameters().text_leases()[index].clone()
            } else {
                code.workspace().lease(*role)?
            };
            crate::resident_program::natural_binding::same_lease(actual, &expected)?;
        }
        // The finite selector depends on the checked immutable geometry alone.
        // Retain the exact cold sources/specs; fresh operands are checked above.
        code.require_same_recipe(old)?;
        code.require_arithmetic(&self.arithmetic)?;
        Ok(Arc::new(Self {
            arithmetic: self.arithmetic.clone(),
            stages: self.stages.clone(),
            packing: self.packing.clone(),
            inverse_zero: self.inverse_zero.clone(),
            initial: self.initial.clone(),
            scatter: self.scatter.clone(),
            buffers: self.buffers.clone(),
            service: self.service.clone(),
            phase: self.phase,
            slot: self.slot,
            typed_outputs: self.typed_outputs,
            code: ComponentCode::Text(code),
        }))
    }
    pub(crate) async fn prepare_text_outputs(
        code: Arc<super::text::TextGdnCode>,
        service: Arc<ResidentService>,
        inputs: GdnInputs,
        outputs: GdnTypedOutputs,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        code.require_inputs(&service, &inputs, &outputs)?;
        let geometry = code.geometry();
        let result = Self::prepare_impl(
            service,
            geometry.phase,
            geometry.slot,
            inputs,
            Some(outputs),
            ComponentCode::Text(code.clone()),
            cancelled,
        )
        .await?;
        code.require_arithmetic(result.arithmetic())?;
        Ok(result)
    }
}

impl GdnPrepared {
    pub(crate) fn storage_accesses_partition(
        &self,
        recurrent: bool,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let partition = if recurrent {
            StagePartition::Recurrent
        } else {
            StagePartition::Convolution
        };
        let mut roles = BTreeMap::<GdnBuffer, DeviceAccess>::new();
        let mut insert = |role, access| {
            let old = roles.entry(role).or_insert(access);
            if access == DeviceAccess::Write {
                *old = access;
            }
        };
        let copies: Vec<&Copy> = if recurrent {
            self.inverse_zero
                .iter()
                .chain(self.initial.iter())
                .chain(self.scatter.iter())
                .collect()
        } else {
            self.packing.iter().collect()
        };
        for copy in copies {
            let geometry = copy.arithmetic.geometry;
            if let Some(source) = geometry.source {
                insert(source, DeviceAccess::Read);
            }
            insert(geometry.destination, DeviceAccess::Write);
        }
        for stage in self.stages.iter().filter(|s| partition.contains(s.stage)) {
            for binding in &stage.spec.bindings {
                insert(
                    binding.role,
                    if binding.io == BufferIOAttr::ReadOnly {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    },
                );
            }
        }
        roles
            .into_iter()
            .map(|(role, access)| Ok((self.bound_lease(role)?.clone(), access)))
            .collect()
    }
}

#[cfg(all(feature = "cuda-native-qualification", target_os = "linux"))]
pub(crate) fn prepare_state_code(queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
    state_adapter::prepare_code(queue)
}
