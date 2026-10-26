//! Actual B4 instruction stages over current per-row source/root bindings.
//! Source definitions and scratch are shared across layers; the body/cursor
//! owns order and one physical flight. No copied reference tensor is an input.
use super::super::*;
use super::{
    gdn_assets, gdn_assets_2b,
    gdn_packing::{CohortGdnPacking, Partition},
    gdn_state::CohortGdnState,
    values::CohortValues,
};
use crate::cuda::kernel::{PinnedKernel, PreparedPinnedLaunch};
use crate::cuda::native::model_gdn::ModelGdnParameters;
use crate::cuda::native::text_native::TextKernelCapture;
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, text::TextCallInput};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use cubecl_runtime::kernel::BufferIOAttr;
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Role {
    Packed,
    Pitched,
    Conv,
    Z,
    Gated,
    ConvWeight,
    ALog,
    DtBias,
    Gamma,
    CacheConv,
    CacheH,
    StateIndices,
    HasInitial,
    CuSeq,
    ConvBatch,
    ConvOffset,
    A,
    B,
    Q,
    K,
    V,
    G,
    Beta,
    GCum,
    MatrixA,
    MatrixInv,
    W,
    U,
    VNew,
    History,
    H0,
    Ht,
    RawO,
    NormInput,
    Rstd,
    ChunkIndices,
    ChunkOffsets,
}
impl Role {
    fn bytes(self, initial: bool) -> usize {
        let m = if initial { 188 } else { 4 };
        match self {
            Self::Packed => m * 8192 * 2,
            Self::Pitched => 4 * 8192 * 2,
            Self::Conv => m * 6144 * 2,
            Self::Z
            | Self::Gated
            | Self::Q
            | Self::K
            | Self::V
            | Self::W
            | Self::U
            | Self::VNew
            | Self::RawO
            | Self::NormInput => m * 2048 * 2,
            Self::A | Self::B => m * 16 * 2,
            Self::G | Self::Beta | Self::GCum | Self::Rstd => m * 16 * 4,
            Self::MatrixA => m * 16 * 64 * 4,
            Self::MatrixInv => m * 16 * 64 * 2,
            Self::History => 4 * 16 * 128 * 128 * 2,
            Self::H0 | Self::Ht => 4 * 16 * 128 * 128 * 4,
            Self::CacheConv => 17 * 1_114_112,
            Self::CacheH => 17 * 1_114_112 - 3 * 6144 * 2,
            Self::ConvWeight => 6144 * 4 * 2,
            Self::ALog => 16 * 4,
            Self::DtBias => 16 * 2,
            Self::Gamma => 128 * 2,
            Self::StateIndices => 4 * 4,
            Self::HasInitial => 4,
            Self::CuSeq => 5 * 4,
            Self::ConvBatch | Self::ConvOffset => 2048 * 4,
            Self::ChunkIndices => 4 * 2 * 4,
            Self::ChunkOffsets => 5 * 8,
        }
    }
    fn external(self) -> bool {
        matches!(
            self,
            Self::Packed
                | Self::Conv
                | Self::Z
                | Self::Gated
                | Self::ConvWeight
                | Self::ALog
                | Self::DtBias
                | Self::Gamma
                | Self::CacheConv
                | Self::CacheH
        )
    }
}
pub(super) struct StageSource {
    pub(super) name: &'static str,
    pub(super) source: &'static str,
    pub(super) entry: &'static str,
    pub(super) threads: u32,
    pub(super) grid: [u32; 3],
    pub(super) bindings: &'static [(Role, BufferIOAttr)],
}
struct Definition {
    stage: &'static StageSource,
    kernel: PinnedKernel,
}
pub(super) struct CohortGdnLibrary {
    source: VerifiedTensorSource,
    initial: bool,
    capture: TextKernelCapture,
    definitions: Box<[Definition]>,
    _metadata: WindowWorkspace,
}
impl CohortGdnLibrary {
    pub(super) fn prepare_code(&self, queue: &cubecl::cuda::OwnedCudaQueue) -> Result<usize> {
        for definition in &self.definitions {
            definition.kernel.prepare_code(queue)?;
        }
        Ok(self.definitions.len())
    }
    fn stages(capture: TextKernelCapture, initial: bool) -> Result<&'static [StageSource]> {
        match (capture.recipe_sha256(), initial) {
            ("39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d", true) => {
                Ok(gdn_assets::PREFILL)
            }
            ("39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d", false) => {
                Ok(gdn_assets::DECODE)
            }
            ("0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0", false) => {
                Ok(gdn_assets_2b::DECODE)
            }
            ("0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0", true) => Err(
                Error::Unsupported("INITIAL B4 GDN is outside this captured catalogue".into()),
            ),
            _ => Err(Error::Unsupported(
                "B4 GDN unknown exact native capture".into(),
            )),
        }
    }
    pub(super) fn host_bytes(capture: TextKernelCapture, initial: bool) -> Result<usize> {
        let stages = Self::stages(capture, initial)?;
        let bound = stages.iter().try_fold(128 * 1024usize, |sum, stage| {
            sum.checked_add(stage.source.len() * 4 + 4096 + PinnedKernel::KEY_METADATA_BYTES)
                .ok_or_else(|| Error::Limit("B4 GDN actual retained instruction metadata".into()))
        })?;
        Ok(bound)
    }
    pub(super) fn prepare(
        source: VerifiedTensorSource,
        capture: TextKernelCapture,
        initial: bool,
    ) -> Result<Arc<Self>> {
        if !crate::cuda::native::cohort_native::CohortPhases::for_capture(capture).contains(initial)
        {
            return Err(Error::Unsupported(
                "GDN phase outside this full source profile".into(),
            ));
        }
        let stages = Self::stages(capture, initial)?;
        let bound = Self::host_bytes(capture, initial)?;
        let metadata = source.reserve_workspace(bound)?;
        let definitions = stages
            .iter()
            .map(|stage| {
                let bindings = stage
                    .bindings
                    .iter()
                    .map(|&(role, access)| (role.bytes(initial), access))
                    .collect::<Vec<_>>();
                Ok(Definition {
                    stage,
                    kernel: PinnedKernel::new(
                        stage.entry,
                        Arc::from(stage.source),
                        stage.threads,
                        bindings.into(),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Arc::new(Self {
            source,
            initial,
            capture,
            definitions: definitions.into_boxed_slice(),
            _metadata: metadata,
        }))
    }
}

/// Only private intermediates/closed controls. Learned weights remain the
/// existing immutable source owners; one admitted cache is shared with KV.
pub(super) struct CohortGdnStorage {
    library: Arc<CohortGdnLibrary>,
    service: Arc<ResidentService>,
    cache: ResidentLease,
    status: ResidentLease,
    leases: BTreeMap<Role, ResidentLease>,
    indices: [ResidentLease; 3],
    _metadata: Arc<WindowWorkspace>,
}
fn i32_bytes(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
impl CohortGdnStorage {
    pub(super) fn allocation_layout(initial: bool) -> Vec<usize> {
        let stages = if initial {
            gdn_assets::PREFILL
        } else {
            gdn_assets::DECODE
        };
        let mut roles = stages
            .iter()
            .flat_map(|stage| stage.bindings.iter().map(|(role, _)| *role))
            .filter(|role| !role.external())
            .collect::<BTreeSet<_>>();
        roles.extend([Role::A, Role::B, Role::RawO, Role::NormInput]);
        if !initial {
            roles.insert(Role::Pitched);
        }
        roles.remove(&Role::StateIndices);
        roles
            .into_iter()
            .map(|role| role.bytes(initial))
            .chain([16; 3])
            .collect()
    }
    pub(super) fn allocate(
        library: Arc<CohortGdnLibrary>,
        service: Arc<ResidentService>,
        cache: ResidentLease,
        status: ResidentLease,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        if library.source.source_identity() != service.source_identity()
            || cache.bytes() != Role::CacheConv.bytes(library.initial)
            || cache.role() != BufferRole::RecurrentState
            || !cache.is_initialized()
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || !status.is_initialized()
            || [&cache, &status]
                .into_iter()
                .any(|lease| !Arc::ptr_eq(lease.service(), &service))
            || BufferRange::new(&cache.handle())?.overlaps(&BufferRange::new(&status.handle())?)
        {
            return Err(Error::State(
                "B4 GDN admitted native cache/status/source".into(),
            ));
        }
        let metadata = Arc::new(library.source.reserve_workspace(128 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(library.clone())?;
        let mut required: BTreeSet<Role> = library
            .definitions
            .iter()
            .flat_map(|d| d.stage.bindings.iter().map(|(role, _)| *role))
            .filter(|role| !role.external())
            .collect();
        required.extend([Role::A, Role::B, Role::RawO, Role::NormInput]);
        if !library.initial {
            required.insert(Role::Pitched);
        }
        // Actual original cache-group block selection depends on GDN layer
        // residue. Preserve all three witnessed request-to-slot controls.
        required.remove(&Role::StateIndices);
        let indices = (0..3)
            .map(|residue| {
                init.upload_bytes(
                    &i32_bytes((0..4).map(|row| 1 + residue + 4 * row)),
                    BufferRole::ImmutableWeights,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("B4 exact three GDN group controls".into()))?;
        let controls = if library.initial {
            // Pinned original compute_causal_conv1d_metadata uses range(num),
            // not token offsets. The kernel multiplies each index by BLOCK_M8.
            let mut batch = vec![-1i32; 2048];
            let mut offsets = vec![-1i32; 2048];
            for row in 0..4 {
                for chunk in 0..6 {
                    batch[row * 6 + chunk] = row as i32;
                    offsets[row * 6 + chunk] = chunk as i32;
                }
            }
            BTreeMap::from([
                (Role::HasInitial, vec![0; 4]),
                (Role::CuSeq, i32_bytes([0, 47, 94, 141, 188])),
                (Role::ConvBatch, i32_bytes(batch)),
                (Role::ConvOffset, i32_bytes(offsets)),
                (Role::ChunkIndices, i32_bytes([0, 0, 1, 0, 2, 0, 3, 0])),
                (
                    Role::ChunkOffsets,
                    (0i64..5).flat_map(i64::to_le_bytes).collect(),
                ),
            ])
        } else {
            BTreeMap::new()
        };
        if controls.keys().any(|role| !required.contains(role)) {
            return Err(Error::State("B4 GDN closed control role population".into()));
        }
        let mut leases = BTreeMap::new();
        for role in required {
            let bytes = role.bytes(library.initial);
            let lease = if let Some(control) = controls.get(&role) {
                if control.len() != bytes {
                    return Err(Error::State("B4 source control extent".into()));
                }
                init.upload_bytes(control, BufferRole::ImmutableWeights)?
            } else {
                init.allocate_empty(
                    bytes,
                    if matches!(role, Role::H0 | Role::Ht) {
                        BufferRole::RecurrentState
                    } else {
                        BufferRole::Scratch
                    },
                )?
            };
            leases.insert(role, lease);
        }
        let result = Arc::new(Self {
            library,
            service,
            cache,
            status,
            leases,
            indices,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn lease(&self, role: Role) -> Result<&ResidentLease> {
        if role == Role::CacheConv {
            return Ok(&self.cache);
        }
        self.leases
            .get(&role)
            .ok_or_else(|| Error::State("B4 GDN private role absent".into()))
    }
    pub(super) fn prefill_h(&self) -> Result<Option<(ResidentLease, ResidentLease)>> {
        if self.library.initial {
            Ok(Some((
                self.lease(Role::H0)?.clone(),
                self.lease(Role::Ht)?.clone(),
            )))
        } else {
            Ok(None)
        }
    }
}

struct Stage {
    source: &'static StageSource,
    launch: PreparedPinnedLaunch,
}
pub(super) struct CohortGdnCode {
    inputs: Box<[Arc<TextCallInput>]>,
    values: Arc<CohortValues>,
    storage: Arc<CohortGdnStorage>,
    packing: Arc<CohortGdnPacking>,
    transport: Arc<CohortGdnState>,
    _parameters: Arc<ModelGdnParameters>,
    nodes: [usize; 2],
    stages: Box<[Stage]>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    // Explicit two-node census split; the binder union never gets assigned to
    // one guessed producer node.
    conv_accesses: Vec<(ResidentLease, DeviceAccess)>,
    recurrent_accesses: Vec<(ResidentLease, DeviceAccess)>,
    _metadata: WindowWorkspace,
}
impl CohortGdnCode {
    pub(super) fn bind(
        inputs: &[Arc<TextCallInput>],
        values: Arc<CohortValues>,
        layer: usize,
        storage: Arc<CohortGdnStorage>,
        packing: Arc<CohortGdnPacking>,
        transport: Arc<CohortGdnState>,
        parameters: Arc<ModelGdnParameters>,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        crate::cuda::native::cohort_native::require_input_shape(inputs)?;
        if inputs.len() != 4 || !Arc::ptr_eq(values.service(), &storage.service) {
            return Err(Error::State("B4 GDN exact all-row/source/service".into()));
        }
        let initial = (inputs[0].rows(), inputs[0].base()) == (47, 0);
        if initial != storage.library.initial
            || !inputs[0].source().same_issuer(&storage.library.source)
        {
            return Err(Error::State("B4 GDN current phase/source library".into()));
        }
        let site = inputs[0].scope().gdn_site(layer)?;
        for input in inputs {
            let actual = input.scope().gdn_site(layer)?;
            if input.scope().native_capture() != storage.library.capture
                || input.prompt_tokens() != 47
                || !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.rows() != inputs[0].rows()
                || actual.nodes != site.nodes
                || actual.operands != site.operands
            {
                return Err(Error::State(
                    "B4 GDN every actual row source layer/frontier".into(),
                ));
            }
            parameters.require_text(input.scope(), layer)?;
        }
        packing.require_storage(
            &values,
            inputs,
            storage.lease(Role::A)?,
            storage.lease(Role::B)?,
            if initial {
                None
            } else {
                Some(storage.lease(Role::Pitched)?)
            },
        )?;
        transport.require_storage(
            inputs,
            &storage.cache,
            &storage.status,
            storage.prefill_h()?.as_ref(),
        )?;
        let mut leases = storage.leases.clone();
        leases.insert(Role::StateIndices, storage.indices[layer % 4].clone());
        leases.insert(Role::CacheConv, storage.cache.clone());
        for (role, id) in [Role::Packed, Role::Conv, Role::Z, Role::Gated]
            .into_iter()
            .zip([
                site.operands[0],
                site.operands[2],
                site.operands[3],
                site.operands[4],
            ])
        {
            leases.insert(role, values.lease(id)?.clone());
        }
        for (role, lease) in [Role::ConvWeight, Role::ALog, Role::DtBias, Role::Gamma]
            .into_iter()
            .zip(parameters.text_leases())
        {
            leases.insert(role, lease.clone());
        }
        // Private scratch/controls cannot overlap any original value or learned
        // source. The indexed H alias is an explicit view of CacheConv below.
        let actual = leases.iter().collect::<Vec<_>>();
        for (index, (_, lease)) in actual.iter().enumerate() {
            let range = BufferRange::new(&lease.handle())?;
            if !Arc::ptr_eq(lease.service(), values.service())
                || actual[..index].iter().any(|(_, other)| {
                    BufferRange::new(&other.handle()).is_ok_and(|r| range.overlaps(&r))
                })
                || range.overlaps(&BufferRange::new(&storage.status.handle())?)
            {
                return Err(Error::State(
                    "B4 GDN physical role/storage disjointness".into(),
                ));
            }
        }
        for lease in storage
            .leases
            .values()
            .chain([&storage.cache, &storage.status])
        {
            let range = BufferRange::new(&lease.handle())?;
            for value in values.buffers() {
                if range.overlaps(&BufferRange::new(&value.handle)?) {
                    return Err(Error::State(
                        "B4 GDN scratch aliases any original SSA value".into(),
                    ));
                }
            }
        }
        let mut accesses = Vec::new();
        let mut conv_accesses = Vec::new();
        let mut recurrent_accesses = Vec::new();
        let stages = storage
            .library
            .definitions
            .iter()
            .enumerate()
            .map(|(stage_index, definition)| {
                let handles = definition
                    .stage
                    .bindings
                    .iter()
                    .map(|&(role, access)| {
                        let (lease, offset) = if role == Role::CacheH {
                            (&storage.cache, 3 * 6144 * 2)
                        } else {
                            (
                                leases.get(&role).ok_or_else(|| {
                                    Error::State("B4 source role not bound".into())
                                })?,
                                0,
                            )
                        };
                        if lease.bytes() != role.bytes(initial) + offset {
                            return Err(Error::State("B4 exact source pointer footprint".into()));
                        }
                        let census = if access == BufferIOAttr::ReadOnly {
                            (lease.clone(), DeviceAccess::Read)
                        } else {
                            (lease.clone(), DeviceAccess::Write)
                        };
                        accesses.push(census.clone());
                        // Stage 0 belongs to the Conv node; the remaining
                        // stages run inside the Recurrent node.
                        if stage_index == 0 {
                            conv_accesses.push(census);
                        } else {
                            recurrent_accesses.push(census);
                        }
                        Ok(lease.handle().offset_start(offset as u64))
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Stage {
                    source: definition.stage,
                    launch: definition.kernel.prepare(
                        values.service().client(),
                        definition.stage.grid,
                        handles,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        accesses.extend([
            (storage.lease(Role::RawO)?.clone(), DeviceAccess::Read),
            (storage.lease(Role::NormInput)?.clone(), DeviceAccess::Write),
            (storage.status.clone(), DeviceAccess::Write),
        ]);
        // The exact prefill/cached layer-norm D2D witness runs inside the
        // Recurrent node, right before its layer_norm stage.
        recurrent_accesses.extend([
            (storage.lease(Role::RawO)?.clone(), DeviceAccess::Read),
            (storage.lease(Role::NormInput)?.clone(), DeviceAccess::Write),
            (storage.status.clone(), DeviceAccess::Write),
        ]);
        conv_accesses.push((storage.status.clone(), DeviceAccess::Write));
        accesses.extend(transport.accesses());
        for part in [
            Partition::BeforeConvolution,
            Partition::AfterConvolution,
            Partition::BeforeRecurrent,
        ] {
            accesses.extend(packing.accesses(part));
        }
        // Conv node: transport kinds 0/1 + Before/AfterConvolution packing.
        conv_accesses.extend(transport.partition_accesses(0));
        conv_accesses.extend(transport.partition_accesses(1));
        conv_accesses.extend(packing.accesses(Partition::BeforeConvolution));
        conv_accesses.extend(packing.accesses(Partition::AfterConvolution));
        // Recurrent node: transport kinds 2/3 + BeforeRecurrent packing.
        recurrent_accesses.extend(transport.partition_accesses(2));
        recurrent_accesses.extend(transport.partition_accesses(3));
        recurrent_accesses.extend(packing.accesses(Partition::BeforeRecurrent));
        let metadata = inputs[0].source().reserve_workspace(64 * 1024)?;
        let nodes = site.nodes;
        Ok(Arc::new(Self {
            inputs: inputs.to_vec().into_boxed_slice(),
            values,
            storage,
            packing,
            transport,
            _parameters: parameters,
            nodes,
            stages: stages.into_boxed_slice(),
            accesses,
            conv_accesses,
            recurrent_accesses,
            _metadata: metadata,
        }))
    }
    /// Original node cursor calls once for Conv and once for Recurrent. Every
    /// source/phase/state authority is checked for all rows before commands.
    /// Returns (native launches, copies, grouped transport launches).
    pub(super) fn enqueue(
        &self,
        node: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<(usize, usize, usize)> {
        if permits.len() != 4 || effects.len() != 4 || !self.nodes.contains(&node) {
            return Err(Error::State(
                "B4 GDN original node/permit/effect population".into(),
            ));
        }
        self.values.require_inputs(&self.inputs)?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, self.values.service())?;
        }
        let mut native = 0;
        let mut copies = 0;
        let mut grouped = 0;
        if node == self.nodes[0] {
            // State transport itself validates all actual root effects before
            // its first copy; no sibling row has placeholder authority.
            self.transport.require_partition(0, permits, effects)?;
            copies += self
                .packing
                .enqueue(node, Partition::BeforeConvolution, permits)?;
            let (launches, transports) = self.transport.enqueue(0, permits, effects)?;
            copies += launches;
            grouped += transports;
            self.stages[0]
                .launch
                .launch_on(self.values.service().client())?;
            native += 1;
            copies += self
                .packing
                .enqueue(node, Partition::AfterConvolution, permits)?;
            let (launches, transports) = self.transport.enqueue(1, permits, effects)?;
            copies += launches;
            grouped += transports;
        } else {
            self.transport.require_partition(2, permits, effects)?;
            copies += self
                .packing
                .enqueue(node, Partition::BeforeRecurrent, permits)?;
            let (launches, transports) = self.transport.enqueue(2, permits, effects)?;
            copies += launches;
            grouped += transports;
            for stage in &self.stages[1..] {
                if stage.source.name == "layer_norm_fwd_kernel" {
                    let from = self.storage.lease(Role::RawO)?;
                    let to = self.storage.lease(Role::NormInput)?;
                    let words = from.bytes() / 4;
                    // Exact original18 prefill D2D witnesses; cached norm is
                    // the same source producer. This private copy preserves
                    // all BF16 bits without arithmetic or producer mutation.
                    unsafe {
                        crate::program_kernels::copy_strided_words::launch_unchecked(
                            self.values.service().client(),
                            grid(words),
                            threads(),
                            arg(&from.handle(), words),
                            arg(&to.handle(), words),
                            arg(&self.storage.status.handle(), 1),
                            0,
                            0,
                            words as u32,
                            words as u32,
                            1,
                            words,
                            false,
                        );
                    }
                    copies += 1;
                }
                stage.launch.launch_on(self.values.service().client())?;
                native += 1;
            }
            let (launches, transports) = self.transport.enqueue(3, permits, effects)?;
            copies += launches;
            grouped += transports;
        }
        Ok((native, copies, grouped))
    }
    pub(super) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    /// Explicit two-node census split: [(conv node, its accesses), (recurrent
    /// node, its accesses)]. Ambiguous binder-union leases stay attributed to
    /// their actual stage/partition side, never to a guessed producer.
    pub(super) fn node_accesses(&self) -> [(usize, Vec<(ResidentLease, DeviceAccess)>); 2] {
        [
            (self.nodes[0], self.conv_accesses.clone()),
            (self.nodes[1], self.recurrent_accesses.clone()),
        ]
    }
}

#[cfg(test)]
mod capture_tests {
    use super::*;
    #[test]
    fn exact_captured_stage_sets_do_not_grant_wide_initial() -> Result<()> {
        let narrow = TextKernelCapture::bind(
            "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d",
        )?;
        let wide = TextKernelCapture::bind(
            "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0",
        )?;
        assert!(std::ptr::eq(
            CohortGdnLibrary::stages(narrow, true)?,
            gdn_assets::PREFILL
        ));
        assert!(std::ptr::eq(
            CohortGdnLibrary::stages(narrow, false)?,
            gdn_assets::DECODE
        ));
        assert!(std::ptr::eq(
            CohortGdnLibrary::stages(wide, false)?,
            gdn_assets_2b::DECODE
        ));
        assert!(CohortGdnLibrary::stages(wide, true).is_err());
        assert!(CohortGdnLibrary::host_bytes(wide, true).is_err());
        assert!(TextKernelCapture::bind("unknown").is_err());
        Ok(())
    }
}
