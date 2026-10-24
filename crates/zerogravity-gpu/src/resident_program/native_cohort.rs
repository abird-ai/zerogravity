//! Genuine per-row authorities over one physical B4 execution in launch_impl.
mod attention;
mod bindings;
mod embedding;
mod gdn_assets;
mod gdn_assets_2b;
mod gdn_code;
mod gdn_packing;
mod gdn_state;
mod head_input;
mod kv;
pub(crate) const CACHE_BYTES: usize = kv::CACHE_BYTES;
mod native_nodes;
pub(crate) mod values;
use self::{
    attention::CohortAttention,
    embedding::CohortEmbedding,
    gdn_code::{CohortGdnCode, CohortGdnLibrary, CohortGdnStorage},
    gdn_packing::CohortGdnPacking,
    gdn_state::CohortGdnState,
    head_input::CohortHeadInput,
    native_nodes::CohortNativeNodes,
};
use self::{bindings::CohortBindings, values::CohortValues};
use super::bf16_greedy::cohort::PreparedCohortGreedy;
use super::*;
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, short_hf::ShortHfCallState};
use crate::resident_leases::backing_pool::ResidentWorkspaceGeometry;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService, TiedWeight};
use crate::vllm_cuda_input_rms::cohort::{CohortGamma, CohortRmsNode, CohortRmsStorage};
use crate::vllm_cuda_qualification::cohort_native::CohortPhases;
use crate::vllm_cuda_qualification::{
    cohort_native::CohortNativeLibrary, model_gdn::ModelGdnParameters, model_rotary::ModelRotary,
    short_hf_native::ShortHfNativeStorage,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::FullModelProfile;
use zerogravity_core::device_events::{BufferLeaseId, BufferRole, DeviceAccess};

/// Explicit diagnostic cuts after actual submitted B4 source kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortHfCohortFault {
    UnknownAfterFirstConvolution,
    UnknownAfterLastFa,
    UnknownAfterHead,
}
/// Root-free immutable instruction definitions on the original model context.
/// No runtime row, StateRoot, scratch epoch or currently bound program is kept.
pub(crate) struct CohortLibrary {
    native: Arc<CohortNativeLibrary>,
    gdn: [Option<Arc<CohortGdnLibrary>>; 2],
}
impl CohortLibrary {
    pub(crate) fn prepare_code(
        &self,
        source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
        queue: &cubecl::cuda::OwnedCudaQueue,
    ) -> Result<usize> {
        let mut count = 0;
        for gdn in self.gdn.iter().flatten() {
            count += gdn.prepare_code(queue)?;
        }
        for incoming in [true, false] {
            kv::definition(incoming)?.prepare_code(queue)?;
            count += 1;
        }
        for (kernel, _) in gdn_state::definitions()? {
            kernel.prepare_code(queue)?;
            count += 1;
        }
        count +=
            crate::vllm_cuda_input_rms::cohort::prepare_code(source, self.native.profile(), queue)?;
        let profile = self.native.profile();
        for initial in CohortPhases::for_profile(profile).iter() {
            let rows = if initial { 188 } else { 4 };
            for width in [
                32,
                512,
                profile.hidden_size(),
                2048,
                profile.intermediate_size(),
                5120,
                6144,
                2 * profile.intermediate_size(),
                8192,
            ] {
                super::native::finite_definition(rows * width)?.prepare_code(queue)?;
                count += 1;
            }
        }
        super::native::finite_definition(4 * 248320)?.prepare_code(queue)?;
        Ok(count + 1)
    }
    /// # Safety
    /// Original close barrier and armed retirement flight retain the complete
    /// cold cohort; no request can prepare/enqueue. Partial failure quarantines.
    pub(crate) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        // SAFETY: caller supplies the original physical completion authority.
        unsafe { self.native.retire_after_physical_completion() }
    }
    pub(crate) fn host_bytes(profile: FullModelProfile) -> Result<usize> {
        let initial = if CohortPhases::for_profile(profile).contains(true) {
            CohortGdnLibrary::host_bytes(profile, true)?
        } else {
            0
        };
        initial
            .checked_add(CohortGdnLibrary::host_bytes(profile, false)?)
            .and_then(|n| n.checked_add(CohortNativeLibrary::host_bytes(profile)))
            .ok_or_else(|| Error::Limit("B4 immutable native/GDN metadata".into()))
    }
    pub(crate) fn prepare(
        storage: Arc<ShortHfNativeStorage>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let source = storage.weights().source().clone();
        let phases = CohortPhases::for_profile(storage.profile());
        let gdn = [
            if phases.contains(true) {
                Some(CohortGdnLibrary::prepare(source.clone(), true)?)
            } else {
                None
            },
            Some(CohortGdnLibrary::prepare(source, false)?),
        ];
        let native = CohortNativeLibrary::prepare(storage, service, init, cancel)?;
        let result = Arc::new(Self { native, gdn });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
}
/// Call-local: contains actual prior/candidate owners and must drop before the
/// next candidates. Immutable weights, gammas and native images remain shared.
pub(crate) struct FullModelCohort {
    bindings: CohortBindings,
    values: Arc<CohortValues>,
    status: ResidentLease,
    owned: OwnedRanges,
    embedding: Arc<CohortEmbedding>,
    rms: BTreeMap<usize, Arc<CohortRmsNode>>,
    sums: BTreeMap<usize, ResidentLease>,
    gdn: BTreeMap<usize, Arc<CohortGdnCode>>,
    native: Arc<CohortNativeNodes>,
    head: Arc<CohortHeadInput>,
    greedy: Arc<PreparedCohortGreedy>,
    certificates: native::CertificateSet,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    begun: AtomicBool,
    next_node: AtomicUsize,
    finished_storage: AtomicUsize,
    output: AtomicBool,
    native_launches: AtomicUsize,
    projection_launches: AtomicUsize,
    sampler_dispatches: AtomicUsize,
    device_copies: AtomicUsize,
    clears: AtomicUsize,
    fault: Option<ShortHfCohortFault>,
    _metadata: WindowWorkspace,
}
impl FullModelCohort {
    /// Complete call-local metadata reservations. Both phases retain their own
    /// original row records; only one body executes under model.workspace.
    pub(crate) fn host_bytes(initial: bool, profile: FullModelProfile) -> Result<usize> {
        let rms = CohortRmsNode::metadata_bound(initial, false, profile)?
            .checked_add(
                CohortRmsNode::metadata_bound(initial, true, profile)?
                    .checked_mul(48)
                    .ok_or_else(|| Error::Limit("B4 RMS metadata population".into()))?,
            )
            .ok_or_else(|| Error::Limit("B4 RMS metadata sum".into()))?;
        [
            8 * 1024 * 1024,             // body/roles/certificates/native sites
            128 * 1024,                  // actual327 value views
            2 * 128 * 1024,              // GDN and RMS private-storage wrappers
            2 * 64 * 1024,               // source embedding and full[B,V] reducer
            18 * (512 + 64 + 32) * 1024, // GDN state/code/packing
            6 * (512 + 512) * 1024,      // attention controls and canonical KV transport
            rms,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| {
            sum.checked_add(bytes)
                .ok_or_else(|| Error::Limit("B4 complete body host metadata".into()))
        })
    }
    /// Source-derived private allocation population for the only active body.
    /// Existing weights/rotary/native scratch and canonical roots priced elsewhere.
    pub(crate) fn allocation_layout(initial: bool, profile: FullModelProfile) -> Vec<usize> {
        let m = if initial { 188 } else { 4 };
        let mut result = CohortGdnStorage::allocation_layout(initial);
        result.extend(CohortRmsStorage::allocation_layout(initial, profile));
        result.extend([m * profile.hidden_size() * 4; 24]); // optional full source dual-sum diagnostics
        result.extend([m * 4, 4 * profile.hidden_size() * 2]); // issued IDs and final-row gather
        // Four immutable row packets +packed packet +actual 256-way partials.
        result.extend(
            PreparedCohortGreedy::allocation_layout(4).expect("closed four-row allocation layout"),
        );
        for _ in 0..6 {
            result.extend([3 * 513 * 8, m * 8, 20, 16, 64, 4, 4]);
            if initial {
                result.push(20);
            }
            result.push(if initial { 4 * 8 * 47 * 4 } else { 128 });
            if !initial {
                result.extend([16384, 16384, 128]);
            }
        }
        result
    }
    /// All physical values/root candidates and the private arena/status are
    /// admitted before this factory. No captured state or N1 Ready is accepted.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        service: Arc<ResidentService>,
        states: &[Arc<ShortHfCallState>],
        permits: &[PrefixPhaseEnqueue<'_>],
        geometry: Arc<ResidentWorkspaceGeometry>,
        status: ResidentLease,
        cache: ResidentLease,
        native_storage: Arc<ShortHfNativeStorage>,
        library: Arc<CohortLibrary>,
        rotary: Arc<ModelRotary>,
        parameters: &BTreeMap<usize, Arc<ModelGdnParameters>>,
        gammas: BTreeMap<usize, CohortGamma>,
        diagnostic: bool,
        fault: Option<ShortHfCohortFault>,
        tied: Arc<TiedWeight>,
        init: &mut ResidentInitialization<'_>,
        cancel: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        native::cancelled(cancel)?;
        init.require_service(&service)?;
        if states.len() != 4
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || !status.is_initialized()
            || !Arc::ptr_eq(status.service(), &service)
            || cache.bytes() != kv::CACHE_BYTES
            || !cache.is_initialized()
            || cache.role() != BufferRole::RecurrentState
            || !Arc::ptr_eq(cache.service(), &service)
        {
            return Err(Error::State(
                "B4 whole-body admitted source workspace/cache/status".into(),
            ));
        }
        let mut owned = OwnedRanges::new(service.client().service_id());
        for lease in geometry.leases().iter().chain([&status, &cache]) {
            owned.insert(BufferRange::new(&lease.handle())?)?;
        }
        let bindings = CohortBindings::bind(&service, states, permits, &owned)?;
        let inputs = bindings.inputs();
        let source = bindings.source().clone();
        let metadata = source.source().reserve_workspace(8 * 1024 * 1024)?;
        let values = CohortValues::bind(&bindings, service.clone(), geometry)?;
        let initial = (inputs[0].rows(), inputs[0].base()) == (47, 0);
        let gdn_library = library.gdn[usize::from(!initial)].clone().ok_or_else(|| {
            Error::Unsupported("B4 phase has no source-prepared GDN library".into())
        })?;
        let gdn_storage = CohortGdnStorage::allocate(
            gdn_library,
            service.clone(),
            cache.clone(),
            status.clone(),
            init,
        )?;
        let rms_storage = CohortRmsStorage::allocate(&inputs, service.clone(), init)?;
        let embedding =
            CohortEmbedding::prepare(&inputs, values.clone(), tied.clone(), status.clone(), init)?;
        let head = CohortHeadInput::prepare(&inputs, values.clone(), status.clone(), init)?;
        let greedy = PreparedCohortGreedy::prepare(
            source.clone(),
            &bindings.records(),
            values.lease(inputs[0].record().logits())?.clone(),
            status.clone(),
            init,
            cancel,
        )?;
        let mut rms = BTreeMap::new();
        let mut sums = BTreeMap::new();
        for (node, gamma) in gammas {
            if diagnostic
                && matches!(&inputs[0].record().plan().nodes()[node].operation,
                PlanOperation::TypedVllm08{descriptor,..} if matches!(descriptor.kind(),
                    zerogravity_core::accelerator::program::Vllm08Kind::ResidualRms(
                        zerogravity_core::accelerator::program::Vllm08RmsSite::Input |
                        zerogravity_core::accelerator::program::Vllm08RmsSite::Final)))
            {
                sums.insert(
                    node,
                    init.allocate_empty(
                        if initial { 188 } else { 4 }
                            * inputs[0].call().model_profile().hidden_size()
                            * 4,
                        BufferRole::Scratch,
                    )?,
                );
            }
            rms.insert(
                node,
                CohortRmsNode::bind(&inputs, rms_storage.clone(), node, gamma, &values)?,
            );
        }
        if diagnostic && sums.len() != 24 {
            return Err(Error::State(
                "B4 all24 independently observed dual RMS sums".into(),
            ));
        }
        if rms.len() != 49 || !rms.contains_key(&1) || !rms.contains_key(&253) {
            return Err(Error::State("B4 all49 genuine source RMS nodes".into()));
        }
        let pitched = if initial {
            None
        } else {
            Some(gdn_storage.lease(gdn_code::Role::Pitched)?.clone())
        };
        let mut gdn = BTreeMap::new();
        let mut attention = BTreeMap::new();
        let controls = attention::CohortAttentionControls::prepare(&inputs, init)?;
        for layer in 0..24 {
            native::cancelled(cancel)?;
            let row_states = bindings.states_at(layer)?;
            if layer % 4 == 3 {
                attention.insert(
                    layer,
                    CohortAttention::prepare(
                        &inputs,
                        &row_states,
                        values.clone(),
                        layer,
                        attention::AttentionOwners {
                            native: native_storage.clone(),
                            rotary: rotary.clone(),
                            cache: cache.clone(),
                            status: status.clone(),
                            controls: controls.clone(),
                        },
                        init,
                    )?,
                );
            } else {
                let packing = CohortGdnPacking::bind(
                    values.clone(),
                    &inputs,
                    layer,
                    gdn_storage.lease(gdn_code::Role::A)?.clone(),
                    gdn_storage.lease(gdn_code::Role::B)?.clone(),
                    pitched.clone(),
                    status.clone(),
                )?;
                let transport = CohortGdnState::prepare(
                    service.clone(),
                    &inputs,
                    &row_states,
                    cache.clone(),
                    status.clone(),
                    gdn_storage.prefill_h()?,
                    init,
                )?;
                let code = CohortGdnCode::bind(
                    &inputs,
                    values.clone(),
                    layer,
                    gdn_storage.clone(),
                    packing,
                    transport,
                    parameters.get(&layer).cloned().ok_or_else(|| {
                        Error::State("B4 original learned GDN source owner absent".into())
                    })?,
                )?;
                let site = inputs[0].scope().gdn_site(layer)?;
                for &node in &site.nodes {
                    if gdn.insert(node, code.clone()).is_some() {
                        return Err(Error::State("B4 repeated GDN node".into()));
                    }
                }
            }
        }
        let native = CohortNativeNodes::bind(
            &inputs,
            values.clone(),
            library.native.clone(),
            head.clone(),
            tied,
            attention,
            status.clone(),
        )?;
        // Complete lowering population before the first queued command.
        for index in 0..255 {
            if usize::from(index == 0)
                + usize::from(rms.contains_key(&index))
                + usize::from(gdn.contains_key(&index))
                + usize::from(native.contains(index))
                != 1
            {
                return Err(Error::State(
                    "B4 exact closed255 source node lowerings".into(),
                ));
            }
        }
        let canonical = bindings
            .state_accesses()
            .into_iter()
            .map(|(lease, _)| lease.id())
            .collect::<BTreeSet<_>>();
        let mut access_map: BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)> =
            BTreeMap::new();
        let all = embedding
            .accesses()
            .into_iter()
            .chain(head.accesses())
            .chain(native.accesses())
            .chain(
                rms.values()
                    .flat_map(|node| node.accesses().iter().cloned()),
            )
            .chain(
                gdn.values()
                    .flat_map(|node| node.accesses().iter().cloned()),
            )
            .chain(greedy.accesses().iter().cloned())
            .chain(
                sums.values()
                    .cloned()
                    .map(|lease| (lease, DeviceAccess::Write)),
            )
            .chain(
                values
                    .buffers()
                    .iter()
                    .map(|value| Ok((value.original_lease()?.clone(), DeviceAccess::Write)))
                    .collect::<Result<Vec<_>>>()?,
            );
        for (lease, access) in all {
            if !canonical.contains(&lease.id()) {
                owned.insert(BufferRange::new(&lease.handle())?)?;
            }
            let entry = access_map.entry(lease.id()).or_insert((lease, access));
            if access == DeviceAccess::Write {
                entry.1 = access;
            }
        }
        // Exclude canonical state planes from owned workspace ranges; compare
        // every candidate against all of them in one global proof instead.
        bindings.require_current(&service, permits, &owned)?;
        for (lease, access) in bindings.state_accesses() {
            let entry = access_map.entry(lease.id()).or_insert((lease, access));
            if access == DeviceAccess::Write {
                entry.1 = access;
            }
        }
        // State adapters include canonical planes in their access unions; those
        // were compared above as states, never as permission to alias scratch.
        let certificates = native::prepare_certificates(&service, values.buffers(), &status)?;
        let result = Arc::new(Self {
            bindings,
            values,
            status,
            owned,
            embedding,
            rms,
            sums,
            gdn,
            native,
            head,
            greedy,
            certificates,
            accesses: access_map.into_values().collect(),
            begun: AtomicBool::new(false),
            next_node: AtomicUsize::new(0),
            finished_storage: AtomicUsize::new(0),
            output: AtomicBool::new(false),
            native_launches: AtomicUsize::new(0),
            projection_launches: AtomicUsize::new(0),
            sampler_dispatches: AtomicUsize::new(0),
            device_copies: AtomicUsize::new(0),
            clears: AtomicUsize::new(0),
            fault,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn require_current(&self, permits: &[PrefixPhaseEnqueue<'_>]) -> Result<()> {
        self.bindings
            .require_current(self.values.service(), permits, &self.owned)
    }
    pub(crate) fn begin_launch(
        &self,
        permits: &[PrefixPhaseEnqueue<'_>],
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        native::cancelled(cancel)?;
        self.require_current(permits)?;
        if self.begun.swap(true, Ordering::AcqRel) {
            return Err(Error::State("B4 actual body launched once".into()));
        }
        Ok(())
    }
    pub(crate) fn nodes(&self) -> &[PlanNode] {
        self.bindings
            .record(0)
            .expect("closed B4 population")
            .plan()
            .nodes()
    }
    pub(crate) fn enqueue(
        &self,
        index: usize,
        node: &PlanNode,
        permits: &[PrefixPhaseEnqueue<'_>],
        cancel: Option<&AtomicBool>,
    ) -> Result<usize> {
        native::cancelled(cancel)?;
        self.bindings
            .require_enqueue(self.values.service(), permits)?;
        if !self.begun.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != index
            || self.finished_storage.load(Ordering::Acquire) != index
            || self
                .nodes()
                .get(index)
                .is_none_or(|actual| !std::ptr::eq(actual, node))
        {
            return Err(Error::State("B4 same original loop/storage cursor".into()));
        }
        let mut n = 0;
        let mut copies = 0;
        let mut clears = 0;
        if index == 0 {
            copies += self.embedding.enqueue(index, permits)?;
        } else if let Some(rms) = self.rms.get(&index) {
            n += rms.enqueue(index, permits)?;
        } else if let Some(gdn) = self.gdn.get(&index) {
            let (a, b) = gdn.enqueue(index, permits, &self.bindings.effects(index)?)?;
            n += a;
            copies += b;
        } else if self.native.contains(index) {
            let effects = if node.state.is_some() {
                self.bindings.effects(index)?
            } else {
                Vec::new()
            };
            let (a, b, c) = self.native.enqueue(index, permits, &effects, cancel)?;
            n += a;
            copies += b;
            clears += c;
        } else {
            return Err(Error::Unsupported(
                "B4 actual source node has no mathematical lowering".into(),
            ));
        }
        if let Some(sum) = self.sums.get(&index) {
            let actual = self
                .rms
                .get(&index)
                .ok_or_else(|| Error::State("B4 diagnostic sum original RMS producer".into()))?
                .sum();
            if actual.bytes() != sum.bytes() {
                return Err(Error::State("B4 full F32 sum diagnostic extent".into()));
            }
            let words = sum.bytes() / 4;
            unsafe {
                crate::program_kernels::copy_strided_words::launch_unchecked(
                    self.values.service().client(),
                    grid(words),
                    threads(),
                    arg(&actual.handle(), words),
                    arg(&sum.handle(), words),
                    arg(&self.status.handle(), 1),
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
        if index == 253 {
            copies += self.head.enqueue(index, permits)?;
        }
        if matches!(&node.operation,PlanOperation::TypedVllm08 {descriptor,..} if matches!(descriptor.kind(),zerogravity_core::accelerator::program::Vllm08Kind::Projection(_)))
        {
            self.projection_launches.fetch_add(n, Ordering::AcqRel);
        }
        self.native_launches.fetch_add(n, Ordering::AcqRel);
        self.device_copies.fetch_add(copies, Ordering::AcqRel);
        self.clears.fetch_add(clears, Ordering::AcqRel);
        self.next_node.store(index + 1, Ordering::Release);
        let cut = match self.fault {
            Some(ShortHfCohortFault::UnknownAfterFirstConvolution) => matches!(&node.operation,
                PlanOperation::TypedVllm08 { descriptor, .. } if descriptor.kind() == zerogravity_core::accelerator::program::Vllm08Kind::GdnConv && descriptor.layer().is_some_and(|l| l.index() == 0)),
            Some(ShortHfCohortFault::UnknownAfterLastFa) => matches!(&node.operation,
                PlanOperation::TypedVllm08 { descriptor, .. } if descriptor.kind() == zerogravity_core::accelerator::program::Vllm08Kind::FlashAttention && descriptor.layer().is_some_and(|l| l.index() == 23)),
            Some(ShortHfCohortFault::UnknownAfterHead) => index == 254,
            None => false,
        };
        if cut {
            return Err(Error::State(
                "B4 deliberate submitted unknown completion cut".into(),
            ));
        }
        Ok(n + copies + clears)
    }
    pub(crate) fn certify(&self, value: &DeviceBuffer, status: &Handle) -> Result<usize> {
        native::certify_original(&self.certificates, value, status, &self.status)?;
        Ok(1)
    }
    pub(crate) fn finish_value_storage_node(&self, index: usize) -> Result<()> {
        if self.next_node.load(Ordering::Acquire) != index + 1
            || self
                .finished_storage
                .compare_exchange(index, index + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return Err(Error::State(
                "B4 finish original producer and full certificates once".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn enqueue_output(&self, cancel: Option<&AtomicBool>) -> Result<usize> {
        native::cancelled(cancel)?;
        if self.next_node.load(Ordering::Acquire) != 255
            || self.finished_storage.load(Ordering::Acquire) != 255
            || self.output.swap(true, Ordering::AcqRel)
        {
            return Err(Error::State(
                "B4 output after full actual traversal once".into(),
            ));
        }
        let count = self.greedy.enqueue(cancel)?;
        self.sampler_dispatches.store(count, Ordering::Release);
        Ok(count)
    }
    pub(crate) fn sums(&self) -> &BTreeMap<usize, ResidentLease> {
        &self.sums
    }
    pub(crate) fn fault_counters(&self) -> [usize; 5] {
        [
            self.next_node.load(Ordering::Acquire),
            self.native_launches.load(Ordering::Acquire),
            self.projection_launches.load(Ordering::Acquire),
            self.device_copies.load(Ordering::Acquire),
            self.sampler_dispatches.load(Ordering::Acquire),
        ]
    }
    pub(crate) fn head_input(&self) -> &ResidentLease {
        self.head.lease()
    }
    pub(crate) fn output_readbacks(&self) -> Result<Vec<crate::resident_leases::ResidentReadback>> {
        self.greedy.readbacks()
    }
    pub(crate) fn validate_output_readbacks(
        &self,
        raw: &[cubecl::bytes::Bytes],
    ) -> Result<Vec<(u32, u32)>> {
        self.greedy.validate_readback(raw)
    }
    pub(crate) fn require_finished(&self) -> Result<()> {
        if !self.output.load(Ordering::Acquire)
            || self.next_node.load(Ordering::Acquire) != 255
            || self.finished_storage.load(Ordering::Acquire) != 255
        {
            return Err(Error::State(
                "B4 full source traversal/Head/output incomplete".into(),
            ));
        }
        self.values.service().ensure_usable()
    }
    pub(crate) fn proposals(&self) -> Result<Vec<Vec<(StateVersion, StateBuffers)>>> {
        self.bindings.proposals()
    }
    pub(crate) fn state_bindings(&self) -> impl Iterator<Item = &[StateBinding]> {
        self.bindings.bindings()
    }
    pub(crate) fn values(&self) -> &[DeviceBuffer] {
        self.values.buffers()
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        &self.status
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn packets(&self) -> &[ResidentLease] {
        self.greedy.packets()
    }
    pub(crate) fn native_launches(&self) -> usize {
        self.native_launches.load(Ordering::Acquire)
    }
    pub(crate) fn projection_launches(&self) -> usize {
        self.projection_launches.load(Ordering::Acquire)
    }
    pub(crate) fn sampler_dispatches(&self) -> usize {
        self.sampler_dispatches.load(Ordering::Acquire)
    }
}

impl ResidentProgram {
    pub(crate) fn from_cohort_body(body: Arc<FullModelCohort>) -> Result<Self> {
        let record = body.bindings.record(0)?;
        let plan = record.plan().clone();
        let states = native::state_requirements(body.bindings.source(), &plan)?;
        let state_bytes = checked_sum(states.iter().map(|state| state.bytes))?
            .checked_mul(body.bindings.width())
            .ok_or_else(|| Error::Limit("B4 all-row state preflight extent".into()))?;
        let values = body.values().to_vec();
        let value_bytes = checked_sum(
            values
                .iter()
                .map(|value| value.original_lease().map(|lease| lease.bytes()))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let canonical = body
            .bindings
            .state_accesses()
            .into_iter()
            .map(|(lease, _)| lease.id())
            .collect::<BTreeSet<_>>();
        let value_ids = values
            .iter()
            .map(|value| value.original_lease().map(|lease| lease.id()))
            .collect::<Result<BTreeSet<_>>>()?;
        let mut source_bytes = 0usize;
        let mut workspace_bytes = 0usize;
        let mut owned_ranges = OwnedRanges::new(body.values.service().client().service_id());
        for (lease, access) in body.accesses() {
            if canonical.contains(&lease.id()) {
                continue;
            }
            owned_ranges.insert(BufferRange::new(&lease.handle())?)?;
            if value_ids.contains(&lease.id()) || lease.id() == body.status.id() {
                continue;
            }
            let total = if *access == DeviceAccess::Read {
                &mut source_bytes
            } else {
                &mut workspace_bytes
            };
            *total = total
                .checked_add(lease.bytes())
                .ok_or_else(|| Error::Limit("B4 retained flight byte preflight".into()))?;
        }
        let preflight = ResidentPreflight {
            source: body.bindings.source().source_identity(),
            source_bytes,
            value_bytes,
            workspace_bytes,
            status_bytes: 4,
            parameter_bytes: 0,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states,
            scratch: [0; SCRATCH_SLOTS],
        };
        preflight.total_with_state()?;
        Ok(Self {
            _source_owner: ResidentProgramSource::Windows(body.bindings.source().clone()),
            plan,
            preflight,
            values,
            status: body.status.handle(),
            owned_ranges,
            body: ResidentProgramBody::NativeCohort(body),
        })
    }
}
