//! Finite actual B4 native packets on the existing source/module context.
//! Original node loop, independent root authorities and flight remain callers.
mod generated;
mod generated_2b;
mod packets;
use super::{
    cubin_manifests::ProjectionImage,
    projection_cubin::{native_error, not_cancelled, resident::PreparedImages},
    short_hf_native::ShortHfNativeStorage,
};
use crate::resident_inference::native_fragment::{PrefixPhaseEnqueue, short_hf::ShortHfCallInput};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::resident_program::BufferRange;
use cubecl::{
    cuda::{CubinIo, CubinKernelManifest, OwnedCudaKernel},
    prelude::{BufferArg, CubeCount, CubeDim},
};
pub(crate) use packets::Role;
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    Error, Result,
    accelerator::program::PlanOperation,
    device_events::{BufferRole, DeviceAccess},
};

/// Source-family phase geometry only; native eligibility still requires the
/// exact source catalogue, arithmetic contract and completed admitted owners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CohortPhases {
    InitialAndCached,
    CachedOnly,
}
impl CohortPhases {
    pub(crate) const fn for_profile(
        profile: zerogravity_core::accelerator::program::FullModelProfile,
    ) -> Self {
        match profile {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                Self::InitialAndCached
            }
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => Self::CachedOnly,
        }
    }
    pub(crate) const fn contains(self, initial: bool) -> bool {
        !initial || matches!(self, Self::InitialAndCached)
    }
    pub(crate) fn iter(self) -> impl Iterator<Item = bool> {
        [true, false]
            .into_iter()
            .filter(move |initial| self.contains(*initial))
    }
}

/// Family shape only: logical time/teacher issuers stay in every actual call.
/// This metadata grants no source, current-root, physical or native authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CohortInputGeometry {
    pub(crate) profile: zerogravity_core::accelerator::program::FullModelProfile,
    pub(crate) initial: bool,
    pub(crate) logical_rows: usize,
    pub(crate) physical_rows: usize,
    pub(crate) native_call: usize,
}
pub(crate) fn input_geometry(inputs: &[Arc<ShortHfCallInput>]) -> Result<CohortInputGeometry> {
    if inputs.len() != 4 {
        return Err(Error::Unsupported("B4 actual request population".into()));
    }
    for input in inputs {
        if input.scope().call().as_vllm_call() != input.call() {
            return Err(Error::State("B4 actual call/record issuer differs".into()));
        }
    }
    let calls: [_; 4] = std::array::from_fn(|row| inputs[row].scope().call());
    packets::geometry(&calls)
}

pub(crate) fn require_input_shape(inputs: &[Arc<ShortHfCallInput>]) -> Result<usize> {
    packets::call(inputs)
}

struct Record {
    manifest: CubinKernelManifest,
    kernel: Option<Arc<OwnedCudaKernel>>,
}
pub(crate) struct CohortNativeLibrary {
    storage: Arc<ShortHfNativeStorage>,
    _images: Arc<PreparedImages>,
    service: Arc<ResidentService>,
    records: Box<[Record]>,
    _metadata: WindowWorkspace,
}
impl CohortNativeLibrary {
    pub(crate) const fn host_bytes(
        profile: zerogravity_core::accelerator::program::FullModelProfile,
    ) -> usize {
        match profile {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                generated::METADATA_BOUND
            }
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
                generated_2b::METADATA_BOUND
            }
        }
    }
    pub(crate) fn profile(&self) -> zerogravity_core::accelerator::program::FullModelProfile {
        self.storage.profile()
    }
    pub(crate) fn weights(&self) -> &Arc<super::model_weights::ModelWeights> {
        self.storage.weights()
    }
    pub(crate) fn workspace(&self) -> &ResidentLease {
        self.storage.cohort_workspace()
    }
    /// Enclosing whole-body admission allocates every value/private/root owner
    /// first. This extends records on the original native set; no second context.
    pub(crate) fn prepare(
        storage: Arc<ShortHfNativeStorage>,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        let source = storage.weights().source();
        if source.source_identity() != service.source_identity() {
            return Err(Error::State(
                "B4 native existing immutable source/service".into(),
            ));
        }
        storage.weights().require_initialized()?;
        // Definitions are selected by the SAME sealed source profile; the2B
        // catalogue contains cached calls only and never INITIAL permission.
        let metadata = source
            .source()
            .reserve_workspace(Self::host_bytes(storage.profile()))?;
        let images = storage.cohort_images().clone();
        images.require_service(&service)?;
        init.retain_completion_owner(storage.clone())?;
        init.retain_completion_owner(images.clone())?;
        let (definition_count, definitions) = match storage.profile() {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                (generated::DEFINITION_COUNT, generated::definitions())
            }
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
                (generated_2b::DEFINITION_COUNT, generated_2b::definitions())
            }
        };
        let mut records = Vec::with_capacity(definition_count);
        for definition in definitions {
            not_cancelled(cancelled)?;
            let kernel = if definition.image == Some(ProjectionImage::RotaryBf16CastSm86) {
                // Only proven bit-copy bijections use the existing integer
                // transport below; keep their complete original ABI provenance.
                None
            } else {
                let image = match definition.image {
                    Some(kind) => images.image(kind)?,
                    None => images.activation()?,
                };
                // SAFETY: generated exact complete B4 source ABI on the same
                // source-admitted image/set; real initialization retains owners.
                let kernel = unsafe { images.set().prepare(image, definition.manifest.clone()) }
                    .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                Some(kernel)
            };
            records.push(Record {
                manifest: definition.manifest,
                kernel,
            });
        }
        if records.len() != definition_count {
            return Err(Error::State("B4 complete native record catalogue".into()));
        }
        let result = Arc::new(Self {
            storage,
            _images: images,
            service,
            records: records.into_boxed_slice(),
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    /// # Safety
    /// No live requests/proposals; original quiescence barrier completed. The
    /// armed retirement flight retains this library and its complete source
    /// graph through partial failure. Retire records before their shared set.
    pub(crate) unsafe fn retire_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        for record in &self.records {
            if let Some(kernel) = &record.kernel {
                // SAFETY: caller holds the original completed-work barrier and
                // armed retirement flight; each record belongs to this library.
                unsafe { kernel.retire_after_physical_completion() }?;
            }
        }
        Ok(())
    }
    /// The actual source node is checked in every original row. Callers derive
    /// semantic leases from the sealed source weights/B values/state controls.
    pub(crate) fn bind(
        self: &Arc<Self>,
        inputs: &[Arc<ShortHfCallInput>],
        node: usize,
        roles: &BTreeMap<Role, ResidentLease>,
        status: ResidentLease,
    ) -> Result<CohortNativeSite> {
        self.service.ensure_usable()?;
        if inputs.len() != 4
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || !Arc::ptr_eq(status.service(), &self.service)
        {
            return Err(Error::State("B4 native actual four rows/status".into()));
        }
        let call = packets::call(inputs)?;
        let mut kind = None;
        for input in inputs {
            if input.call().model_profile() != self.profile()
                || !Arc::ptr_eq(input.source().source(), self.storage.weights().source())
            {
                return Err(Error::State("B4 actual per-row native source/call".into()));
            }
            let actual = input
                .record()
                .plan()
                .nodes()
                .get(node)
                .ok_or_else(|| Error::State("B4 actual source node absent".into()))?;
            let PlanOperation::TypedVllm08 { descriptor, .. } = &actual.operation else {
                return Err(Error::State("B4 native actual typed source node".into()));
            };
            descriptor.validate()?;
            if descriptor.call() != input.call() || kind.is_some_and(|k| k != descriptor.kind()) {
                return Err(Error::State(
                    "B4 every original source descriptor/invocation".into(),
                ));
            }
            kind = Some(descriptor.kind());
        }
        let mut bound = Vec::new();
        let mut accesses = Vec::new();
        let packets = match self.profile() {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                generated::site(call, node)?
            }
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
                generated_2b::site(call, node)?
            }
        };
        for packet in packets {
            let record = self
                .records
                .get(packet.definition)
                .ok_or_else(|| Error::State("B4 native exact record index".into()))?;
            if packet.roles.len() != record.manifest.bindings.len() || packet.clears.len() > 1 {
                return Err(Error::State("B4 native role/clear population".into()));
            }
            let leases = packet
                .roles
                .iter()
                .zip(&record.manifest.bindings)
                .map(|(role, spec)| {
                    let lease = roles
                        .get(role)
                        .ok_or_else(|| Error::State(format!("B4 source role absent: {role:?}")))?;
                    if !Arc::ptr_eq(lease.service(), &self.service)
                        || lease.bytes() as u64 != spec.bytes
                        || lease.source_identity() != self.service.source_identity()
                        || BufferRange::new(&lease.handle())?
                            .overlaps(&BufferRange::new(&status.handle())?)
                    {
                        return Err(Error::State(format!(
                            "B4 exact physical source role extent: {role:?}"
                        )));
                    }
                    accesses.push((
                        lease.clone(),
                        if spec.io == CubinIo::Read {
                            DeviceAccess::Read
                        } else {
                            DeviceAccess::Write
                        },
                    ));
                    Ok(lease.clone())
                })
                .collect::<Result<Vec<_>>>()?;
            // Every distinct semantic role has disjoint physical storage. The
            // manifest's duplicate pointers are typed offsets into one role.
            for (index, lease) in leases.iter().enumerate() {
                let range = BufferRange::new(&lease.handle())?;
                for other in &leases[..index] {
                    if range.overlaps(&BufferRange::new(&other.handle())?) {
                        return Err(Error::State("B4 native semantic roles overlap".into()));
                    }
                }
            }
            let copy = if record.kernel.is_none() {
                let mode = match packet.roles.as_slice() {
                    [Role::PrivateQ, Role::Q] => 0,
                    [Role::O, Role::PrivateO] => 1,
                    [Role::PublicLse, Role::Lse] => 2,
                    _ => {
                        return Err(Error::Unsupported(
                            "unproved B4 original integer copy".into(),
                        ));
                    }
                };
                let bytes = if mode == 2 { 128 } else { 16384 };
                if leases.iter().any(|l| l.bytes() != bytes) || !packet.clears.is_empty() {
                    return Err(Error::State("B4 exact copy bijection extents".into()));
                }
                Some(mode)
            } else {
                None
            };
            let clear =
                packet
                    .clears
                    .first()
                    .map(|(role, offset, bytes)| {
                        let index =
                            packet.roles.iter().position(|r| r == role).ok_or_else(|| {
                                Error::State("B4 clear source role absent".into())
                            })?;
                        if *role != Role::Workspace || *offset != 0 || !match self.profile() {
                            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                                matches!(*bytes, 24 | 64 | 160)
                            }
                            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
                                *bytes == 128
                            }
                        } || *bytes > leases[index].bytes()
                        {
                            return Err(Error::Unsupported(
                                "B4 source/SASS exact semaphore clear".into(),
                            ));
                        }
                        Ok((index, *bytes / 4))
                    })
                    .transpose()?;
            bound.push(BoundPacket {
                index: packet.definition,
                leases,
                copy,
                clear,
            });
        }
        accesses.push((status.clone(), DeviceAccess::Write));
        Ok(CohortNativeSite {
            library: self.clone(),
            inputs: inputs.to_vec().into_boxed_slice(),
            node,
            status,
            packets: bound.into_boxed_slice(),
            accesses,
            submitted: AtomicBool::new(false),
        })
    }
}
struct BoundPacket {
    index: usize,
    leases: Vec<ResidentLease>,
    copy: Option<u32>,
    clear: Option<(usize, usize)>,
}
pub(crate) struct CohortNativeSite {
    library: Arc<CohortNativeLibrary>,
    inputs: Box<[Arc<ShortHfCallInput>]>,
    node: usize,
    status: ResidentLease,
    packets: Box<[BoundPacket]>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    submitted: AtomicBool,
}
unsafe fn words(lease: &ResidentLease) -> BufferArg {
    // SAFETY: private exact source/role extent validated by the enclosing binder.
    unsafe { BufferArg::from_raw_parts(lease.handle(), lease.bytes() / 4) }
}
impl CohortNativeSite {
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    /// Caller retains the SAME registered whole B flight and actual node cursor.
    pub(crate) fn enqueue(
        &self,
        node: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        cancelled: Option<&AtomicBool>,
    ) -> Result<(usize, usize, usize)> {
        if node != self.node || permits.len() != 4 {
            return Err(Error::State(
                "B4 native original node/all-row permits".into(),
            ));
        }
        let service = &self.library.service;
        service.ensure_usable()?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, service)?;
        }
        not_cancelled(cancelled)?;
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State(
                "B4 native source node already submitted".into(),
            ));
        }
        let (mut native, mut copies, mut clears) = (0, 0, 0);
        for packet in &self.packets {
            not_cancelled(cancelled)?;
            if let Some((index, count)) = packet.clear {
                let destination = &packet.leases[index];
                unsafe {
                    crate::program_kernels::cohort_copy::clear_prefix::launch_unchecked(
                        service.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(64),
                        words(destination),
                        words(&self.status),
                        count,
                    );
                }
                clears += 1;
            }
            if let Some(mode) = packet.copy {
                let count = packet.leases[0].bytes() / 4;
                unsafe {
                    crate::program_kernels::cohort_copy::copy_words::launch_unchecked(
                        service.client(),
                        CubeCount::Static((count as u32).div_ceil(128), 1, 1),
                        CubeDim::new_1d(128),
                        words(&packet.leases[1]),
                        words(&packet.leases[0]),
                        words(&self.status),
                        mode,
                    );
                }
                copies += 1;
            } else {
                let record = &self.library.records[packet.index];
                let bindings = packet
                    .leases
                    .iter()
                    .map(ResidentLease::cubin_binding)
                    .collect::<Result<Vec<_>>>()?;
                // SAFETY: exact generated record, checked immutable roles and
                // surrounding original armed physical flight/completion owner.
                unsafe {
                    record
                        .kernel
                        .as_ref()
                        .expect("sealed native record")
                        .enqueue_workspace_bindings(bindings, None)
                }
                .map_err(native_error)?;
                native += 1;
            }
        }
        Ok((native, copies, clears))
    }
}

/// Descriptive captured compiler-input identity, never a runtime source permit.
pub fn short_hf_cohort_source_scope_sha256(
    profile: zerogravity_core::accelerator::program::FullModelProfile,
) -> &'static str {
    match profile {
        zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
            "d2b241e5e4feed9a8d16b871693e69169efa40c1b32572d08bb7cc342da80415"
        }
        zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
            generated_2b::SOURCE_INPUT_SHA256
        }
    }
}
