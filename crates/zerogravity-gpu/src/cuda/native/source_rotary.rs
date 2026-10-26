//! Original source factories before learned-W loading; no Root/model permit.
use super::RetainedSource;
use super::projection_cubin::resident::{AdmittedImages, NativeRetentionWitness, PreparedImages};
use super::projection_cubin::{native_error, not_cancelled};
use super::{CubinQualificationInputs, finite_bf16};
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{Device, bytes::Bytes, cuda::OwnedCudaKernel};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::{
    Error, Result,
    device_events::{BufferRole, DeviceAccess},
    resource_arbiter::{AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner},
};
mod manifests;
pub use manifests::RotaryOutput;
const MANAGED_BYTES: usize = 809501320;
const NATIVE_BYTES: usize = 64 * 1024 * 1024;
const CAP: usize = 896 * 1024 * 1024;

pub(super) fn required_images()
-> std::collections::BTreeSet<super::cubin_manifests::ProjectionImage> {
    manifests::required_images()
}
mod layout;
mod qualification;
use layout::{Layout, SourceKind};
pub use qualification::{RotaryMode, qualify_source_rotary_cubins, qualify_text_rotary_cubins};

pub(crate) fn text_rotary_retained_host_bytes(positions: usize) -> Result<usize> {
    Layout::new(SourceKind::Text, positions)?.retained_host_bytes()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RotaryComparison {
    pub status: &'static str,
    pub native_launches: usize,
    pub completed_readback_flights: usize,
    pub authentic_compared_bytes: usize,
    pub checked_source_scalar_bytes: usize,
    pub source_prefix_bytes: usize,
    pub native_cache_full_extent: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub physical_reserved_bytes: u64,
    pub backend_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub all_uninitialized_retained: bool,
    pub retained_images: usize,
    pub retained_kernels: usize,
    pub native_charge_bytes: usize,
}
/// The source-derived payload is deliberately absent from the public qualifier API.
/// Only same-engine cold model code can consume the privately sealed prefix.
pub struct RotaryOutcome {
    pub comparison: RotaryComparison,
    prefix: Option<Arc<SourceRotaryPrefix>>,
    retired: bool,
}
impl RotaryOutcome {
    fn require_complete_for(&self, kind: SourceKind, positions: usize) -> Result<()> {
        let prefix = self.prefix.as_ref().ok_or_else(|| {
            Error::State("rotary preparation has no completed source prefix".into())
        })?;
        prefix.require_source(prefix.source())?;
        prefix.require_kind(kind)?;
        prefix.require_span(0, positions)?;
        if prefix.positions() != positions {
            return Err(Error::State(
                "rotary prepared context extent differs".into(),
            ));
        }
        if !self.retired {
            return Err(Error::State(
                "rotary source preparation has not retired".into(),
            ));
        }
        Ok(())
    }
    pub fn require_fixed_complete(&self, positions: usize) -> Result<()> {
        self.require_complete_for(SourceKind::FixedPrompt, positions)
    }
    pub fn require_text_complete(&self, positions: usize) -> Result<()> {
        self.require_complete_for(SourceKind::Text, positions)
    }
    pub fn discard_source_prefix(self) -> Result<()> {
        let prefix = self
            .prefix
            .ok_or_else(|| Error::State("rotary completion has no source prefix".into()))?;
        prefix.require_source(prefix.source())?;
        prefix.require_span(0, prefix.owner.profile.positions())?;
        drop(prefix);
        Ok(())
    }
    pub(crate) fn into_fixed_prefix(self) -> Result<Arc<SourceRotaryPrefix>> {
        self.into_prefix_for(SourceKind::FixedPrompt)
    }
    pub(crate) fn into_text_prefix(self) -> Result<Arc<SourceRotaryPrefix>> {
        self.into_prefix_for(SourceKind::Text)
    }
    fn into_prefix_for(self, kind: SourceKind) -> Result<Arc<SourceRotaryPrefix>> {
        let prefix = self
            .prefix
            .ok_or_else(|| Error::State("rotary completion has no source prefix".into()))?;
        prefix.require_kind(kind)?;
        if !self.retired {
            return Err(Error::State(
                "rotary source preparation has not retired".into(),
            ));
        }
        Ok(prefix)
    }
}
struct MetadataOwner {
    profile: Layout,
    source: Arc<RetainedSource>,
    _metadata: WindowWorkspace,
    _readback: WindowWorkspace,
    _prefix: WindowWorkspace,
}
/// Bytes drop before their original admission. No public bytes constructor or Root authority.
pub(crate) struct SourceRotaryPrefix {
    bytes: Box<[u8]>,
    owner: Arc<MetadataOwner>,
}
impl SourceRotaryPrefix {
    pub(crate) fn require_source(&self, source: &VerifiedTensorSource) -> Result<()> {
        if self.bytes.len() != self.owner.profile.bytes()
            || !self.owner.source.source().same_issuer(source)
            || self.owner.source.source_identity() != source.source_identity()
            || !finite_bf16(self.bytes.as_ref())
        {
            return Err(Error::State(
                "computed rotary original immutable source issuer differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }
    pub(crate) fn source(&self) -> &VerifiedTensorSource {
        self.owner.source.source()
    }
    pub(crate) fn require_span(&self, base: usize, rows: usize) -> Result<()> {
        self.owner.profile.require_span(base, rows)
    }
    fn require_kind(&self, kind: SourceKind) -> Result<()> {
        self.owner.profile.require_kind(kind)
    }
    pub(crate) fn require_fixed_route(&self) -> Result<()> {
        self.require_kind(SourceKind::FixedPrompt)
    }
    pub(crate) fn require_text_route(&self) -> Result<()> {
        self.require_kind(SourceKind::Text)
    }
    pub(crate) fn positions(&self) -> usize {
        self.owner.profile.positions()
    }
    pub(crate) fn retained_host_bytes(&self) -> Result<usize> {
        let actual = self
            .owner
            ._metadata
            .bytes()
            .checked_add(self.owner._readback.bytes())
            .and_then(|bytes| bytes.checked_add(self.owner._prefix.bytes()))
            .ok_or_else(|| Error::Limit("rotary retained source extent".into()))?;
        if actual != self.owner.profile.retained_host_bytes()? {
            return Err(Error::State(
                "rotary retained source ownership extent differs".into(),
            ));
        }
        Ok(actual)
    }
}

struct Owner {
    metadata: Arc<MetadataOwner>,
    leases: Vec<ResidentLease>,
}
impl Owner {
    fn lease(&self, role: RotaryOutput) -> Result<&ResidentLease> {
        self.leases
            .get(role as usize)
            .ok_or_else(|| Error::State("source rotary lease role absent".into()))
    }
}
struct Launch {
    kernel: Arc<OwnedCudaKernel>,
    roles: &'static [RotaryOutput],
}
struct Programs {
    owner: Arc<Owner>,
    images: Arc<PreparedImages>,
    launches: Vec<Launch>,
}
impl Programs {
    fn prepare(
        owner: Arc<Owner>,
        images: Arc<PreparedImages>,
        init: &mut crate::resident_leases::ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        let mut launches = Vec::with_capacity(RotaryOutput::ALL.len());
        init.retain_completion_owner(owner.clone())?;
        init.retain_completion_owner(images.clone())?;
        let service = owner
            .leases
            .first()
            .ok_or_else(|| Error::State("source rotary original buffers absent".into()))?
            .service();
        init.require_service(service)?;
        images.require_service(service)?;
        if owner.leases.len() != RotaryOutput::ALL.len()
            || owner.leases.iter().zip(RotaryOutput::ALL).any(|(l, r)| {
                l.bytes() != r.bytes()
                    || !Arc::ptr_eq(l.service(), service)
                    || l.source_identity() != owner.metadata.source.source_identity()
            })
        {
            return Err(Error::State(
                "source rotary same source/service/exact role leases".into(),
            ));
        }
        for record in manifests::launches() {
            not_cancelled(cancelled)?;
            let image = images.image(record.image)?;
            if image.sha256() != record.image.sha256()
                || image.byte_len() != record.image.bytes()
                || image.binary_sm() != 86
            {
                return Err(Error::State("rotary immutable source image differs".into()));
            }
            // SAFETY: exact closed source/selected-code ABI and original armed preparation.
            let kernel =
                unsafe { images.set().prepare(image, record.manifest) }.map_err(native_error)?;
            init.retain_completion_owner(kernel.clone())?;
            launches.push(Launch {
                kernel,
                roles: record.roles,
            });
        }
        if launches.len() != RotaryOutput::ALL.len() {
            return Err(Error::State(
                "source rotary complete native output records".into(),
            ));
        }
        let result = Arc::new(Self {
            owner,
            images,
            launches,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn witness(&self) -> Result<NativeRetentionWitness> {
        self.images.retention_witness(
            &self
                .launches
                .iter()
                .map(|l| l.kernel.clone())
                .collect::<Vec<_>>(),
        )
    }
    unsafe fn enqueue(&self, index: usize, cancelled: Option<&AtomicBool>) -> Result<()> {
        not_cancelled(cancelled)?;
        let launch = self
            .launches
            .get(index)
            .ok_or_else(|| Error::State("source rotary stage absent".into()))?;
        let service = self.owner.leases[0].service();
        service.ensure_usable()?;
        self.images.require_service(service)?;
        let handles = launch
            .roles
            .iter()
            .map(|&role| Ok(self.owner.lease(role)?.handle()))
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: caller's same original armed flight retains complete owner/access union.
        unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)
    }
    unsafe fn retire(&self) -> Result<()> {
        for launch in &self.launches {
            // SAFETY: same original trusted completion, then retained read-only flight.
            unsafe { launch.kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        // SAFETY: every original record retired; armed read-only guard owns partial unload.
        unsafe { self.images.set().retire_after_physical_completion() }.map_err(native_error)
    }
}
fn check_snapshot_zero(
    key: &zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    backend: u64,
) -> Result<()> {
    let r = ResourceArbiter::process_shared().snapshot(key)?;
    if r.reserved != backend
        || r.fenced
        || (backend == 0 && (r.reservations != 0 || !r.per_class.is_empty()))
        || (backend != 0
            && (r.reservations != 1
                || r.active_models
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    != ["backend/cuda"]))
    {
        return Err(Error::State(
            "source rotary original physical envelope did not retire".into(),
        ));
    }
    Ok(())
}
struct CompletedRotary {
    prefix: Arc<SourceRotaryPrefix>,
    readback: Vec<Bytes>,
    profile: Layout,
    native_launches: usize,
    backend_reserved: u64,
}
impl CompletedRotary {
    fn into_outcome(self, status: &'static str) -> RotaryOutcome {
        RotaryOutcome {
            comparison: RotaryComparison {
                status,
                native_launches: self.native_launches,
                completed_readback_flights: 2,
                physical_reserved_bytes: self.backend_reserved,
                backend_reserved_bytes: self.backend_reserved,
                authentic_compared_bytes: RotaryOutput::ALL
                    .into_iter()
                    .map(|role| self.profile.prefix_bytes(role))
                    .sum::<usize>()
                    - 8,
                checked_source_scalar_bytes: 8,
                source_prefix_bytes: self.profile.bytes(),
                native_cache_full_extent: 134217728,
                ..Default::default()
            },
            prefix: Some(self.prefix),
            retired: true,
        }
    }
}

/// Compute and retain a fixed-prompt source span with original native math.
/// No reference case, expected output or fault selection enters model loading.
pub async fn prepare_fixed_prompt_rotary(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: CubinQualificationInputs,
    positions: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<RotaryOutcome> {
    let profile = Layout::new(SourceKind::FixedPrompt, positions)?;
    Ok(
        execute_rotary(device, source, device_cap, inputs, cancelled, profile)
            .await?
            .into_outcome(profile.production_ready_status()),
    )
}

/// Compute and retain a text source span with original native math.
pub async fn prepare_text_rotary(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: CubinQualificationInputs,
    positions: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<RotaryOutcome> {
    let profile = Layout::new(SourceKind::Text, positions)?;
    Ok(
        execute_rotary(device, source, device_cap, inputs, cancelled, profile)
            .await?
            .into_outcome(profile.production_ready_status()),
    )
}

/// Fault-free producer used by model loading and complete qualification cases.
async fn execute_rotary(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: CubinQualificationInputs,
    cancelled: Option<&AtomicBool>,
    profile: Layout,
) -> Result<CompletedRotary> {
    not_cancelled(cancelled)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA source rotary producer".into(),
        ));
    }
    // Reserve before any own descriptor/source/lease/queue clones or allocations.
    let metadata = source.source().reserve_workspace(64 * 1024)?;
    let readback = source
        .source()
        .reserve_workspace(profile.readback_bytes()?)?;
    let prefix = source
        .source()
        .reserve_workspace(profile.prefix_reservation_bytes()?)?;
    let metadata = Arc::new(MetadataOwner {
        profile,
        source,
        _metadata: metadata,
        _readback: readback,
        _prefix: prefix,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        metadata.source.source_identity(),
        ResourceOwner::new(
            profile.production_owner(),
            "source-rotary-before-model-weights",
        )?,
        CAP,
        device_cap,
    )?;
    let compiler_context = if matches!(profile.kind, SourceKind::Text) {
        Some(service.admit_cuda_compiler_context(metadata.source.source())?)
    } else {
        None
    };
    let backend_reserved = if compiler_context.is_some() {
        crate::physical_service::CUDA_COMPILER_DEVICE_BYTES
    } else {
        0
    };
    let key = service.physical_key();
    let initial = ResourceArbiter::process_shared().snapshot(&key)?;
    if initial.fenced || initial.reserved != CAP as u64 + backend_reserved {
        return Err(Error::State(
            "source rotary requires solitary original physical envelope".into(),
        ));
    }
    if inputs.device_cap != device_cap
        || inputs.images.capacity() > 16
        || inputs.images.len() != 10
        || inputs
            .images
            .iter()
            .map(|i| i.kind)
            .collect::<std::collections::BTreeSet<_>>()
            != required_images()
    {
        return Err(Error::Malformed(
            "source rotary exact ten admitted images/cap".into(),
        ));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: CAP as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(metadata.clone())?;
    let mut leases = Vec::with_capacity(RotaryOutput::ALL.len());
    for role in RotaryOutput::ALL {
        leases.push(init.allocate_empty(role.bytes(), BufferRole::Activation)?);
    }
    if leases.iter().map(ResidentLease::bytes).sum::<usize>() != MANAGED_BYTES {
        return Err(Error::State("source rotary full1M managed extent".into()));
    }
    let owner = Arc::new(Owner { metadata, leases });
    init.retain_completion_owner(owner.clone())?;
    for image in &inputs.images {
        if image.bytes.len() != image.kind.bytes()
            || image.bytes.capacity() > 32 * 1024 * 1024
            || image.admission.bytes() < image.bytes.capacity()
            || !image
                .admission
                .source()
                .same_issuer(owner.metadata.source.source())
        {
            return Err(Error::Malformed(
                "source rotary full image capacity/source admission".into(),
            ));
        }
    }
    let admitted = AdmittedImages::reserve(&service, owner.clone(), inputs.images)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let images = admitted.prepare_source_rotary_images(sources, &mut init, cancelled)?;
    let programs = Programs::prepare(owner.clone(), images, &mut init, cancelled)?;
    let initialized = init.complete_sync().await?;
    drop(initialized);
    slot.finish()?;
    let accesses = owner
        .leases
        .iter()
        .cloned()
        .map(|l| (l, DeviceAccess::Write))
        .collect::<Vec<_>>();
    let reads = RotaryOutput::ALL
        .iter()
        .map(|&role| owner.lease(role)?.readback(0, profile.prefix_bytes(role)))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: MANAGED_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(programs.clone())?;
    for index in 0..programs.launches.len() {
        // SAFETY: one original flight retains all twelve writes and their dependency graph.
        unsafe { programs.enqueue(index, cancelled) }?;
    }
    let native_launches = programs.launches.len();
    if native_launches != RotaryOutput::ALL.len() {
        return Err(Error::State(
            "source rotary incomplete actual native sequence".into(),
        ));
    }
    let (ready, raw) = flight
        .complete_with_readback(&owner.leases, &reads, |raw| {
            if raw.len() != RotaryOutput::ALL.len()
                || raw
                    .iter()
                    .zip(RotaryOutput::ALL)
                    .any(|(b, r)| b.len() != profile.prefix_bytes(r))
                || raw[2].as_ref() != 10_000_000i64.to_le_bytes()
                || !finite_bf16(raw[11].as_ref())
            {
                return Err(Error::State(
                    "source rotary exact completed prefix/scalar cohort".into(),
                ));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    let mut bytes = vec![0u8; profile.bytes()].into_boxed_slice();
    bytes.copy_from_slice(raw[11].as_ref());
    let prefix = Arc::new(SourceRotaryPrefix {
        bytes,
        owner: owner.metadata.clone(),
    });
    prefix.require_source(owner.metadata.source.source())?;
    prefix.require_span(0, profile.positions())?;
    // Original initialized read lease guards every record/module unload error.
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: profile.bytes() as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut retirement = service.submit(
        0,
        &[(
            owner.lease(RotaryOutput::Cache)?.clone(),
            DeviceAccess::Read,
        )],
    )?;
    retirement.retain_completion_owner(programs.clone())?;
    // SAFETY: all producer math physically completed. Same original guard retains cache/native owners.
    unsafe { programs.retire() }?;
    let retired = retirement
        .complete_read_only_readback(
            &[owner
                .lease(RotaryOutput::Cache)?
                .readback(0, profile.bytes())?],
            |actual| {
                if actual.len() != 1 || &actual[0][..] != prefix.bytes() {
                    return Err(Error::State(
                        "source rotary computed prefix changed during native retirement".into(),
                    ));
                }
                Ok(())
            },
        )
        .await?;
    slot.finish()?;
    drop((retired, ready, reads, accesses, programs, owner));
    if service.resident_bytes()? != 0
        || service.pending_events()? != 0
        || service.uploaded_value_bytes() != 0
        || quarantined_handles_for(service.client())? != 0
        || quarantined_completion_owners_for(service.client())? != 0
    {
        return Err(Error::State(
            "source rotary positive managed/native events did not retire".into(),
        ));
    }
    service.ensure_usable()?;
    drop(service);
    check_snapshot_zero(&key, backend_reserved)?;
    Ok(CompletedRotary {
        prefix,
        readback: raw,
        profile,
        native_launches,
        backend_reserved,
    })
}
