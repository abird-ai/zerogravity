//! Original source factories before learned-W loading; no Root/model permit.
use super::RetainedSource;
use super::projection_cubin::resident::{AdmittedImages, NativeRetentionWitness, PreparedImages};
use super::projection_cubin::{native_error, not_cancelled};
use super::{CubinQualificationInputs, finite_bf16};
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{Device, cuda::OwnedCudaKernel};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
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
    let layout = Layout::new(SourceKind::Text, positions)?;
    (64usize * 1024)
        .checked_add(layout.readback_bytes()?)
        .and_then(|bytes| bytes.checked_add(layout.prefix_reservation_bytes().ok()?))
        .ok_or_else(|| Error::Limit("rotary source preparation extent".into()))
}

#[derive(Clone, Copy, Default)]
struct FaultPlan {
    pressure: bool,
    cancel_before: Option<usize>,
    unknown_after: Option<usize>,
}
impl FaultPlan {
    fn is_fault(self) -> bool {
        self.cancel_before.is_some() || self.unknown_after.is_some()
    }
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
}
impl RotaryOutcome {
    pub fn require_complete(&self, positions: usize) -> Result<()> {
        let prefix = self.prefix.as_ref().ok_or_else(|| {
            Error::State("rotary preparation has no completed source prefix".into())
        })?;
        prefix.require_source(prefix.source())?;
        prefix.require_span(0, positions)?;
        if prefix.positions() != positions {
            return Err(Error::State(
                "rotary prepared context extent differs".into(),
            ));
        }
        let state = &self.comparison;
        if state.resident_bytes != 0
            || state.pending_events != 0
            || state.physical_fenced
            || state.physical_reserved_bytes != state.backend_reserved_bytes
        {
            return Err(Error::State(
                "rotary source preparation has not retired".into(),
            ));
        }
        Ok(())
    }
    pub fn discard_source_prefix(self) -> Result<()> {
        let prefix = self.into_prefix()?;
        prefix.require_source(prefix.source())?;
        prefix.require_span(0, prefix.owner.profile.positions())?;
        drop(prefix);
        Ok(())
    }
    pub(crate) fn into_prefix(self) -> Result<Arc<SourceRotaryPrefix>> {
        self.prefix
            .ok_or_else(|| Error::State("rotary completion has no source prefix".into()))
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
    pub(crate) fn is_text(&self) -> bool {
        self.owner.profile.kind == SourceKind::Text
    }
    pub(crate) fn positions(&self) -> usize {
        self.owner.profile.positions()
    }
    pub(crate) fn retained_host_bytes(&self) -> Result<usize> {
        self.owner
            ._metadata
            .bytes()
            .checked_add(self.owner._readback.bytes())
            .and_then(|bytes| bytes.checked_add(self.owner._prefix.bytes()))
            .ok_or_else(|| Error::Limit("rotary retained source extent".into()))
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
    unsafe fn enqueue(
        &self,
        index: usize,
        external: Option<&AtomicBool>,
        local: &AtomicBool,
    ) -> Result<()> {
        not_cancelled(external)?;
        not_cancelled(Some(local))?;
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
/// Compute and retain the requested source span with original native math.
/// No reference case, expected output or fault selection enters model loading.
pub async fn prepare_text_rotary(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: CubinQualificationInputs,
    positions: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<RotaryOutcome> {
    let layout = Layout::new(SourceKind::Text, positions)?;
    prepare_rotary_inner(
        device,
        source,
        device_cap,
        Some(inputs),
        FaultPlan::default(),
        cancelled,
        |_| Ok(()),
        layout,
    )
    .await
}

#[expect(clippy::too_many_arguments)]
async fn prepare_rotary_inner<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: FaultPlan,
    cancelled: Option<&AtomicBool>,
    validate: F,
    profile: Layout,
) -> Result<RotaryOutcome>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
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
            match profile.kind {
                SourceKind::FixedPrompt => "original-hf08",
                SourceKind::Text => "qwen35-text",
            },
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
    if mode.pressure {
        if !matches!(
            service.allocate_uninitialized(CAP + 1, BufferRole::Scratch),
            Err(Error::Limit(_))
        ) || service.resident_bytes()? != 0
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != 0
            || quarantined_handles_for(service.client())? != 0
            || quarantined_completion_owners_for(service.client())? != 0
        {
            return Err(Error::State(
                "source rotary cap+1 did not refuse before buffer/native/H2D".into(),
            ));
        }
        drop((inputs, metadata, service));
        check_snapshot_zero(&key, backend_reserved)?;
        return Ok(RotaryOutcome {
            comparison: RotaryComparison {
                status: "SOURCE_ROTARY_PRESSURE_REFUSED_BEFORE_BUFFER_NATIVE_H2D",
                physical_reserved_bytes: backend_reserved,
                backend_reserved_bytes: backend_reserved,
                ..Default::default()
            },
            prefix: None,
        });
    }
    let inputs =
        inputs.ok_or_else(|| Error::Malformed("source rotary original images absent".into()))?;
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
    let native_witness = programs.witness()?;
    let programs_weak = Arc::downgrade(&programs);
    let owner_weak = Arc::downgrade(&owner);
    let source_weak = owner.metadata.source.weak_owner();
    let service_weak = Arc::downgrade(&service);
    let local_cancel = AtomicBool::new(false);
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
    let mut native_launches = 0;
    let mut cut = false;
    for index in 0..programs.launches.len() {
        if mode.cancel_before == Some(index) {
            local_cancel.store(true, Ordering::Release);
        }
        // SAFETY: one original flight retains all twelve writes and their dependency graph.
        let result = unsafe { programs.enqueue(index, cancelled, &local_cancel) };
        if mode.cancel_before == Some(index) {
            if !matches!(result, Err(Error::State(_))) {
                return Err(Error::State(
                    "source rotary actual pre-enqueue cancellation failed".into(),
                ));
            }
            cut = true;
            break;
        }
        result?;
        native_launches += 1;
        if mode.unknown_after == Some(native_launches) {
            cut = true;
            break;
        }
    }
    if mode.is_fault() {
        if !cut {
            return Err(Error::State(
                "source rotary exact real fault cut absent".into(),
            ));
        }
        drop(flight);
        drop(slot);
        drop((reads, accesses, programs, owner, service));
        let service = service_weak
            .upgrade()
            .ok_or_else(|| Error::State("source rotary unknown lost service".into()))?;
        let owner = owner_weak.upgrade().ok_or_else(|| {
            Error::State("source rotary unknown lost original leases/source".into())
        })?;
        if programs_weak.upgrade().is_none()
            || source_weak.upgrade().is_none()
            || owner.leases.len() != RotaryOutput::ALL.len()
            || owner.leases.iter().any(ResidentLease::is_initialized)
        {
            return Err(Error::State(
                "source rotary unknown lost uninitialized producer graph".into(),
            ));
        }
        let retained = native_witness.inspect()?;
        let physical = ResourceArbiter::process_shared().snapshot(&key)?;
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        if retained.image_owners != manifests::required_images().len()
            || retained.kernel_owners != RotaryOutput::ALL.len()
            || retained.native_charge_bytes != NATIVE_BYTES
            || resident_bytes != MANAGED_BYTES + NATIVE_BYTES
            || pending_events != 1
            || handles != RotaryOutput::ALL.len()
            || owners == 0
            || service.uploaded_value_bytes() != 0
            || physical.reserved != CAP as u64 + backend_reserved
            || !physical.fenced
        {
            return Err(Error::State(
                "source rotary actual unknown quarantine/accounting differs".into(),
            ));
        }
        return Ok(RotaryOutcome {
            comparison: RotaryComparison {
                status: if mode.cancel_before.is_some() {
                    "SOURCE_ROTARY_CANCELLED_UNKNOWN_RETAINED"
                } else {
                    "SOURCE_ROTARY_UNKNOWN_RETAINED"
                },
                native_launches,
                resident_bytes,
                pending_events,
                physical_reserved_bytes: physical.reserved,
                backend_reserved_bytes: backend_reserved,
                physical_fenced: true,
                quarantined_handles: handles,
                quarantined_completion_owners: owners,
                source_retained: true,
                all_uninitialized_retained: true,
                retained_images: manifests::required_images().len(),
                retained_kernels: RotaryOutput::ALL.len(),
                native_charge_bytes: NATIVE_BYTES,
                ..Default::default()
            },
            prefix: None,
        });
    }
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
            let records = RotaryOutput::ALL
                .iter()
                .copied()
                .zip(raw.iter().map(|b| b.as_ref()))
                .collect::<Vec<_>>();
            validate(&records)
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
    drop((retired, ready, raw, reads, accesses, programs, owner));
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
    Ok(RotaryOutcome {
        comparison: RotaryComparison {
            status: match profile.kind {
                SourceKind::FixedPrompt => "SOURCE_ROTARY_FIXED_PREFIX_READY",
                SourceKind::Text => "SOURCE_ROTARY_TEXT_PREFIX_READY",
            },
            native_launches,
            completed_readback_flights: 2,
            physical_reserved_bytes: backend_reserved,
            backend_reserved_bytes: backend_reserved,
            authentic_compared_bytes: RotaryOutput::ALL
                .into_iter()
                .map(|r| profile.prefix_bytes(r))
                .sum::<usize>()
                - 8,
            checked_source_scalar_bytes: 8,
            source_prefix_bytes: profile.bytes(),
            native_cache_full_extent: 134217728,
            ..Default::default()
        },
        prefix: Some(prefix),
    })
}
