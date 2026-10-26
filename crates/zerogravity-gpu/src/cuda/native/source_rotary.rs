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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixProfile {
    FixedPrompt,
    Text,
}
impl PrefixProfile {
    fn rows(self) -> usize {
        match self {
            Self::FixedPrompt => 67,
            Self::Text => 79,
        }
    }
    fn bytes(self) -> usize {
        self.rows() * 128
    }
    fn prefix_bytes(self, role: RotaryOutput) -> usize {
        if self == Self::FixedPrompt {
            return role.prefix_bytes();
        }
        match role {
            RotaryOutput::Range1048576 => self.rows() * 4,
            RotaryOutput::Frequencies | RotaryOutput::Cos | RotaryOutput::Sin => {
                self.rows() * 32 * 4
            }
            RotaryOutput::Cat => self.rows() * 64 * 4,
            RotaryOutput::Cache => self.bytes(),
            _ => role.prefix_bytes(),
        }
    }
    fn require_span(self, base: usize, rows: usize) -> Result<()> {
        if rows == 0 || base.checked_add(rows).is_none_or(|end| end > self.rows()) {
            return Err(Error::Unsupported(
                "computed rotary finite prefix interval".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotaryMode {
    Complete,
    Pressure,
    CancelBeforeFirstKernel,
    UnknownAfterPow,
    CancelAfterOuter,
    UnknownAfterCat,
    CancelBeforeCast,
    UnknownAfterCast,
}
impl RotaryMode {
    fn cancel_before(self) -> Option<usize> {
        match self {
            Self::CancelBeforeFirstKernel => Some(0),
            Self::CancelAfterOuter => Some(8),
            Self::CancelBeforeCast => Some(11),
            _ => None,
        }
    }
    fn unknown_after(self) -> Option<usize> {
        match self {
            Self::UnknownAfterPow => Some(4),
            Self::UnknownAfterCat => Some(11),
            Self::UnknownAfterCast => Some(12),
            _ => None,
        }
    }
    fn is_fault(self) -> bool {
        !matches!(self, Self::Complete | Self::Pressure)
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
    pub fn discard_source_prefix(self) -> Result<()> {
        let prefix = self.into_prefix()?;
        prefix.require_source(prefix.source())?;
        prefix.require_span(0, prefix.owner.profile.rows())?;
        drop(prefix);
        Ok(())
    }
    pub(crate) fn into_prefix(self) -> Result<Arc<SourceRotaryPrefix>> {
        self.prefix
            .ok_or_else(|| Error::State("rotary completion has no source prefix".into()))
    }
}
struct MetadataOwner {
    profile: PrefixProfile,
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
        self.owner.profile == PrefixProfile::Text
    }
}
#[cfg(test)]
fn require_prefix_span(base: usize, rows: usize) -> Result<()> {
    PrefixProfile::FixedPrompt.require_span(base, rows)
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
        let mut launches = Vec::with_capacity(12);
        init.retain_completion_owner(owner.clone())?;
        init.retain_completion_owner(images.clone())?;
        let service = owner
            .leases
            .first()
            .ok_or_else(|| Error::State("source rotary original buffers absent".into()))?
            .service();
        init.require_service(service)?;
        images.require_service(service)?;
        if owner.leases.len() != 12
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
        if launches.len() != 12 {
            return Err(Error::State("source rotary exact12 native records".into()));
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
/// All mathematical inputs are original source factories/constants. Callback
/// may compare independent expected bytes only AFTER the actual native math.
/// No captured inv_freq/frequencies/cos/sin/cache can enter device preparation.
pub async fn qualify_source_rotary_cubins<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<RotaryOutcome>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
    qualify_rotary_inner(
        device,
        source,
        device_cap,
        inputs,
        mode,
        cancelled,
        validate,
        PrefixProfile::FixedPrompt,
    )
    .await
}

/// Same original native source constructor; authentic .8B/2B and positions0..79.
pub async fn qualify_text_rotary_cubins<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
) -> Result<RotaryOutcome>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
    qualify_rotary_inner(
        device,
        source,
        device_cap,
        inputs,
        mode,
        cancelled,
        validate,
        PrefixProfile::Text,
    )
    .await
}

#[expect(clippy::too_many_arguments)]
async fn qualify_rotary_inner<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
    profile: PrefixProfile,
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
    let readback = source.source().reserve_workspace(1024 * 1024)?;
    let prefix = source.source().reserve_workspace(profile.bytes() * 3)?;
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
            match profile {
                PrefixProfile::FixedPrompt => "original-hf08",
                PrefixProfile::Text => "qwen35-text",
            },
            "source-rotary-before-model-weights",
        )?,
        CAP,
        device_cap,
    )?;
    let compiler_context = if matches!(profile, PrefixProfile::Text) {
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
    if mode == RotaryMode::Pressure {
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
    let mut leases = Vec::with_capacity(12);
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
    for index in 0..12 {
        if mode.cancel_before() == Some(index) {
            local_cancel.store(true, Ordering::Release);
        }
        // SAFETY: one original flight retains all twelve writes and their dependency graph.
        let result = unsafe { programs.enqueue(index, cancelled, &local_cancel) };
        if mode.cancel_before() == Some(index) {
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
        if mode.unknown_after() == Some(native_launches) {
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
            || owner.leases.len() != 12
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
        if retained.image_owners != 10
            || retained.kernel_owners != 12
            || retained.native_charge_bytes != NATIVE_BYTES
            || resident_bytes != MANAGED_BYTES + NATIVE_BYTES
            || pending_events != 1
            || handles != 12
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
                status: if mode.cancel_before().is_some() {
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
                retained_images: 10,
                retained_kernels: 12,
                native_charge_bytes: NATIVE_BYTES,
                ..Default::default()
            },
            prefix: None,
        });
    }
    if native_launches != 12 {
        return Err(Error::State(
            "source rotary incomplete actual native sequence".into(),
        ));
    }
    let (ready, raw) = flight
        .complete_with_readback(&owner.leases, &reads, |raw| {
            if raw.len() != 12
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
    prefix.require_span(0, profile.rows())?;
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
            status: match profile {
                PrefixProfile::FixedPrompt => "SOURCE_ROTARY_FIRST67_EXACT_NO_MODEL_QUALIFICATION",
                PrefixProfile::Text => "SOURCE_ROTARY_FIRST79_EXACT_NO_MODEL_QUALIFICATION",
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

#[cfg(test)]
mod source_span_tests {
    use super::*;
    #[test]
    fn short_profile_has_complete_79_positions_and_preserves_legacy67() {
        let short = PrefixProfile::Text;
        for (base, rows) in [
            (0, 47),
            (0, 57),
            (0, 62),
            (0, 67),
            (0, 72),
            (47, 1),
            (78, 1),
            (0, 79),
        ] {
            short.require_span(base, rows).unwrap();
        }
        for (base, rows) in [
            (0, 0),
            (79, 1),
            (78, 2),
            (0, 80),
            (usize::MAX, 1),
            (1, usize::MAX),
        ] {
            assert!(short.require_span(base, rows).is_err());
        }
        assert_eq!(short.bytes(), 10112);
        assert_eq!(short.prefix_bytes(RotaryOutput::Range1048576), 316);
        assert_eq!(short.prefix_bytes(RotaryOutput::Cat), 20224);
        assert_eq!(
            RotaryOutput::ALL
                .into_iter()
                .map(|r| short.prefix_bytes(r))
                .sum::<usize>(),
            61636
        );
        assert_eq!(PrefixProfile::FixedPrompt.bytes(), 8576);
        for role in RotaryOutput::ALL {
            assert_eq!(
                PrefixProfile::FixedPrompt.prefix_bytes(role),
                role.prefix_bytes()
            );
            assert!(short.prefix_bytes(role) <= role.bytes());
        }
        assert!(PrefixProfile::FixedPrompt.require_span(67, 1).is_err());
    }
    #[test]
    fn only_actual_defined_prefix_intervals_are_available() {
        for (base, rows) in [(0, 1), (1, 1), (0, 64), (64, 2), (66, 1), (0, 67)] {
            assert!(require_prefix_span(base, rows).is_ok());
        }
        for (base, rows) in [
            (0, 0),
            (67, 1),
            (66, 2),
            (0, 68),
            (usize::MAX, 1),
            (1, usize::MAX),
        ] {
            assert!(require_prefix_span(base, rows).is_err());
        }
    }
}
