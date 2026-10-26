//! Diagnostic cases over the shared source constructor.
use super::*;
use std::sync::atomic::Ordering;

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
}

#[derive(Clone, Copy)]
struct FaultPlan {
    cancel_before: Option<usize>,
    unknown_after: Option<usize>,
}
impl FaultPlan {
    fn new(mode: RotaryMode) -> Self {
        Self {
            cancel_before: mode.cancel_before(),
            unknown_after: mode.unknown_after(),
        }
    }
}

struct QualificationEnvironment {
    profile: Layout,
    device_cap: u64,
    metadata: Arc<MetadataOwner>,
    service: Arc<ResidentService>,
    key: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    backend_reserved: u64,
}
impl QualificationEnvironment {
    fn new(
        device: Device,
        source: Arc<RetainedSource>,
        device_cap: u64,
        profile: Layout,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
            return Err(Error::Unsupported(
                "release original CUDA source rotary producer".into(),
            ));
        }
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
        Ok(Self {
            profile,
            device_cap,
            metadata,
            service,
            key,
            backend_reserved,
        })
    }

    fn refuse_pressure(self, inputs: Option<CubinQualificationInputs>) -> Result<RotaryOutcome> {
        if !matches!(
            self.service
                .allocate_uninitialized(CAP + 1, BufferRole::Scratch),
            Err(Error::Limit(_))
        ) || self.service.resident_bytes()? != 0
            || self.service.pending_events()? != 0
            || self.service.uploaded_value_bytes() != 0
            || quarantined_handles_for(self.service.client())? != 0
            || quarantined_completion_owners_for(self.service.client())? != 0
        {
            return Err(Error::State(
                "source rotary cap+1 did not refuse before buffer/native/H2D".into(),
            ));
        }
        drop((inputs, self.metadata, self.service));
        check_snapshot_zero(&self.key, self.backend_reserved)?;
        Ok(RotaryOutcome {
            comparison: RotaryComparison {
                status: "SOURCE_ROTARY_PRESSURE_REFUSED_BEFORE_BUFFER_NATIVE_H2D",
                physical_reserved_bytes: self.backend_reserved,
                backend_reserved_bytes: self.backend_reserved,
                ..Default::default()
            },
            prefix: None,
            retired: true,
        })
    }

    async fn prepare_fault(
        self,
        inputs: Option<CubinQualificationInputs>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<FaultExecution> {
        let inputs = inputs
            .ok_or_else(|| Error::Malformed("source rotary original images absent".into()))?;
        if inputs.device_cap != self.device_cap
            || inputs.images.capacity() > 16
            || inputs.images.len() != 10
            || inputs
                .images
                .iter()
                .map(|image| image.kind)
                .collect::<std::collections::BTreeSet<_>>()
                != required_images()
        {
            return Err(Error::Malformed(
                "source rotary exact ten admitted images/cap".into(),
            ));
        }
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: CAP as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut init = self.service.begin_initialization(0)?;
        init.retain_completion_owner(self.metadata.clone())?;
        let mut leases = Vec::with_capacity(RotaryOutput::ALL.len());
        for role in RotaryOutput::ALL {
            leases.push(init.allocate_empty(role.bytes(), BufferRole::Activation)?);
        }
        if leases.iter().map(ResidentLease::bytes).sum::<usize>() != MANAGED_BYTES {
            return Err(Error::State("source rotary full1M managed extent".into()));
        }
        let owner = Arc::new(Owner {
            metadata: self.metadata,
            leases,
        });
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
        let admitted = AdmittedImages::reserve(&self.service, owner.clone(), inputs.images)?;
        let sources = self
            .service
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
            .map(|lease| (lease, DeviceAccess::Write))
            .collect::<Vec<_>>();
        let reads = RotaryOutput::ALL
            .iter()
            .map(|&role| {
                owner
                    .lease(role)?
                    .readback(0, self.profile.prefix_bytes(role))
            })
            .collect::<Result<Vec<_>>>()?;
        let native_witness = programs.witness()?;
        Ok(FaultExecution {
            owner,
            programs,
            service: self.service,
            accesses,
            reads,
            native_witness,
            key: self.key,
            backend_reserved: self.backend_reserved,
        })
    }
}

struct FaultExecution {
    owner: Arc<Owner>,
    programs: Arc<Programs>,
    service: Arc<ResidentService>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    reads: Vec<crate::resident_leases::ResidentReadback>,
    native_witness: NativeRetentionWitness,
    key: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    backend_reserved: u64,
}
impl FaultExecution {
    async fn run(self, plan: FaultPlan, cancelled: Option<&AtomicBool>) -> Result<RotaryOutcome> {
        let programs_weak = Arc::downgrade(&self.programs);
        let owner_weak = Arc::downgrade(&self.owner);
        let source_weak = self.owner.metadata.source.weak_owner();
        let service_weak = Arc::downgrade(&self.service);
        let ticket = self.service.queue().submit(
            self.service.budget(),
            self.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: MANAGED_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        not_cancelled(cancelled)?;
        slot.begin_submission()?;
        let mut flight = self.service.submit(0, &self.accesses)?;
        flight.retain_completion_owner(self.programs.clone())?;
        let local_cancel = AtomicBool::new(false);
        let mut native_launches = 0;
        let mut cancelled_cut = false;
        let mut cut = false;
        for index in 0..self.programs.launches.len() {
            not_cancelled(cancelled)?;
            let cancel_before = plan.cancel_before == Some(index);
            if cancel_before {
                local_cancel.store(true, Ordering::Release);
            }
            // SAFETY: the same real flight retains all output writes and owners.
            let result = unsafe { self.programs.enqueue(index, Some(&local_cancel)) };
            if cancel_before {
                if !matches!(result, Err(Error::State(_))) {
                    return Err(Error::State(
                        "source rotary actual pre-enqueue cancellation failed".into(),
                    ));
                }
                cancelled_cut = true;
                cut = true;
                break;
            }
            result?;
            native_launches += 1;
            if plan.unknown_after == Some(native_launches) {
                cut = true;
                break;
            }
        }
        if !cut {
            return Err(Error::State(
                "source rotary exact real fault cut absent".into(),
            ));
        }
        drop(flight);
        drop(slot);
        drop((
            self.reads,
            self.accesses,
            self.programs,
            self.owner,
            self.service,
        ));
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
        let retained = self.native_witness.inspect()?;
        let physical = ResourceArbiter::process_shared().snapshot(&self.key)?;
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
            || physical.reserved != CAP as u64 + self.backend_reserved
            || !physical.fenced
        {
            return Err(Error::State(
                "source rotary actual unknown quarantine/accounting differs".into(),
            ));
        }
        Ok(RotaryOutcome {
            comparison: RotaryComparison {
                status: if cancelled_cut {
                    "SOURCE_ROTARY_CANCELLED_UNKNOWN_RETAINED"
                } else {
                    "SOURCE_ROTARY_UNKNOWN_RETAINED"
                },
                native_launches,
                resident_bytes,
                pending_events,
                physical_reserved_bytes: physical.reserved,
                backend_reserved_bytes: self.backend_reserved,
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
            retired: false,
        })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "source, physical admission, diagnostic plan and comparison are independent inputs"
)]
async fn qualify_rotary<F>(
    device: Device,
    source: Arc<RetainedSource>,
    device_cap: u64,
    inputs: Option<CubinQualificationInputs>,
    mode: RotaryMode,
    cancelled: Option<&AtomicBool>,
    validate: F,
    profile: Layout,
    complete_status: &'static str,
) -> Result<RotaryOutcome>
where
    F: FnOnce(&[(RotaryOutput, &[u8])]) -> Result<()>,
{
    if mode == RotaryMode::Complete {
        let inputs = inputs
            .ok_or_else(|| Error::Malformed("source rotary original images absent".into()))?;
        let completed =
            execute_rotary(device, source, device_cap, inputs, cancelled, profile).await?;
        let records = RotaryOutput::ALL
            .iter()
            .copied()
            .zip(completed.readback.iter().map(|bytes| bytes.as_ref()))
            .collect::<Vec<_>>();
        validate(&records)?;
        return Ok(completed.into_outcome(complete_status));
    }
    let environment =
        QualificationEnvironment::new(device, source, device_cap, profile, cancelled)?;
    if mode == RotaryMode::Pressure {
        return environment.refuse_pressure(inputs);
    }
    environment
        .prepare_fault(inputs, cancelled)
        .await?
        .run(FaultPlan::new(mode), cancelled)
        .await
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
    qualify_rotary(
        device,
        source,
        device_cap,
        inputs,
        mode,
        cancelled,
        validate,
        Layout::new(SourceKind::FixedPrompt, 67)?,
        "SOURCE_ROTARY_FIRST67_EXACT_NO_MODEL_QUALIFICATION",
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
    qualify_rotary(
        device,
        source,
        device_cap,
        inputs,
        mode,
        cancelled,
        validate,
        Layout::new(SourceKind::Text, 79)?,
        "SOURCE_ROTARY_FIRST79_EXACT_NO_MODEL_QUALIFICATION",
    )
    .await
}
