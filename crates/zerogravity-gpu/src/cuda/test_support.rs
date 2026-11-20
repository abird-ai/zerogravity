//! Model-free diagnostics: original service plus finite compiler/payload charges.
use crate::resident_leases::ResidentService;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::{
    Error, Result,
    accelerator::SourceIdentity,
    resource_arbiter::{
        AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey,
        ResourceArbiter, ResourceOwner,
    },
};

#[test]
#[ignore = "physical CUDA same-service secondary stream and event ownership"]
fn owned_secondary_stream_events_are_disjoint_and_exact() -> Result<()> {
    use cubecl_common::bytes::Bytes;

    const BYTES: usize = 8 * 1024 * 1024;
    let diagnostic = CudaDiagnostic::new("secondary-stream-event")?;
    let service = &diagnostic.service;
    let primary = service.pinned_cuda_queue()?.clone();
    let primary_client = service.client().clone();
    let (secondary_client, secondary) = unsafe {
        primary
            .fork_stream(service.client().clone())
            .map_err(|error| Error::State(format!("secondary stream fork: {error}")))?
    };
    if primary.service_id() != secondary.service_id()
        || primary_client.service_id() != secondary_client.service_id()
        || primary.logical_stream_id() == secondary.logical_stream_id()
    {
        return Err(Error::State(
            "secondary CUDA stream service/identity differs".into(),
        ));
    }

    let primary_handle = primary_client.empty(BYTES);
    let secondary_handle = secondary_client.empty(BYTES);
    let primary_bytes = vec![0xa5u8; BYTES];
    let secondary_bytes = vec![0x5au8; BYTES];
    primary_client.write(
        &primary_handle,
        Bytes::from_bytes_vec(primary_bytes.clone()),
    );
    secondary_client.write(
        &secondary_handle,
        Bytes::from_bytes_vec(secondary_bytes.clone()),
    );

    // Both physical events are recorded before either is waited. Their buffers
    // are disjoint, so no cross-stream dependency or synchronization is needed.
    let mut primary_event = primary
        .record_event()
        .map_err(|error| Error::State(format!("primary event record: {error}")))?;
    let mut secondary_event = secondary
        .record_event()
        .map_err(|error| Error::State(format!("secondary event record: {error}")))?;
    if primary_event.stream_id() != primary.logical_stream_id()
        || secondary_event.stream_id() != secondary.logical_stream_id()
    {
        return Err(Error::State(
            "owned CUDA event stream identity differs".into(),
        ));
    }
    let primary_ready_before_wait = primary_event
        .ready()
        .map_err(|error| Error::State(format!("primary event query: {error}")))?;
    let secondary_ready_before_wait = secondary_event
        .ready()
        .map_err(|error| Error::State(format!("secondary event query: {error}")))?;
    primary_event
        .wait()
        .map_err(|error| Error::State(format!("primary event wait: {error}")))?;
    secondary_event
        .wait()
        .map_err(|error| Error::State(format!("secondary event wait: {error}")))?;

    let actual_primary = primary_client
        .read_one(primary_handle)
        .map_err(|error| Error::State(format!("primary readback: {error}")))?;
    let actual_secondary = secondary_client
        .read_one(secondary_handle)
        .map_err(|error| Error::State(format!("secondary readback: {error}")))?;
    if actual_primary.as_ref() != primary_bytes || actual_secondary.as_ref() != secondary_bytes {
        return Err(Error::State(
            "secondary CUDA stream disjoint readback differs".into(),
        ));
    }
    println!(
        "P12_CUDA_SECONDARY_STREAM_EVENT_PASS same_service=true streams_distinct=true disjoint_buffers=true events_recorded_before_wait=true primary_ready_before_wait={primary_ready_before_wait} secondary_ready_before_wait={secondary_ready_before_wait} readback=exact"
    );
    Ok(())
}

#[test]
#[ignore = "physical CUDA resident event-backed in-flight ownership lifecycle"]
fn resident_cuda_in_flight_event_lifecycle_is_exact() -> Result<()> {
    use cubecl_common::bytes::Bytes;
    use zerogravity_core::device_events::{BufferRole, DeviceAccess};

    const BYTES: usize = 8 * 1024 * 1024;
    let diagnostic = CudaDiagnostic::new("resident-event-flight")?;
    let service = diagnostic.service.clone();
    let primary = service.pinned_cuda_queue()?.clone();
    let (secondary_client, secondary) = unsafe {
        primary
            .fork_stream(service.client().clone())
            .map_err(|error| Error::State(format!("resident secondary stream: {error}")))?
    };
    let slot = || -> Result<crate::physical_queue::ActiveSlot> {
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 1,
            }],
            None,
        )?;
        cubecl::future::block_on(ticket.wait())
    };

    // Known completion: event first proves physical quiescence, then the
    // existing readback/validator path alone may initialize outputs/Ready.
    let output = service.allocate_uninitialized(BYTES, BufferRole::Output)?;
    let mut queue = slot()?;
    queue.begin_submission()?;
    let mut flight = service.submit(0, &[(output.clone(), DeviceAccess::Write)])?;
    let owner = Arc::new(());
    let owner_weak = Arc::downgrade(&owner);
    flight.retain_completion_owner(owner.clone())?;
    drop(owner);
    let expected = vec![0x31u8; BYTES];
    secondary_client.write(&output.handle(), Bytes::from_bytes_vec(expected.clone()));
    let mut in_flight = unsafe { flight.into_cuda_in_flight(&secondary)? };
    assert!(owner_weak.upgrade().is_some());
    let _ = in_flight.ready()?;
    let readback = output.readback(0, BYTES)?;
    let (ready, raw) = cubecl::future::block_on(in_flight.complete_with_readback(
        &[output.clone()],
        &[readback],
        |raw| {
            if raw.len() != 1 || raw[0].as_ref() != expected {
                return Err(Error::State("resident event known readback differs".into()));
            }
            Ok(())
        },
    ))?;
    queue.finish()?;
    assert!(output.is_initialized());
    drop((ready, raw));
    assert!(owner_weak.upgrade().is_none());
    service.ensure_usable()?;

    // Cancelled completion: the logical event is cancelled before the CUDA
    // completion witness retires it. Writes never become initialized/Ready.
    let cancelled = service.allocate_uninitialized(BYTES, BufferRole::Output)?;
    let mut queue = slot()?;
    queue.begin_submission()?;
    let mut flight = service.submit(0, &[(cancelled.clone(), DeviceAccess::Write)])?;
    let cancel_owner = Arc::new(());
    let cancel_weak = Arc::downgrade(&cancel_owner);
    flight.retain_completion_owner(cancel_owner.clone())?;
    drop(cancel_owner);
    secondary_client.write(
        &cancelled.handle(),
        Bytes::from_bytes_vec(vec![0x72u8; BYTES]),
    );
    let mut in_flight = unsafe { flight.into_cuda_in_flight(&secondary)? };
    in_flight.request_cancel()?;
    assert!(cancel_weak.upgrade().is_some());
    in_flight.retire_cancelled()?;
    queue.finish()?;
    assert!(!cancelled.is_initialized());
    assert!(cancel_weak.upgrade().is_none());
    service.ensure_usable()?;

    // Unresolved drop: no query/wait result authorizes retirement. The original
    // armed submission must fence/quarantine exact leases + completion owner.
    let quarantined_before =
        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    let uncertain = service.allocate_uninitialized(BYTES, BufferRole::Output)?;
    let mut queue = slot()?;
    queue.begin_submission()?;
    let mut flight = service.submit(0, &[(uncertain.clone(), DeviceAccess::Write)])?;
    let unknown_owner = Arc::new(());
    let unknown_weak = Arc::downgrade(&unknown_owner);
    flight.retain_completion_owner(unknown_owner.clone())?;
    drop(unknown_owner);
    secondary_client.write(
        &uncertain.handle(),
        Bytes::from_bytes_vec(vec![0xc4u8; BYTES]),
    );
    let in_flight = unsafe { flight.into_cuda_in_flight(&secondary)? };
    assert!(unknown_weak.upgrade().is_some());
    drop(in_flight);
    drop(queue);
    assert!(!uncertain.is_initialized());
    assert!(service.ensure_usable().is_err());
    assert!(unknown_weak.upgrade().is_some());
    let quarantined_after =
        crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    assert!(quarantined_after > quarantined_before);

    println!(
        "P12_CUDA_RESIDENT_INFLIGHT_EVENT_PASS known_readback=exact known_initialized=true cancel_retired=true cancel_initialized=false unresolved_fenced=true unresolved_initialized=false completion_owner_retained=true quarantine_owner_delta={} queue_width_unchanged=true",
        quarantined_after - quarantined_before
    );
    Ok(())
}

pub(crate) struct CudaDiagnostic {
    pub(crate) service: Arc<ResidentService>,
    pub(crate) compiler: Arc<(GlobalResourceLease, GlobalResourceLease)>,
    _payload: GlobalResourceLease,
}
impl CudaDiagnostic {
    pub(crate) fn new(label: &'static str) -> Result<Self> {
        let device =
            cubecl::Device::cuda(0).map_err(|e| Error::State(format!("{label} CUDA: {e}")))?;
        Self::new_on(label, device)
    }
    /// Same finite diagnostic ownership for an explicitly selected CUDA
    /// device. Keep qualification code on one compiler/charge path rather than
    /// constructing ad-hoc clients that silently fall back to PTX.
    pub(crate) fn new_on(label: &'static str, device: cubecl::Device) -> Result<Self> {
        const MIB: usize = 1024 * 1024;
        let service = ResidentService::new_pinned_cuda(
            device,
            SourceIdentity::for_new_allocation()?,
            ResourceOwner::new(label, "isolated-diagnostic")?,
            256 * MIB,
            (768 * MIB) as u64,
        )?;
        let arbiter = ResourceArbiter::process_shared();
        let host = PhysicalDeviceKey::new("host", &format!("zerogravity-{label}-diagnostic"))?;
        arbiter.register_matching(host.clone(), (512 * MIB) as u64)?;
        let reserve = |key: &PhysicalDeviceKey, bytes| {
            arbiter.admit(
                key,
                ResourceOwner::new(label, "compiler-context")?,
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes,
                }],
            )
        };
        let compiler = Arc::new((
            reserve(&host, cubecl::cuda::CudaCompilerContextPolicy::HOST_BYTES)?,
            reserve(
                &service.physical_key(),
                cubecl::cuda::CudaCompilerContextPolicy::DEVICE_BYTES,
            )?,
        ));
        let adopted = Arc::new(AtomicBool::new(false));
        let installed = unsafe {
            service
                .pinned_cuda_queue()?
                .install_compiler_context(compiler.clone(), adopted.clone())
        };
        // Adopted code/context outlives model retirement, until process exit.
        if adopted.load(Ordering::Acquire) || installed.is_err() {
            std::mem::forget(compiler.clone());
        }
        installed.map_err(|e| Error::State(format!("{label} compiler owner: {e}")))?;
        let payload = reserve(&host, (128 * MIB) as u64)?;
        Ok(Self {
            service,
            compiler,
            _payload: payload,
        })
    }
}
