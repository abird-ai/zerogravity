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

#[test]
#[ignore = "physical CUDA private provider transfer and ordered secondary-stream completion"]
fn resident_private_provider_bridge_is_ordered_and_retained() -> Result<()> {
    use crate::cuda::bf16_gemm::{Bf16Gemm, Bf16GemmReduction};
    use cubecl_common::bytes::Bytes;
    use zerogravity_core::{
        accelerator::program::ValueStorageFormat,
        device_events::{BufferRole, DeviceAccess},
        placement::PreparedTensorLayout,
        provider::{ProviderBufferMode, ProviderCompletionMode, ProviderRequirements},
    };

    let diagnostic = CudaDiagnostic::new("private-provider-bridge")?;
    let service = &diagnostic.service;
    let primary = service.pinned_cuda_queue()?.clone();
    let (_, secondary) = unsafe { primary.fork_stream(service.client().clone()) }
        .map_err(|error| Error::State(format!("provider secondary queue: {error}")))?;
    let secondary = Arc::new(secondary);
    let (rows, inner, columns) = (8, 32, 64);
    let gemm = Arc::new(Bf16Gemm::with_reduction_certified(
        columns,
        rows,
        inner,
        Bf16GemmReduction::Ordered,
    )?);
    gemm.prepare_code(&primary)?;
    let input_layout = PreparedTensorLayout::dense(2, inner * 2, rows)?;
    let output_layout = PreparedTensorLayout::dense(2, columns * 2, rows)?;
    let workspace_bytes = input_layout.byte_len()? + output_layout.byte_len()? + 4;
    let requirements = ProviderRequirements {
        workspace_bytes,
        opaque_device_bytes: 0,
        buffers: ProviderBufferMode::Private,
        completion: ProviderCompletionMode::ExternalFence,
    };
    let slot = || -> Result<crate::physical_queue::ActiveSlot> {
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: workspace_bytes as u64,
            }],
            None,
        )?;
        cubecl::future::block_on(ticket.wait())
    };
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>()
    };
    let weights_bytes = encode(&vec![1.0; columns * inner]);
    let inputs = (0..rows)
        .flat_map(|row| std::iter::repeat_n((row + 1) as f32, inner))
        .collect::<Vec<_>>();
    let expected = (0..rows)
        .flat_map(|row| std::iter::repeat_n(((row + 1) * inner) as f32, columns))
        .collect::<Vec<_>>();
    let expected = encode(&expected);

    let mut active = slot()?;
    active.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let weights = init.upload_bytes(&weights_bytes, BufferRole::ImmutableWeights)?;
    let input = init.allocate_empty(input_layout.byte_len()?, BufferRole::Activation)?;
    let private_input = init.allocate_empty(input_layout.byte_len()?, BufferRole::Scratch)?;
    let private_output = init.allocate_empty(output_layout.byte_len()?, BufferRole::Scratch)?;
    let status = init.allocate_empty(4, BufferRole::Scratch)?;
    let output = init.allocate_empty(output_layout.byte_len()?, BufferRole::Output)?;
    let uncertain = init.allocate_empty(output_layout.byte_len()?, BufferRole::Output)?;
    drop(cubecl::future::block_on(init.complete_sync())?);
    active.finish()?;
    let workspace = [&private_input, &private_output, &status];
    use crate::resident_leases::ProviderView;
    let input_view = ProviderView::new(&input, 0, input_layout, ValueStorageFormat::Bf16)?;
    let private_input_view =
        ProviderView::new(&private_input, 0, input_layout, ValueStorageFormat::Bf16)?;
    let private_output_view =
        ProviderView::new(&private_output, 0, output_layout, ValueStorageFormat::Bf16)?;
    let output_view = ProviderView::new(&output, 0, output_layout, ValueStorageFormat::Bf16)?;
    // Same bytes do not authorize an encoding/layout conversion.
    let wrong = ProviderView::new(
        &private_input,
        0,
        PreparedTensorLayout::dense(4, inner * 2, rows)?,
        ValueStorageFormat::F32,
    )?;
    assert!(matches!(
        input_view.require_copy_to(wrong),
        Err(Error::Unsupported(_))
    ));
    assert!(input_view.require_copy_to(input_view).is_err());
    let wrong_rows = ProviderView::new(
        &private_input,
        0,
        PreparedTensorLayout::dense(2, inner * 4, rows / 2)?,
        ValueStorageFormat::Bf16,
    )?;
    assert!(input_view.require_copy_to(wrong_rows).is_err());
    let padded = PreparedTensorLayout::strided(2, inner, inner * 2, rows)?;
    let padded_input = ProviderView::new(&input, 0, padded, ValueStorageFormat::Bf16)?;
    let padded_private = ProviderView::new(&private_input, 0, padded, ValueStorageFormat::Bf16)?;
    assert!(padded_input.require_copy_to(padded_private).is_err());
    assert!(ProviderView::new(&input, 1, input_layout, ValueStorageFormat::Bf16).is_err());
    assert!(
        service
            .bind_cuda_provider(&primary, requirements, &workspace, &[])
            .is_err()
    );
    assert!(
        service
            .bind_cuda_provider(
                &secondary,
                ProviderRequirements {
                    completion: ProviderCompletionMode::OriginalQueue,
                    ..requirements
                },
                &workspace,
                &[]
            )
            .is_err()
    );
    assert!(
        service
            .bind_cuda_provider(&secondary, requirements, &workspace[..2], &[])
            .is_err()
    );
    assert!(
        service
            .bind_cuda_provider(
                &secondary,
                requirements,
                &[&private_input, &private_input, &status],
                &[]
            )
            .is_err()
    );
    service.ensure_usable()?;
    let binding = service.bind_cuda_provider(&secondary, requirements, &workspace, &[])?;
    println!(
        "CUDA_PROVIDER_PREFLIGHT_PASS incompatible_stream_layout_extent_refused=true unfenced=true"
    );

    let mut active = slot()?;
    active.begin_submission()?;
    let flight = service.submit(
        0,
        &[
            (weights.clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Write),
            (private_input.clone(), DeviceAccess::Write),
            (private_output.clone(), DeviceAccess::Write),
            (status.clone(), DeviceAccess::Write),
            (output.clone(), DeviceAccess::Write),
        ],
    )?;
    // Real primary producer, not an already completed input. The outbound
    // event orders the private transfer and the math behind this write.
    service
        .client()
        .write(&input.handle(), Bytes::from_bytes_vec(encode(&inputs)));
    let flight = unsafe {
        flight.with_cuda_provider(binding, gemm.clone(), |commands| {
            commands.copy(input_view, private_input_view)?;
            commands.require_accesses([
                (&weights, DeviceAccess::Read),
                (&private_input, DeviceAccess::Read),
                (&private_output, DeviceAccess::Write),
                (&status, DeviceAccess::Write),
            ])?;
            commands
                .client()
                .write(&status.handle(), Bytes::from_bytes_vec(vec![0; 4]));
            gemm.prepare_certified(
                commands.client(),
                weights.handle(),
                private_input.handle(),
                private_output.handle(),
                status.handle(),
            )?
            .launch()?;
            commands.copy(private_output_view, output_view)
        })
    }?;
    assert!(!output.is_initialized()); // No Ready merely from inserting a wait.
    let readback = [output.readback(0, output.bytes())?, status.readback(0, 4)?];
    let (ready, _) = cubecl::future::block_on(flight.complete_with_readback(
        &[output.clone()],
        &readback,
        |actual| {
            if actual.len() != 2 || actual[0].as_ref() != expected || actual[1].as_ref() != [0; 4] {
                return Err(Error::State(
                    "private provider GEMM/copy output differs".into(),
                ));
            }
            Ok(())
        },
    ))?;
    drop(ready);
    active.finish()?;
    assert!(output.is_initialized());
    service.ensure_usable()?;

    // Error after actual external enqueue cannot hand the original submission
    // back for a premature primary readback. Keep the provider/leases fenced.
    let mut active = slot()?;
    active.begin_submission()?;
    let flight = service.submit(
        0,
        &[
            (input.clone(), DeviceAccess::Read),
            (private_input.clone(), DeviceAccess::Write),
            (private_output.clone(), DeviceAccess::Write),
            (status.clone(), DeviceAccess::Write),
            (uncertain.clone(), DeviceAccess::Write),
        ],
    )?;
    let weak = Arc::downgrade(&gemm);
    let handles_before = crate::physical_quarantine::quarantined_handles_for(service.client())?;
    let result = unsafe {
        flight.with_cuda_provider(binding, gemm, |commands| {
            commands.copy(input_view, private_input_view)?;
            Err(Error::State(
                "diagnostic interrupted external provider".into(),
            ))
        })
    };
    assert!(
        matches!(result, Err(Error::State(message)) if message == "diagnostic interrupted external provider")
    );
    drop(active);
    assert!(service.ensure_usable().is_err());
    assert!(!uncertain.is_initialized());
    assert!(weak.upgrade().is_some());
    assert!(
        crate::physical_quarantine::quarantined_handles_for(service.client())? > handles_before
    );
    println!(
        "CUDA_PRIVATE_PROVIDER_BRIDGE_PASS exact_native_gemm=true original_event_return=true private_copy=true no_early_ready=true interrupted_fenced=true retained_owner=true"
    );
    Ok(())
}

#[test]
#[ignore = "physical CUDA bounded dual resident flights on disjoint streams"]
fn resident_dual_flights_retire_independently() -> Result<()> {
    use cubecl_common::bytes::Bytes;
    use zerogravity_core::device_events::{BufferRole, DeviceAccess};

    const BYTES: usize = 8 * 1024 * 1024;
    let diagnostic = CudaDiagnostic::new("resident-dual-flight")?;
    let service = diagnostic.service.clone();
    let primary = service.pinned_cuda_queue()?.clone();
    let primary_client = service.client().clone();
    let (secondary_client, secondary) = unsafe {
        primary
            .fork_stream(service.client().clone())
            .map_err(|error| Error::State(format!("dual-flight secondary stream: {error}")))?
    };
    let queue = Arc::new(crate::physical_queue::PhysicalQueue::new(
        service.physical_key(),
        4,
        2,
    )?);
    let ticket = || {
        queue.submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 1,
            }],
            None,
        )
    };
    let mut one_slot = cubecl::future::block_on(ticket()?.wait())?;
    let mut two_slot = cubecl::future::block_on(ticket()?.wait())?;
    one_slot.begin_submission()?;
    two_slot.begin_submission()?;

    let one = service.allocate_uninitialized(BYTES, BufferRole::Output)?;
    let two = service.allocate_uninitialized(BYTES, BufferRole::Output)?;
    let one_expected = vec![0x11u8; BYTES];
    let two_expected = vec![0x22u8; BYTES];
    let one_flight = service.submit(11, &[(one.clone(), DeviceAccess::Write)])?;
    let two_flight = service.submit(22, &[(two.clone(), DeviceAccess::Write)])?;
    primary_client.write(&one.handle(), Bytes::from_bytes_vec(one_expected.clone()));
    secondary_client.write(&two.handle(), Bytes::from_bytes_vec(two_expected.clone()));
    let one_flight = unsafe { one_flight.into_cuda_in_flight(&primary)? };
    let two_flight = unsafe { two_flight.into_cuda_in_flight(&secondary)? };

    // Retire in reverse order. The shared prepaid envelope remains Submitted
    // until the final known flight completes; each queue slot retires alone.
    let (two_ready, two_raw) = cubecl::future::block_on(two_flight.complete_with_readback(
        &[two.clone()],
        &[two.readback(0, BYTES)?],
        |raw| {
            if raw.len() != 1 || raw[0].as_ref() != two_expected {
                return Err(Error::State("dual-flight second readback differs".into()));
            }
            Ok(())
        },
    ))?;
    two_slot.finish()?;
    assert!(two.is_initialized());
    assert!(!one.is_initialized());

    let (one_ready, one_raw) = cubecl::future::block_on(one_flight.complete_with_readback(
        &[one.clone()],
        &[one.readback(0, BYTES)?],
        |raw| {
            if raw.len() != 1 || raw[0].as_ref() != one_expected {
                return Err(Error::State("dual-flight first readback differs".into()));
            }
            Ok(())
        },
    ))?;
    one_slot.finish()?;
    assert!(one.is_initialized());
    drop((one_ready, one_raw, two_ready, two_raw));
    service.ensure_usable()?;
    assert_eq!(queue.stats()?.completed, 2);

    println!(
        "P12_CUDA_RESIDENT_DUAL_FLIGHT_PASS active_capacity=2 same_prepaid_envelope=true streams_distinct=true disjoint_outputs=true events_before_completion=true reverse_retirement=true readback=exact outputs_initialized=true service_reusable=true"
    );
    Ok(())
}

pub(crate) struct CudaDiagnostic {
    pub(crate) service: Arc<ResidentService>,
    _compiler: Arc<(GlobalResourceLease, GlobalResourceLease)>,
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
            _compiler: compiler,
            _payload: payload,
        })
    }
}
