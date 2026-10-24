//! Expected-only completed callback, on the original physical owner path.
use super::*;
use cubecl::{Device, bytes::Bytes};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SliceFault {
    None,
    AfterRms,
    AfterProjection,
}
#[derive(Debug)]
pub struct SliceOutcome {
    pub physical_key: zerogravity_core::resource_arbiter::PhysicalDeviceKey,
    pub completed_nodes: Option<usize>,
    pub submitted_nodes: usize,
    pub math_enqueues: usize,
    pub dispatches: Option<usize>,
    pub readbacks: usize,
    pub callbacks: usize,
    pub ready: bool,
    pub source_graph_retained: bool,
    pub outputs_unpromoted: bool,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub fenced: bool,
    pub physical_reserved: u64,
}
pub struct CompletedSlice<'a> {
    record: &'a RecordedOperatorSlice,
    token: u32,
    names: Vec<&'static str>,
    raw: &'a [Bytes],
}
impl CompletedSlice<'_> {
    pub fn record(&self) -> &RecordedOperatorSlice {
        self.record
    }
    pub fn token(&self) -> u32 {
        self.token
    }
    pub fn value(&self, name: &str) -> Result<&[u8]> {
        self.names
            .iter()
            .position(|n| *n == name)
            .map(|i| self.raw[i].as_ref())
            .ok_or_else(|| Error::State("portable completed observation absent".into()))
    }
}
async fn slot(
    service: &Arc<ResidentService>,
    cancelled: Option<&AtomicBool>,
) -> Result<crate::physical_queue::ActiveSlot> {
    cancel(cancelled)?;
    let selected = service
        .queue()
        .submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 1,
            }],
            None,
        )?
        .wait()
        .await?;
    cancel(cancelled)?;
    Ok(selected)
}

fn validate(body: &TypedBody, raw: &[Bytes]) -> Result<()> {
    let observations = body.observations();
    if raw.len() != observations.len()
        || raw
            .iter()
            .zip(&observations)
            .any(|(r, (_, l))| r.len() != l.bytes())
    {
        return Err(Error::State(
            "portable typed complete observation extents".into(),
        ));
    }
    for (bytes, (name, _)) in raw.iter().zip(&observations) {
        if matches!(*name, "status" | "projection_status") {
            if bytes.iter().any(|b| *b != 0) {
                return Err(Error::State(format!("portable typed {name} refused")));
            }
        } else if *name == "rms_packed" {
            if bytes
                .chunks_exact(2)
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
            {
                return Err(Error::State("portable packed RMS finite words".into()));
            }
        } else {
            for word in bytes.chunks_exact(4) {
                let bits = u32::from_le_bytes(
                    word.try_into()
                        .map_err(|_| Error::State("portable F32 observation".into()))?,
                );
                if !f32::from_bits(bits).is_finite()
                    || matches!(*name, "embedding" | "rms_bf16" | "projection_bf16")
                        && bits & 0xffff != 0
                {
                    return Err(Error::State(format!(
                        "portable typed finite/precision stage {name}"
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Qualifies a state-free recorded fragment only. The caller authenticates any
/// independent reference and reads its payload solely in this completed callback.
pub async fn qualify_portable_bf16_slice<F>(
    device: Device,
    source: VerifiedTensorSource,
    slice: RecordedOperatorSlice,
    token: u32,
    fault: SliceFault,
    cancelled: Option<&AtomicBool>,
    validate_expected: F,
) -> Result<SliceOutcome>
where
    F: FnOnce(CompletedSlice<'_>) -> Result<()>,
{
    cancel(cancelled)?;
    crate::backend::validate_provider(&device)?;
    if !matches!(
        crate::backend::Provider::selected(&device)?,
        crate::backend::Provider::Hip | crate::backend::Provider::Vulkan
    ) {
        return Err(Error::Unsupported(
            "portable typed slice requires explicit HIP/Vulkan".into(),
        ));
    }
    Shape::bind(&slice, &source, token)?;
    if source.metadata_reserved_bytes() > 64 * 1024 * 1024
        || source.staging_chunk_bytes() > 1024 * 1024
    {
        return Err(Error::Limit(
            "portable typed source metadata/window envelope".into(),
        ));
    }
    let service = ResidentService::new(
        device.clone(),
        source.source_identity(),
        ResourceOwner::new("portable-cpu-bf16", "typed-source-fragment")?,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    )?;
    let key = service.physical_key();
    let inputs = TypedBody::prepare_inputs(
        service.clone(),
        source.clone(),
        Arc::new(slice),
        token,
        cancelled,
    )?;
    #[cfg(test)]
    super::tests::cold_physical_guards(&device, &inputs, &service).await?;
    let mut queue = slot(&service, cancelled).await?;
    cancel(cancelled)?;
    queue.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let pending = TypedBody::prepare(inputs, &mut init, cancelled, fault).await?;
    drop(init.complete_sync().await?);
    queue.finish()?;
    let (mut program, body) = pending.bind()?;
    let readbacks = body.readbacks()?;
    let outputs = body.outputs();
    let readback_bytes =
        readbacks
            .iter()
            .zip(body.observations())
            .try_fold(0usize, |a, (_, (_, l))| {
                a.checked_add(l.bytes())
                    .ok_or_else(|| Error::Limit("portable typed readbacks overflow".into()))
            })?;
    let readback_charge = source.reserve_workspace(2 * readback_bytes + 64 * 1024)?;
    let mut queue = slot(&service, cancelled).await?;
    cancel(cancelled)?;
    queue.begin_submission()?;
    let mut flight = service.submit(0, body.accesses())?;
    let submitted = program.launch_typed_slice(&mut flight, cancelled);
    if fault != SliceFault::None {
        if !matches!(submitted,Err(Error::State(ref e)) if e=="deliberate portable typed after actual node submission")
        {
            return Err(Error::State(
                "portable typed missed selected actual fault".into(),
            ));
        }
        let weak = Arc::downgrade(&body);
        drop(flight);
        drop(queue);
        drop((readbacks, outputs, program, body, source, readback_charge));
        let retained = weak.upgrade().ok_or_else(|| {
            Error::State("portable typed quarantine lost original source/program graph".into())
        })?;
        let outputs_unpromoted = retained.outputs().iter().all(|l| !l.is_initialized());
        let physical = ResourceArbiter::process_shared().snapshot(&key)?;
        if service.ensure_usable().is_ok()
            || !physical.fenced
            || service.pending_events()? == 0
            || service.resident_bytes()? == 0
            || !outputs_unpromoted
        {
            return Err(Error::State("portable typed unknown disposition".into()));
        }
        return Ok(SliceOutcome {
            physical_key: key,
            completed_nodes: None,
            submitted_nodes: retained.submitted_nodes.load(Ordering::Acquire),
            math_enqueues: retained.commands.load(Ordering::Acquire),
            dispatches: None,
            readbacks: 0,
            callbacks: 0,
            ready: false,
            source_graph_retained: true,
            outputs_unpromoted,
            resident_bytes: service.resident_bytes()?,
            pending_events: service.pending_events()?,
            fenced: physical.fenced,
            physical_reserved: physical.reserved,
        });
    }
    let submitted = submitted?;
    if !submitted.proposals.is_empty() || submitted.logits.is_some() {
        return Err(Error::State(
            "portable state-free fragment produced state/logits".into(),
        ));
    }
    let callbacks = AtomicUsize::new(0);
    let (ready, raw) = flight
        .complete_with_readback(&outputs, &readbacks, |raw| {
            validate(&body, raw)?;
            callbacks.fetch_add(1, Ordering::AcqRel);
            validate_expected(CompletedSlice {
                record: &body.slice,
                token: body.token,
                names: body.observations().iter().map(|(n, _)| *n).collect(),
                raw,
            })
        })
        .await?;
    queue.finish()?;
    body.require_finished()?;
    let mut result = SliceOutcome {
        physical_key: key.clone(),
        completed_nodes: Some(body.next.load(Ordering::Acquire)),
        submitted_nodes: body.submitted_nodes.load(Ordering::Acquire),
        math_enqueues: body.commands.load(Ordering::Acquire),
        dispatches: Some(submitted.dispatches),
        readbacks: raw.len(),
        callbacks: callbacks.load(Ordering::Acquire),
        ready: true,
        source_graph_retained: false,
        outputs_unpromoted: false,
        resident_bytes: 0,
        pending_events: 0,
        fenced: false,
        physical_reserved: 0,
    };
    drop((
        ready,
        raw,
        readbacks,
        outputs,
        program,
        body,
        source,
        readback_charge,
    ));
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "portable typed completed owners remain".into(),
        ));
    }
    let weak = Arc::downgrade(&service);
    drop(service);
    if weak.upgrade().is_some() {
        return Err(Error::State("portable typed service remains".into()));
    }
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    result.fenced = physical.fenced;
    result.physical_reserved = physical.reserved;
    if result.fenced || result.physical_reserved != 0 {
        return Err(Error::State("portable typed physical final0".into()));
    }
    Ok(result)
}

/// Actual typed work through one provider must fence a preexisting peer on
/// the same reported PCI. Each provider retains only its own original handles.
pub async fn qualify_portable_bf16_slice_cross_provider(
    device: Device,
    peer: Device,
    source: VerifiedTensorSource,
    slice: RecordedOperatorSlice,
    token: u32,
    fault: SliceFault,
) -> Result<SliceOutcome> {
    crate::backend::validate_provider(&device)?;
    crate::backend::validate_provider(&peer)?;
    use crate::backend::Provider;
    let primary_provider = Provider::selected(&device)?;
    let peer_provider = Provider::selected(&peer)?;
    if fault == SliceFault::None
        || !matches!(primary_provider, Provider::Hip | Provider::Vulkan)
        || !matches!(peer_provider, Provider::Hip | Provider::Vulkan)
        || primary_provider == peer_provider
    {
        return Err(Error::Unsupported(
            "typed fault requires distinct HIP/Vulkan providers".into(),
        ));
    }
    let primary_client = device.client();
    let peer_client = peer.client();
    let pci = primary_client
        .properties()
        .identity
        .physical
        .as_ref()
        .and_then(|p| p.pci_address);
    if pci.is_none()
        || pci
            != peer_client
                .properties()
                .identity
                .physical
                .as_ref()
                .and_then(|p| p.pci_address)
    {
        return Err(Error::State(
            "typed cross-provider actual same PCI required".into(),
        ));
    }
    let peer = ResidentService::new(
        peer,
        source.source_identity(),
        ResourceOwner::new("portable-typed-peer", "quarantine")?,
        64 * 1024 * 1024,
        256 * 1024 * 1024,
    )?;
    let peer_key = peer.physical_key();
    let mut queue = slot(&peer, None).await?;
    queue.begin_submission()?;
    let mut init = peer.begin_initialization(0)?;
    let peer_lease = init.upload_bytes(&[1, 2, 3, 4], BufferRole::Activation)?;
    drop(init.complete_sync().await?);
    queue.finish()?;
    let outcome = qualify_portable_bf16_slice(device, source, slice, token, fault, None, |_| {
        Err(Error::State("fault called expected validator".into()))
    })
    .await?;
    if outcome.physical_key != peer_key
        || outcome.physical_reserved != 128 * 1024 * 1024
        || peer.ensure_usable().is_ok()
        || !Arc::ptr_eq(peer.queue(), &crate::physical_queue::queue_for(peer_key)?)
    {
        return Err(Error::State(
            "typed peer physical envelope/queue/fence mismatch".into(),
        ));
    }
    let weak = Arc::downgrade(&peer);
    drop((peer_lease, peer));
    let retained = weak
        .upgrade()
        .ok_or_else(|| Error::State("typed fenced peer original owner lost".into()))?;
    if crate::physical_quarantine::quarantined_handles_for(retained.client())? == 0 {
        return Err(Error::State(
            "typed fenced peer original handles lost".into(),
        ));
    }
    Ok(outcome)
}
