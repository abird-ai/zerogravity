//! One model-free source/transport/lifecycle probe; no numerical eligibility.
use super::*;
use crate::physical_queue::ActiveSlot;
use zerogravity_core::{
    resource_arbiter::{AllocationClass, AllocationRequest, ResourceArbiter},
    session_root::SessionAuthority,
};

const PROVIDER: usize = 64 * 1024 * 1024;
const DEVICE: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixTransportMode {
    Healthy,
    UnknownFirstUpload,
    UnknownLastUpload,
    UnknownRows,
    #[cfg(test)]
    InvalidDeviceId,
}

#[derive(Debug)]
pub struct MatrixTransportOutcome {
    pub physical_key: PhysicalDeviceKey,
    pub upload_commands: usize,
    pub upload_checkpoints: usize,
    pub id_uploads: usize,
    pub integer_dispatches: usize,
    pub compared_bytes: usize,
    /// Selected-row phase only; upload/table verification is separately scoped.
    pub row_readbacks: usize,
    pub row_callbacks: usize,
    pub row_ready: usize,
    pub published_roots: usize,
    pub source_retained: bool,
    pub transport_retained: bool,
    pub service_usable: bool,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub physical_reserved: u64,
    pub fenced: bool,
}

async fn slot(service: &Arc<ResidentService>) -> Result<ActiveSlot> {
    service
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
        .await
}

/// Explicit model-free qualification surface. `source` is a genuine sealed
/// SourceWeights owner, never a caller-provided raw table/expected operand.
/// Unknown modes require a fresh owned process and intentionally retain charges.
pub async fn qualify_bf16_matrix_transport(
    device: cubecl::Device,
    source: VerifiedTensorSource,
    tensor: &str,
    ids: Arc<[u32]>,
    mode: MatrixTransportMode,
) -> Result<MatrixTransportOutcome> {
    crate::backend::validate_provider(&device)?;
    if !matches!(
        crate::backend::Provider::selected(&device)?,
        crate::backend::Provider::Hip | crate::backend::Provider::Vulkan
    ) {
        return Err(Error::Unsupported(
            "portable matrix probe explicitly selects HIP/Vulkan".into(),
        ));
    }
    let matrix = source.matrix(tensor, MatrixAxes::OutputInput)?;
    let geometry = ResidentMatrix::authenticate(&source, &matrix)?;
    geometry.output_bytes(&ids)?;
    // Bounds for this diagnostic cohort,not a production matrix-size policy.
    if geometry.bytes() > 3 * layout::WINDOW || source.staging_chunk_bytes() > layout::WINDOW {
        return Err(Error::Limit(
            "portable matrix probe source/diagnostic bound".into(),
        ));
    }
    let service = ResidentService::new(
        device,
        source.source_identity(),
        ResourceOwner::new("verified-matrix-transport", "probe")?,
        PROVIDER,
        DEVICE,
    )?;
    require_portable(&service, geometry)?;
    let key = service.physical_key();
    let mut queue = slot(&service).await?;
    queue.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let cut = match mode {
        MatrixTransportMode::UnknownFirstUpload => upload::Cut::First,
        MatrixTransportMode::UnknownLastUpload => upload::Cut::Last,
        _ => upload::Cut::Never,
    };
    let pending = if cut == upload::Cut::Never {
        init.upload_bf16_matrix(source.clone(), matrix.clone(), None)
            .await?
    } else {
        match init
            .upload_bf16_matrix_cut(source.clone(), matrix.clone(), None, cut)
            .await?
        {
            upload::Attempt::Complete(_) => {
                return Err(Error::State("portable upload missed actual cut".into()));
            }
            upload::Attempt::Cut(witness) => {
                let weak_service = Arc::downgrade(&service);
                drop(init);
                drop(queue);
                drop((source, matrix, service));
                let owner = witness
                    .upgrade()
                    .ok_or_else(|| Error::State("unknown upload lost original owner".into()))?;
                let snapshot = owner.snapshot()?;
                let service = weak_service
                    .upgrade()
                    .ok_or_else(|| Error::State("unknown upload lost service".into()))?;
                if !snapshot.pending
                    || snapshot.initialized
                    || snapshot.source != service.source_identity()
                    || snapshot.commands != snapshot.checkpoints + 1
                {
                    return Err(Error::State(
                        "unknown original matrix upload disposition".into(),
                    ));
                }
                return outcome(
                    &service,
                    [snapshot.commands, snapshot.checkpoints],
                    [0, 0],
                    0,
                    0,
                    true,
                    true,
                );
            }
        }
    };
    let counts = pending.counts()?;
    drop(init.complete_sync().await?);
    let uploaded = pending.bind_initialized()?;
    drop(matrix);
    let matrix = uploaded;
    queue.finish()?;

    // Full table comparison occurs only inside the original completed callback.
    // The authenticated source read is an oracle here,never a dispatch operand.
    let readback_charge = source.reserve_workspace(geometry.bytes())?;
    let mut queue = slot(&service).await?;
    queue.begin_submission()?;
    let mut flight = service.submit(0, &[(matrix.lease().clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(matrix.clone())?;
    let raw = flight
        .complete_read_only_readback(&[matrix.lease().readback(0, geometry.bytes())?], |raw| {
            let expected = source.read_rows(&matrix.matrix, 0..geometry.rows())?;
            if raw.len() != 1 || raw[0][..] != *expected.as_slice() {
                return Err(Error::State(
                    "portable full original matrix bytes differ".into(),
                ));
            }
            Ok(())
        })
        .await?;
    drop((raw, readback_charge));
    queue.finish()?;
    let precancel = AtomicBool::new(true);
    let resident_before = service.resident_bytes()?;
    if !matches!(matrix.prepare_rows(ids.clone(),Some(&precancel)),Err(Error::State(ref s)) if s == "verified matrix rows cancelled")
        || service.resident_bytes()? != resident_before
        || !matches!(
            service.allocate_uninitialized(service.remaining_capacity()? + 1, BufferRole::Scratch),
            Err(Error::Limit(_))
        )
        || service.resident_bytes()? != resident_before
    {
        return Err(Error::State(
            "portable precancel/actual provider pressure allocation changed".into(),
        ));
    }
    let mut sessions = [
        SessionAuthority::new(source.source_identity())?,
        SessionAuthority::new(source.source_identity())?,
    ];
    let before = [sessions[0].binding(), sessions[1].binding()];
    let mut copied = geometry.bytes();
    let mut publications = 0;
    for (index, session) in sessions.iter_mut().enumerate() {
        let selected: Arc<[u32]> = if index == 0 {
            ids.clone()
        } else {
            ids.iter().rev().copied().collect::<Vec<_>>().into()
        };
        let prepared = matrix.prepare_rows(selected.clone(), None)?;
        #[cfg(test)]
        let prepared = if mode == MatrixTransportMode::InvalidDeviceId {
            let mut prepared = prepared;
            Arc::get_mut(&mut prepared)
                .ok_or_else(|| Error::State("invalid-ID probe owner shared".into()))?
                .inject_invalid_first();
            prepared
        } else {
            prepared
        };
        let attempt = session.begin_advance(selected.len() as u64)?;
        let cohort_charge =
            source.reserve_workspace(2 * prepared.output().bytes() + selected.len() * 4)?;
        let mut queue = slot(&service).await?;
        queue.begin_submission()?;
        let mut flight = service.submit(session.binding().root(), &prepared.accesses())?;
        let commands = prepared.enqueue(&mut flight)?;
        if commands != 2 {
            return Err(Error::State("portable actual command accounting".into()));
        }
        if mode == MatrixTransportMode::UnknownRows {
            let weak = Arc::downgrade(&prepared);
            let matrix_weak = Arc::downgrade(&matrix);
            let service_weak = Arc::downgrade(&service);
            drop(flight);
            drop(queue);
            drop((prepared, matrix, source, service, cohort_charge));
            let retained = weak
                .upgrade()
                .ok_or_else(|| Error::State("unknown gather lost control/source owner".into()))?;
            let service = service_weak
                .upgrade()
                .ok_or_else(|| Error::State("unknown gather lost service".into()))?;
            if retained.output().is_initialized()
                || matrix_weak.upgrade().is_none()
                || session.binding() != before[index]
            {
                return Err(Error::State(
                    "unknown gather published/retired original outputs".into(),
                ));
            }
            return outcome(&service, counts, [1, 1], 0, 0, true, true);
        }
        #[cfg(test)]
        if mode == MatrixTransportMode::InvalidDeviceId {
            let checked = AtomicBool::new(false);
            let result = flight
                .complete_with_readback(
                    &[prepared.output().clone()],
                    &prepared.readbacks()?,
                    |raw| {
                        if raw.len() != 2
                            || raw[0].len() != prepared.output().bytes()
                            || raw[1].len() != selected.len() * 4
                            || raw[1][..4] != 1u32.to_le_bytes()
                            || raw[1][4..].iter().any(|b| *b != 0)
                            || raw[0][..geometry.row_bytes()].iter().any(|b| *b != 0)
                        {
                            return Err(Error::State(
                                "invalid device ID did not write complete zero/error row".into(),
                            ));
                        }
                        for (r, id) in selected.iter().enumerate().skip(1) {
                            let expected =
                                source.read_rows(&matrix.matrix, *id as usize..*id as usize + 1)?;
                            if raw[0][r * geometry.row_bytes()..(r + 1) * geometry.row_bytes()]
                                != *expected.as_slice()
                            {
                                return Err(Error::State(
                                    "invalid device ID corrupted another complete row".into(),
                                ));
                            }
                        }
                        checked.store(true, Ordering::Release);
                        Err(Error::State(
                            "deliberate invalid device ID status refusal".into(),
                        ))
                    },
                )
                .await;
            if !matches!(result,Err(Error::State(ref e)) if e == "deliberate invalid device ID status refusal")
                || !checked.load(Ordering::Acquire)
            {
                return Err(Error::State(
                    "invalid-ID actual destination/status check failed".into(),
                ));
            }
            drop(queue);
            if prepared.output().is_initialized() || session.binding() != before[index] {
                return Err(Error::State(
                    "invalid device ID published Ready/root".into(),
                ));
            }
            drop((prepared, matrix, source, cohort_charge));
            let mut result = outcome(
                &service,
                counts,
                [1, 1],
                geometry.output_bytes(&ids)?,
                0,
                true,
                true,
            )?;
            result.row_readbacks = 1;
            result.row_callbacks = 1;
            return Ok(result);
        }
        let late_cancel = AtomicBool::new(false);
        let (ready, raw) = flight
            .complete_with_readback(
                &[prepared.output().clone()],
                &prepared.readbacks()?,
                |raw| {
                    // Status is checked before bytes can authorize private Ready.
                    prepared.validate(raw)?;
                    let mut expected = Vec::with_capacity(prepared.output().bytes());
                    for id in selected.iter() {
                        let row =
                            source.read_rows(&matrix.matrix, *id as usize..*id as usize + 1)?;
                        expected.extend_from_slice(row.as_slice());
                    }
                    if expected.capacity() != prepared.output().bytes()
                        || raw[0][..] != expected[..]
                    {
                        return Err(Error::State(
                            "portable selected original row bytes differ".into(),
                        ));
                    }
                    if index == 0 {
                        late_cancel.store(true, Ordering::Release);
                    }
                    Ok(())
                },
            )
            .await?;
        queue.finish()?;
        copied += prepared.output().bytes();
        if late_cancel.load(Ordering::Acquire) {
            if session.binding() != before[index]
                || !prepared.output().is_initialized()
                || service.ensure_usable().is_err()
            {
                return Err(Error::State(
                    "portable completed cancellation changed root/service".into(),
                ));
            }
            drop((ready, raw, prepared, cohort_charge));
            let retry = matrix.prepare_rows(selected.clone(), None)?;
            let retry_attempt = session.begin_advance(selected.len() as u64)?;
            let charge =
                source.reserve_workspace(2 * retry.output().bytes() + selected.len() * 4)?;
            let mut queue = slot(&service).await?;
            queue.begin_submission()?;
            let mut flight = service.submit(session.binding().root(), &retry.accesses())?;
            retry.enqueue(&mut flight)?;
            let (ready, raw) = flight
                .complete_with_readback(&[retry.output().clone()], &retry.readbacks()?, |raw| {
                    retry.validate(raw)?;
                    let mut expected = Vec::with_capacity(retry.output().bytes());
                    for id in selected.iter() {
                        let row =
                            source.read_rows(&matrix.matrix, *id as usize..*id as usize + 1)?;
                        expected.extend_from_slice(row.as_slice());
                    }
                    if raw[0][..] != expected[..] {
                        return Err(Error::State("portable retry source rows differ".into()));
                    }
                    Ok(())
                })
                .await?;
            queue.finish()?;
            copied += retry.output().bytes();
            service.publish_if_usable(|| {
                session.prepare_publish(retry_attempt)?.commit();
                Ok(())
            })?;
            let committed = session.binding();
            let stale_refused = session.prepare_publish(attempt).is_err();
            if !stale_refused || session.binding() != committed {
                return Err(Error::State(
                    "portable stale completed attempt accepted".into(),
                ));
            }
            drop((ready, raw, retry, charge));
        } else {
            service.publish_if_usable(|| {
                session.prepare_publish(attempt)?.commit();
                Ok(())
            })?;
            drop((ready, raw, prepared, cohort_charge));
        }
        publications += 1;
    }
    drop((matrix, source));
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 || publications != 2 {
        return Err(Error::State(
            "portable healthy transport final resident/pending".into(),
        ));
    }
    let mut result = outcome(&service, counts, [3, 3], copied, publications, false, false)?;
    let weak = Arc::downgrade(&service);
    drop(service);
    if weak.upgrade().is_some() {
        return Err(Error::State("portable healthy service retained".into()));
    }
    result.physical_reserved = ResourceArbiter::process_shared().snapshot(&key)?.reserved;
    if result.physical_reserved != 0 {
        return Err(Error::State("portable healthy physical final0".into()));
    }
    Ok(result)
}

fn outcome(
    service: &Arc<ResidentService>,
    uploads: [usize; 2],
    rows: [usize; 2],
    compared_bytes: usize,
    published_roots: usize,
    source_retained: bool,
    transport_retained: bool,
) -> Result<MatrixTransportOutcome> {
    let physical = ResourceArbiter::process_shared().snapshot(&service.physical_key())?;
    let result = MatrixTransportOutcome {
        physical_key: service.physical_key(),
        upload_commands: uploads[0],
        upload_checkpoints: uploads[1],
        id_uploads: rows[0],
        integer_dispatches: rows[1],
        compared_bytes,
        published_roots,
        source_retained,
        transport_retained,
        service_usable: service.ensure_usable().is_ok(),
        resident_bytes: service.resident_bytes()?,
        pending_events: service.pending_events()?,
        quarantined_handles: crate::physical_quarantine::quarantined_handles_for(service.client())?,
        physical_reserved: physical.reserved,
        fenced: physical.fenced,
        row_readbacks: if published_roots == 0 { 0 } else { rows[1] },
        row_callbacks: if published_roots == 0 { 0 } else { rows[1] },
        row_ready: if published_roots == 0 { 0 } else { rows[1] },
    };
    if transport_retained
        && (result.service_usable
            || !result.fenced
            || result.pending_events == 0
            || result.resident_bytes == 0
            || result.quarantined_handles == 0
            || result.physical_reserved == 0)
    {
        return Err(Error::State(
            "portable unknown lost original fence/quarantine/reservation".into(),
        ));
    }
    Ok(result)
}

/// Same actual PCI/queue/cap across two explicitly selected providers. No raw
/// handle crosses providers; uncertainty through one must fence both.
pub async fn qualify_bf16_matrix_cross_provider(
    device: cubecl::Device,
    peer: cubecl::Device,
    source: VerifiedTensorSource,
    tensor: &str,
    ids: Arc<[u32]>,
) -> Result<MatrixTransportOutcome> {
    crate::backend::validate_provider(&peer)?;
    if crate::backend::Provider::selected(&device)? == crate::backend::Provider::selected(&peer)? {
        return Err(Error::Unsupported(
            "portable cross-provider probe needs distinct providers".into(),
        ));
    }
    let peer = ResidentService::new(
        peer,
        source.source_identity(),
        ResourceOwner::new("verified-matrix-peer", "probe")?,
        PROVIDER,
        DEVICE,
    )?;
    let peer_key = peer.physical_key();
    let mut queue = slot(&peer).await?;
    queue.begin_submission()?;
    let mut init = peer.begin_initialization(0)?;
    let peer_lease = init.upload_bytes(&[1, 2, 3, 4], BufferRole::Activation)?;
    drop(init.complete_sync().await?);
    queue.finish()?;
    let outcome = qualify_bf16_matrix_transport(
        device,
        source,
        tensor,
        ids,
        MatrixTransportMode::UnknownRows,
    )
    .await?;
    if outcome.physical_key != peer_key
        || outcome.physical_reserved != (PROVIDER * 2) as u64
        || !Arc::ptr_eq(
            peer.queue(),
            &crate::physical_queue::queue_for(peer_key.clone())?,
        )
        || peer.ensure_usable().is_ok()
    {
        return Err(Error::State(
            "portable same-PCI peer cap/queue/fence mismatch".into(),
        ));
    }
    let weak = Arc::downgrade(&peer);
    drop((peer_lease, peer));
    let retained = weak
        .upgrade()
        .ok_or_else(|| Error::State("portable fenced peer provider lost".into()))?;
    if crate::physical_quarantine::quarantined_handles_for(retained.client())? == 0 {
        return Err(Error::State("portable fenced peer handles lost".into()));
    }
    Ok(outcome)
}

#[cfg(test)]
mod tests;
