//! Serial original W/source comparisons and original guarded native retirement.
use super::*;
use crate::resident_leases::ResidentReadback;
use zerogravity_core::device_events::DeviceAccess;

fn immutable(cold: &ColdModel) -> Result<Vec<(ResidentLease, &[u8])>> {
    let mut all = Vec::new();
    for p in cold.parameters.values() {
        all.extend(p.immutable_weights());
    }
    for values in &cold.values {
        all.extend(values.immutable_rms()?);
    }
    for workspace in &cold.private {
        all.extend(workspace.immutable_controls());
    }
    all.push((cold.rotary.lease().clone(), cold.rotary.source_bytes()));
    all.extend(cold.native.immutable_controls());
    Ok(all)
}
pub(super) async fn verify(
    cold: Arc<ColdModel>,
    cancelled: Option<&AtomicBool>,
) -> Result<(usize, usize, usize)> {
    not_cancelled(cancelled)?;
    let weights = cold.native.weights();
    let w = weights
        .verify_source_windows(cold.clone(), cancelled)
        .await?
        .0;
    let service = cold.values[0].body.values().service();
    let source = &cold.source;
    let admission = Arc::new(source.reserve_workspace(32 * 1024 * 1024)?);
    let all = immutable(&cold)?;
    let reads = all
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<ResidentReadback>>>()?;
    let accesses = all
        .iter()
        .map(|(l, _)| (l.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: 32 * 1024 * 1024,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(cold.clone())?;
    flight.retain_completion_owner(admission)?;
    drop(
        flight
            .complete_read_only_readback(&reads, |r| {
                if r.len() != all.len() || r.iter().zip(&all).any(|(a, (_, b))| &a[..] != *b) {
                    return Err(Error::State(
                        "short HF immutable source parameters/controls changed".into(),
                    ));
                }
                Ok(())
            })
            .await?,
    );
    slot.finish()?;
    // Streaming keeps its authenticated current-call source window. Resident
    // mode independently compares every immutable tied row before/after unload.
    let tied_windows = match cold.native.head().resident_weight() {
        Some(tied) => tied.verify_source_windows(cold.clone(), cancelled).await?,
        None => 0,
    };
    Ok((w, 0, tied_windows))
}
pub(super) async fn retire(
    cold: Arc<ColdModel>,
    cohort: Option<Arc<cohort::ColdCohort>>,
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    let all = immutable(&cold)?;
    let mut accesses = cold.native.weights().accesses();
    if let Some(tied) = cold.native.head().resident_weight() {
        accesses.push((tied.lease().clone(), DeviceAccess::Read));
    }

    accesses.extend(all.iter().map(|(l, _)| (l.clone(), DeviceAccess::Read)));
    let service = cold.values[0].body.values().service();
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 1,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(cold.clone())?;
    if let Some(cohort) = cohort {
        flight.retain_completion_owner(cohort.clone())?;
        // SAFETY: no live sessions/proposals and original quiescence barrier
        // completed. This SAME armed initialized-W Read flight retains all B4
        // owners through partial failure. Shared set retirement still requires
        // every B4 and N1 record to retire through the original actor.
        unsafe { cohort.retire_after_physical_completion() }
            .map_err(|e| Error::State(format!("short HF B4 native retirement: {e}")))?;
    }
    // SAFETY: same original barrier/armed flight; all B4 records retired above.
    // Partial unload still fences/quarantines the complete original owner graph.
    unsafe { cold.native.retire_after_physical_completion() }
        .map_err(|e| Error::State(format!("short HF original native retirement: {e}")))?;
    let reads = all
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    drop(
        flight
            .complete_read_only_readback(&reads, |r| {
                if r.len() != all.len() || r.iter().zip(&all).any(|(a, (_, b))| &a[..] != *b) {
                    return Err(Error::State(
                        "short HF source changed at native retirement".into(),
                    ));
                }
                Ok(())
            })
            .await?,
    );
    slot.finish()?;
    Ok(())
}
