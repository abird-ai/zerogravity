//! Serial original W/source comparisons and original guarded native retirement.
use super::*;
use crate::resident_leases::{ResidentReadback, TIED_BYTES, TIED_WINDOW_BYTES, TIED_WINDOWS};
use zerogravity_core::device_events::DeviceAccess;
use zerogravity_model_qwen35::original_binding::QwenMatrixRole;
fn immutable(cold: &ColdModel) -> Vec<(ResidentLease, &[u8])> {
    let mut all = Vec::new();
    for p in cold.parameters.values() {
        all.extend(p.immutable_weights());
    }
    for p in cold.attention_parameters.values() {
        all.extend(p.immutable_weights());
    }
    all.push((cold.rotary.lease().clone(), cold.rotary.source_bytes()));
    all.extend(cold.native.immutable_controls());
    all
}
pub(super) async fn verify(
    cold: Arc<ColdModel>,
    cancelled: Option<&AtomicBool>,
) -> Result<(usize, usize)> {
    not_cancelled(cancelled)?;
    let weights = cold.native.weights();
    let w = weights
        .verify_source_windows(cold.clone(), cancelled)
        .await?
        .0;
    let source = cold.source.source();
    let tied = &cold.tied;
    let service = tied.lease().service();
    if !Arc::ptr_eq(tied.source(), source)
        || tied.copy_counts()? != (TIED_WINDOWS, TIED_WINDOWS)
        || tied.lease().bytes() != TIED_BYTES
    {
        return Err(Error::State(
            "canonical complete tied source upload coverage".into(),
        ));
    }
    let d = source.matrix_descriptor(QwenMatrixRole::Output)?;
    source.validate_matrix(QwenMatrixRole::Output, &d)?;
    source.validate_matrix(QwenMatrixRole::Embedding, &d)?;
    let m = source.source().matrix(&d.name, d.axes)?;
    if (m.rows(), m.columns()) != (248320, 1024) {
        return Err(Error::State("canonical tied geometry".into()));
    }
    let admission = Arc::new(source.source().reserve_workspace(32 * 1024 * 1024)?);
    for i in 0..TIED_WINDOWS {
        not_cancelled(cancelled)?;
        let first = i * 512;
        let window = Arc::new(match cancelled {
            Some(f) => source
                .source()
                .read_rows_cancellable(&m, first..first + 512, f)?,
            None => source.source().read_rows(&m, first..first + 512)?,
        });
        if window.as_slice().len() != TIED_WINDOW_BYTES {
            return Err(Error::State("canonical tied complete source window".into()));
        }
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
        let mut flight = service.submit(0, &[(tied.lease().clone(), DeviceAccess::Read)])?;
        flight.retain_completion_owner(cold.clone())?;
        flight.retain_completion_owner(admission.clone())?;
        flight.retain_completion_owner(window.clone())?;
        let r = [tied
            .lease()
            .readback(i * TIED_WINDOW_BYTES, TIED_WINDOW_BYTES)?];
        drop(
            flight
                .complete_read_only_readback(&r, |r| {
                    if r.len() != 1 || &r[0][..] != window.as_slice() {
                        return Err(Error::State("canonical full tied table changed".into()));
                    }
                    Ok(())
                })
                .await?,
        );
        slot.finish()?;
    }
    let all = immutable(&cold);
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
                        "canonical immutable source parameters/controls changed".into(),
                    ));
                }
                Ok(())
            })
            .await?,
    );
    slot.finish()?;
    Ok((w, TIED_WINDOWS))
}
pub(super) async fn retire(cold: Arc<ColdModel>, cancelled: Option<&AtomicBool>) -> Result<()> {
    not_cancelled(cancelled)?;
    let all = immutable(&cold);
    let mut accesses = cold.native.weights().accesses();
    accesses.push((cold.tied.lease().clone(), DeviceAccess::Read));
    accesses.extend(all.iter().map(|(l, _)| (l.clone(), DeviceAccess::Read)));
    let service = cold.tied.lease().service();
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
    // SAFETY: no live sessions/proposals and original quiescence barrier completed.
    // This armed initialized-W Read flight keeps every native/source owner alive
    // through any partial unload failure; completion is never manufactured.
    unsafe { cold.native.retire() }?;
    let reads = all
        .iter()
        .map(|(l, _)| l.readback(0, l.bytes()))
        .collect::<Result<Vec<_>>>()?;
    drop(
        flight
            .complete_read_only_readback(&reads, |r| {
                if r.len() != all.len() || r.iter().zip(&all).any(|(a, (_, b))| &a[..] != *b) {
                    return Err(Error::State(
                        "canonical source changed at native retirement".into(),
                    ));
                }
                Ok(())
            })
            .await?,
    );
    slot.finish()?;
    Ok(())
}
