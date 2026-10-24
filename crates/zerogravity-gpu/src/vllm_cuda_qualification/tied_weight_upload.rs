//! Full original-table transfer/coverage proof, without a head/native engine.
use crate::physical_quarantine::{quarantined_completion_owners_for, quarantined_handles_for};
use crate::resident_leases::{
    ResidentService, TIED_BYTES, TIED_ROW_BYTES, TIED_WINDOW_BYTES, TIED_WINDOWS,
    TiedUploadAttempt, TiedUploadCut,
};
use cubecl::Device;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::OriginalWindowModelSource;

// Full485MiB source +27MiB bounded local headroom. No native64MiB is used here.
const PROVIDER_CAP: usize = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiedWeightUploadMode {
    Complete,
    Pressure,
    CancelBeforeFirstCopy,
    UnknownAfterMiddleCopy,
    CancelAfterMiddleCopy,
}
#[derive(Clone, Copy, Debug)]
pub struct TiedWeightUploadOutcome {
    pub mode: TiedWeightUploadMode,
    pub status: &'static str,
    pub copies_enqueued: usize,
    pub copy_checkpoints: usize,
    pub initialized: bool,
    pub compared_bytes: usize,
    pub readback_flights: usize,
    pub resident_bytes: usize,
    pub pending_events: usize,
    pub uploaded_bytes: u64,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub source_retained: bool,
    pub upload_owner_retained: bool,
    pub current_window_retained: bool,
    pub service_retained: bool,
    pub current_window_staging_bytes: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub host_final_zero_established: bool,
}
fn cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State(
            "original tied upload qualification cancelled".into(),
        ));
    }
    Ok(())
}

/// CPU-authenticated source is supplied by the caller. A true flag preempts
/// geometry, host workspace, Device/client lookup and actual CUDA allocation.
/// This qualifies the full original upload only, not logits or StateRoot.
pub async fn qualify_tied_weight_upload(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    device_cap: u64,
    mode: TiedWeightUploadMode,
    cancel: Option<&AtomicBool>,
) -> Result<TiedWeightUploadOutcome> {
    cancelled(cancel)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-original-tied-table", "finite-source-upload")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    if mode == TiedWeightUploadMode::Pressure {
        if service.resident_bytes()? != 0
            || service.pending_events()? != 0
            || service.uploaded_value_bytes() != 0
        {
            return Err(Error::State(
                "tied upload pressure service not fresh".into(),
            ));
        }
        if !matches!(
            service.allocate_uninitialized(PROVIDER_CAP + 1, BufferRole::ImmutableWeights),
            Err(Error::Limit(_))
        ) {
            return Err(Error::State(
                "original tied upload ledger failed to refuse pressure".into(),
            ));
        }
        let resident_bytes = service.resident_bytes()?;
        let pending_events = service.pending_events()?;
        let uploaded_bytes = service.uploaded_value_bytes();
        let handles = quarantined_handles_for(service.client())?;
        let owners = quarantined_completion_owners_for(service.client())?;
        service.ensure_usable()?;
        if resident_bytes != 0
            || pending_events != 0
            || uploaded_bytes != 0
            || handles != 0
            || owners != 0
        {
            return Err(Error::State(
                "tied upload pressure created buffers/copies/events/quarantine".into(),
            ));
        }
        drop(source);
        drop(service);
        let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
        if snapshot.reserved != 0
            || snapshot.reservations != 0
            || snapshot.fenced
            || !snapshot.per_class.is_empty()
        {
            return Err(Error::State(
                "tied upload pressure retained physical envelope".into(),
            ));
        }
        return Ok(TiedWeightUploadOutcome {
            mode,
            status: "ADMISSION_REFUSED_BEFORE_SOURCE_WINDOW_H2D",
            copies_enqueued: 0,
            copy_checkpoints: 0,
            initialized: false,
            compared_bytes: 0,
            readback_flights: 0,
            resident_bytes,
            pending_events,
            uploaded_bytes,
            quarantined_handles: handles,
            quarantined_completion_owners: owners,
            source_retained: false,
            upload_owner_retained: false,
            current_window_retained: false,
            service_retained: false,
            current_window_staging_bytes: 0,
            physical_reserved_bytes: snapshot.reserved,
            physical_fenced: snapshot.fenced,
            host_final_zero_established: false,
        });
    }
    cancelled(cancel)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::ImmutableWeights,
            bytes: TIED_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    cancelled(cancel)?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    let attempt = if mode == TiedWeightUploadMode::Complete {
        TiedUploadAttempt::Complete(
            init.upload_original_tied_bf16(source.clone(), cancel)
                .await?,
        )
    } else {
        let cut = match mode {
            TiedWeightUploadMode::CancelBeforeFirstCopy => TiedUploadCut::CancelBeforeFirstEnqueue,
            TiedWeightUploadMode::UnknownAfterMiddleCopy => {
                TiedUploadCut::UnknownAfterMiddleEnqueue
            }
            TiedWeightUploadMode::CancelAfterMiddleCopy => TiedUploadCut::CancelAfterMiddleEnqueue,
            TiedWeightUploadMode::Complete | TiedWeightUploadMode::Pressure => unreachable!(),
        };
        init.upload_tied_bf16_inner(source.clone(), cut, cancel)
            .await?
    };
    let pending = match attempt {
        TiedUploadAttempt::Complete(pending) => pending,
        TiedUploadAttempt::Cut(witness) => {
            // No barrier/readback/Ready/unload follows this actual abandoned
            // enqueue. These are ALL local strong owners; witness is Weak-only.
            drop(init);
            drop(slot);
            drop(source);
            drop(service);
            let retained = witness.inspect()?;
            if retained.reserved != PROVIDER_CAP as u64 || !retained.fenced || retained.initialized
            {
                return Err(Error::State(
                    "tied upload unknown envelope/readiness differs".into(),
                ));
            }
            return Ok(TiedWeightUploadOutcome {
                mode,
                status: match mode {
                    TiedWeightUploadMode::UnknownAfterMiddleCopy => "UNKNOWN_RETAINED",
                    TiedWeightUploadMode::CancelAfterMiddleCopy => "CANCELLED_UNKNOWN_RETAINED",
                    TiedWeightUploadMode::CancelBeforeFirstCopy => "CANCELLED_BEFORE_H2D_RETAINED",
                    _ => unreachable!(),
                },
                copies_enqueued: retained.enqueued,
                copy_checkpoints: retained.checkpoints,
                initialized: retained.initialized,
                compared_bytes: 0,
                readback_flights: 0,
                resident_bytes: retained.resident_bytes,
                pending_events: retained.pending_events,
                uploaded_bytes: retained.uploaded_bytes,
                quarantined_handles: retained.handles,
                quarantined_completion_owners: retained.completion_owners,
                source_retained: retained.source,
                upload_owner_retained: retained.owner,
                current_window_retained: retained.window,
                service_retained: retained.service,
                current_window_staging_bytes: retained.staging_bytes,
                physical_reserved_bytes: retained.reserved,
                physical_fenced: retained.fenced,
                host_final_zero_established: false,
            });
        }
    };
    drop(init.complete_sync().await?);
    slot.finish()?;
    let weight = pending.bind_initialized()?;
    let (copies_enqueued, copy_checkpoints) = weight.copy_counts()?;
    let initialized = weight.lease().is_initialized();
    if !Arc::ptr_eq(weight.source(), &source) || !Arc::ptr_eq(weight.lease().service(), &service) {
        return Err(Error::State(
            "initialized tied upload original source/service differs".into(),
        ));
    }
    // One bounded D2H at a time, independently compared with freshly verified
    // source bytes. The admission owner also rides every original read flight.
    let readback_admission = Arc::new(source.source().reserve_workspace(32 * 1024 * 1024)?);
    let descriptor = source
        .matrix_descriptor(zerogravity_model_qwen35::original_binding::QwenMatrixRole::Output)?;
    let matrix = source.source().matrix(&descriptor.name, descriptor.axes)?;
    let mut compared = 0usize;
    for index in 0..TIED_WINDOWS {
        cancelled(cancel)?;
        let first = index * (TIED_WINDOW_BYTES / TIED_ROW_BYTES);
        let end = first + TIED_WINDOW_BYTES / TIED_ROW_BYTES;
        let source_window = Arc::new(match cancel {
            Some(flag) => source
                .source()
                .read_rows_cancellable(&matrix, first..end, flag)?,
            None => source.source().read_rows(&matrix, first..end)?,
        });
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::ImmutableWeights,
                bytes: TIED_BYTES as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        cancelled(cancel)?;
        slot.begin_submission()?;
        let mut flight = service.submit(
            1 + index as u64,
            &[(weight.lease().clone(), DeviceAccess::Read)],
        )?;
        flight.retain_completion_owner(weight.clone())?;
        flight.retain_completion_owner(readback_admission.clone())?;
        flight.retain_completion_owner(source_window.clone())?;
        let raw = flight
            .complete_read_only_readback(
                &[weight
                    .lease()
                    .readback(index * TIED_WINDOW_BYTES, TIED_WINDOW_BYTES)?],
                |raw| {
                    if raw.len() != 1 || &raw[0][..] != source_window.as_slice() {
                        return Err(Error::State(
                            "full tied-table physical source comparison differs".into(),
                        ));
                    }
                    Ok(())
                },
            )
            .await?;
        drop(raw);
        slot.finish()?;
        compared = compared
            .checked_add(TIED_WINDOW_BYTES)
            .ok_or_else(|| Error::Limit("tied compared bytes overflow".into()))?;
    }
    if compared != TIED_BYTES
        || service.uploaded_value_bytes() != TIED_BYTES as u64
        || service.pending_events()? != 0
        || service.resident_bytes()? != TIED_BYTES
    {
        return Err(Error::State(
            "full tied upload/readback coverage differs".into(),
        ));
    }
    let handles = quarantined_handles_for(service.client())?;
    let owners = quarantined_completion_owners_for(service.client())?;
    if handles != 0 || owners != 0 {
        return Err(Error::State(
            "completed tied upload entered quarantine".into(),
        ));
    }
    drop(readback_admission);
    drop(weight);
    if service.resident_bytes()? != 0 {
        return Err(Error::State("tied upload retained original lease".into()));
    }
    drop(source);
    drop(service);
    let snapshot = ResourceArbiter::process_shared().snapshot(&key)?;
    if snapshot.reserved != 0
        || snapshot.reservations != 0
        || snapshot.fenced
        || !snapshot.per_class.is_empty()
    {
        return Err(Error::State(
            "completed tied upload retained physical envelope".into(),
        ));
    }
    Ok(TiedWeightUploadOutcome {
        mode,
        status: "FULL_ORIGINAL_TIED_UPLOAD_COMPARED_NO_HEAD_QUALIFICATION",
        copies_enqueued,
        copy_checkpoints,
        initialized,
        compared_bytes: compared,
        readback_flights: TIED_WINDOWS,
        resident_bytes: 0,
        pending_events: 0,
        uploaded_bytes: TIED_BYTES as u64,
        quarantined_handles: handles,
        quarantined_completion_owners: owners,
        source_retained: false,
        upload_owner_retained: false,
        current_window_retained: false,
        service_retained: false,
        current_window_staging_bytes: 0,
        physical_reserved_bytes: snapshot.reserved,
        physical_fenced: snapshot.fenced,
        host_final_zero_established: false,
    })
}
