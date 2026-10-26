//! Same real preparation/flight for compare and diagnostic cuts.
use super::resident::Prepared;
use super::*;
use cubecl::Device;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};

pub async fn qualify_solve2b_owned_leaves(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: Solve2bFixture,
    device_cap: u64,
    validate: impl FnMut(Solve2bReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<Solve2bReceipt> {
    let mut probe = None;
    run(
        device,
        model,
        fixture,
        device_cap,
        RunHooks {
            fault: None,
            probe: &mut probe,
            validate,
        },
        cancelled,
    )
    .await
}
pub(super) struct RunHooks<'a, F> {
    pub(super) fault: Option<Solve2bFault>,
    pub(super) probe: &'a mut Option<super::faults::Probe>,
    pub(super) validate: F,
}
pub(super) async fn run(
    device: Device,
    model: Arc<OriginalWindowModelSource>,
    fixture: Solve2bFixture,
    device_cap: u64,
    hooks: RunHooks<'_, impl FnMut(Solve2bReadback<'_>) -> Result<()>>,
    cancelled: Option<&AtomicBool>,
) -> Result<Solve2bReceipt> {
    let RunHooks {
        fault,
        probe,
        mut validate,
    } = hooks;
    not_cancelled(cancelled)?;
    // Exact held source/authentication/fixture admission before a CUDA client.
    let owner = LeafOwner::new(model, fixture)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release CUDA required for original2B/T64 solve leaf".into(),
        ));
    }
    let service = ResidentService::new_pinned_cuda(
        device,
        owner.bound.source().source_identity(),
        ResourceOwner::new("qwen2b-source", "finite-owned-solve4-recompute3-leaf")?,
        PROVIDER,
        device_cap,
    )?;
    let key = service.physical_key();
    let requests = [AllocationRequest {
        kind: AllocationClass::Other,
        bytes: PROVIDER as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(1)?;
    init.retain_completion_owner(owner.clone())?;
    let prepared = Prepared::prepare(service.clone(), owner.clone(), &mut init, cancelled)?;
    drop(init.complete_sync().await?);
    slot.finish()?;
    if prepared.leases().len() != 13 || service.uploaded_value_bytes() != INPUT_BYTES as u64 {
        return Err(Error::State(
            "solve2b actual13 allocations/seven complete H2D inputs".into(),
        ));
    }
    if fault.is_some() {
        *probe = Some(super::faults::Probe::new(&prepared)?);
    }
    for index in 0..2 {
        not_cancelled(cancelled)?;
        let job = &prepared.jobs[index];
        let ticket =
            service
                .queue()
                .submit(service.budget(), service.owner().clone(), &requests, None)?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        not_cancelled(cancelled)?;
        let mut flight = service.submit(index as u64 + 1, &job.accesses)?;
        // Positive and cuts retain the SAME actual full prepared/source/code graph.
        flight.retain_completion_owner(prepared.clone())?;
        let dispatches = prepared.through_solve(index, cancelled)?;
        if let Some(fault) = fault {
            if let Some(probe) = probe {
                probe.dispatches = dispatches;
            }
            return match fault {
                Solve2bFault::UnknownAfterSolve => Err(Error::State(super::faults::UNKNOWN.into())),
                Solve2bFault::CancelBeforeRecompute => {
                    let flag = AtomicBool::new(true);
                    not_cancelled(Some(&flag))?;
                    unreachable!("true cancellation must refuse")
                }
                Solve2bFault::Pressure => {
                    Err(Error::State("solve2b pressure cannot reach enqueue".into()))
                }
            };
        }
        prepared.recompute(index, cancelled)?;
        let (ready, raw) = flight
            .complete_with_readback(&job.outputs, &job.reads, |raw| {
                let actual = prepared.validate(index, raw)?;
                validate(actual)
            })
            .await?;
        slot.finish()?;
        drop((ready, raw));
    }
    let uploaded = service.uploaded_value_bytes();
    drop((prepared, owner));
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "solve2b final actual allocations/events did not retire".into(),
        ));
    }
    drop(service);
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "solve2b final original physical envelope not0/unfenced".into(),
        ));
    }
    Ok(Solve2bReceipt {
        completed_cohorts: 2,
        arithmetic: OWNED,
        original_reference: REFERENCE,
        original_selected_tuple_equals_owned: false,
        zero_dispatches: 2,
        solve_dispatches: 2,
        recompute_dispatches: 2,
        compared_output_bytes: 2 * OUTPUT_BYTES,
        immutable_operand_bytes_checked: 2 * INPUT_BYTES,
        uploaded_value_bytes: uploaded,
        final_resident_bytes: 0,
        final_physical_reserved_bytes: 0,
        parent_fullmodel_accepted: false,
    })
}
