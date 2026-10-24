//! Weak-only diagnostic witness; real original armed flight retains owners.
use super::*;
use crate::resident_inference::native_fragment::{ColdModelSlots, NaturalColdSnapshot};
use crate::resident_program::native_full::FullModelProgram;
use zerogravity_core::device_events::BufferLeaseId;

struct Witness {
    cold: Weak<ColdModelSlots>,
    body: Weak<FullModelProgram>,
    input: Weak<NaturalCallInput>,
    state: Weak<NaturalCallState>,
    source: Weak<OriginalWindowModelSource>,
    service: Weak<ResidentService>,
    snapshot: NaturalColdSnapshot,
    current: Vec<BufferLeaseId>,
    roots: [RootBinding; 2],
    roots_unchanged: bool,
}

/// Reuse the SAME actual pre-H2D cold pressure cut; no natural record is bound.
pub(super) async fn pressure(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
) -> Result<Vllm08NaturalFaultOutcome> {
    use crate::resident_inference::native_fragment::{
        Vllm08ModelFault, qualify_vllm08_model_fault,
    };
    let actual = qualify_vllm08_model_fault(
        device,
        source,
        records,
        inputs,
        rotary,
        Vllm08ModelFault::AdmissionPressure,
        cancelled,
    )
    .await?;
    if actual.status != "REFUSED_BEFORE_ALLOCATION"
        || actual.ready_completions != 0
        || actual.readback_commands != 0
        || actual.pending_events != 0
        || actual.resident_bytes != 0
        || actual.physical_reserved_bytes != 0
        || actual.physical_fenced
    {
        return Err(Error::State(
            "natural shared cold pressure cut did not refuse before physical work".into(),
        ));
    }
    Ok(Vllm08NaturalFaultOutcome {
        fault: Vllm08NaturalFault::AdmissionPressure,
        status: actual.status,
        roots: None,
        completed_phases: 0,
        committed_calls: 0,
        emitted_tokens: 0,
        stale_inputs_refused: 0,
        native_launches: 0,
        convolution_dispatches: 0,
        pending_events: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        all_prepared_owners_retained: false,
        all_future_uninitialized: true,
        prepared_state_buffers: 0,
        retained_plan_values: 0,
        retained_source_weight_groups: 0,
        submitted_current_leases: 0,
        native_image_owners: 0,
        native_head_kernel_owners: 0,
        native_charge_bytes: 0,
        image_admission_bytes: 0,
        resident_bytes: 0,
        physical_reserved_bytes: 0,
        physical_fenced: false,
        source_weight_windows: 0,
        tied_weight_windows: 0,
    })
}

pub(super) async fn qualify(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    execution: Vllm08NaturalFaultExecution<'_>,
) -> Result<Vllm08NaturalFaultOutcome> {
    let Vllm08NaturalFaultExecution {
        rotary,
        fault,
        cancelled,
    } = execution;
    not_cancelled(cancelled)?;
    source.require_vllm08_full_source()?;
    // Admission covers witness metadata through post-drop inspection. It keeps
    // the verified-source admission owner, never an OriginalWindowModelSource,
    // body, cold-slot or device witness used by the Weak checks below.
    let _controls = source.source().reserve_workspace(1024 * 1024)?;
    let mut witness = None;
    // Every strong local is confined to this original preparation/submission.
    // After it returns only the genuine armed flight may retain the graph.
    let result: Result<()> = async {
        not_cancelled(cancelled)?;
        source.require_vllm08_full_source()?;
        let prompt = records[0].call().request().teachers().ids().to_vec();
        if prompt.len() != 1 {
            return Err(Error::State(
                "natural unknown first source prompt geometry".into(),
            ));
        }
        let cold = prepare_natural_cold(device, source.clone(), records, inputs, rotary, cancelled)
            .await?;
        let model = cold.model;
        let sessions = cold.sessions;
        let slots = cold.slots;
        let pool = cold.state_pool;
        let input = sessions[0].bind_natural_prompt(
            source.clone(),
            NaturalSlot::InitialOne,
            &prompt,
            cancelled,
        )?;
        let state = sessions[0].bind_natural_state(input.clone(), &pool)?;
        let program = slots.bind_call(&sessions[0], input.clone(), state.clone(), cancelled)?;
        let body = program.full_model_body()?.clone();
        body.arm_fault(match fault {
            Vllm08NaturalFault::UnknownAfterConvolution => 1,
            Vllm08NaturalFault::CancelBeforeRecurrent => 2,
            _ => return Err(Error::State("natural unknown cut not selected".into())),
        })?;
        let snapshot = slots.quarantine_snapshot()?;
        let current = body
            .accesses()?
            .iter()
            .map(|(lease, _)| lease.id())
            .collect::<Vec<_>>();
        if current.iter().any(|id| !snapshot.prepared_ids.contains(id)) {
            return Err(Error::State(
                "natural fresh call escaped original cold prepared lease cohort".into(),
            ));
        }
        let roots = [sessions[0].binding(), sessions[1].binding()];
        witness = Some(Witness {
            cold: Arc::downgrade(&slots),
            body: Arc::downgrade(&body),
            input: Arc::downgrade(&input),
            state: Arc::downgrade(&state),
            source: Arc::downgrade(&source),
            service: Arc::downgrade(&model.service),
            snapshot,
            current,
            roots,
            roots_unchanged: false,
        });
        let result = sessions[0]
            .prepare_natural_call(
                NaturalSubmission {
                    input,
                    state,
                    program,
                    cold_owner: slots.completion_owner(),
                },
                |_| {
                    Err(Error::State(
                        "natural uncertain cut must not enqueue readback or create Ready".into(),
                    ))
                },
                cancelled,
            )
            .await;
        witness.as_mut().unwrap().roots_unchanged =
            sessions[0].binding() == roots[0] && sessions[1].binding() == roots[1];
        match result {
            Ok(_) => Err(Error::State(
                "natural uncertain cut returned completed proposal".into(),
            )),
            Err(error) => Err(error),
        }
    }
    .await;
    let intended = matches!((&result,fault),(Err(Error::State(message)),Vllm08NaturalFault::UnknownAfterConvolution)
        if message=="deliberate full24 unknown after actual source Conv and canonical copy")
        || matches!((&result,fault),(Err(Error::State(message)),Vllm08NaturalFault::CancelBeforeRecurrent)
        if message=="closed original cubin qualification cancelled");
    if !intended {
        return match result {
            Err(error) => Err(error),
            Ok(()) => Err(Error::State(
                "natural uncertain cut unexpectedly succeeded".into(),
            )),
        };
    }
    // Explicitly drop the entry source before checking retention. No witness
    // stores a strong source/body/cold/service owner merely to pass inspection.
    drop(source);
    let witness = witness
        .ok_or_else(|| Error::State("natural fault failed before actual original flight".into()))?;
    inspect(witness, fault)
}

fn inspect(w: Witness, fault: Vllm08NaturalFault) -> Result<Vllm08NaturalFaultOutcome> {
    let cold = w
        .cold
        .upgrade()
        .ok_or_else(|| Error::State("natural original armed flight lost cold graph".into()))?;
    let body = w
        .body
        .upgrade()
        .ok_or_else(|| Error::State("natural original armed flight lost fresh body".into()))?;
    let input = w
        .input
        .upgrade()
        .ok_or_else(|| Error::State("natural original armed flight lost trusted input".into()))?;
    let state = w.state.upgrade().ok_or_else(|| {
        Error::State("natural original armed flight lost canonical state binding".into())
    })?;
    let source = w.source.upgrade().ok_or_else(|| {
        Error::State("natural original armed flight lost immutable source".into())
    })?;
    let service = w
        .service
        .upgrade()
        .ok_or_else(|| Error::State("natural original armed flight lost service".into()))?;
    let actual = cold.quarantine_snapshot()?;
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&service.physical_key())?;
    let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
    let owners = crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    let current = body
        .accesses()?
        .iter()
        .map(|(lease, _)| lease.id())
        .collect::<Vec<_>>();
    if actual != w.snapshot
        || !actual.future_uninitialized
        || !w.roots_unchanged
        || current != w.current
        || !body.fault_cut_observed()
        || body.convolution_dispatches() == 0
        || body.native_launches() != 2
        || !Arc::ptr_eq(state.input(), &input)
        || !Arc::ptr_eq(input.source(), &source)
        || body.record().call() != input.call()
        || service.pending_events()? != 1
        || service.ensure_usable().is_ok()
        || !physical.fenced
        || physical.reserved != service.capacity() as u64
        || handles < w.current.len()
        || owners == 0
    {
        return Err(Error::State("natural uncertain original graph/source/current root/future initialization/quarantine differs".into()));
    }
    Ok(Vllm08NaturalFaultOutcome {
        fault,
        status: "UNKNOWN_RETAINED_NO_PUBLICATION",
        roots: Some(w.roots),
        completed_phases: 0,
        committed_calls: 0,
        emitted_tokens: 0,
        stale_inputs_refused: 0,
        native_launches: body.native_launches(),
        convolution_dispatches: body.convolution_dispatches(),
        pending_events: service.pending_events()?,
        quarantined_handles: handles,
        quarantined_completion_owners: owners,
        all_prepared_owners_retained: true,
        all_future_uninitialized: actual.future_uninitialized,
        prepared_state_buffers: actual.prepared_state_buffers,
        retained_plan_values: actual.retained_plan_values,
        retained_source_weight_groups: actual.retained_source_weight_groups,
        submitted_current_leases: w.current.len(),
        native_image_owners: actual.native_images,
        native_head_kernel_owners: actual.native_head_kernels,
        native_charge_bytes: actual.native_charge_bytes,
        image_admission_bytes: actual.image_admission_bytes,
        resident_bytes: actual.resident_bytes,
        physical_reserved_bytes: physical.reserved,
        physical_fenced: physical.fenced,
        source_weight_windows: 0,
        tied_weight_windows: 0,
    })
}
