//! Same natural binding/submit/complete/commit path; no recovery on unknown work.
use super::completion::NaturalSubmission;
use super::*;
use crate::Device;
use crate::resident_inference::native_fragment::{
    NaturalRetirementProbe, Vllm08ModelInputs, Vllm08ModelReadback, prepare_natural_cold,
};
use crate::vllm_cuda_qualification::source_rotary::RotaryOutcome;

mod quarantine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vllm08NaturalFault {
    AdmissionPressure,
    CancelBeforeCommit,
    RestoreReplay,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
}
pub struct Vllm08NaturalFaultExecution<'a> {
    pub rotary: Option<RotaryOutcome>,
    pub fault: Vllm08NaturalFault,
    pub cancelled: Option<&'a AtomicBool>,
}
#[derive(Debug)]
pub struct Vllm08NaturalFaultOutcome {
    pub fault: Vllm08NaturalFault,
    pub status: &'static str,
    pub roots: Option<[RootBinding; 2]>,
    pub completed_phases: usize,
    pub committed_calls: usize,
    pub emitted_tokens: usize,
    pub stale_inputs_refused: usize,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub all_prepared_owners_retained: bool,
    pub all_future_uninitialized: bool,
    pub prepared_state_buffers: usize,
    pub retained_plan_values: usize,
    pub retained_source_weight_groups: usize,
    pub submitted_current_leases: usize,
    pub native_image_owners: usize,
    pub native_head_kernel_owners: usize,
    pub native_charge_bytes: usize,
    pub image_admission_bytes: usize,
    pub resident_bytes: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
}

pub async fn qualify_vllm08_natural_fault<F>(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    execution: Vllm08NaturalFaultExecution<'_>,
    mut validate: F,
) -> Result<Vllm08NaturalFaultOutcome>
where
    F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
{
    let Vllm08NaturalFaultExecution {
        rotary,
        fault,
        cancelled,
    } = execution;
    // True entry cancellation wins before source checks, driver or admissions.
    not_cancelled(cancelled)?;
    if fault == Vllm08NaturalFault::AdmissionPressure {
        return quarantine::pressure(device, source, records, inputs, rotary, cancelled).await;
    }
    if matches!(
        fault,
        Vllm08NaturalFault::UnknownAfterConvolution | Vllm08NaturalFault::CancelBeforeRecurrent
    ) {
        return quarantine::qualify(
            device,
            source,
            records,
            inputs,
            Vllm08NaturalFaultExecution {
                rotary,
                fault,
                cancelled,
            },
        )
        .await;
    }
    source.require_vllm08_full_source()?;
    let _controls = source.source().reserve_workspace(1024 * 1024)?;
    let prompt = records[0].call().request().teachers().ids().to_vec();
    if prompt.len() != 1 {
        return Err(Error::State(
            "natural fault first source prompt geometry".into(),
        ));
    }
    let cold =
        prepare_natural_cold(device, source.clone(), records, inputs, rotary, cancelled).await?;
    let model = cold.model;
    let mut sessions = cold.sessions;
    let slots = cold.slots;
    let pool = cold.state_pool;
    let service = model.service.clone();
    let key = service.physical_key();
    let initial = [sessions[0].binding(), sessions[1].binding()];
    let snapshot = sessions[0].snapshot();
    let input = sessions[0].bind_natural_prompt(
        source.clone(),
        NaturalSlot::InitialOne,
        &prompt,
        cancelled,
    )?;
    if input.require_session(&sessions[1]).is_ok() {
        return Err(Error::State(
            "natural foreign original Session accepted a trusted input".into(),
        ));
    }
    let state = sessions[0].bind_natural_state(input.clone(), &pool)?;
    let program = slots.bind_call(&sessions[0], input.clone(), state.clone(), cancelled)?;
    let body = program.full_model_body()?.clone();
    let prepared = sessions[0]
        .prepare_natural_call(
            NaturalSubmission {
                input: input.clone(),
                state: state.clone(),
                program,
                cold_owner: slots.completion_owner(),
            },
            &mut validate,
            cancelled,
        )
        .await?;
    if sessions[0].binding() != initial[0] || sessions[1].binding() != initial[1] {
        return Err(Error::State(
            "natural fault completion published without original commit".into(),
        ));
    }
    let mut stale_inputs_refused = 1;
    let (kind, committed_calls, emitted_tokens, status) = match fault {
        Vllm08NaturalFault::CancelBeforeCommit => {
            let flag = AtomicBool::new(true);
            match sessions[0].commit_natural_call(prepared, Some(&flag)) {
                Err(Error::State(message))
                    if message == "natural full24 cancelled before publication" => {}
                _ => {
                    return Err(Error::State(
                        "natural cancellation did not refuse original commit at intended boundary"
                            .into(),
                    ));
                }
            }
            if sessions[0].binding() != initial[0] {
                return Err(Error::State(
                    "natural cancelled proposal changed published root".into(),
                ));
            }
            (
                NaturalRetirementProbe::CancelBeforeCommit,
                0,
                0,
                "COMPLETED_PRIVATE_CANCELLED_NO_PUBLICATION",
            )
        }
        Vllm08NaturalFault::RestoreReplay => {
            let committed = sessions[0].commit_natural_call(prepared, cancelled)?;
            let feedback = committed.feedback().cloned().ok_or_else(|| {
                Error::State("natural restore probe lacks original committed packet".into())
            })?;
            if committed.root().root() != 1 || committed.root().frontier() != 1 {
                return Err(Error::State(
                    "natural restore first original commit differs".into(),
                ));
            }
            sessions[0].restore(&snapshot)?;
            let now = sessions[0].binding();
            if now.root() != 2
                || now.session_epoch() != 1
                || now.frontier() != 0
                || now.session() != initial[0].session()
            {
                return Err(Error::State(
                    "natural restore did not use original epoch/root transition".into(),
                ));
            }
            if input.require_session(&sessions[0]).is_ok()
                || feedback.require_current(&sessions[0], &source).is_ok()
                || sessions[0]
                    .bind_natural_feedback(
                        source.clone(),
                        NaturalSlot::FeedbackAtOne,
                        feedback.clone(),
                        cancelled,
                    )
                    .is_ok()
                || slots
                    .bind_call(&sessions[0], input.clone(), state.clone(), cancelled)
                    .is_ok()
            {
                return Err(Error::State("natural restored root accepted stale input, completed packet or claimed slot replay".into()));
            }
            stale_inputs_refused += 4;
            drop(feedback);
            drop(committed);
            (
                NaturalRetirementProbe::RestoredFirst,
                1,
                1,
                "RESTORE_REVOKED_INPUT_PACKET_AND_SLOT",
            )
        }
        _ => unreachable!(),
    };
    if sessions[1].binding() != initial[1] {
        return Err(Error::State(
            "natural fault affected independent original history".into(),
        ));
    }
    let roots = [sessions[0].binding(), sessions[1].binding()];
    let native_launches = body.native_launches();
    let convolution_dispatches = body.convolution_dispatches();
    let (source_weight_windows, tied_weight_windows) = slots
        .verify_source_and_retire_probe(&sessions, kind, cancelled)
        .await?;
    drop(body);
    drop(input);
    drop(state);
    drop(snapshot);
    drop(pool);
    drop(slots);
    drop(sessions);
    drop(model);
    let resident_bytes = service.resident_bytes()?;
    if resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "natural safe fault retained completed graph".into(),
        ));
    }
    drop(service);
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "natural safe fault failed original physical final0".into(),
        ));
    }
    Ok(Vllm08NaturalFaultOutcome {
        fault,
        status,
        roots: Some(roots),
        completed_phases: 1,
        committed_calls,
        emitted_tokens,
        stale_inputs_refused,
        native_launches,
        convolution_dispatches,
        pending_events: 0,
        quarantined_handles: 0,
        quarantined_completion_owners: 0,
        all_prepared_owners_retained: false,
        all_future_uninitialized: true,
        resident_bytes,
        physical_reserved_bytes: physical.reserved,
        prepared_state_buffers: 0,
        retained_plan_values: 0,
        retained_source_weight_groups: 0,
        submitted_current_leases: 0,
        native_image_owners: 0,
        native_head_kernel_owners: 0,
        native_charge_bytes: 0,
        image_admission_bytes: 0,
        physical_fenced: physical.fenced,
        source_weight_windows,
        tied_weight_windows,
    })
}
