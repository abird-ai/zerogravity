//! Same autoregressive binding/submit/complete/commit path; no recovery on unknown work.
use super::completion::AutoregressiveSubmission;
use super::*;
use crate::Device;
use crate::cuda::native::source_rotary::RotaryOutcome;
use crate::resident_inference::native_fragment::{
    AutoregressiveRetirementProbe, SourceModelInputs, SourceModelReadback, SourceModelScopes,
    prepare_autoregressive_cold,
};

mod quarantine;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoregressiveFault {
    AdmissionPressure,
    CancelBeforeCommit,
    RestoreReplay,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
}
pub struct AutoregressiveFaultExecution<'a> {
    pub rotary: Option<RotaryOutcome>,
    pub fault: AutoregressiveFault,
    pub cancelled: Option<&'a AtomicBool>,
}
#[derive(Debug)]
pub struct AutoregressiveFaultOutcome {
    pub fault: AutoregressiveFault,
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

pub async fn qualify_autoregressive_fault<F>(
    device: Device,
    scopes: SourceModelScopes,
    inputs: SourceModelInputs,
    execution: AutoregressiveFaultExecution<'_>,
    mut validate: F,
) -> Result<AutoregressiveFaultOutcome>
where
    F: FnMut(SourceModelReadback<'_>) -> Result<()>,
{
    let AutoregressiveFaultExecution {
        rotary,
        fault,
        cancelled,
    } = execution;
    // True entry cancellation wins before source checks, driver or admissions.
    not_cancelled(cancelled)?;
    let (source, records) = scopes.into_parts();
    if fault == AutoregressiveFault::AdmissionPressure {
        return quarantine::pressure(
            device,
            SourceModelScopes::new(source, records)?,
            inputs,
            rotary,
            cancelled,
        )
        .await;
    }
    if matches!(
        fault,
        AutoregressiveFault::UnknownAfterConvolution | AutoregressiveFault::CancelBeforeRecurrent
    ) {
        return quarantine::qualify(
            device,
            SourceModelScopes::new(source, records)?,
            inputs,
            AutoregressiveFaultExecution {
                rotary,
                fault,
                cancelled,
            },
        )
        .await;
    }
    let _controls = source.reserve_workspace(1024 * 1024)?;
    let prompt = records[0].invocation().tokens().ids().to_vec();
    if prompt.len() != 1 {
        return Err(Error::State(
            "autoregressive fault first source prompt geometry".into(),
        ));
    }
    let initial_record = records[0].clone();
    let cold = prepare_autoregressive_cold(
        device,
        SourceModelScopes::new(source.clone(), records)?,
        inputs,
        rotary,
        cancelled,
    )
    .await?;
    let model = cold.model;
    let mut sessions = cold.sessions;
    let slots = cold.slots;
    let pool = cold.state_pool;
    let service = model.service.clone();
    let key = service.physical_key();
    let initial = [sessions[0].binding(), sessions[1].binding()];
    let snapshot = sessions[0].snapshot();
    let input = sessions[0].bind_autoregressive_prompt(
        initial_record.clone(),
        AutoregressiveSlot::InitialOne,
        cancelled,
    )?;
    if input.require_session(&sessions[1]).is_ok() {
        return Err(Error::State(
            "autoregressive foreign original Session accepted a trusted input".into(),
        ));
    }
    let state = sessions[0].bind_autoregressive_state(input.clone(), &pool)?;
    let program = slots.bind_call(&sessions[0], input.clone(), state.clone(), cancelled)?;
    let body = program.full_model_body()?.clone();
    let prepared = sessions[0]
        .prepare_autoregressive_call(
            AutoregressiveSubmission {
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
            "autoregressive fault completion published without original commit".into(),
        ));
    }
    let mut stale_inputs_refused = 1;
    let (kind, committed_calls, emitted_tokens, status) = match fault {
        AutoregressiveFault::CancelBeforeCommit => {
            let flag = AtomicBool::new(true);
            match sessions[0].commit_autoregressive_call(prepared, Some(&flag)) {
                Err(Error::State(message))
                    if message == "autoregressive full24 cancelled before publication" => {}
                _ => {
                    return Err(Error::State(
                        "autoregressive cancellation did not refuse original commit at intended boundary"
                            .into(),
                    ));
                }
            }
            if sessions[0].binding() != initial[0] {
                return Err(Error::State(
                    "autoregressive cancelled proposal changed published root".into(),
                ));
            }
            (
                AutoregressiveRetirementProbe::CancelBeforeCommit,
                0,
                0,
                "COMPLETED_PRIVATE_CANCELLED_NO_PUBLICATION",
            )
        }
        AutoregressiveFault::RestoreReplay => {
            let committed = sessions[0].commit_autoregressive_call(prepared, cancelled)?;
            let feedback = committed.feedback().cloned().ok_or_else(|| {
                Error::State("autoregressive restore probe lacks original committed packet".into())
            })?;
            if committed.root().root() != 1 || committed.root().frontier() != 1 {
                return Err(Error::State(
                    "autoregressive restore first original commit differs".into(),
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
                    "autoregressive restore did not use original epoch/root transition".into(),
                ));
            }
            if input.require_session(&sessions[0]).is_ok()
                || feedback.require_current(&sessions[0], &source).is_ok()
                || feedback.request(&sessions[0]).is_ok()
                || slots
                    .bind_call(&sessions[0], input.clone(), state.clone(), cancelled)
                    .is_ok()
            {
                return Err(Error::State("autoregressive restored root accepted stale input, completed packet or claimed slot replay".into()));
            }
            stale_inputs_refused += 4;
            drop(feedback);
            drop(committed);
            (
                AutoregressiveRetirementProbe::RestoredFirst,
                1,
                1,
                "RESTORE_REVOKED_INPUT_PACKET_AND_SLOT",
            )
        }
        _ => unreachable!(),
    };
    if sessions[1].binding() != initial[1] {
        return Err(Error::State(
            "autoregressive fault affected independent original history".into(),
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
            "autoregressive safe fault retained completed graph".into(),
        ));
    }
    drop(service);
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "autoregressive safe fault failed original physical final0".into(),
        ));
    }
    Ok(AutoregressiveFaultOutcome {
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
