//! Bounded qualification calls through ORIGINAL Session prepare/commit.
//! Model arithmetic stays in the original ResidentProgram node traversal.
use super::completion::NaturalSubmission;
use super::*;
use crate::Device;
use crate::resident_inference::native_fragment::{
    Vllm08ModelInputs, Vllm08ModelReadback, prepare_natural_cold,
};
use crate::vllm_cuda_qualification::source_rotary::RotaryOutcome;

#[derive(Debug)]
pub struct Vllm08NaturalReceipt {
    pub phases: usize,
    pub roots: [RootBinding; 2],
    pub phase_roots: Vec<RootBinding>,
    pub short_tokens: [u32; 2],
    pub long_tokens: [u32; 2],
    pub prefix64_emissions: usize,
    pub native_launches: usize,
    pub workspace_clears: usize,
    pub device_copies: usize,
    pub dispatches: usize,
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
    pub final_resident_bytes: usize,
    pub final_physical_reserved_bytes: u64,
}

/// The supplied records establish cold storage geometry only. Their cached
/// teacher IDs are NEVER natural decode inputs: only actual committed private
/// BF16 packets may mint the subsequent one-token source requests.
/// Expected archives are opened only by `validate` after genuine arithmetic.
pub async fn qualify_vllm08_natural_model<F>(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
    mut validate: F,
) -> Result<Vllm08NaturalReceipt>
where
    F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
{
    // Wins even when no rotary producer or provider exists. No fake prefix.
    not_cancelled(cancelled)?;
    source.require_vllm08_full_source()?;
    let _controls = source.source().reserve_workspace(1024 * 1024)?;
    // Bounded source prompt/control IDs, never captured arithmetic payloads.
    let prompts = [
        records[0].call().request().teachers().ids().to_vec(),
        records[2].call().request().teachers().ids().to_vec(),
        records[3].call().request().teachers().ids().to_vec(),
    ];
    if prompts.iter().map(Vec::len).collect::<Vec<_>>() != [1, 64, 2] {
        return Err(Error::State("natural exact source prompt controls".into()));
    }
    let cold =
        prepare_natural_cold(device, source.clone(), records, inputs, rotary, cancelled).await?;
    let model = cold.model;
    let mut sessions = cold.sessions;
    let slots = cold.slots;
    let state_pool = cold.state_pool;
    let service = model.service.clone();
    let key = service.physical_key();
    let initial = [sessions[0].binding(), sessions[1].binding()];
    if initial.iter().any(|root| {
        root.source() != source.source_identity()
            || root.frontier() != 0
            || root.root() != 0
            || root.session_epoch() != 0
    }) || initial[0].session() == initial[1].session()
    {
        return Err(Error::State(
            "natural two genuine independent original sessions".into(),
        ));
    }
    let mut feedback: [Option<Arc<CommittedBf16Feedback>>; 2] = [None, None];
    let mut current = initial;
    let mut short = Vec::with_capacity(2);
    let mut long = Vec::with_capacity(2);
    let mut phase_roots = Vec::with_capacity(5);
    let mut dispatches = 0usize;
    let mut native_launches = 0usize;
    let mut workspace_clears = 0usize;
    let mut device_copies = 0usize;
    let mut prefix64_emissions = 0usize;
    for slot in NaturalSlot::ALL {
        not_cancelled(cancelled)?;
        let h = usize::from(slot.index() >= 2);
        let input = if slot.feedback() {
            sessions[h].bind_natural_feedback(
                source.clone(),
                slot,
                feedback[h].clone().ok_or_else(|| {
                    Error::State("natural next input lacks actual committed producer".into())
                })?,
                cancelled,
            )?
        } else {
            let ids = match slot {
                NaturalSlot::InitialOne => &prompts[0],
                NaturalSlot::Prefix64 => &prompts[1],
                NaturalSlot::FinalTwo => &prompts[2],
                _ => unreachable!(),
            };
            sessions[h].bind_natural_prompt(source.clone(), slot, ids, cancelled)?
        };
        let state = sessions[h].bind_natural_state(input.clone(), &state_pool)?;
        let program = slots.bind_call(&sessions[h], input.clone(), state.clone(), cancelled)?;
        let body = program.full_model_body()?.clone();
        let before = sessions[h].binding();
        let prepared = sessions[h]
            .prepare_natural_call(
                NaturalSubmission {
                    input,
                    state,
                    program,
                    cold_owner: slots.completion_owner(),
                },
                &mut validate,
                cancelled,
            )
            .await?;
        // The physical result is still private; only original commit publishes.
        if sessions[h].binding() != before {
            return Err(Error::State(
                "natural physical completion published before original commit".into(),
            ));
        }
        let committed = sessions[h].commit_natural_call(prepared, cancelled)?;
        let after = committed.root();
        if after.root() != before.root() + 1
            || after.frontier() != (slot.base() + slot.rows()) as u64
            || after.session() != before.session()
            || after.session_epoch() != before.session_epoch()
            || sessions[1 - h].binding() != current[1 - h]
        {
            return Err(Error::State(
                "natural original current-root progression differs".into(),
            ));
        }
        if let Some(next) = committed.feedback() {
            let (id, _) = next.selected();
            if h == 0 {
                short.push(id);
            } else {
                long.push(id);
            }
            feedback[h] = Some(next.clone());
        } else if slot.emits() {
            return Err(Error::State(
                "natural emitted phase omitted actual committed packet".into(),
            ));
        }
        if slot == NaturalSlot::Prefix64 {
            prefix64_emissions += usize::from(committed.readback().device_token.is_some());
            if committed.feedback().is_some() || prefix64_emissions != 0 {
                return Err(Error::State(
                    "natural prompt prefix must publish state without token emission".into(),
                ));
            }
        }
        dispatches = dispatches
            .checked_add(committed.readback().dispatches)
            .ok_or_else(|| Error::Limit("natural dispatch count overflow".into()))?;
        native_launches = native_launches
            .checked_add(body.native_launches())
            .ok_or_else(|| Error::Limit("natural native count overflow".into()))?;
        workspace_clears = workspace_clears
            .checked_add(body.workspace_clears())
            .ok_or_else(|| Error::Limit("natural clear count overflow".into()))?;
        device_copies = device_copies
            .checked_add(body.device_copies())
            .ok_or_else(|| Error::Limit("natural copy count overflow".into()))?;
        phase_roots.push(after);
        current[h] = after;
        drop(committed);
        drop(body);
    }
    if short.len() != 2
        || long.len() != 2
        || phase_roots.len() != 5
        || prefix64_emissions != 0
        || sessions[0].binding().frontier() != 2
        || sessions[1].binding().frontier() != 67
    {
        return Err(Error::State(
            "natural complete finite publication/emission coverage".into(),
        ));
    }
    let roots = [sessions[0].binding(), sessions[1].binding()];
    let (source_weight_windows, tied_weight_windows) =
        slots.verify_source_and_retire(&sessions, cancelled).await?;
    let short_tokens = [short[0], short[1]];
    let long_tokens = [long[0], long[1]];
    drop(feedback);
    drop(state_pool);
    drop(slots);
    drop(sessions);
    drop(model);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "natural all completed source/state/native owners failed to retire".into(),
        ));
    }
    drop(service);
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "natural original physical envelope failed final0".into(),
        ));
    }
    Ok(Vllm08NaturalReceipt {
        phases: 5,
        roots,
        phase_roots,
        short_tokens,
        long_tokens,
        prefix64_emissions,
        native_launches,
        workspace_clears,
        device_copies,
        dispatches,
        source_weight_windows,
        tied_weight_windows,
        final_resident_bytes,
        final_physical_reserved_bytes: physical.reserved,
    })
}
