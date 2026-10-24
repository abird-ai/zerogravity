//! Two fresh request pairs, one loaded model and the SAME original scratch.
use super::*;
use crate::Device;
use crate::resident_inference::native_fragment::natural_reuse::{
    PreparedReusableNatural, ReusableNaturalStorage, prepare_reusable_natural_cold,
};
use crate::resident_inference::native_fragment::{Vllm08ModelInputs, Vllm08ModelReadback};
use crate::vllm_cuda_qualification::source_rotary::RotaryOutcome;
use std::collections::BTreeSet;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::{BufferLeaseId, DeviceAccess};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

mod faults;
pub use faults::{
    Vllm08NaturalReuseFault, Vllm08NaturalReuseFaultExecution, Vllm08NaturalReuseFaultOutcome,
    qualify_vllm08_natural_reuse_fault,
};

#[derive(Debug)]
pub struct Vllm08NaturalReuseRound {
    pub roots: [RootBinding; 2],
    pub phase_roots: Vec<RootBinding>,
    pub short_tokens: [u32; 2],
    pub long_tokens: [u32; 2],
}
#[derive(Debug)]
pub struct Vllm08NaturalReuseReceipt {
    pub rounds: Vec<Vllm08NaturalReuseRound>,
    pub calls: usize,
    pub reused_geometries: usize,
    pub distinct_packets: usize,
    pub preserved_committed_packets: usize,
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

fn value_ids(
    body: &crate::resident_program::native_full::FullModelProgram,
) -> Result<Vec<BufferLeaseId>> {
    (0..327)
        .map(|i| body.values().lease(ValueId(i)).map(ResidentLease::id))
        .chain(std::iter::once(Ok(body.values().status().id())))
        .collect()
}

/// Same accepted1→1 and64→2→1 coordinates on two independent Session pairs.
/// Expected archives remain callback-only after actual original completion.
/// No claim for other lengths, repeated decode beyond2/67 or concurrent GPU math.
pub async fn qualify_vllm08_natural_reuse<F>(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    rotary: Option<RotaryOutcome>,
    cancelled: Option<&AtomicBool>,
    mut validate: F,
) -> Result<Vllm08NaturalReuseReceipt>
where
    F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
{
    not_cancelled(cancelled)?;
    source.require_vllm08_full_source()?;
    let _controls = source.source().reserve_workspace(2 * 1024 * 1024)?;
    let prompts = [
        records[0].call().request().teachers().ids().to_vec(),
        records[2].call().request().teachers().ids().to_vec(),
        records[3].call().request().teachers().ids().to_vec(),
    ];
    if prompts.iter().map(Vec::len).collect::<Vec<_>>() != [1, 64, 2] {
        return Err(Error::State(
            "reusable natural original prompt coordinates".into(),
        ));
    }
    let PreparedReusableNatural { model, storage } =
        prepare_reusable_natural_cold(device, source.clone(), records, inputs, rotary, cancelled)
            .await?;
    let service = model.service.clone();
    let key = service.physical_key();
    let mut sessions = Vec::with_capacity(4);
    let mut rounds = Vec::with_capacity(2);
    let mut geometry_ids: [Option<Vec<BufferLeaseId>>; 5] = std::array::from_fn(|_| None);
    let mut packets = BTreeSet::new();
    let mut candidate_ids = BTreeSet::new();
    let mut kept_feedback = Vec::with_capacity(8);
    let mut calls = 0;
    let mut reused_geometries = 0;
    let mut prefix64_emissions = 0;
    let mut native_launches = 0;
    let mut workspace_clears = 0;
    let mut device_copies = 0;
    let mut dispatches = 0;
    for round in 0..2 {
        not_cancelled(cancelled)?;
        let start = sessions.len();
        sessions.push(storage.session(cancelled).await?);
        sessions.push(storage.session(cancelled).await?);
        let initial = [sessions[start].binding(), sessions[start + 1].binding()];
        if initial.iter().any(|r| {
            r.source() != source.source_identity()
                || r.frontier() != 0
                || r.root() != 0
                || r.session_epoch() != 0
        }) || sessions.iter().enumerate().any(|(i, a)| {
            sessions[..i]
                .iter()
                .any(|b| a.binding().session() == b.binding().session())
        }) {
            return Err(Error::State(
                "reusable natural requires four independent original Sessions".into(),
            ));
        }
        let mut feedback: [Option<Arc<CommittedBf16Feedback>>; 2] = [None, None];
        let mut current = initial;
        let mut short = Vec::with_capacity(2);
        let mut long = Vec::with_capacity(2);
        let mut roots = Vec::with_capacity(5);
        for slot in NaturalSlot::ALL {
            let h = usize::from(slot.index() >= 2);
            let s = start + h;
            storage.require_open(&sessions[s])?;
            let input = if slot.feedback() {
                sessions[s].bind_natural_feedback(
                    source.clone(),
                    slot,
                    feedback[h].clone().ok_or_else(|| {
                        Error::State("reusable natural actual feedback missing".into())
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
                sessions[s].bind_natural_prompt(source.clone(), slot, ids, cancelled)?
            };
            let state = sessions[s]
                .bind_fresh_natural_state(input.clone(), cancelled)
                .await?;
            for id in state.outputs().iter().map(ResidentLease::id) {
                if !candidate_ids.insert(id) {
                    return Err(Error::State(
                        "reusable natural candidate was reused across calls".into(),
                    ));
                }
            }
            state.require_reusable_state()?;
            let before = sessions[s].binding();
            let mut observed_body = None;
            let prepared = sessions[s]
                .prepare_natural_call_bound(
                    input.clone(),
                    state.clone(),
                    storage.completion_owner(),
                    || {
                        let program = storage.bind_call(&sessions[s], input, state, cancelled)?;
                        observed_body = Some(program.full_model_body()?.clone());
                        Ok(program)
                    },
                    &mut validate,
                    cancelled,
                )
                .await?;
            let body = observed_body
                .ok_or_else(|| Error::State("reusable actual locked binding absent".into()))?;
            let ids = value_ids(&body)?;
            if round == 0 {
                geometry_ids[slot.index()] = Some(ids);
            } else if geometry_ids[slot.index()].as_ref() == Some(&ids) {
                reused_geometries += 1;
            } else {
                return Err(Error::State(
                    "reusable call replaced the original geometry scratch".into(),
                ));
            }
            if !packets.insert(body.publication_packet()?.id()) {
                return Err(Error::State(
                    "reusable immutable packet allocation was reused".into(),
                ));
            }
            if sessions[s].binding() != before {
                return Err(Error::State(
                    "reusable physical completion published early".into(),
                ));
            }
            let committed = sessions[s].commit_natural_call(prepared, cancelled)?;
            let after = committed.root();
            if after.session() != before.session()
                || after.session_epoch() != before.session_epoch()
                || after.root() != before.root() + 1
                || after.frontier() != (slot.base() + slot.rows()) as u64
                || sessions[start + 1 - h].binding() != current[1 - h]
            {
                return Err(Error::State(
                    "reusable original roots crossed request boundaries".into(),
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
                kept_feedback.push(next.clone());
            } else if slot.emits() {
                return Err(Error::State(
                    "reusable emitted phase lacks actual packet".into(),
                ));
            }
            if slot == NaturalSlot::Prefix64 {
                prefix64_emissions += usize::from(committed.readback().device_token.is_some());
                if committed.feedback().is_some() || prefix64_emissions != 0 {
                    return Err(Error::State("reusable prefix64 emission".into()));
                }
            }
            native_launches += body.native_launches();
            workspace_clears += body.workspace_clears();
            device_copies += body.device_copies();
            dispatches += committed.readback().dispatches;
            roots.push(after);
            current[h] = after;
            calls += 1;
        }
        if short.len() != 2
            || long.len() != 2
            || sessions[start].binding().frontier() != 2
            || sessions[start + 1].binding().frontier() != 67
        {
            return Err(Error::State(
                "reusable complete original request pair".into(),
            ));
        }
        rounds.push(Vllm08NaturalReuseRound {
            roots: [sessions[start].binding(), sessions[start + 1].binding()],
            phase_roots: roots,
            short_tokens: [short[0], short[1]],
            long_tokens: [long[0], long[1]],
        });
    }
    if calls != 10
        || reused_geometries != 5
        || packets.len() != 10
        || candidate_ids.len() != 480
        || kept_feedback.len() != 8
    {
        return Err(Error::State(
            "reusable actual storage/state/packet coverage".into(),
        ));
    }
    verify_packets(&model, &storage, &kept_feedback, cancelled).await?;
    let (source_weight_windows, tied_weight_windows) =
        storage.verify_source_and_retire(cancelled).await?;
    drop(kept_feedback);
    drop(sessions);
    drop(storage);
    drop(model);
    let final_resident_bytes = service.resident_bytes()?;
    if final_resident_bytes != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "reusable final source/packet/state owners did not retire".into(),
        ));
    }
    drop(service);
    let physical =
        zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State("reusable physical envelope final0".into()));
    }
    Ok(Vllm08NaturalReuseReceipt {
        rounds,
        calls,
        reused_geometries,
        distinct_packets: packets.len(),
        preserved_committed_packets: 8,
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

async fn verify_packets(
    model: &Arc<ResidentModel>,
    storage: &Arc<ReusableNaturalStorage>,
    packets: &[Arc<CommittedBf16Feedback>],
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    let service = &model.service;
    for packet in packets {
        let packet_model = packet.model.upgrade().ok_or_else(|| {
            Error::State("reusable preserved packet original model retired".into())
        })?;
        if !Arc::ptr_eq(&packet_model, model)
            || !Arc::ptr_eq(&packet.source, storage.source())
            || packet.root.source() != service.source_identity()
            || packet.token.lease.source_identity() != service.source_identity()
            || packet.token.producer != Some(packet.root)
            || !Arc::ptr_eq(packet.token.lease.service(), service)
            || !packet.token.lease.is_initialized()
            || packet.token.lease.bytes() != 12
            || packet.token.lease.role() != BufferRole::Output
            || packet.token.lease.allocation_generation() != packet.allocation_generation
            || packet.token.lease.service_generation() != packet.service_generation
        {
            return Err(Error::State(
                "reusable preserved packet original source/root/generation differs".into(),
            ));
        }
    }
    let accesses = packets
        .iter()
        .map(|p| (p.token.lease.clone(), DeviceAccess::Read))
        .collect::<Vec<_>>();
    let reads = packets
        .iter()
        .map(|p| p.token.lease.readback(0, 12))
        .collect::<Result<Vec<_>>>()?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 96,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    let _workspace = model.workspace.lock().await;
    not_cancelled(cancelled)?;
    slot.begin_submission()?;
    let mut flight = service.submit(0, &accesses)?;
    flight.retain_completion_owner(storage.completion_owner())?;
    for packet in packets {
        flight.retain_completion_owner(packet.clone())?;
    }
    drop(
        flight
            .complete_read_only_readback(&reads, |raw| {
                if raw.len() != packets.len() {
                    return Err(Error::State("reusable preserved packet cohort".into()));
                }
                for (raw, packet) in raw.iter().zip(packets) {
                    if raw[..] != packet.packet_bytes {
                        return Err(Error::State(
                            "reusable committed packet changed during scratch reuse".into(),
                        ));
                    }
                }
                Ok(())
            })
            .await?,
    );
    slot.finish()?;
    Ok(())
}
