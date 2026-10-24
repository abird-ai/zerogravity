//! SAME reused template path; safe rollback versus original uncertain quarantine.
use super::*;
use crate::resident_inference::native_fragment::NaturalColdSnapshot;
use crate::resident_program::native_full::FullModelProgram;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Vllm08NaturalReuseFault {
    CancelBeforeCommit,
    RestoreReplay,
    UnknownAfterConvolution,
    CancelBeforeRecurrent,
}
pub struct Vllm08NaturalReuseFaultExecution<'a> {
    pub rotary: Option<RotaryOutcome>,
    pub fault: Vllm08NaturalReuseFault,
    pub cancelled: Option<&'a AtomicBool>,
}
#[derive(Debug)]
pub struct Vllm08NaturalReuseFaultOutcome {
    pub fault: Vllm08NaturalReuseFault,
    pub status: &'static str,
    pub roots: [RootBinding; 2],
    pub completed_calls: usize,
    pub committed_calls: usize,
    pub preserved_packets: usize,
    pub stale_inputs_refused: usize,
    pub candidate_replay_refused: bool,
    pub fresh_retry_committed: bool,
    pub competing_commit_refused: bool,
    pub shared_temporary_ids_reused: bool,
    pub all_current_owners_retained: bool,
    pub current_candidate_uninitialized: bool,
    pub native_launches: usize,
    pub convolution_dispatches: usize,
    pub pending_events: usize,
    pub quarantined_handles: usize,
    pub quarantined_completion_owners: usize,
    pub resident_bytes: usize,
    pub physical_reserved_bytes: u64,
    pub physical_fenced: bool,
    pub source_weight_windows: usize,
    pub tied_weight_windows: usize,
}
struct Witness {
    storage: Weak<ReusableNaturalStorage>,
    body: Weak<FullModelProgram>,
    input: Weak<NaturalCallInput>,
    state: Weak<NaturalCallState>,
    source: Weak<OriginalWindowModelSource>,
    service: Weak<ResidentService>,
    snapshot: NaturalColdSnapshot,
    current: Vec<BufferLeaseId>,
    roots: [RootBinding; 2],
    roots_unchanged: bool,
    shared_ids: bool,
}

pub async fn qualify_vllm08_natural_reuse_fault<F>(
    device: Device,
    source: Arc<OriginalWindowModelSource>,
    records: [RecordedVllm08Model; 5],
    inputs: Vllm08ModelInputs,
    execution: Vllm08NaturalReuseFaultExecution<'_>,
    mut validate: F,
) -> Result<Vllm08NaturalReuseFaultOutcome>
where
    F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
{
    let Vllm08NaturalReuseFaultExecution {
        rotary,
        fault,
        cancelled,
    } = execution;
    not_cancelled(cancelled)?;
    source.require_vllm08_full_source()?;
    let _controls = source.source().reserve_workspace(2 * 1024 * 1024)?;
    let mut witness = None;
    let result: Result<Vllm08NaturalReuseFaultOutcome> = async {
        let prompt=records[0].call().request().teachers().ids().to_vec();
        if prompt.len()!=1 {return Err(Error::State("reuse fault requires original initialone prompt".into()));}
        let PreparedReusableNatural{model,storage}=prepare_reusable_natural_cold(device,source.clone(),records,inputs,rotary,cancelled).await?;
        let service=model.service.clone();let key=service.physical_key();
        let mut first=storage.session(cancelled).await?;
        let mut second=storage.session(cancelled).await?;
        let initial=[first.binding(),second.binding()];
        let input=first.bind_natural_prompt(source.clone(),NaturalSlot::InitialOne,&prompt,cancelled)?;
        let state=first.bind_fresh_natural_state(input.clone(),cancelled).await?;
        state.require_reusable_state()?;
        let mut first_body=None;
        let prepared=first.prepare_natural_call_bound(input.clone(),state.clone(),storage.completion_owner(),|| {
            let program=storage.bind_call(&first,input,state,cancelled)?;
            first_body=Some(program.full_model_body()?.clone());Ok(program)
        }, &mut validate,cancelled).await?;
        let first_body=first_body.ok_or_else(||Error::State("reuse first actual body missing".into()))?;
        let first_ids=value_ids(&first_body)?;
        let first_packet=first_body.publication_packet()?.id();
        let committed=first.commit_natural_call(prepared,cancelled)?;
        let kept=committed.feedback().cloned().ok_or_else(||Error::State("reuse first committed packet missing".into()))?;
        let first_root=first.binding();
        if first_root.root()!=1 || first_root.frontier()!=1 || second.binding()!=initial[1] {
            return Err(Error::State("reuse fault first real commit progression".into()));
        }
        drop(committed);drop(first_body);
        let snapshot=second.snapshot();
        let input=second.bind_natural_prompt(source.clone(),NaturalSlot::InitialOne,&prompt,cancelled)?;
        let state=second.bind_fresh_natural_state(input.clone(),cancelled).await?;
        state.require_reusable_state()?;
        let roots=[first.binding(),second.binding()];
        let uncertain=matches!(fault,Vllm08NaturalReuseFault::UnknownAfterConvolution|Vllm08NaturalReuseFault::CancelBeforeRecurrent);
        let mut actual_body=None;
        let mut shared_ids=false;
        let prepared=second.prepare_natural_call_bound(input.clone(),state.clone(),storage.completion_owner(),|| {
            let program=storage.bind_call(&second,input.clone(),state.clone(),cancelled)?;
            let body=program.full_model_body()?.clone();
            shared_ids=value_ids(&body)?==first_ids;
            if !shared_ids || body.publication_packet()?.id()==first_packet {
                return Err(Error::State("reuse fault requires same scratch and new immutable packet".into()));
            }
            if uncertain {
                body.arm_fault(if fault==Vllm08NaturalReuseFault::UnknownAfterConvolution {1}else{2})?;
                witness=Some(Witness {
                    storage:Arc::downgrade(&storage),body:Arc::downgrade(&body),input:Arc::downgrade(&input),
                    state:Arc::downgrade(&state),source:Arc::downgrade(&source),service:Arc::downgrade(&service),
                    snapshot:storage.quarantine_snapshot()?,
                    current:body.accesses()?.iter().map(|(l,_)|l.id()).collect(),roots,roots_unchanged:false,shared_ids,
                });
            }
            actual_body=Some(body);Ok(program)
        }, |actual| {
            if uncertain {return Err(Error::State("reuse uncertain cut must not perform D2H/Ready".into()));}
            validate(actual)
        },cancelled).await;
        if uncertain {
            if let Some(w)=&mut witness {w.roots_unchanged=first.binding()==roots[0]&&second.binding()==roots[1];}
            return match prepared {Err(e)=>Err(e),Ok(_)=>Err(Error::State("reuse uncertain cut completed".into()))};
        }
        let prepared=prepared?;
        let body=actual_body.ok_or_else(||Error::State("reuse second actual body missing".into()))?;
        // Root is still unchanged after the first physical prepare. Attempting
        // a SECOND immutable body over that SAME Arc candidate must refuse
        // before the bind closure, even though the root is still current.
        let mut rebound=false;
        let replay=second.prepare_natural_call_bound(input.clone(),state.clone(),storage.completion_owner(),|| {
            rebound=true;Err(Error::State("reuse candidate replay reached binder".into()))
        }, |_|Err(Error::State("reuse candidate replay reached completion".into())),cancelled).await;
        let candidate_replay_refused=matches!(replay,Err(Error::State(ref m)) if m=="reusable natural candidate already submitted")
            && !rebound && second.binding()==roots[1] && service.pending_events()?==0;
        if !candidate_replay_refused {return Err(Error::State("reuse same current candidate submitted twice".into()));}
        let mut stale_inputs_refused=0;
        let mut preserved=vec![kept.clone()];
        let committed_calls = match fault {
            Vllm08NaturalReuseFault::CancelBeforeCommit=>{
                let cancel=AtomicBool::new(true);
                if !matches!(second.commit_natural_call(prepared,Some(&cancel)),Err(Error::State(m)) if m=="natural full24 cancelled before publication") || second.binding()!=roots[1] {
                    return Err(Error::State("reuse beforecommit cancellation published candidate".into()));
                }
                2
            }
            Vllm08NaturalReuseFault::RestoreReplay=>{
                let committed=second.commit_natural_call(prepared,cancelled)?;
                if committed.root().root()!=1 {return Err(Error::State("reuse restore actual commit absent".into()));}
                let old_feedback=committed.feedback().cloned().ok_or_else(||Error::State("reuse restore committed packet absent".into()))?;
                drop(committed);
                second.restore(&snapshot)?;
                if second.binding().root()!=2 || second.binding().session_epoch()!=1 || second.binding().frontier()!=0 {
                    return Err(Error::State("reuse original restore epoch differs".into()));
                }
                if revoked(input.require_session(&second)){stale_inputs_refused+=1;}
                if revoked(state.require_current(&second)){stale_inputs_refused+=1;}
                if matches!(old_feedback.require_current(&second,&source),Err(Error::State(m)) if m=="natural feedback is stale/foreign or lacks original completed packet"){stale_inputs_refused+=1;}
                preserved.push(old_feedback);
                if stale_inputs_refused!=3 {return Err(Error::State("reuse restore did not revoke old call/candidate".into()));}
                3
            }
            _=>unreachable!(),
        };
        drop(body);drop(input);drop(state);
        // Fresh claims/packets on the original restored/current root. A second
        // candidate completes privately while the first private Ready remains.
        let retry_base=second.binding();
        let PreparedInitial{prepared:retry,body:retry_body,input:retry_input,state:retry_state}=prepare_initial(&second,&source,&storage,&prompt,&mut validate,cancelled).await?;
        if value_ids(&retry_body)?!=first_ids {return Err(Error::State("reuse retry replaced shared scratch".into()));}
        let PreparedInitial{prepared:competing,body:competing_body,input:competing_input,state:competing_state}=prepare_initial(&second,&source,&storage,&prompt,&mut validate,cancelled).await?;
        if value_ids(&competing_body)?!=first_ids || retry_body.publication_packet()?.id()==competing_body.publication_packet()?.id()
            ||retry_state.outputs().iter().any(|a| competing_state.outputs().iter().any(|b| a.id()==b.id())) {
            return Err(Error::State("reuse competing candidates/packets alias".into()));
        }
        if second.binding()!=retry_base {return Err(Error::State("reuse competing completion published early".into()));}
        let retry=second.commit_natural_call(retry,cancelled)?;
        let after=second.binding();
        if after.root()!=retry_base.root()+1 || after.frontier()!=1 ||after.session_epoch()!=retry_base.session_epoch() {
            return Err(Error::State("reuse fresh retry did not publish actual canonical root".into()));
        }
        preserved.push(retry.feedback().cloned().ok_or_else(||Error::State("reuse actual retry packet missing".into()))?);
        let competing_commit_refused=matches!(second.commit_natural_call(competing,cancelled),Err(Error::State(m)) if m=="natural full24 original current-root/call span differs or was revoked") && second.binding()==after;
        if !competing_commit_refused || first.binding()!=first_root {
            return Err(Error::State("reuse competing stale publication changed committed roots".into()));
        }
        verify_packets(&model,&storage,&preserved,cancelled).await?;
        let final_roots=[first.binding(),second.binding()];
        let native_launches=retry_body.native_launches();
        let convolution_dispatches=retry_body.convolution_dispatches();
        let preserved_packets=preserved.len();
        drop(retry);drop(retry_body);drop(retry_input);drop(retry_state);
        drop(competing_body);drop(competing_input);drop(competing_state);
        let (source_weight_windows,tied_weight_windows)=storage.verify_source_and_retire(cancelled).await?;
        drop(preserved);drop(kept);drop(snapshot);drop(first);drop(second);drop(storage);drop(model);
        let resident_bytes=service.resident_bytes()?;
        if resident_bytes!=0||service.pending_events()?!=0 {return Err(Error::State("reuse safe fault owners did not retire".into()));}
        drop(service);
        let physical=zerogravity_core::resource_arbiter::ResourceArbiter::process_shared().snapshot(&key)?;
        if physical.reserved!=0||physical.fenced {return Err(Error::State("reuse safe fault physical final0".into()));}
        Ok(Vllm08NaturalReuseFaultOutcome {fault,status:"SAFE_REUSE_REFUSED_FINAL0",roots:final_roots,completed_calls:4,committed_calls,
            preserved_packets,stale_inputs_refused,candidate_replay_refused,fresh_retry_committed:true,competing_commit_refused,shared_temporary_ids_reused:shared_ids,
            all_current_owners_retained:false,current_candidate_uninitialized:false,native_launches,convolution_dispatches,
            pending_events:0,quarantined_handles:0,quarantined_completion_owners:0,resident_bytes,physical_reserved_bytes:physical.reserved,
            physical_fenced:false,source_weight_windows,tied_weight_windows})
    }.await;
    if !matches!(
        fault,
        Vllm08NaturalReuseFault::UnknownAfterConvolution
            | Vllm08NaturalReuseFault::CancelBeforeRecurrent
    ) {
        return result;
    }
    let intended = matches!((&result,fault),(Err(Error::State(m)),Vllm08NaturalReuseFault::UnknownAfterConvolution)
        if m=="deliberate full24 unknown after actual source Conv and canonical copy")
        || matches!((&result,fault),(Err(Error::State(m)),Vllm08NaturalReuseFault::CancelBeforeRecurrent)
        if m=="closed original cubin qualification cancelled");
    if !intended {
        return result;
    }
    drop(source);
    inspect(
        witness.ok_or_else(|| Error::State("reuse uncertain original witness absent".into()))?,
        fault,
    )
}

fn inspect(w: Witness, fault: Vllm08NaturalReuseFault) -> Result<Vllm08NaturalReuseFaultOutcome> {
    let storage = w
        .storage
        .upgrade()
        .ok_or_else(|| Error::State("reuse original flight lost storage".into()))?;
    let body = w
        .body
        .upgrade()
        .ok_or_else(|| Error::State("reuse original flight lost body".into()))?;
    let input = w
        .input
        .upgrade()
        .ok_or_else(|| Error::State("reuse original flight lost input".into()))?;
    let state = w
        .state
        .upgrade()
        .ok_or_else(|| Error::State("reuse original flight lost canonical candidate".into()))?;
    let source = w
        .source
        .upgrade()
        .ok_or_else(|| Error::State("reuse original flight lost source".into()))?;
    let service = w
        .service
        .upgrade()
        .ok_or_else(|| Error::State("reuse original flight lost service".into()))?;
    let actual = storage.quarantine_snapshot()?;
    // Completed first-session candidates/packet can retire once that session
    // drops. Compare real COMMON prepared identities, not total resident bytes
    // or a fictitious all-future-uninitialized write cohort.
    if actual.prepared_ids != w.snapshot.prepared_ids
        || actual.readback_cohorts != w.snapshot.readback_cohorts
        || actual.prepared_state_buffers != w.snapshot.prepared_state_buffers
        || actual.retained_plan_values != w.snapshot.retained_plan_values
        || actual.retained_source_weight_groups != w.snapshot.retained_source_weight_groups
        || actual.native_images != w.snapshot.native_images
        || actual.native_head_kernels != w.snapshot.native_head_kernels
        || actual.native_charge_bytes != w.snapshot.native_charge_bytes
        || actual.image_admission_bytes != w.snapshot.image_admission_bytes
    {
        return Err(Error::State(
            "reuse original common source/native/storage graph changed after unknown".into(),
        ));
    }
    let current = body
        .accesses()?
        .iter()
        .map(|(l, _)| l.id())
        .collect::<Vec<_>>();
    let candidate_uninitialized = state.outputs().iter().all(|l| !l.is_initialized());
    let physical = zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&service.physical_key())?;
    let handles = crate::physical_quarantine::quarantined_handles_for(service.client())?;
    let owners = crate::physical_quarantine::quarantined_completion_owners_for(service.client())?;
    if !w.shared_ids
        || !w.roots_unchanged
        || current != w.current
        || !candidate_uninitialized
        || !body.fault_cut_observed()
        || body.native_launches() != 2
        || body.convolution_dispatches() == 0
        || !Arc::ptr_eq(state.input(), &input)
        || !Arc::ptr_eq(input.source(), &source)
        || !Arc::ptr_eq(body.record(), input.record())
        || service.pending_events()? != 1
        || service.ensure_usable().is_ok()
        || !physical.fenced
        || physical.reserved != service.capacity() as u64
        || handles < current.len()
        || owners == 0
    {
        return Err(Error::State(
            "reuse original uncertain current graph/ownership/root/fence differs".into(),
        ));
    }
    Ok(Vllm08NaturalReuseFaultOutcome {
        fault,
        status: "UNKNOWN_REUSE_RETAINED_NO_PUBLICATION",
        roots: w.roots,
        completed_calls: 1,
        committed_calls: 1,
        preserved_packets: 0,
        stale_inputs_refused: 0,
        candidate_replay_refused: false,
        fresh_retry_committed: false,
        competing_commit_refused: false,
        shared_temporary_ids_reused: true,
        all_current_owners_retained: true,
        current_candidate_uninitialized: candidate_uninitialized,
        native_launches: body.native_launches(),
        convolution_dispatches: body.convolution_dispatches(),
        pending_events: service.pending_events()?,
        quarantined_handles: handles,
        quarantined_completion_owners: owners,
        resident_bytes: service.resident_bytes()?,
        physical_reserved_bytes: physical.reserved,
        physical_fenced: physical.fenced,
        source_weight_windows: 0,
        tied_weight_windows: 0,
    })
}

struct PreparedInitial {
    prepared: super::super::completion::NaturalPrepared,
    body: Arc<FullModelProgram>,
    input: Arc<NaturalCallInput>,
    state: Arc<NaturalCallState>,
}
async fn prepare_initial<F>(
    session: &ResidentSession,
    source: &Arc<OriginalWindowModelSource>,
    storage: &Arc<ReusableNaturalStorage>,
    prompt: &[u32],
    validate: &mut F,
    cancelled: Option<&AtomicBool>,
) -> Result<PreparedInitial>
where
    F: FnMut(Vllm08ModelReadback<'_>) -> Result<()>,
{
    let input =
        session.bind_natural_prompt(source.clone(), NaturalSlot::InitialOne, prompt, cancelled)?;
    let state = session
        .bind_fresh_natural_state(input.clone(), cancelled)
        .await?;
    state.require_reusable_state()?;
    let mut body = None;
    let prepared = session
        .prepare_natural_call_bound(
            input.clone(),
            state.clone(),
            storage.completion_owner(),
            || {
                let program =
                    storage.bind_call(session, input.clone(), state.clone(), cancelled)?;
                body = Some(program.full_model_body()?.clone());
                Ok(program)
            },
            validate,
            cancelled,
        )
        .await?;
    Ok(PreparedInitial {
        prepared,
        body: body.ok_or_else(|| Error::State("reuse retry bound body missing".into()))?,
        input,
        state,
    })
}

fn revoked(result: Result<()>) -> bool {
    matches!(result,Err(Error::State(m)) if m=="natural full24 original current-root/call span differs or was revoked")
}
