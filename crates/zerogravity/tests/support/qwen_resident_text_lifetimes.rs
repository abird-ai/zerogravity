//! Original resident-text lifecycle orchestration; never a runtime dependency.
use super::{
    reference::{Call, Expected},
    source::need,
};
use std::{
    future::Future,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity::{
    Error, Result, accelerator::program::InvocationPhase, qwen35::plan::QwenForwardPlan,
};
use zerogravity_gpu::{
    ResidentOutput, ResidentTextModel, ResidentTextPending, ResidentTextPreparation,
    ResidentTextPublication, ResidentTextSession,
};

pub fn supported(mode: &str) -> bool {
    matches!(
        mode,
        "healthy"
            | "session-known"
            | "session-fault"
            | "snapshot-known"
            | "snapshot-fault"
            | "pending-known"
            | "pending-fault"
            | "readback-known"
            | "snapshot-readback-known"
            | "unknown"
    )
}
fn phase(c: &Call) -> InvocationPhase {
    if c.rows == 1 && c.position != 0 {
        InvocationPhase::Decode
    } else {
        InvocationPhase::Prefill
    }
}
fn ready(p: ResidentTextPreparation) -> Result<ResidentTextPending> {
    match p {
        ResidentTextPreparation::Ready(p) => Ok(p),
        _ => Err(Error::State("expected actual source Ready".into())),
    }
}
fn final_reader(probe: zerogravity_gpu::ResidentTextFinalReader) -> Result<()> {
    let observed = probe.observe()?;
    need(
        !observed.model_retained
            && observed.aliases > 0
            && observed.bytes > 0
            && observed.before.retained,
        "actual sole wrapper retired after the original final lease",
    )?;
    println!("SOURCE_LIFECYCLE_FINAL_READER {observed:?}");
    Ok(())
}
fn selected(logits: &[f32]) -> Result<(u32, u32)> {
    let (i, value) = zerogravity::numerics::greedy_finite(logits)?;
    Ok((i, value.to_bits()))
}
async fn prepare(
    session: &ResidentTextSession,
    plan: &QwenForwardPlan,
    call: &Call,
    token: Option<u32>,
    cancelled: Option<Arc<AtomicBool>>,
) -> Result<ResidentTextPreparation> {
    let ids = token.map_or_else(|| call.input_ids.clone(), |t| vec![t]);
    let invocation = plan.bind_invocation(phase(call), call.position, ids.into(), true)?;
    session
        .prepare(
            invocation.bound_invocation(),
            invocation.phase(),
            ResidentOutput::Logits,
            cancelled,
        )
        .await
}
async fn check(session: &ResidentTextSession, expected: &Expected, call: &Call) -> Result<usize> {
    need(
        session.binding().frontier() == (call.position + call.rows) as u64,
        "published original frontier",
    )?;
    let snapshot = session.snapshot()?;
    let data = snapshot.read_domains().await?;
    let (words, tails) = expected.compare_domains(call, &data, 514)?;
    println!(
        "SOURCE_LIFECYCLE_STATE call={} state_words={words} runtime_checked_kv_tail_words={tails} domains={} planes={} mismatches=0 observer=false",
        call.call,
        expected.domain_count(),
        expected.plane_count()
    );
    Ok(words)
}
struct Noop;
impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}
fn until_submitted<F: Future<Output = Result<ResidentTextPreparation>>>(
    model: &ResidentTextModel,
    future: std::pin::Pin<&mut F>,
) -> Result<()> {
    let mut future = future;
    let waker = Waker::from(Arc::new(Noop));
    let mut context = Context::from_waker(&waker);
    let start = Instant::now();
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
            let _ = result?;
            return Err(Error::State(
                "resident-text completed before armed original launch boundary".into(),
            ));
        }
        if model.diagnostic_launch_handles() > 0 {
            return Ok(());
        }
        need(
            start.elapsed() < Duration::from_secs(300),
            "original source launch boundary deadline",
        )?;
        std::thread::yield_now();
    }
}

pub async fn run(
    mode: &str,
    model: Arc<ResidentTextModel>,
    plan: QwenForwardPlan,
    expected: &Expected,
) -> Result<()> {
    need(supported(mode), "resident-text lifecycle mode")?;
    let calls: Vec<_> = expected
        .calls()
        .iter()
        .filter(|c| c.repeat == 0 && c.case == 0)
        .collect();
    need(calls.len() == 3, "original fresh47 and two cached calls")?;
    let first = calls[0];
    let next = calls[1];
    let last = calls[2];
    let native = model.cpu_forward_source_native_probe().await?;
    let before = native.observe()?;
    let resource = before.physical.clone();
    let mut session = model.session().await?;

    if mode == "healthy" {
        let mut peer = model.session().await?;
        need(
            model.session().await.is_err(),
            "session semaphore refuses extra root before allocation",
        )?;
        let root0 = session.binding();
        let peer0 = peer.binding();
        let cancelled = Arc::new(AtomicBool::new(true));
        need(
            matches!(
                prepare(&session, &plan, first, None, Some(cancelled.clone())).await?,
                ResidentTextPreparation::CancelledBeforeLaunch
            ),
            "proven prelaunch cancellation",
        )?;
        let host = ResourceArbiter::process_shared();
        let key = zerogravity::resource_arbiter::PhysicalDeviceKey::new("host", "system-ram")?;
        let budget = host.snapshot(&key)?;
        let pressure = host.admit(
            &key,
            ResourceOwner::new("resident-text-qualification", "capacity-pressure")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: budget.remaining,
            }],
        )?;
        let capacity = prepare(&session, &plan, first, None, None).await;
        drop(pressure);
        need(
            matches!(capacity?, ResidentTextPreparation::Capacity),
            "proven prelaunch host capacity refusal",
        )?;
        need(
            session.binding() == root0
                && peer.binding() == peer0
                && native.observe()?.pending_events == 0,
            "prelaunch leaves roots and completion unchanged",
        )?;
        let pending = ready(prepare(&session, &plan, first, None, None).await?)?;
        let actual = selected(&pending.readback().logits)?;
        need(
            actual == (first.selected, first.max_bits),
            "first actual internal greedy feedback",
        )?;
        expected.compare_logits(first, &pending.readback().logits)?;
        drop(session.commit(pending)?);
        check(&session, expected, first).await?;
        let saved = session.snapshot()?;
        let saved_clone = saved.clone();
        let saved_charge = host.snapshot(&key)?.reserved;
        drop(saved_clone);
        need(
            host.snapshot(&key)?.reserved == saved_charge,
            "snapshot clone shares original allocation/charge",
        )?;
        let stable = session.binding();
        need(
            peer.restore(&saved).is_err() && peer.binding() == peer0,
            "cross-session snapshot refuses unchanged peer",
        )?;
        need(
            peer.commit(ready(
                prepare(&session, &plan, next, Some(actual.0), None).await?,
            )?)
            .is_err(),
            "pending root cannot publish through peer authority",
        )?;
        need(
            session.binding() == stable && peer.binding() == peer0,
            "cross-session pending leaves both roots",
        )?;

        model.diagnostic_pause_next_launch()?;
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut in_flight = Box::pin(prepare(
            &session,
            &plan,
            next,
            Some(actual.0),
            Some(cancelled.clone()),
        ));
        until_submitted(&model, in_flight.as_mut())?;
        let handles = model.diagnostic_launch_handles();
        cancelled.store(true, Ordering::Release);
        model.diagnostic_resume_launch();
        let pending = ready(in_flight.await?)?;
        let completed = pending.readback().dispatches;
        expected.compare_logits(next, &pending.readback().logits)?;
        need(
            completed > 0 && handles > 0,
            "cancelled submitted work completed with actual accounting",
        )?;
        let discarded = session.commit(pending)?;
        match discarded {
            ResidentTextPublication::Discarded(read) => need(
                read.dispatches == completed,
                "known discard preserves completed accounting",
            )?,
            ResidentTextPublication::Committed(_) => {
                return Err(Error::State("cancelled call published".into()));
            }
        }
        need(
            session.binding() == stable,
            "known completed discard does not publish",
        )?;
        check(&session, expected, first).await?;
        let pending = ready(prepare(&session, &plan, next, Some(actual.0), None).await?)?;
        let second = selected(&pending.readback().logits)?;
        need(
            second == (next.selected, next.max_bits),
            "cached actual feedback",
        )?;
        expected.compare_logits(next, &pending.readback().logits)?;
        drop(session.commit(pending)?);
        check(&session, expected, next).await?;
        let pending = ready(prepare(&session, &plan, last, Some(second.0), None).await?)?;
        expected.compare_logits(last, &pending.readback().logits)?;
        drop(session.commit(pending)?);
        check(&session, expected, last).await?;
        session.restore(&saved)?;
        check(&session, expected, first).await?;
        let replay = ready(prepare(&session, &plan, next, Some(actual.0), None).await?)?;
        expected.compare_logits(next, &replay.readback().logits)?;
        drop(session.commit(replay)?);
        check(&session, expected, next).await?;
        session.reset().await?;
        need(
            session.binding().frontier() == 0 && session.binding() != root0,
            "reset advances original root authority",
        )?;
        let replay = ready(prepare(&session, &plan, first, None, None).await?)?;
        expected.compare_logits(first, &replay.readback().logits)?;
        drop(session.commit(replay)?);
        check(&session, expected, first).await?;
        let pending = ready(prepare(&peer, &plan, first, None, None).await?)?;
        expected.compare_logits(first, &pending.readback().logits)?;
        drop(peer.commit(pending)?);
        check(&peer, expected, first).await?;
        need(
            session.binding().session() != peer.binding().session(),
            "independent source sessions",
        )?;
        drop(saved);
        drop(session);
        drop(peer);
        drop(plan);
        model.shutdown_cpu_forward_source().await?;
        println!(
            "SOURCE_LIFECYCLE_HEALTHY prelaunch_cancel=true capacity=true session_limit=true cross_session_refusal=true submitted_cancel=completed_discard dispatches={completed} snapshot_restore_reset_replay=true independent_sessions=true"
        );
    } else if mode == "unknown" {
        let mut peer = model.session().await?;
        let first_pending = ready(prepare(&session, &plan, first, None, None).await?)?;
        let actual = selected(&first_pending.readback().logits)?;
        expected.compare_logits(first, &first_pending.readback().logits)?;
        drop(session.commit(first_pending)?);
        check(&session, expected, first).await?;
        let prior = session.snapshot()?;
        let prior_bytes = prior.read_domains().await?;
        let stable = session.binding();
        let peer0 = peer.binding();
        model.diagnostic_pause_next_launch()?;
        let mut in_flight = Box::pin(prepare(&session, &plan, next, Some(actual.0), None));
        until_submitted(&model, in_flight.as_mut())?;
        let handles = model.diagnostic_launch_handles();
        drop(in_flight);
        need(
            session.binding() == stable && peer.binding() == peer0,
            "unknown completion never published either root",
        )?;
        need(
            prepare(&peer, &plan, first, None, None).await.is_err(),
            "unknown fenced peer cannot prepare",
        )?;
        need(
            session.reset().await.is_err() && peer.reset().await.is_err(),
            "unknown cannot reset/fallback",
        )?;
        let pending = native.observe()?;
        need(
            pending.pending_events > 0
                && pending.quarantined_handles >= handles
                && pending.completion_owners > 0
                && !pending.service_usable
                && pending.retained,
            "original unknown flight retains completion, handles and native owners",
        )?;
        let refusal = match prior.read_domains().await {
            Err(Error::State(reason))
                if reason == "source binding closed or physical owner fenced"
                    || reason == "resident service fenced" =>
            {
                reason
            }
            Err(error) => return Err(error),
            Ok(_) => {
                return Err(Error::State(
                    "post-fence snapshot started a new GPU read".into(),
                ));
            }
        };
        let preserved = expected.compare_domains(first, &prior_bytes, 514)?.0;
        drop(prior_bytes);
        drop(prior);
        drop(session);
        drop(peer);
        drop(plan);
        drop(model);
        println!(
            "SOURCE_LIFECYCLE_UNKNOWN before_unknown_state_words={preserved} post_unknown_state_words_unavailable=true read_refusal={refusal:?} launch_handles={handles} pending_events={} retained_handles={} unchanged_roots=true peer_refused=true pending_outputs_unavailable=true",
            pending.pending_events, pending.quarantined_handles
        );
    } else {
        let pending = ready(prepare(&session, &plan, first, None, None).await?)?;
        expected.compare_logits(first, &pending.readback().logits)?;
        if mode.starts_with("pending-") {
            drop(session);
            drop(plan);
            drop(model);
            let probe = pending.diagnostic_final_reader(mode.ends_with("-fault"))?;
            drop(pending);
            final_reader(probe)?;
        } else {
            let ResidentTextPublication::Committed(output) = session.commit(pending)? else {
                return Err(Error::State("live source proposal discarded".into()));
            };
            check(&session, expected, first).await?;
            match mode {
                "session-known" | "session-fault" => {
                    drop(output);
                    drop(plan);
                    drop(model);
                    let probe = session.diagnostic_final_reader(mode.ends_with("-fault"))?;
                    drop(session);
                    final_reader(probe)?;
                }
                "snapshot-known" | "snapshot-fault" => {
                    let snapshot = session.snapshot()?;
                    drop(output);
                    drop(session);
                    drop(plan);
                    drop(model);
                    let probe = snapshot.diagnostic_final_reader(mode.ends_with("-fault"))?;
                    drop(snapshot);
                    final_reader(probe)?;
                }
                "readback-known" => {
                    drop(session);
                    drop(plan);
                    let weak = Arc::downgrade(&model);
                    need(
                        model.shutdown_cpu_forward_source().await.is_err()
                            && weak.strong_count() > 0,
                        "original final readback prevents premature shutdown",
                    )?;
                    need(
                        native.observe()?.retained,
                        "readback retains original native owners",
                    )?;
                    expected.compare_logits(first, &output.logits)?;
                    drop(output);
                    need(
                        weak.strong_count() == 0,
                        "final readback releases actual model",
                    )?;
                }
                "snapshot-readback-known" => {
                    let snapshot = session.snapshot()?;
                    let data = snapshot.read_domains().await?;
                    drop(snapshot);
                    drop(output);
                    drop(session);
                    drop(plan);
                    let weak = Arc::downgrade(&model);
                    need(
                        model.shutdown_cpu_forward_source().await.is_err()
                            && weak.strong_count() > 0,
                        "snapshot host reader prevents premature shutdown",
                    )?;
                    need(
                        native.observe()?.retained,
                        "snapshot host reader retains native/source",
                    )?;
                    expected.compare_domains(first, &data, 514)?;
                    drop(data);
                    need(
                        weak.strong_count() == 0,
                        "snapshot readback releases actual model",
                    )?;
                }
                _ => unreachable!(),
            }
        }
    }
    let after = native.observe()?;
    let fault = mode.ends_with("-fault") || mode == "unknown";
    let global = ResourceArbiter::process_shared().snapshot(&resource)?;
    if fault {
        need(
            after.retained
                && !after.service_usable
                && global.fenced
                && after.modules == before.modules
                && after.code_bytes == before.code_bytes
                && after.host_bytes == before.host_bytes
                && after.device_bytes == before.device_bytes,
            "fault retains original entire native set and physical fence",
        )?;
        need(
            ResourceArbiter::process_shared()
                .admit(
                    &resource,
                    ResourceOwner::new("resident-text-qualification", "fenced-peer")?,
                    &[AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: 4096,
                    }],
                )
                .is_err(),
            "same physical device refuses peer admission after final-reader fault",
        )?;
    } else {
        need(
            !after.retained
                && after.modules == 0
                && after.resident_bytes == 0
                && !global.fenced
                && global.reserved == 4 * 1024 * 1024 * 1024,
            "known final reader retires original native/state to persistent context",
        )?;
    }
    println!(
        "RESIDENT_TEXT_LIFECYCLE_PASS mode={mode} observer=false native_retained={} fenced={} resident_bytes={} device_reserved_bytes={} sdk=false",
        after.retained, global.fenced, after.resident_bytes, global.reserved
    );
    Ok(())
}
