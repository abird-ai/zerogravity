//! Actual-source lifetime cases; never a production caller or source oracle.
use super::{
    drive,
    reference::Expected,
    source::{Loaded, need},
    source_pins::Pins,
};
use std::{future::Future, sync::Arc, task::Poll};
use zerogravity::gpu::{SourceTextCandidate, SourceTextNativeState};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity::{Error, Result};

#[derive(Clone, Copy)]
pub enum Mode {
    Session(bool),
    Snapshot(bool),
    Pending(bool),
    Unknown,
}
impl Mode {
    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "session-known" => Self::Session(false),
            "session-fault" => Self::Session(true),
            "snapshot-known" => Self::Snapshot(false),
            "snapshot-fault" => Self::Snapshot(true),
            "pending-known" => Self::Pending(false),
            "pending-fault" => Self::Pending(true),
            "unknown" => Self::Unknown,
            _ => return None,
        })
    }
    fn retained(self) -> bool {
        match self {
            Self::Session(f) | Self::Snapshot(f) | Self::Pending(f) => f,
            Self::Unknown => true,
        }
    }
}
fn retained(before: &SourceTextNativeState, after: &SourceTextNativeState) -> Result<()> {
    need(
        after.retained
            && after.modules == before.modules
            && after.code_bytes == before.code_bytes
            && after.host_bytes == before.host_bytes
            && after.device_bytes == before.device_bytes
            && after.service_retained
            && !after.service_usable
            && after.quarantined_handles > 0,
        "original native/source/service/handles retained after uncertainty",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let physical = arbiter.snapshot(&after.physical)?;
    need(
        physical.fenced && physical.reserved > 0,
        "original physical domain remains fenced/charged",
    )?;
    need(
        arbiter
            .admit(
                &after.physical,
                ResourceOwner::new("other-model", "source-fence-control")?,
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 1,
                }],
            )
            .is_err(),
        "physical fence refuses another owner before work",
    )?;
    println!(
        "SOURCE_MODEL_FENCE reserved={} quarantined={} pending={} completion_owners={} recovery=false",
        physical.reserved, after.quarantined_handles, after.pending_events, after.completion_owners
    );
    Ok(())
}
pub fn run(
    mode: Mode,
    name: &str,
    label: &str,
    candidate: Arc<SourceTextCandidate>,
    loaded: Arc<Loaded>,
    expected: &mut Expected,
    pins: Pins,
) -> Result<()> {
    pins.require(true)?;
    let source = Arc::downgrade(&loaded);
    let cache = loaded.cache_witness()?;
    let host_arbiter = loaded.host.arbiter.clone();
    let host_key = loaded.host.resource.clone();
    let witness = candidate.native_probe();
    let before = witness.observe()?;
    need(
        before.retained
            && before.modules > 0
            && before.service_usable
            && before.pending_events == 0
            && before.quarantined_handles == 0,
        "fresh native lifetime scope",
    )?;
    let mut session = drive::block_on(candidate.session())?;
    let mut observer = expected.observe(0, 0, 11, None)?;
    let pending = drive::block_on(session.prepare(11, Some(&mut observer), None))?;
    observer.finish()?;
    println!(
        "SOURCE_MODEL_LIFETIME_SEED records={} words={} mismatches=0",
        observer.completed_records(),
        observer.words
    );
    drop(observer);
    let selected = super::selected(pending.logits())?;
    let call = expected.call(0, 0)?;
    need(
        selected == (call.selected, call.max_bits),
        "lifetime seed actual original fullV selection",
    )?;
    loaded.verify_after()?;
    // Consume the caller's real owner; no extra clone fabricates close refusal.
    need(
        loaded.close().is_err(),
        "live native owner refuses original source close",
    )?;
    if matches!(mode, Mode::Unknown) {
        drop(session.commit(pending)?);
        let root = session.binding();
        let peer = drive::block_on(candidate.session())?;
        let peer_root = peer.binding();
        candidate.diagnostic_pause_next_launch()?;
        let mut observer = expected.observe(0, 1, selected.0, None)?;
        let mut future = Box::pin(session.prepare(selected.0, Some(&mut observer), None));
        let handles = drive::block_on(std::future::poll_fn(|cx| match future.as_mut().poll(cx) {
            Poll::Ready(_) => Poll::Ready(Err(Error::State(
                "armed source flight completed unexpectedly".into(),
            ))),
            Poll::Pending => {
                let n = candidate.diagnostic_launch_handles();
                if n > 0 {
                    Poll::Ready(Ok(n))
                } else {
                    Poll::Pending
                }
            }
        }))?;
        drop(future); // Abandon the actual original armed flight, never a surrogate.
        observer.finish_before_logits()?;
        println!(
            "SOURCE_MODEL_LIFETIME_PARTIAL position=1 records={} words={} mismatches=0 logits_observed=false root_published=false handles={handles}",
            observer.completed_records(),
            observer.words
        );
        drop(observer);
        need(
            session.binding() == root && peer.binding() == peer_root,
            "unknown cannot publish or replace either original root",
        )?;
        need(
            drive::block_on(peer.prepare(11, None, None)).is_err(),
            "fenced model refuses existing peer",
        )?;
        drop(peer);
        need(
            drive::block_on(candidate.session()).is_err(),
            "fenced model refuses new session with a free permit",
        )?;
        let paused = witness.observe()?;
        need(
            paused.pending_events == 1
                && paused.quarantined_handles >= handles
                && paused.completion_owners > 0,
            "actual unknown flight retains all handles/event/owners",
        )?;
        drop(session);
        drop(candidate);
        let after = witness.observe()?;
        retained(&before, &after)?;
        need(
            after.pending_events == 1 && after.completion_owners > 0,
            "unknown survives every caller wrapper drop",
        )?;
    } else {
        let boundary = match mode {
            Mode::Pending(fault) => {
                drop(session);
                drop(candidate);
                let boundary = pending.diagnostic_final_reader(fault)?;
                drop(pending);
                boundary
            }
            Mode::Snapshot(fault) => {
                drop(session.commit(pending)?);
                let snapshot = session.snapshot()?;
                drop(session);
                drop(candidate);
                let boundary = snapshot.diagnostic_final_reader(fault)?;
                drop(snapshot);
                boundary
            }
            Mode::Session(fault) => {
                drop(session.commit(pending)?);
                drop(candidate);
                let boundary = session.diagnostic_final_reader(fault)?;
                drop(session);
                boundary
            }
            Mode::Unknown => unreachable!(),
        };
        let result = boundary.observe()?;
        need(
            !result.candidate_retained
                && result.release_hits == 1
                && result.native_retirements == 1
                && result.reader_hits_before_native == 1,
            "actual final reader precedes native retirement exactly once",
        )?;
        if mode.retained() {
            retained(&result.before, &result.after)?;
        } else {
            need(
                !result.after.retained
                    && result.after.modules == 0
                    && result.after.code_bytes == 0
                    && result.after.host_bytes == 0
                    && result.after.device_bytes == 0
                    && !result.after.service_retained
                    && result.after.quarantined_handles == 0,
                "known final wrapper retires native/service",
            )?;
            let arbiter = ResourceArbiter::process_shared();
            let physical = arbiter.snapshot(&result.after.physical)?;
            need(
                !physical.fenced,
                "known retirement leaves original physical domain usable",
            )?;
            let peer = arbiter.admit(
                &result.after.physical,
                ResourceOwner::new("other-model", "source-retired-admission-control")?,
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 1,
                }],
            )?;
            drop(peer);
            println!(
                "SOURCE_MODEL_HEALTHY_PHYSICAL reserved={} fenced=false peer_admission=true native_peer_work=false",
                physical.reserved
            );
        }
        println!(
            "SOURCE_MODEL_FINAL_READER bytes={} aliases={} release_hits={} native_attempts={} hits_before_native={} modules_before={} modules_after={} retained={}",
            result.bytes,
            result.aliases,
            result.release_hits,
            result.native_retirements,
            result.reader_hits_before_native,
            result.before.modules,
            result.after.modules,
            result.after.retained
        );
    }
    let retained_source = source.upgrade();
    need(
        retained_source.is_some() == mode.retained()
            && cache.upgrade().is_some() == mode.retained(),
        "original source/cache final disposition",
    )?;
    if let Some(source) = retained_source {
        source.verify_after()?;
        need(
            source.close().is_err(),
            "uncertain native owner still refuses source close",
        )?;
    }
    pins.require(mode.retained())?;
    let host = host_arbiter.snapshot(&host_key)?;
    println!(
        "SOURCE_MODEL_LIFETIME_PASS model={label} mode={name} source_retained={} cache_retained={} cache_files={} host_reserved={} host_fenced={} completed_seed_calls=1 ordinary_serving_qualified=false",
        mode.retained(),
        mode.retained(),
        pins.len(),
        host.reserved,
        host.fenced
    );
    Ok(())
}
