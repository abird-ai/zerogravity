//! Scheduling/status controls only: no tensor source, math, readout or successful
//! publication. Actual inference acceptance belongs to real-source coverage.
use super::*;
use crate::model_instance::{PreparedModel, TextTokenizer};
use std::collections::BTreeMap;
use std::{
    convert::Infallible,
    rc::Rc,
    sync::{Mutex, atomic::AtomicUsize},
};
use zerogravity_core::{Error, identity::SourceIdentity, resource_arbiter::AllocationClass};

type Log = Arc<Mutex<Vec<String>>>;
type Compatibility = fn(usize, usize) -> bool;
#[derive(Clone, Copy)]
enum Action {
    Capacity,
    CancelBeforeLaunch,
    Fatal,
    ReadyCancel,
    ReadyDiscard,
}
#[derive(Default)]
struct Control {
    group: usize,
    next: AtomicUsize,
    live: AtomicUsize,
    actions: Mutex<VecDeque<Action>>,
    state_only: Mutex<Vec<bool>>,
    compatibility: Mutex<Option<Compatibility>>,
    log: Log,
}
struct Model {
    source: SourceIdentity,
    control: Arc<Control>,
    prefill_chunk: usize,
}
struct Tokenizer;
impl TextTokenizer for Tokenizer {
    fn eos(&self) -> u32 {
        0
    }
    fn decode(&self, _: &[u32], _: bool) -> zerogravity_core::Result<String> {
        Err(Error::Unsupported("policy fixture has no decoder".into()))
    }
    fn prompt_tokens(&self, _: &str, _: bool) -> zerogravity_core::Result<Vec<u32>> {
        Ok(vec![1, 1])
    }
    fn prompt_token_bound(&self, _: usize, _: bool) -> usize {
        2
    }
    fn admitted_prompt_scratch_bound(&self, _: usize, _: bool) -> zerogravity_core::Result<u64> {
        Ok(0)
    }
    fn decode_scratch_bound(&self, _: usize) -> usize {
        0
    }
}
impl PreparedModel for Model {
    fn source(&self) -> SourceIdentity {
        self.source
    }
    fn tokenizer(&self) -> Arc<dyn TextTokenizer> {
        Arc::new(Tokenizer)
    }
    fn max_context(&self) -> usize {
        512
    }
    fn max_token_bytes(&self) -> usize {
        1
    }
    fn vocabulary_size(&self) -> u64 {
        2
    }
}
struct State {
    id: usize,
    control: Arc<Control>,
    _affine: Rc<()>,
}
impl Drop for State {
    fn drop(&mut self) {
        self.control
            .log
            .lock()
            .unwrap()
            .push(format!("state-drop:{}:{}", self.control.group, self.id));
    }
}
struct Ticket {
    width: usize,
    control: Arc<Control>,
}
impl Drop for Ticket {
    fn drop(&mut self) {
        assert_eq!(self.control.live.fetch_sub(1, Ordering::AcqRel), 1);
        self.control
            .log
            .lock()
            .unwrap()
            .push(format!("ready-drop:{}:{}", self.control.group, self.width));
    }
}
struct StatusOnly;
fn refusal() -> ServiceError {
    ServiceError::new(
        500,
        "no_execution",
        "policy fixture cannot execute or publish",
    )
}
impl Executor for StatusOnly {
    type Model = Arc<Model>;
    type State = State;
    type Lease = ();
    type Ready = Ticket;
    fn open(
        model: &Self::Model,
        _: &str,
        _: &Arc<AtomicBool>,
        _: &HostPool,
    ) -> Result<(State, ()), ServiceError> {
        Ok((
            State {
                id: model.control.next.fetch_add(1, Ordering::Relaxed),
                control: model.control.clone(),
                _affine: Rc::new(()),
            },
            (),
        ))
    }
    fn prefill(model: &Self::Model) -> Prefill {
        Prefill::Chunks(model.prefill_chunk)
    }
    fn group(model: &Self::Model) -> usize {
        model.control.group
    }
    fn frontier(_: &State) -> usize {
        0
    }
    fn selected(_: &mut State) -> Result<u32, ServiceError> {
        Err(refusal())
    }
    fn has_feedback(_: &State) -> bool {
        false
    }
    fn single(row: &mut Active<Self>) -> Result<SingleCompleted, ServiceError> {
        assert_eq!(row.state.control.live.load(Ordering::Acquire), 0);
        row.state.control.log.lock().unwrap().push(format!(
            "single:{}:{}",
            row.state.control.group, row.state.id
        ));
        Err(refusal())
    }
    fn cohort_eligible(_: &Active<Self>) -> bool {
        true
    }
    fn compatible(first: &Active<Self>, row: &Active<Self>) -> bool {
        first
            .state
            .control
            .compatibility
            .lock()
            .unwrap()
            .is_none_or(|compatible| compatible(first.state.id, row.state.id))
    }
    fn prepare(
        rows: &mut [Active<Self>],
        work: CohortWork,
    ) -> Result<Preparation<Ticket>, ServiceError> {
        let control = rows[0].state.control.clone();
        control.state_only.lock().unwrap().push(work.state_only);
        assert_eq!(
            control.live.load(Ordering::Acquire),
            0,
            "previous private Ready still retained"
        );
        control
            .log
            .lock()
            .unwrap()
            .push(format!("prepare:{}:{}", control.group, rows.len()));
        let action = control
            .actions
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(Action::Fatal);
        match action {
            Action::Capacity => Ok(Preparation::Capacity),
            Action::CancelBeforeLaunch => {
                rows[0].cancelled.store(true, Ordering::Release);
                Ok(Preparation::CancelledBeforeLaunch)
            }
            Action::Fatal => Err(refusal()),
            Action::ReadyCancel | Action::ReadyDiscard => {
                if matches!(action, Action::ReadyCancel) {
                    rows[0].cancelled.store(true, Ordering::Release);
                }
                assert_eq!(control.live.fetch_add(1, Ordering::AcqRel), 0);
                Ok(Preparation::Ready(Completed {
                    ready: Ticket {
                        width: rows.len(),
                        control,
                    },
                    fused: true,
                    completion: Completion {
                        quanta: rows.len() as u64,
                        forwards: 1,
                        model_tokens: rows.len() as u64,
                        cohort_width: rows.len(),
                        state_only: work.state_only,
                        ..Completion::default()
                    },
                }))
            }
        }
    }
    fn commit(
        ready: Ticket,
        _: &mut [Active<Self>],
        _: CohortWork,
    ) -> Result<Publication, ServiceError> {
        ready
            .control
            .log
            .lock()
            .unwrap()
            .push(format!("discard:{}:{}", ready.control.group, ready.width));
        Ok(Publication::Discarded)
    }
    fn diagnostics(
        _: &BTreeMap<String, Self::Model>,
        _: &SchedulerSnapshot,
    ) -> Result<serde_json::Value, ServiceError> {
        Ok(serde_json::json!({"kind":"policy-control-only"}))
    }
}
struct Rig {
    host: HostPool,
    queue: pending::PendingQueue<()>,
    stats: Arc<WorkStats>,
    receivers: Vec<event_channel::mpsc::Receiver<ResponseEvent>>,
    _charge: ServingCharge,
}
impl Rig {
    fn new() -> Self {
        let host = HostPool::isolated(256).unwrap();
        let charge = host
            .charge(
                "policy-test",
                "bounded-control",
                AllocationClass::Scratch,
                1024 * 1024,
            )
            .unwrap();
        Self {
            queue: pending::PendingQueue::new(host.clone(), Some(1)).unwrap(),
            host,
            stats: Arc::default(),
            receivers: Vec::new(),
            _charge: charge,
        }
    }
    fn rows(
        &mut self,
        group: usize,
        count: usize,
        actions: Vec<Action>,
        policy: WavePolicy,
        log: &Log,
    ) -> VecDeque<Active<StatusOnly>> {
        self.rows_with_prefill(group, count, actions, policy, log, 1)
    }
    fn rows_with_prefill(
        &mut self,
        group: usize,
        count: usize,
        actions: Vec<Action>,
        policy: WavePolicy,
        log: &Log,
        prefill_chunk: usize,
    ) -> VecDeque<Active<StatusOnly>> {
        let model = Arc::new(Model {
            source: SourceIdentity::for_new_allocation().unwrap(),
            prefill_chunk,
            control: Arc::new(Control {
                group,
                next: AtomicUsize::new(0),
                live: AtomicUsize::new(0),
                actions: Mutex::new(actions.into()),
                state_only: Mutex::new(Vec::new()),
                compatibility: Mutex::new(None),
                log: log.clone(),
            }),
        });
        let mut rows = VecDeque::new();
        for _ in 0..count {
            let (events, receiver) =
                event_channel::mpsc::try_sync_channel(&self.host, "policy-test", "events", 4)
                    .unwrap();
            self.receivers.push(receiver);
            let cancel = self.queue.cancel_token(None);
            let job = Scheduled {
                payload: Some(RequestLease {
                    request: Request {
                        model: "policy-test".into(),
                        input: "fixture".into(),
                        chat: false,
                        max_tokens: 1,
                        stream: false,
                        token_metrics: None,
                    },
                    prefix_sharing: None,
                    _charges: Arc::new(Vec::new()),
                }),
                metrics: MetricsLease {
                    metrics: None,
                    _charges: None,
                },
                events,
                cancelled: cancel.flag().clone(),
                cancel,
                prompt_charge: None,
                transport: None,
                model_reader: None,
            };
            rows.push_back(
                Active::<StatusOnly>::new(
                    job,
                    model.clone(),
                    WorkerConfig {
                        workers: 1,
                        active: 8,
                        context: 512,
                        output: 1,
                        max_pending: Some(1),
                        policy: WorkPolicy::Waves(policy),
                    },
                    &self.host,
                    self.stats.clone(),
                )
                .unwrap(),
            );
        }
        rows
    }
    fn close(self) {
        assert_eq!(self.stats.snapshot().active, 0);
        assert_eq!(self.stats.snapshot().committed_roots, 0);
        assert_eq!(self.stats.snapshot().output_tokens, 0);
        let host = self.host.clone();
        drop(self);
        assert_eq!(host.snapshot().unwrap().3, 0);
    }
}
fn policy(rows: usize, retry: RetryOrder) -> WavePolicy {
    WavePolicy {
        max_members: rows,
        max_tokens: 64,
        max_scratch: 256 * 1024 * 1024,
        fairness: Fairness {
            round_tokens: 72,
            prefill_age_rounds: 2,
        },
        prefill_quantum: 32,
        unequal_prefill: true,
        unequal_decode: true,
        cohort_width: rows,
        cohorts: Cohorts::Fused {
            widths: if retry == RetryOrder::Regroup {
                &[8, 4, 2, 1]
            } else {
                &[4, 1]
            },
            split_prefix: retry == RetryOrder::Regroup,
            retry,
        },
    }
}
fn turn(rows: &mut VecDeque<Active<StatusOnly>>, rig: &Rig, policy: WavePolicy) {
    advance_wave(
        rows,
        &mut VecDeque::new(),
        &mut VecDeque::new(),
        &rig.stats,
        policy,
    );
}
fn operations(log: &Log) -> Vec<String> {
    log.lock()
        .unwrap()
        .iter()
        .filter(|s| !s.starts_with("state-drop:"))
        .cloned()
        .collect()
}

#[test]
fn prelaunch_capacity_reduces_only_supported_cohorts_without_padding() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(8, RetryOrder::Regroup);
    let mut rows = rig.rows(1, 8, vec![Action::Capacity; 7], policy, &log);
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(
        operations(&log),
        [
            "prepare:1:8",
            "prepare:1:4",
            "prepare:1:2",
            "single:1:0",
            "single:1:1",
            "prepare:1:2",
            "single:1:2",
            "single:1:3",
            "prepare:1:4",
            "prepare:1:2",
            "single:1:4",
            "single:1:5",
            "prepare:1:2",
            "single:1:6",
            "single:1:7"
        ]
    );
    assert_eq!(rig.stats.snapshot().cohort_admission_refusals, 7);
    assert_eq!(rig.stats.snapshot().completed, 0);
    rig.close();
}
#[test]
fn known_completed_cancel_drops_ready_before_regrouping_survivors() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(8, RetryOrder::Regroup);
    let mut rows = rig.rows(1, 8, vec![Action::ReadyCancel], policy, &log);
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(
        operations(&log),
        [
            "prepare:1:8",
            "ready-drop:1:8",
            "prepare:1:4",
            "prepare:1:2",
            "single:1:7"
        ]
    );
    let stats = rig.stats.snapshot();
    assert_eq!(
        (
            stats.completed,
            stats.model_tokens,
            stats.cancelled,
            stats.discarded_cohorts
        ),
        (8, 8, 1, 1)
    );
    rig.close();
}
#[test]
fn bounded_prefill_round_retains_deferred_owners_and_rotates_oldest_first() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let mut policy = policy(2, RetryOrder::Regroup);
    policy.max_tokens = 2;
    policy.fairness.round_tokens = 2;
    policy.fairness.prefill_age_rounds = 2;
    let mut rows = rig.rows(
        1,
        6,
        vec![Action::Fatal, Action::Fatal, Action::Fatal],
        policy,
        &log,
    );

    turn(&mut rows, &rig, policy);
    assert_eq!(
        rows.iter().map(|row| row.state.id).collect::<Vec<_>>(),
        [2, 3, 4, 5]
    );
    assert!(rows.iter().all(|row| row.prefill_wait_rounds == 1));

    turn(&mut rows, &rig, policy);
    assert_eq!(
        rows.iter().map(|row| row.state.id).collect::<Vec<_>>(),
        [4, 5]
    );
    assert!(rows.iter().all(|row| row.prefill_wait_rounds == 2));

    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(
        operations(&log),
        ["prepare:1:2", "prepare:1:2", "prepare:1:2"]
    );
    rig.close();
}

#[test]
fn captured_discard_returns_to_next_round_without_overtaking_other_ready_work() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(4, RetryOrder::NextRound);
    let mut rows = rig.rows(1, 4, vec![Action::ReadyDiscard], policy, &log);
    rows.extend(rig.rows(2, 4, vec![Action::Fatal], policy, &log));
    turn(&mut rows, &rig, policy);
    assert_eq!(rows.len(), 4);
    assert_eq!(
        operations(&log),
        [
            "prepare:1:4",
            "discard:1:4",
            "ready-drop:1:4",
            "prepare:2:4"
        ]
    );
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(operations(&log).last().unwrap(), "prepare:1:4");
    rig.close();
}
#[test]
fn fatal_or_unknown_preparation_is_never_retried_as_smaller_work() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(8, RetryOrder::Regroup);
    let mut rows = rig.rows(1, 8, vec![Action::Fatal], policy, &log);
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(operations(&log), ["prepare:1:8"]);
    assert_eq!(
        (
            rig.stats.snapshot().failed,
            rig.stats.snapshot().cohort_admission_refusals
        ),
        (8, 0)
    );
    rig.close();
}
#[test]
fn admitted_session_high_water_survives_warmup_and_completed_drops() {
    let stats = Arc::new(WorkStats::default());
    let warmup = stats.enter();
    assert_eq!(stats.snapshot().active, 1);
    drop(warmup);
    assert_eq!((stats.snapshot().active, stats.snapshot().peak), (0, 1));
    let mut sessions = (0..8).map(|_| stats.enter()).collect::<Vec<_>>();
    assert_eq!((stats.snapshot().active, stats.snapshot().peak), (8, 8));
    sessions.truncate(2);
    let next = stats.enter();
    assert_eq!(stats.snapshot().peak, 8);
    drop(next);
    drop(sessions);
    assert_eq!((stats.snapshot().active, stats.snapshot().peak), (0, 8));
}

// No session or inference exists in the real-thread retirement fixture.
struct Retirement<const FAIL: bool>;
impl<const FAIL: bool> Executor for Retirement<FAIL> {
    type Model = Arc<Model>;
    type State = ();
    type Lease = ();
    type Ready = Infallible;
    fn open(
        _: &Self::Model,
        _: &str,
        _: &Arc<AtomicBool>,
        _: &HostPool,
    ) -> Result<((), ()), ServiceError> {
        Err(refusal())
    }
    fn prefill(_: &Self::Model) -> Prefill {
        Prefill::Chunks(1)
    }
    fn group(_: &Self::Model) -> usize {
        0
    }
    fn frontier(_: &()) -> usize {
        0
    }
    fn selected(_: &mut ()) -> Result<u32, ServiceError> {
        Err(refusal())
    }
    fn has_feedback(_: &()) -> bool {
        false
    }
    fn single(_: &mut Active<Self>) -> Result<SingleCompleted, ServiceError> {
        Err(refusal())
    }
    fn selective_retirement() -> Option<ModelRetirement<Self::Model>> {
        Some(|_| if FAIL { Err(refusal()) } else { Ok(()) })
    }
    fn retire(_: &BTreeMap<String, Self::Model>) -> Result<(), ServiceError> {
        if FAIL { Err(refusal()) } else { Ok(()) }
    }
    fn diagnostics(
        _: &BTreeMap<String, Self::Model>,
        _: &SchedulerSnapshot,
    ) -> Result<serde_json::Value, ServiceError> {
        Ok(serde_json::json!({"kind":"retirement-only"}))
    }
}

#[test]
fn shutdown_rejects_new_retirement_failures_after_real_join_and_keeps_them_sticky() {
    fn run<const FAIL: bool>(previous: u64) {
        let host = HostPool::isolated(256).unwrap();
        let mut router = Router::retirement_test::<Retirement<FAIL>>(host.clone());
        router.retirement_counter(previous);
        assert_eq!(router.shutdown().is_err(), FAIL);
        assert_eq!(router.shutdown().is_err(), FAIL);
        drop(router);
        assert_eq!(host.snapshot().unwrap().3, 0);
    }
    run::<false>(7);
    run::<true>(7);
    run::<true>(u64::MAX);
}

#[test]
fn failed_selective_retirement_is_visible_and_never_reopens_admission() {
    use crate::{
        Runtime,
        diagnostics::{ModelLifecycle, RuntimePhase},
    };
    let host = HostPool::isolated(256).unwrap();
    let model = || {
        Arc::new(Model {
            source: SourceIdentity::for_new_allocation().unwrap(),
            control: Arc::default(),
            prefill_chunk: 1,
        })
    };
    let a = model();
    let b = model();
    let runtime = Runtime::builder().build().unwrap();
    runtime
        .configure_text(|| {
            Router::with_executor::<Retirement<true>>(
                BTreeMap::from([("a".into(), a), ("b".into(), b)]),
                host.clone(),
                WorkerConfig {
                    workers: 1,
                    active: 1,
                    context: 16,
                    output: 1,
                    max_pending: Some(1),
                    policy: WorkPolicy::Sequential,
                },
            )
        })
        .unwrap();
    let target = runtime.models().resolve("a").unwrap();
    assert!(runtime.models().unload(&target).is_err());
    let failed = runtime.snapshot().unwrap();
    assert_eq!(failed.status.phase, RuntimePhase::Ready);
    assert!(failed.status.failed && failed.failure.is_some());
    assert_eq!(failed.catalog[0].lifecycle, ModelLifecycle::Failed);
    assert!(failed.catalog[0].capabilities.is_none());
    assert_eq!(failed.catalog[1].lifecycle, ModelLifecycle::Ready);
    assert!(runtime.models().resolve("a").is_err());
    assert!(runtime.models().resolve("b").is_ok());
    assert!(runtime.shutdown().is_err());
    let closed = runtime.snapshot().unwrap();
    assert_eq!(closed.status.phase, RuntimePhase::Closed);
    assert_eq!(closed.failure, failed.failure);
    drop((target, runtime));
    assert_eq!(host.snapshot().unwrap().1, 0);
}

#[test]
fn captured_capacity_falls_directly_back_to_original_singletons() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(4, RetryOrder::NextRound);
    let mut rows = rig.rows(1, 4, vec![Action::Capacity], policy, &log);
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(
        operations(&log),
        [
            "prepare:1:4",
            "single:1:0",
            "single:1:1",
            "single:1:2",
            "single:1:3"
        ]
    );
    assert_eq!(rig.stats.snapshot().completed, 0);
    rig.close();
}
#[test]
fn prelaunch_cancel_keeps_captured_survivors_for_the_next_fair_round() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(4, RetryOrder::NextRound);
    let mut rows = rig.rows(1, 4, vec![Action::CancelBeforeLaunch], policy, &log);
    turn(&mut rows, &rig, policy);
    assert_eq!(rows.len(), 3);
    assert_eq!(operations(&log), ["prepare:1:4"]);
    assert_eq!(
        (
            rig.stats.snapshot().completed,
            rig.stats.snapshot().cancelled
        ),
        (0, 1)
    );
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(
        operations(&log),
        ["prepare:1:4", "single:1:1", "single:1:2", "single:1:3"]
    );
    rig.close();
}
#[test]
fn non_fused_prompt_tails_still_enter_one_complete_cohort() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let mut policy = policy(4, RetryOrder::Regroup);
    policy.cohorts = Cohorts::PromptTails;
    policy.cohort_width = 0;
    let mut rows = rig.rows(1, 3, vec![Action::Fatal], policy, &log);
    // Scheduling inputs only; the fixture refuses every physical operation.
    for row in &mut rows {
        row.cursor = 1;
    }
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(operations(&log), ["prepare:1:3"]);
    rig.close();
}

#[test]
fn chunked_prefill_split_marks_only_nonterminal_chunks_state_only() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let mut policy = policy(2, RetryOrder::Regroup);
    policy.unequal_prefill = false;

    // The fixture prompt has two tokens. With a two-token chunk this is the
    // terminal prefill span and must request readout, not state-only output.
    let mut terminal = rig.rows_with_prefill(1, 2, vec![Action::Fatal], policy, &log, 2);
    let terminal_control = terminal[0].state.control.clone();
    turn(&mut terminal, &rig, policy);
    assert!(terminal.is_empty());
    assert_eq!(*terminal_control.state_only.lock().unwrap(), [false]);

    // With one-token chunks the same prompt has another prefill span pending,
    // so the first cohort is correctly state-only.
    let mut prefix = rig.rows_with_prefill(2, 2, vec![Action::Fatal], policy, &log, 1);
    let prefix_control = prefix[0].state.control.clone();
    turn(&mut prefix, &rig, policy);
    assert!(prefix.is_empty());
    assert_eq!(*prefix_control.state_only.lock().unwrap(), [true]);
    rig.close();
}

#[test]
fn asymmetric_compatibility_never_forms_an_order_dependent_cohort() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(2, RetryOrder::Regroup);
    let mut rows = rig.rows(1, 2, vec![Action::Fatal], policy, &log);
    let control = rows[0].state.control.clone();
    *control.compatibility.lock().unwrap() = Some(|first, row| first < row);

    turn(&mut rows, &rig, policy);

    assert!(rows.is_empty());
    assert_eq!(operations(&log), ["single:1:0", "single:1:1"]);
    assert_eq!(rig.stats.snapshot().failed, 2);
    rig.close();
}

#[test]
fn malformed_discard_cannot_recurse_without_removing_a_member() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let policy = policy(4, RetryOrder::Regroup);
    let mut rows = rig.rows(1, 4, vec![Action::ReadyDiscard], policy, &log);
    turn(&mut rows, &rig, policy);
    assert!(rows.is_empty());
    assert_eq!(
        operations(&log),
        ["prepare:1:4", "discard:1:4", "ready-drop:1:4"]
    );
    assert_eq!(rig.stats.snapshot().failed, 4);
    rig.close();
}
#[test]
fn invalid_wave_controls_refuse_before_worker_or_session_creation() {
    let mut rig = Rig::new();
    let log = Arc::default();
    let original = policy(4, RetryOrder::Regroup);
    let rows = rig.rows(1, 1, vec![], original, &log);
    let model = rows[0].model.clone();
    drop(rows);
    let mut bad = Vec::new();
    let mut p = original;
    p.fairness.round_tokens = 0;
    bad.push(p);
    let mut p = original;
    p.fairness.round_tokens = 71;
    bad.push(p);
    let mut p = original;
    p.fairness.prefill_age_rounds = 0;
    bad.push(p);
    let mut p = original;
    p.prefill_quantum = 0;
    bad.push(p);
    let mut p = original;
    p.prefill_quantum = 513;
    bad.push(p);
    let mut p = original;
    p.cohort_width = 8;
    bad.push(p);
    let mut p = original;
    p.cohorts = Cohorts::Fused {
        widths: &[8, 4, 2, 1, 1],
        split_prefix: true,
        retry: RetryOrder::Regroup,
    };
    bad.push(p);
    let mut p = original;
    p.cohorts = Cohorts::None;
    bad.push(p);
    let mut p = original;
    p.max_scratch = usize::MAX;
    bad.push(p);
    for policy in bad {
        assert!(
            Router::with_executor::<StatusOnly>(
                BTreeMap::from([("control".into(), model.clone())]),
                rig.host.clone(),
                WorkerConfig {
                    workers: 1,
                    active: 8,
                    context: 512,
                    output: 1,
                    max_pending: Some(1),
                    policy: WorkPolicy::Waves(policy)
                }
            )
            .is_err()
        );
    }
    assert!(operations(&log).is_empty());
    drop(model);
    rig.close();
}
