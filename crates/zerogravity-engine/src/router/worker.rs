//! One admitted queue/worker and presentation policy for typed executors.
use super::pending::ServingHost as HostPool;
use super::pending::{CancelToken, Pop, ServingCharge, WorkerQueue};
use super::statistics::{Counters, WorkStats};
use super::waves::{Phase, Ready};
use super::*;
use crate::model_instance::execution::*;
use crate::state_cache::PrefixSharingScope;
use std::result::Result;
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

mod cohorts;

#[derive(Clone, Debug, serde::Serialize)]
pub struct SchedulerSnapshot {
    pub config: WorkerConfig,
    pub queue: super::pending::QueueSnapshot,
    pub host: Option<super::pending::HostBudgetSnapshot>,
    pub active_limit: usize,
    pub cohort_width: usize,
    pub counters: Counters,
}
pub(crate) struct ActiveCharge(Arc<WorkStats>);
impl WorkStats {
    pub(crate) fn enter(self: &Arc<Self>) -> ActiveCharge {
        let active = self.active.fetch_add(1, Ordering::AcqRel) + 1;
        self.peak.fetch_max(active, Ordering::AcqRel);
        ActiveCharge(self.clone())
    }
}
impl Drop for ActiveCharge {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}
/// Executor-private physical state is dropped first; its source-reader lease
/// remains until presentation, prompt and transport owners have retired.
pub struct Active<A: Executor> {
    pub state: A::State,
    pub model: A::Model,
    pub cancelled: Arc<AtomicBool>,
    pub errors: event_channel::mpsc::SyncSender<ResponseEvent>,
    pub output: GeneratedText,
    pub prompt: Vec<u32>,
    pub cursor: usize,
    pub prefix_sharing: Option<PrefixSharingScope>,
    pub packet_presented: bool,
    /// Eligible prefill rounds deferred by decode protection.
    pub prefill_wait_rounds: usize,
    pub blocked_since: Option<Instant>,
    pub token_commit_observed_at: Option<Instant>,
    pub token_observation: Option<GpuTokenObservation>,
    context: usize,
    policy: WorkPolicy,
    _active: ActiveCharge,
    _prompt_charge: Option<ServingCharge>,
    _transport: Option<Arc<ServingCharge>>,
    _source_lease: A::Lease,
    _cancel: CancelToken,
    _model_reader: Option<ModelRead>,
}
impl<A: Executor> Active<A> {
    pub fn new(
        job: Scheduled,
        model: A::Model,
        config: WorkerConfig,
        host: &HostPool,
        stats: Arc<WorkStats>,
    ) -> Result<Self, ServiceError> {
        let Scheduled {
            payload,
            metrics,
            events,
            cancelled,
            cancel,
            prompt_charge,
            transport,
            model_reader,
        } = job;
        let lease = payload.ok_or_else(|| {
            ServiceError::new(
                500,
                "scheduled_payload",
                "scheduled request lost its payload",
            )
        })?;
        let request = &lease.request;
        let prefix_sharing = lease.prefix_sharing;
        let tokenizer = model.model().tokenizer();
        let prompt = tokenizer
            .prompt_tokens(&request.input, request.chat)
            .map_err(ServiceError::inference)?;
        if request.max_tokens == 0
            || request.max_tokens > config.output
            || prompt.is_empty()
            || prompt.len() > config.context
            || prompt
                .len()
                .checked_add(request.max_tokens - 1)
                .is_none_or(|n| n > config.context)
        {
            return Err(ServiceError::new(
                400,
                "context_overflow",
                "request exceeds bounded model context",
            ));
        }
        let (mut state, source_lease) = A::open(&model, &request.model, &cancelled, host)?;
        let cursor = A::restore_prefix(&model, &mut state, &prompt, prefix_sharing)?;
        if cursor > prompt.len()
            || A::frontier(&state) != cursor
            || (cursor == prompt.len() && !A::has_feedback(&state))
        {
            return Err(ServiceError::new(
                500,
                "prefix_restore",
                "restored prefix differs from executor frontier/feedback",
            ));
        }
        let output = GeneratedText::new(
            tokenizer,
            events.clone(),
            request.stream,
            request.max_tokens,
            prompt.len(),
        )
        .observed_charged(
            metrics.metrics,
            model.model().vocabulary_size(),
            metrics._charges,
        );
        drop(lease);
        let active = stats.enter();
        Ok(Self {
            state,
            model,
            cancelled,
            errors: events,
            output,
            prompt,
            cursor,
            prefix_sharing,
            packet_presented: false,
            prefill_wait_rounds: 0,
            blocked_since: None,
            token_commit_observed_at: None,
            token_observation: None,
            context: config.context,
            policy: config.policy,
            _active: active,
            _prompt_charge: prompt_charge,
            _transport: transport,
            _source_lease: source_lease,
            _cancel: cancel,
            _model_reader: model_reader,
        })
    }
    pub fn prefill_count(&self) -> usize {
        let remaining = self.prompt.len().saturating_sub(self.cursor);
        match A::prefill(&self.model) {
            Prefill::Chunks(n) => remaining.min(n),
            Prefill::Atomic => remaining,
        }
    }
    /// Whether the current admitted prefill span is followed by another
    /// prefill span. This is scheduler semantics, independent of model/backend.
    pub fn prefill_has_successor(&self) -> bool {
        self.prompt.len().saturating_sub(self.cursor) > self.prefill_count()
    }
    pub fn ready(&self) -> Ready {
        Ready {
            source: A::group(&self.model),
            phase: if self.cursor < self.prompt.len() {
                Phase::Prefill
            } else {
                Phase::Decode
            },
            frontier: A::frontier(&self.state),
            capacity: self.context,
            tokens: if self.cursor < self.prompt.len() {
                self.prefill_count()
            } else {
                1
            },
            // Finite scheduler credit, never a physical workspace quote.
            scratch_bytes: 32 * 1024 * 1024,
        }
    }
    pub fn present(&mut self, stats: &WorkStats) -> Result<Presentation, ServiceError> {
        if matches!(self.policy, WorkPolicy::Sequential) {
            let id = A::selected(&mut self.state)?;
            let committed = self
                .token_commit_observed_at
                .ok_or_else(missing_observation)?;
            let before = self.output.generated_token_count();
            let presented = self
                .output
                .accept_observed(id, committed, self.token_observation);
            stats.output_tokens.fetch_add(
                (self.output.generated_token_count() - before) as u64,
                Ordering::Relaxed,
            );
            return presented.map(|done| {
                if done {
                    Presentation::Finished
                } else {
                    Presentation::Ready
                }
            });
        }
        let shown = if self.packet_presented {
            self.output.resume()?
        } else {
            let id = A::selected(&mut self.state)?;
            let before = self.output.generated_token_count();
            let committed = self
                .token_commit_observed_at
                .ok_or_else(missing_observation)?;
            let shown = self
                .output
                .present_observed(id, committed, self.token_observation)?;
            stats.output_tokens.fetch_add(
                (self.output.generated_token_count() - before) as u64,
                Ordering::Relaxed,
            );
            self.packet_presented = true;
            shown
        };
        if shown == Presentation::Blocked {
            let since = self.blocked_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= Duration::from_secs(30) {
                return Err(ServiceError::new(
                    503,
                    "backpressure_timeout",
                    "client output credit deadline expired",
                ));
            }
        } else {
            self.blocked_since = None;
        }
        Ok(shown)
    }
    /// One existing source quantum, including bounded presentation. Caller
    /// owns fair ordering/attempt accounting; no in-flight future may escape.
    pub fn step(&mut self, stats: &WorkStats) -> Result<Progress, ServiceError> {
        if self.cursor >= self.prompt.len() {
            match self.present(stats)? {
                Presentation::Finished => return Ok(Progress::Finished),
                Presentation::Blocked => return Ok(Progress::Parked),
                Presentation::Ready => {}
            }
        }
        if matches!(self.policy, WorkPolicy::Waves(_)) && self.cancelled.load(Ordering::Acquire) {
            return Ok(Progress::Cancelled);
        }
        let completed = A::single(self)?;
        stats.record(completed.completion);
        self.install(completed.published?, None)?;
        Ok(Progress::Forwarded)
    }
    pub fn install(
        &mut self,
        mut published: Published,
        cohort: Option<FusedCohortObservation>,
    ) -> Result<(), ServiceError> {
        let cache_prefix = if self.cursor < self.prompt.len() {
            let end = self
                .cursor
                .checked_add(published.consumed)
                .filter(|&end| published.consumed != 0 && end <= self.prompt.len())
                .ok_or_else(|| {
                    ServiceError::new(
                        500,
                        "source_progress",
                        "published prompt progress exceeds its original input",
                    )
                })?;
            self.cursor = end;
            !published.readout
        } else if published.consumed != 1 {
            return Err(ServiceError::new(
                500,
                "source_progress",
                "feedback publication must consume one original token",
            ));
        } else {
            false
        };
        if cache_prefix {
            A::cache_prefix(
                &self.model,
                &self.state,
                &self.prompt[..self.cursor],
                self.prefix_sharing,
            )?;
        }
        if published.readout {
            if let Some(observation) = &mut published.observation {
                observation.fused_cohort = cohort;
            }
            self.token_observation = published.observation;
            self.token_commit_observed_at = Some(published.observed_at);
            self.packet_presented = false;
            self.blocked_since = None;
        }
        Ok(())
    }
}
fn missing_observation() -> ServiceError {
    ServiceError::new(
        500,
        "missing_commit_observation",
        "committed packet has no publication observation",
    )
}
pub fn fail<A: Executor>(row: &Active<A>, stats: &WorkStats, error: ServiceError) {
    if error.status == 499 && error.code == "cancelled" {
        stats.cancelled.fetch_add(1, Ordering::Relaxed);
    } else {
        stats.failed.fetch_add(1, Ordering::Relaxed);
    }
    let _ = row.errors.try_send(ResponseEvent::Error(error));
}
fn eligible<A: Executor>(
    mut row: Active<A>,
    parked: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
) -> Option<Active<A>> {
    if row.cancelled.load(Ordering::Acquire) {
        stats.cancelled.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if row.cursor < row.prompt.len() {
        return Some(row);
    }
    match row.present(stats) {
        Ok(Presentation::Ready) => Some(row),
        Ok(Presentation::Blocked) => {
            parked.push_back(row);
            None
        }
        Ok(Presentation::Finished) => {
            stats.attempted.fetch_add(1, Ordering::Relaxed);
            stats.completed.fetch_add(1, Ordering::Relaxed);
            None
        }
        Err(error) => {
            stats.attempted.fetch_add(1, Ordering::Relaxed);
            fail(&row, stats, error);
            None
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    Forwarded,
    Finished,
    Parked,
    Cancelled,
}
pub fn advance_single<A: Executor>(
    row: Active<A>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
) {
    let mut row = if matches!(row.policy, WorkPolicy::Sequential) {
        if row.cancelled.load(Ordering::Acquire) {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
            return;
        }
        row
    } else {
        let Some(row) = eligible(row, active, stats) else {
            return;
        };
        row
    };
    stats.attempted.fetch_add(1, Ordering::Relaxed);
    match row.step(stats) {
        Ok(Progress::Parked) => active.push_back(row),
        Ok(Progress::Cancelled) => {
            stats.cancelled.fetch_add(1, Ordering::Relaxed);
        }
        Ok(progress) => {
            stats.completed.fetch_add(1, Ordering::Relaxed);
            if progress == Progress::Forwarded {
                active.push_back(row);
            }
        }
        Err(error) => fail(&row, stats, error),
    }
}

pub(crate) struct WorkerCaptures<A: Executor> {
    pub queue: WorkerQueue<Scheduled>,
    pub models: Arc<super::registry::Registry<A>>,
    pub host: HostPool,
    pub stats: Arc<WorkStats>,
    pub config: WorkerConfig,
}
impl<A: Executor> WorkerCaptures<A> {
    pub fn run(self) {
        let _worker = super::WorkerScope::enter();
        // The consumed, exactly-priced captures remain inside the worker scope.
        run::<A>(self.queue, self.models, self.host, self.stats, self.config);
    }
}
fn run<A: Executor>(
    queue: WorkerQueue<Scheduled>,
    models: Arc<super::registry::Registry<A>>,
    host: HostPool,
    stats: Arc<WorkStats>,
    config: WorkerConfig,
) {
    let mut active = VecDeque::<Active<A>>::with_capacity(config.active);
    let wave_capacity = if matches!(config.policy, WorkPolicy::Waves(_)) {
        config.active
    } else {
        0
    };
    let mut eligible_rows = VecDeque::with_capacity(wave_capacity);
    let mut parked = VecDeque::with_capacity(wave_capacity);
    let mut closing = false;
    let cancelled = |job: &Scheduled| job.cancel.is_cancelled();
    loop {
        let _ = queue.sweep(cancelled);
        if matches!(config.policy, WorkPolicy::Waves(_)) {
            active.retain(|row| {
                if row.cancelled.load(Ordering::Acquire) {
                    stats.cancelled.fetch_add(1, Ordering::Relaxed);
                    false
                } else {
                    true
                }
            });
        }
        if !closing {
            while active.len() < config.active {
                let next = if active.is_empty() {
                    let wait = if matches!(config.policy, WorkPolicy::Sequential) {
                        30
                    } else {
                        25
                    };
                    queue.pop_wait(Duration::from_millis(wait), cancelled)
                } else {
                    queue.try_pop(cancelled)
                };
                let job = match next {
                    Pop::Item(job) => job,
                    Pop::Empty => break,
                    Pop::Closed => {
                        closing = true;
                        break;
                    }
                };
                if job.cancelled.load(Ordering::Acquire) {
                    stats.cancelled.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let alias = &job
                    .payload
                    .as_ref()
                    .expect("scheduled payload")
                    .request
                    .model;
                let errors = job.events.clone();
                let model = job
                    .model_reader
                    .as_ref()
                    .ok_or_else(|| {
                        ServiceError::new(
                            500,
                            "model_reader",
                            "queued request lacks its registry owner",
                        )
                    })
                    .and_then(|reader| models.model_for_job(alias, reader));
                match model
                    .and_then(|model| Active::<A>::new(job, model, config, &host, stats.clone()))
                {
                    Ok(row) => active.push_back(row),
                    Err(error) => {
                        if error.status == 499 && error.code == "cancelled" {
                            stats.cancelled.fetch_add(1, Ordering::Relaxed);
                        } else {
                            stats.failed.fetch_add(1, Ordering::Relaxed);
                        }
                        let _ = errors.try_send(ResponseEvent::Error(error));
                    }
                }
            }
        }
        if active.is_empty() {
            if closing {
                if let Err(error) = models.retire_all() {
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                    stats.retirement_failed.store(true, Ordering::Release);
                    eprintln!("source retirement failed: {error}");
                }
                return;
            }
            continue;
        }
        match config.policy {
            WorkPolicy::Sequential => {
                let row = active.pop_front().expect("nonempty active queue");
                advance_single(row, &mut active, &stats);
            }
            WorkPolicy::Waves(policy) => {
                let progressed =
                    advance_wave(&mut active, &mut eligible_rows, &mut parked, &stats, policy);
                if !progressed && !active.is_empty() {
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        }
    }
}

/// One complete fair wave, including credit parking, membership and retries.
/// Both the real worker and physical diagnostics use this same primitive.
/// Caller admits the bounded active and scratch containers before entry.
pub fn advance_wave<A: Executor>(
    active: &mut VecDeque<Active<A>>,
    ready: &mut VecDeque<Active<A>>,
    parked: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: WavePolicy,
) -> bool {
    ready.clear();
    parked.clear();
    for row in active.drain(..) {
        if let Some(row) = eligible(row, parked, stats) {
            ready.push_back(row);
        }
    }
    std::mem::swap(active, parked);
    stats.parked.store(active.len(), Ordering::Relaxed);
    if ready.is_empty() {
        return false;
    }
    advance_round(ready, active, stats, policy);
    stats.parked.store(
        active.iter().filter(|r| r.blocked_since.is_some()).count(),
        Ordering::Relaxed,
    );
    true
}

/// Shared work-forming primitive used by the worker and physical diagnostics.
/// Callers admit their bounded row/reference storage before entering it.
pub fn advance_round<A: Executor>(
    eligible_rows: &mut VecDeque<Active<A>>,
    active: &mut VecDeque<Active<A>>,
    stats: &WorkStats,
    policy: WavePolicy,
) {
    let ready = eligible_rows.iter().map(Active::ready).collect::<Vec<_>>();
    let ages = eligible_rows
        .iter()
        .map(|row| row.prefill_wait_rounds)
        .collect::<Vec<_>>();
    let bounds = waves::Limits {
        max_members: policy.max_members,
        max_tokens: policy.max_tokens,
        max_scratch: policy.max_scratch,
    };
    let planned = waves::form_with_frontiers(
        &ready,
        bounds,
        policy.unequal_decode,
        policy.unequal_prefill,
        policy.prefill_quantum,
    )
    .and_then(|waves| waves::fair_round(&ready, &ages, waves, policy.fairness));
    let round = match planned {
        Ok(round) => round,
        Err(error) => {
            for row in eligible_rows.drain(..) {
                fail(&row, stats, error.clone());
            }
            return;
        }
    };
    let mut pending = eligible_rows.drain(..).map(Some).collect::<Vec<_>>();
    // Deferred work keeps its original state/source owners and goes before rows
    // that consume this round, so equal-age prefills cannot be perpetually
    // overtaken by the same younger wave.
    for index in round.deferred {
        let mut row = pending[index].take().expect("deferred scheduler row");
        debug_assert!(row.cursor < row.prompt.len());
        row.prefill_wait_rounds = row.prefill_wait_rounds.saturating_add(1);
        active.push_back(row);
    }
    for wave in round.waves {
        for &index in &wave.indices {
            pending[index]
                .as_mut()
                .expect("selected scheduler row")
                .prefill_wait_rounds = 0;
        }
        let admitted = wave
            .indices
            .iter()
            .filter_map(|&i| pending[i].as_ref())
            .filter(|r| !r.cancelled.load(Ordering::Acquire))
            .collect::<Vec<_>>();
        if let Err(error) = A::validate_wave(&admitted) {
            stats
                .failed
                .fetch_add(admitted.len() as u64, Ordering::Relaxed);
            for index in wave.indices {
                if let Some(row) = pending[index].take() {
                    let _ = row.errors.try_send(ResponseEvent::Error(error.clone()));
                }
            }
            continue;
        }
        stats.groups.fetch_add(1, Ordering::Relaxed);
        if wave.indices.len() > 1 {
            stats
                .co_scheduled
                .fetch_add(wave.indices.len() as u64, Ordering::Relaxed);
        }
        let rows = wave
            .indices
            .iter()
            .map(|&i| pending[i].take().expect("unique scheduler wave row"))
            .collect::<Vec<_>>();
        cohorts::advance(rows, active, stats, policy.cohorts, policy.cohort_width);
    }
    debug_assert!(pending.iter().all(Option::is_none));
}

#[cfg(test)]
mod tests;
