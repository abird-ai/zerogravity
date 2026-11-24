//! Coarse execution seam. Scheduling metadata never grants backend eligibility.
//! Associated state/Ready/lease types retain the original physical owners.
use super::{LogitModel, PreparedModel};
pub use crate::router::waves::Fairness;
use crate::router::waves::Phase;
use crate::router::worker::{Active, SchedulerSnapshot};
use crate::router::{GpuTokenObservation, ServiceError, pending::ServingHost as HostPool};
use crate::state_cache::{PrefixCacheEviction, PrefixSharingScope};
use std::{collections::BTreeMap, sync::Arc, time::Instant};

type Result<T> = std::result::Result<T, ServiceError>;

/// Cold adapter cleanup of one independently drained prepared owner.
pub type ModelRetirement<M> = fn(&M) -> Result<()>;

/// The shared registry and typed workers hold the very same model handles.
/// Identical erased Arc identity must mean the same typed execution owner;
/// clones retain that owner, including its original cleanup authority.
pub trait ModelHandle: Clone + Send + Sync + 'static {
    fn model(&self) -> &dyn PreparedModel;
    fn erased(&self) -> Arc<dyn PreparedModel>;
}
impl<T: PreparedModel + 'static> ModelHandle for Arc<T> {
    fn model(&self) -> &dyn PreparedModel {
        self.as_ref()
    }
    fn erased(&self) -> Arc<dyn PreparedModel> {
        self.clone()
    }
}
impl ModelHandle for Arc<dyn LogitModel> {
    fn model(&self) -> &dyn PreparedModel {
        self.as_ref()
    }
    fn erased(&self) -> Arc<dyn PreparedModel> {
        self.clone()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prefill {
    Chunks(usize),
    Atomic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryOrder {
    Regroup,
    NextRound,
}
#[derive(Clone, Copy, Debug)]
pub enum Cohorts {
    None,
    /// Independent final-prompt calls still publish through one all-row barrier.
    PromptTails,
    Fused {
        widths: &'static [usize],
        split_prefix: bool,
        retry: RetryOrder,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct WavePolicy {
    pub max_members: usize,
    pub max_tokens: usize,
    pub max_scratch: usize,
    /// Total logical source/decode tokens issued in one scheduler round.
    pub fairness: Fairness,
    pub prefill_quantum: usize,
    pub unequal_prefill: bool,
    pub unequal_decode: bool,
    pub cohorts: Cohorts,
    pub cohort_width: usize,
}
#[derive(Clone, Copy, Debug)]
pub enum WorkPolicy {
    /// One cooperative quantum before checking pending work again.
    Sequential,
    /// Every eligible row gets one fair round; output-blocked rows are parked.
    Waves(WavePolicy),
}
#[derive(Clone, Copy, Debug)]
pub struct WorkerConfig {
    pub workers: usize,
    pub active: usize,
    pub context: usize,
    pub output: usize,
    pub max_pending: Option<usize>,
    pub policy: WorkPolicy,
}

#[derive(Clone, Copy, Debug)]
pub struct CohortWork {
    pub phase: Phase,
    pub state_only: bool,
    /// Zero: composite all-row publication, not a fused cohort width.
    pub cohort_width: usize,
}
/// Actual completed work, distinct from publication and output usage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Completion {
    pub quanta: u64,
    pub forwards: u64,
    pub dispatches: u64,
    pub model_tokens: u64,
    pub committed_roots: u64,
    pub fused_projections: u64,
    pub fused_sampler: u64,
    pub cohort_width: usize,
    pub state_only: bool,
}
/// Only returned after the original source/session publisher succeeds.
#[derive(Clone, Copy, Debug)]
pub struct Published {
    pub consumed: usize,
    pub readout: bool,
    pub observed_at: Instant,
    pub observation: Option<GpuTokenObservation>,
}
pub struct SingleCompleted {
    pub completion: Completion,
    pub published: Result<Published>,
}
pub struct Completed<R> {
    pub ready: R,
    pub completion: Completion,
    pub fused: bool,
}
pub enum Preparation<R> {
    Ready(Completed<R>),
    /// Proven no launch/state mutation. The scheduler may reduce the band.
    Capacity,
    CancelledBeforeLaunch,
}
pub enum Publication {
    Committed(Vec<Result<Published>>),
    /// Known completion; private results discarded, original roots unchanged.
    Discarded,
}

/// No threads, queue, fairness, presentation or fallback policy in an executor.
/// Submitted calls must reach original completion/quarantine before returning;
/// never race/drop an in-flight future on cancellation. Presentation IDs cannot
/// be used to reconstruct native feedback: retain it inside `State`.
pub trait Executor: Sized + Send + Sync + 'static {
    type Model: ModelHandle;
    type State;
    type Lease;
    type Ready;

    fn open(
        model: &Self::Model,
        alias: &str,
        cancelled: &Arc<std::sync::atomic::AtomicBool>,
        host: &HostPool,
    ) -> Result<(Self::State, Self::Lease)>;
    fn prefill(model: &Self::Model) -> Prefill;
    fn group(model: &Self::Model) -> usize;
    fn frontier(state: &Self::State) -> usize;
    fn selected(state: &mut Self::State) -> Result<u32>;
    fn has_feedback(state: &Self::State) -> bool;
    fn single(row: &mut Active<Self>) -> Result<SingleCompleted>;
    /// Restore an explicitly authorized complete-domain prefix into a freshly
    /// opened state. Zero means miss/no sharing. The worker verifies the returned
    /// cursor against the executor's actual frontier.
    fn restore_prefix(
        _model: &Self::Model,
        _state: &mut Self::State,
        _prompt: &[u32],
        _sharing: Option<PrefixSharingScope>,
    ) -> Result<usize> {
        Ok(0)
    }
    /// Retain an already-published state-only prompt prefix for future authorized
    /// requests. This runs only after the worker has advanced its committed cursor.
    fn cache_prefix(
        _model: &Self::Model,
        _state: &Self::State,
        _prefix: &[u32],
        _sharing: Option<PrefixSharingScope>,
    ) -> Result<()> {
        Ok(())
    }
    fn invalidate_prefix_scope(
        _models: &BTreeMap<String, Self::Model>,
        _sharing: PrefixSharingScope,
    ) -> Result<PrefixCacheEviction> {
        Ok(PrefixCacheEviction::default())
    }

    fn validate_wave(_rows: &[&Active<Self>]) -> Result<()> {
        Ok(())
    }
    fn validate_cohort(_rows: &[Active<Self>], _work: CohortWork) -> Result<()> {
        Ok(())
    }
    fn cohort_eligible(_row: &Active<Self>) -> bool {
        false
    }
    /// Pairwise physical compatibility for fused cohort formation. The
    /// scheduler requires this relation in both directions for every pair, so
    /// implementations need not be trusted to be symmetric or transitive and
    /// must not rely on one distinguished anchor row.
    fn compatible(_first: &Active<Self>, _row: &Active<Self>) -> bool {
        true
    }
    fn prepare(_rows: &mut [Active<Self>], _work: CohortWork) -> Result<Preparation<Self::Ready>> {
        Err(ServiceError::new(
            400,
            "unsupported_cohort",
            "no cohort executor",
        ))
    }
    fn commit(
        _ready: Self::Ready,
        _rows: &mut [Active<Self>],
        _work: CohortWork,
    ) -> Result<Publication> {
        Err(ServiceError::new(
            400,
            "unsupported_cohort",
            "no cohort publisher",
        ))
    }
    /// Independent model cleanup on the lifecycle caller, after all accepted
    /// requests of that exact owner have retired. Other models may be running.
    /// Return a hook only when the adapter's original ownership allows this.
    fn selective_retirement() -> Option<ModelRetirement<Self::Model>> {
        None
    }
    fn retirement_pending(_models: &BTreeMap<String, Self::Model>) -> bool {
        false
    }
    fn retire(_models: &BTreeMap<String, Self::Model>) -> Result<()> {
        Ok(())
    }
    fn diagnostics(
        models: &BTreeMap<String, Self::Model>,
        snapshot: &SchedulerSnapshot,
    ) -> Result<serde_json::Value>;
    /// Additional bounded adapter-side row/reference containers, not device
    /// workspace or state (those remain admitted by their physical owner).
    fn scratch_bytes(_active: usize) -> Result<u64> {
        Ok(0)
    }
}
