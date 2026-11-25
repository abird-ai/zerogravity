//! On-demand observations of the original owners; no execution permission.
use crate::{
    model_instance::execution::Prefill, router::worker::SchedulerSnapshot,
    scheduler::SchedulerLimits,
};
use serde::Serialize;
use zerogravity_artifacts::artifact::ArtifactFormat;
use zerogravity_core::{
    identity::SourceIdentity,
    provider::{
        ExecutionDescription, ResidentModelMemory, ResidentStatePolicy, ResidentTextAdmission,
    },
    resource_arbiter::{HostBackingSnapshot, ResourceSnapshot},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimePhase {
    Empty,
    Loading,
    Ready,
    Retiring,
    Joining,
    Closed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RuntimeStatus {
    /// Global admission gate, including an in-progress shutdown.
    pub closed: bool,
    pub phase: RuntimePhase,
    pub failed: bool,
    /// Legacy count of currently registered aliases.
    pub loaded_models: usize,
    /// Distinct prepared execution owners, independent of source-ID equality.
    pub loaded_owners: usize,
    pub execution_adapters: usize,
}

/// Stable alias slots during one cold route set. Retired/failed entries are
/// not loaded or eligible for requests; they disappear when that set closes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLifecycle {
    Ready,
    Draining,
    Retiring,
    Retired,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SourceRevision {
    pub repository: String,
    pub revision: String,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ArtifactDescription {
    pub family: String,
    pub format: ArtifactFormat,
    /// Authenticated remote revision, only when known by the source owner.
    pub revision: Option<SourceRevision>,
}

/// Existing admission bounds / lease observations. They are not additive
/// reservations on top of the physical resource ledger.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelMemory {
    Host {
        session_state_bytes: u64,
        session_scratch_bytes: u64,
    },
    Resident(ResidentModelMemory),
    ResidentText {
        admission: ResidentTextAdmission,
        resident_bytes: usize,
    },
}

/// Adapter-supplied facts, not guessed from the hardware, a route name or
/// the requested configuration. Unreported facts stay absent.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ModelDetails {
    pub artifact: Option<ArtifactDescription>,
    pub execution: Option<ExecutionDescription>,
    pub state_policy: Option<ResidentStatePolicy>,
    /// Whether the loaded adapter enables scoped prefix sharing.
    pub prefix_sharing: Option<bool>,
    /// The actual retained local source envelope, where available. Does not
    /// include separate tokenizer, backend or request credits.
    pub source_host_reserved_bytes: Option<u64>,
    pub memory: Option<ModelMemory>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelCapabilities {
    pub max_context: usize,
    pub vocabulary_size: u64,
    pub max_token_bytes: usize,
    pub prefill: Prefill,
    /// Endpoint policy only; exact input/shape validation still applies.
    pub chat_allowed: bool,
    pub details: ModelDetails,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ModelSnapshot {
    pub source: SourceIdentity,
    pub aliases: Vec<String>,
    pub lifecycle: ModelLifecycle,
    /// Readers spanning admission, queued work and active session destruction.
    pub admitted_requests: usize,
    pub independent_retirement: bool,
    /// Observe prepared facts only while the owner is available for admission.
    /// Draining/retiring entries never retain an extra executable reader.
    pub capabilities: Option<ModelCapabilities>,
}

/// One cold snapshot of the runtime and original process-wide resource
/// ledger. Lifecycle/catalog are serialized with admission; counters and
/// physical memory may advance between their individual observations.
#[derive(Clone, Debug, Serialize)]
pub struct RuntimeSnapshot {
    pub status: RuntimeStatus,
    pub failure: Option<String>,
    pub limits: SchedulerLimits,
    pub scheduler: Option<SchedulerSnapshot>,
    pub catalog: Vec<ModelSnapshot>,
    /// Includes other runtimes and retained/fenced owners, not OS telemetry.
    pub resources: Vec<ResourceSnapshot>,
    /// Aggregate constraint over those host scopes, not an additional pool.
    pub host_backing: HostBackingSnapshot,
}
