//! Observations of the existing foreground worker, not another task owner.
#[cfg(feature = "tui")]
use super::work::Completed;
use super::{actions::ActionSummary, work::Request};
use std::{sync::Mutex, time::Instant};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Subject {
    Snapshot(String),
    Catalog(String),
    Registration(u64),
    // A pull intent can resolve a different snapshot than any cached row.
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Repository {
        provider: String,
        repository: String,
        revision: String,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Stage {
    #[default]
    Starting,
    Resolving,
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Verifying,
    Loading,
    Generating,
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Downloading,
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Inspecting,
    Retiring,
}
impl Stage {
    #[cfg(feature = "tui")]
    pub fn label(self) -> &'static str {
        match self {
            Self::Starting => "Starting",
            Self::Resolving => "Resolving source and execution policy",
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Self::Verifying => "Verifying payloads",
            Self::Loading => "Preparing and loading model",
            Self::Generating => "Generating",
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Self::Downloading => "Downloading",
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Self::Inspecting => "Inspecting metadata",
            Self::Retiring => "Retiring",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Progress {
    pub stage: Stage,
    pub resolved: Option<String>,
    #[cfg(all(feature = "cache", target_os = "linux"))]
    pub verification: Option<zerogravity::cache::VerificationProgress>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Task {
    pub action: ActionSummary,
    #[cfg_attr(
        not(feature = "tui"),
        allow(dead_code, reason = "native model detail association")
    )]
    pub subject: Option<Subject>,
    #[cfg_attr(
        not(feature = "tui"),
        allow(dead_code, reason = "native requested route display")
    )]
    pub requested: Option<String>,
    pub started: Instant,
    pub progress: Progress,
}
#[cfg(feature = "tui")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FinishedTask {
    pub task: Task,
    pub elapsed: std::time::Duration,
    pub failed: bool,
    // Native display only. Browser output must use its existing safe projection.
    pub message: String,
}

#[cfg(feature = "tui")]
impl FinishedTask {
    pub fn new(task: Task, result: &zerogravity::Result<Completed>) -> Self {
        let (failed, message) = match result {
            Ok(Completed::Message(message)) => (false, message.clone()),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Ok(Completed::Refused {
                message,
                native_detail,
            }) => (
                true,
                native_detail
                    .as_ref()
                    .map_or_else(|| message.clone(), |detail| format!("{message}: {detail}")),
            ),
            Ok(Completed::Settings(outcome)) => (false, outcome.message.clone()),
            Err(error) => (true, error.to_string()),
        };
        // Bound the retained observation, not the underlying typed result.
        let mut end = message.len().min(2048);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        Self {
            elapsed: task.started.elapsed(),
            task,
            failed,
            message: message[..end].to_owned(),
        }
    }
}

impl Task {
    pub fn new(action: ActionSummary, request: &Request, profile: &str) -> Self {
        let (subject, requested) = match request {
            Request::Load {
                source, execution, ..
            } => (source_subject(source), Some(execution.summary(profile))),
            Request::Verify(digest)
            | Request::Inspect(digest)
            | Request::Reclaim { digest, .. }
            | Request::Alias { digest, .. } => (Some(Subject::Snapshot(digest.clone())), None),
            Request::Invoke { owner, .. } | Request::Unload(owner) => {
                (Some(Subject::Registration(owner.registration)), None)
            }
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Request::Pull(intent) => (Some(intent.subject()), None),
            _ => (None, None),
        };
        Self {
            action,
            subject,
            requested,
            started: Instant::now(),
            progress: Progress::default(),
        }
    }
}

fn source_subject(source: &str) -> Option<Subject> {
    use zerogravity::run_policy::SourceSelector;
    let source = source
        .split_once('=')
        .filter(|(alias, _)| zerogravity::runtime::text::validate_alias(alias).is_ok())
        .map_or(source, |(_, source)| source);
    match SourceSelector::parse(source).ok()? {
        SourceSelector::Snapshot(digest) => Some(Subject::Snapshot(digest)),
        SourceSelector::Name(name) | SourceSelector::Catalog(name) => Some(Subject::Catalog(name)),
        _ => None, // no inferred source identity for a local path or floating ref
    }
}

pub(crate) fn update(progress: &Mutex<Progress>, change: impl FnOnce(&mut Progress)) {
    change(
        &mut progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
    );
}

pub(crate) fn stage(progress: &Mutex<Progress>, stage: Stage) {
    update(progress, |progress| progress.stage = stage);
}
