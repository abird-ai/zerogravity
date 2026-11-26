//! Bounded operational observations; no prompts, model text or credentials.
use std::collections::VecDeque;
use zerogravity::control::{ControlEvent, ControlEventKind, ReclaimStatus};

pub(crate) const CAPACITY: usize = 128;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Level {
    Info,
    Warning,
    Error,
}
impl Level {
    pub fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Component {
    Action,
    Models,
    Requests,
    Serve,
    Cache,
    #[cfg(any(feature = "tui", all(feature = "acquisition", target_os = "linux")))]
    Download,
    Control,
}
impl Component {
    #[cfg(feature = "tui")]
    pub const ALL: [Self; 7] = [
        Self::Action,
        Self::Models,
        Self::Requests,
        Self::Serve,
        Self::Cache,
        Self::Download,
        Self::Control,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Action => "action",
            Self::Models => "models",
            Self::Requests => "requests",
            Self::Serve => "serve",
            Self::Cache => "cache",
            #[cfg(any(feature = "tui", all(feature = "acquisition", target_os = "linux")))]
            Self::Download => "download",
            Self::Control => "control",
        }
    }
}

#[derive(Clone)]
pub(crate) struct Entry {
    pub sequence: u64,
    pub level: Level,
    pub component: Component,
    pub correlation: String,
    pub message: String,
}
#[cfg(feature = "tui")]
impl Entry {
    pub fn line(&self) -> String {
        format!(
            "{:>5} {:5} {:9} {} · {}",
            self.sequence,
            self.level.label(),
            self.component.label(),
            self.message,
            self.correlation
        )
    }
}

#[derive(Default)]
pub(crate) struct Ring {
    entries: VecDeque<Entry>,
    next: u64,
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    download: Option<(
        u64,
        zerogravity::control::DownloadPhase,
        zerogravity::control::DownloadStatus,
        bool,
    )>,
}

impl Ring {
    fn push(
        &mut self,
        level: Level,
        component: Component,
        correlation: String,
        message: impl Into<String>,
    ) {
        let sequence = self.next;
        self.next += 1;
        if self.entries.len() == CAPACITY {
            self.entries.pop_front();
        }
        self.entries.push_back(Entry {
            sequence,
            level,
            component,
            correlation,
            message: message.into(),
        });
    }
    pub fn start(&mut self, id: u64, title: &str, correlation: String) {
        self.push(
            Level::Info,
            Component::Action,
            format!("action #{id} · {correlation}"),
            format!("{title} started"),
        );
    }
    pub fn finish(&mut self, id: u64, title: &str, success: bool) {
        self.push(
            if success { Level::Info } else { Level::Error },
            Component::Action,
            format!("action #{id}"),
            format!(
                "{title} {}",
                if success {
                    "completed"
                } else {
                    "returned error"
                }
            ),
        );
    }
    pub fn cancel(&mut self, id: u64) {
        self.push(
            Level::Info,
            Component::Action,
            format!("action #{id}"),
            "Cancellation requested",
        );
    }
    pub fn gap(&mut self, missed: u64) {
        self.push(
            Level::Warning,
            Component::Control,
            "events".into(),
            format!("Missed {missed} transient notifications; resnapshot requested"),
        );
    }
    pub fn event(&mut self, event: ControlEvent) {
        use ControlEventKind as E;
        let outcome = |ok| if ok { Level::Info } else { Level::Error };
        let result = |ok| if ok { "succeeded" } else { "failed" };
        let (level, component, correlation, message) = match event.kind {
            E::ModelLoadFinished { success } => (
                outcome(success),
                Component::Models,
                "route set".into(),
                format!("Load {}", result(success)),
            ),
            E::ModelUnloadFinished {
                registration_id,
                success,
            } => (
                outcome(success),
                Component::Models,
                format!("registration {registration_id}"),
                format!("Unload {}", result(success)),
            ),
            E::RequestAccepted { registration_id } => (
                Level::Info,
                Component::Requests,
                format!("registration {registration_id}"),
                "Request accepted (not completed)".into(),
            ),
            E::CacheAliasFinished { success } => (
                outcome(success),
                Component::Cache,
                "alias".into(),
                format!("Alias {}", result(success)),
            ),
            E::CacheVerificationFinished { success } => (
                outcome(success),
                Component::Cache,
                "verification".into(),
                format!("Verification {}", result(success)),
            ),
            E::CacheReclamationFinished { outcome } => (
                match outcome {
                    ReclaimStatus::Failed => Level::Error,
                    ReclaimStatus::Busy | ReclaimStatus::Referenced | ReclaimStatus::Incomplete => {
                        Level::Warning
                    }
                    _ => Level::Info,
                },
                Component::Cache,
                "reclaim".into(),
                format!("Reclamation {outcome:?}"),
            ),
            E::ServeStartFinished { success } => (
                outcome(success),
                Component::Serve,
                "listener".into(),
                format!("Start {}", result(success)),
            ),
            E::ServeStopFinished {
                generation,
                success,
            } => (
                outcome(success),
                Component::Serve,
                format!("generation {generation}"),
                format!("Stop {}", result(success)),
            ),
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            E::DownloadChanged { .. } => return, // progress is not log history
        };
        self.push(
            level,
            component,
            format!("event {} · {correlation}", event.sequence),
            message,
        );
    }
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    pub fn observe_download(&mut self, row: Option<&zerogravity::control::DownloadSummary>) {
        use zerogravity::control::DownloadStatus;
        let current = row.map(|row| (row.id, row.phase, row.status, row.cancel_requested));
        if current == self.download {
            return;
        }
        self.download = current;
        if let Some((id, phase, status, cancel)) = current {
            self.push(
                match status {
                    DownloadStatus::Failed => Level::Error,
                    DownloadStatus::DurabilityUnknown => Level::Warning,
                    _ => Level::Info,
                },
                Component::Download,
                format!("operation {id}"),
                format!(
                    "{phase:?} · {status:?}{}",
                    if cancel {
                        " · cancellation requested"
                    } else {
                        ""
                    }
                ),
            );
        }
    }

    pub fn revision(&self) -> u64 {
        self.next
    }
    pub fn entries(&self) -> &VecDeque<Entry> {
        &self.entries
    }
    pub fn dropped(&self) -> u64 {
        self.next.saturating_sub(self.entries.len() as u64)
    }
}
