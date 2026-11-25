//! Bounded operational observations; no prompts, model text or credentials.
use std::collections::VecDeque;
use zerogravity::control::{ControlEvent, ControlEventKind, ReclaimStatus};

const CAPACITY: usize = 128;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Level {
    Info,
    Warning,
    Error,
}
impl Level {
    fn label(self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    Action,
    Models,
    Requests,
    Serve,
    Cache,
    Download,
    Control,
}
impl Component {
    const ALL: [Self; 7] = [
        Self::Action,
        Self::Models,
        Self::Requests,
        Self::Serve,
        Self::Cache,
        Self::Download,
        Self::Control,
    ];
    fn label(self) -> &'static str {
        match self {
            Self::Action => "action",
            Self::Models => "models",
            Self::Requests => "requests",
            Self::Serve => "serve",
            Self::Cache => "cache",
            Self::Download => "download",
            Self::Control => "control",
        }
    }
}

struct Entry {
    sequence: u64,
    level: Level,
    component: Component,
    correlation: String,
    message: String,
}
impl Entry {
    fn line(&self) -> String {
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
pub(super) struct Logs {
    entries: VecDeque<Entry>,
    next: u64,
    level: Option<Level>,
    component: Option<Component>,
    pub query: String,
    paused_at: Option<u64>,
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    download: Option<(
        u64,
        zerogravity::control::DownloadPhase,
        zerogravity::control::DownloadStatus,
        bool,
    )>,
}

impl Logs {
    fn push(
        &mut self,
        level: Level,
        component: Component,
        correlation: String,
        message: impl Into<String>,
    ) -> u64 {
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
        sequence
    }
    pub fn start(&mut self, title: &str, correlation: String) -> u64 {
        self.push(
            Level::Info,
            Component::Action,
            format!("action #{} · {correlation}", self.next),
            format!("{title} started"),
        )
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
    pub fn following(&self) -> bool {
        self.paused_at.is_none()
    }
    pub fn toggle_follow(&mut self) {
        self.paused_at = if self.following() {
            Some(self.next)
        } else {
            None
        };
    }
    pub fn cycle_level(&mut self) {
        self.level = match self.level {
            None => Some(Level::Info),
            Some(Level::Info) => Some(Level::Warning),
            Some(Level::Warning) => Some(Level::Error),
            Some(Level::Error) => None,
        };
    }
    pub fn cycle_component(&mut self) {
        self.component = match self.component {
            None => Some(Component::ALL[0]),
            Some(value) => Component::ALL
                .iter()
                .position(|component| *component == value)
                .and_then(|i| Component::ALL.get(i + 1))
                .copied(),
        };
    }
    pub fn heading(&self) -> String {
        format!(
            "{} · level {} · component {} · search {}",
            if self.following() {
                "Following"
            } else {
                "Paused view (collection continues)"
            },
            self.level.map_or("all", Level::label),
            self.component.map_or("all", Component::label),
            if self.query.is_empty() {
                "all"
            } else {
                &self.query
            }
        )
    }
    pub fn lines(&self) -> Vec<String> {
        let query = self.query.to_ascii_lowercase();
        self.entries
            .iter()
            .filter(|entry| {
                self.paused_at.is_none_or(|end| entry.sequence < end)
                    && self.level.is_none_or(|level| entry.level == level)
                    && self
                        .component
                        .is_none_or(|component| entry.component == component)
            })
            .map(Entry::line)
            .filter(|line| query.is_empty() || line.to_ascii_lowercase().contains(&query))
            .collect()
    }
    pub fn dropped(&self) -> u64 {
        self.next.saturating_sub(self.entries.len() as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn logs_follow_filter_and_retain_only_bounded_operational_rows() {
        let mut logs = Logs::default();
        logs.event(ControlEvent {
            sequence: 2,
            kind: ControlEventKind::RequestAccepted {
                registration_id: 17,
            },
        });
        logs.toggle_follow();
        logs.finish(1, "Run", false);
        assert_eq!(logs.lines().len(), 1);
        logs.toggle_follow();
        logs.query = "registration 17".into();
        assert_eq!(logs.lines().len(), 1);
        logs.query.clear();
        for _ in 0..CAPACITY {
            logs.gap(1);
        }
        assert_eq!(logs.lines().len(), CAPACITY);
        assert_eq!(logs.dropped(), 2);
        logs.cycle_level();
        assert!(logs.lines().is_empty());
    }
}
