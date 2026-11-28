//! Bounded operational observations; no prompts, model text or credentials.
use super as operational;
use std::collections::VecDeque;
use zerogravity::control::{ControlEvent, ControlEventKind, ReclaimStatus};
use zerogravity::runtime::text::FailureDetails;

pub(crate) const CAPACITY: usize = 128;
pub(crate) use super::Level;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Component {
    #[cfg(any(feature = "tui", feature = "web"))]
    Action,
    Models,
    Requests,
    Serve,
    Cache,
    #[cfg(any(
        feature = "tui",
        all(feature = "web", feature = "acquisition", target_os = "linux")
    ))]
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
            #[cfg(any(feature = "tui", feature = "web"))]
            Self::Action => "action",
            Self::Models => "models",
            Self::Requests => "requests",
            Self::Serve => "serve",
            Self::Cache => "cache",
            #[cfg(any(
                feature = "tui",
                all(feature = "web", feature = "acquisition", target_os = "linux")
            ))]
            Self::Download => "download",
            Self::Control => "control",
        }
    }
}

#[derive(Clone)]
pub(crate) struct Entry {
    pub sequence: u64,
    pub timestamp_unix_ms: u64,
    pub level: Level,
    pub component: Component,
    pub correlation: String,
    pub message: String,
    pub details: Option<FailureDetails>,
    pub stage: Option<String>,
    pub elapsed_ms: Option<u64>,
    pub causes: Vec<String>,
}
impl Entry {
    fn new(
        level: Level,
        component: Component,
        correlation: String,
        message: impl Into<String>,
    ) -> Self {
        Self {
            sequence: 0,
            timestamp_unix_ms: operational::now_ms(),
            level,
            component,
            correlation,
            message: message.into(),
            details: None,
            stage: None,
            elapsed_ms: None,
            causes: Vec::new(),
        }
    }
}

#[cfg(feature = "tui")]
impl Entry {
    pub fn line(&self) -> String {
        format!(
            "{} {:5} {:9} {}{} · {}{}",
            operational::clock(self.timestamp_unix_ms),
            self.level.label(),
            self.component.label(),
            self.stage
                .as_ref()
                .map_or_else(String::new, |stage| format!("{stage} · ")),
            self.message,
            self.correlation,
            self.causes
                .first()
                .map_or_else(String::new, |cause| format!(" · {cause}"))
        )
    }
}

#[derive(Default)]
pub(crate) struct Ring {
    entries: VecDeque<Entry>,
    next: u64,
    #[cfg(all(
        any(feature = "tui", feature = "web"),
        feature = "acquisition",
        target_os = "linux"
    ))]
    download: Option<(
        u64,
        zerogravity::control::DownloadPhase,
        zerogravity::control::DownloadStatus,
        bool,
    )>,
    #[cfg(all(
        any(feature = "tui", feature = "web"),
        feature = "acquisition",
        target_os = "linux"
    ))]
    download_sample: Option<(std::time::Instant, u64)>,
}

impl Ring {
    fn append(&mut self, mut entry: Entry) {
        entry.sequence = self.next;
        self.next = self.next.saturating_add(1);
        if self.entries.len() == CAPACITY {
            self.entries.pop_front();
        }
        operational::record(operational::Record {
            timestamp_unix_ms: entry.timestamp_unix_ms,
            level: entry.level,
            component: entry.component.label(),
            correlation: &entry.correlation,
            message: &entry.message,
            details: entry.details,
            stage: entry.stage.as_deref(),
            elapsed_ms: entry.elapsed_ms,
            causes: &entry.causes,
        });
        self.entries.push_back(entry);
    }

    fn push(
        &mut self,
        level: Level,
        component: Component,
        correlation: String,
        message: impl Into<String>,
    ) {
        self.append(Entry::new(level, component, correlation, message));
    }

    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn start(&mut self, id: u64, title: &str, correlation: String) {
        self.push(
            Level::Info,
            Component::Action,
            format!("action #{id} · {correlation}"),
            format!("{title} started"),
        );
    }

    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn task(
        &mut self,
        task: &crate::control_host::progress::Task,
        terminal: Option<(bool, Option<FailureDetails>, Vec<String>)>,
    ) {
        let mut message = match terminal.as_ref() {
            Some((true, _, _)) => format!("{} completed", task.action.title),
            Some(_) => format!("{} returned error", task.action.title),
            None => task.action.title.to_owned(),
        };
        if let Some(requested) = &task.requested {
            message.push_str(&format!(
                " · requested {}",
                operational::bounded_message(requested)
            ));
        }
        if let Some(resolved) = &task.progress.resolved {
            message.push_str(&format!(
                " · selected {}",
                operational::bounded_message(resolved)
            ));
        }
        #[cfg(all(feature = "cache", target_os = "linux"))]
        if let Some(v) = task.progress.verification {
            message.push_str(&format!(
                " · hashed {}/{} bytes · verified {}/{} files",
                v.bytes_read, v.total_bytes, v.files_verified, v.total_files
            ));
        }
        let (success, details, causes) = terminal.unwrap_or((true, None, Vec::new()));
        let mut entry = Entry::new(
            if success { Level::Info } else { Level::Error },
            Component::Action,
            format!("action #{} · {}", task.action.id, task.correlation()),
            message,
        );
        entry.stage = Some(task.progress.stage.label().into());
        entry.elapsed_ms = Some(task.started.elapsed().as_millis().min(u64::MAX as u128) as u64);
        entry.details = details;
        entry.causes = causes;
        self.append(entry);
    }

    #[cfg(all(test, feature = "tui"))]
    pub fn finish(&mut self, id: u64, title: &str, success: bool, details: Option<FailureDetails>) {
        let mut entry = Entry::new(
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
        entry.details = details;
        self.append(entry);
    }
    #[cfg(any(feature = "tui", feature = "web"))]
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
    #[cfg(feature = "serve")]
    pub fn drain(&mut self, events: &mut zerogravity::control::ControlEvents<'_>) {
        use zerogravity::control::ControlPoll;
        for _ in 0..64 {
            match events.poll(std::time::Duration::ZERO) {
                ControlPoll::Event(event) => self.event(event),
                ControlPoll::Resync { missed } => self.gap(missed),
                ControlPoll::Pending => break,
            }
        }
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
            E::ServeRequestFailed {
                generation,
                details,
            } => {
                let mut entry = Entry::new(
                    if details.status.is_some_and(|status| status >= 500) {
                        Level::Error
                    } else {
                        Level::Warning
                    },
                    Component::Serve,
                    format!("event {} · generation {generation}", event.sequence),
                    details.public_message(),
                );
                entry.details = Some(details);
                entry.timestamp_unix_ms = event.timestamp_unix_ms;
                self.append(entry);
                return;
            }
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
        let mut entry = Entry::new(
            level,
            component,
            format!("event {} · {correlation}", event.sequence),
            message,
        );
        entry.timestamp_unix_ms = event.timestamp_unix_ms;
        self.append(entry);
    }
    #[cfg(all(
        any(feature = "tui", feature = "web"),
        feature = "acquisition",
        target_os = "linux"
    ))]
    pub fn observe_download(&mut self, row: Option<&zerogravity::control::DownloadSummary>) {
        use zerogravity::control::DownloadStatus;
        let current = row.map(|row| (row.id, row.phase, row.status, row.cancel_requested));
        let changed = current != self.download;
        if !changed
            && !row.is_some_and(|row| {
                self.download_sample.is_none_or(|(time, bytes)| {
                    time.elapsed() >= std::time::Duration::from_secs(1) && bytes != row.total_bytes
                })
            })
        {
            return;
        }
        self.download = current;
        if let Some((id, phase, status, cancel)) = current {
            let row = row.expect("observed download");
            self.download_sample = Some((std::time::Instant::now(), row.total_bytes));
            self.push(
                match status {
                    DownloadStatus::Failed => Level::Error,
                    DownloadStatus::DurabilityUnknown => Level::Warning,
                    _ => Level::Info,
                },
                Component::Download,
                format!("operation {id}"),
                format!(
                    "{phase:?} · {status:?} · {} bytes · {}/{} files{}{}",
                    row.total_bytes,
                    row.completed_files,
                    row.selected_files,
                    row.total_expected_bytes
                        .map_or_else(String::new, |n| format!(" · expected {n} bytes")),
                    if cancel {
                        " · cancellation requested"
                    } else {
                        ""
                    },
                ),
            );
        }
    }

    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn revision(&self) -> u64 {
        self.next
    }
    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn entries(&self) -> &VecDeque<Entry> {
        &self.entries
    }
    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn dropped(&self) -> u64 {
        self.next.saturating_sub(self.entries.len() as u64)
    }
}
