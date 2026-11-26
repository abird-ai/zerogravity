//! One control-owned operation view, not another transfer/cache executor.
//! The caller runs the original synchronous transaction on its chosen worker.
use super::{ControlEventKind, ControlPlane};
use crate::{
    Error, Result,
    acquisition::{
        self, PullControl, PullObservation, PullPhase, PullProgress, PullReceipt, ResolvedPullPlan,
        http::HttpTransport,
    },
    artifacts::PreparedHubPull,
    cache::{CacheStore, CommitOutcome, TransferAdmission},
};
use serde::Serialize;
use std::{
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadPhase {
    Resolving,
    Planning,
    Transferring,
    Verifying,
    Publishing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStatus {
    Running,
    PauseRequested,
    Paused,
    Durable,
    DurabilityUnknown,
    Cancelled,
    Failed,
}
impl DownloadStatus {
    pub fn terminal(self) -> bool {
        matches!(
            self,
            Self::Durable | Self::DurabilityUnknown | Self::Cancelled | Self::Failed
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DownloadSummary {
    pub id: u64,
    pub phase: DownloadPhase,
    pub status: DownloadStatus,
    pub cancel_requested: bool,
    pub source: Option<crate::cache::ManifestSource>,
    pub selected_files: usize,
    pub completed_files: usize,
    pub path: Option<String>,
    pub file_complete: bool,
    pub file_bytes: u64,
    pub file_expected_bytes: Option<u64>,
    pub file_maximum_bytes: u64,
    pub total_bytes: u64,
    pub total_expected_bytes: Option<u64>,
    pub total_maximum_bytes: u64,
    pub cached_bytes: u64,
    pub resumed_bytes: u64,
    /// Last observed proper-prefix checkpoint of the current file, NOT a
    /// claim that these bytes remain on disk or will be eligible next time.
    pub checkpoint_bytes: u64,
    /// Rate of newly accepted logical bytes, never network throughput.
    pub logical_bytes_per_second: Option<u64>,
    pub snapshot: Option<String>,
}

impl DownloadSummary {
    fn new(id: u64, phase: DownloadPhase) -> Self {
        Self {
            id,
            phase,
            status: DownloadStatus::Running,
            cancel_requested: false,
            source: None,
            selected_files: 0,
            completed_files: 0,
            path: None,
            file_complete: false,
            file_bytes: 0,
            file_expected_bytes: None,
            file_maximum_bytes: 0,
            total_bytes: 0,
            total_expected_bytes: None,
            total_maximum_bytes: 0,
            cached_bytes: 0,
            resumed_bytes: 0,
            checkpoint_bytes: 0,
            logical_bytes_per_second: None,
            snapshot: None,
        }
    }
}

struct Active {
    summary: DownloadSummary,
    cancel: Arc<AtomicBool>,
    sample: Option<(Instant, u64)>,
    last_event: Instant,
}
#[derive(Default)]
struct Slot {
    next: u64,
    current: Option<Active>,
}

#[derive(Default)]
pub(super) struct Downloads {
    slot: Mutex<Slot>,
    changed: Condvar,
}
impl Downloads {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.slot
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    pub(super) fn observe(&self) -> Option<DownloadSummary> {
        self.lock().current.as_ref().map(|active| {
            let mut summary = active.summary.clone();
            summary.cancel_requested = active.cancel.load(Ordering::Acquire);
            summary
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadCommand {
    Pause,
    Resume,
    Cancel,
}

impl ControlPlane<'_> {
    pub fn download(&self) -> Option<DownloadSummary> {
        self.state.downloads.observe()
    }

    /// Reserve the single operation view for this runtime. No worker, cache,
    /// transport or credentials are created; caller owns the actual execution.
    /// The same cancel flag may be used by the existing process signal owner.
    pub fn begin_download(&self, cancel: Arc<AtomicBool>) -> Result<DownloadOperation<'_>> {
        self.begin_download_in(cancel, DownloadPhase::Resolving)
    }

    /// Already-resolved plans start at Planning, not an invented resolve phase.
    pub fn begin_resolved_download(
        &self,
        cancel: Arc<AtomicBool>,
    ) -> Result<DownloadOperation<'_>> {
        self.begin_download_in(cancel, DownloadPhase::Planning)
    }

    fn begin_download_in(
        &self,
        cancel: Arc<AtomicBool>,
        phase: DownloadPhase,
    ) -> Result<DownloadOperation<'_>> {
        self.runtime.engine.require_open()?;
        let mut slot = self.state.downloads.lock();
        if slot
            .current
            .as_ref()
            .is_some_and(|row| !row.summary.status.terminal())
        {
            return Err(Error::State(
                "a control-owned download is already active".into(),
            ));
        }
        slot.next = slot
            .next
            .checked_add(1)
            .ok_or_else(|| Error::Limit("download operation identity exhausted".into()))?;
        let id = slot.next;
        slot.current = Some(Active {
            summary: DownloadSummary::new(id, phase),
            cancel: cancel.clone(),
            sample: None,
            last_event: Instant::now(),
        });
        self.emit(ControlEventKind::DownloadChanged {
            id,
            status: DownloadStatus::Running,
        });
        Ok(DownloadOperation {
            control: *self,
            id,
            cancel,
            finished: false,
            expected_snapshot: None,
        })
    }

    /// Stale selections never affect replacement operations. Pause is a
    /// request, not an acknowledgement; only the executor can park safely.
    pub fn control_download(&self, id: u64, command: DownloadCommand) -> Result<()> {
        let downloads = &self.state.downloads;
        let mut slot = downloads.lock();
        let active = slot
            .current
            .as_mut()
            .filter(|row| row.summary.id == id && !row.summary.status.terminal())
            .ok_or_else(|| Error::State("download operation is no longer active".into()))?;
        let state = &mut active.summary;
        match command {
            DownloadCommand::Cancel => active.cancel.store(true, Ordering::Release),
            DownloadCommand::Pause
                if !active.cancel.load(Ordering::Acquire)
                    && state.phase != DownloadPhase::Publishing
                    && state.status == DownloadStatus::Running =>
            {
                state.status = DownloadStatus::PauseRequested;
            }
            DownloadCommand::Resume
                if !active.cancel.load(Ordering::Acquire)
                    && matches!(
                        state.status,
                        DownloadStatus::PauseRequested | DownloadStatus::Paused
                    ) =>
            {
                state.status = DownloadStatus::Running;
                active.sample = None;
                state.logical_bytes_per_second = None;
            }
            _ => {
                return Err(Error::State(
                    "download control is unavailable in this state".into(),
                ));
            }
        }
        self.emit(ControlEventKind::DownloadChanged {
            id,
            status: state.status,
        });
        downloads.changed.notify_all();
        Ok(())
    }
}

/// Single synchronous owner guard. Clients hold views/cursors, not this
/// guard. Drop without a terminal receipt records failure, never completion.
pub struct DownloadOperation<'a> {
    control: ControlPlane<'a>,
    id: u64,
    cancel: Arc<AtomicBool>,
    finished: bool,
    expected_snapshot: Option<String>,
}
impl DownloadOperation<'_> {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn cancel_flag(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }

    /// Refuse any other manifest before publication. This is an additional
    /// caller restriction, never a source or model execution grant.
    pub fn expect_snapshot(mut self, digest: &str) -> Result<Self> {
        crate::cache::paths::validate_digest(digest)?;
        self.expected_snapshot = Some(digest.into());
        Ok(self)
    }

    fn update(&self, update: impl FnOnce(&mut Active) -> bool) {
        let mut slot = self.control.state.downloads.lock();
        if let Some(active) = slot
            .current
            .as_mut()
            .filter(|row| row.summary.id == self.id && !row.summary.status.terminal())
            && update(active)
        {
            self.control.emit(ControlEventKind::DownloadChanged {
                id: self.id,
                status: active.summary.status,
            });
        }
    }

    fn progress(&self, event: &PullProgress<'_>) {
        self.update(|active| {
            let row = &mut active.summary;
            let new_file = row.path.as_deref() != Some(event.path());
            if new_file {
                row.path = Some(event.path().into());
                row.checkpoint_bytes = 0;
                row.file_complete = false;
            }
            if event.file_complete() && !row.file_complete {
                row.completed_files += 1;
                row.file_complete = true;
            }
            row.file_bytes = event.file_bytes();
            row.file_expected_bytes = event.file_total_bytes();
            row.file_maximum_bytes = event.file_maximum_bytes();
            row.total_bytes = event.total_bytes();
            row.total_expected_bytes = event.total_expected_bytes();
            row.total_maximum_bytes = event.total_maximum_bytes();
            row.cached_bytes = event.cached_bytes();
            row.resumed_bytes = event.resumed_bytes();
            if event.file_complete() {
                row.checkpoint_bytes = 0;
            }
            let now = Instant::now();
            let accepted = event.downloaded_bytes();
            if let Some((at, previous)) = active.sample {
                let elapsed = now.duration_since(at);
                if elapsed >= Duration::from_millis(200) {
                    row.logical_bytes_per_second = Some(
                        (accepted.saturating_sub(previous) as f64 / elapsed.as_secs_f64()) as u64,
                    );
                    active.sample = Some((now, accepted));
                }
            } else {
                active.sample = Some((now, accepted));
            }
            let changed = new_file
                || event.file_complete()
                || now.duration_since(active.last_event) >= Duration::from_millis(200);
            if changed {
                active.last_event = now;
            }
            changed
        });
    }

    fn planned(&self, plan: &ResolvedPullPlan) {
        self.update(|active| {
            active.summary.source = Some(plan.source().clone());
            active.summary.selected_files = plan.files().len();
            active.summary.total_expected_bytes = plan.total_bytes();
            active.summary.total_maximum_bytes = plan.maximum_bytes();
            true
        });
    }

    pub fn execute_prepared(
        mut self,
        prepared: &PreparedHubPull<'_>,
        store: &CacheStore,
        transport: &impl HttpTransport,
        admission: &TransferAdmission,
        progress: &mut impl FnMut(&PullProgress<'_>),
    ) -> Result<PullReceipt> {
        self.planned(prepared.plan());
        let result = prepared.execute_controlled(
            store,
            transport,
            admission,
            &self.cancel,
            PullObservation {
                control: &self,
                progress: &mut |event: &PullProgress<'_>| {
                    self.progress(event);
                    progress(event);
                },
            },
        );
        self.finish(&result);
        result
    }

    /// Explicit already-resolved provider plan, same original executor.
    pub fn execute_plan(
        mut self,
        store: &CacheStore,
        plan: &ResolvedPullPlan,
        transport: &impl HttpTransport,
        admission: &TransferAdmission,
        reference: Option<&str>,
        progress: &mut impl FnMut(&PullProgress<'_>),
    ) -> Result<PullReceipt> {
        self.planned(plan);
        let result = acquisition::execute_controlled(
            store,
            plan,
            transport,
            admission,
            &self.cancel,
            reference,
            PullObservation {
                control: &self,
                progress: &mut |event: &PullProgress<'_>| {
                    self.progress(event);
                    progress(event);
                },
            },
        );
        self.finish(&result);
        result
    }

    /// Resolution failed before entering the executor. Preserve the actual
    /// original error and do not convert a cancel request into proof of stop.
    pub fn fail(mut self, error: &Error) {
        self.terminal(DownloadStatus::from_error(error), None);
    }

    fn finish(&mut self, result: &Result<PullReceipt>) {
        match result {
            Ok(receipt) => match receipt.outcome() {
                CommitOutcome::Durable(snapshot) => {
                    self.terminal(DownloadStatus::Durable, Some(snapshot.digest()))
                }
                CommitOutcome::DurabilityUnknown { snapshot, .. } => {
                    self.terminal(DownloadStatus::DurabilityUnknown, Some(snapshot.digest()))
                }
            },
            Err(error) => self.terminal(DownloadStatus::from_error(error), None),
        }
    }
    fn terminal(&mut self, status: DownloadStatus, snapshot: Option<&str>) {
        self.update(|active| {
            active.summary.status = status;
            active.summary.snapshot = snapshot.map(str::to_owned);
            active.summary.logical_bytes_per_second = None;
            true
        });
        self.finished = true;
        self.control.state.downloads.changed.notify_all();
    }
}
impl Drop for DownloadOperation<'_> {
    fn drop(&mut self) {
        if !self.finished {
            self.terminal(DownloadStatus::Failed, None);
        }
    }
}

impl DownloadStatus {
    /// Recognize only the original explicit cooperative cancellation outcomes.
    /// A simultaneous request and unrelated error remains a failure.
    pub fn from_error(error: &Error) -> Self {
        match error {
            Error::State(message)
                if matches!(
                    message.as_str(),
                    "acquisition canceled"
                        | "cache operation canceled"
                        | "cache descriptor operation canceled"
                ) =>
            {
                Self::Cancelled
            }
            _ => Self::Failed,
        }
    }
}

impl PullControl for DownloadOperation<'_> {
    fn expected_snapshot(&self) -> Option<&str> {
        self.expected_snapshot.as_deref()
    }
    fn phase(&self, phase: PullPhase) {
        self.update(|active| {
            active.summary.phase = match phase {
                PullPhase::Planning => DownloadPhase::Planning,
                PullPhase::Transferring => DownloadPhase::Transferring,
                PullPhase::Verifying => DownloadPhase::Verifying,
                PullPhase::Publishing => DownloadPhase::Publishing,
            };
            // A pause racing the publication boundary cannot stop publication.
            if phase == PullPhase::Publishing
                && active.summary.status == DownloadStatus::PauseRequested
            {
                active.summary.status = DownloadStatus::Running;
            }
            true
        });
    }
    fn checkpoint(&self, bytes: u64) {
        self.update(|active| {
            active.summary.checkpoint_bytes = bytes;
            true
        });
    }
    fn pause_requested(&self) -> bool {
        self.control
            .state
            .downloads
            .lock()
            .current
            .as_ref()
            .is_some_and(|row| {
                row.summary.id == self.id && row.summary.status == DownloadStatus::PauseRequested
            })
    }
    fn wait_paused(&self, cancel: &AtomicBool) -> Result<()> {
        let downloads = &self.control.state.downloads;
        let mut slot = downloads.lock();
        if let Some(active) = slot
            .current
            .as_mut()
            .filter(|row| row.summary.id == self.id)
            && active.summary.status == DownloadStatus::PauseRequested
            && !cancel.load(Ordering::Acquire)
        {
            active.summary.status = DownloadStatus::Paused;
            active.summary.logical_bytes_per_second = None;
            active.sample = None;
            self.control.emit(ControlEventKind::DownloadChanged {
                id: self.id,
                status: active.summary.status,
            });
        }
        // Signals use the original atomic flag, so bounded polling also
        // notices external cancellation without signal-handler locking.
        while !cancel.load(Ordering::Acquire)
            && slot.current.as_ref().is_some_and(|row| {
                row.summary.id == self.id && row.summary.status == DownloadStatus::Paused
            })
        {
            (slot, _) = downloads
                .changed
                .wait_timeout(slot, Duration::from_millis(100))
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        if cancel.load(Ordering::Acquire) {
            Err(Error::State("acquisition canceled".into()))
        } else {
            Ok(())
        }
    }
}
