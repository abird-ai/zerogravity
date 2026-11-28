//! One foreground action slot shared by both clients. No effect queue.
//! The worker is joined before replacement or host retirement.
#[cfg(feature = "tui")]
use super::progress::FinishedTask;
use super::{
    Selection,
    logs::Ring,
    progress::{Observer, Progress, Task},
    work::{self, ActionResult, Completed, Request, Response},
};
use std::{
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
};
#[cfg(feature = "web")]
use zerogravity::runtime::text::FailureCategory;
use zerogravity::runtime::text::ServiceError;
use zerogravity::{Error, Result, Runtime};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Origin {
    #[cfg(feature = "tui")]
    Native,
    #[cfg(feature = "web")]
    Web,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ActionSummary {
    pub id: u64,
    pub origin: Origin,
    pub title: &'static str,
    pub cancel_requested: bool,
}
struct Active {
    id: u64,
    origin: Origin,
    title: &'static str,
    generates: bool,
    #[cfg(feature = "web")]
    settings: bool,
    cancel: Arc<AtomicBool>,
    response: Arc<Mutex<Response>>,
    task: Task,
    progress: Arc<Mutex<Progress>>,
    worker: JoinHandle<ActionResult<Completed>>,
}
impl Active {
    fn summary(&self) -> ActionSummary {
        ActionSummary {
            id: self.id,
            origin: self.origin,
            title: self.title,
            cancel_requested: self.cancel.load(Ordering::Acquire),
        }
    }
    fn task(&self) -> Task {
        let mut task = self.task.clone();
        task.action = self.summary();
        task.progress = self
            .progress
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        task
    }
    fn sample(&self, revision: Option<u64>) -> Option<Response> {
        if !self.generates {
            return None;
        }
        let response = self
            .response
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (revision != Some(response.revision)).then(|| response.clone())
    }
    fn join(self) -> ActionResult<Completed> {
        self.worker
            .join()
            .map_err(|_| Error::State("control action worker panicked".into()))?
    }
}

#[cfg(feature = "web")]
#[derive(Clone)]
pub(crate) struct WebResult {
    pub summary: ActionSummary,
    pub state: &'static str,
    pub message: Option<String>,
    pub response: Option<Response>,
}

#[derive(Default)]
struct Slot {
    next: u64,
    closed: bool,
    revision: u64,
    active: Option<Active>,
    logs: Arc<Mutex<Ring>>,
    #[cfg(feature = "tui")]
    last_task: Option<FinishedTask>,
    #[cfg(feature = "web")]
    latest: Option<WebResult>,
}
impl Slot {
    fn logs(&self) -> MutexGuard<'_, Ring> {
        self.logs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn changed(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }
    fn record_terminal(&self, task: &Task, result: &ActionResult<Completed>) {
        let (success, details, causes) = match result {
            Err(error) => {
                let details = error.details();
                let load_failure = matches!(task.progress.stage, super::progress::Stage::Loading)
                    || (task.progress.stage == super::progress::Stage::Resolving
                        && matches!(
                            details.category,
                            zerogravity::runtime::text::FailureCategory::Resource
                                | zerogravity::runtime::text::FailureCategory::Unsupported
                        ));
                let sensitive = (task.subject.is_none() && !load_failure)
                    || task.progress.stage == super::progress::Stage::Generating;
                #[cfg(all(feature = "acquisition", target_os = "linux"))]
                let sensitive =
                    sensitive || task.progress.stage == super::progress::Stage::Downloading;
                let causes = if sensitive {
                    vec![details.public_message().into()]
                } else {
                    crate::output::operational::causes(error)
                };
                (false, Some(details), causes)
            }
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Ok(Completed::Refused {
                message,
                native_detail,
            }) => {
                let mut causes = vec![crate::output::operational::bounded_message(message)];
                if let Some(detail) = native_detail {
                    causes.push(crate::output::operational::bounded_message(detail));
                }
                (false, None, causes)
            }
            _ => (true, None, Vec::new()),
        };
        self.logs().task(task, Some((success, details, causes)));
    }
    #[cfg(feature = "web")]
    fn reap_web(&mut self) {
        if !self
            .active
            .as_ref()
            .is_some_and(|job| job.origin == Origin::Web && job.worker.is_finished())
        {
            return;
        }
        let job = self.active.take().expect("finished Web worker");
        let summary = job.summary();
        let task = job.task();
        let response = job.sample(None);
        let settings = job.settings;
        let result = job.join();
        self.record_terminal(&task, &result);
        #[cfg(feature = "tui")]
        if task.subject.is_some() {
            self.last_task = Some(FinishedTask::new(task, &result));
        }
        let (state, message) = match result {
            Ok(Completed::Message(message)) => ("completed", message),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            Ok(Completed::Refused {
                message,
                native_detail,
            }) => {
                let _ = native_detail;
                ("failed", message)
            }
            Ok(Completed::Settings(outcome)) => {
                let _ = outcome.editor;
                ("completed", outcome.message)
            }
            Err(error) => ("failed", if settings {
                match error.details().category {
                    FailureCategory::State => "Configuration changed, already exists, or publication is uncertain; refresh before retrying.",
                    FailureCategory::Unsupported => "Configuration action refused; default and active profiles are protected.",
                    FailureCategory::Artifact => "Configuration validation refused the draft.",
                    _ => safe_error(&error),
                }
            } else { safe_error(&error) }.into()),
        };
        self.latest = Some(WebResult {
            summary,
            state,
            message: Some(message),
            response,
        });
        self.changed();
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        if let Some(job) = self.active.take() {
            job.cancel.store(true, Ordering::Release);
            let _ = job.join();
        }
    }
}

/// Trusted host clones share this slot, not the execution worker. Only the
/// foreground host calls shutdown; browser disconnection cannot cancel it.
#[derive(Clone, Default)]
pub(crate) struct Actions(Arc<Mutex<Slot>>);
impl Actions {
    fn lock(&self) -> MutexGuard<'_, Slot> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub fn with_logs<T>(&self, f: impl FnOnce(&Ring) -> T) -> T {
        f(&self.lock().logs())
    }
    pub fn with_logs_mut<T>(&self, f: impl FnOnce(&mut Ring) -> T) -> T {
        f(&mut self.lock().logs())
    }

    pub fn start(
        &self,
        runtime: &Runtime,
        selection: &Selection,
        request: Request,
        origin: Origin,
    ) -> Result<u64> {
        let mut slot = self.lock();
        #[cfg(feature = "web")]
        slot.reap_web();
        if slot.closed || slot.active.is_some() {
            return Err(Error::State(
                "An action is already in progress or the host is closing".into(),
            ));
        }
        let id = slot
            .next
            .checked_add(1)
            .ok_or_else(|| Error::Limit("control action identities exhausted".into()))?;
        let title = request.title();

        let generates = request.generates();
        #[cfg(feature = "web")]
        let settings = matches!(&request, Request::Settings(_));
        let runtime = runtime.clone();
        let selection = selection.clone();
        let document = selection.snapshot();
        let cancel = Arc::new(AtomicBool::new(false));
        let response = Arc::new(Mutex::new(Response::default()));
        let task = Task::new(
            ActionSummary {
                id,
                origin,
                title,
                cancel_requested: false,
            },
            &request,
            &document.execution_default,
        );
        let progress = Arc::new(Mutex::new(Progress::default()));
        let observation =
            Observer::new(task.clone(), Arc::clone(&progress), Arc::clone(&slot.logs));
        slot.next = id;
        slot.logs().start(id, title, task.correlation());
        let flag = Arc::clone(&cancel);
        let output = Arc::clone(&response);
        let worker = thread::Builder::new()
            .name("zg-control-action".into())
            .spawn(move || {
                work::perform(
                    &runtime,
                    &selection,
                    &document,
                    request,
                    &flag,
                    origin,
                    work::Observation {
                        response: &output,
                        progress: &observation,
                    },
                )
            });
        let worker = match worker {
            Ok(worker) => worker,
            Err(error) => {
                let error = ServiceError::inference(Error::Io(error));
                slot.record_terminal(&task, &Err(error.clone()));
                slot.changed();
                return Err(error.into_core());
            }
        };

        #[cfg(feature = "web")]
        if origin == Origin::Web {
            slot.latest = None;
        }
        slot.active = Some(Active {
            id,
            origin,
            title,
            generates,
            #[cfg(feature = "web")]
            settings,
            cancel,
            response,
            task,
            progress,
            worker,
        });
        slot.changed();
        Ok(id)
    }

    #[cfg(feature = "tui")]
    pub fn active(&self) -> Option<ActionSummary> {
        let slot = &mut *self.lock();
        #[cfg(feature = "web")]
        slot.reap_web();
        slot.active.as_ref().map(Active::summary)
    }

    /// Progress is sampled separately from the stable action identity.
    #[cfg(feature = "tui")]
    pub fn tasks(&self) -> (Option<Task>, Option<FinishedTask>) {
        let slot = &mut *self.lock();
        #[cfg(feature = "web")]
        slot.reap_web();
        (
            slot.active.as_ref().map(Active::task),
            slot.last_task.clone(),
        )
    }

    /// Exact cancellation, including an observed Web action from the native
    /// host. Browser callers pass their own origin and cannot cancel native
    /// private work through a guessed or stale identity.
    pub fn cancel(&self, id: u64, origin: Option<Origin>) -> Result<()> {
        let mut slot = self.lock();
        #[cfg(feature = "web")]
        slot.reap_web();
        let job = slot
            .active
            .as_ref()
            .filter(|job| job.id == id && origin.is_none_or(|origin| origin == job.origin))
            .ok_or_else(|| Error::State("Selected action is no longer active".into()))?;
        if !job.cancel.swap(true, Ordering::AcqRel) {
            slot.logs().cancel(id);
            slot.changed();
        }
        Ok(())
    }

    #[cfg(feature = "tui")]
    pub fn poll_native(
        &self,
        id: u64,
        revision: u64,
    ) -> (Option<Response>, Option<ActionResult<Completed>>) {
        let mut slot = self.lock();
        let Some(job) = slot
            .active
            .as_ref()
            .filter(|job| job.id == id && job.origin == Origin::Native)
        else {
            return (
                None,
                Some(Err(
                    Error::State("Native action ownership lost".into()).into()
                )),
            );
        };
        let response = job.sample(Some(revision));
        if !job.worker.is_finished() {
            return (response, None);
        }
        let job = slot.active.take().expect("finished native worker");
        // Completion may have arrived between the first sample and finished
        // check; take the exact final bounded output before joining.
        let response = job.sample(Some(revision));
        let task = job.task();
        let result = job.join();
        slot.record_terminal(&task, &result);
        if task.subject.is_some() {
            slot.last_task = Some(FinishedTask::new(task, &result));
        }
        slot.changed();
        (response, Some(result))
    }

    #[cfg(feature = "web")]
    pub fn observe(&self) -> (u64, Option<Task>, Option<u64>) {
        let mut slot = self.lock();
        slot.reap_web();
        let active = slot.active.as_ref().map(Active::task);
        let web = slot
            .active
            .as_ref()
            .filter(|job| job.origin == Origin::Web)
            .map(|job| job.id)
            .or_else(|| slot.latest.as_ref().map(|job| job.summary.id));
        (slot.revision, active, web)
    }

    #[cfg(feature = "web")]
    pub fn web_result(&self, id: u64) -> Option<WebResult> {
        let mut slot = self.lock();
        slot.reap_web();
        if let Some(job) = slot
            .active
            .as_ref()
            .filter(|job| job.id == id && job.origin == Origin::Web)
        {
            return Some(WebResult {
                summary: job.summary(),
                state: "running",
                message: None,
                response: job.sample(None),
            });
        }
        slot.latest
            .as_ref()
            .filter(|job| job.summary.id == id)
            .cloned()
    }

    pub fn shutdown(&self) -> Result<()> {
        let active = {
            let mut slot = self.lock();
            slot.closed = true;
            slot.active.take()
        };
        if let Some(job) = active {
            job.cancel.store(true, Ordering::Release);
            let progress = Arc::clone(&job.progress);
            let mut task = job.task();
            let result = job.join();
            task.progress = progress
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .clone();
            self.lock().record_terminal(&task, &result);
            result.map(|_| ()).map_err(ServiceError::into_core)
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "web")]
fn safe_error(error: &ServiceError) -> &'static str {
    // The original native error is useful locally but may contain paths,
    // provider URLs or arbitrary artifact text. Never forward it verbatim.
    match error.details().category {
        FailureCategory::Io => {
            "Source or local I/O unavailable; inspect the selected cache/configuration in the native client."
        }
        FailureCategory::Artifact => {
            "The original source or artifact validator refused this action."
        }
        FailureCategory::Input => {
            "The configuration or request is invalid; check the selected settings and input."
        }
        FailureCategory::Unsupported => {
            "The selected source or execution route is unavailable; check the cached model and compiled backend."
        }
        FailureCategory::Resource | FailureCategory::Limit => {
            "The original resource or execution bound refused this action."
        }
        FailureCategory::State => {
            "The action could not complete in the current runtime state; refresh and inspect the owner."
        }
    }
}

#[cfg(all(test, feature = "tui", feature = "web"))]
mod tests {
    use super::*;
    use crate::control_host::Document;

    #[test]
    fn one_shared_slot_exact_cancel_and_private_native_result() {
        let runtime = super::super::runtime().unwrap();
        let selection = Selection::new(Document::test_default());
        let actions = Actions::default();
        let (release, wait) = std::sync::mpsc::channel();
        // Hold the same production slot with a deterministic worker
        // boundary. No model or transport mock in the execution path.
        let cancel = Arc::new(AtomicBool::new(false));
        {
            let mut slot = actions.lock();
            slot.next = 1;
            slot.active = Some(Active {
                id: 1,
                origin: Origin::Native,
                title: "Run",
                generates: true,
                settings: false,
                cancel: cancel.clone(),
                response: Arc::new(Mutex::new(Response::default())),
                task: Task::new(
                    ActionSummary {
                        id: 1,
                        origin: Origin::Native,
                        title: "Run",
                        cancel_requested: false,
                    },
                    &Request::Stop(1),
                    "Auto",
                ),
                progress: Arc::new(Mutex::new(Progress::default())),
                worker: thread::spawn(move || {
                    wait.recv().unwrap();
                    Ok(Completed::Message("native private result".into()))
                }),
            });
        }
        assert!(
            actions
                .start(&runtime, &selection, Request::Stop(1), Origin::Web)
                .is_err()
        );
        assert!(actions.web_result(1).is_none());
        assert!(actions.cancel(1, Some(Origin::Web)).is_err());
        assert!(actions.cancel(2, None).is_err());
        assert!(!cancel.load(Ordering::Acquire));
        actions.cancel(1, None).unwrap();
        assert!(cancel.load(Ordering::Acquire));
        release.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if let (_, Some(result)) = actions.poll_native(1, 0) {
                assert!(
                    matches!(result.unwrap(), Completed::Message(text) if text == "native private result")
                );
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            thread::yield_now();
        }
        let id = actions
            .start(
                &runtime,
                &selection,
                Request::Unload(work::Owner {
                    alias: "missing".into(),
                    registration: 1,
                }),
                Origin::Web,
            )
            .unwrap();
        assert_eq!(id, 2);
        assert!(actions.cancel(1, None).is_err());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if let Some(result) = actions.web_result(id)
                && result.state != "running"
            {
                assert_eq!(result.state, "failed");
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            thread::yield_now();
        }
        assert!(actions.active().is_none());
        assert!(actions.web_result(1).is_none());
        actions.shutdown().unwrap();
        runtime.shutdown().unwrap();
    }
}
