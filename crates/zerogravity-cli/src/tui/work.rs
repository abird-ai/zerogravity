//! One scoped, client-owned action worker. All effects go through the original
//! resident loader and ControlPlane; closing a view does not own retirement.
use super::{library::Key, settings};
use crate::resident;
use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
use zerogravity::{
    Error, Result, Runtime,
    control::RegisteredModel,
    diagnostics::RuntimePhase,
    models::ModelLifecycle,
    run::{
        OutputPoll, ResponseEvent, TextControls,
        config::{RunOverrides, resolve_serving},
    },
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Owner {
    pub alias: String,
    pub registration: u64,
}
impl Owner {
    fn reference(&self) -> RegisteredModel<'_> {
        RegisteredModel {
            alias: &self.alias,
            registration_id: self.registration,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Prompt {
    pub text: String,
    pub chat: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Request {
    Load {
        source: String,
        prompt: Option<Prompt>,
        device: Option<String>,
    },
    Invoke {
        owner: Owner,
        prompt: Prompt,
    },
    Unload(Owner),
    Start(String),
    Stop(u64),
    Settings(settings::Action),
    Inspect(String),
    Verify(String),
    Reclaim {
        digest: String,
        apply: bool,
    },
    Alias {
        digest: String,
        name: String,
    },
    CheckFit {
        source: String,
        device: String,
    },
    #[cfg(all(feature = "acquisition", target_os = "linux"))]
    Pull(crate::pull::native::Intent),
}
impl Request {
    pub fn title(&self) -> &'static str {
        match self {
            Self::Load {
                prompt: Some(_), ..
            } => "Load and run",
            Self::Load { .. } => "Load",
            Self::Invoke { .. } => "Run",
            Self::Unload(_) => "Unload",
            Self::Start(_) => "Start listener",
            Self::Stop(_) => "Stop listener",
            Self::Settings(action) => action.title(),
            Self::Inspect(_) => "Inspect snapshot",
            Self::Verify(_) => "Verify snapshot",
            Self::Reclaim { apply: false, .. } => "Removal preview",
            Self::Reclaim { apply: true, .. } => "Remove snapshot",
            Self::Alias { .. } => "Create alias",
            Self::CheckFit { .. } => "Check device policy",
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            Self::Pull(_) => "Pull model",
        }
    }
    pub fn correlation(&self) -> String {
        match self {
            Self::Invoke { owner, .. } | Self::Unload(owner) => {
                format!("registration {}", owner.registration)
            }
            Self::Stop(generation) => format!("generation {generation}"),
            Self::Inspect(digest)
            | Self::Verify(digest)
            | Self::Reclaim { digest, .. }
            | Self::Alias { digest, .. } => format!("sha256:{digest}"),
            Self::Settings(_) => "configuration".into(),
            _ => "local".into(),
        }
    }
    pub fn detail_key(&self) -> Option<Key> {
        match self {
            Self::Inspect(digest)
            | Self::Verify(digest)
            | Self::Reclaim { digest, .. }
            | Self::Alias { digest, .. } => Some(Key::Snapshot(digest.clone())),
            Self::CheckFit { device, .. } => Some(Key::Device(device.clone())),
            _ => None,
        }
    }
    pub fn generates(&self) -> bool {
        matches!(
            self,
            Self::Invoke { .. }
                | Self::Load {
                    prompt: Some(_),
                    ..
                }
        )
    }
}

pub(super) use crate::control_host::Document;

/// Bounded presentation copy only, independent of the original output stream.
const MAX_DISPLAY_BYTES: usize = 64 * 1024;
#[derive(Clone, Debug, Default)]
pub(super) struct Response {
    pub owner: Option<Owner>,
    pub text: String,
    pub clipped: bool,
    pub revision: u64,
}
impl Response {
    fn append(&mut self, text: &str) {
        let limit = text
            .len()
            .min(MAX_DISPLAY_BYTES.saturating_sub(self.text.len()));
        let mut end = limit;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        self.text.push_str(&text[..end]);
        self.clipped |= end != text.len();
        self.revision = self.revision.wrapping_add(1);
    }
    fn done(&mut self, text: &str) {
        self.text.clear();
        self.clipped = false;
        self.append(text);
    }
}

pub(super) enum Completed {
    Message(String),
    Settings(Box<settings::Outcome>),
}

pub(super) struct Job {
    title: &'static str,
    worker: Option<JoinHandle<Result<Completed>>>,
    cancel: Arc<AtomicBool>,
    response: Arc<Mutex<Response>>,
    generates: bool,
    pub detail_key: Option<Key>,
    pub log_id: u64,
    pub retry: Option<settings::Action>,
}
impl Job {
    pub fn start(
        runtime: &Runtime,
        document: &Document,
        request: Request,
        log_id: u64,
    ) -> Result<Self> {
        let title = request.title();
        let generates = request.generates();
        let detail_key = request.detail_key();
        let retry = match &request {
            Request::Settings(
                action @ (settings::Action::Set { .. } | settings::Action::Replace { .. }),
            ) => Some(action.clone()),
            _ => None,
        };
        let runtime = runtime.clone();
        let document = document.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let response = Arc::new(Mutex::new(Response::default()));
        let flag = Arc::clone(&cancel);
        let output = Arc::clone(&response);
        let worker = thread::Builder::new()
            .name("zg-control-action".into())
            .spawn(move || match request {
                Request::Settings(action) if !flag.load(Ordering::Acquire) => {
                    settings::perform(action, &document.target)
                        .map(|outcome| Completed::Settings(Box::new(outcome)))
                }
                Request::Settings(_) => Ok(Completed::Message("Cancelled before starting".into())),
                request => {
                    execute(&runtime, &document, request, &flag, &output).map(Completed::Message)
                }
            })?;
        Ok(Self {
            title,
            worker: Some(worker),
            cancel,
            response,
            generates,
            detail_key,
            log_id,
            retry,
        })
    }
    pub fn title(&self) -> &'static str {
        self.title
    }
    pub fn requested(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    pub fn finished(&self) -> bool {
        self.worker
            .as_ref()
            .is_some_and(|worker| worker.is_finished())
    }
    pub fn sample(&self, revision: u64) -> Option<Response> {
        if !self.generates {
            return None;
        }
        let response = self
            .response
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (response.revision != revision).then(|| response.clone())
    }
    pub fn join(&mut self) -> Result<Completed> {
        self.worker
            .take()
            .expect("one owned action worker")
            .join()
            .map_err(|_| Error::State("control action worker panicked".into()))?
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        if self.worker.is_some() {
            self.cancel();
            let _ = self.join();
        }
    }
}

fn resolve_request(
    document: &Document,
    source: &str,
    device: Option<&str>,
) -> Result<(
    zerogravity::run_policy::Request,
    zerogravity::pipelines::ServeDoc,
)> {
    let models = if source.trim().is_empty() {
        Vec::new()
    } else {
        // One exact selector, or ID=MODEL. URLs containing '=' remain
        // opaque when passed as the source after a valid alias.
        let source = source.trim();
        let explicit = source
            .split_once('=')
            .filter(|(id, _)| zerogravity::runtime::text::validate_alias(id).is_ok());
        let route = if let Some((id, source)) = explicit {
            resident::route(Some(id), source)
        } else {
            resident::route(None, source)
        };
        vec![route.map_err(|e| e.into_core())?]
    };
    resolve_serving(
        document.read()?,
        RunOverrides {
            devices: device.map(|id| vec![id.to_owned()]),
            ..Default::default()
        },
        models,
        &std::env::current_dir()?,
    )
}

fn execute(
    runtime: &Runtime,
    document: &Document,
    request: Request,
    cancel: &Arc<AtomicBool>,
    output: &Mutex<Response>,
) -> Result<String> {
    if cancel.load(Ordering::Acquire) {
        return Ok("Cancelled before starting".into());
    }
    let control = runtime.control();
    match request {
        Request::Settings(_) => unreachable!("settings worker dispatch"),
        Request::Inspect(digest) => cache_action(control, CacheAction::Inspect, &digest, cancel),
        Request::Verify(digest) => cache_action(control, CacheAction::Verify, &digest, cancel),
        Request::Reclaim { digest, apply } => {
            cache_action(control, CacheAction::Reclaim(apply), &digest, cancel)
        }
        Request::Alias { digest, name } => {
            cache_action(control, CacheAction::Alias(name), &digest, cancel)
        }
        Request::CheckFit { source, device } => {
            let (request, settings) = resolve_request(document, &source, Some(&device))?;
            let plan = resident::prepare(
                request,
                settings,
                None,
                resident::Acquisition {
                    offer: false,
                    pull: false,
                    cancel: Arc::clone(cancel),
                },
            )
            .map_err(|error| error.into_core())?;
            if plan.is_some() {
                Ok(format!(
                    "Cold source/device policy accepted for {device}.\nMetadata and budget observation only; actual load must revalidate and admit."
                ))
            } else {
                Ok("Cancelled before device policy check completed".into())
            }
        }
        #[cfg(all(feature = "acquisition", target_os = "linux"))]
        Request::Pull(intent) => intent.execute(runtime, cancel, document.offline),
        Request::Load {
            source,
            prompt,
            device,
        } => {
            if runtime.status().phase != RuntimePhase::Empty {
                return Err(Error::State(
                    "Retire the current route set before loading another".into(),
                ));
            }
            let (request, settings) = resolve_request(document, &source, device.as_deref())?;
            if prompt.is_some() && request.graph.models.len() != 1 {
                return Err(Error::Malformed("Quick run requires one source; load the configured route set and select an owner".into()));
            }
            let alias = request.graph.models.first().map(|model| model.id.clone());
            let Some(plan) = resident::prepare(
                request,
                settings,
                None,
                resident::Acquisition {
                    offer: false,
                    pull: false,
                    cancel: Arc::clone(cancel),
                },
            )
            .map_err(|e| e.into_core())?
            else {
                return Ok("Cancelled before model load".into());
            };
            if cancel.load(Ordering::Acquire) {
                return Ok("Cancelled before model load".into());
            }
            plan.load(control).map_err(|e| e.into_core())?;
            if let Some(prompt) = prompt {
                if cancel.load(Ordering::Acquire) {
                    return Ok("Load completed; run skipped after cancellation request".into());
                }
                let alias = alias.expect("validated single route");
                let snapshot = control.snapshot()?;
                let model = snapshot
                    .runtime
                    .catalog
                    .iter()
                    .find(|model| {
                        model.lifecycle == ModelLifecycle::Ready && model.aliases.contains(&alias)
                    })
                    .ok_or_else(|| Error::State("Loaded registration is no longer ready".into()))?;
                invoke(
                    runtime,
                    Owner {
                        alias,
                        registration: model.registration_id,
                    },
                    prompt,
                    cancel,
                    output,
                )
            } else {
                Ok(if cancel.load(Ordering::Acquire) {
                    "Model route set loaded; synchronous load could not be interrupted"
                } else {
                    "Model route set loaded"
                }
                .into())
            }
        }
        Request::Invoke { owner, prompt } => invoke(runtime, owner, prompt, cancel, output),
        Request::Unload(owner) => {
            control.unload(owner.reference())?;
            Ok(format!("Registration {} retired", owner.registration))
        }
        Request::Start(address) => {
            #[cfg(feature = "serve")]
            {
                if runtime.status().loaded_owners == 0 {
                    return Err(Error::State(
                        "Load a model before starting the inference API".into(),
                    ));
                }
                let settings = document
                    .read()?
                    .and_then(|(_, doc)| doc.serve)
                    .unwrap_or_default();
                let address = if address.trim().is_empty() {
                    settings.listen.as_deref().unwrap_or("127.0.0.1:8000")
                } else {
                    address.trim()
                };
                let listener = crate::serve::Listener::new(
                    address,
                    settings.enable_per_request_metrics.unwrap_or(false),
                )
                .map_err(|e| e.into_core())?;
                if cancel.load(Ordering::Acquire) {
                    return Ok("Cancelled before listener start".into());
                }
                let binding = control.start_serving(listener.options())?;
                Ok(format!("Listening: http://{}/v1", binding.address))
            }
            #[cfg(not(feature = "serve"))]
            {
                let _ = address;
                Err(Error::Unsupported(
                    "Inference API requires the optional serve feature".into(),
                ))
            }
        }
        Request::Stop(generation) => {
            #[cfg(feature = "serve")]
            {
                control.stop_serving(generation)?;
                Ok("Listener stopped; loaded models retained".into())
            }
            #[cfg(not(feature = "serve"))]
            {
                let _ = generation;
                Err(Error::Unsupported(
                    "Inference API requires the optional serve feature".into(),
                ))
            }
        }
    }
}

enum CacheAction {
    Inspect,
    Verify,
    Reclaim(bool),
    Alias(String),
}
fn cache_action(
    control: zerogravity::control::ControlPlane<'_>,
    action: CacheAction,
    digest: &str,
    cancel: &AtomicBool,
) -> Result<String> {
    #[cfg(all(feature = "cache", target_os = "linux"))]
    {
        use zerogravity::{
            cache::{ReclaimMode, ReclaimOutcome},
            run_policy::SourceSelector,
        };
        let root = super::library::cache_root()?;
        let selector = SourceSelector::Snapshot(digest.to_owned());
        match action {
            CacheAction::Inspect | CacheAction::Verify => {
                let detail = if matches!(action, CacheAction::Verify) {
                    control.verify_cached(&root, &selector, None, cancel)?
                } else {
                    control.inspect_cached(&root, &selector, None)?
                };
                let manifest = detail.manifest;
                let source = manifest.source();
                Ok(format!(
                    "{}\nSnapshot: {}\n{}:{}@{}\n{} files · {} bytes\nNo execution or remote freshness qualification.",
                    if detail.payload_verified {
                        "Payloads verified"
                    } else {
                        "Manifest identity checked; payloads not verified"
                    },
                    detail.snapshot,
                    source.provider(),
                    source.repository(),
                    source.revision(),
                    manifest.entries().len(),
                    manifest.total_bytes()
                ))
            }
            CacheAction::Alias(name) => {
                let result = control.alias_cached(&root, digest, &name)?;
                Ok(format!(
                    "Alias by-name:{} {}\nSnapshot: {}{}",
                    result.name(),
                    if result.changed() {
                        "created"
                    } else {
                        "already bound"
                    },
                    digest,
                    if result.projection_warning().is_some() {
                        "\nBy-name projection needs refresh"
                    } else {
                        ""
                    }
                ))
            }
            CacheAction::Reclaim(apply) => {
                let result = control.reclaim_cached(
                    &root,
                    digest,
                    if apply {
                        ReclaimMode::Apply
                    } else {
                        ReclaimMode::DryRun
                    },
                    cancel,
                )?;
                match result {
                    ReclaimOutcome::DryRun(report) => Ok(format!(
                        "Removal preview; no mutation\n{} candidate files · {} candidate directories\nApply revalidates exact target.",
                        report.candidate_files(),
                        report.candidate_directories()
                    )),
                    ReclaimOutcome::Durable(report) => Ok(format!(
                        "Removal durable\n{} removed files · {} removed directories{}",
                        report.removed_files(),
                        report.removed_directories(),
                        if report.by_name_warning().is_some() {
                            "\nBy-name projection needs refresh"
                        } else {
                            ""
                        }
                    )),
                    ReclaimOutcome::Busy => Err(Error::State(
                        "Cache removal busy; retry after active readers close".into(),
                    )),
                    ReclaimOutcome::Referenced { references } => Err(Error::State(format!(
                        "Snapshot retained by {references} references; none removed"
                    ))),
                    ReclaimOutcome::Incomplete { cause, .. } => Err(Error::State(format!(
                        "Removal incomplete; inspect and retry the same digest. Completed changes were not rolled back: {cause}"
                    ))),
                }
            }
        }
    }
    #[cfg(not(all(feature = "cache", target_os = "linux")))]
    {
        match action {
            CacheAction::Reclaim(apply) => {
                let _ = apply;
            }
            CacheAction::Alias(name) => {
                let _ = name;
            }
            _ => {}
        }
        let _ = (control, digest, cancel);
        Err(Error::Unsupported(
            "Cache operations require Linux and cache feature".into(),
        ))
    }
}

fn invoke(
    runtime: &Runtime,
    owner: Owner,
    prompt: Prompt,
    cancel: &AtomicBool,
    output: &Mutex<Response>,
) -> Result<String> {
    if cancel.load(Ordering::Acquire) {
        return Ok("Cancelled before request admission".into());
    }
    let scheduler = runtime
        .scheduler_snapshot()?
        .ok_or_else(|| Error::State("No ready text scheduler".into()))?;
    let max_output = NonZeroUsize::new(scheduler.config.output)
        .ok_or_else(|| Error::State("No admitted output capacity".into()))?;
    let mut controls = TextControls::new(max_output);
    controls.chat = prompt.chat;
    let mut stream = runtime
        .control()
        .invoke(owner.reference(), &prompt.text, controls)
        .map_err(|e| e.into_core())?;
    {
        let mut response = output
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        response.owner = Some(owner);
        response.revision = response.revision.wrapping_add(1);
    }
    loop {
        if cancel.load(Ordering::Acquire) {
            stream.cancel()?;
            return Ok("Cancellation requested; output stream closed".into());
        }
        match stream
            .poll_next(Duration::from_millis(100))
            .map_err(|e| e.into_core())?
        {
            OutputPoll::Pending => {}
            OutputPoll::Event(
                ResponseEvent::Text(text) | ResponseEvent::TokenText { text, .. },
            ) => {
                output
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .append(&text);
            }
            OutputPoll::Event(ResponseEvent::Done {
                output: text,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            }) => {
                output
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .done(&text);
                return Ok(format!(
                    "{completion_tokens} output tokens · {prompt_tokens} prompt tokens · {finish_reason}"
                ));
            }
            OutputPoll::Event(ResponseEvent::Error(error)) => return Err(error.into_core()),
            OutputPoll::Finished => {
                return Err(Error::State(
                    "Text stream ended without a terminal result".into(),
                ));
            }
        }
    }
}

pub(super) use crate::control_host::{combine, retire};
