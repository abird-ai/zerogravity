//! Shared action execution. Original resident planner, ControlPlane and
//! OutputStream remain the only execution owners.
use super::settings;
use super::{Document, execution::Execution};
use crate::resident;
use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};
use zerogravity::{
    Error, Result, Runtime,
    control::RegisteredModel,
    diagnostics::RuntimePhase,
    models::ModelLifecycle,
    run::{OutputPoll, ResponseEvent, TextControls, config::resolve_serving},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Owner {
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
pub(crate) struct Prompt {
    pub text: String,
    pub chat: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Request {
    Load {
        source: String,
        prompt: Option<Prompt>,
        execution: Execution,
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
    #[cfg(all(feature = "web", feature = "acquisition", target_os = "linux"))]
    PullPreview(crate::pull::native::Intent),
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
            #[cfg(all(feature = "web", feature = "acquisition", target_os = "linux"))]
            Self::PullPreview(_) => "Pull preview",
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

/// Bounded presentation copy only, independent of the original output stream.
const MAX_DISPLAY_BYTES: usize = 64 * 1024;
#[derive(Clone, Debug, Default)]
pub(crate) struct Response {
    pub owner: Option<Owner>,
    pub text: String,
    pub clipped: bool,
    pub revision: u64,
    #[cfg(feature = "web")]
    pub usage: Option<Usage>,
}
#[cfg(feature = "web")]
#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    finish_reason: &'static str,
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

pub(crate) enum Completed {
    Message(String),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Refused {
        message: String,
        native_detail: Option<String>,
    },
    Settings(Box<settings::Outcome>),
}

fn resolve_request(
    document: &Document,
    source: &str,
    execution: Execution,
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
        execution.overrides(),
        models,
        &std::env::current_dir()?,
    )
}

pub(super) fn perform(
    runtime: &Runtime,
    selection: &super::Selection,
    document: &Document,
    request: Request,
    cancel: &Arc<AtomicBool>,
    output: &Mutex<Response>,
    origin: super::Origin,
) -> Result<Completed> {
    if let Request::Settings(action) = request {
        if cancel.load(Ordering::Acquire) {
            return Ok(Completed::Message("Cancelled before starting".into()));
        }
        let changes = !matches!(&action, settings::Action::Validate(_));
        let result = settings::perform(action, &document.target);
        // A failed publication can still have changed durable bytes.
        if result.is_err() && changes {
            selection.configuration_changed();
        }
        let mut outcome = result?;
        #[cfg(feature = "web")]
        if origin == super::Origin::Web
            && outcome
                .active
                .as_ref()
                .is_some_and(|(target, _)| target.profile.is_none())
        {
            outcome.message = "Updated portable configuration for future actions".into();
        }
        // Commit selection while the same action is still the slot owner.
        // A competing Web action must not read the previous selection.
        if let Some((target, config)) = outcome.active.take() {
            selection.update(Document::from_config(target, config, document.offline));
        }
        if changes {
            selection.configuration_changed();
        }
        return Ok(Completed::Settings(Box::new(outcome)));
    }
    if cancel.load(Ordering::Acquire) {
        return Ok(Completed::Message("Cancelled before starting".into()));
    }
    let control = runtime.control();
    let browser = match origin {
        #[cfg(feature = "web")]
        super::Origin::Web => true,
        #[cfg(feature = "tui")]
        super::Origin::Native => false,
    };
    match request {
        Request::Inspect(digest) => {
            cache_action(control, CacheAction::Inspect, &digest, cancel, browser)
        }
        Request::Verify(digest) => {
            cache_action(control, CacheAction::Verify, &digest, cancel, browser)
        }
        Request::Reclaim { digest, apply } => cache_action(
            control,
            CacheAction::Reclaim(apply),
            &digest,
            cancel,
            browser,
        ),
        Request::Alias { digest, name } => {
            cache_action(control, CacheAction::Alias(name), &digest, cancel, browser)
        }
        other => execute(runtime, document, other, cancel, output).map(Completed::Message),
    }
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
        Request::Inspect(_)
        | Request::Verify(_)
        | Request::Reclaim { .. }
        | Request::Alias { .. } => unreachable!("cache worker dispatch"),
        Request::CheckFit { source, device } => {
            let (request, settings) =
                resolve_request(document, &source, Execution::on_device(device.clone()))?;
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
        #[cfg(all(feature = "web", feature = "acquisition", target_os = "linux"))]
        Request::PullPreview(intent) => intent.confirmation(),
        Request::Load {
            source,
            prompt,
            execution,
        } => {
            if runtime.status().phase != RuntimePhase::Empty {
                return Err(Error::State(
                    "Retire the current route set before loading another".into(),
                ));
            }
            let (request, settings) = resolve_request(document, &source, execution)?;
            if prompt.is_some() && request.graph.models.len() != 1 {
                return Err(Error::InvalidInput("Quick run requires one source; load the configured route set and select an owner".into()));
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
                    settings.enable_error_details.unwrap_or(false),
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
    browser: bool,
) -> Result<Completed> {
    #[cfg(all(feature = "cache", target_os = "linux"))]
    {
        use zerogravity::{
            cache::{ReclaimMode, ReclaimOutcome},
            run_policy::SourceSelector,
        };
        let root = super::inventory::cache_root()?;
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
                Ok(Completed::Message(format!(
                    "{}\nSnapshot: {}\n{}:{}@{}\n{} files · {} bytes\nNo execution or remote freshness qualification.",
                    if detail.payload_verified {
                        "Payloads verified"
                    } else {
                        "Manifest identity checked; payloads not verified"
                    },
                    detail.snapshot,
                    source.provider(),
                    if browser {
                        super::inventory::public_repository(source.provider(), source.repository())
                    } else {
                        source.repository()
                    },
                    if browser {
                        super::inventory::public_revision(source.provider(), source.revision())
                    } else {
                        source.revision()
                    },
                    manifest.entries().len(),
                    manifest.total_bytes()
                )))
            }
            CacheAction::Alias(name) => {
                let result = control.alias_cached(&root, digest, &name)?;
                Ok(Completed::Message(format!(
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
                )))
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
                    ReclaimOutcome::DryRun(report) => Ok(Completed::Message(format!(
                        "Removal preview; no mutation\n{} candidate files · {} candidate directories\nApply revalidates exact target.",
                        report.candidate_files(),
                        report.candidate_directories()
                    ))),
                    ReclaimOutcome::Durable(report) => Ok(Completed::Message(format!(
                        "Removal durable\n{} removed files · {} removed directories{}",
                        report.removed_files(),
                        report.removed_directories(),
                        if report.by_name_warning().is_some() {
                            "\nBy-name projection needs refresh"
                        } else {
                            ""
                        }
                    ))),
                    ReclaimOutcome::Busy => Ok(Completed::Refused {
                        message: "Cache removal busy; retry after active readers close".into(), native_detail: None,
                    }),
                    ReclaimOutcome::Referenced { references } => Ok(Completed::Refused {
                        message: format!("Snapshot retained by {references} references; none removed"), native_detail: None,
                    }),
                    ReclaimOutcome::Incomplete { cause, .. } => Ok(Completed::Refused {
                        message: "Removal incomplete; inspect and retry the same digest. Completed changes were not rolled back".into(),
                        native_detail: Some(cause.to_string()),
                    }),
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
        let _ = (control, digest, cancel, browser);
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
                ..
            }) => {
                let mut response = output
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                response.done(&text);
                #[cfg(feature = "web")]
                {
                    response.usage = Some(Usage {
                        prompt_tokens,
                        completion_tokens,
                        finish_reason,
                    });
                }
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
