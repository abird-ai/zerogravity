//! One scoped, client-owned action worker. All effects go through the original
//! resident loader and ControlPlane; closing a view does not own retirement.
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
    control::{RegisteredModel, ServeSummary},
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
    },
    Invoke {
        owner: Owner,
        prompt: Prompt,
    },
    Unload(Owner),
    Start(String),
    Stop(u64),
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

#[derive(Clone)]
pub(super) struct Document {
    selection: Option<(String, String)>,
    pub chat: bool,
}
impl Document {
    #[cfg(test)]
    pub fn test_default() -> Self {
        Self {
            selection: None,
            chat: true,
        }
    }

    pub fn new(selection: Option<(&str, &str)>) -> Result<Self> {
        let chat = crate::config_cli::read_selected_document(selection)?
            .and_then(|(_, doc)| doc.run?.task?.raw)
            != Some(true);
        Ok(Self {
            selection: selection.map(|(flag, value)| (flag.to_owned(), value.to_owned())),
            chat,
        })
    }
    fn read(
        &self,
    ) -> Result<
        Option<(
            std::path::PathBuf,
            zerogravity::configuration::ConfigDocument,
        )>,
    > {
        crate::config_cli::read_selected_document(
            self.selection
                .as_ref()
                .map(|(flag, value)| (flag.as_str(), value.as_str())),
        )
    }
}

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

pub(super) struct Job {
    title: &'static str,
    worker: Option<JoinHandle<Result<String>>>,
    cancel: Arc<AtomicBool>,
    response: Arc<Mutex<Response>>,
    generates: bool,
}
impl Job {
    pub fn start(runtime: &Runtime, document: &Document, request: Request) -> Result<Self> {
        let title = request.title();
        let generates = request.generates();
        let runtime = runtime.clone();
        let document = document.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let response = Arc::new(Mutex::new(Response::default()));
        let flag = Arc::clone(&cancel);
        let output = Arc::clone(&response);
        let worker = thread::Builder::new()
            .name("zg-control-action".into())
            .spawn(move || execute(&runtime, &document, request, &flag, &output))?;
        Ok(Self {
            title,
            worker: Some(worker),
            cancel,
            response,
            generates,
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
    pub fn join(&mut self) -> Result<String> {
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
        Request::Load { source, prompt } => {
            if runtime.status().phase != RuntimePhase::Empty {
                return Err(Error::State(
                    "Retire the current route set before loading another".into(),
                ));
            }
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
            let (request, settings) = resolve_serving(
                document.read()?,
                RunOverrides::default(),
                models,
                &std::env::current_dir()?,
            )?;
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

/// Foreground process exit, not a client navigation operation. The worker is
/// joined by the caller first; no concurrent start/stop remains in this UI.
pub(super) fn retire(runtime: &Runtime) -> Result<()> {
    #[cfg(feature = "serve")]
    let stopped = (|| match runtime.control().serving()? {
        ServeSummary::Listening { binding, .. } | ServeSummary::Finished { binding } => {
            runtime.control().stop_serving(binding.generation)
        }
        ServeSummary::Idle | ServeSummary::Unavailable => Ok(()),
        ServeSummary::Transitioning => {
            Err(Error::State("Listener transition still in progress".into()))
        }
    })();
    #[cfg(not(feature = "serve"))]
    let stopped = {
        let _ = ServeSummary::Unavailable;
        Ok(())
    };
    combine(stopped, runtime.shutdown())
}

pub(super) fn combine(primary: Result<()>, cleanup: Result<()>) -> Result<()> {
    match (primary, cleanup) {
        (Ok(()), result) | (result, Ok(())) => result,
        (Err(primary), Err(cleanup)) => Err(Error::State(format!("{primary}; cleanup: {cleanup}"))),
    }
}
