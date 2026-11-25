//! Shared Rust control plane below CLI/TUI/Web. Original runtime, cache and
//! transport owners remain authoritative; no frontend-owned execution state.
#[cfg(all(feature = "cache", target_os = "linux"))]
mod cache;
mod events;
#[cfg(feature = "serve")]
mod serving;

#[cfg(all(feature = "cache", target_os = "linux"))]
pub use cache::{CachedModelDetail, CachedModelSummary};
pub use events::{ControlEvent, ControlEventKind, ControlEvents, ControlPoll, ReclaimStatus};
#[cfg(feature = "serve")]
pub use serving::ServeOptions;

use crate::{
    Error, Result, Runtime,
    devices::{DeviceRef, Inventory, Observation},
    diagnostics::{RuntimeSnapshot, TaskObservation},
    models::{ModelLoadRequest, TextLoadPolicy},
    pipelines::{CheckedDocument, ModelInstance},
    run::{OutputStream, TextControls, TextTask},
    runtime::text::ServiceError,
};
use serde::Serialize;
use std::{net::SocketAddr, path::Path};

#[derive(Default)]
pub(crate) struct ControlState {
    events: events::Events,
    #[cfg(feature = "serve")]
    serving: serving::ServeState,
}

/// A borrowed view of the same Runtime. Closing a client view never shuts
/// down models, requests, or the listener owned by the process/runtime.
#[derive(Clone, Copy)]
pub struct ControlPlane<'a> {
    runtime: &'a Runtime,
    state: &'a ControlState,
}

/// Exact observed registration in this runtime, not a source or alias permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisteredModel<'a> {
    pub alias: &'a str,
    pub registration_id: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ServeBinding {
    pub generation: u64,
    pub address: SocketAddr,
}

/// Only the listener owned by this ControlPlane; an independently started
/// SDK listener or another process is not inferred from port observations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ServeSummary {
    Unavailable,
    Idle,
    Transitioning,
    Listening {
        binding: ServeBinding,
        accepted_requests: u64,
        active_connections: usize,
    },
    /// Worker finished, but its terminal result is not observed until join.
    Finished {
        binding: ServeBinding,
    },
}

/// Bounded live owner observations, not a stored second catalog. Cold device,
/// cache and document views are requested explicitly, independently of Home.
#[derive(Clone, Debug, Serialize)]
pub struct ControlSnapshot {
    pub runtime: RuntimeSnapshot,
    pub serving: ServeSummary,
}

/// Trusted native actions use the existing typed SDK requests, not raw argv.
/// The future wire adapter must validate/authenticate its own input before
/// constructing these actions. No action implicitly downloads or starts a UI.
pub enum ControlAction<'a> {
    Load(ModelLoadRequest),
    LoadDeclared {
        models: &'a [ModelInstance],
        policy: TextLoadPolicy,
        cache_root: Option<&'a Path>,
    },
    Unload(RegisteredModel<'a>),
    Invoke {
        model: RegisteredModel<'a>,
        input: &'a str,
        controls: TextControls,
    },
    InspectTask(TextTask<'a>),
    Devices,
    InspectDevice(&'a DeviceRef),
    CheckConfig(&'a Path),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    CachedModels(&'a Path),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    InspectCached {
        root: &'a Path,
        selector: &'a crate::run_policy::SourceSelector,
        pin: Option<&'a str>,
    },
    #[cfg(all(feature = "cache", target_os = "linux"))]
    VerifyCached {
        root: &'a Path,
        selector: &'a crate::run_policy::SourceSelector,
        pin: Option<&'a str>,
        cancel: &'a std::sync::atomic::AtomicBool,
    },
    #[cfg(all(feature = "cache", target_os = "linux"))]
    ReclaimCached {
        root: &'a Path,
        digest: &'a str,
        mode: crate::cache::ReclaimMode,
        cancel: &'a std::sync::atomic::AtomicBool,
    },
    #[cfg(feature = "serve")]
    StartServing(ServeOptions),
    #[cfg(feature = "serve")]
    StopServing {
        generation: u64,
    },
}

pub enum ControlReply {
    Done,
    Output(Box<dyn OutputStream>),
    Task(Box<TaskObservation>),
    Devices(Inventory),
    Device(Observation),
    Config(Box<CheckedDocument>),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    CachedModels(Vec<CachedModelSummary>),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    CachedModel(CachedModelDetail),
    #[cfg(all(feature = "cache", target_os = "linux"))]
    Reclamation(crate::cache::ReclaimOutcome),
    #[cfg(feature = "serve")]
    Serving(ServeBinding),
}

/// Preserve original core and structured service refusal semantics.
#[derive(Debug)]
pub enum ControlError {
    Core(Error),
    Service(ServiceError),
}
impl From<Error> for ControlError {
    fn from(value: Error) -> Self {
        Self::Core(value)
    }
}
impl From<ServiceError> for ControlError {
    fn from(value: ServiceError) -> Self {
        Self::Service(value)
    }
}
impl std::fmt::Display for ControlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core(error) => error.fmt(f),
            Self::Service(error) => error.fmt(f),
        }
    }
}
impl std::error::Error for ControlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            Self::Service(error) => Some(error),
        }
    }
}

impl<'a> ControlPlane<'a> {
    pub(crate) fn new(runtime: &'a Runtime, state: &'a ControlState) -> Self {
        Self { runtime, state }
    }

    pub fn snapshot(&self) -> Result<ControlSnapshot> {
        Ok(ControlSnapshot {
            runtime: self.runtime.snapshot()?,
            #[cfg(feature = "serve")]
            serving: self.state.serving.observe()?,
            #[cfg(not(feature = "serve"))]
            serving: ServeSummary::Unavailable,
        })
    }

    /// Transient events of actions invoked through this plane. Direct SDK,
    /// HTTP requests and external cache changes remain visible on fresh owner
    /// observations, not an invented complete activity history.
    pub fn subscribe(&self) -> ControlEvents<'_> {
        self.state.events.subscribe()
    }

    fn emit(&self, kind: ControlEventKind) {
        self.state.events.publish(kind);
    }

    pub fn load(&self, request: ModelLoadRequest) -> std::result::Result<(), ServiceError> {
        let result = self.runtime.models().load(request);
        self.emit(ControlEventKind::ModelLoadFinished {
            success: result.is_ok(),
        });
        result
    }

    pub fn load_declared(
        &self,
        models: &[ModelInstance],
        policy: TextLoadPolicy,
        cache_root: Option<&Path>,
    ) -> std::result::Result<(), ServiceError> {
        let result = self
            .runtime
            .models()
            .load_declared(models, policy, cache_root);
        self.emit(ControlEventKind::ModelLoadFinished {
            success: result.is_ok(),
        });
        result
    }

    pub fn unload(&self, model: RegisteredModel<'_>) -> Result<()> {
        let result = self
            .runtime
            .engine
            .unload_registration(model.alias, model.registration_id);
        self.emit(ControlEventKind::ModelUnloadFinished {
            registration_id: model.registration_id,
            success: result.is_ok(),
        });
        result
    }

    /// Return the original cancellable output stream, not a replicated token
    /// or completion channel. Accepted is not started/completed; terminal
    /// results remain with that stream and scheduler observations.
    pub fn invoke(
        &self,
        model: RegisteredModel<'_>,
        input: &str,
        controls: TextControls,
    ) -> std::result::Result<Box<dyn OutputStream>, ServiceError> {
        let result = self.runtime.engine.submit_registered_task(
            model.alias,
            model.registration_id,
            input,
            controls,
        );
        if result.is_ok() {
            self.emit(ControlEventKind::RequestAccepted {
                registration_id: model.registration_id,
            });
        }
        result
    }

    pub fn inspect_task(
        &self,
        task: TextTask<'_>,
    ) -> std::result::Result<TaskObservation, ServiceError> {
        self.runtime.inspect_task(task)
    }

    pub fn devices(&self) -> Result<Inventory> {
        self.runtime.devices().inventory()
    }

    pub fn inspect_device(&self, selector: &DeviceRef) -> Result<Observation> {
        let inventory = self.devices()?;
        Ok(crate::devices::observe_selected(&inventory, selector)?.clone())
    }

    pub fn check_config(&self, path: &Path) -> Result<CheckedDocument> {
        self.runtime.engine.require_open()?;
        crate::pipelines::load_document(path)?.check()
    }

    /// Synchronous owner operations; callers may run slow work on their
    /// bounded worker. No extra executor, mutable model table or task queue.
    pub fn dispatch(
        &self,
        action: ControlAction<'_>,
    ) -> std::result::Result<ControlReply, ControlError> {
        use ControlAction as A;
        use ControlReply as R;
        Ok(match action {
            A::Load(request) => {
                self.load(request)?;
                R::Done
            }
            A::LoadDeclared {
                models,
                policy,
                cache_root,
            } => {
                self.load_declared(models, policy, cache_root)?;
                R::Done
            }
            A::Unload(model) => {
                self.unload(model)?;
                R::Done
            }
            A::Invoke {
                model,
                input,
                controls,
            } => R::Output(self.invoke(model, input, controls)?),
            A::InspectTask(task) => R::Task(Box::new(self.inspect_task(task)?)),
            A::Devices => R::Devices(self.devices()?),
            A::InspectDevice(device) => R::Device(self.inspect_device(device)?),
            A::CheckConfig(path) => R::Config(Box::new(self.check_config(path)?)),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            A::CachedModels(root) => R::CachedModels(self.cached_models(root)?),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            A::InspectCached {
                root,
                selector,
                pin,
            } => R::CachedModel(self.inspect_cached(root, selector, pin)?),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            A::VerifyCached {
                root,
                selector,
                pin,
                cancel,
            } => R::CachedModel(self.verify_cached(root, selector, pin, cancel)?),
            #[cfg(all(feature = "cache", target_os = "linux"))]
            A::ReclaimCached {
                root,
                digest,
                mode,
                cancel,
            } => R::Reclamation(self.reclaim_cached(root, digest, mode, cancel)?),
            #[cfg(feature = "serve")]
            A::StartServing(options) => R::Serving(self.start_serving(options)?),
            #[cfg(feature = "serve")]
            A::StopServing { generation } => {
                self.stop_serving(generation)?;
                R::Done
            }
        })
    }
}
