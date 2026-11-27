use crate::{
    Error, Result,
    catalog::ModelCatalog,
    diagnostics::{ModelSnapshot, RuntimePhase, RuntimeSnapshot, RuntimeStatus},
    model_instance::ModelInstance,
    output::{OutputStream, TextOutputStream},
    request::{TaskInput, TaskRequest, TaskTarget, TextControls, TextTask},
    router::{Request, RequestEvents, Router, ServiceError},
    scheduler::SchedulerLimits,
};
use std::{
    borrow::Cow,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_core::{physical::ResourceArbiter, resource_arbiter::AllocationClass};

/// Clones share the actual scheduler, loaded sources, lifecycle and physical ledger.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}
struct Inner {
    closed: AtomicBool,
    resources: ResourceArbiter,
    limits: SchedulerLimits,
    text: Mutex<TextControl>,
    changed: Condvar,
    #[cfg(test)]
    waiters_changed: Condvar,
}
struct TextControl {
    phase: TextPhase,
    // Preserve the first lifecycle/worker failure for every shutdown caller.
    failure: Option<String>,
    #[cfg(test)]
    shutdown_waiters: usize,
}
// One inline router per runtime avoids another allocation and ownership handle.
#[allow(clippy::large_enum_variant)]
enum TextPhase {
    Empty,
    Loading(std::thread::ThreadId),
    Ready(Router),
    Retiring(Router, std::thread::ThreadId),
    Joining(std::thread::ThreadId),
    Closed,
}
impl TextControl {
    fn router(&self) -> Option<&Router> {
        match &self.phase {
            TextPhase::Ready(router) | TextPhase::Retiring(router, _) => Some(router),
            _ => None,
        }
    }
    fn status(&self, closed: bool) -> RuntimeStatus {
        let router = self.router();
        let (loaded_models, loaded_owners) = router.map_or((0, 0), Router::ready_counts);
        RuntimeStatus {
            closed,
            phase: match &self.phase {
                TextPhase::Empty => RuntimePhase::Empty,
                TextPhase::Loading(_) => RuntimePhase::Loading,
                TextPhase::Ready(_) => RuntimePhase::Ready,
                TextPhase::Retiring(..) => RuntimePhase::Retiring,
                TextPhase::Joining(_) => RuntimePhase::Joining,
                TextPhase::Closed => RuntimePhase::Closed,
            },
            failed: self.failure.is_some(),
            loaded_models,
            loaded_owners,
            execution_adapters: usize::from(router.is_some()),
        }
    }
    fn terminal_result(&self) -> Result<()> {
        match &self.failure {
            Some(message) => Err(Error::State(message.clone())),
            None => Ok(()),
        }
    }
}

enum UnloadTarget<'a> {
    Handle(&'a ModelInstance),
    Registration {
        alias: &'a str,
        registration_id: u64,
    },
}

/// A load/retirement owns the busy phase, not the mutex. On unwind its local
/// router retires first; then this guard releases waiters with a sticky failure.
struct LifecycleOperation<'a> {
    inner: &'a Inner,
    panic_message: &'static str,
    armed: bool,
}
impl<'a> LifecycleOperation<'a> {
    fn new(inner: &'a Inner, panic_message: &'static str) -> Self {
        Self {
            inner,
            panic_message,
            armed: true,
        }
    }
    fn finish(mut self) {
        self.complete(None);
    }
    fn complete(&mut self, failure: Option<&str>) {
        let mut text = self
            .inner
            .text
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if text.failure.is_none() {
            text.failure = failure.map(str::to_owned);
        }
        if matches!(text.phase, TextPhase::Retiring(..)) {
            let TextPhase::Retiring(router, _) =
                std::mem::replace(&mut text.phase, TextPhase::Empty)
            else {
                unreachable!()
            };
            // Even if shutdown raced, the remaining router must still join.
            text.phase = TextPhase::Ready(router);
        } else {
            text.phase = if self.inner.closed.load(Ordering::Acquire) {
                TextPhase::Closed
            } else {
                TextPhase::Empty
            };
        }
        self.armed = false;
        self.inner.changed.notify_all();
    }
}
impl Drop for LifecycleOperation<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.complete(Some(self.panic_message));
        }
    }
}

#[derive(Default)]
pub struct RuntimeBuilder {
    limits: SchedulerLimits,
}
impl RuntimeBuilder {
    pub fn limits(mut self, limits: SchedulerLimits) -> Self {
        self.limits = limits;
        self
    }
    pub fn build(self) -> Result<Runtime> {
        self.limits.validate()?;
        Ok(Runtime {
            inner: Arc::new(Inner {
                closed: AtomicBool::new(false),
                resources: ResourceArbiter::process_shared(),
                limits: self.limits,
                text: Mutex::new(TextControl {
                    phase: TextPhase::Empty,
                    failure: None,
                    #[cfg(test)]
                    shutdown_waiters: 0,
                }),
                changed: Condvar::new(),
                #[cfg(test)]
                waiters_changed: Condvar::new(),
            }),
        })
    }
}
impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }
    pub fn status(&self) -> RuntimeStatus {
        let text = self
            .inner
            .text
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        text.status(self.inner.closed.load(Ordering::Acquire))
    }
    /// Read-only even during load/retirement/shutdown. The process resource
    /// ledger remains observable after the runtime releases its route set.
    pub fn snapshot(&self) -> Result<RuntimeSnapshot> {
        let (status, failure, scheduler, catalog) = {
            let text = self
                .inner
                .text
                .lock()
                .map_err(|_| Error::State("runtime lock poisoned".into()))?;
            let router = text.router();
            (
                text.status(self.inner.closed.load(Ordering::Acquire)),
                text.failure.clone(),
                router.map(Router::scheduler_snapshot),
                router
                    .map(Router::catalog_snapshot)
                    .transpose()?
                    .unwrap_or_default(),
            )
        };
        let (resources, host_backing) = self.inner.resources.snapshots_with_host()?;
        Ok(RuntimeSnapshot {
            status,
            failure,
            limits: self.inner.limits,
            scheduler,
            catalog,
            resources,
            host_backing,
        })
    }
    /// Current original worker/queue counters without model or resource
    /// inventory. Presentation callers may sample this at a coarse cadence.
    pub fn scheduler_snapshot(&self) -> Result<Option<crate::router::worker::SchedulerSnapshot>> {
        let text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        Ok(text.router().map(Router::scheduler_snapshot))
    }
    pub(crate) fn catalog_snapshot(&self) -> Result<Vec<ModelSnapshot>> {
        let text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        text.router()
            .map(Router::catalog_snapshot)
            .transpose()
            .map(Option::unwrap_or_default)
    }
    pub fn resources(&self) -> &ResourceArbiter {
        &self.inner.resources
    }
    pub fn models(&self) -> ModelCatalog<'_> {
        ModelCatalog::new(self)
    }
    pub fn limits(&self) -> SchedulerLimits {
        self.inner.limits
    }
    /// Reserve the one load slot, then prepare sources outside the mutex. A
    /// racing shutdown closes admission and waits for returned workers to retire.
    /// Loader status callbacks are allowed; synchronous self-shutdown is rejected.
    pub fn configure_text(
        &self,
        load: impl FnOnce() -> std::result::Result<Router, ServiceError>,
    ) -> std::result::Result<(), ServiceError> {
        self.configure_text_with(load, |_| Ok(()))
    }

    /// Prepare another bounded cold owner against the unpublished router.
    /// Both owners become available only after the same load slot commits.
    /// Finalization must not expose the private router or run request work.
    pub(crate) fn configure_text_with<T>(
        &self,
        load: impl FnOnce() -> std::result::Result<Router, ServiceError>,
        finalize: impl FnOnce(&Router) -> std::result::Result<T, ServiceError>,
    ) -> std::result::Result<T, ServiceError> {
        {
            let mut text = self.inner.text.lock().map_err(|_| runtime_lock_error())?;
            if self.inner.closed.load(Ordering::Acquire) {
                return Err(runtime_closed());
            }
            if !matches!(text.phase, TextPhase::Empty) {
                return Err(ServiceError::new(
                    409,
                    "runtime_configured",
                    "text scheduler already configured or loading",
                ));
            }
            if let Some(failure) = &text.failure {
                return Err(ServiceError::new(503, "runtime_failed", failure.clone()));
            }
            text.phase = TextPhase::Loading(std::thread::current().id());
        }
        // Declared before the returned router: unwinding retires that owner
        // before the guard can mark loading complete or wake shutdown callers.
        let mut operation = LifecycleOperation::new(&self.inner, "text loader panicked");
        let router = match load() {
            Ok(router) => router,
            Err(error) => {
                operation.finish();
                return Err(error);
            }
        };
        let valid = router.active_limit() <= self.inner.limits.max_active
            && self
                .inner
                .limits
                .allows_pending(router.queue_snapshot().limit);
        let prepared = if !valid {
            Err(ServiceError::new(
                400,
                "invalid_config",
                "text scheduler exceeds runtime limits",
            ))
        } else if self.inner.closed.load(Ordering::Acquire) {
            Err(runtime_closed())
        } else {
            finalize(&router)
        };
        let refusal = match prepared {
            Ok(prepared) => {
                let mut text = self
                    .inner
                    .text
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                if !self.inner.closed.load(Ordering::Acquire) {
                    text.phase = TextPhase::Ready(router);
                    operation.armed = false;
                    self.inner.changed.notify_all();
                    return Ok(prepared);
                }
                drop(text);
                // The private result may retain admitted plan metadata.
                drop(prepared);
                runtime_closed()
            }
            Err(error) => error,
        };
        // Keep Loading until original workers, source readers and charges
        // retire. No router or private result is destroyed under the lock.
        let failure = self.retire_text_router(router);
        operation.finish();
        if let Some(message) = failure {
            Err(ServiceError::new(500, "worker_shutdown", message))
        } else if self.inner.closed.load(Ordering::Acquire) {
            Err(runtime_closed())
        } else {
            Err(refusal)
        }
    }

    /// Short snapshot/admission callback on the actual scheduler. Never hold
    /// this lock across transport IO or re-enter the runtime from the callback.
    pub fn with_text_router<T>(
        &self,
        callback: impl FnOnce(&Router) -> std::result::Result<T, ServiceError>,
    ) -> std::result::Result<T, ServiceError> {
        let text = self.inner.text.lock().map_err(|_| runtime_lock_error())?;
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(runtime_closed());
        }
        let router = text.router().ok_or_else(|| {
            ServiceError::new(503, "runtime_unconfigured", "no text scheduler configured")
        })?;
        callback(router)
    }
    /// Internal cold/task operation under the same original lifecycle lock.
    pub(crate) fn with_ready_text<T>(
        &self,
        callback: impl FnOnce(&Router) -> Result<T>,
    ) -> Result<T> {
        let text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        self.require_open()?;
        let router = text
            .router()
            .ok_or_else(|| Error::Unsupported("no source-bound text adapter configured".into()))?;
        callback(router)
    }

    /// Legacy text request adapter into the original bounded semantic queue.
    pub fn request_text(
        &self,
        request: Request,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        self.request_semantic(request.into())
    }
    pub fn request_semantic(
        &self,
        request: crate::router::SemanticRequest,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        self.with_text_router(|router| {
            request.validate_input(self.inner.limits.max_input_bytes)?;
            router.request_with(request, None, None, self.inner.limits.max_input_bytes)
        })
    }
    /// Native finite encoder batch on the same admitted runtime/router.
    pub fn request_embeddings(
        &self,
        request: crate::embedding::EmbeddingRequest,
    ) -> std::result::Result<crate::embedding::EmbeddingReceipt, ServiceError> {
        self.with_text_router(|router| {
            router.request_embeddings_with(request, self.inner.limits.max_input_bytes, None)
        })
    }

    pub fn public_embedding_dimensions(
        &self,
        alias: &str,
    ) -> std::result::Result<usize, ServiceError> {
        self.with_text_router(|router| router.public_embedding_dimensions(alias))
    }
    pub fn request_public_embeddings(
        &self,
        request: crate::embedding::EmbeddingRequest,
        dimensions: usize,
    ) -> std::result::Result<crate::embedding::EmbeddingReceipt, ServiceError> {
        self.with_text_router(|router| {
            router.request_embeddings_with(
                request,
                self.inner.limits.max_input_bytes,
                Some(dimensions),
            )
        })
    }

    /// Exact selected-model input count under the original lifecycle and
    /// configured semantic/rendered input limits. Does not admit generation.
    pub fn count_input_tokens(
        &self,
        model: &str,
        input: &crate::router::TextInput,
    ) -> std::result::Result<usize, ServiceError> {
        self.with_text_router(|router| {
            router.count_input_tokens_with(model, input, self.inner.limits.max_input_bytes)
        })
    }

    /// Explicit cross-request prefix sharing by the trusted embedding layer.
    pub fn request_text_scoped(
        &self,
        request: Request,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        self.request_semantic_scoped(request.into(), sharing)
    }
    pub fn request_semantic_scoped(
        &self,
        request: crate::router::SemanticRequest,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        self.with_text_router(|router| {
            request.validate_input(self.inner.limits.max_input_bytes)?;
            router.request_with(
                request,
                None,
                Some(sharing),
                self.inner.limits.max_input_bytes,
            )
        })
    }
    /// Admitted protocol payload, retaining the caller's existing ingress
    /// charges and this runtime's configured semantic/rendered input limit.
    pub fn request_semantic_ingress(
        &self,
        request: crate::router::SemanticRequest,
        ingress: Vec<crate::router::pending::ServingCharge>,
        sharing: Option<crate::state_cache::PrefixSharingScope>,
    ) -> std::result::Result<RequestEvents, ServiceError> {
        self.with_text_router(|router| {
            request.validate_input(self.inner.limits.max_input_bytes)?;
            router.request_with(
                request,
                Some(ingress),
                sharing,
                self.inner.limits.max_input_bytes,
            )
        })
    }
    pub fn invalidate_prefix_scope(
        &self,
        sharing: crate::state_cache::PrefixSharingScope,
    ) -> std::result::Result<crate::state_cache::PrefixCacheEviction, ServiceError> {
        self.with_text_router(|router| router.invalidate_prefix_scope(sharing))
    }
    pub fn run(&self, request: &TaskRequest) -> Result<Box<dyn OutputStream>> {
        self.require_open()?;
        request.validate(self.inner.limits.max_input_bytes)?;
        let TaskInput::Text(input) = &request.input else {
            return Err(Error::Unsupported("Fabric task driver unavailable".into()));
        };
        self.submit_task(TextTask {
            target: TaskTarget::Model(&request.model),
            input,
            controls: TextControls::new(request.max_output),
        })
        .map_err(task_error)
    }

    /// Same original router for model and pipeline work. Preserve the actual
    /// structured service refusal rather than flattening it to legacy Error.
    pub fn submit_task(
        &self,
        task: TextTask<'_>,
    ) -> std::result::Result<Box<dyn OutputStream>, ServiceError> {
        self.submit_task_checked(task, None)
    }

    /// Submit against the exact registration selected by a control client.
    /// Alias replacement cannot retarget the request between check and admission.
    pub fn submit_registered_task(
        &self,
        alias: &str,
        registration_id: u64,
        input: &str,
        controls: TextControls,
    ) -> std::result::Result<Box<dyn OutputStream>, ServiceError> {
        self.submit_task_checked(
            TextTask {
                target: TaskTarget::Model(alias),
                input,
                controls,
            },
            Some(registration_id),
        )
    }

    fn submit_task_checked(
        &self,
        task: TextTask<'_>,
        expected_registration: Option<u64>,
    ) -> std::result::Result<Box<dyn OutputStream>, ServiceError> {
        self.require_open().map_err(|_| runtime_closed())?;
        task.validate(self.inner.limits.max_input_bytes)?;
        self.with_text_router(|router| {
            if let Some(expected) = expected_registration
                && !matches!(task.target, TaskTarget::Model(alias)
                    if router.registration(alias).is_some_and(|owner| owner.id == expected))
            {
                return Err(ServiceError::new(
                    409,
                    "stale_registration",
                    "selected model registration is no longer ready",
                ));
            }
            task.require(router)?;
            match task.target {
                TaskTarget::Model(alias) => {
                    let charge = router.host().charge(
                        alias,
                        "sdk-output-stream",
                        AllocationClass::Output,
                        (std::mem::size_of::<TextOutputStream>() + 64) as u64,
                    )?;
                    let events =
                        submit_text(router, alias, Cow::Borrowed(task.input), task.controls)?;
                    Ok(Box::new(TextOutputStream::new(events, charge)) as Box<dyn OutputStream>)
                }
                TaskTarget::Pipeline(plan) => {
                    crate::pipeline::submit_pipeline(self, router, plan, task.input, task.controls)
                }
            }
        })
    }

    /// Cold selected-owner observation, serialized with original catalog
    /// admission. Never opens a source, admits a request or retains a reader.
    pub fn inspect_task(
        &self,
        task: TextTask<'_>,
    ) -> std::result::Result<crate::diagnostics::TaskObservation, ServiceError> {
        use crate::diagnostics::{TaskObservation, TaskStageObservation};
        self.require_open().map_err(|_| runtime_closed())?;
        task.validate(self.inner.limits.max_input_bytes)?;
        self.with_text_router(|router| {
            task.require(router)?;
            let (pipeline, stages) = match task.target {
                TaskTarget::Model(alias) => {
                    let registration = router.registration(alias).ok_or_else(|| {
                        ServiceError::new(404, "model_not_found", "model is not registered")
                    })?;
                    (
                        None,
                        vec![TaskStageObservation {
                            step: None,
                            alias: alias.into(),
                            registration_id: registration.id,
                        }],
                    )
                }
                TaskTarget::Pipeline(plan) => (
                    Some(plan.name().to_owned()),
                    plan.stages()
                        .iter()
                        .map(|stage| TaskStageObservation {
                            step: Some(stage.id().into()),
                            alias: stage.alias().into(),
                            registration_id: stage.registration_id(),
                        })
                        .collect(),
                ),
            };
            let registrations: Vec<_> = stages.iter().map(|stage| stage.registration_id).collect();
            let models = router
                .selected_catalog_snapshot(&registrations)
                .map_err(ServiceError::inference)?;
            if stages.iter().any(|stage| {
                !models.iter().any(|model| {
                    model.registration_id == stage.registration_id && model.capabilities.is_some()
                })
            }) {
                return Err(ServiceError::new(
                    503,
                    "model_retired",
                    "task selected owner is unavailable",
                ));
            }
            Ok(TaskObservation {
                pipeline,
                input_bytes: task.input.len(),
                input_limit_bytes: TextTask::input_limit(self.inner.limits.max_input_bytes),
                controls: task.controls,
                limits: self.inner.limits,
                scheduler: router.scheduler_snapshot(),
                stages,
                models,
            })
        })
    }
    pub(crate) fn loaded_text_models(&self) -> Result<Vec<ModelInstance>> {
        let text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        self.require_open()?;
        Ok(text.router().map_or_else(Vec::new, |router| {
            router
                .model_ids()
                .filter_map(|id| router.model(id))
                .collect()
        }))
    }
    pub(crate) fn resolve_text(&self, alias: &str) -> Result<ModelInstance> {
        let text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        self.require_open()?;
        let router = text
            .router()
            .ok_or_else(|| Error::Unsupported("no source-bound text adapter configured".into()))?;
        router
            .model(alias)
            .ok_or_else(|| Error::State(format!("model is not registered: {alias}")))
    }
    pub(crate) fn text_model_aliases(&self, model: &ModelInstance) -> Result<Vec<String>> {
        let text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        self.require_open()?;
        let router = text
            .router()
            .ok_or_else(|| Error::State("no source-bound text adapter configured".into()))?;
        router
            .aliases(model)
            .map(|aliases| aliases.map(str::to_owned).collect())
    }

    /// Close only the selected owner's admission, drain its accepted readers
    /// unlocked, then retire it. The last owner uses the existing whole join.
    pub(crate) fn unload_text_model(&self, model: &ModelInstance) -> Result<()> {
        self.unload_text_target(UnloadTarget::Handle(model))
    }

    /// Retire exactly the observed registration, under the original lifecycle
    /// lock. A stale UI row cannot unload a new owner that reused its alias.
    pub fn unload_registration(&self, alias: &str, registration_id: u64) -> Result<()> {
        self.unload_text_target(UnloadTarget::Registration {
            alias,
            registration_id,
        })
    }

    fn unload_text_target(&self, target: UnloadTarget<'_>) -> Result<()> {
        if crate::router::on_worker_thread() {
            return Err(Error::State(
                "blocking model unload from an engine worker is unsupported".into(),
            ));
        }
        let current = std::thread::current().id();
        let mut text = self
            .inner
            .text
            .lock()
            .map_err(|_| Error::State("runtime lock poisoned".into()))?;
        self.require_open()?;
        let TextPhase::Ready(router) = &text.phase else {
            return Err(Error::State(
                "no ready text adapter for model retirement".into(),
            ));
        };
        let selected = match target {
            UnloadTarget::Handle(model) => router.begin_unload(model)?,
            UnloadTarget::Registration {
                alias,
                registration_id,
            } => {
                if router
                    .registration(alias)
                    .is_none_or(|owner| owner.id != registration_id)
                {
                    return Err(Error::State(
                        "selected model registration is no longer ready".into(),
                    ));
                }
                router.begin_unload_registration(alias, registration_id)?
            }
        };
        if selected.is_none() {
            router.close_admission();
        }
        let TextPhase::Ready(router) =
            std::mem::replace(&mut text.phase, TextPhase::Joining(current))
        else {
            unreachable!("matched ready router");
        };
        if let Some(selected) = selected {
            text.phase = TextPhase::Retiring(router, current);
            drop(text);
            let mut operation =
                LifecycleOperation::new(&self.inner, "selective model retirement panicked");
            let result = selected.finish();
            let failure = result.as_ref().err().map(ToString::to_string);
            operation.complete(failure.as_deref());
            result
        } else {
            drop(text);
            let operation = LifecycleOperation::new(&self.inner, "text model retirement panicked");
            let failure = self.retire_text_router(router);
            operation.finish();
            failure.map_or(Ok(()), |message| Err(Error::State(message)))
        }
    }

    /// Close admission under the lifecycle lock before waiting for retirement.
    /// One caller retires outside the mutex; other callers see the same result.
    /// Calls from engine workers or this runtime's loader/retirement owner are
    /// rejected before changing admission: blocking joins could wait on themselves.
    pub fn shutdown(&self) -> Result<()> {
        if crate::router::on_worker_thread() {
            return Err(Error::State(
                "blocking runtime shutdown from an engine worker is unsupported".into(),
            ));
        }
        let current = std::thread::current().id();
        let router = {
            let mut text = self
                .inner
                .text
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if matches!(&text.phase, TextPhase::Loading(owner) | TextPhase::Joining(owner) | TextPhase::Retiring(_, owner) if *owner == current)
            {
                return Err(Error::State(
                    "blocking runtime shutdown from its loader/retirement owner is unsupported"
                        .into(),
                ));
            }
            self.inner.closed.store(true, Ordering::Release);
            loop {
                match &text.phase {
                    TextPhase::Empty => {
                        text.phase = TextPhase::Closed;
                        self.inner.changed.notify_all();
                        return text.terminal_result();
                    }
                    TextPhase::Closed => return text.terminal_result(),
                    TextPhase::Ready(router) => {
                        router.close_admission();
                        let TextPhase::Ready(router) =
                            std::mem::replace(&mut text.phase, TextPhase::Joining(current))
                        else {
                            unreachable!("matched ready router");
                        };
                        break router;
                    }
                    TextPhase::Loading(_) | TextPhase::Joining(_) | TextPhase::Retiring(..) => {
                        #[cfg(test)]
                        {
                            text.shutdown_waiters += 1;
                            self.inner.waiters_changed.notify_all();
                        }
                        text = self
                            .inner
                            .changed
                            .wait(text)
                            .unwrap_or_else(std::sync::PoisonError::into_inner);
                        #[cfg(test)]
                        {
                            text.shutdown_waiters -= 1;
                        }
                    }
                }
            }
        };
        let operation = LifecycleOperation::new(&self.inner, "text shutdown panicked");
        let _ = self.retire_text_router(router);
        operation.finish();
        self.inner
            .text
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .terminal_result()
    }
    /// Consumes the actual router so its source/charge destruction also occurs
    /// unlocked and before the lifecycle operation releases any waiters.
    fn retire_text_router(&self, mut router: Router) -> Option<String> {
        let failure = router.shutdown().err().map(|error| error.to_string());
        if let Some(message) = &failure {
            let mut text = self
                .inner
                .text
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if text.failure.is_none() {
                text.failure = Some(message.clone());
            }
        }
        drop(router);
        failure
    }
    pub fn require_open(&self) -> Result<()> {
        if self.inner.closed.load(Ordering::Acquire) {
            Err(Error::State("Fabric runtime is shut down".into()))
        } else {
            Ok(())
        }
    }
    #[cfg(test)]
    pub(crate) fn wait_for_shutdown_waiters(&self, count: usize) {
        let text = self.inner.text.lock().unwrap();
        let (text, _) = self
            .inner
            .waiters_changed
            .wait_timeout_while(text, std::time::Duration::from_secs(5), |text| {
                text.shutdown_waiters < count
            })
            .unwrap();
        assert!(
            text.shutdown_waiters >= count,
            "shutdown callers did not reach terminal wait"
        );
    }
}
fn runtime_lock_error() -> ServiceError {
    ServiceError::new(500, "runtime_state", "runtime lock poisoned")
}
fn runtime_closed() -> ServiceError {
    ServiceError::new(503, "runtime_closed", "Fabric runtime is shut down")
}
/// Input ownership remains with the caller until the original ingress
/// reservation takes over. Owned pipeline text is moved, not copied.
pub(crate) fn submit_text(
    router: &Router,
    alias: &str,
    input: Cow<'_, str>,
    controls: TextControls,
) -> std::result::Result<RequestEvents, ServiceError> {
    let input_bytes = match &input {
        Cow::Borrowed(text) => text.len(),
        Cow::Owned(text) => text.capacity(),
    };
    let bytes = alias
        .len()
        .checked_add(input_bytes)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| ServiceError::new(400, "size_overflow", "text request copy overflow"))?;
    let ingress =
        router
            .host()
            .charge(alias, "sdk-request-copy", AllocationClass::Scratch, bytes)?;
    router.request_semantic_admitted(
        crate::router::SemanticRequest {
            model: alias.to_owned(),
            input: crate::router::TextInput::from_text(input.into_owned(), controls.chat),
            max_tokens: controls.max_output.get(),
            stream: controls.stream,
            token_metrics: controls
                .observe_tokens
                .then(|| crate::router::TokenMetricsRequest {
                    started: std::time::Instant::now(),
                    wire_request_id: None,
                }),
        },
        vec![ingress],
    )
}

/// Legacy SDK conversion uses the same typed service-error boundary.
pub(crate) fn task_error(error: ServiceError) -> Error {
    error.into_core()
}

pub(crate) fn service_error(error: ServiceError) -> Error {
    error.into_core()
}
