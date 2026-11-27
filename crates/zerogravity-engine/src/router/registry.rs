//! One immutable alias table; aliases share the exact typed prepared owner.
//! Locks occur at admission, request handoff and retirement, never per token.
use super::{
    ServiceError,
    performance::{RegistrationPerformance, RequestTiming},
    worker::SchedulerSnapshot,
};
use crate::model_instance::{
    EmbeddingEncoder, ModelInstance, PreparedModel,
    execution::{Executor, ModelHandle, ModelRetirement},
};
use crate::state_cache::{PrefixCacheEviction, PrefixSharingScope};
use crate::transcription::SpeechRecognizer;
use crate::{
    Error, Result,
    diagnostics::{ModelCapabilities, ModelLifecycle, ModelSnapshot},
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Condvar, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};
use zerogravity_core::identity::SourceIdentity;

/// A single owner may be visible, draining, or retired. Its stable alias
/// keys stay until the enclosing route set retires; no separate live catalog.
struct OwnerState<M> {
    model: Option<M>,
    lifecycle: ModelLifecycle,
    readers: usize,
}

/// Process-local registration, distinct from the source identity. Aliases
/// share it; even registering the same Arc in a new route set issues a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegisteredOwner {
    pub(crate) id: u64,
    pub(crate) source: SourceIdentity,
}
static NEXT_REGISTRATION: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
enum RouteModel<M> {
    Text(M),
    Encoder(Arc<dyn EmbeddingEncoder>),
    Speech(Arc<dyn SpeechRecognizer>),
}
impl<M: ModelHandle> RouteModel<M> {
    fn source(&self) -> SourceIdentity {
        match self {
            Self::Text(model) => model.model().source(),
            Self::Encoder(model) => model.source(),
            Self::Speech(model) => model.source(),
        }
    }
}

pub(super) struct Route<A: Executor> {
    registration: RegisteredOwner,
    state: Mutex<OwnerState<RouteModel<A::Model>>>,
    drained: Condvar,
    retire: Option<ModelRetirement<A::Model>>,
    independent_retirement: bool,
    public_embeddings: bool,
    public_transcription: bool,
    performance: RegistrationPerformance,
}
impl<A: Executor> Route<A> {
    fn new(model: RouteModel<A::Model>) -> Result<Self> {
        let retire = A::selective_retirement();
        let independent_retirement = !matches!(&model, RouteModel::Text(_)) || retire.is_some();
        let public_embeddings =
            matches!(&model, RouteModel::Encoder(encoder) if encoder.public_embeddings());
        let public_transcription =
            matches!(&model, RouteModel::Speech(speech) if speech.public_transcription());
        let id = NEXT_REGISTRATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| Error::Limit("model registration identity exhausted".into()))?;
        Ok(Self {
            registration: RegisteredOwner {
                id,
                source: model.source(),
            },
            state: Mutex::new(OwnerState {
                model: Some(model),
                lifecycle: ModelLifecycle::Ready,
                readers: 0,
            }),
            drained: Condvar::new(),
            retire,
            independent_retirement,
            public_embeddings,
            public_transcription,
            performance: RegistrationPerformance::new(std::time::Instant::now()),
        })
    }
    fn state(&self) -> MutexGuard<'_, OwnerState<RouteModel<A::Model>>> {
        // No executor work or destructors run in this critical section.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn accepting(&self) -> bool {
        self.state().lifecycle == ModelLifecycle::Ready
    }
    fn model(&self) -> Option<RouteModel<A::Model>> {
        let state = self.state();
        if state.lifecycle == ModelLifecycle::Ready {
            state.model.clone()
        } else {
            None
        }
    }
    fn matches(&self, model: &Arc<dyn PreparedModel>) -> bool {
        matches!(self.model(), Some(RouteModel::Text(current)) if Arc::ptr_eq(&current.erased(), model))
    }
    fn matches_encoder(&self, model: &Arc<dyn EmbeddingEncoder>) -> bool {
        matches!(self.model(), Some(RouteModel::Encoder(current)) if Arc::ptr_eq(&current, model))
    }
    fn acquire(
        self: &Arc<Self>,
    ) -> std::result::Result<(RouteModel<A::Model>, ModelRead), ServiceError> {
        let mut state = self.state();
        if state.lifecycle != ModelLifecycle::Ready {
            return Err(missing_model());
        }
        let model = state.model.as_ref().ok_or_else(missing_model)?.clone();
        state.readers = state.readers.checked_add(1).ok_or_else(|| {
            ServiceError::new(503, "model_readers", "model request reader count exhausted")
        })?;
        Ok((
            model,
            ModelRead(self.clone(), RequestTiming::new(std::time::Instant::now())),
        ))
    }
    fn close(self: &Arc<Self>) -> Result<SelectiveRetirement> {
        // Only a concrete executor that can retire one independent owner
        // while other owners run may opt in.
        if !self.independent_retirement {
            return Err(Error::Unsupported(
                "executor has no independent model retirement".into(),
            ));
        }
        let mut state = self.state();
        if state.lifecycle != ModelLifecycle::Ready {
            return Err(Error::State("model is already retiring or retired".into()));
        }
        state.lifecycle = ModelLifecycle::Draining;
        Ok(SelectiveRetirement(self.clone()))
    }
}

trait RouteControl: Send + Sync {
    fn performance(&self) -> &RegistrationPerformance;
    fn release(&self);
    fn retire(&self) -> Result<()>;
    fn failed_retirement(&self);
}
impl<A: Executor> RouteControl for Route<A> {
    fn performance(&self) -> &RegistrationPerformance {
        &self.performance
    }
    fn release(&self) {
        let mut state = self.state();
        state.readers -= 1;
        if state.readers == 0 && state.lifecycle == ModelLifecycle::Draining {
            self.drained.notify_all();
        }
    }
    fn retire(&self) -> Result<()> {
        if !self.independent_retirement {
            return Err(Error::Unsupported(
                "executor has no independent model retirement".into(),
            ));
        }
        let mut state = self
            .drained
            .wait_while(self.state(), |state| state.readers != 0)
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.lifecycle != ModelLifecycle::Draining {
            return Err(Error::State(
                "model retirement requires draining admission".into(),
            ));
        }
        let model = state
            .model
            .take()
            .ok_or_else(|| Error::State("model already retired".into()))?;
        state.lifecycle = ModelLifecycle::Retiring;
        drop(state);
        // Actual adapter cleanup and final source destruction are unlocked.
        let result = match &model {
            RouteModel::Text(model) => self.retire.expect("text retirement admitted")(model)
                .map_err(ServiceError::into_core),
            // Synchronous stateless encoder; the last admitted call has
            // completed. Original source/scratch owners drop normally.
            RouteModel::Encoder(_) | RouteModel::Speech(_) => Ok(()),
        };
        drop(model);
        self.state().lifecycle = if result.is_ok() {
            ModelLifecycle::Retired
        } else {
            ModelLifecycle::Failed
        };
        result
    }
    fn failed_retirement(&self) {
        let mut state = self.state();
        if matches!(
            state.lifecycle,
            ModelLifecycle::Draining | ModelLifecycle::Retiring
        ) {
            state.lifecycle = ModelLifecycle::Failed;
        }
    }
}

/// Original registry reader, moved from admitted request to active row.
/// The last field of each owner, after its session/physical/decoder readers.
pub struct ModelRead(Arc<dyn RouteControl>, RequestTiming);
impl ModelRead {
    pub(super) fn observe_token(&mut self, committed: std::time::Instant) {
        self.0.performance().observe(&mut self.1, committed);
    }
    pub(crate) fn complete(&mut self) {
        self.0.performance().complete(&mut self.1);
    }
}
impl Drop for ModelRead {
    fn drop(&mut self) {
        self.0.release();
    }
}

pub(crate) struct SelectiveRetirement(Arc<dyn RouteControl>);
impl SelectiveRetirement {
    pub(crate) fn finish(self) -> Result<()> {
        self.0.retire()
    }
}
impl Drop for SelectiveRetirement {
    fn drop(&mut self) {
        // On unwind preserve a failed lifecycle observation, not Ready.
        self.0.failed_retirement();
    }
}

pub(crate) struct Registry<A: Executor> {
    routes: BTreeMap<String, Arc<Route<A>>>,
    has_finite_tasks: bool,
}
impl<A: Executor> Registry<A> {
    pub(super) fn new(
        models: BTreeMap<String, A::Model>,
        encoders: BTreeMap<String, Arc<dyn EmbeddingEncoder>>,
        speech: BTreeMap<String, Arc<dyn SpeechRecognizer>>,
    ) -> Result<Self> {
        let mut routes = BTreeMap::<String, Arc<Route<A>>>::new();
        for (alias, model) in models {
            let identity = model.erased();
            let route = match routes.values().find(|route| route.matches(&identity)) {
                Some(route) => route.clone(),
                None => Arc::new(Route::new(RouteModel::Text(model))?),
            };
            routes.insert(alias, route);
        }
        let has_finite_tasks = !encoders.is_empty() || !speech.is_empty();
        for (alias, model) in encoders {
            let route = match routes.values().find(|route| route.matches_encoder(&model)) {
                Some(route) => route.clone(),
                None => Arc::new(Route::new(RouteModel::Encoder(model))?),
            };
            routes.insert(alias, route);
        }
        for (alias, model) in speech {
            let route=match routes.values().find(|route|matches!(route.model(),Some(RouteModel::Speech(current)) if Arc::ptr_eq(&current,&model))){
                Some(route)=>route.clone(),
                None=>Arc::new(Route::new(RouteModel::Speech(model))?),
            };
            routes.insert(alias, route);
        }
        Ok(Self {
            routes,
            has_finite_tasks,
        })
    }

    pub(super) fn has_finite_tasks(&self) -> bool {
        self.has_finite_tasks
    }

    fn retire_route(&self, route: &Arc<Route<A>>) -> Result<Option<SelectiveRetirement>> {
        if self
            .routes
            .values()
            .any(|other| !Arc::ptr_eq(route, other) && other.accepting())
        {
            route.close().map(Some)
        } else {
            // Last owner uses the established whole-queue drain and join.
            Ok(None)
        }
    }

    pub(super) fn model_for_job(
        &self,
        alias: &str,
        reader: &ModelRead,
    ) -> std::result::Result<A::Model, ServiceError> {
        let route = self.routes.get(alias).ok_or_else(missing_model)?;
        if !std::ptr::addr_eq(route.as_ref() as &dyn RouteControl, reader.0.as_ref()) {
            return Err(ServiceError::new(
                500,
                "model_reader",
                "queued request belongs to another prepared owner",
            ));
        }
        // Closed admission does not revoke a previously accepted reader.
        match route.state().model.clone() {
            Some(RouteModel::Text(model)) => Ok(model),
            _ => Err(ServiceError::new(
                500,
                "model_reader",
                "accepted text reader lost its owner",
            )),
        }
    }

    // Bounded cold adapter views, not a second registry. Retired/draining
    // aliases are no longer exposed. Full shutdown waits for selective
    // retirement before closing the queue and invoking its original hooks.
    fn snapshot(&self) -> BTreeMap<String, A::Model> {
        self.routes
            .iter()
            .filter_map(|(alias, route)| match route.model() {
                Some(RouteModel::Text(model)) => Some((alias.clone(), model)),
                _ => None,
            })
            .collect()
    }
    pub(super) fn retire_all(&self) -> std::result::Result<(), ServiceError> {
        A::retire(&self.snapshot())
    }
}

fn missing_model() -> ServiceError {
    ServiceError::new(404, "model_not_found", "model is not registered")
}
pub(crate) trait Models: Send + Sync {
    fn model(&self, alias: &str) -> Option<Arc<dyn PreparedModel>>;
    fn registration(&self, alias: &str) -> Option<RegisteredOwner>;
    fn acquire(
        &self,
        alias: &str,
    ) -> std::result::Result<(Arc<dyn PreparedModel>, ModelRead), ServiceError>;
    fn acquire_encoder(
        &self,
        alias: &str,
        public: bool,
    ) -> std::result::Result<(Arc<dyn EmbeddingEncoder>, ModelRead), ServiceError>;
    fn acquire_speech(
        &self,
        alias: &str,
        public: bool,
    ) -> std::result::Result<(Arc<dyn SpeechRecognizer>, ModelRead), ServiceError>;
    fn begin_unload_registration(
        &self,
        alias: &str,
        id: u64,
    ) -> Result<Option<SelectiveRetirement>>;
    fn alias_slots(&self) -> usize;
    fn ready_counts(&self) -> (usize, usize);
    fn catalog_snapshot(&self, registrations: Option<&[u64]>) -> Result<Vec<ModelSnapshot>>;
    fn alias(&self, index: usize, public: bool) -> Option<&str>;
    fn begin_unload(&self, model: &ModelInstance) -> Result<Option<SelectiveRetirement>>;
    fn admission(
        &self,
        snapshot: &SchedulerSnapshot,
    ) -> std::result::Result<serde_json::Value, ServiceError>;
    fn invalidate_prefix_scope(
        &self,
        sharing: PrefixSharingScope,
    ) -> std::result::Result<PrefixCacheEviction, ServiceError>;
    fn retirement_pending(&self) -> bool;
}
impl<A: Executor> Models for Registry<A> {
    fn model(&self, alias: &str) -> Option<Arc<dyn PreparedModel>> {
        match self.routes.get(alias)?.model()? {
            RouteModel::Text(model) => Some(model.erased()),
            RouteModel::Encoder(_) | RouteModel::Speech(_) => None,
        }
    }
    fn registration(&self, alias: &str) -> Option<RegisteredOwner> {
        let route = self.routes.get(alias)?;
        route.accepting().then_some(route.registration)
    }

    fn acquire(
        &self,
        alias: &str,
    ) -> std::result::Result<(Arc<dyn PreparedModel>, ModelRead), ServiceError> {
        let (model, reader) = self
            .routes
            .get(alias)
            .ok_or_else(missing_model)?
            .acquire()?;
        let reader = reader;
        let model = model;
        match model {
            RouteModel::Text(model) => Ok((model.erased(), reader)),
            RouteModel::Encoder(_) | RouteModel::Speech(_) => Err(ServiceError::new(
                400,
                "unsupported_task",
                "model is not a text generator",
            )),
        }
    }
    fn acquire_encoder(
        &self,
        alias: &str,
        public: bool,
    ) -> std::result::Result<(Arc<dyn EmbeddingEncoder>, ModelRead), ServiceError> {
        let route = self.routes.get(alias).ok_or_else(missing_model)?;
        let (model, reader) = route.acquire()?;
        let reader = reader;
        let model = model;
        match model {
            RouteModel::Encoder(model) if !public || route.public_embeddings => Ok((model, reader)),
            RouteModel::Encoder(_) => Err(ServiceError::new(
                404,
                "not_found",
                "encoder is not qualified for public serving",
            )),
            RouteModel::Text(_) | RouteModel::Speech(_) => Err(ServiceError::new(
                400,
                "unsupported_task",
                "model is not an encoder",
            )),
        }
    }
    fn acquire_speech(
        &self,
        alias: &str,
        public: bool,
    ) -> std::result::Result<(Arc<dyn SpeechRecognizer>, ModelRead), ServiceError> {
        let route = self.routes.get(alias).ok_or_else(missing_model)?;
        let (model, reader) = route.acquire()?;
        let reader = reader;
        let model = model;
        match model {
            RouteModel::Speech(model) if !public || route.public_transcription => {
                Ok((model, reader))
            }
            RouteModel::Speech(_) => Err(ServiceError::new(
                404,
                "not_found",
                "speech recognizer is not qualified for public serving",
            )),
            _ => Err(ServiceError::new(
                400,
                "unsupported_task",
                "model is not a speech recognizer",
            )),
        }
    }
    fn begin_unload_registration(
        &self,
        alias: &str,
        id: u64,
    ) -> Result<Option<SelectiveRetirement>> {
        let route = self
            .routes
            .get(alias)
            .filter(|route| route.registration.id == id && route.accepting())
            .ok_or_else(|| Error::State("selected model registration is no longer ready".into()))?;
        self.retire_route(route)
    }
    fn alias_slots(&self) -> usize {
        self.routes.len()
    }
    fn ready_counts(&self) -> (usize, usize) {
        let mut aliases = 0;
        let mut owners = 0;
        for (index, route) in self.routes.values().enumerate() {
            if route.accepting() {
                aliases += 1;
                if !self
                    .routes
                    .values()
                    .take(index)
                    .any(|prior| Arc::ptr_eq(prior, route))
                {
                    owners += 1;
                }
            }
        }
        (aliases, owners)
    }
    fn catalog_snapshot(&self, registrations: Option<&[u64]>) -> Result<Vec<ModelSnapshot>> {
        let mut entries = Vec::new();
        for (index, route) in self.routes.values().enumerate() {
            if registrations.is_some_and(|ids| !ids.contains(&route.registration.id))
                || self
                    .routes
                    .values()
                    .take(index)
                    .any(|prior| Arc::ptr_eq(prior, route))
            {
                continue;
            }
            let (lifecycle, admitted_requests, model) = {
                let state = route.state();
                (
                    state.lifecycle,
                    state.readers,
                    if state.lifecycle == ModelLifecycle::Ready {
                        state.model.clone()
                    } else {
                        None
                    },
                )
            };
            let capabilities = model
                .map(|model| {
                    Ok::<_, Error>(match model {
                        RouteModel::Text(model) => {
                            let prepared = model.model();
                            ModelCapabilities {
                                max_context: prepared.max_context(),
                                vocabulary_size: prepared.vocabulary_size(),
                                max_token_bytes: Some(prepared.max_token_bytes()),
                                prefill: Some(A::prefill(&model)),
                                embedding_dimensions: None,
                                transcription: false,
                                chat_allowed: prepared.validate_chat().is_ok(),
                                details: A::describe_model(&model)?,
                            }
                        }
                        RouteModel::Encoder(model) => ModelCapabilities {
                            max_context: model.max_context(),
                            vocabulary_size: model.vocabulary_size(),
                            max_token_bytes: None,
                            prefill: None,
                            embedding_dimensions: Some(model.dimensions()),
                            transcription: false,
                            chat_allowed: false,
                            details: model.describe()?,
                        },
                        RouteModel::Speech(model) => ModelCapabilities {
                            max_context: model.max_context(),
                            vocabulary_size: model.vocabulary_size(),
                            max_token_bytes: None,
                            prefill: None,
                            embedding_dimensions: None,
                            transcription: true,
                            chat_allowed: false,
                            details: model.describe()?,
                        },
                    })
                })
                .transpose()?;
            entries.push(ModelSnapshot {
                registration_id: route.registration.id,
                source: route.registration.source,
                aliases: self
                    .routes
                    .iter()
                    .filter(|(_, other)| Arc::ptr_eq(other, route))
                    .map(|(alias, _)| alias.clone())
                    .collect(),
                lifecycle,
                admitted_requests,
                performance: route.performance.snapshot(std::time::Instant::now()),
                independent_retirement: route.independent_retirement,
                capabilities,
            });
        }
        Ok(entries)
    }
    fn alias(&self, index: usize, public: bool) -> Option<&str> {
        let (alias, route) = self.routes.iter().nth(index)?;
        let state = route.state();
        (state.lifecycle == ModelLifecycle::Ready
            && (matches!(state.model.as_ref(), Some(RouteModel::Text(_)))
                || (public && (route.public_embeddings || route.public_transcription))))
            .then_some(alias.as_str())
    }
    fn begin_unload(&self, model: &ModelInstance) -> Result<Option<SelectiveRetirement>> {
        let route = self
            .routes
            .values()
            .find(|route| route.matches(model.prepared()))
            .ok_or_else(|| Error::State("model is not the original registered owner".into()))?;
        self.retire_route(route)
    }
    fn admission(
        &self,
        snapshot: &SchedulerSnapshot,
    ) -> std::result::Result<serde_json::Value, ServiceError> {
        A::diagnostics(&self.snapshot(), snapshot)
    }
    fn invalidate_prefix_scope(
        &self,
        sharing: PrefixSharingScope,
    ) -> std::result::Result<PrefixCacheEviction, ServiceError> {
        A::invalidate_prefix_scope(&self.snapshot(), sharing)
    }
    fn retirement_pending(&self) -> bool {
        A::retirement_pending(&self.snapshot())
    }
}

/// Allocation-free cold enumeration over stable alias slots. A retired slot
/// is skipped; it cannot be reused within this cold route set.
pub struct ModelIds<'a> {
    pub(crate) models: &'a dyn Models,
    pub(crate) index: usize,
    pub(crate) public: bool,
}
impl<'a> Iterator for ModelIds<'a> {
    type Item = &'a str;
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.models.alias_slots() {
            let index = self.index;
            self.index += 1;
            if let Some(alias) = self.models.alias(index, self.public) {
                return Some(alias);
            }
        }
        None
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.models.alias_slots() - self.index))
    }
}
impl std::iter::FusedIterator for ModelIds<'_> {}

pub(crate) fn instance(models: &dyn Models, alias: &str) -> Option<ModelInstance> {
    models.model(alias).map(ModelInstance::new)
}
