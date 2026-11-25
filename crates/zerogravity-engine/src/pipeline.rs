//! Bounded serial language composition over the original text runtime.
//! No pipeline worker, model reader, additional state publisher or registry.
use crate::{
    Error, Result, Runtime,
    model_instance::ModelInstance,
    output::{OutputPoll, OutputStream, ResponseEvent, next_blocking, poll_event},
    request::{TaskTarget, TextControls, TextTask},
    router::{
        RegisteredOwner, RequestEvents, Router, ServiceError,
        pending::{ServingCharge, checked_sum, vec_bytes},
    },
    runtime::{service_error, submit_text, task_error},
};
use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};
use zerogravity_core::{
    identity::SourceIdentity,
    pipeline::{Config, ModelInstance as DeclaredModel, PipelineNode},
    resource_arbiter::AllocationClass,
};

/// Explicitly associate a declared model ID with an exact loaded handle and
/// its registered alias. Target metadata never loads or substitutes a model.
pub struct LanguageBinding<'a> {
    pub model: &'a str,
    pub alias: &'a str,
    pub owner: &'a ModelInstance,
}

#[derive(Clone, Debug)]
pub struct LanguageStage {
    id: String,
    alias: String,
    registration: RegisteredOwner,
}
impl LanguageStage {
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn alias(&self) -> &str {
        &self.alias
    }
    pub fn source(&self) -> SourceIdentity {
        self.registration.source
    }
    pub fn registration_id(&self) -> u64 {
        self.registration.id
    }
    pub(crate) fn require(&self, router: &Router) -> std::result::Result<(), ServiceError> {
        if router.registration(&self.alias) != Some(self.registration) {
            return Err(ServiceError::new(
                409,
                "stale_pipeline",
                format!(
                    "pipeline stage {} no longer has its original registered model",
                    self.id
                ),
            ));
        }
        Ok(())
    }
}

/// A reusable immutable plan, not a retained model or runtime reader. A new
/// registration must be bound again, even if it has the same source identity.
#[derive(Clone)]
pub struct PreparedLanguagePipeline(Arc<PreparedChain>);
struct PreparedChain {
    name: String,
    stages: Vec<LanguageStage>,
    _charge: ServingCharge,
}
impl PreparedLanguagePipeline {
    pub fn name(&self) -> &str {
        &self.0.name
    }
    pub fn stages(&self) -> &[LanguageStage] {
        &self.0.stages
    }
}

impl Runtime {
    /// Validate the existing graph and bind every selected stage before any
    /// request. Each used declaration requires one explicit loaded binding.
    pub fn prepare_pipeline(
        &self,
        config: &Config,
        name: &str,
        bindings: &[LanguageBinding<'_>],
    ) -> Result<PreparedLanguagePipeline> {
        self.require_open()?;
        let chain = config.language_chain(name)?;
        // Bounded by existing graph/alias limits. Reject ambiguity before
        // allocating the retained plan or observing any model.
        if bindings.len() > chain.len()
            || bindings.iter().enumerate().any(|(i, binding)| {
                bindings[..i]
                    .iter()
                    .any(|prior| prior.model == binding.model)
                    || !chain
                        .iter()
                        .any(|stage| stage.model.as_deref() == Some(binding.model))
            })
        {
            return Err(Error::Malformed(
                "duplicate or unused pipeline model binding".into(),
            ));
        }
        let binding_for = |model: Option<&str>| {
            bindings
                .iter()
                .find(|binding| Some(binding.model) == model)
                .ok_or_else(|| {
                    Error::Malformed("pipeline model requires an explicit loaded binding".into())
                })
        };
        self.with_ready_text(|router| {
            let mut stages = Vec::with_capacity(chain.len());
            for stage in chain {
                let binding = binding_for(stage.model.as_deref())?;
                if !router
                    .model(binding.alias)
                    .is_some_and(|registered| registered.shares_source(binding.owner))
                {
                    return Err(Error::State(format!(
                        "pipeline stage {} requires its exact registered owner",
                        stage.id
                    )));
                }
                stages.push((stage, binding.alias));
            }
            bind_chain(router, name, &stages).map_err(service_error)
        })
    }

    /// Cold source-declared language graph. The trusted embedding loader
    /// resolves the selected declarations through its actual source/provider
    /// owners, registering exactly their IDs as aliases in one private router.
    /// No source authorization is inferred from the graph. The original
    /// runtime load slot atomically publishes the router and admitted plan.
    pub fn configure_language_pipeline(
        &self,
        config: &Config,
        name: &str,
        load: impl FnOnce(&[&DeclaredModel]) -> std::result::Result<Router, ServiceError>,
    ) -> std::result::Result<PreparedLanguagePipeline, ServiceError> {
        self.require_open()
            .map_err(|_| ServiceError::new(503, "runtime_closed", "Fabric runtime is shut down"))?;
        let chain = config
            .language_chain_structure(name)
            .map_err(ServiceError::inference)?;
        let mut models: Vec<&DeclaredModel> = Vec::new();
        let mut stages = Vec::with_capacity(chain.len());
        for stage in chain {
            let model = config
                .models
                .iter()
                .find(|m| Some(m.id.as_str()) == stage.model.as_deref())
                .ok_or_else(|| {
                    ServiceError::new(400, "invalid_model", "pipeline declaration absent")
                })?;
            if !models.iter().any(|prior| prior.id == model.id) {
                models.push(model);
            }
            stages.push((stage, model.id.as_str()));
        }
        crate::router::validate_aliases(models.iter().map(|m| m.id.as_str()))?;
        self.configure_text_with(
            || load(&models),
            |router| {
                if router.model_ids().count() != models.len()
                    || models.iter().any(|m| router.registration(&m.id).is_none())
                {
                    return Err(ServiceError::new(
                        400,
                        "pipeline_binding",
                        "loaded aliases differ from selected pipeline declarations",
                    ));
                }
                bind_chain(router, name, &stages)
            },
        )
    }

    /// Each stage receives the same finite output-token ceiling. Only the
    /// final stage's original stream and usage are returned. Intermediate
    /// stages publish their own committed text and hand it to the next request.
    /// Future stages are not pre-admitted; unloading one can stop the chain
    /// at the next boundary, never substitute another registration.
    pub fn run_pipeline(
        &self,
        plan: &PreparedLanguagePipeline,
        input: &str,
        max_output: NonZeroUsize,
    ) -> Result<Box<dyn OutputStream>> {
        self.submit_task(TextTask {
            target: TaskTarget::Pipeline(plan),
            input,
            controls: TextControls::new(max_output),
        })
        .map_err(task_error)
    }
}

fn stage_controls(controls: TextControls, final_stage: bool) -> TextControls {
    TextControls {
        stream: final_stage && controls.stream,
        observe_tokens: final_stage && controls.observe_tokens,
        ..controls
    }
}

/// Called with the original ready router lock held, after exact selection
/// and controls have been checked. No second task queue.
pub(crate) fn submit_pipeline(
    runtime: &Runtime,
    router: &Router,
    plan: &PreparedLanguagePipeline,
    input: &str,
    controls: TextControls,
) -> std::result::Result<Box<dyn OutputStream>, ServiceError> {
    let charge = router.host().charge(
        plan.name(),
        "pipeline-stream",
        AllocationClass::Output,
        std::mem::size_of::<LanguageStream>() as u64 + 64,
    )?;
    let stages = plan.stages();
    let active = submit_text(
        router,
        stages[0].alias(),
        Cow::Borrowed(input),
        stage_controls(controls, stages.len() == 1),
    )?;
    Ok(Box::new(LanguageStream {
        active: Some(active),
        plan: plan.clone(),
        runtime: runtime.clone(),
        stage: 0,
        finished: false,
        controls,
        _charge: charge,
    }))
}

fn bind_chain(
    router: &Router,
    name: &str,
    selected: &[(&PipelineNode, &str)],
) -> std::result::Result<PreparedLanguagePipeline, ServiceError> {
    let mut bytes = checked_sum(&[
        std::mem::size_of::<PreparedChain>() as u64 + 64,
        vec_bytes::<LanguageStage>(selected.len())?,
        name.len() as u64,
    ])?;
    for (stage, alias) in selected {
        if router.registration(alias).is_none() {
            return Err(ServiceError::new(
                503,
                "model_retired",
                "pipeline model admission closed",
            ));
        }
        bytes = checked_sum(&[bytes, stage.id.len() as u64, alias.len() as u64])?;
    }
    let charge = router
        .host()
        .charge(name, "pipeline-plan", AllocationClass::Scratch, bytes)?;
    let mut stages = Vec::with_capacity(selected.len());
    for (stage, alias) in selected {
        let registration = router.registration(alias).ok_or_else(|| {
            ServiceError::new(503, "model_retired", "pipeline model admission closed")
        })?;
        stages.push(LanguageStage {
            id: stage.id.clone(),
            alias: (*alias).to_owned(),
            registration,
        });
    }
    Ok(PreparedLanguagePipeline(Arc::new(PreparedChain {
        name: name.to_owned(),
        stages,
        _charge: charge,
    })))
}

struct LanguageStream {
    // Drop the original request/cancellation and transport before the
    // runtime handle and stream/plan credits.
    active: Option<RequestEvents>,
    plan: PreparedLanguagePipeline,
    runtime: Runtime,
    stage: usize,
    finished: bool,
    controls: TextControls,
    _charge: ServingCharge,
}
impl LanguageStream {
    fn advance(&mut self, output: String) -> std::result::Result<(), ServiceError> {
        // Check only the next original stage; already accepted work and
        // unrelated later stages are not revalidated/substituted here.
        let next = self.stage + 1;
        let stages = self.plan.stages();
        TextTask {
            target: TaskTarget::Model(stages[next].alias()),
            input: &output,
            controls: self.controls,
        }
        .validate(self.runtime.limits().max_input_bytes)?;
        let active = self.runtime.with_text_router(|router| {
            stages[next].require(router)?;
            // Predecessor's original transport stays live until the
            // successor admits the moved String's capacity.
            submit_text(
                router,
                stages[next].alias(),
                Cow::Owned(output),
                stage_controls(self.controls, next + 1 == stages.len()),
            )
        })?;
        self.active = Some(active);
        self.stage = next;
        Ok(())
    }
}
impl OutputStream for LanguageStream {
    fn next(&mut self) -> Result<Option<ResponseEvent>> {
        next_blocking(self)
    }
    fn poll_next(
        &mut self,
        timeout: std::time::Duration,
    ) -> std::result::Result<OutputPoll, ServiceError> {
        if self.finished {
            return Ok(OutputPoll::Finished);
        }
        let Some(active) = self.active.as_ref() else {
            self.finished = true;
            return Ok(OutputPoll::Finished);
        };
        let event = match poll_event(active, timeout) {
            Ok(OutputPoll::Event(event)) => event,
            Ok(other) => return Ok(other),
            Err(error) => {
                self.active.take();
                self.finished = true;
                return Err(error);
            }
        };
        let intermediate = self.stage + 1 < self.plan.stages().len();
        match event {
            ResponseEvent::Done { output, .. } if intermediate => {
                if let Err(error) = self.advance(output) {
                    self.finished = true;
                    return Err(error);
                }
                Ok(OutputPoll::Pending)
            }
            event => {
                if matches!(event, ResponseEvent::Done { .. } | ResponseEvent::Error(_)) {
                    // Keep terminal transport through consumer stream lifetime.
                    self.finished = true;
                } else if intermediate {
                    self.active.take();
                    self.finished = true;
                    return Err(ServiceError::new(
                        500,
                        "pipeline_output",
                        "non-streaming pipeline stage emitted a delta",
                    ));
                }
                Ok(OutputPoll::Event(event))
            }
        }
    }
    fn cancel(&mut self) -> Result<()> {
        if !self.finished {
            if let Some(active) = &self.active {
                active.cancel();
            }
            self.finished = true;
        }
        Ok(())
    }
}
