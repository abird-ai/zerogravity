//! Bounded serial language composition over the original text runtime.
//! No pipeline worker, model reader, additional state publisher or registry.
use crate::{
    Error, Result, Runtime,
    model_instance::ModelInstance,
    output::{OutputStream, ResponseEvent, recv_event},
    router::{
        RegisteredOwner, RequestEvents, Router,
        pending::{ServingCharge, checked_sum, vec_bytes},
    },
    runtime::{service_error, submit_text},
};
use std::{borrow::Cow, num::NonZeroUsize, sync::Arc};
use zerogravity_core::{
    identity::SourceIdentity, pipeline::Config, resource_arbiter::AllocationClass,
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
    fn require(&self, router: &Router) -> Result<()> {
        if router.registration(&self.alias) != Some(self.registration) {
            return Err(Error::State(format!(
                "pipeline stage {} no longer has its original registered model",
                self.id
            )));
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
            let mut bytes = checked_sum(&[
                std::mem::size_of::<PreparedChain>() as u64 + 64,
                vec_bytes::<LanguageStage>(chain.len()).map_err(service_error)?,
                name.len() as u64,
            ])
            .map_err(service_error)?;
            for stage in &chain {
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
                bytes = checked_sum(&[bytes, stage.id.len() as u64, binding.alias.len() as u64])
                    .map_err(service_error)?;
            }
            let charge = router
                .host()
                .charge(name, "pipeline-plan", AllocationClass::Scratch, bytes)
                .map_err(service_error)?;
            let mut stages = Vec::with_capacity(chain.len());
            for stage in chain {
                let binding = binding_for(stage.model.as_deref())?;
                let registration = router
                    .registration(binding.alias)
                    .ok_or_else(|| Error::State("pipeline model admission closed".into()))?;
                stages.push(LanguageStage {
                    id: stage.id.clone(),
                    alias: binding.alias.to_owned(),
                    registration,
                });
            }
            Ok(PreparedLanguagePipeline(Arc::new(PreparedChain {
                name: name.to_owned(),
                stages,
                _charge: charge,
            })))
        })
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
        self.validate_pipeline_input(input)?;
        self.with_ready_text(|router| {
            let stages = plan.stages();
            for stage in stages {
                stage.require(router)?;
            }
            if max_output.get() > router.max_output() {
                return Err(Error::Limit(
                    "pipeline output exceeds configured text limit".into(),
                ));
            }
            let charge = router
                .host()
                .charge(
                    plan.name(),
                    "pipeline-stream",
                    AllocationClass::Output,
                    std::mem::size_of::<LanguageStream>() as u64 + 64,
                )
                .map_err(service_error)?;
            let active = submit_text(
                router,
                stages[0].alias(),
                Cow::Borrowed(input),
                max_output.get(),
                stages.len() == 1,
            )?;
            Ok(Box::new(LanguageStream {
                active: Some(active),
                plan: plan.clone(),
                runtime: self.clone(),
                stage: 0,
                finished: false,
                max_output,
                _charge: charge,
            }) as Box<dyn OutputStream>)
        })
    }

    fn validate_pipeline_input(&self, input: &str) -> Result<()> {
        self.require_open()?;
        if input.is_empty() {
            return Err(Error::Malformed("pipeline stage input is empty".into()));
        }
        if input.len() > self.limits().max_input_bytes {
            return Err(Error::Limit(
                "pipeline stage input exceeds runtime limit".into(),
            ));
        }
        Ok(())
    }
}

struct LanguageStream {
    // Drop the original request/cancellation and transport before the
    // runtime handle and stream/plan credits.
    active: Option<RequestEvents>,
    plan: PreparedLanguagePipeline,
    runtime: Runtime,
    stage: usize,
    finished: bool,
    max_output: NonZeroUsize,
    _charge: ServingCharge,
}
impl LanguageStream {
    fn advance(&mut self, output: String) -> Result<()> {
        self.runtime.validate_pipeline_input(&output)?;
        let next = self.stage + 1;
        let stages = self.plan.stages();
        let active = self.runtime.with_ready_text(|router| {
            stages[next].require(router)?;
            // The predecessor's original event transport charge remains
            // live until the successor admits the moved String's capacity.
            submit_text(
                router,
                stages[next].alias(),
                Cow::Owned(output),
                self.max_output.get(),
                next + 1 == stages.len(),
            )
        })?;
        self.active = Some(active);
        self.stage = next;
        Ok(())
    }
}
impl OutputStream for LanguageStream {
    fn next(&mut self) -> Result<Option<ResponseEvent>> {
        if self.finished {
            return Ok(None);
        }
        while let Some(active) = self.active.as_ref() {
            let event = match recv_event(active) {
                Ok(event) => event,
                Err(error) => {
                    self.active.take();
                    return Err(error);
                }
            };
            match event {
                ResponseEvent::Done { output, .. } if self.stage + 1 < self.plan.stages().len() => {
                    if let Err(error) = self.advance(output) {
                        self.finished = true;
                        return Err(error);
                    }
                }
                event => {
                    // Intermediate stages request only one terminal. Never
                    // expose their deltas as the final stage's output.
                    if matches!(event, ResponseEvent::Done { .. } | ResponseEvent::Error(_)) {
                        // As with TextOutputStream, retain the terminal event
                        // transport through the consumer stream lifetime.
                        self.finished = true;
                    } else if self.stage + 1 < self.plan.stages().len() {
                        self.active.take();
                        return Err(Error::State(
                            "non-streaming pipeline stage emitted a delta".into(),
                        ));
                    }
                    return Ok(Some(event));
                }
            }
        }
        Ok(None)
    }
    fn cancel(&mut self) -> Result<()> {
        if let Some(active) = &self.active {
            active.cancel();
        }
        self.finished = true;
        Ok(())
    }
}
