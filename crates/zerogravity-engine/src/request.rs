//! Requests carry intent, never an authenticated model/execution permit.
use crate::{Error, Result};
use std::num::NonZeroUsize;
use zerogravity_core::{run_policy::SourceSelector, task::TaskKind};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaskInput {
    Text(String),
    /// Future typed media/encoder stage. No payload access or decoding in S0.
    Unavailable(TaskKind),
}
#[derive(Clone, Debug)]
pub struct TaskRequest {
    pub model: String,
    pub input: TaskInput,
    pub max_output: NonZeroUsize,
}
impl TaskRequest {
    pub fn validate(&self, max_input_bytes: usize) -> Result<()> {
        SourceSelector::parse(&self.model)?;
        match &self.input {
            TaskInput::Text(text) if text.len() > max_input_bytes => {
                Err(Error::Limit("Fabric request input bytes".into()))
            }
            TaskInput::Unavailable(task) => Err(Error::Unsupported(format!(
                "Fabric task driver {task:?} unavailable"
            ))),
            _ => Ok(()),
        }
    }
}

/// An already loaded alias or an exact prepared serial language pipeline.
/// Neither form authorizes source acquisition or backend fallback.
#[derive(Clone, Copy)]
pub enum TaskTarget<'a> {
    Model(&'a str),
    Pipeline(&'a crate::pipeline::PreparedLanguagePipeline),
}

/// Per-request controls, not another model/scheduler configuration. A pipeline
/// applies the token ceiling to each stage, with presentation/observations
/// selected only for its final stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct TextControls {
    pub max_output: NonZeroUsize,
    pub chat: bool,
    pub stream: bool,
    pub observe_tokens: bool,
}
impl TextControls {
    pub const fn new(max_output: NonZeroUsize) -> Self {
        Self {
            max_output,
            chat: false,
            stream: true,
            observe_tokens: false,
        }
    }
}

/// Borrowed SDK text intent. Admission owns any eventual input copy.
#[derive(Clone, Copy)]
pub struct TextTask<'a> {
    pub target: TaskTarget<'a>,
    pub input: &'a str,
    pub controls: TextControls,
}
impl TextTask<'_> {
    pub(crate) fn input_limit(max_input_bytes: usize) -> usize {
        max_input_bytes.min(crate::router::MAX_INPUT_BYTES)
    }

    pub(crate) fn validate(
        self,
        max_input_bytes: usize,
    ) -> std::result::Result<(), crate::router::ServiceError> {
        use crate::router::ServiceError;
        if self.input.is_empty() {
            return Err(ServiceError::new(
                400,
                "invalid_request",
                "empty task input",
            ));
        }
        if self.input.len() > Self::input_limit(max_input_bytes) {
            return Err(ServiceError::new(
                400,
                "limit_exceeded",
                "task input exceeds runtime limit",
            ));
        }
        match self.target {
            TaskTarget::Model(alias) => crate::router::validate_alias(alias),
            TaskTarget::Pipeline(_) if self.controls.chat => Err(ServiceError::new(
                400,
                "unsupported",
                "serial language pipelines accept raw text only",
            )),
            TaskTarget::Pipeline(_) => Ok(()),
        }
    }

    /// The same selected registrations and output bounds constrain cold
    /// observation and actual submission, under the original router lock.
    pub(crate) fn require(
        self,
        router: &crate::router::Router,
    ) -> std::result::Result<(), crate::router::ServiceError> {
        use crate::router::ServiceError;
        if self.controls.max_output.get() > router.max_output() {
            return Err(ServiceError::new(
                400,
                "limit_exceeded",
                "task output exceeds configured text limit",
            ));
        }
        match self.target {
            TaskTarget::Model(alias) => {
                if router.registration(alias).is_none() {
                    return Err(ServiceError::new(
                        404,
                        "model_not_found",
                        "model is not registered",
                    ));
                }
                if self.controls.chat {
                    router.validate_chat(alias)?;
                }
            }
            TaskTarget::Pipeline(plan) => {
                for stage in plan.stages() {
                    stage.require(router)?;
                }
            }
        }
        Ok(())
    }
}
