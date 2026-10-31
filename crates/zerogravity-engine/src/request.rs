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
