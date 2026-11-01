//! Prepared-source routing and the original bounded scheduler/presentation owners.
mod cpu;
pub mod event_channel;
pub mod pending;
pub mod token_metrics;
pub use cpu::*;

use crate::{Result, Runtime, model_instance::ModelInstance};

pub fn resolve(runtime: &Runtime, alias: &str) -> Result<ModelInstance> {
    runtime.require_open()?;
    zerogravity_core::run_policy::SourceSelector::parse(alias)?;
    runtime.resolve_text(alias)
}
