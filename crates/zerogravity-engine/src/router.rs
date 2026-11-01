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

// Set once per worker, including capture destruction; never per token/step.
thread_local! {
    static ON_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) fn on_worker_thread() -> bool {
    ON_WORKER.with(std::cell::Cell::get)
}
struct WorkerScope(bool);
impl WorkerScope {
    fn enter() -> Self {
        Self(ON_WORKER.with(|active| active.replace(true)))
    }
}
impl Drop for WorkerScope {
    fn drop(&mut self) {
        ON_WORKER.with(|active| active.set(self.0));
    }
}
