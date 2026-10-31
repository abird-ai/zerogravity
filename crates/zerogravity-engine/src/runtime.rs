use crate::{
    Error, Result, catalog::ModelCatalog, diagnostics::RuntimeStatus, output::OutputStream,
    request::TaskRequest, scheduler::SchedulerLimits,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_core::physical::ResourceArbiter;

/// Clones share lifecycle and the process-wide physical ledger.
/// An open shell has no loaded models or migrated execution adapters.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<Inner>,
}
struct Inner {
    closed: AtomicBool,
    resources: ResourceArbiter,
    limits: SchedulerLimits,
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
            }),
        })
    }
}
impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }
    pub fn status(&self) -> RuntimeStatus {
        RuntimeStatus {
            closed: self.inner.closed.load(Ordering::Acquire),
            loaded_models: 0,
            execution_adapters: 0,
        }
    }
    pub fn resources(&self) -> &ResourceArbiter {
        &self.inner.resources
    }
    pub fn models(&self) -> ModelCatalog<'_> {
        ModelCatalog::new(self)
    }
    pub fn run(&self, request: &TaskRequest) -> Result<Box<dyn OutputStream>> {
        self.require_open()?;
        request.validate(self.inner.limits.max_input_bytes)?;
        Err(Error::Unsupported(
            "Fabric task execution awaits F0 adapters; existing source-specific APIs unchanged"
                .into(),
        ))
    }
    /// Closes only this empty shell, not other runtimes or physical namespaces.
    pub fn shutdown(&self) -> Result<()> {
        self.inner.closed.store(true, Ordering::Release);
        Ok(())
    }
    pub fn require_open(&self) -> Result<()> {
        if self.inner.closed.load(Ordering::Acquire) {
            Err(Error::State("Fabric runtime is shut down".into()))
        } else {
            Ok(())
        }
    }
}
