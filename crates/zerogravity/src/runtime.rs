//! Public composition root. Cold builders perform no model or device IO.
use crate::{
    Result,
    artifacts::ArtifactService,
    devices::Devices,
    diagnostics::RuntimeStatus,
    models::ModelCatalog,
    run::{OutputStream, TaskRequest},
};

#[derive(Clone)]
pub struct Runtime {
    engine: zerogravity_engine::Runtime,
}
#[derive(Default)]
pub struct RuntimeBuilder {
    engine: zerogravity_engine::RuntimeBuilder,
}
impl RuntimeBuilder {
    pub fn limits(mut self, limits: zerogravity_engine::scheduler::SchedulerLimits) -> Self {
        self.engine = self.engine.limits(limits);
        self
    }
    pub fn build(self) -> Result<Runtime> {
        Ok(Runtime {
            engine: self.engine.build()?,
        })
    }
}
impl Runtime {
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }
    pub fn models(&self) -> ModelCatalog<'_> {
        self.engine.models()
    }
    pub fn artifacts(&self) -> ArtifactService<'_> {
        ArtifactService::new(&self.engine)
    }
    pub fn devices(&self) -> Devices<'_> {
        Devices::new(&self.engine)
    }
    pub fn run(&self, request: &TaskRequest) -> Result<Box<dyn OutputStream>> {
        self.engine.run(request)
    }
    pub fn status(&self) -> RuntimeStatus {
        self.engine.status()
    }
    pub fn shutdown(&self) -> Result<()> {
        self.engine.shutdown()
    }
    #[cfg(feature = "serve")]
    pub fn server(&self) -> crate::serving::ServerBuilder {
        crate::serving::ServerBuilder::new(self.engine.clone())
    }
}
/// Transfer the same engine handle into the independently reusable server.
/// No second runtime, model registry or resource owner is constructed.
impl From<Runtime> for zerogravity_engine::Runtime {
    fn from(runtime: Runtime) -> Self {
        runtime.engine
    }
}

pub use zerogravity_engine::scheduler::SchedulerLimits;
