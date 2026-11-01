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
    /// Load admitted CPU text sources into this runtime's one scheduler.
    /// Aliases of one canonical path share the original source/tokenizer owner.
    #[cfg(all(feature = "cpu", feature = "text", feature = "cache"))]
    pub fn load_text(
        &self,
        declared: Vec<(String, std::path::PathBuf)>,
        config: TextConfig,
    ) -> std::result::Result<(), text::ServiceError> {
        self.engine.configure_text(|| {
            let limits = self.engine.limits();
            if config
                .workers
                .checked_mul(config.active_per_worker)
                .is_none_or(|n| n > limits.max_active)
                || !limits.allows_pending(config.max_pending)
            {
                return Err(text::ServiceError::new(
                    400,
                    "invalid_config",
                    "text scheduler exceeds runtime limits",
                ));
            }
            crate::builtins::text::load_router(
                declared,
                config.workers,
                config.active_per_worker,
                config.max_context,
                config.max_output,
                config.host_mib,
                config.max_pending,
            )
        })
    }
    /// Direct bounded text events, with the original transport-compatible errors.
    pub fn request_text(
        &self,
        request: text::Request,
    ) -> std::result::Result<text::RequestEvents, text::ServiceError> {
        self.engine.request_text(request)
    }
    pub fn status(&self) -> RuntimeStatus {
        self.engine.status()
    }
    /// Close admission and join workers. Engine-worker and same-runtime loader/
    /// retirement callbacks return an error rather than synchronously self-joining.
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

pub use zerogravity_engine::model_instance::{
    ModelInstance, PreparedModel, TextSession, TextTokenizer,
};
pub use zerogravity_engine::router as text;
pub use zerogravity_engine::scheduler::SchedulerLimits;

/// Explicit bounded CPU scheduler/source admission configuration.
#[derive(Clone, Copy, Debug)]
pub struct TextConfig {
    pub workers: usize,
    pub active_per_worker: usize,
    pub max_context: usize,
    pub max_output: usize,
    pub host_mib: usize,
    pub max_pending: Option<usize>,
}
impl Default for TextConfig {
    fn default() -> Self {
        Self {
            workers: 1,
            active_per_worker: 1,
            max_context: 8192,
            max_output: 512,
            host_mib: 4096,
            max_pending: Some(16),
        }
    }
}
