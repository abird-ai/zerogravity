//! Public composition root. Cold builders perform no model or device IO.
use crate::{
    Result,
    artifacts::ArtifactService,
    devices::Devices,
    diagnostics::{RuntimeSnapshot, RuntimeStatus},
    models::ModelCatalog,
    run::{OutputStream, TaskRequest},
};

#[derive(Clone)]
pub struct Runtime {
    pub(crate) engine: zerogravity_engine::Runtime,
    control: std::sync::Arc<std::sync::OnceLock<crate::control::ControlState>>,
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
            control: Default::default(),
        })
    }
}
impl Runtime {
    /// Every facade clone observes the same control state. Initialization is
    /// cold: no listener, worker, source, cache or device is opened.
    pub fn control(&self) -> crate::control::ControlPlane<'_> {
        crate::control::ControlPlane::new(self, self.control.get_or_init(Default::default))
    }
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }
    pub fn models(&self) -> ModelCatalog<'_> {
        ModelCatalog::new(self)
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
    /// Unified borrowed model or prepared-pipeline text request.
    /// Retains the original structured service refusal and output events.
    pub fn submit_task(
        &self,
        task: crate::run::TextTask<'_>,
    ) -> std::result::Result<Box<dyn OutputStream>, text::ServiceError> {
        self.engine.submit_task(task)
    }
    /// On-demand selected owners/controls; no model loading or task admission.
    pub fn inspect_task(
        &self,
        task: crate::run::TextTask<'_>,
    ) -> std::result::Result<crate::diagnostics::TaskObservation, text::ServiceError> {
        self.engine.inspect_task(task)
    }
    /// Bind a structurally checked serial language graph to exact loaded models.
    pub fn prepare_pipeline(
        &self,
        config: &crate::pipelines::Config,
        name: &str,
        bindings: &[crate::pipelines::LanguageBinding<'_>],
    ) -> Result<crate::pipelines::PreparedLanguagePipeline> {
        self.engine.prepare_pipeline(config, name, bindings)
    }
    /// Cold-load only the selected serial language chain's declared sources,
    /// with explicit execution policy. Empty runtime only. All exact aliases
    /// and the admitted plan publish together, or the private router retires.
    pub fn load_pipeline(
        &self,
        config: &crate::pipelines::Config,
        name: &str,
        policy: crate::pipelines::LanguageLoadPolicy,
    ) -> std::result::Result<crate::pipelines::PreparedLanguagePipeline, text::ServiceError> {
        self.load_pipeline_with_cache(config, name, policy, None)
    }

    /// Same selected chain, with an optional explicit existing GGUF cache
    /// root. Original-HF policies carry their own CacheOptions.
    pub fn load_pipeline_with_cache(
        &self,
        config: &crate::pipelines::Config,
        name: &str,
        policy: crate::pipelines::LanguageLoadPolicy,
        cache_root: Option<&std::path::Path>,
    ) -> std::result::Result<crate::pipelines::PreparedLanguagePipeline, text::ServiceError> {
        crate::pipelines::load_language(&self.engine, config, name, policy, cache_root)
    }

    /// Execute through the original scheduler, with a finite per-stage limit.
    pub fn run_pipeline(
        &self,
        plan: &crate::pipelines::PreparedLanguagePipeline,
        input: &str,
        max_output: std::num::NonZeroUsize,
    ) -> Result<Box<dyn OutputStream>> {
        self.engine.run_pipeline(plan, input, max_output)
    }
    /// Load admitted CPU text sources into this runtime's one scheduler.
    /// Aliases of one canonical path share the original source/tokenizer owner.
    #[cfg(all(feature = "cpu", feature = "text", feature = "cache"))]
    pub fn load_text(
        &self,
        declared: Vec<(String, std::path::PathBuf)>,
        config: TextConfig,
    ) -> std::result::Result<(), text::ServiceError> {
        self.models()
            .load(crate::models::ModelLoadRequest::CpuText { declared, config })
    }

    /// Ordinary GGUF device text on the same engine Router/worker. Captured
    /// finite routes remain explicit diagnostics, not ordinary SDK capability.
    #[cfg(all(
        feature = "text",
        any(feature = "vulkan", feature = "cuda", feature = "hip")
    ))]
    pub fn load_gpu_text(
        &self,
        declared: Vec<(String, std::path::PathBuf)>,
        backend: crate::devices::GpuBackend,
        profile: crate::builtins::text::GpuProfile,
    ) -> std::result::Result<(), text::ServiceError> {
        self.models()
            .load(crate::models::ModelLoadRequest::GpuText {
                declared,
                backend,
                profile,
            })
    }

    /// Direct bounded text events, with the original transport-compatible errors.
    pub fn request_text(
        &self,
        request: text::Request,
    ) -> std::result::Result<text::RequestEvents, text::ServiceError> {
        self.engine.request_text(request)
    }
    pub fn request_text_scoped(
        &self,
        request: text::Request,
        sharing: PrefixSharingScope,
    ) -> std::result::Result<text::RequestEvents, text::ServiceError> {
        self.engine.request_text_scoped(request, sharing)
    }
    pub fn invalidate_prefix_scope(
        &self,
        sharing: PrefixSharingScope,
    ) -> std::result::Result<PrefixCacheEviction, text::ServiceError> {
        self.engine.invalidate_prefix_scope(sharing)
    }
    pub fn status(&self) -> RuntimeStatus {
        self.engine.status()
    }
    /// On-demand effective runtime, model and process resource facts.
    pub fn snapshot(&self) -> Result<RuntimeSnapshot> {
        self.engine.snapshot()
    }
    pub fn scheduler_snapshot(&self) -> Result<Option<text::worker::SchedulerSnapshot>> {
        self.engine.scheduler_snapshot()
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
    LogitModel, ModelInstance, PreparedModel, TextSession, TextTokenizer,
};
pub use zerogravity_engine::router as text;
pub use zerogravity_engine::scheduler::SchedulerLimits;
pub use zerogravity_engine::state_cache::{PrefixCacheEviction, PrefixSharingScope};

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
