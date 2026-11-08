//! Source-bound cold registration through the runtime's one engine/catalog.
#[cfg(all(feature = "qwen35-hip", target_os = "linux"))]
pub use crate::builtins::text::QwenHipProfile;
#[cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
pub use crate::builtins::text::QwenSourceProfile;
pub use crate::model::DeclaredModel;
pub use zerogravity_core::model::{ModelAssembly, ModelDefinition, ModelProgram, ProgramRequest};
pub use zerogravity_core::provider::{PreparedExecutable, RegionExecutionProvider};
pub use zerogravity_core::source::{SourceTensorRef, TensorSource};
pub use zerogravity_engine::model_instance::{ModelInstance, PreparedModel};
pub use zerogravity_engine::{
    executable::ExecutableBundle, prepare::prepare as prepare_executable,
};

/// One complete cold route set. This is not live model replacement or residency.
/// Aliases share the original source; an already configured runtime refuses.
pub enum ModelLoadRequest {
    /// Canonical original source; shared CUDA worker and bounded matrix placement.
    #[cfg(all(feature = "qwen35-cuda", target_os = "linux"))]
    QwenCuda {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenSourceProfile,
        /// Provider capacity, not total device memory.
        provider_bytes: usize,
        /// Physical device admission, including the backend's context owners.
        device_bytes: u64,
    },
    /// Canonical original HF source through the shared worker; cold HIP only.
    #[cfg(all(feature = "qwen35-hip", target_os = "linux"))]
    QwenHip {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenHipProfile,
    },
    /// Existing serving source profiles, including explicitly finite diagnostics.
    #[cfg(feature = "serve")]
    ServingText {
        declared: Vec<(String, std::path::PathBuf)>,
        profile: crate::serving::TextProfile,
    },
    CpuText {
        declared: Vec<(String, std::path::PathBuf)>,
        config: crate::runtime::TextConfig,
    },
    #[cfg(all(
        feature = "text",
        any(feature = "vulkan", feature = "cuda", feature = "hip")
    ))]
    GpuText {
        declared: Vec<(String, std::path::PathBuf)>,
        backend: crate::devices::GpuBackend,
        profile: crate::builtins::text::GpuProfile,
    },
}

pub struct ModelCatalog<'a> {
    runtime: &'a crate::Runtime,
}
impl<'a> ModelCatalog<'a> {
    pub(crate) fn new(runtime: &'a crate::Runtime) -> Self {
        Self { runtime }
    }
    pub fn list(&self) -> crate::Result<Vec<ModelInstance>> {
        self.runtime.engine.models().list()
    }
    /// Resolve once to the actual retained model, never a new source identity.
    pub fn resolve(&self, alias: &str) -> crate::Result<ModelInstance> {
        self.runtime.engine.models().resolve(alias)
    }
    /// Closed/loading/limit checks precede source IO; preparation and worker
    /// retirement run outside the lifecycle mutex, exactly as direct loading.
    pub fn load(
        &self,
        request: ModelLoadRequest,
    ) -> std::result::Result<(), crate::runtime::text::ServiceError> {
        self.runtime
            .engine
            .configure_text(|| crate::builtins::load_text(request, self.runtime.engine.limits()))
    }
    /// Individual model retirement remains future work; shutdown retires the
    /// configured route set only after closing admission and joining workers.
    pub fn unload(&self, model: &ModelInstance) -> crate::Result<()> {
        self.runtime.engine.models().unload(model)
    }
}
