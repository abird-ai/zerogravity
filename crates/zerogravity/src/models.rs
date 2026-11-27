//! Source-bound cold registration through the runtime's one engine/catalog.
#[cfg(all(
    any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    ),
    target_os = "linux"
))]
pub use crate::builtins::text::QwenServingProfile;
#[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
pub use crate::gpu::HipSourceBudget;

pub use crate::smart_config::{SmartConfig, SmartConfigMode};

#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QwenCudaSmartOverrides {
    pub state_policy: Option<QwenCudaStatePolicy>,
    /// `Some(0|4|8)` is an explicit user value. `None` leaves cohort width
    /// eligible for smart selection when the layer is Enabled/Exclusive.
    pub cohort_width: Option<usize>,
}

#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QwenCudaStatePolicy {
    #[default]
    Contiguous,
    Paged {
        page_tokens: usize,
        max_append_tokens: usize,
    },
}
#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
impl From<QwenCudaStatePolicy> for crate::gpu::CudaSourceStatePolicy {
    fn from(policy: QwenCudaStatePolicy) -> Self {
        match policy {
            QwenCudaStatePolicy::Contiguous => Self::Contiguous,
            QwenCudaStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            } => Self::Paged {
                page_tokens,
                max_append_tokens,
            },
        }
    }
}
#[cfg(feature = "model-qwen35")]
#[path = "qwen35_source/source_inventory.rs"]
pub(crate) mod qwen_source_inventory;

/// Approved exact HF source metadata, independent of any compiled backend.
/// This neither acquires the source nor qualifies any execution.
#[cfg(feature = "model-qwen35")]
pub fn qwen_source_snapshot(target: &str) -> Option<&'static str> {
    qwen_source_inventory::snapshot_for_target(target)
}

/// Original source identity for the finite approved Qwen inventory. A match
/// identifies the source only; actual model and device admission remains later.
#[cfg(feature = "model-qwen35")]
pub fn qwen_target_for_snapshot(snapshot: &str) -> Option<&'static str> {
    qwen_source_inventory::for_snapshot(snapshot).map(|spec| spec.target)
}

/// Only an already approved immutable repository/revision is a catalog
/// default. Floating or unregistered locators never become acquisition recipes.
#[cfg(all(feature = "model-qwen35", feature = "cache", target_os = "linux"))]
pub fn qwen_target_for_locator(locator: &str) -> crate::Result<Option<&'static str>> {
    let locator = crate::cache::provenance::ProviderLocator::parse("huggingface", locator)?;
    Ok(locator
        .revision
        .as_deref()
        .and_then(|revision| qwen_source_inventory::for_locator(&locator.repository, revision))
        .map(|spec| spec.target))
}

/// Common cache-name resolution for model management and execution. Existing
/// explicit claims take precedence over a built-in default, never over an
/// explicit snapshot pin. A known catalog default is only an exact approved
/// snapshot, not an arbitrary repo/quantization guess.
#[cfg(all(feature = "cache", target_os = "linux"))]
pub fn select_cached(
    metadata: &crate::cache::CacheMetadata,
    selector: &crate::run_policy::SourceSelector,
    pin: Option<&str>,
) -> crate::Result<crate::cache::SnapshotMetadata> {
    use crate::{Error, run_policy::SourceSelector};
    if let SourceSelector::Name(name) = selector {
        return find_named_cached(metadata, name, pin)?.ok_or_else(|| {
            Error::Unsupported(
                "model name is not acquired; use model list or an explicit model pull".into(),
            )
        });
    }
    metadata.select(selector, pin)
}

/// Missing name/default is distinct from invalid, ambiguous or dangling state.
#[cfg(all(feature = "cache", target_os = "linux"))]
pub fn find_named_cached(
    metadata: &crate::cache::CacheMetadata,
    name: &str,
    pin: Option<&str>,
) -> crate::Result<Option<crate::cache::SnapshotMetadata>> {
    use crate::Error;
    if let Some(selected) = metadata.named(name)? {
        if pin.is_some_and(|pin| pin != selected.digest) {
            return Err(Error::Malformed(
                "explicit snapshot does not match requested model name".into(),
            ));
        }
        return Ok(Some(selected));
    }
    #[cfg(feature = "model-qwen35")]
    if let Some(digest) = qwen_source_snapshot(name) {
        if pin.is_some_and(|pin| pin != digest) {
            return Err(Error::Malformed("explicit snapshot does not match the approved catalog default; select the desired digest directly".into()));
        }
        return metadata.snapshot_optional(digest);
    }
    Ok(None)
}

#[cfg(all(feature = "model-qwen35", feature = "acquisition", target_os = "linux"))]
#[path = "qwen35_source/pull_recipe.rs"]
mod pull_recipe;
#[cfg(all(feature = "model-qwen35", feature = "acquisition", target_os = "linux"))]
pub use pull_recipe::QwenPullRecipe;

pub use crate::model::DeclaredModel;
pub use zerogravity_core::model::{
    ModelAssembly, ModelDefinition, ModelProgram, ModelRegion, ProgramRequest, SourceRegionOperator,
};
pub use zerogravity_core::provider::{
    ExecutionDescription, PreparedExecutable, ProgramDescription, RegionDescription,
    RegionExecutionProvider,
};
pub use zerogravity_core::source::{ProgramSource, RegionSource, SourceTensorRef, TensorSource};
pub use zerogravity_engine::diagnostics::{
    ArtifactDescription, LayerDevice, LayerPlacement, LayerSpan, ModelCapabilities, ModelDetails,
    ModelLifecycle, ModelMemory, ModelSnapshot, SourceRevision,
};
pub use zerogravity_engine::model_instance::{ModelInstance, PreparedModel};
pub use zerogravity_engine::router::performance::{ModelPerformance, RequestPerformance};
pub use zerogravity_engine::{
    executable::ExecutableBundle,
    prepare::{prepare as prepare_executable, prepare_region},
};

#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
const MIB: usize = 1024 * 1024;
#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
const QWEN_CUDA_PROVIDER_BYTES: usize = 4608 * MIB;
#[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
const QWEN_CUDA_DEVICE_BYTES: u64 = 5120 * MIB as u64;

/// Explicit execution policy for one selected text route set.
/// Source selectors, model IDs and snapshot pins come from explicit
/// declarations, optionally selected by a graph. No acquisition or fallback.
pub enum TextLoadPolicy {
    Cpu(crate::runtime::TextConfig),
    #[cfg(all(
        feature = "text",
        any(feature = "vulkan", feature = "cuda", feature = "hip")
    ))]
    Gpu {
        backend: crate::devices::GpuBackend,
        profile: crate::builtins::text::GpuProfile,
    },
    /// All used declarations must resolve to the SAME approved HF snapshot.
    #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
    QwenHipNative {
        cache: crate::cache::CacheOptions,
        profile: crate::models::QwenServingProfile,
        budget: crate::models::HipSourceBudget,
    },
    /// Same-snapshot aliases; existing smart startup and device envelopes.
    #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
    QwenCuda {
        cache: crate::cache::CacheOptions,
        profile: crate::models::QwenServingProfile,
    },
}

/// One complete cold route set. Aliases share the original source.
/// A configured runtime refuses a new load until its last owner is retired.
pub enum ModelLoadRequest {
    /// Canonical original source with smart startup configuration Enabled.
    #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
    QwenCuda {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        /// Provider capacity, not total device memory.
        provider_bytes: usize,
        /// Physical device admission, including the backend's context owners.
        device_bytes: u64,
    },
    /// User state-policy value; under Enabled it overrides this knob while
    /// future unspecified smart-config knobs remain eligible for Auto selection.
    #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
    QwenCudaWithStatePolicy {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        provider_bytes: usize,
        device_bytes: u64,
        state_policy: QwenCudaStatePolicy,
    },
    /// Explicit control of the shared startup smart-config layer.
    #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
    QwenCudaWithSmartConfig {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        provider_bytes: usize,
        device_bytes: u64,
        smart_config: SmartConfig<QwenCudaSmartOverrides>,
    },
    /// Canonical original HF source through the shared worker; cold HIP only.
    #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
    QwenHip {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
    },
    /// Same cold HIP route with an explicit total device envelope, including context.
    #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
    QwenHipWithBudget {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        budget: HipSourceBudget,
    },
    /// Native HIP BF16 projection profile; requires admitted resident weights.
    #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
    QwenHipNative {
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        budget: HipSourceBudget,
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

impl ModelLoadRequest {
    /// Canonical production Qwen CUDA load. Backend/model selection remains
    /// caller policy; this constructor owns the adapter's qualified envelopes.
    #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
    pub fn qwen_cuda(
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
    ) -> Self {
        Self::QwenCuda {
            aliases,
            cache,
            snapshot,
            profile,
            provider_bytes: QWEN_CUDA_PROVIDER_BYTES,
            device_bytes: QWEN_CUDA_DEVICE_BYTES,
        }
    }

    /// Explicit native ROCm projection route, distinct from the CPU reference.
    #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
    pub fn qwen_hip_native(
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        budget: HipSourceBudget,
    ) -> Self {
        Self::QwenHipNative {
            aliases,
            cache,
            snapshot,
            profile,
            budget,
        }
    }

    /// Canonical production Qwen HIP load with an explicit device envelope.
    #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
    pub fn qwen_hip(
        aliases: Vec<String>,
        cache: crate::cache::CacheOptions,
        snapshot: String,
        profile: QwenServingProfile,
        budget: HipSourceBudget,
    ) -> Self {
        Self::QwenHipWithBudget {
            aliases,
            cache,
            snapshot,
            profile,
            budget,
        }
    }
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
    pub fn snapshot(&self) -> crate::Result<Vec<ModelSnapshot>> {
        self.runtime.engine.models().snapshot()
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
    /// Trusted prepared encoder composition. The original cold load slot
    /// covers preparation, publication and worker retirement.
    pub fn load_encoders(
        &self,
        config: crate::embedding::EncoderConfig,
        load: impl FnOnce(
            &crate::embedding::ServingHost,
        ) -> std::result::Result<
            crate::embedding::EncoderModels,
            crate::runtime::text::ServiceError,
        >,
    ) -> std::result::Result<(), crate::runtime::text::ServiceError> {
        self.runtime.engine.configure_text(|| {
            let worker = config.worker(self.runtime.engine.limits())?;
            let host = crate::embedding::ServingHost::new(config.host_mib)?;
            let models = load(&host)?;
            zerogravity_engine::router::Router::with_embeddings(models, host, worker)
        })
    }

    /// Trusted prepared speech owners on the original cold runtime slot.
    /// No implicit model acquisition or public endpoint eligibility.
    pub fn load_transcribers(
        &self,
        config: crate::transcription::TranscriptionConfig,
        load: impl FnOnce(
            &crate::embedding::ServingHost,
        ) -> std::result::Result<
            crate::transcription::SpeechModels,
            crate::runtime::text::ServiceError,
        >,
    ) -> std::result::Result<(), crate::runtime::text::ServiceError> {
        self.runtime.engine.configure_text(|| {
            let worker = config.worker(self.runtime.engine.limits())?;
            let host = crate::embedding::ServingHost::new(config.host_mib)?;
            zerogravity_engine::router::Router::with_transcription(load(&host)?, host, worker)
        })
    }

    /// Exact observed registration, for any task kind in the catalog.
    pub fn unload_registration(&self, alias: &str, registration_id: u64) -> crate::Result<()> {
        self.runtime
            .engine
            .unload_registration(alias, registration_id)
    }

    /// Resolve explicit existing source declarations through the same cold
    /// route preparation as graph-selected models. None uses the default
    /// cache root; original-HF policies own their explicit CacheOptions.
    pub fn load_declared(
        &self,
        declared: &[crate::pipeline::ModelInstance],
        policy: TextLoadPolicy,
        cache_root: Option<&std::path::Path>,
    ) -> std::result::Result<(), crate::runtime::text::ServiceError> {
        self.runtime.engine.configure_text(|| {
            zerogravity_engine::router::validate_aliases(declared.iter().map(|m| m.id.as_str()))?;
            let models = declared.iter().collect::<Vec<_>>();
            crate::builtins::load_declared_text(
                &models,
                policy,
                cache_root,
                self.runtime.engine.limits(),
            )
        })
    }

    /// Current aliases of the exact registered prepared owner.
    pub fn aliases(&self, model: &ModelInstance) -> crate::Result<Vec<String>> {
        self.runtime.engine.models().aliases(model)
    }
    /// Retire this prepared model and all its aliases without closing the
    /// runtime. Accepted work drains; independent owners continue serving
    /// when the adapter supports selective retirement. The last owner joins
    /// the original workers; retained source/tokenizer readers remain valid.
    pub fn unload(&self, model: &ModelInstance) -> crate::Result<()> {
        self.runtime.engine.models().unload(model)
    }
}
