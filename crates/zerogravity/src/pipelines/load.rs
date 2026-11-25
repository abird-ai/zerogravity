//! Resolve only the selected graph's declared sources through existing loaders.
use super::ModelInstance;
use crate::{
    models::ModelLoadRequest,
    run_policy::SourceSelector,
    runtime::{TextConfig, text::ServiceError},
};
use std::{collections::BTreeMap, path::PathBuf};
use zerogravity_engine::{router::Router, scheduler::SchedulerLimits};

/// Explicit execution policy for one selected serial language chain.
/// Source paths, model IDs and snapshot pins come from the graph, never
/// from a parallel alias/source table. No automatic acquisition or fallback.
pub enum LanguageLoadPolicy {
    Cpu(TextConfig),
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

struct LocalSources {
    declared: Vec<(String, PathBuf)>,
    targets: BTreeMap<String, String>,
}
fn unsupported(message: &'static str) -> ServiceError {
    ServiceError::new(400, "unsupported", message)
}
fn require_original_sidecars(model: &ModelInstance) -> Result<(), ServiceError> {
    if model.model_config.is_some() || model.tokenizer.is_some() || model.native_artifacts.is_some()
    {
        return Err(unsupported(
            "pipeline source sidecar overrides are not qualified; use the original source's sidecars",
        ));
    }
    Ok(())
}
fn local_sources(models: &[&ModelInstance]) -> Result<LocalSources, ServiceError> {
    let mut declared = Vec::with_capacity(models.len());
    let mut targets = BTreeMap::new();
    for model in models {
        require_original_sidecars(model)?;
        if model.snapshot.is_some()
            || !matches!(model.target.as_str(), "qwen3.5-0.8b" | "qwen3.5-2b")
        {
            return Err(unsupported(
                "local language pipeline requires a qualified Qwen GGUF target and no HF snapshot pin",
            ));
        }
        let source = model.source.as_deref().ok_or_else(|| {
            ServiceError::new(
                400,
                "invalid_model",
                "pipeline declaration requires an explicit source",
            )
        })?;
        let SourceSelector::Local(path) =
            SourceSelector::parse(source).map_err(ServiceError::inference)?
        else {
            return Err(unsupported(
                "local language pipeline requires a local source path; no acquisition",
            ));
        };
        declared.push((model.id.clone(), path));
        targets.insert(model.id.clone(), model.target.clone());
    }
    Ok(LocalSources { declared, targets })
}

#[cfg(all(
    feature = "model-qwen35",
    any(feature = "cuda", feature = "hip"),
    target_os = "linux"
))]
fn original_snapshot(models: &[&ModelInstance]) -> Result<(Vec<String>, String), ServiceError> {
    use crate::models::qwen_source_inventory::select_snapshot;
    let mut selected = None;
    let mut aliases = Vec::with_capacity(models.len());
    for model in models {
        require_original_sidecars(model)?;
        let pin = model.snapshot.as_deref().ok_or_else(|| {
            ServiceError::new(
                400,
                "invalid_model",
                "original HF pipeline requires an exact snapshot pin",
            )
        })?;
        let spec = select_snapshot(pin).map_err(ServiceError::inference)?;
        if spec.target != model.target {
            return Err(ServiceError::new(
                400,
                "invalid_model",
                "declared target differs from approved snapshot",
            ));
        }
        if let Some(source) = &model.source {
            if !matches!(SourceSelector::parse(source).map_err(ServiceError::inference)?,
                SourceSelector::HuggingFace(repository) if repository == spec.repository)
            {
                return Err(unsupported(
                    "HF pipeline source selector must match the pinned approved repository",
                ));
            }
        }
        if selected.is_some_and(|prior| prior != spec.snapshot_sha) {
            return Err(unsupported(
                "original HF pipeline supports one distinct source per runtime",
            ));
        }
        selected = Some(spec.snapshot_sha);
        aliases.push(model.id.clone());
    }
    Ok((
        aliases,
        selected
            .ok_or_else(|| {
                ServiceError::new(400, "invalid_model", "empty pipeline source selection")
            })?
            .into(),
    ))
}

impl LanguageLoadPolicy {
    pub(super) fn load(
        self,
        models: &[&ModelInstance],
        limits: SchedulerLimits,
    ) -> Result<Router, ServiceError> {
        let (request, targets) = match self {
            Self::Cpu(config) => {
                let LocalSources { declared, targets } = local_sources(models)?;
                (ModelLoadRequest::CpuText { declared, config }, targets)
            }
            #[cfg(all(
                feature = "text",
                any(feature = "vulkan", feature = "cuda", feature = "hip")
            ))]
            Self::Gpu { backend, profile } => {
                let LocalSources { declared, targets } = local_sources(models)?;
                (
                    ModelLoadRequest::GpuText {
                        declared,
                        backend,
                        profile,
                    },
                    targets,
                )
            }
            #[cfg(all(feature = "model-qwen35", feature = "hip", target_os = "linux"))]
            Self::QwenHipNative {
                cache,
                profile,
                budget,
            } => {
                let (aliases, snapshot) = original_snapshot(models)?;
                (
                    ModelLoadRequest::qwen_hip_native(aliases, cache, snapshot, profile, budget),
                    BTreeMap::new(),
                )
            }
            #[cfg(all(feature = "model-qwen35", feature = "cuda", target_os = "linux"))]
            Self::QwenCuda { cache, profile } => {
                let (aliases, snapshot) = original_snapshot(models)?;
                (
                    ModelLoadRequest::qwen_cuda(aliases, cache, snapshot, profile),
                    BTreeMap::new(),
                )
            }
        };
        crate::builtins::load_text_declared(request, limits, &targets)
    }
}
