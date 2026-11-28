//! Shared canonical resident text route planning for serve and interactive
//! clients. Only the original SDK loader admits and publishes a route set.
use engine::ServiceError;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity::{
    Error,
    control::ControlPlane,
    pipeline::ModelInstance,
    pipelines::ServeDoc,
    run_policy::{Request, SourceSelector},
    runtime::{SchedulerLimits, TextConfig, text as engine},
};

fn invalid(message: &'static str) -> ServiceError {
    ServiceError::new(400, "invalid_config", message)
}
fn core(error: Error) -> ServiceError {
    ServiceError::inference(error)
}

#[cfg(any(
    feature = "serve",
    all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux"
    )
))]
pub(crate) const DEFAULT_WORKERS: usize = 2;
pub(crate) const DEFAULT_ACTIVE: usize = 2;
#[cfg(any(
    feature = "serve",
    all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux"
    )
))]
pub(crate) const DEFAULT_HOST_MIB: usize = 4096;

pub(crate) struct Acquisition {
    pub offer: bool,
    pub pull: bool,
    pub cancel: Arc<AtomicBool>,
}
impl Acquisition {
    fn requested(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
}

pub(crate) struct Plan {
    #[cfg_attr(
        not(feature = "serve"),
        allow(
            dead_code,
            reason = "standalone serving uses exact outer limits; interactive host has a reusable upper envelope"
        )
    )]
    pub limits: SchedulerLimits,
    load: Load,
}
enum Load {
    #[cfg_attr(
        not(all(
            feature = "model-qwen35",
            feature = "text",
            feature = "cache",
            target_os = "linux"
        )),
        allow(dead_code, reason = "no canonical resident route compiled")
    )]
    Declared(Vec<ModelInstance>, TextConfig),
    #[cfg(all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux",
        any(feature = "cuda", feature = "hip")
    ))]
    Native(zerogravity::models::ModelLoadRequest),
}
impl Plan {
    pub(crate) fn load(self, control: ControlPlane<'_>) -> Result<(), ServiceError> {
        match self.load {
            Load::Declared(models, config) => control.load_declared(
                &models,
                zerogravity::models::TextLoadPolicy::Cpu(config),
                None,
            ),
            #[cfg(all(
                feature = "model-qwen35",
                feature = "text",
                feature = "cache",
                target_os = "linux",
                any(feature = "cuda", feature = "hip")
            ))]
            Load::Native(request) => control.load(request),
        }
    }
}

pub(crate) fn route(id: Option<&str>, source: &str) -> Result<ModelInstance, ServiceError> {
    let selector = SourceSelector::parse(source).map_err(core)?;
    let id = if let Some(id) = id {
        id.to_owned()
    } else {
        match &selector {
            SourceSelector::Local(path) => {
                path.file_stem().and_then(|s| s.to_str()).map(str::to_owned)
            }
            SourceSelector::Name(name)
            | SourceSelector::ByName(name)
            | SourceSelector::Snapshot(name) => Some(name.clone()),
            SourceSelector::Catalog(name) => name.rsplit('/').next().map(str::to_owned),
            SourceSelector::HuggingFace(locator) | SourceSelector::GitHub(locator) => locator
                .split('@')
                .next()
                .and_then(|repo| repo.rsplit('/').next())
                .map(str::to_owned),
            SourceSelector::Http(_) => None,
        }
        .ok_or_else(|| invalid("use --model ID=MODEL for this source"))?
    };
    engine::validate_alias(&id)?;
    Ok(ModelInstance {
        id,
        target: String::new(),
        source: Some(source.into()),
        model_config: None,
        tokenizer: None,
        snapshot: None,
        native_artifacts: None,
    })
}

pub(crate) fn prepare(
    request: Request,
    settings: ServeDoc,
    pending_override: Option<Option<usize>>,
    acquisition: Acquisition,
    observe_route: &dyn Fn(&str),
) -> Result<Option<Plan>, ServiceError> {
    let pending = pending_override.unwrap_or(settings.max_pending_requests);
    let active = settings.active_per_worker.unwrap_or(DEFAULT_ACTIVE);
    // Validate the complete declaration and transport policy before source or
    // credential IO, including an absent or duplicate model ID.
    engine::validate_aliases(request.graph.models.iter().map(|m| m.id.as_str()))?;
    engine::validate_config(
        request.graph.models.len(),
        settings.workers.unwrap_or(1),
        active,
        request.task.context,
        request.task.max_tokens,
        pending,
    )?;
    zerogravity::configuration::validate_serving_capacity(&settings).map_err(core)?;
    if request.diagnostics.dry_run
        || request.diagnostics.audit_logits_jsonl.is_some()
        || request.diagnostics.audit_layers_jsonl.is_some()
        || request.draft_device.is_some()
    {
        return Err(invalid(
            "serve does not run one-shot diagnostics or draft-device overrides",
        ));
    }
    zerogravity::run::validate_serving_source_request(&request).map_err(core)?;
    if acquisition.requested() {
        return Ok(None);
    }
    #[cfg(all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux"
    ))]
    {
        prepare_routes(
            request,
            settings,
            pending,
            active,
            acquisition,
            observe_route,
        )
    }
    #[cfg(not(all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux"
    )))]
    {
        let _ = (
            request,
            settings,
            pending,
            active,
            acquisition.offer,
            acquisition.pull,
            observe_route,
        );
        Err(ServiceError::new(
            400,
            "backend_not_compiled",
            "canonical serving requires Linux model-qwen35, text and cache with a compiled backend",
        ))
    }
}

#[cfg(all(
    feature = "model-qwen35",
    feature = "text",
    feature = "cache",
    target_os = "linux"
))]
fn limits(max_active: usize, max_queued: Option<usize>) -> SchedulerLimits {
    SchedulerLimits {
        max_active,
        max_queued,
        ..Default::default()
    }
}

/// Cold source route observation shared by the canonical planner and
/// both control clients. Candidate is not authenticated or admitted.
#[cfg(all(
    feature = "cache",
    target_os = "linux",
    any(
        feature = "tui",
        feature = "web",
        all(feature = "model-qwen35", feature = "text")
    )
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachedRoute {
    #[cfg(feature = "model-qwen35")]
    OriginalQwen {
        cuda: bool,
        hip: bool,
    },
    Gguf,
    Refused {
        code: &'static str,
        reason: &'static str,
    },
}
#[cfg(all(
    feature = "cache",
    target_os = "linux",
    any(
        feature = "tui",
        feature = "web",
        all(feature = "model-qwen35", feature = "text")
    )
))]
impl CachedRoute {
    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn label(self) -> &'static str {
        match self {
            #[cfg(feature = "model-qwen35")]
            Self::OriginalQwen { .. } => "Original Qwen · GPU candidate",
            Self::Gguf => "GGUF · CPU candidate",
            Self::Refused { .. } => "No canonical execution route",
        }
    }
    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn permits(self, execution: &crate::control_host::execution::Execution) -> bool {
        use zerogravity::run_policy::Backend;
        let backend = execution.backend.as_ref();
        let device = execution.device.as_deref();
        match self {
            Self::Refused { .. } => backend.is_none() && device.is_none(),
            Self::Gguf => {
                matches!(backend, None | Some(Backend::Auto | Backend::Cpu))
                    && matches!(device, None | Some("cpu:0"))
            }
            #[cfg(feature = "model-qwen35")]
            Self::OriginalQwen { cuda, hip } => {
                device != Some("cpu:0")
                    && match backend {
                        None | Some(Backend::Auto) => cuda || hip,
                        Some(Backend::Cuda) => cuda,
                        Some(Backend::Rocm) => hip,
                        _ => false,
                    }
            }
        }
    }
    #[cfg(any(feature = "tui", feature = "web"))]
    pub fn refusal(self) -> Option<&'static str> {
        match self {
            Self::Refused { reason, .. } => Some(reason),
            _ => None,
        }
    }

    #[cfg(feature = "web")]
    pub fn refusal_code(self) -> Option<&'static str> {
        match self {
            Self::Refused { code, .. } => Some(code),
            _ => None,
        }
    }
}

#[cfg(all(
    feature = "cache",
    target_os = "linux",
    any(
        feature = "tui",
        feature = "web",
        all(feature = "model-qwen35", feature = "text")
    )
))]
pub(crate) fn cached_route(digest: &str, layout: zerogravity::cache::SourceLayout) -> CachedRoute {
    use CachedRoute::*;
    let unsupported = |reason| Refused {
        code: "unsupported_source_route",
        reason,
    };
    let uncompiled = |reason| Refused {
        code: "backend_not_compiled",
        reason,
    };
    #[cfg(not(feature = "model-qwen35"))]
    let _ = digest;
    if layout.weights == 0 {
        return unsupported("Snapshot has no weight payloads; acquire a complete model source");
    }
    if !cfg!(all(feature = "model-qwen35", feature = "text")) {
        return uncompiled("Canonical text model consumer is not compiled");
    }
    #[cfg(feature = "model-qwen35")]
    if let Some(target) = zerogravity::models::qwen_target_for_snapshot(digest) {
        let cuda = cfg!(feature = "cuda") && target != "qwen3.8-27b";
        let hip = cfg!(feature = "hip");
        return if cuda || hip {
            OriginalQwen { cuda, hip }
        } else if target == "qwen3.8-27b" {
            uncompiled(
                "Original Qwen 27B requires the qualified ROCm route; CUDA support is not yet qualified",
            )
        } else {
            uncompiled("Original Qwen requires a compiled CUDA or ROCm backend")
        };
    }
    if let Some(reason) = layout.single_gguf_refusal() {
        return unsupported(reason);
    }
    if !cfg!(feature = "cpu") {
        return uncompiled("Canonical GGUF requires the compiled CPU route");
    }
    Gguf
}

#[cfg(all(
    feature = "model-qwen35",
    feature = "text",
    feature = "cache",
    target_os = "linux"
))]
fn prepare_routes(
    mut request: Request,
    settings: ServeDoc,
    pending: Option<usize>,
    active: usize,
    acquisition: Acquisition,
    observe_route: &dyn Fn(&str),
) -> Result<Option<Plan>, ServiceError> {
    use crate::run_cli::source;
    use zerogravity::run_policy::Backend;
    // Resolve all declarations cold. A missing source is not permission to
    // download: first select one supported original route set and its policy.
    let mut missing = None;
    let mut original: Option<String> = None;
    let mut gguf = false;
    for model in &mut request.graph.models {
        if model.source.is_none()
            || model.native_artifacts.is_some()
            || model.model_config.is_some()
            || model.tokenizer.is_some()
        {
            return Err(invalid(
                "canonical serve requires an explicit source and original role-bound sidecars; use legacy diagnostic routes for overrides",
            ));
        }
        if model.snapshot.is_some()
            && matches!(
                SourceSelector::parse(model.source.as_deref().expect("checked source"))
                    .map_err(core)?,
                SourceSelector::Local(_)
            )
        {
            return Err(invalid(
                "local serving source cannot carry a cache snapshot pin",
            ));
        }
        let original_digest = match source::select_model(model).map_err(core)? {
            source::Selection::Local => None,
            source::Selection::Missing(selected) => {
                let digest = selected.expected_snapshot().map_err(core)?.to_owned();
                if missing.is_none() {
                    missing = Some(selected);
                }
                source::bind(model, digest.clone()).map_err(core)?;
                Some(digest)
            }
            source::Selection::Cached(snapshot) => {
                match cached_route(&snapshot.digest, snapshot.manifest.source_layout()) {
                    CachedRoute::OriginalQwen { .. } => Some(snapshot.digest),
                    CachedRoute::Gguf => None,
                    CachedRoute::Refused { code, reason } => {
                        return Err(ServiceError::from_core(
                            400,
                            code,
                            Error::Unsupported(reason.into()),
                        ));
                    }
                }
            }
        };
        if let Some(digest) = original_digest {
            if original.as_ref().is_some_and(|old| *old != digest) {
                return Err(invalid(
                    "original GPU serving currently supports one distinct approved snapshot shared by aliases",
                ));
            }
            original = Some(digest);
        } else {
            gguf = true;
        }
    }
    if let Some(missing) = &missing
        && !(acquisition.offer || acquisition.pull)
    {
        return Err(core(missing.not_cached()));
    }
    if gguf && original.is_some() {
        return Err(invalid(
            "mixed GGUF and original GPU source serving is not qualified",
        ));
    }
    let inventory = zerogravity::devices::discover().map_err(core)?;
    if acquisition.requested() {
        return Ok(None);
    }
    if let Some(digest) = original {
        #[cfg(any(feature = "cuda", feature = "hip"))]
        {
            use crate::run_cli::qwen_source;
            if settings.host_mib.is_some() {
                return Err(invalid(
                    "original GPU serving uses its source-owned host envelope; --host-mib is a legacy/CPU setting",
                ));
            }
            if settings.workers.unwrap_or(1) != 1 {
                return Err(invalid("original GPU serving requires one worker"));
            }
            let profile = zerogravity::models::QwenServingProfile {
                active,
                context: request.task.context,
                output: request.task.max_tokens,
                max_pending: pending,
                cohort_width: 0,
            };
            profile.validate_base()?;
            let target =
                zerogravity::models::qwen_target_for_snapshot(&digest).expect("approved digest");
            let selected =
                qwen_source::require_source_backend(&request, &inventory, target).map_err(core)?;
            // Record the selected route before device budget/weight admission.
            // The loaded registration remains authoritative for physical placement.
            observe_route(match selected {
                qwen_source::SelectedBackend::Cuda => "CUDA · device 0",
                qwen_source::SelectedBackend::Hip => "ROCm · device 0",
            });
            if settings.device_mib.is_some() && selected != qwen_source::SelectedBackend::Hip {
                return Err(invalid(
                    "canonical device-mib currently requires original ROCm source",
                ));
            }
            if let Some(missing) = &missing {
                match source::acquire_missing(
                    &request,
                    &inventory,
                    acquisition.offer,
                    acquisition.pull,
                    missing,
                    Some(acquisition.cancel.clone()),
                ) {
                    Ok(acquired) if acquired == digest => {}
                    Ok(_) => return Err(invalid("acquisition returned a different snapshot")),
                    Err(error) => {
                        #[cfg(feature = "acquisition")]
                        if acquisition.requested()
                            && zerogravity::control::DownloadStatus::from_error(&error)
                                == zerogravity::control::DownloadStatus::Cancelled
                        {
                            return Ok(None);
                        }
                        return Err(core(error));
                    }
                }
            }
            if acquisition.requested() {
                return Ok(None);
            }
            // Consent/transfer can outlive the original observation. Refresh
            // and require the SAME selected backend, never silently fall back.
            let inventory = zerogravity::devices::discover().map_err(core)?;
            if qwen_source::select_backend(&request, &inventory, target).map_err(core)?
                != Some(selected)
            {
                return Err(invalid(
                    "original selected GPU route changed during startup",
                ));
            }
            let root = zerogravity::cache::paths::resolve_root(
                None,
                std::env::var_os("XDG_CACHE_HOME").as_deref(),
                std::env::var_os("HOME").as_deref(),
            )
            .map_err(core)?;
            let metadata = zerogravity::cache::CacheMetadata::open(&root).map_err(core)?;
            let cache = zerogravity::cache::CacheOptions::new(
                Some(root),
                metadata.disk_limit_bytes().map_err(core)?,
            );
            let aliases = request
                .graph
                .models
                .iter()
                .map(|model| model.id.clone())
                .collect::<Vec<_>>();
            // Keep the common policy alive through request construction.
            // Aliases are borrowed from the still-resolved declaration.
            let load = qwen_source::load_request(
                &request,
                &inventory,
                selected,
                aliases,
                cache,
                digest,
                qwen_source::SourceLoadOptions {
                    profile,
                    cohort_width: settings.cohort_width,
                    device_mib: settings.device_mib,
                },
            )
            .map_err(core)?;
            return Ok(Some(Plan {
                limits: limits(active, pending),
                load: Load::Native(load),
            }));
        }
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        {
            let _ = (digest, missing, inventory);
            return Err(ServiceError::new(
                400,
                "backend_not_compiled",
                "Original-source Qwen models require CUDA or ROCm/HIP. Neither backend is compiled. Build the default CLI or enable cuda/hip; use a supported GGUF for CPU.",
            ));
        }
    }

    if settings.cohort_width.is_some() || settings.device_mib.is_some() {
        return Err(invalid(
            "canonical CPU GGUF serving does not accept GPU cohort/device limits",
        ));
    }
    if !cfg!(feature = "cpu")
        || !matches!(request.backend, Backend::Auto | Backend::Cpu)
        || request.numerical.as_deref() == Some("native")
    {
        return Err(invalid(
            "canonical GGUF serving requires the compiled CPU route; explicit legacy GPU profiles remain separate",
        ));
    }
    let workers = settings.workers.unwrap_or(DEFAULT_WORKERS);
    engine::validate_config(
        request.graph.models.len(),
        workers,
        active,
        request.task.context,
        request.task.max_tokens,
        pending,
    )?;
    let allowed = zerogravity::devices::allowed(&request, &inventory).map_err(core)?;
    let cpu = allowed
        .iter()
        .find(|device| device.class == "host-memory")
        .ok_or_else(|| invalid("selected devices do not include CPU"))?;
    observe_route("CPU · cpu:0");
    let (ceiling, available) = cpu
        .ceiling
        .zip(cpu.available)
        .ok_or_else(|| invalid("CPU memory availability unknown"))?;
    let backing = zerogravity::resource_arbiter::ResourceArbiter::process_shared()
        .host_backing()
        .map_err(core)?;
    let margin = request.headroom.bytes(ceiling, true).map_err(core)?;
    let host_mib = settings.host_mib.unwrap_or(DEFAULT_HOST_MIB);
    let usable = backing.conservative_available(ceiling, available, margin);
    if host_mib as u64 * 1024 * 1024 > usable {
        return Err(core(Error::resource(
            "host memory",
            "serving envelope after headroom",
            host_mib as u64 * 1024 * 1024,
            usable,
        )));
    }
    for model in &mut request.graph.models {
        model.target =
            zerogravity::prepare::gguf_target(model, request.task.context).map_err(core)?;
    }
    let config = zerogravity::runtime::TextConfig {
        workers,
        active_per_worker: active,
        max_context: request.task.context,
        max_output: request.task.max_tokens,
        host_mib,
        max_pending: pending,
    };
    Ok(Some(Plan {
        limits: limits(workers * active, pending),
        load: Load::Declared(request.graph.models, config),
    }))
}

#[cfg(all(
    test,
    feature = "model-qwen35",
    feature = "cache",
    target_os = "linux",
    any(feature = "tui", feature = "web")
))]
mod tests {
    use super::*;
    use crate::control_host::execution::Execution;
    use zerogravity::{cache::SourceLayout, run_policy::Backend};

    #[test]
    fn source_candidates_filter_choices_without_granting_execution() {
        let missing = cached_route("unrelated", SourceLayout::default());
        assert!(missing.refusal().unwrap().contains("no weight payloads"));
        assert!(missing.permits(&Execution::default()));
        let gguf = CachedRoute::Gguf;
        let hip_only = CachedRoute::OriginalQwen {
            cuda: false,
            hip: true,
        };
        for (backend, gguf_ok, hip_ok) in [
            (Backend::Auto, true, true),
            (Backend::Cpu, true, false),
            (Backend::Cuda, false, false),
            (Backend::Rocm, false, true),
            (Backend::Vulkan, false, false),
        ] {
            let choice = Execution {
                backend: Some(backend),
                device: None,
            };
            assert_eq!(gguf.permits(&choice), gguf_ok);
            assert_eq!(hip_only.permits(&choice), hip_ok);
            assert!(!missing.permits(&choice));
        }
        assert!(!gguf.permits(&Execution {
            backend: Some(Backend::Auto),
            device: Some("pci:0000:01:00.0".into())
        }));
        assert!(!hip_only.permits(&Execution::on_device("cpu:0".into())));
        let original = SourceLayout {
            weights: 2,
            safetensors: 2,
            configs: 1,
            tokenizers: 1,
            ..Default::default()
        };
        assert!(cached_route("unapproved", original).refusal().is_some());
        #[cfg(all(feature = "model-qwen35", feature = "text", feature = "hip"))]
        assert_eq!(
            cached_route(
                zerogravity::models::qwen_source_snapshot("qwen3.8-27b").unwrap(),
                original
            ),
            hip_only
        );
    }
}
