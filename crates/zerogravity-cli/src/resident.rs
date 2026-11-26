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
    if settings
        .host_mib
        .is_some_and(|n| !(256..=1_048_576).contains(&n))
    {
        return Err(invalid("host-mib must be 256..=1048576"));
    }
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
        prepare_routes(request, settings, pending, active, acquisition)
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
        let selected = source::select_model(model).map_err(core)?;
        let original_digest = if let Some(selected) = selected {
            let digest = selected.expected_snapshot().map_err(core)?.to_owned();
            if missing.is_none() {
                missing = Some(selected);
            }
            source::bind(model, digest.clone()).map_err(core)?;
            Some(digest)
        } else {
            model
                .snapshot
                .as_ref()
                .filter(|digest| zerogravity::models::qwen_target_for_snapshot(digest).is_some())
                .cloned()
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
                &request, &inventory, selected, aliases, cache, digest, profile,
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
