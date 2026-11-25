//! Canonical serve grammar, using the same typed document and source policy
//! as run. Legacy explicit diagnostic/local forms are compatibility adapters.
use super::*;
use zerogravity::{
    Error,
    pipeline::ModelInstance,
    pipelines::ServeDoc,
    run::config::{self, DocumentSelection, RunOverrides},
    run_policy::{Backend, Headroom, Request, SourceSelector},
};

fn invalid(message: &'static str) -> ServiceError {
    ServiceError::new(400, "invalid_config", message)
}
fn core(error: Error) -> ServiceError {
    ServiceError::inference(error)
}

pub(super) enum Invocation {
    Legacy(Box<YamlConfig>),
    Canonical(Box<Options>),
}

pub(super) struct Options {
    request: Request,
    settings: ServeDoc,
    pending_override: Option<Option<usize>>,
    pull: bool,
    offline: bool,
    ask_before_pull: bool,
}

#[derive(Default)]
struct Parsed {
    models: Vec<ModelInstance>,
    overrides: RunOverrides,
    settings: ServeDoc,
    pending: Option<Option<usize>>,
    file: Option<String>,
    profile: Option<String>,
    pull: bool,
    offline: bool,
    // Old advanced options and explicit local-only routes retain their
    // original owner and defaults. Never silently mix old/new policies.
    advanced: bool,
    canonical: bool,
    explicit_local: bool,
}

impl Invocation {
    pub(super) fn parse(args: &[String]) -> Result<Self, ServiceError> {
        if args.len() > 2048
            || args
                .iter()
                .try_fold(0usize, |n, s| n.checked_add(s.len()))
                .is_none_or(|n| n > 2 * 1024 * 1024)
        {
            return Err(invalid("bounded serve argument count/bytes"));
        }
        let mut p = Parsed::default();
        let mut seen = std::collections::BTreeSet::new();
        let mut it = args.iter();
        let mut literal = false;
        while let Some(raw) = it.next() {
            if !literal && raw == "--" {
                literal = true;
                continue;
            }
            if literal || !raw.starts_with('-') {
                p.models.push(route(None, raw)?);
                p.canonical = true;
                continue;
            }
            let (flag, inline) = raw
                .split_once('=')
                .map_or((raw.as_str(), None), |(k, v)| (k, Some(v)));
            let flag = match flag {
                "-d" => "--device",
                "--context" => "--max-context",
                "--max-tokens" => "--max-output",
                flag => flag,
            };
            if !matches!(flag, "--model" | "--device") && !seen.insert(flag) {
                return Err(invalid("duplicate serve option"));
            }
            if matches!(
                flag,
                "--pull" | "--offline" | "--enable-per-request-metrics"
            ) {
                if inline.is_some() {
                    return Err(invalid("serve switch takes no value"));
                }
                match flag {
                    "--pull" => {
                        p.pull = true;
                        p.canonical = true;
                    }
                    "--offline" => {
                        p.offline = true;
                        p.canonical = true;
                    }
                    _ => p.settings.enable_per_request_metrics = Some(true),
                }
                continue;
            }
            let value = inline
                .or_else(|| it.next().map(String::as_str))
                .ok_or_else(|| invalid("serve option requires a value"))?;
            match flag {
                "--model" => {
                    let (id, source) = value
                        .split_once('=')
                        .ok_or_else(|| invalid("--model requires ID=MODEL"))?;
                    let local = matches!(
                        SourceSelector::parse(source).map_err(core)?,
                        SourceSelector::Local(_)
                    );
                    p.explicit_local |= local;
                    p.canonical |= !local;
                    p.models.push(route(Some(id), source)?);
                }
                "--config" => p.file = Some(value.into()),
                "--profile" => {
                    p.profile = Some(value.into());
                    p.canonical = true;
                }
                "--listen" => p.settings.listen = Some(value.into()),
                "--workers" => p.settings.workers = Some(parse_usize(value, flag)?),
                "--active-per-worker" => {
                    p.settings.active_per_worker = Some(parse_usize(value, flag)?)
                }
                "--max-context" => p.overrides.task.context = Some(parse_usize(value, flag)?),
                "--max-output" => p.overrides.task.max_tokens = Some(parse_usize(value, flag)?),
                "--host-mib" => p.settings.host_mib = Some(parse_usize(value, flag)?),
                "--max-pending-requests" => p.pending = Some(parse_optional_usize(value, flag)?),
                "--backend" if matches!(value, "vulkan-amd" | "vulkan-nvidia") => p.advanced = true,
                "--backend" => {
                    p.overrides.backend = Some(Backend::parse(value).map_err(|_| {
                        ServiceError::new(
                            400,
                            "invalid_backend",
                            "backend must be auto, cpu, cuda, rocm or vulkan",
                        )
                    })?);
                    p.canonical |= value == "auto" || value == "vulkan";
                }
                "--device" => {
                    p.overrides
                        .devices
                        .get_or_insert_with(Vec::new)
                        .push(value.into());
                    p.canonical = true;
                }
                "--headroom" => {
                    p.overrides.headroom = Some(Headroom::parse(value).map_err(core)?);
                    p.canonical = true;
                }
                "--numerical" => {
                    p.overrides.numerical = Some(value.into());
                    p.canonical = true;
                }
                "--source-profile"
                | "--native-artifacts"
                | "--provider-mib"
                | "--device-mib"
                | "--batch-mode"
                | "--batch-rows"
                | "--native-batch-rows" => p.advanced = true,
                _ => return Err(invalid("unknown serve option; use serve --help")),
            }
        }
        if p.pull && p.offline {
            return Err(invalid("--offline conflicts with --pull"));
        }
        if p.file.is_some() && p.profile.is_some() {
            return Err(invalid("--config and --profile are mutually exclusive"));
        }
        // Keep the established finite/GGUF explicit local flag grammar,
        // including old benchmark profiles, without guessing new source paths.
        if p.file.is_none()
            && p.profile.is_none()
            && !p.canonical
            && (p.advanced || p.explicit_local)
        {
            return YamlConfig::cli(args).map(Box::new).map(Self::Legacy);
        }
        if p.advanced {
            return Err(invalid(
                "advanced legacy serving profiles require the explicit local --model ID=PATH grammar",
            ));
        }
        let cwd = std::env::current_dir().map_err(|error| core(error.into()))?;
        let selection = match (&p.file, &p.profile) {
            (Some(file), None) => DocumentSelection::File(Path::new(file)),
            (None, Some(profile)) => DocumentSelection::Profile(profile),
            (None, None) => DocumentSelection::Default,
            _ => unreachable!(),
        };
        let selected = config::select_document(
            if selection == DocumentSelection::Default {
                DocumentSelection::None
            } else {
                selection
            },
            &cwd,
            std::env::var_os("XDG_CONFIG_HOME").as_deref(),
            std::env::var_os("HOME").as_deref(),
        )
        .map_err(core)?;
        let document = if selection == DocumentSelection::Default {
            zerogravity::configuration::load_selected(
                selection,
                &cwd,
                std::env::var_os("XDG_CONFIG_HOME").as_deref(),
                std::env::var_os("HOME").as_deref(),
            )
            .map_err(core)?
        } else if let Some(path) = selected {
            let bytes = zerogravity::pipelines::read_document_bytes(&path).map_err(core)?;
            match zerogravity::pipelines::parse_document(&bytes) {
                Ok(doc) => Some((path, doc)),
                Err(error) => {
                    if p.file.is_some()
                        && args.len() == 2
                        && p.models.is_empty()
                        && !p.canonical
                        && let Ok(legacy) = YamlConfig::from_yaml(&path, &bytes)
                    {
                        return Ok(Self::Legacy(Box::new(legacy)));
                    }
                    return Err(core(error));
                }
            }
        } else {
            None
        };
        let preferences = document
            .as_ref()
            .map(|(_, doc)| {
                (
                    doc.ui.unwrap_or_default(),
                    doc.downloads.unwrap_or_default(),
                )
            })
            .unwrap_or_default();
        let (request, mut settings) =
            config::resolve_serving(document, p.overrides, p.models, &cwd).map_err(core)?;
        macro_rules! overlay {
            ($($field:ident),*) => { $(settings.$field = p.settings.$field.or(settings.$field);)* }
        }
        overlay!(
            listen,
            workers,
            active_per_worker,
            host_mib,
            enable_per_request_metrics
        );
        #[cfg(not(test))]
        crate::output::install_document(preferences.0);
        Ok(Self::Canonical(Box::new(Options {
            request,
            settings,
            pending_override: p.pending,
            pull: p.pull,
            offline: p.offline,
            ask_before_pull: preferences.1.ask_before_pull.unwrap_or(true),
        })))
    }
}

fn route(id: Option<&str>, source: &str) -> Result<ModelInstance, ServiceError> {
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

pub(super) fn run(
    options: Box<Options>,
    shutdown: &mut shutdown::Shutdown,
) -> Result<(), ServiceError> {
    let Options {
        request,
        settings,
        pending_override,
        pull,
        offline,
        ask_before_pull,
    } = *options;
    let listener = Listener::new(
        settings.listen.as_deref().unwrap_or("127.0.0.1:8000"),
        settings.enable_per_request_metrics.unwrap_or(false),
    )?;
    let pending = pending_override.unwrap_or(settings.max_pending_requests);
    let active = settings.active_per_worker.unwrap_or(default_active());
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
    let offer = crate::run_cli::input::interaction().map_err(core)? && !offline && ask_before_pull;
    if shutdown.is_requested() {
        return Ok(());
    }
    #[cfg(all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux"
    ))]
    {
        execute(
            request, settings, pending, active, listener, offer, pull, shutdown,
        )
    }
    #[cfg(not(all(
        feature = "model-qwen35",
        feature = "text",
        feature = "cache",
        target_os = "linux"
    )))]
    {
        let _ = (request, settings, pending, active, listener, offer, pull);
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
#[allow(clippy::too_many_arguments)]
fn execute(
    mut request: Request,
    settings: ServeDoc,
    pending: Option<usize>,
    active: usize,
    listener: Listener,
    offer: bool,
    pull: bool,
    shutdown: &mut shutdown::Shutdown,
) -> Result<(), ServiceError> {
    use crate::run_cli::source;
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
        && !(offer || pull)
    {
        return Err(core(missing.not_cached()));
    }
    if gguf && original.is_some() {
        return Err(invalid(
            "mixed GGUF and original GPU source serving is not qualified",
        ));
    }
    let inventory = zerogravity::devices::discover().map_err(core)?;
    if shutdown.is_requested() {
        return Ok(());
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
            let selected = qwen_source::select_backend(&request, &inventory, target).map_err(core)?
                .ok_or_else(|| invalid("no eligible compiled original-source GPU route; select an acquired GGUF for CPU"))?;
            if let Some(missing) = &missing {
                match source::acquire_missing(
                    &request,
                    &inventory,
                    offer,
                    pull,
                    missing,
                    Some(shutdown.requested.clone()),
                ) {
                    Ok(acquired) if acquired == digest => {}
                    Ok(_) => return Err(invalid("acquisition returned a different snapshot")),
                    Err(error) => {
                        #[cfg(feature = "acquisition")]
                        if shutdown.is_requested()
                            && zerogravity::control::DownloadStatus::from_error(&error)
                                == zerogravity::control::DownloadStatus::Cancelled
                        {
                            return Ok(());
                        }
                        return Err(core(error));
                    }
                }
            }
            if shutdown.is_requested() {
                return Ok(());
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
            return launch(shutdown, listener, active, pending, |control| {
                control.load(load)
            });
        }
        #[cfg(not(any(feature = "cuda", feature = "hip")))]
        {
            let _ = (digest, missing, offer, pull, inventory);
            return Err(ServiceError::new(
                400,
                "backend_not_compiled",
                "original source serving requires a compiled CUDA/ROCm source route",
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
    let workers = settings.workers.unwrap_or(default_workers());
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
    let host_mib = settings.host_mib.unwrap_or(default_host());
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
    launch(shutdown, listener, workers * active, pending, |control| {
        control.load_declared(
            &request.graph.models,
            zerogravity::models::TextLoadPolicy::Cpu(config),
            None,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_routes_and_document_share_run_settings() {
        let root = std::env::temp_dir().join(format!("zg-serve-config-{}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("run.yaml");
        {
            std::fs::write(
                &path,
                r#"schema: zerogravity.pipeline.v2
model:
  - id: configured
    target: qwen3.5-0.8b
    source: ./weights.gguf
run:
  entrypoint: {model: configured}
  backend: cpu
  device: [cpu:0]
  headroom: '0'
  task: {context: 32, max_tokens: 4, prompt: example}
serve:
  listen: 127.0.0.1:0
  workers: 1
  active_per_worker: 1
  max_context: 64
  max_output: 256
  max_pending_requests: 3
"#,
            )
            .unwrap();
            let parse = |args: &[&str]| {
                Invocation::parse(&args.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
            };
            let path = path.to_str().unwrap();
            let Invocation::Canonical(doc) = parse(&["--config", path]).unwrap() else {
                panic!("canonical document")
            };
            assert_eq!(
                doc.request.graph.models[0].source.as_deref(),
                Some(root.join("./weights.gguf").to_str().unwrap())
            );
            assert_eq!(doc.request.task.context, 64);
            assert_eq!(doc.request.task.max_tokens, 256);
            assert!(doc.request.task.prompt.is_none());
            assert_eq!(doc.request.backend, Backend::Cpu);
            assert_eq!(doc.request.headroom, Headroom::Bytes(0));
            assert_eq!(doc.settings.workers, Some(1));
            assert_eq!(doc.settings.max_pending_requests, Some(3));
            let Invocation::Canonical(overridden) = parse(&[
                "--config",
                path,
                "by-name:tiny",
                "--model",
                "fast=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "--max-output",
                "1",
                "--max-pending-requests",
                "none",
                "--offline",
            ])
            .unwrap() else {
                panic!("canonical routes")
            };
            assert_eq!(
                overridden
                    .request
                    .graph
                    .models
                    .iter()
                    .map(|m| m.id.as_str())
                    .collect::<Vec<_>>(),
                ["tiny", "fast"]
            );
            assert_eq!(overridden.request.task.max_tokens, 1);
            assert_eq!(overridden.pending_override, Some(None));
            assert!(overridden.offline);
            assert!(parse(&["--offline", "tiny", "--pull"]).is_err());
            assert!(parse(&["--config", path, "--profile", "other"]).is_err());
            assert!(parse(&["tiny", "--batch-mode", "fused"]).is_err());
            assert!(matches!(
                parse(&["--model", "local=./weights.gguf"]),
                Ok(Invocation::Legacy(_))
            ));
            assert!(matches!(
                parse(&["--model", "remote=by-name:tiny"]),
                Ok(Invocation::Canonical(_))
            ));
        }
        std::fs::remove_dir_all(root).unwrap();
    }
}
