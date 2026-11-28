//! Canonical serve grammar, using the same typed document and source policy
//! as run. Legacy explicit diagnostic/local forms are compatibility adapters.
use super::*;
use crate::resident::route;
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
    cohort: Option<Option<usize>>,
    device_mib: Option<Option<usize>>,
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
                "--pull" | "--offline" | "--enable-per-request-metrics" | "--enable-error-details"
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
                    "--enable-error-details" => {
                        p.settings.enable_error_details = Some(true);
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
                "--cohort-width" => {
                    p.cohort = Some(if value == "auto" {
                        None
                    } else {
                        Some(parse_usize(value, flag)?)
                    });
                    p.canonical = true;
                }
                "--device-mib" => {
                    p.device_mib = Some(if value == "auto" {
                        p.canonical = true;
                        None
                    } else {
                        Some(parse_usize(value, flag)?)
                    });
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
            enable_per_request_metrics,
            enable_error_details
        );
        settings.cohort_width = p.cohort.unwrap_or(settings.cohort_width);
        settings.device_mib = p.device_mib.unwrap_or(settings.device_mib);
        zerogravity::configuration::validate_serving_capacity(&settings).map_err(core)?;
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
        settings.enable_error_details.unwrap_or(false),
    )?;
    let offer = crate::run_cli::input::interaction().map_err(core)? && !offline && ask_before_pull;
    crate::output::operational::status(
        "models",
        crate::output::operational::Level::Info,
        &format!(
            "Resolving model routes · backend {:?} · context {} · output {}",
            request.backend, request.task.context, request.task.max_tokens
        ),
    );
    let Some(plan) = crate::resident::prepare(
        request,
        settings,
        pending_override,
        crate::resident::Acquisition {
            offer,
            pull,
            cancel: shutdown.requested.clone(),
        },
        &|route| {
            crate::output::operational::status(
                "models",
                crate::output::operational::Level::Info,
                &format!("Selected route · {route}"),
            )
        },
    )?
    else {
        return Ok(());
    };
    launch(
        shutdown,
        listener,
        plan.limits.max_active,
        plan.limits.max_queued,
        |control| plan.load(control),
    )
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
            let Invocation::Canonical(native) = parse(&[
                "--config",
                path,
                "by-name:tiny",
                "--backend",
                "rocm",
                "--device",
                "auto",
                "--numerical",
                "native",
                "--active-per-worker",
                "8",
                "--cohort-width",
                "4",
                "--device-mib",
                "24576",
            ])
            .unwrap() else {
                panic!("canonical native policy")
            };
            assert_eq!(native.settings.cohort_width, Some(4));
            assert_eq!(native.settings.device_mib, Some(24576));
            assert!(parse(&["by-name:tiny", "--cohort-width", "8"]).is_err());
            assert!(parse(&["by-name:tiny", "--device-mib", "0"]).is_err());
            let Invocation::Canonical(auto) = parse(&[
                "by-name:tiny",
                "--cohort-width",
                "auto",
                "--device-mib",
                "auto",
            ])
            .unwrap() else {
                panic!("canonical automatic policy")
            };
            assert_eq!(auto.settings.cohort_width, None);
            assert_eq!(auto.settings.device_mib, None);
            assert!(matches!(
                parse(&["--model", "legacy=./model.gguf", "--device-mib", "4096"]),
                Ok(Invocation::Legacy(_))
            ));
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
