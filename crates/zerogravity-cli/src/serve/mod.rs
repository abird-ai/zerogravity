//! Opt-in local OpenAI-compatible HTTP serving.
//! Facade loading + shared engine/transport; CLI owns configuration and signals.

#[cfg(test)]
use engine::Router;
use engine::ServiceError;
use zerogravity::runtime::text as engine;
use zerogravity::runtime::text::pending;
mod canonical;
mod shutdown;

use serde::{Deserialize, Deserializer};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::Arc;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct YamlConfig {
    #[serde(default = "default_listen")]
    listen: String,
    models: Vec<YamlModel>,
    #[serde(default = "default_workers")]
    workers: usize,
    #[serde(default = "default_active")]
    active_per_worker: usize,
    #[serde(default = "default_context")]
    max_context: usize,
    #[serde(default = "default_output")]
    max_output: usize,
    #[serde(default = "default_host")]
    host_mib: usize,
    #[serde(default = "default_backend")]
    backend: String,
    #[serde(default = "default_source_profile")]
    source_profile: String,
    #[serde(default)]
    native_artifacts: Option<PathBuf>,
    #[serde(default = "default_provider_mib")]
    provider_mib: usize,
    #[serde(default = "default_device_mib")]
    device_mib: usize,
    #[serde(default = "default_batch_mode")]
    batch_mode: String,
    #[serde(default = "default_batch_rows")]
    batch_rows: usize,
    #[serde(default)]
    native_batch_rows: usize,
    #[serde(default, deserialize_with = "yaml_pending_limit")]
    max_pending_requests: Option<usize>,
    #[serde(default)]
    enable_per_request_metrics: bool,
    #[serde(default)]
    enable_error_details: bool,
}
fn yaml_pending_limit<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_yaml::Value>::deserialize(deserializer)?;
    match value {
        None | Some(serde_yaml::Value::Null) => Ok(None),
        Some(serde_yaml::Value::Number(number)) => number
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("max_pending_requests out of range")),
        Some(serde_yaml::Value::String(value))
            if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("null") =>
        {
            Ok(None)
        }
        Some(serde_yaml::Value::String(value)) => value.parse::<usize>().map(Some).map_err(|_| {
            serde::de::Error::custom("max_pending_requests must be a positive count or none")
        }),
        Some(_) => Err(serde::de::Error::custom(
            "max_pending_requests must be an integer or none",
        )),
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct YamlModel {
    id: String,
    path: PathBuf,
}
fn default_listen() -> String {
    "127.0.0.1:8000".into()
}
fn default_workers() -> usize {
    crate::resident::DEFAULT_WORKERS
}
fn default_active() -> usize {
    crate::resident::DEFAULT_ACTIVE
}
fn default_context() -> usize {
    1024
}
fn default_output() -> usize {
    128
}
fn default_host() -> usize {
    crate::resident::DEFAULT_HOST_MIB
}
fn default_source_profile() -> String {
    "gguf".into()
}
fn default_backend() -> String {
    "cpu".into()
}
fn default_provider_mib() -> usize {
    2048
}
fn default_device_mib() -> usize {
    4096
}
fn default_batch_rows() -> usize {
    2
}
fn default_batch_mode() -> String {
    "interleaved".into()
}

impl YamlConfig {
    fn from_yaml(path: &Path, raw: &[u8]) -> Result<Self, ServiceError> {
        if raw.is_empty() || raw.len() > 64 * 1024 {
            return Err(ServiceError::new(
                400,
                "config_limit",
                "serving YAML exceeds 64 KiB",
            ));
        }
        let mut parsed: Self = serde_yaml::from_slice(raw)
            .map_err(|_| ServiceError::new(400, "config_invalid", "invalid typed serving YAML"))?;
        let base = path.parent().unwrap_or(Path::new("."));
        for model in &mut parsed.models {
            if model.path.is_relative() {
                model.path = base.join(&model.path);
            }
        }
        if let Some(native) = &mut parsed.native_artifacts
            && native.is_relative()
        {
            *native = std::fs::canonicalize(base.join(&*native)).map_err(|_| {
                ServiceError::new(
                    400,
                    "native_artifacts_missing",
                    "cannot resolve native artifacts directory",
                )
            })?;
        }
        Ok(parsed)
    }

    fn cli(args: &[String]) -> Result<Self, ServiceError> {
        let mut config = Self {
            listen: default_listen(),
            models: Vec::new(),
            workers: default_workers(),
            active_per_worker: default_active(),
            max_context: default_context(),
            max_output: default_output(),
            host_mib: default_host(),
            backend: default_backend(),
            source_profile: default_source_profile(),
            native_artifacts: None,
            provider_mib: default_provider_mib(),
            device_mib: default_device_mib(),
            batch_mode: default_batch_mode(),
            batch_rows: default_batch_rows(),
            native_batch_rows: 0,
            max_pending_requests: None,
            enable_per_request_metrics: false,
            enable_error_details: false,
        };
        if args.first().is_some_and(|x| x == "--config") {
            if args.len() != 2 {
                return Err(ServiceError::new(
                    400,
                    "invalid_config",
                    "--config must stand alone",
                ));
            }
            let path = Path::new(&args[1]);
            let raw = zerogravity::pipelines::read_document_bytes(path).map_err(|_| {
                ServiceError::new(400, "config_unreadable", "cannot read bounded serving YAML")
            })?;
            return Self::from_yaml(path, &raw);
        }
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if flag == "--enable-per-request-metrics" {
                config.enable_per_request_metrics = true;
                continue;
            }
            if flag == "--enable-error-details" {
                config.enable_error_details = true;
                continue;
            }
            let value = it.next().ok_or_else(|| {
                ServiceError::new(400, "invalid_flag", format!("{flag} requires a value"))
            })?;
            match flag.as_str() {
                "--model" => {
                    let (id, path) = value.split_once('=').ok_or_else(|| {
                        ServiceError::new(400, "invalid_model", "--model requires ID=PATH")
                    })?;
                    config.models.push(YamlModel {
                        id: id.into(),
                        path: PathBuf::from(path),
                    });
                }
                "--listen" => config.listen = value.clone(),
                "--workers" => config.workers = parse_usize(value, flag)?,
                "--active-per-worker" => config.active_per_worker = parse_usize(value, flag)?,
                "--max-context" => config.max_context = parse_usize(value, flag)?,
                "--max-output" => config.max_output = parse_usize(value, flag)?,
                "--host-mib" => config.host_mib = parse_usize(value, flag)?,
                "--backend" => config.backend = value.clone(),
                "--source-profile" => config.source_profile = value.clone(),
                "--native-artifacts" => config.native_artifacts = Some(PathBuf::from(value)),
                "--provider-mib" => config.provider_mib = parse_usize(value, flag)?,
                "--device-mib" => config.device_mib = parse_usize(value, flag)?,
                "--batch-mode" => config.batch_mode = value.clone(),
                "--batch-rows" => config.batch_rows = parse_usize(value, flag)?,
                "--native-batch-rows" => config.native_batch_rows = parse_usize(value, flag)?,
                "--max-pending-requests" => {
                    config.max_pending_requests = parse_optional_usize(value, flag)?
                }
                _ => {
                    return Err(ServiceError::new(
                        400,
                        "invalid_flag",
                        format!("unknown serve flag {flag}"),
                    ));
                }
            }
        }
        Ok(config)
    }
}
fn parse_optional_usize(value: &str, flag: &str) -> Result<Option<usize>, ServiceError> {
    if value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("null") {
        return Ok(None);
    }
    parse_usize(value, flag).map(Some)
}
fn parse_usize(value: &str, flag: &str) -> Result<usize, ServiceError> {
    value.parse::<usize>().map_err(|_| {
        ServiceError::new(
            400,
            "invalid_integer",
            format!("invalid integer for {flag}"),
        )
    })
}
impl YamlConfig {
    fn profile(&self) -> zerogravity::serving::TextProfile {
        zerogravity::serving::TextProfile {
            workers: self.workers,
            active_per_worker: self.active_per_worker,
            max_context: self.max_context,
            max_output: self.max_output,
            host_mib: self.host_mib,
            backend: self.backend.clone(),
            source_profile: self.source_profile.clone(),
            native_artifacts: self.native_artifacts.clone(),
            provider_mib: self.provider_mib,
            device_mib: self.device_mib,
            batch_mode: self.batch_mode.clone(),
            batch_rows: self.batch_rows,
            native_batch_rows: self.native_batch_rows,
            max_pending_requests: self.max_pending_requests,
        }
    }
}

pub(super) fn help() {
    println!(
        "USAGE: zerogravity serve MODEL... | --model ID=MODEL [--model ID=MODEL ...]\n\
             zerogravity serve [--config FILE | --profile NAME] [MODEL...]\n\
             Canonical routes: local GGUF or cached model names/digests/provider locators.\n\
             --backend auto|cpu|cuda|rocm|vulkan, --device ID, --headroom LIMIT, --pull, --offline.\n\
             Direct GGUF CPU or one approved original GPU source shared by aliases.\n\
             Original GPU: --numerical source|native (native ROCm), --cohort-width auto|0|4|8.\n\
             Original ROCm: optional --device-mib auto|MiB sets an exact admission envelope within live headroom.\n\
             Legacy explicit local/diagnostic serving: --model ID=MODEL.gguf [--listen 127.0.0.1:8000] \
             [--backend cpu|cuda|vulkan-nvidia|vulkan-amd|rocm] \
             [--source-profile gguf|fixed-prompt-bf16|qwen35-cuda-bf16] [--native-artifacts DIR] \
             [--workers 2] [--active-per-worker 2] \
             [--max-context 1024] [--max-output 128] \
             [--host-mib 4096|8192] [--provider-mib 2048] [--device-mib 4096] \
             [--batch-mode interleaved|fused] [--batch-rows 2|4|8] \
             [--max-pending-requests N|none] \
             [--enable-per-request-metrics] [--enable-error-details]\n\
             ENDPOINTS (eligible loaded models only):\n\
             GET /health, GET /v1/models[/ID].\n\
             POST /v1/responses (text, streaming, resources), /v1/responses/input_tokens.\n\
             /v1/conversations and response resources (bounded listener-local state).\n\
             POST /v1/chat/completions, /v1/completions (text).\n\
             POST /v1/embeddings, /v1/audio/transcriptions, /v1/audio/translations \
             (explicitly eligible SDK adapters; no built-in public media grant).\n\
             WebSocket /v1/realtime?model=ID (qualified CPU text).\n\
             Speech synthesis, image generation, audio Realtime and WebRTC unavailable.\n\
             Qwen3.5 greedy text CPU or resident GPU, no inference fallback. \
             GPU: workers=1, context=512. Default interleaved; fused enables \
             CUDA 2/4/8-row compatible prompt tails and independent decode frontiers \
             with active>=rows. Default rows=2; survivors regroup without padding. \
             Prefill groups require equal frontiers; unmatched rows run as singletons. \
             CUDA needs CUDA_PATH with runtime+CCCL headers. \
             Opt-in metrics permit return_token_ids=true with bounded token IDs, \
             publication-observed times and separate HTTP delivery observations. \
             Loopback only; optional ZEROGRAVITY_API_KEY."
    );
}

pub fn run(args: &[String]) -> Result<(), ServiceError> {
    if args.len() == 1 && args[0] == "--help" {
        help();
        return Ok(());
    }

    // Install before config/source loading, including when serving is PID 1.
    // Keep the guard alive until all HTTP and inference workers have joined.
    let mut shutdown = shutdown::Shutdown::install()
        .map_err(|error| ServiceError::new(500, "signal_handler", error.to_string()))?;
    match canonical::Invocation::parse(args)? {
        canonical::Invocation::Legacy(options) => legacy(*options, &mut shutdown),
        canonical::Invocation::Canonical(options) => canonical::run(options, &mut shutdown),
    }
}

pub(crate) struct Listener {
    address: SocketAddr,
    key: Option<String>,
    metrics: bool,
    error_details: bool,
}
impl Listener {
    pub(crate) fn new(
        listen: &str,
        metrics: bool,
        error_details: bool,
    ) -> Result<Self, ServiceError> {
        let address: SocketAddr = listen.parse().map_err(|_| {
            ServiceError::new(400, "invalid_listen", "--listen requires numeric IP:PORT")
        })?;
        if !address.ip().is_loopback() {
            return Err(ServiceError::new(
                403,
                "remote_bind_unsupported",
                "HTTP service is loopback-only; expose through a TLS/auth reverse proxy",
            ));
        }
        let key = std::env::var("ZEROGRAVITY_API_KEY")
            .ok()
            .filter(|v| !v.is_empty());
        zerogravity::serving::auth::validate_key(key.as_deref())
            .map_err(ServiceError::inference)?;
        if error_details && key.is_none() {
            return Err(ServiceError::new(
                400,
                "invalid_config",
                "--enable-error-details requires ZEROGRAVITY_API_KEY; use only for trusted local debugging",
            ));
        }
        Ok(Self {
            address,
            key,
            metrics,
            error_details,
        })
    }
    pub(crate) fn options(self) -> zerogravity::control::ServeOptions {
        zerogravity::control::ServeOptions {
            address: self.address,
            api_key: self.key,
            limits: zerogravity::serving::ingress::IngressLimits {
                connections: 64,
                ..Default::default()
            },
            enable_per_request_metrics: self.metrics,
            enable_error_details: self.error_details,
            prefix_sharing_scope: None,
        }
    }
}

fn legacy(options: YamlConfig, shutdown: &mut shutdown::Shutdown) -> Result<(), ServiceError> {
    let listener = Listener::new(
        &options.listen,
        options.enable_per_request_metrics,
        options.enable_error_details,
    )?;
    let profile = options.profile();
    profile.validate()?;
    let max_active = options
        .workers
        .checked_mul(options.active_per_worker)
        .ok_or_else(|| {
            ServiceError::new(
                400,
                "invalid_config",
                "text scheduler exceeds runtime limits",
            )
        })?;
    launch(
        shutdown,
        listener,
        max_active,
        options.max_pending_requests,
        |control| {
            control.load(zerogravity::models::ModelLoadRequest::ServingText {
                declared: options.models.into_iter().map(|m| (m.id, m.path)).collect(),
                profile,
            })
        },
    )
}

/// Foreground ownership is shared by canonical and legacy commands. Stop
/// ingress and join handlers before retiring original inference workers.
fn launch(
    shutdown: &mut shutdown::Shutdown,
    listener: Listener,
    max_active: usize,
    max_pending: Option<usize>,
    load: impl FnOnce(zerogravity::control::ControlPlane<'_>) -> Result<(), ServiceError>,
) -> Result<(), ServiceError> {
    if shutdown.is_requested() {
        return Ok(());
    }
    let runtime = zerogravity::Runtime::builder()
        .limits(zerogravity::runtime::SchedulerLimits {
            max_active,
            max_queued: max_pending,
            ..Default::default()
        })
        .build()
        .map_err(|error| ServiceError::new(400, "invalid_config", error.to_string()))?;
    let control = runtime.control();
    let result = (|| {
        crate::output::emit!(Status, "zerogravity: loading selected model routes");
        load(control)?;
        shutdown.admit(&runtime)?;
        if shutdown.is_requested() {
            return Ok(());
        }
        // Get actual original model/worker facts before announcing listener readiness.
        let snapshot = control.snapshot().map_err(ServiceError::inference)?;
        let binding = control
            .start_serving(listener.options())
            .map_err(|error| ServiceError::new(500, "bind_failed", error.to_string()))?;
        // A monitoring/presentation error must still stop the exact listener.
        let waited = {
            if !shutdown.is_requested() {
                for model in &snapshot.runtime.catalog {
                    let details = model.capabilities.as_ref().map(|caps| &caps.details);
                    let backend = details
                        .and_then(|d| d.execution.as_ref())
                        .map(|e| format!("{:?}", e.selected.provider))
                        .unwrap_or_else(|| "unreported".into());
                    let device = details.and_then(|d| d.memory.as_ref()).and_then(|memory| {
                        use zerogravity::models::ModelMemory;
                        match memory {
                            ModelMemory::Resident(memory) => Some(memory.device.device.as_str()),
                            ModelMemory::ResidentText { admission, .. } => {
                                Some(admission.device.device.as_str())
                            }
                            _ => None,
                        }
                    });
                    let device = device.or((backend == "Cpu").then_some("cpu:0"));
                    crate::output::emit!(
                        Status,
                        "Model {} · {}{}",
                        model.aliases.join(", "),
                        backend,
                        device.map_or_else(String::new, |device| format!(" · {device}"))
                    );
                }
                crate::output::emit!(Status, "API http://{}/v1", binding.address);
                if let Some(scheduler) = &snapshot.runtime.scheduler {
                    crate::output::emit!(
                        Status,
                        "zerogravity: serving on {}; max-context {}, max-output {}, active {}, Ctrl-C to stop",
                        binding.address,
                        scheduler.config.context,
                        scheduler.config.output,
                        scheduler.active_limit
                    );
                } else {
                    crate::output::emit!(
                        Status,
                        "zerogravity: serving on {}; Ctrl-C to stop",
                        binding.address
                    );
                }
            }
            shutdown.wait_control(control, &runtime, binding)
        };
        let stopped = control
            .stop_serving(binding.generation)
            .map_err(|error| ServiceError::new(500, "http_error", error.to_string()));
        combine(waited, stopped, "listener retirement")
    })();
    let joined = runtime
        .shutdown()
        .map_err(|error| ServiceError::new(500, "shutdown_failed", error.to_string()));
    let result = combine(result, joined, "source retirement");
    if result.is_ok() {
        crate::output::emit!(
            Status,
            "zerogravity: shutdown complete; HTTP handlers and inference workers joined"
        );
    }
    result
}

fn combine(
    first: Result<(), ServiceError>,
    second: Result<(), ServiceError>,
    phase: &str,
) -> Result<(), ServiceError> {
    match (first, second) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(error), Err(close)) => Err(ServiceError::new(
            error.status,
            error.code,
            format!("{}; {phase}: {}", error.message, close.message),
        )),
    }
}

#[cfg(test)]
fn join_router(router: Arc<Router>) -> std::io::Result<()> {
    let mut router = Arc::try_unwrap(router)
        .map_err(|_| std::io::Error::other("HTTP handler retained the serving router"))?;
    router.shutdown()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_profile_defaults_and_finite_hf_flags_are_cold() {
        let old = YamlConfig::cli(&[]).unwrap();
        assert_eq!(old.source_profile, "gguf");
        assert!(old.native_artifacts.is_none());
        assert!(old.profile().validate().is_ok());
        let args = [
            "--source-profile",
            "fixed-prompt-bf16",
            "--native-artifacts",
            "/bundle",
            "--backend",
            "cuda",
            "--workers",
            "1",
            "--active-per-worker",
            "2",
            "--max-context",
            "512",
            "--max-output",
            "8",
            "--provider-mib",
            "2304",
            "--device-mib",
            "5120",
            "--host-mib",
            "3072",
        ]
        .map(str::to_owned);
        let mut cfg = YamlConfig::cli(&args).unwrap();
        if cfg!(all(
            feature = "qwen35-cuda-serving",
            feature = "qwen35-cuda-qualification",
            target_os = "linux"
        )) {
            assert!(cfg.profile().validate().is_ok());
        } else {
            assert_eq!(
                cfg.profile().validate().unwrap_err().code,
                "backend_not_compiled"
            );
        }
        cfg.active_per_worker = 3;
        assert_eq!(
            cfg.profile().validate().unwrap_err().code,
            "unsupported_fixed-prompt_profile"
        );
        cfg.active_per_worker = 1;
        cfg.max_output = 9;
        assert!(cfg.profile().validate().is_err());
        cfg.max_output = 8;
        cfg.max_context = 513;
        assert!(cfg.profile().validate().is_err());
        cfg.max_context = 512;
        cfg.batch_mode = "fused".into();
        assert!(cfg.profile().validate().is_err());
        cfg.source_profile = "gguf".into();
        cfg.batch_mode = "interleaved".into();
        assert_eq!(
            cfg.profile().validate().unwrap_err().code,
            "unsupported_source_profile"
        );
    }

    #[test]
    fn qwen35_cuda_native_four_opt_in_is_reachable_and_bounded_before_source_load() {
        let args = [
            "--model",
            "original=/not-opened",
            "--source-profile",
            "qwen35-cuda-bf16",
            "--backend",
            "cuda",
            "--workers",
            "1",
            "--active-per-worker",
            "8",
            "--max-context",
            "512",
            "--max-output",
            "8",
            "--host-mib",
            "4096",
            "--provider-mib",
            "4096",
            "--device-mib",
            "5120",
            "--batch-mode",
            "interleaved",
            "--native-artifacts",
            "/not-opened-native",
            "--native-batch-rows",
            "4",
        ]
        .map(str::to_owned);
        let mut cfg = YamlConfig::cli(&args).unwrap();
        assert_eq!(cfg.native_batch_rows, 4);
        let expect_supported = |cfg: &YamlConfig| {
            if cfg!(all(feature = "qwen35-cuda-serving", target_os = "linux")) {
                assert!(cfg.profile().validate().is_ok());
            } else {
                assert_eq!(
                    cfg.profile().validate().unwrap_err().code,
                    "backend_not_compiled"
                );
            }
        };
        expect_supported(&cfg);
        cfg.native_batch_rows = 0;
        expect_supported(&cfg);
        for width in [1, 2, 8] {
            cfg.native_batch_rows = width;
            assert_eq!(
                cfg.profile().validate().unwrap_err().code,
                "unsupported_native_batch_profile"
            );
        }
        cfg.native_batch_rows = 4;
        cfg.active_per_worker = 3;
        assert!(cfg.profile().validate().is_err());
        cfg.active_per_worker = 8;
        cfg.batch_mode = "fused".into();
        assert!(cfg.profile().validate().is_err());
        cfg.batch_mode = "interleaved".into();
        for backend in ["cpu", "rocm", "vulkan-nvidia"] {
            cfg.backend = backend.into();
            assert!(cfg.profile().validate().is_err());
        }
        cfg.backend = "cuda".into();
        for source in ["gguf", "fixed-prompt-bf16"] {
            cfg.source_profile = source.into();
            assert!(cfg.profile().validate().is_err());
        }
    }

    #[test]
    fn serve_cli_is_bounded_and_configured_only_explicitly() {
        let cfg = YamlConfig::cli(&[
            "--model".into(),
            "a=weights.gguf".into(),
            "--workers".into(),
            "3".into(),
        ])
        .unwrap();
        assert_eq!(cfg.models.len(), 1);
        assert_eq!(cfg.workers, 3);
        assert_eq!(cfg.backend, "cpu");
        assert!(!cfg.enable_per_request_metrics);
        let observed = YamlConfig::cli(&[
            "--enable-per-request-metrics".into(),
            "--model".into(),
            "a=weights.gguf".into(),
        ])
        .unwrap();
        assert!(observed.enable_per_request_metrics);
        assert!(YamlConfig::cli(&["--model".into(), "missing".into()]).is_err());
        assert!(YamlConfig::cli(&["--workers".into(), "no".into()]).is_err());
        assert_eq!(
            YamlConfig::cli(&["--max-pending-requests".into(), "none".into()])
                .unwrap()
                .max_pending_requests,
            None
        );
        assert_eq!(
            YamlConfig::cli(&["--max-pending-requests".into(), "12".into()])
                .unwrap()
                .max_pending_requests,
            Some(12)
        );
        assert!(YamlConfig::cli(&["--max-pending-requests".into(), "0".into()]).is_ok());
        assert!(YamlConfig::cli(&["--max-pending-requests".into(), "nope".into()]).is_err());
        let null_pending: YamlConfig =
            serde_yaml::from_str("listen: 127.0.0.1:8000\nmodels: []\nmax_pending_requests: null")
                .unwrap();
        assert_eq!(null_pending.max_pending_requests, None);
        let none_pending: YamlConfig =
            serde_yaml::from_str("listen: 127.0.0.1:8000\nmodels: []\nmax_pending_requests: none")
                .unwrap();
        assert_eq!(none_pending.max_pending_requests, None);
        let counted_pending: YamlConfig =
            serde_yaml::from_str("listen: 127.0.0.1:8000\nmodels: []\nmax_pending_requests: 7")
                .unwrap();
        assert_eq!(counted_pending.max_pending_requests, Some(7));
        assert!(
            serde_yaml::from_str::<YamlConfig>(
                "listen: 127.0.0.1:8000\nmodels: []\nmax_pending_requests: -1",
            )
            .is_err()
        );
        assert!(
            YamlConfig::cli(&[
                "--config".into(),
                "config.yaml".into(),
                "--workers".into(),
                "2".into()
            ])
            .is_err()
        );
        let example = include_str!("../../../../examples/serving/qwen-cpu.yaml");
        let parsed: YamlConfig = serde_yaml::from_str(example).unwrap();
        assert_eq!(parsed.listen, "127.0.0.1:8000");
        assert_eq!(parsed.models.len(), 2);
        assert_eq!(parsed.models[0].path, parsed.models[1].path);
        assert_eq!((parsed.workers, parsed.active_per_worker), (2, 2));
        assert_eq!(parsed.host_mib, 4096);
        let gpu_config = include_str!("../../../../examples/serving/qwen-cuda.yaml");
        let gpu: YamlConfig = serde_yaml::from_str(gpu_config).unwrap();
        assert_eq!(gpu.backend, "cuda");
        assert_eq!(gpu.batch_mode, "interleaved");
        assert_eq!(gpu.max_context, 512);
        assert_eq!((gpu.workers, gpu.active_per_worker), (1, 2));
        assert_eq!((gpu.provider_mib, gpu.device_mib), (2048, 4096));
        assert_eq!(gpu.models.len(), 2);
        assert!(
            serde_yaml::from_str::<YamlConfig>(
                "listen: 127.0.0.1:8080\\nmodels: []\\nunknown_provider: yes"
            )
            .is_err()
        );
        let err = run(&[
            "--listen".into(),
            "0.0.0.0:9911".into(),
            "--model".into(),
            "unqualified=missing.gguf".into(),
        ])
        .unwrap_err();
        assert_eq!(err.code, "remote_bind_unsupported");
        let unsupported = run(&["--batch-mode".into(), "fused".into()]).unwrap_err();
        assert_eq!(unsupported.code, "unsupported_batch_mode");
        // Refuse uncompiled GPU backend before initializing a device.
        #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "hip")))]
        {
            let missing = run(&[
                "--backend".into(),
                "cuda".into(),
                "--model".into(),
                "unqualified=missing.gguf".into(),
            ])
            .unwrap_err();
            assert_eq!(missing.code, "backend_not_compiled");
        }
    }
}
