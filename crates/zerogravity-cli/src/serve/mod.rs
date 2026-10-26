//! Opt-in local OpenAI-compatible HTTP serving.
//! Uses original Qwen CPU source and private independent model sessions.

mod engine;
mod event_channel;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
mod gpu_batch;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
mod gpu_engine;
mod http;
mod pending;
mod token_metrics;

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
use crate::GpuBackend;
use engine::{Request, RequestEvents, ServiceError};

enum Router {
    Cpu(engine::Router),
    #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
    Gpu(gpu_engine::GpuRouter),
}
impl Router {
    fn request_admitted(
        &self,
        request: Request,
        ingress: Vec<pending::ServingCharge>,
    ) -> Result<RequestEvents, ServiceError> {
        match self {
            Self::Cpu(v) => v.request_admitted(request, ingress),
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => v.request_admitted(request, ingress),
        }
    }
    fn max_output(&self) -> usize {
        match self {
            Self::Cpu(v) => v.max_output,
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => v.max_output,
        }
    }
    fn charge_ingress(
        &self,
        session: &str,
        bytes: u64,
    ) -> Result<pending::ServingCharge, ServiceError> {
        match self {
            Self::Cpu(v) => v.charge_ingress(session, bytes),
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => v.charge_ingress(session, bytes),
        }
    }
    fn model_ids(&self) -> Box<dyn Iterator<Item = &str> + '_> {
        match self {
            Self::Cpu(v) => Box::new(v.model_ids()),
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => Box::new(v.model_ids()),
        }
    }
    fn admission(&self) -> serde_json::Value {
        match self {
            Self::Cpu(v) => {
                let (limit, reserved, remaining, serving) =
                    v.memory_snapshot().unwrap_or((0, 0, 0, 0));
                let model = reserved.checked_sub(serving);
                let queue = v.queue_snapshot();
                serde_json::json!({"kind":"configured_host_cpu",
                    "host_limit_bytes":limit,
                    "total_host_limit_bytes":limit,
                    "total_host_reserved_bytes":reserved,
                    "reserved_bytes":reserved,
                    "remaining_bytes":remaining,
                    "model_host_reserved_bytes":model,
                    "backend_host_reserved_bytes":0,
                    "serving_host_reserved_bytes":serving,
                    "queued_requests":queue.pending,
                    "pending_admission_in_progress":queue.in_progress,
                    "pending_queue_capacity":queue.capacity,
                    "pending_queue_charged_bytes":queue.backing_bytes,
                    "pending_queue_consumers":queue.consumers,
                    "queued_requests_limit":queue.limit,
                    "active_sessions":v.active_sessions(),
                    "peak_active_sessions":v.peak_active_sessions(),
                    "active_sessions_limit":v.active_limit()})
            }
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => {
                let m = v.metrics();
                let mut value = serde_json::json!({"kind":"resident_gpu",
                    "physical_accounting":"ResidentService/ResourceArbiter",
                    "batch_mode":if m.native_rows != 0 {"fused"} else {"interleaved"},
                    "native_batch_rows_max":m.native_rows.max(1),
                    "gpu_step_attempts":m.attempted,
                    "completed_gpu_quanta":m.completed,
                    "completed_quantum_unit":"request work, including terminal presentation",
                    "gpu_forward_steps_completed":m.gpu_forwards,
                    "gpu_forward_calls_completed":m.gpu_forwards,
                    "completed_model_token_steps":m.model_tokens,
                    "completed_forward_kernel_dispatches":m.kernel_dispatches,
                    "committed_gpu_roots":m.committed_roots,
                    "native_projection_dispatches_completed":m.native_projections,
                    "native_sampler_dispatches_completed":m.native_sampler,
                    "gpu_fused_rows_completed":m.fused_rows,
                    "gpu_fused_state_only_batches":m.fused_state_only_batches,
                    "gpu_fused_state_only_rows":m.fused_state_only_rows,
                    "gpu_fused_greedy_batches":m.fused_greedy_batches,
                    "gpu_fused_batches_by_rows":{"2":m.native_bands[0],"4":m.native_bands[1],"8":m.native_bands[2]},
                    "native_admission_refusals":m.native_admission_refusals,
                    "discarded_ready_cohorts":m.discarded_cohorts,
                    "accepted_output_tokens":m.output_tokens,
                    "output_credit_parked_requests":m.parked,
                    "failed_gpu_quanta":m.failed,
                    "cancelled_requests":m.cancelled});
                let tail = serde_json::json!({
                    "active_sessions":m.active,
                    "peak_active_sessions":m.peak_active,
                    "active_sessions_limit":m.active_limit,
                    "fixed-prompt_resources":m.canonical_resources,
                    "hf_source":m.hf_source,"hf_resources":m.hf_resources,
                    "queued_requests":m.queue.pending,
                    "pending_admission_in_progress":m.queue.in_progress,
                    "pending_queue_capacity":m.queue.capacity,
                    "pending_queue_charged_bytes":m.queue.backing_bytes,
                    "pending_queue_consumers":m.queue.consumers,
                    "queued_requests_limit":m.queue.limit,
                    "total_host_limit_bytes":m.host.map(|h|h.0),
                    "total_host_reserved_bytes":m.host.map(|h|h.1),
                    "model_host_reserved_bytes":m.host.map(|h|h.2),
                    "backend_host_reserved_bytes":m.host.map(|h|h.3),
                    "serving_host_reserved_bytes":m.host.map(|h|h.4),
                    "scheduling_groups":m.groups,
                    "co_scheduled_members":m.co_scheduled,
                    "state_cohort_publications":m.cohort_publications,
                    "state_cohort_rows":m.cohort_rows,
                    "gpu_fused_batches":m.fused_batches});
                if let (Some(target), Some(source)) = (value.as_object_mut(), tail.as_object()) {
                    target.extend(source.iter().map(|(key, item)| (key.clone(), item.clone())));
                }
                value
            }
        }
    }
}

use serde::{Deserialize, Deserializer};
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
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
    2
}
fn default_active() -> usize {
    2
}
fn default_context() -> usize {
    1024
}
fn default_output() -> usize {
    128
}
fn default_host() -> usize {
    4096
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
            let raw = std::fs::read(path).map_err(|_| {
                ServiceError::new(400, "config_unreadable", "cannot read serving YAML")
            })?;
            if raw.is_empty() || raw.len() > 64 * 1024 {
                return Err(ServiceError::new(
                    400,
                    "config_limit",
                    "serving YAML exceeds 64 KiB",
                ));
            }
            let mut parsed: Self = serde_yaml::from_slice(&raw).map_err(|_| {
                ServiceError::new(400, "config_invalid", "invalid typed serving YAML")
            })?;
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
            return Ok(parsed);
        }
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            if flag == "--enable-per-request-metrics" {
                config.enable_per_request_metrics = true;
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
fn validate_batch_mode(
    mode: &str,
    backend: &str,
    active: usize,
    rows: usize,
) -> Result<(), ServiceError> {
    match mode {
        "interleaved" => Ok(()),
        "fused" if backend == "cuda" && matches!(rows, 2 | 4 | 8) && active >= rows => Ok(()),
        _ => Err(ServiceError::new(
            400,
            "unsupported_batch_mode",
            "use interleaved, or fused with CUDA, rows=2/4/8 and active>=rows",
        )),
    }
}

fn fixed_prompt_owner_profile(active: usize, host: usize, provider: usize) -> bool {
    matches!(
        (provider, host, active),
        (2304, 3072..=16384, 1..=2) | (4096, 4096, 1..=8)
    )
}

fn qwen35_cuda_host_profile(host_mib: usize) -> bool {
    matches!(host_mib, 4096 | 8192)
}

fn validate_source_profile(options: &YamlConfig) -> Result<(), ServiceError> {
    if options.native_batch_rows != 0
        && (options.source_profile != "qwen35-cuda-bf16"
            || options.backend != "cuda"
            || options.batch_mode != "interleaved"
            || options.native_batch_rows != 4
            || options.active_per_worker < 4)
    {
        return Err(ServiceError::new(
            400,
            "unsupported_native_batch_profile",
            "native-batch-rows=4 requires qwen35-cuda-bf16/CUDA/interleaved/active>=4; default0",
        ));
    }
    match options.source_profile.as_str() {
        "gguf" if options.native_artifacts.is_none() => Ok(()),
        "fixed-prompt-bf16" => {
            if options.backend != "cuda"
                || options.workers != 1
                || options.max_context != 512
                || !(1..=8).contains(&options.max_output)
                || options.batch_mode != "interleaved"
                || options.device_mib != 5120
                || !fixed_prompt_owner_profile(
                    options.active_per_worker,
                    options.host_mib,
                    options.provider_mib,
                )
            {
                return Err(ServiceError::new(
                    400,
                    "unsupported_fixed-prompt_profile",
                    "fixed-prompt requires CUDA/1worker/context512/maxoutput1..8/interleaved/device5120 and finite2 or exact4GiB active8 owner profile",
                ));
            }
            if !cfg!(all(feature = "qwen35-cuda-serving", target_os = "linux")) {
                return Err(ServiceError::new(
                    400,
                    "backend_not_compiled",
                    "compile Linux qwen35-cuda-serving for original HF native serving",
                ));
            }
            Ok(())
        }
        "qwen35-cuda-bf16" => {
            if options.backend != "cuda"
                || options.workers != 1
                || options.max_context != 512
                || !(1..=8).contains(&options.max_output)
                || !(1..=8).contains(&options.active_per_worker)
                || options.batch_mode != "interleaved"
                || !qwen35_cuda_host_profile(options.host_mib)
                || options.provider_mib == 0
                || options.provider_mib > options.device_mib
                || options.device_mib != 5120
                || options.native_artifacts.is_none()
            {
                return Err(ServiceError::new(
                    400,
                    "unsupported_qwen35_cuda_profile",
                    "Qwen CUDA requires CUDA/1worker/context512/maxoutput1..8/active1..8/interleaved/4096or8192host/provider1..5120/5120device and actual native-artifacts",
                ));
            }
            if !cfg!(all(feature = "qwen35-cuda-serving", target_os = "linux")) {
                return Err(ServiceError::new(
                    400,
                    "backend_not_compiled",
                    "compile Linux qwen35-cuda-serving",
                ));
            }
            Ok(())
        }
        _ => Err(ServiceError::new(
            400,
            "unsupported_source_profile",
            "source-profile is gguf, fixed-prompt-bf16 or qwen35-cuda-bf16; native-artifacts is HF-only",
        )),
    }
}

pub fn run(args: &[String]) -> Result<(), ServiceError> {
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "USAGE: zerogravity serve --config config.yaml | \
             --model ID=MODEL.gguf [--listen 127.0.0.1:8000] \
             [--backend cpu|cuda|vulkan-nvidia|vulkan-amd|rocm] \
             [--source-profile gguf|fixed-prompt-bf16|qwen35-cuda-bf16] [--native-artifacts DIR] \
             [--workers 2] [--active-per-worker 2] \
             [--max-context 1024] [--max-output 128] \
             [--host-mib 4096|8192] [--provider-mib 2048] [--device-mib 4096] \
             [--batch-mode interleaved|fused] [--batch-rows 2|4|8] \
             [--max-pending-requests N|none] \
             [--enable-per-request-metrics]\n\
             ENDPOINTS: GET /health, GET /v1/models, \
             POST /v1/chat/completions, POST /v1/completions.\n\
             Qwen3.5 greedy text CPU or resident GPU, no inference fallback. \
             GPU: workers=1, context=512. Default interleaved; fused enables \
             CUDA B2/B4/B8 compatible prompt tails and independent decode frontiers \
             with active>=rows. Default rows=2; survivors regroup without padding. \
             Prefill groups require equal frontiers; unmatched rows use B1. \
             CUDA needs CUDA_PATH with runtime+CCCL headers. \
             Opt-in metrics permit return_token_ids=true with bounded token IDs, \
             publication-observed times and separate HTTP delivery observations. \
             Loopback only; optional ZEROGRAVITY_API_KEY."
        );
        return Ok(());
    }

    let options = YamlConfig::cli(args)?;
    let address: SocketAddr = options.listen.parse().map_err(|_| {
        ServiceError::new(400, "invalid_listen", "--listen requires numeric IP:PORT")
    })?;
    let key = std::env::var("ZEROGRAVITY_API_KEY")
        .ok()
        .filter(|v| !v.is_empty());
    // No native TLS; a Bearer token over unencrypted non-loopback
    // sockets must not be mistaken for secure remote authentication.
    // Use a trusted TLS reverse proxy forwarding to local loopback.
    if !address.ip().is_loopback() {
        return Err(ServiceError::new(
            403,
            "remote_bind_unsupported",
            "HTTP service is loopback-only; expose through a TLS/auth reverse proxy",
        ));
    }
    if options.max_pending_requests == Some(0) {
        return Err(ServiceError::new(
            400,
            "invalid_pending_limit",
            "--max-pending-requests must be a positive count or none",
        ));
    }
    validate_batch_mode(
        &options.batch_mode,
        &options.backend,
        options.active_per_worker,
        options.batch_rows,
    )?;
    validate_source_profile(&options)?;
    let declared = options.models.into_iter().map(|m| (m.id, m.path)).collect();
    let backend = options.backend.as_str();
    let router = match backend {
        "cpu" => Router::Cpu(engine::Router::new(
            declared,
            options.workers,
            options.active_per_worker,
            options.max_context,
            options.max_output,
            options.host_mib,
            options.max_pending_requests,
        )?),
        "vulkan-nvidia" | "vulkan-amd" | "cuda" | "rocm" => {
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            {
                let selected = match backend {
                    "vulkan-nvidia" => GpuBackend::VulkanNvidia,
                    "vulkan-amd" => GpuBackend::VulkanAmd,
                    "cuda" => GpuBackend::Cuda,
                    _ => GpuBackend::Hip,
                };
                if !selected.compiled() {
                    return Err(ServiceError::new(
                        400,
                        "backend_not_compiled",
                        "selected GPU backend was not compiled into this binary",
                    ));
                }
                let profile = gpu_engine::GpuProfile {
                    workers: options.workers,
                    active: options.active_per_worker,
                    context: options.max_context,
                    output: options.max_output,
                    provider_mib: options.provider_mib,
                    device_mib: options.device_mib,
                    native_rows: if options.source_profile == "qwen35-cuda-bf16" {
                        options.native_batch_rows
                    } else if options.batch_mode == "fused" {
                        options.batch_rows
                    } else {
                        0
                    },
                    host_mib: options.host_mib,
                    max_pending: options.max_pending_requests,
                };
                if options.source_profile == "fixed-prompt-bf16" {
                    #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
                    {
                        Router::Gpu(gpu_engine::GpuRouter::new_canonical(
                            declared,
                            selected,
                            profile,
                            options.host_mib,
                            options.native_artifacts.as_deref(),
                        )?)
                    }
                    #[cfg(not(all(feature = "qwen35-cuda-serving", target_os = "linux")))]
                    {
                        return Err(ServiceError::new(
                            400,
                            "backend_not_compiled",
                            "qwen35-cuda-serving feature required",
                        ));
                    }
                } else if options.source_profile == "qwen35-cuda-bf16" {
                    #[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
                    {
                        Router::Gpu(gpu_engine::GpuRouter::new_qwen35_cuda(
                            declared,
                            selected,
                            profile,
                            options.host_mib,
                            options.native_artifacts.as_deref(),
                        )?)
                    }
                    #[cfg(not(all(feature = "qwen35-cuda-serving", target_os = "linux")))]
                    {
                        return Err(ServiceError::new(
                            400,
                            "backend_not_compiled",
                            "qwen35-cuda-serving feature required",
                        ));
                    }
                } else {
                    Router::Gpu(gpu_engine::GpuRouter::new(declared, selected, profile)?)
                }
            }
            #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "hip")))]
            {
                return Err(ServiceError::new(
                    400,
                    "backend_not_compiled",
                    "compile with --features serve,vulkan or serve,cuda or serve,hip",
                ));
            }
        }
        _ => {
            return Err(ServiceError::new(
                400,
                "invalid_backend",
                "backend must be cpu, vulkan-nvidia, vulkan-amd, cuda or rocm",
            ));
        }
    };
    let router = Arc::new(router);
    let listener = TcpListener::bind(address)
        .map_err(|e| ServiceError::new(500, "bind_failed", e.to_string()))?;
    eprintln!(
        "zerogravity: serving {:?} Qwen on {address}, {} worker(s); max-context {}, max-output {}",
        backend, options.workers, options.max_context, options.max_output
    );
    http::serve(listener, router, key, options.enable_per_request_metrics)
        .map_err(|e| ServiceError::new(500, "http_error", e.to_string()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn canonical_eight_profile_needs_all_three_exact_owner_settings() {
        for active in 1..=8 {
            assert!(super::fixed_prompt_owner_profile(active, 4096, 4096));
        }
        for (active, host, provider) in [
            (0, 4096, 4096),
            (9, 4096, 4096),
            (8, 3072, 4096),
            (8, 4095, 4096),
            (8, 4097, 4096),
            (8, 4096, 2304),
            (8, 4096, 4095),
            (8, 4096, 4097),
            (3, 4096, 2304),
        ] {
            assert!(!super::fixed_prompt_owner_profile(active, host, provider));
        }
        assert!(super::fixed_prompt_owner_profile(1, 3072, 2304));
        assert!(super::fixed_prompt_owner_profile(2, 16384, 2304));
    }

    use super::*;
    #[test]
    fn source_profile_defaults_and_finite_hf_flags_are_cold() {
        let old = YamlConfig::cli(&[]).unwrap();
        assert_eq!(old.source_profile, "gguf");
        assert!(old.native_artifacts.is_none());
        assert!(validate_source_profile(&old).is_ok());
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
        if cfg!(all(feature = "qwen35-cuda-serving", target_os = "linux")) {
            assert!(validate_source_profile(&cfg).is_ok());
        } else {
            assert_eq!(
                validate_source_profile(&cfg).unwrap_err().code,
                "backend_not_compiled"
            );
        }
        cfg.active_per_worker = 3;
        assert_eq!(
            validate_source_profile(&cfg).unwrap_err().code,
            "unsupported_fixed-prompt_profile"
        );
        cfg.active_per_worker = 1;
        cfg.max_output = 9;
        assert!(validate_source_profile(&cfg).is_err());
        cfg.max_output = 8;
        cfg.max_context = 513;
        assert!(validate_source_profile(&cfg).is_err());
        cfg.max_context = 512;
        cfg.batch_mode = "fused".into();
        assert!(validate_source_profile(&cfg).is_err());
        cfg.source_profile = "gguf".into();
        assert_eq!(
            validate_source_profile(&cfg).unwrap_err().code,
            "unsupported_source_profile"
        );
    }

    #[test]
    fn native_serving_mode_is_bounded_before_model_or_device_lookup() {
        for backend in ["cpu", "cuda", "vulkan-nvidia", "vulkan-amd", "rocm"] {
            assert!(validate_batch_mode("interleaved", backend, 1, 2).is_ok());
            if backend != "cuda" {
                let err = run(&[
                    "--backend".into(),
                    backend.into(),
                    "--batch-mode".into(),
                    "fused".into(),
                    "--model".into(),
                    "unqualified=missing.gguf".into(),
                ])
                .unwrap_err();
                assert_eq!(err.code, "unsupported_batch_mode");
            }
        }
        assert!(validate_batch_mode("fused", "cuda", 2, 2).is_ok());
        assert!(validate_batch_mode("fused", "cuda", 4, 4).is_ok());
        assert!(validate_batch_mode("fused", "cuda", 8, 8).is_ok());
        for rows in [0, 1, 3, 5, 6, 7, 9, 16] {
            assert!(validate_batch_mode("fused", "cuda", 8, rows).is_err());
        }
        assert!(validate_batch_mode("fused", "cuda", 4, 8).is_err());
        for mode in ["fused", "fused-b4", "unknown"] {
            let err = run(&[
                "--backend".into(),
                "cuda".into(),
                "--batch-mode".into(),
                mode.into(),
                "--active-per-worker".into(),
                "1".into(),
                "--model".into(),
                "unqualified=missing.gguf".into(),
            ])
            .unwrap_err();
            assert_eq!(err.code, "unsupported_batch_mode");
        }
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
                assert!(validate_source_profile(cfg).is_ok());
            } else {
                assert_eq!(
                    validate_source_profile(cfg).unwrap_err().code,
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
                validate_source_profile(&cfg).unwrap_err().code,
                "unsupported_native_batch_profile"
            );
        }
        cfg.native_batch_rows = 4;
        cfg.active_per_worker = 3;
        assert!(validate_source_profile(&cfg).is_err());
        cfg.active_per_worker = 8;
        cfg.batch_mode = "fused".into();
        assert!(validate_source_profile(&cfg).is_err());
        cfg.batch_mode = "interleaved".into();
        for backend in ["cpu", "rocm", "vulkan-nvidia"] {
            cfg.backend = backend.into();
            assert!(validate_source_profile(&cfg).is_err());
        }
        cfg.backend = "cuda".into();
        for source in ["gguf", "fixed-prompt-bf16"] {
            cfg.source_profile = source.into();
            assert!(validate_source_profile(&cfg).is_err());
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
            let missing = run(&["--backend".into(), "cuda".into()]).unwrap_err();
            assert_eq!(missing.code, "backend_not_compiled");
        }
    }
}
