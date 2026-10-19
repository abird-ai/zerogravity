//! Opt-in local OpenAI-compatible HTTP serving.
//! Uses original Qwen CPU source and private independent model sessions.

mod engine;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
mod gpu_batch;
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
mod gpu_engine;
mod http;

#[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
use crate::GpuBackend;
use engine::{Request, RequestEvents, ServiceError};

enum Router {
    Cpu(engine::Router),
    #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
    Gpu(gpu_engine::GpuRouter),
}
impl Router {
    fn request(&self, request: Request) -> Result<RequestEvents, ServiceError> {
        match self {
            Self::Cpu(v) => v.request(request),
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => v.request(request),
        }
    }
    fn max_output(&self) -> usize {
        match self {
            Self::Cpu(v) => v.max_output,
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => v.max_output,
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
                let (limit, reserved, remaining) = v.memory_snapshot().unwrap_or((0, 0, 0));
                serde_json::json!({"kind":"configured_host_cpu",
                    "host_limit_bytes":limit,"reserved_bytes":reserved,
                    "remaining_bytes":remaining})
            }
            #[cfg(any(feature = "vulkan", feature = "cuda", feature = "hip"))]
            Self::Gpu(v) => {
                let m = v.metrics();
                serde_json::json!({"kind":"resident_gpu",
                    "physical_accounting":"ResidentService/ResourceArbiter",
                    "batch_mode":"interleaved",
                    "gpu_step_attempts":m.attempted,
                    "completed_gpu_quanta":m.completed,
                    "gpu_forward_steps_completed":m.gpu_forwards,
                    "failed_gpu_quanta":m.failed,
                    "cancelled_requests":m.cancelled,
                    "queued_requests":m.queued,
                    "active_sessions":m.active,
                    "scheduling_groups":m.groups,
                    "co_scheduled_members":m.co_scheduled,
                    "state_cohort_publications":m.cohort_publications,
                    "state_cohort_rows":m.cohort_rows,
                    "gpu_fused_batches":0})
            }
        }
    }
}

use serde::Deserialize;
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
    #[serde(default = "default_provider_mib")]
    provider_mib: usize,
    #[serde(default = "default_device_mib")]
    device_mib: usize,
    #[serde(default = "default_batch_mode")]
    batch_mode: String,
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
fn default_backend() -> String {
    "cpu".into()
}
fn default_provider_mib() -> usize {
    2048
}
fn default_device_mib() -> usize {
    4096
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
            provider_mib: default_provider_mib(),
            device_mib: default_device_mib(),
            batch_mode: default_batch_mode(),
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
            return Ok(parsed);
        }
        let mut it = args.iter();
        while let Some(flag) = it.next() {
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
                "--provider-mib" => config.provider_mib = parse_usize(value, flag)?,
                "--device-mib" => config.device_mib = parse_usize(value, flag)?,
                "--batch-mode" => config.batch_mode = value.clone(),
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
fn parse_usize(value: &str, flag: &str) -> Result<usize, ServiceError> {
    value.parse::<usize>().map_err(|_| {
        ServiceError::new(
            400,
            "invalid_integer",
            format!("invalid integer for {flag}"),
        )
    })
}
pub fn run(args: &[String]) -> Result<(), ServiceError> {
    if args.len() == 1 && args[0] == "--help" {
        println!(
            "USAGE: zerogravity serve --config config.yaml | \
             --model ID=MODEL.gguf [--listen 127.0.0.1:8000] \
             [--backend cpu|cuda|vulkan-nvidia|vulkan-amd|rocm] \
             [--workers 2] [--active-per-worker 2] \
             [--max-context 1024] [--max-output 128] \
             [--host-mib 4096] [--provider-mib 2048] [--device-mib 4096] \
             [--batch-mode interleaved]\n\
             ENDPOINTS: GET /health, GET /v1/models, \
             POST /v1/chat/completions, POST /v1/completions.\n\
             Qwen3.5 greedy text CPU or resident GPU, no inference fallback. \
             GPU: workers=1, context=512; source-compatible requests \
             share scheduling waves but NOT fused device kernels. \
             CUDA needs CUDA_PATH with runtime+CCCL headers. \
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
    // No silent provider fallback; reject explicitly requested fusion
    // until ONE actual grouped GPU graph has passed numerical/physical gates.
    if options.batch_mode != "interleaved" {
        return Err(ServiceError::new(
            400,
            "unsupported_batch_mode",
            "cross-request fused GPU kernels are not yet qualified; use interleaved",
        ));
    }
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
                Router::Gpu(gpu_engine::GpuRouter::new(
                    declared,
                    selected,
                    gpu_engine::GpuProfile {
                        workers: options.workers,
                        active: options.active_per_worker,
                        context: options.max_context,
                        output: options.max_output,
                        provider_mib: options.provider_mib,
                        device_mib: options.device_mib,
                    },
                )?)
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
    http::serve(listener, router, key)
        .map_err(|e| ServiceError::new(500, "http_error", e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        assert!(YamlConfig::cli(&["--model".into(), "missing".into()]).is_err());
        assert!(YamlConfig::cli(&["--workers".into(), "no".into()]).is_err());
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
