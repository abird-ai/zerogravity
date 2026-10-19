//! Opt-in local OpenAI-compatible HTTP serving.
//! Uses original Qwen CPU source and private independent model sessions.

mod engine;
mod http;

use engine::{Router, ServiceError};
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
            "USAGE: zerogravity serve [--config config.yaml] or [--model ID=MODEL.gguf ...] [--listen 127.0.0.1:8000] [--workers 2] [--active-per-worker 2] [--max-context 1024] [--max-output 128] [--host-mib 4096]\nEndpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions. Greedy Qwen3.5 CPU only. Loopback only; optional ZEROGRAVITY_API_KEY."
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
    // Source preparation before listener advertises readiness; no
    // implicit model downloads, CPU/GPU fallback or registry aliases.
    let router = Arc::new(Router::new(
        options.models.into_iter().map(|m| (m.id, m.path)).collect(),
        options.workers,
        options.active_per_worker,
        options.max_context,
        options.max_output,
        options.host_mib,
    )?);
    let listener = TcpListener::bind(address)
        .map_err(|e| ServiceError::new(500, "bind_failed", e.to_string()))?;
    eprintln!(
        "zerogravity: serving native CPU Qwen on {address}, {} decode workers; max-context {}, max-output {}",
        options.workers, options.max_context, options.max_output
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
    }
}
