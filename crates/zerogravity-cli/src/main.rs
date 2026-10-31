//! Explicit CPU/resident GPU inference and truthful source-inspection CLI.
//! Reserved runtime commands remain explicit unsupported operations.

use std::env;
mod args;
mod commands;
mod output;
#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod signals;
use args::{Command, parse};
use commands::run;
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use std::fs::File;
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::process::ExitCode;
use zerogravity::Result;
use zerogravity::artifact::{Artifact, ArtifactFormat};
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use zerogravity::cpu::ops::top_logit_ranks;
use zerogravity::model::DeclaredModel;
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use zerogravity::prepare::PreparedQwen;
#[cfg(all(
    feature = "model-qwen35",
    feature = "text",
    any(feature = "vulkan", feature = "cuda", feature = "hip")
))]
mod gpu;

#[cfg(all(feature = "acquisition", target_os = "linux"))]
mod pull;

#[cfg(all(feature = "cache", target_os = "linux"))]
mod cache_manage;

#[cfg(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
))]
mod cache_run;

#[cfg(all(feature = "gemma4-cpu-candidate", target_os = "linux"))]
mod gemma4_cpu;

mod pipeline;
mod run_cli;

#[cfg(feature = "serve")]
mod serve;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuBackend {
    VulkanAmd,
    VulkanNvidia,
    Cuda,
    Hip,
}

impl GpuBackend {
    fn command(self) -> &'static str {
        match self {
            Self::VulkanAmd => "run-vulkan-amd",
            Self::VulkanNvidia => "run-vulkan-nvidia",
            Self::Cuda => "run-cuda",
            Self::Hip => "run-rocm",
        }
    }

    fn feature(self) -> &'static str {
        match self {
            Self::VulkanAmd | Self::VulkanNvidia => "vulkan",
            Self::Cuda => "cuda",
            Self::Hip => "hip",
        }
    }

    fn compiled(self) -> bool {
        match self {
            Self::VulkanAmd | Self::VulkanNvidia => cfg!(feature = "vulkan"),
            Self::Cuda => cfg!(feature = "cuda"),
            Self::Hip => cfg!(feature = "hip"),
        }
    }

    #[cfg(all(
        feature = "model-qwen35",
        feature = "text",
        any(feature = "vulkan", feature = "cuda", feature = "hip")
    ))]
    fn device(self) -> Result<zerogravity::gpu::Device> {
        use zerogravity::gpu::Device;
        let device = match self {
            #[cfg(feature = "vulkan")]
            Self::VulkanAmd => Device::vulkan(zerogravity::gpu::WgpuDeviceKind::IntegratedGpu(0)),
            #[cfg(feature = "vulkan")]
            Self::VulkanNvidia => Device::vulkan(zerogravity::gpu::WgpuDeviceKind::DiscreteGpu(0)),
            #[cfg(feature = "cuda")]
            Self::Cuda => Device::cuda(0),
            #[cfg(feature = "hip")]
            Self::Hip => Device::rocm(0),
            #[cfg(not(all(feature = "vulkan", feature = "cuda", feature = "hip")))]
            _ => return Err(self.unavailable()),
        };
        device.map_err(|error| {
            zerogravity::Error::Unsupported(format!(
                "{} device unavailable: {error}",
                self.command(),
            ))
        })
    }

    fn unavailable(self) -> zerogravity::Error {
        zerogravity::Error::Unsupported(format!(
            "{} requires compiled features {}, model-qwen35 and text",
            self.command(),
            self.feature(),
        ))
    }
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn inspect(path: &Path) -> Result<()> {
    if path.extension().is_some_and(|v| v == "json") {
        let model = DeclaredModel::from_file(path)?;
        println!("Config: {}", path.display());
        print!("{model}");
        println!(
            "Status: CONFIG GEOMETRY ONLY — source weights/tokenizer/execution not validated."
        );
    } else {
        let artifact = Artifact::open(path)?;
        let format = match artifact.info.format {
            ArtifactFormat::Safetensors => "safetensors",
            ArtifactFormat::Gguf => "GGUF v3",
        };
        println!("Artifact: {}", path.display());
        println!("Format: {format}");
        println!("File bytes: {}", artifact.info.file_len);
        println!("Tensor directory: {} tensors", artifact.info.tensors.len());
        if let Some(ref arch) = artifact.info.architecture {
            println!("Declared architecture: {}", arch.escape_debug());
        }
        for tensor in artifact.info.tensors.iter().take(12) {
            println!(
                "  {} {:?} {} @ byte {}",
                tensor.name.escape_debug(),
                tensor.shape,
                tensor.kind.name(),
                tensor.byte_offset
            );
        }
        if artifact.info.tensors.len() > 12 {
            println!(
                "  ... {} additional tensors",
                artifact.info.tensors.len() - 12
            );
        }
        println!(
            "Status: TENSOR DIRECTORY ONLY — no full-source hashes, tokenizer or model execution."
        );
    }
    Ok(())
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
struct InferenceOptions {
    context: usize,
    prompt: Option<String>,
    max_tokens: usize,
    raw: bool,
    config: Option<String>,
    tokenizer: Option<String>,
    audit_jsonl: Option<String>,
    audit_layers_jsonl: Option<String>,
    teacher_ids: Option<Vec<u32>>,
    prefill_chunk: usize,
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn infer(model: &str, options: InferenceOptions) -> Result<()> {
    let path = Path::new(model);
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    let default_config = root.join("config.json");
    let default_tokenizer = root.join("tokenizer.json");
    let config = options
        .config
        .as_deref()
        .map(Path::new)
        .unwrap_or(&default_config);
    let tokenizer = options
        .tokenizer
        .as_deref()
        .map(Path::new)
        .unwrap_or(&default_tokenizer);
    let prepared = PreparedQwen::from_paths(path, config, tokenizer)?;
    infer_prepared(model, prepared, options)
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn infer_prepared(model: &str, prepared: PreparedQwen, options: InferenceOptions) -> Result<()> {
    let InferenceOptions {
        prompt,
        max_tokens,
        raw,
        config: _,
        context,
        tokenizer: _,
        audit_jsonl,
        audit_layers_jsonl,
        teacher_ids,
        prefill_chunk,
    } = options;
    let (model_source, tokenizer) = prepared.into_parts();
    let mut engine = model_source.new_session()?;
    let prompt = match prompt {
        Some(p) => p,
        None => {
            let mut text = String::new();
            io::stdin()
                .take(1024 * 1024 + 1)
                .read_to_string(&mut text)?;
            if text.len() > 1024 * 1024 {
                return Err(zerogravity::Error::Limit(
                    "prompt stdin exceeds 1 MiB".into(),
                ));
            }
            text
        }
    };
    let tokens = tokenizer.prompt_tokens(&prompt, !raw)?;
    if tokens.is_empty() {
        return Err(zerogravity::Error::Malformed(
            "prompt tokenization empty".into(),
        ));
    }
    if tokens
        .len()
        .checked_add(max_tokens - 1)
        .is_none_or(|n| n > context.min(model_source.geometry().max_position_embeddings as usize))
    {
        return Err(zerogravity::Error::Limit(
            "CPU validation maximum is 32768 input tokens; actual throughput is not guaranteed"
                .into(),
        ));
    }
    eprintln!(
        "zerogravity: CPU prefill {} tokens on {}",
        tokens.len(),
        model
    );
    let mut layer_writer = match audit_layers_jsonl {
        Some(path) => Some(BufWriter::new(File::create(path)?)),
        None => None,
    };
    let mut logits = Vec::new();
    let mut done = 0usize;
    for chunk in tokens.chunks(prefill_chunk) {
        let final_chunk = done + chunk.len() == tokens.len();
        if final_chunk && let Some(writer) = layer_writer.as_mut() {
            let (result, observations) = engine.prefill_transaction_with_layer_trace(chunk)?;
            logits = result;
            // One numeric stage vector per JSONL record, at the final prefill
            // token; no 248K-vocabulary vector or entire KV cache is emitted.
            for observation in observations {
                serde_json::to_writer(&mut *writer, &observation).map_err(|e| {
                    zerogravity::Error::Malformed(format!("layer trace encoding: {e}"))
                })?;
                writer.write_all(b"\n")?;
            }
            writer.flush()?;
        } else if final_chunk {
            logits = engine.prefill_transaction(chunk)?;
        } else {
            // Intermediate chunk outputs are never sampled. Preserve the
            // complete atomic StateRoot update, but avoid a redundant full
            // 248K-vocabulary projection and allocation at each boundary.
            engine.prefill_chunk_without_logits(chunk)?;
        }
        done += chunk.len();
        if done.is_multiple_of(128) || done == tokens.len() {
            eprintln!("zerogravity: prefill {done}/{}", tokens.len());
        }
    }
    let mut generated = Vec::with_capacity(max_tokens);
    let mut audit = match audit_jsonl {
        Some(path) => Some(BufWriter::new(File::create(path)?)),
        None => None,
    };
    for generation_step in 0..max_tokens {
        if let Some(writer) = audit.as_mut() {
            let ranks = top_logit_ranks(&logits, 20)?;
            let selected = ranks[0].id;
            serde_json::to_writer(
                &mut *writer,
                &serde_json::json!({
                    "stage":"decode",
                    "step": generated.len(),
                    "prefill_tokens":tokens.len(),
                    "position":engine.position(),
                    "root_id":engine.root_id(),
                    "prefill_chunk":prefill_chunk,
                    "selected_token":selected,
                    "committed_token":teacher_ids.as_ref().map(|ids| ids[generated.len()]).unwrap_or(selected),
                    "state_fingerprint":engine.state_fingerprint(),
                    "state_bytes":engine.state_bytes(),
                    "top":ranks,
                }),
            )
            .map_err(|e| {
                zerogravity::Error::Malformed(format!("cannot encode logits audit: {e}"))
            })?;
            writer.write_all(b"\n")?;
        }
        if std::env::var_os("ZEROGRAVITY_DIAGNOSTIC_LOGITS").is_some() {
            let mut ranks: Vec<_> = logits.iter().enumerate().collect();
            ranks.sort_unstable_by(|a, b| b.1.total_cmp(a.1));
            let top = *ranks[0].1;
            eprintln!(
                "ZG_LOGITS step={} {:?}",
                generated.len(),
                ranks
                    .iter()
                    .take(8)
                    .map(|(id, logit)| (*id, **logit - top))
                    .collect::<Vec<_>>()
            );
        }
        let greedy_id = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .ok_or_else(|| zerogravity::Error::Malformed("model produced no logits".into()))?
            .0 as u32;
        let id = teacher_ids
            .as_ref()
            .map(|ids| ids[generated.len()])
            .unwrap_or(greedy_id);
        if id == tokenizer.eos() {
            break;
        }
        generated.push(id);
        // Last emitted token is visible output; decoding it again is wasted
        // work and can spuriously reject a valid final context-boundary token.
        if generation_step + 1 < max_tokens {
            logits = engine.forward(id)?;
        }
    }
    if let Some(writer) = audit.as_mut() {
        writer.flush()?;
    }
    println!("{}", tokenizer.decode(&generated, true)?);
    eprintln!(
        "zerogravity: {} generated tokens, {} total consumed (CPU)",
        generated.len(),
        engine.position()
    );
    Ok(())
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn cpu_options(command: Command) -> (String, InferenceOptions) {
    let Command::Run {
        model,
        prompt,
        max_tokens,
        raw,
        config,
        tokenizer,
        audit_jsonl,
        audit_layers_jsonl,
        teacher_ids,
        prefill_chunk,
    } = command
    else {
        unreachable!("only CPU run reaches the CPU bridge")
    };
    (
        model,
        InferenceOptions {
            context: 32_768,
            prompt,
            max_tokens,
            raw,
            config,
            tokenizer,
            audit_jsonl,
            audit_layers_jsonl,
            teacher_ids,
            prefill_chunk,
        },
    )
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn execute_cpu(command: Command) -> Result<()> {
    let (model, options) = cpu_options(command);
    infer(&model, options)
}

#[cfg(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
))]
fn execute_qwen_cache(args: Vec<String>) -> Result<()> {
    cache_run::run(args)
}

#[cfg(not(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
)))]
fn execute_qwen_cache(_: Vec<String>) -> Result<()> {
    Err(zerogravity::Error::Unsupported(
        "run-cache requires Linux and compiled cache, cpu, model-qwen35 and text features".into(),
    ))
}

fn execute_cache(args: Vec<String>) -> Result<()> {
    #[cfg(all(feature = "gemma4-cpu-candidate", target_os = "linux"))]
    if gemma4_cpu::selected(&args) {
        return gemma4_cpu::run(&args);
    }
    execute_qwen_cache(args)
}

#[cfg(not(all(feature = "cpu", feature = "model-qwen35", feature = "text")))]
fn execute_cpu(_: Command) -> Result<()> {
    Err(zerogravity::Error::Unsupported(
        "run requires compiled features cpu, model-qwen35 and text".into(),
    ))
}

#[cfg(all(
    feature = "model-qwen35",
    feature = "text",
    any(feature = "vulkan", feature = "cuda", feature = "hip")
))]
fn execute_gpu(backend: GpuBackend, args: Vec<String>) -> Result<()> {
    if !backend.compiled() {
        return Err(backend.unavailable());
    }
    futures_lite::future::block_on(gpu::run(backend, args.into_iter()))
}

#[cfg(not(all(
    feature = "model-qwen35",
    feature = "text",
    any(feature = "vulkan", feature = "cuda", feature = "hip")
)))]
fn execute_gpu(backend: GpuBackend, _: Vec<String>) -> Result<()> {
    Err(backend.unavailable())
}

#[cfg(all(feature = "cache", target_os = "linux"))]
fn execute_cache_manage(args: Vec<String>) -> Result<()> {
    cache_manage::run(&args)
}

#[cfg(not(all(feature = "cache", target_os = "linux")))]
fn execute_cache_manage(_: Vec<String>) -> Result<()> {
    Err(zerogravity::Error::Unsupported(
        "cache commands require Linux and the optional cache feature".into(),
    ))
}

/// Offline model family INTENT matrix, not the source-bound loaded model catalog.
/// Neither model admission, network, GPU nor cache initialization occurs here.
fn models(args: &[String]) -> Result<()> {
    use zerogravity::targets::{MODEL_TARGETS, find};
    match args {
        [] | [_]
            if args.is_empty()
                || args
                    .first()
                    .is_some_and(|one| matches!(one.as_str(), "list" | "--help" | "help")) =>
        {
            if args
                .first()
                .is_some_and(|value| matches!(value.as_str(), "--help" | "help"))
            {
                println!(
                    "USAGE: zerogravity models [list | inspect TARGET | run TARGET | stream TARGET]\nStatic family intents are not runnable model registrations."
                );
                return Ok(());
            }
            println!(
                "PRIORITY  TARGET                        TASK              SOURCE STATUS           EXECUTION"
            );
            for target in MODEL_TARGETS {
                println!(
                    "{:>2}        {:<29} {:<17} {:<23} {}",
                    target.priority,
                    target.slug,
                    target.task.label(),
                    target.source_status.label(),
                    target.execution_status.label()
                );
            }
            println!(
                "NOTE: offline target registry; no weights, cache, GPU, license or runtime started."
            );
            Ok(())
        }
        [command, slug] if matches!(command.as_str(), "inspect" | "run" | "stream") => {
            let entry = find(slug).ok_or_else(|| {
                zerogravity::Error::Unsupported("unknown model target; use models list".into())
            })?;
            if command == "run" {
                return entry.require_source_bound_execution();
            }
            if command == "stream" {
                return entry.require_streaming_media();
            }
            println!(
                "Target: {}\nPriority: {}\nTask: {}\nUpstream: {}\nSource layout: {}\nSource coverage: {}\nExecution: {}\nOptimization: NOT IMPLEMENTED\nStreaming media: NOT IMPLEMENTED\nPayload hold: {}\nNotes: {}",
                entry.slug,
                entry.priority,
                entry.task.label(),
                entry
                    .repository
                    .unwrap_or("unverified; requires exact source"),
                entry.source_layout.label(),
                entry.source_status.label(),
                entry.execution_status.label(),
                if entry.payload_hold {
                    "YES - PIPELINE ONLY"
                } else {
                    "no implicit download"
                },
                entry.note
            );
            println!(
                "No source, tensor format, execution, quant, MTP, SSD or media transport is authorized by this entry."
            );
            Ok(())
        }
        _ => Err(zerogravity::Error::Malformed(
            "invalid models arguments; use models [list|inspect TARGET|run TARGET|stream TARGET]"
                .into(),
        )),
    }
}

fn main() -> ExitCode {
    let args = env::args_os()
        .skip(1)
        .map(|value| value.to_string_lossy().into_owned());
    let args = args.collect::<Vec<_>>();
    let result = match args.first().map(String::as_str) {
        Some("run") => Some(run_cli::run(args[1..].to_vec(), None)),
        Some(
            alias @ ("run-cpu" | "run-cuda" | "run-rocm" | "run-vulkan" | "run-vulkan-amd"
            | "run-vulkan-nvidia"),
        ) => Some(run_cli::run(args[1..].to_vec(), Some(alias))),
        Some("devices") => Some(run_cli::devices(&args[1..])),
        _ => None,
    };
    if let Some(result) = result {
        return match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity run: {error}");
                ExitCode::from(2)
            }
        };
    }
    run(parse(args))
}

#[cfg(test)]
mod tests {
    use super::{Command, parse};

    #[test]
    fn empty_and_help() {
        assert_eq!(parse([] as [&str; 0]), Command::Help);
        assert_eq!(parse(["--help"]), Command::Help);
        assert_eq!(parse(["-h"]), Command::Help);
    }

    #[test]
    fn version_forms() {
        assert_eq!(parse(["--version"]), Command::Version);
        assert_eq!(parse(["-V"]), Command::Version);
    }

    #[test]
    fn commands_remain_explicitly_unimplemented() {
        assert_eq!(
            parse(["serve", "--listen", "127.0.0.1:8000"]),
            Command::Serve(vec!["--listen".into(), "127.0.0.1:8000".into()])
        );
        assert_eq!(
            parse(["run", "model.gguf"]),
            Command::Run {
                model: "model.gguf".into(),
                prompt: None,
                max_tokens: 16,
                raw: false,
                tokenizer: None,
                config: None,
                audit_jsonl: None,
                audit_layers_jsonl: None,
                teacher_ids: None,
                prefill_chunk: 128,
            }
        );
    }

    #[test]
    fn inspect_is_a_real_separate_command() {
        assert_eq!(
            parse(["inspect", "weights.gguf"]),
            Command::Inspect("weights.gguf".into())
        );
        assert!(matches!(parse(["inspect"]), Command::Invalid(_)));
        assert!(matches!(parse(["inspect", "a", "b"]), Command::Invalid(_)));
    }

    #[test]
    fn unknown_is_not_accepted() {
        assert_eq!(
            parse(["--some-unimplemented-option"]),
            Command::Unknown("--some-unimplemented-option".into())
        );
    }
}
