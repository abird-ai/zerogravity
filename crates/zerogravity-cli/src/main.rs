//! Explicit CPU/resident GPU inference and truthful source-inspection CLI.
//! Reserved runtime commands remain explicit unsupported operations.

use std::env;
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use std::fs::File;
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;
use std::process::ExitCode;
use zerogravity::Result;
use zerogravity::artifact::{Artifact, ArtifactFormat};
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
use zerogravity::math::top_logit_ranks;
use zerogravity::model::{Family, LayerKind, ModelGeometry};
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

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Help,
    Version,
    Inspect(String),
    Run {
        model: String,
        prompt: Option<String>,
        max_tokens: usize,
        raw: bool,
        tokenizer: Option<String>,
        config: Option<String>,
        audit_jsonl: Option<String>,
        audit_layers_jsonl: Option<String>,
        teacher_ids: Option<Vec<u32>>,
        prefill_chunk: usize,
    },
    GpuRun {
        backend: GpuBackend,
        args: Vec<String>,
    },
    Pull(Vec<String>),
    Models(Vec<String>),
    Pipeline(Vec<String>),
    Serve(Vec<String>),
    CacheRun(Vec<String>),
    CacheManage(Vec<String>),
    NotImplemented(String),
    Invalid(String),
    Unknown(String),
}

fn parse<I, S>(args: I) -> Command
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut args = args.into_iter();
    match args.next() {
        None => Command::Help,
        Some(first) => {
            match first.as_ref() {
                "-h" | "--help" | "help" => Command::Help,
                "-V" | "--version" | "version" => Command::Version,
                "inspect" => match (args.next(), args.next()) {
                    (Some(path), None) if !path.as_ref().is_empty() => {
                        Command::Inspect(path.as_ref().to_owned())
                    }
                    _ => Command::Invalid("inspect requires exactly one local path".into()),
                },
                "run" => {
                    let Some(model) = args.next() else {
                        return Command::Invalid("run requires a local model GGUF file".into());
                    };
                    let model = model.as_ref().to_owned();
                    if model.is_empty() {
                        return Command::Invalid("missing model path".into());
                    }
                    let mut prompt = None;
                    let mut max_tokens = 16usize;
                    let mut raw = false;
                    let mut tokenizer = None;
                    let mut config = None;
                    let mut audit_jsonl = None;
                    let mut audit_layers_jsonl = None;
                    let mut teacher_ids = None;
                    let mut prefill_chunk = 128usize;
                    while let Some(flag) = args.next() {
                        let value = flag.as_ref();
                        match value {
                            "--raw" => raw = true,
                            "--prompt"
                            | "-p"
                            | "--max-tokens"
                            | "-n"
                            | "--tokenizer"
                            | "--config"
                            | "--audit-logits-jsonl"
                            | "--audit-layers-jsonl"
                            | "--teacher-ids"
                            | "--prefill-chunk" => {
                                let Some(next) = args.next() else {
                                    return Command::Invalid(format!("{value} requires a value"));
                                };
                                let n = next.as_ref().to_owned();
                                match value {
                                    "--prompt" | "-p" => prompt = Some(n),
                                    "--max-tokens" | "-n" => match n.parse::<usize>() {
                                        Ok(x) if (1..=128).contains(&x) => max_tokens = x,
                                        _ => {
                                            return Command::Invalid(
                                                "max tokens must be in 1..128".into(),
                                            );
                                        }
                                    },
                                    "--tokenizer" => tokenizer = Some(n),
                                    "--config" => config = Some(n),
                                    "--audit-logits-jsonl" => audit_jsonl = Some(n),
                                    "--audit-layers-jsonl" => audit_layers_jsonl = Some(n),
                                    "--prefill-chunk" => match n.parse::<usize>() {
                                        Ok(value) if (1..=32768).contains(&value) => {
                                            prefill_chunk = value
                                        }
                                        _ => {
                                            return Command::Invalid(
                                                "prefill chunk must be 1..32768".into(),
                                            );
                                        }
                                    },
                                    "--teacher-ids" => {
                                        let ids = n
                                            .split(',')
                                            .map(str::parse::<u32>)
                                            .collect::<std::result::Result<Vec<_>, _>>();
                                        match ids {
                                        Ok(v) if !v.is_empty() && v.len() <= 128 => teacher_ids = Some(v),
                                        _ => return Command::Invalid("invalid comma-separated teacher token IDs (max 128)".into()),
                                    }
                                    }
                                    _ => unreachable!(),
                                }
                            }
                            _ => return Command::Invalid(format!("unknown run option '{value}'")),
                        }
                    }
                    if let Some(ref ids) = teacher_ids
                        && (ids.len() != max_tokens || audit_jsonl.is_none())
                    {
                        return Command::Invalid(
                            "--teacher-ids requires --audit-logits-jsonl and -n matching teacher count".into()
                        );
                    }
                    Command::Run {
                        model,
                        prompt,
                        max_tokens,
                        raw,
                        tokenizer,
                        config,
                        audit_jsonl,
                        audit_layers_jsonl,
                        teacher_ids,
                        prefill_chunk,
                    }
                }
                "run-vulkan-amd" | "run-vulkan-nvidia" | "run-cuda" | "run-rocm" => {
                    let backend = match first.as_ref() {
                        "run-vulkan-amd" => GpuBackend::VulkanAmd,
                        "run-vulkan-nvidia" => GpuBackend::VulkanNvidia,
                        "run-cuda" => GpuBackend::Cuda,
                        _ => GpuBackend::Hip,
                    };
                    Command::GpuRun {
                        backend,
                        args: args.map(|arg| arg.as_ref().to_owned()).collect(),
                    }
                }
                "pull" => Command::Pull(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "models" => Command::Models(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "pipeline" => Command::Pipeline(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "cache" => Command::CacheManage(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "run-cache" => Command::CacheRun(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "serve" => Command::Serve(args.map(|arg| arg.as_ref().to_owned()).collect()),
                "config" => Command::NotImplemented(first.as_ref().to_owned()),
                other => Command::Unknown(other.to_owned()),
            }
        }
    }
}

fn usage() {
    if !cfg!(all(
        feature = "cpu",
        feature = "model-qwen35",
        feature = "text"
    )) || cfg!(any(
        feature = "vulkan",
        feature = "cuda",
        feature = "hip",
        feature = "acquisition",
        feature = "cache",
        feature = "serve"
    )) {
        composed_usage();
        return;
    }

    println!(
        "zerogravity v{VERSION}

A composable, high-performance AI execution fabric.
Phase 1: real Qwen3.5 0.8B/2B CPU text inference and source inspection.

USAGE:
    zerogravity [--help | --version]
    zerogravity inspect <local-config.json|model.safetensors|model.gguf>
    zerogravity run <model.gguf> --prompt Hello --max-tokens 16
    zerogravity run <model.gguf> --prompt Hello --raw -n 8
    zerogravity run <model.gguf> --raw -p Hello --prefill-chunk 128
    zerogravity run <model.gguf> --raw -p Hello --audit-logits-jsonl ./target/logits.jsonl
    zerogravity run <model.gguf> --raw -p Hello --audit-layers-jsonl ./target/layers.jsonl

IMPLEMENTED:
    inspect     Validate bounded source tensor directory or model config geometry
    run         Qwen3.5 0.8B/2B CPU text inference (no server, first reference path)

FUTURE COMMANDS (NOT IMPLEMENTED):
    serve       Run the shared model router and OpenAI-compatible API
    models      Manage registered and resident models
    config      View and update model/runtime configuration

Only qualified text-only dense Qwen3.5 is implemented; CPU is the first
reference backend. Trained MTP, vision, Gemma, GPU, and HTTP are NOT IMPLEMENTED.
Use --model-config and --tokenizer for colocated sidecars; --config selects pipeline YAML. See docs/ENGINEERING_SPECIFICATION.md."
    );
}

/// Keep the existing default CPU help byte-for-byte; other feature sets name
/// only their compiled command capabilities without initializing a device.
fn composed_usage() {
    println!(
        "zerogravity v{VERSION}\n\nA composable, high-performance AI execution fabric.\n\nUSAGE:\n    zerogravity [--help | --version]\n    zerogravity inspect <local-config.json|model.safetensors|model.gguf>"
    );
    if cfg!(all(
        feature = "cpu",
        feature = "model-qwen35",
        feature = "text"
    )) {
        println!("    zerogravity run <model.gguf> --prompt Hello --max-tokens 16");
    }
    if cfg!(all(feature = "model-qwen35", feature = "text")) {
        for backend in [
            GpuBackend::VulkanAmd,
            GpuBackend::VulkanNvidia,
            GpuBackend::Cuda,
            GpuBackend::Hip,
        ] {
            if backend.compiled() {
                println!(
                    "    zerogravity {} <model.gguf> --prompt Hello --max-tokens 16",
                    backend.command()
                );
            }
        }
    }
    if cfg!(all(feature = "cache", target_os = "linux")) {
        println!("    zerogravity cache --help  (explicit offline exact-snapshot removal)");
    }
    if cfg!(all(feature = "acquisition", target_os = "linux")) {
        println!(
            "    zerogravity pull --help  (explicit data acquisition and offline cache inspection)"
        );
    }
    if cfg!(all(
        feature = "cache",
        feature = "cpu",
        feature = "model-qwen35",
        feature = "text",
        target_os = "linux"
    )) {
        println!("    zerogravity run-cache --help  (explicit offline verified cache model)");
    }
    if cfg!(feature = "serve") {
        println!("    zerogravity serve --help  (loopback OpenAI-compatible Qwen3.5 CPU router)");
    }
    println!("\nCOMPILED CAPABILITIES:");
    println!("    source inspection: enabled");
    for (name, enabled) in [
        ("CPU", cfg!(feature = "cpu")),
        ("Qwen3.5", cfg!(feature = "model-qwen35")),
        ("text tokenizer", cfg!(feature = "text")),
        ("OpenAI text HTTP (CPU)", cfg!(feature = "serve")),
        ("Vulkan", cfg!(feature = "vulkan")),
        ("CUDA", cfg!(feature = "cuda")),
        ("HIP/ROCm", cfg!(feature = "hip")),
    ] {
        println!(
            "    {name}: {}",
            if enabled { "enabled" } else { "disabled" }
        );
    }
    if cfg!(all(
        feature = "model-qwen35",
        feature = "text",
        any(feature = "vulkan", feature = "cuda", feature = "hip")
    )) {
        println!(
            "\nGPU OPTIONS:\n    --prompt TEXT | stdin (at most 1 MiB)\n    --max-tokens N  (1..=512, default 16)\n    --raw\n    --provider-mib N  (default 2048)\n    --device-mib N  (default 4096)\n\nResident GPU prompt plus generation admits at most 512 consumed tokens.\nOriginal model.gguf, config.json and tokenizer.json must be colocated."
        );
    }
    println!("\nMODEL MATRIX: zerogravity models [list|inspect SLUG] (offline)");
    println!(
        "PIPELINES: zerogravity pipeline [check|run] --config FILE.yaml (run NOT IMPLEMENTED)"
    );
    if cfg!(feature = "serve") {
        println!("\nFUTURE COMMANDS (NOT IMPLEMENTED):\n    config      Runtime configuration");
    } else {
        println!(
            "\nFUTURE COMMANDS (NOT IMPLEMENTED):\n    serve       HTTP router (requires serve feature)\n    config      Runtime configuration"
        );
    }
    println!(
        "\nRun routing: zerogravity run --help; inventory: zerogravity devices list [--json].\nForced backends never fall back. Generic GPU routing refuses missing cold costs/physical binding."
    );
}

fn inspect(path: &Path) -> Result<()> {
    if path.extension().is_some_and(|v| v == "json") {
        let geometry = ModelGeometry::from_file(path)?;
        println!("Config: {}", path.display());
        println!("Family: {:?}", geometry.family);
        println!(
            "Text geometry: {} layers, hidden {}, head dim {}, attention heads {}, KV heads {}",
            geometry.layers.len(),
            geometry.hidden_size,
            geometry.head_dim,
            geometry.num_attention_heads,
            geometry.num_key_value_heads
        );
        println!(
            "Vocabulary: {}; max declared positions: {}",
            geometry.vocab_size, geometry.max_position_embeddings
        );
        let full = geometry
            .layers
            .iter()
            .filter(|&&k| k == LayerKind::FullAttention)
            .count();
        println!("Full attention layers: {full}");
        match geometry.family {
            Family::QwenHybrid => println!(
                "Gated DeltaNet layers: {}; recurrent K heads: {:?}, V heads: {:?}",
                geometry
                    .layers
                    .iter()
                    .filter(|&&k| k == LayerKind::LinearAttention)
                    .count(),
                geometry.recurrent_key_heads,
                geometry.recurrent_value_heads
            ),
            Family::Gemma4Moe => println!(
                "Sliding attention layers: {}; window: {:?}; experts: {:?}; selected experts: {:?}",
                geometry
                    .layers
                    .iter()
                    .filter(|&&k| k == LayerKind::SlidingAttention)
                    .count(),
                geometry.sliding_window,
                geometry.num_experts,
                geometry.top_k_experts
            ),
        }
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

fn run(command: Command) -> ExitCode {
    match command {
        Command::Help => {
            usage();
            ExitCode::SUCCESS
        }
        Command::Version => {
            println!("zerogravity v{VERSION}");
            ExitCode::SUCCESS
        }
        Command::CacheManage(args) => match execute_cache_manage(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity cache: {error}");
                ExitCode::from(2)
            }
        },
        Command::CacheRun(args) => match execute_cache(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity: {error}");
                ExitCode::from(2)
            }
        },
        command @ Command::Run { .. } => match execute_cpu(command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("zerogravity run: {e}");
                ExitCode::from(2)
            }
        },
        Command::GpuRun { backend, args } => match execute_gpu(backend, args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("zerogravity {}: {e}", backend.command());
                ExitCode::from(2)
            }
        },
        Command::Inspect(path) => match inspect(Path::new(&path)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("zerogravity inspect: {e}");
                ExitCode::from(2)
            }
        },
        Command::Pull(args) => {
            #[cfg(all(feature = "acquisition", target_os = "linux"))]
            let result = pull::run(&args);
            #[cfg(not(all(feature = "acquisition", target_os = "linux")))]
            let result: Result<()> = {
                let _ = args;
                Err(zerogravity::Error::Unsupported(
                    "pull requires the optional acquisition feature on Linux".into(),
                ))
            };
            match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("zerogravity pull: {error}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Pipeline(args) => match pipeline::run(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity pipeline: {error}");
                ExitCode::from(2)
            }
        },
        Command::Serve(args) => {
            #[cfg(feature = "serve")]
            let result = serve::run(&args);
            #[cfg(not(feature = "serve"))]
            let result: std::result::Result<(), String> = {
                let _ = args;
                Err("native HTTP serving requires --features serve".into())
            };
            match result {
                Ok(()) => ExitCode::SUCCESS,
                Err(error) => {
                    eprintln!("zerogravity serve: {error:?}");
                    ExitCode::from(2)
                }
            }
        }
        Command::Models(args) => match models(&args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("zerogravity models: {error}");
                ExitCode::from(2)
            }
        },
        Command::NotImplemented(name) => {
            eprintln!(
                "zerogravity: '{name}' is not implemented (local Qwen3.5 CPU run/inspect exist; router, API and model management are future work)"
            );
            ExitCode::from(2)
        }
        Command::Invalid(detail) => {
            eprintln!("zerogravity: {detail}");
            ExitCode::from(2)
        }
        Command::Unknown(name) => {
            eprintln!("zerogravity: unknown command '{name}'. Try --help.");
            ExitCode::from(2)
        }
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
