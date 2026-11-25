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
use std::io::{self, Read};
use std::path::Path;
use std::process::ExitCode;
use zerogravity::Result;
use zerogravity::artifact::{Artifact, ArtifactFormat};
use zerogravity::model::DeclaredModel;

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

use zerogravity::devices::GpuBackend;

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
    audit_jsonl: Option<String>,
    audit_layers_jsonl: Option<String>,
    teacher_ids: Option<Vec<u32>>,
    prefill_chunk: usize,
}

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn infer_prepared(
    model: &str,
    prepared: impl Into<zerogravity::diagnostics::cpu_text::Input>,
    options: InferenceOptions,
) -> Result<()> {
    use zerogravity::diagnostics::cpu_text::{self, Diagnostics, Observation, TextOptions};
    let prompt = match options.prompt {
        Some(prompt) => prompt,
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
    let options_text = TextOptions {
        prompt,
        context: options.context,
        max_tokens: options.max_tokens,
        raw: options.raw,
        prefill_chunk: options.prefill_chunk,
        teacher_ids: options.teacher_ids,
    };
    let mut logits = options.audit_jsonl.map(output::CpuAuditFile::new);
    let mut layers = options.audit_layers_jsonl.map(output::CpuAuditFile::new);
    let diagnostics = Diagnostics {
        logits: logits
            .as_mut()
            .map(|writer| writer as &mut dyn cpu_text::AuditSink),
        layers: layers
            .as_mut()
            .map(|writer| writer as &mut dyn cpu_text::AuditSink),
        top_logits: std::env::var_os("ZEROGRAVITY_DIAGNOSTIC_LOGITS").is_some(),
    };
    let output = cpu_text::generate(prepared, options_text, diagnostics, |event| {
        match event {
            Observation::Prefill { done: 0, total } => {
                crate::output::emit!(
                    Progress,
                    "zerogravity: CPU prefill {total} tokens on {model}"
                )
            }
            Observation::Prefill { done, total } => {
                crate::output::emit!(Progress, "zerogravity: prefill {done}/{total}")
            }
            Observation::Logits { step, values } => {
                crate::output::emit!(Explicit, "ZG_LOGITS step={step} {values:?}")
            }
        }
        Ok(())
    })?;
    println!("{}", output.text());
    crate::output::emit!(
        Status,
        "zerogravity: {} generated tokens, {} total consumed (CPU)",
        output.generated().len(),
        output.consumed()
    );
    Ok(())
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
    match args::presentation(&args) {
        Ok((options, command)) => {
            output::install(options);
            run(parse(command))
        }
        Err(detail) => {
            output::fail(output::Failure::new(
                "",
                "invalid_usage",
                &detail,
                Some("Run zerogravity --help."),
            ));
            ExitCode::from(2)
        }
    }
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
    fn run_and_serve_forward_to_their_single_frontends() {
        assert_eq!(
            parse(["serve", "--listen", "127.0.0.1:8000"]),
            Command::Serve(vec!["--listen".into(), "127.0.0.1:8000".into()])
        );
        assert_eq!(
            parse(["run", "model.gguf"]),
            Command::Run {
                args: vec!["model.gguf".into()],
                alias: None
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
