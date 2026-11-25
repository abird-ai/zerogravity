//! One typed run frontend; graph/source/device validation precedes effects.
use std::path::Path;
use zerogravity::pipelines::{DiagnosticDoc, TaskDoc};
use zerogravity::{Error, Result, run_policy::*};
mod args;
mod config;
mod devices;
mod execute;
pub(crate) mod input;
mod qwen35_cuda;
#[cfg(all(
    feature = "model-qwen35",
    feature = "cache",
    feature = "text",
    target_os = "linux",
    any(feature = "cuda", feature = "hip")
))]
pub(crate) mod qwen_source;
pub(crate) mod source;

#[derive(Clone, Debug, Default)]
pub(super) struct Arguments {
    positional: Option<String>,
    direct_positional: bool,
    prompt_file: Option<String>,
    pull: bool,
    offline: bool,
    config: Option<String>,
    profile: Option<String>,
    model: Option<String>,
    model_config: Option<String>,
    tokenizer: Option<String>,
    snapshot: Option<String>,
    native_artifacts: Option<String>,
    backend: Option<Backend>,
    devices: Option<Vec<String>>,
    headroom: Option<Headroom>,
    entrypoint: Option<Entrypoint>,
    task: TaskDoc,
    diagnostics: DiagnosticDoc,
    numerical: Option<String>,
    speculation: Option<Speculation>,
    draft_device: Option<Vec<String>>,
    graph: Vec<String>,
    vendor: Option<String>,
    help: bool,
}
use args::parse;
pub(crate) use args::takes_value;
pub(super) fn run(
    args: Vec<String>,
    alias: Option<&str>,
    format: crate::args::Format,
) -> Result<()> {
    let mut parsed = parse(&args, alias)?;
    if parsed.help {
        help();
        return Ok(());
    }
    let pull = parsed.pull;
    let offline = parsed.offline;
    if let Some(file) = parsed.prompt_file.take() {
        parsed.task.prompt = Some(input::prompt_file(Path::new(&file))?);
    }
    let (mut request, ask_before_pull) = config::resolve_selected(
        parsed,
        &std::env::current_dir()?,
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    )?;
    if format == crate::args::Format::Json && !request.diagnostics.dry_run {
        return Err(bad(
            "run --format json requires a dry-run plan; generated text remains stdout",
        ));
    }
    if pull && request.diagnostics.dry_run {
        return Err(bad("--pull cannot be combined with dry-run"));
    }
    zerogravity::run::validate_source_request(&request)?;
    let interaction = input::interaction()?;
    let interactive = input::prepare(&mut request, interaction)?;
    execute::run(
        request,
        execute::Mode {
            interactive,
            offer: interaction && !offline && ask_before_pull,
            pull,
        },
    )
}
pub(super) fn devices(args: &[String]) -> Result<()> {
    if !args.is_empty()
        && !matches!(args,[one]if one=="list")
        && !matches!(args,[one,two]if one=="list"&&two=="--json")
    {
        return Err(Error::Malformed("devices list [--json]".into()));
    }
    let runtime = zerogravity::Runtime::builder().build()?;
    let inventory = runtime.control().devices()?;
    if args.iter().any(|s| s == "--json") {
        println!(
            "{}",
            serde_json::to_string_pretty(&inventory).map_err(|e| Error::State(e.to_string()))?
        );
    } else {
        println!("Observed hardware only; execution eligibility is checked per model by `run`.");
        for d in inventory.devices {
            println!("{}", devices::display(&d));
        }
    }
    Ok(())
}
pub(super) fn device_inspect(selector: &str, format: crate::args::Format) -> Result<()> {
    let selector = DeviceRef::parse(selector)?;
    let runtime = zerogravity::Runtime::builder().build()?;
    let device = runtime.control().inspect_device(&selector)?;
    if format == crate::args::Format::Json {
        crate::output::json(&device)
    } else {
        println!(
            "Observed hardware only; no model admission.\n{}\nObservation: {}",
            devices::display(&device),
            device.provenance
        );
        Ok(())
    }
}

fn bad(s: impl Into<String>) -> Error {
    Error::Malformed(s.into())
}
pub(super) fn help() {
    println!(
        "RUN A MODEL\n    zerogravity run MODEL [PROMPT] [--pull] [-d auto|cpu|gpu:N|pci:ID|/dev/dri/renderDN|/dev/nvidiaN]\n    zerogravity run CONFIG.yaml [--explain|--dry-run]\n    zerogravity device list [--format json]\n\nADVANCED: zerogravity run [MODEL|CONFIG.yaml] [--config FILE] [--model SOURCE]\n    --model-config FILE --tokenizer FILE --snapshot SHA --native-artifacts DIR --prompt TEXT | --prompt-file FILE --max-tokens N --context N\n    --raw[=true|false] --teacher-ids IDS --prefill-chunk N\n    --backend auto|cpu|cuda|rocm|vulkan --device auto|cpu|DEVICE|/dev/NODE[,DEVICE...]\n    --headroom PERCENT|BYTES --numerical source|native --speculation auto|off|mtp|draft:MODEL\n    --draft-device DEVICE --entrypoint model:ID|pipeline:NAME\n    --explain[=true|false] --dry-run[=true|false] --profile NAME\n    --audit-logits-jsonl FILE --audit-layers-jsonl FILE\n    --define-model ID=TARGET[@SOURCE] --name NAME --input TYPE --output TYPE\n    --step ID=KIND[:MODEL] --link FROM=TO --route PATH=PROTOCOL\nLinux -d /dev: DRM card/render nodes or /dev/nvidiaN; /dev/kfd is shared and refused.\nAliases: run-cpu/run-cuda/run-rocm/run-vulkan (same options).\nNative numerical profile: explicit ROCm canonical Qwen source, qualified gfx1151 with resident weights and rocBLAS; source remains the default.\nSources: exact local files, SHA256 snapshots, catalog:/by-name: references, hf:/gh:/http:/https: locators. Bare lowercase names resolve existing durable names or exact approved catalog defaults.\nPrompt: one positional prompt, --prompt, --prompt-file, configured prompt, or piped stdin. No prompt on a capable TTY opens repeated independent requests.\n--pull explicitly consents to a supported exact catalog download; offline and dry-run never acquire.\nCatalogue references: catalog:REF or a pull-shaped huggingface/..., huggingface-bundle/..., github/..., github-bundle/... name.\nAbsolute and ./ ../ paths stay local; use ./huggingface/... to name such a local directory.\nOnly local .yaml/.yml implies run config; provider/catalog/URL schemes take precedence.\nImplicit config.yaml supplies defaults when present; --profile/--config replace it. Explicit positional model selects a fresh direct route.\nGraph/split/NPU/remote/speculative execution requires independently qualified adapters; parsing grants none."
    );
    let mut compiled = vec!["auto"];
    if cfg!(all(
        feature = "cpu",
        feature = "model-qwen35",
        feature = "text"
    )) {
        compiled.push("cpu");
    }
    if cfg!(all(
        feature = "cuda",
        feature = "model-qwen35",
        feature = "text"
    )) {
        compiled.push("cuda");
    }
    if cfg!(all(
        feature = "hip",
        feature = "model-qwen35",
        feature = "text"
    )) {
        compiled.push("rocm");
    }
    if cfg!(all(
        feature = "vulkan",
        feature = "model-qwen35",
        feature = "text"
    )) {
        compiled.push("vulkan");
    }
    println!(
        "Compiled run choices: {} (source/device eligibility checked separately)",
        compiled.join(", ")
    );
}
#[cfg(test)]
mod tests;

#[cfg(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
))]
pub(super) fn cached_cpu_options(args: &[String]) -> Result<crate::InferenceOptions> {
    let parsed = parse(args, None)?;
    let prompt_supplied = parsed.task.prompt.is_some();
    let options = zerogravity::diagnostics::cpu_text::TextOptions {
        prompt: parsed.task.prompt.unwrap_or_default(),
        context: 32768,
        max_tokens: parsed.task.max_tokens.unwrap_or(16),
        raw: parsed.task.raw.unwrap_or(false),
        prefill_chunk: parsed.task.prefill_chunk.unwrap_or(128),
        teacher_ids: parsed.task.teacher_ids,
    };
    options.validate(parsed.diagnostics.audit_logits_jsonl.is_some())?;
    Ok(crate::InferenceOptions {
        context: options.context,
        prompt: prompt_supplied.then_some(options.prompt),
        max_tokens: options.max_tokens,
        raw: options.raw,
        prefill_chunk: options.prefill_chunk,
        teacher_ids: options.teacher_ids,
        audit_jsonl: parsed.diagnostics.audit_logits_jsonl,
        audit_layers_jsonl: parsed.diagnostics.audit_layers_jsonl,
    })
}
