//! One typed run frontend; graph/source/device validation precedes effects.
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};
use zerogravity::{Error, Result, run_policy::*};
mod args;
mod config;
mod devices;
mod execute;
mod qwen35_cuda;
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
mod source;

/// Narrow HOST-only precheck shared by direct and serving Qwen CUDA frontends.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(crate) fn qwen35_cuda_host_available(host_bytes: u64, headroom: &Headroom) -> Result<()> {
    devices::host_available(host_bytes, headroom)
}
/// Transactional pre-Client card admission shared by the direct run and
/// serving Qwen CUDA frontends; one seam, never forked. Serve wiring needs
/// its own device-identity qualification first (see the plan TODO).
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(crate) use qwen35_cuda::admit_pre_client_card;
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunDoc {
    pub entrypoint: Option<EntryDoc>,
    pub backend: Option<String>,
    pub device: Option<Vec<String>>,
    pub headroom: Option<String>,
    pub task: Option<TaskDoc>,
    pub numerical: Option<String>,
    pub speculation: Option<String>,
    pub draft_device: Option<Vec<String>>,
    pub diagnostics: Option<DiagnosticDoc>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum EntryDoc {
    Model(ModelEntry),
    Pipeline(PipelineEntry),
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelEntry {
    pub model: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PipelineEntry {
    pub pipeline: String,
}
impl EntryDoc {
    pub(crate) fn typed(&self) -> Entrypoint {
        match self {
            Self::Model(v) => Entrypoint::Model(v.model.clone()),
            Self::Pipeline(v) => Entrypoint::Pipeline(v.pipeline.clone()),
        }
    }
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TaskDoc {
    pub prompt: Option<String>,
    pub context: Option<usize>,
    pub max_tokens: Option<usize>,
    pub raw: Option<bool>,
    pub teacher_ids: Option<Vec<u32>>,
    pub prefill_chunk: Option<usize>,
}
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DiagnosticDoc {
    pub explain: Option<bool>,
    pub dry_run: Option<bool>,
    pub audit_logits_jsonl: Option<String>,
    pub audit_layers_jsonl: Option<String>,
}
#[derive(Clone, Debug, Default)]
pub(super) struct Arguments {
    positional: Option<String>,
    config: Option<String>,
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
pub(super) fn run(args: Vec<String>, alias: Option<&str>) -> Result<()> {
    let parsed = parse(&args, alias)?;
    if parsed.help {
        help();
        return Ok(());
    }
    let request = config::resolve(
        parsed,
        &std::env::current_dir()?,
        std::env::var_os("XDG_CONFIG_HOME"),
        std::env::var_os("HOME"),
    )?;
    execute::run(request)
}
pub(super) fn devices(args: &[String]) -> Result<()> {
    if !args.is_empty()
        && !matches!(args,[one]if one=="list")
        && !matches!(args,[one,two]if one=="list"&&two=="--json")
    {
        return Err(Error::Malformed("devices list [--json]".into()));
    }
    let inventory = devices::discover()?;
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
fn absolute(base: &Path, s: &str) -> PathBuf {
    let p = Path::new(s);
    if p.is_absolute() {
        p.into()
    } else {
        base.join(p)
    }
}
fn bad(s: impl Into<String>) -> Error {
    Error::Malformed(s.into())
}
fn help() {
    println!(
        "RUN A MODEL\n    zerogravity run MODEL [--prompt TEXT] [-d auto|cpu|gpu:N|pci:ID|/dev/dri/renderDN|/dev/nvidiaN]\n    zerogravity run CONFIG.yaml [--explain|--dry-run]\n    zerogravity devices list [--json]\n\nADVANCED: zerogravity run [MODEL|CONFIG.yaml] [--config FILE] [--model SOURCE]\n    --model-config FILE --tokenizer FILE --snapshot SHA --native-artifacts DIR --prompt TEXT --max-tokens N --context N\n    --raw[=true|false] --teacher-ids IDS --prefill-chunk N\n    --backend auto|cpu|cuda|rocm|vulkan --device auto|cpu|DEVICE|/dev/NODE[,DEVICE...]\n    --headroom PERCENT|BYTES --numerical PROFILE --speculation auto|off|mtp|draft:MODEL\n    --draft-device DEVICE --entrypoint model:ID|pipeline:NAME\n    --explain[=true|false] --dry-run[=true|false]\n    --audit-logits-jsonl FILE --audit-layers-jsonl FILE\n    --define-model ID=TARGET[@SOURCE] --name NAME --input TYPE --output TYPE\n    --step ID=KIND[:MODEL] --link FROM=TO --route PATH=PROTOCOL\nLinux -d /dev: DRM card/render nodes or /dev/nvidiaN; /dev/kfd is shared and refused.\nAliases: run-cpu/run-cuda/run-rocm/run-vulkan (same options).\nSources: exact local files, acquired cache references, hf:/gh:/http:/https: locators. No implicit network or download.\nCatalogue references: catalog:REF or a pull-shaped huggingface/..., huggingface-bundle/..., github/..., github-bundle/... name.\nAbsolute and ./ ../ paths stay local; use ./huggingface/... to name such a local directory.\nOnly local .yaml/.yml implies run config; provider/catalog/URL schemes take precedence.\nSelector-free run loads XDG_CONFIG_HOME/abird/zerogravity/config.yaml (absolute HOME fallback).\nGraph/split/NPU/remote/speculative execution requires independently qualified adapters; parsing grants none."
    );
}
#[cfg(test)]
mod tests;
