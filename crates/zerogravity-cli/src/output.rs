//! CLI presentation.
mod presentation;
pub(crate) use presentation::{
    Event, Failure, Options, Switch, Verbosity, emit, fail, install, report,
};

use super::{GpuBackend, VERSION};
#[cfg(all(feature = "acquisition", target_os = "linux"))]
pub(crate) use presentation::reporter;

pub(super) fn usage() {
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
        presentation_usage();
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
    presentation_usage();
}

fn presentation_usage() {
    println!(
        "\nPRESENTATION OPTIONS (before command):\n    -q, --quiet | -v, --verbose | -vv\n    --format auto|text   (text selects plain presentation)\n    --color auto|always|never\n    --progress auto|always|never\nExisting command-specific JSON, JSONL and audit options are unchanged."
    );
}

/// Keep the existing default CPU command help; other feature sets name
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
        println!("    zerogravity serve --help  (loopback OpenAI-compatible inference router)");
    }
    println!("\nCOMPILED CAPABILITIES:");
    println!("    source inspection: enabled");
    for (name, enabled) in [
        ("CPU", cfg!(feature = "cpu")),
        ("Qwen3.5", cfg!(feature = "model-qwen35")),
        ("text tokenizer", cfg!(feature = "text")),
        ("OpenAI HTTP frontend", cfg!(feature = "serve")),
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

#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
pub(crate) struct CpuAuditFile {
    path: String,
    writer: Option<std::io::BufWriter<std::fs::File>>,
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
impl CpuAuditFile {
    pub(crate) fn new(path: String) -> Self {
        Self { path, writer: None }
    }
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
impl zerogravity::diagnostics::cpu_text::AuditSink for CpuAuditFile {
    fn open(&mut self) -> zerogravity::Result<&mut dyn std::io::Write> {
        if self.writer.is_none() {
            self.writer = Some(std::io::BufWriter::new(std::fs::File::create(&self.path)?));
        }
        Ok(self.writer.as_mut().expect("opened audit file"))
    }
}
