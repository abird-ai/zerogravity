//! One-shot CLI adapter onto the same source-bound Qwen Runtime used by serving.
use super::*;
use zerogravity::cache::{CacheMetadata, CacheOptions};
use zerogravity::models::{ModelLoadRequest, QwenServingProfile};
use zerogravity::runtime::text::{Request as TextRequest, ResponseEvent};

const GIB: u64 = 1024 * 1024 * 1024;
const QWEN_TARGETS: &[&str] = &["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.8-27b"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SelectedBackend {
    Cuda,
    Hip,
}

struct Resolved {
    alias: String,
    cache: CacheOptions,
    snapshot: String,
    target: String,
}

pub(super) fn candidate(request: &Request, model: &zerogravity::pipeline::ModelInstance) -> bool {
    if model.native_artifacts.is_some()
        || !matches!(
            request.backend,
            Backend::Auto | Backend::Cuda | Backend::Rocm
        )
    {
        return false;
    }
    target(model).is_some()
}

pub(super) fn selected_backend(
    request: &Request,
    model: &zerogravity::pipeline::ModelInstance,
    inventory: &devices::Inventory,
) -> Result<Option<SelectedBackend>> {
    if !candidate(request, model) {
        return Ok(None);
    }
    zerogravity::devices::fresh(inventory)?;
    let target = target(model).expect("candidate has Qwen target");
    select_backend(request, inventory, &target)
}

pub(super) fn run(
    request: Request,
    inventory: devices::Inventory,
    backend: SelectedBackend,
) -> Result<()> {
    let Entrypoint::Model(id) = &request.entrypoint else {
        return Err(bad(
            "source-bound Qwen run requires direct model entrypoint",
        ));
    };
    let model = request
        .graph
        .models
        .iter()
        .find(|model| &model.id == id)
        .ok_or_else(|| bad("direct entrypoint model absent"))?;
    let resolved = resolve(model, id)?;
    let native = request.numerical.as_deref() == Some("native");
    if native && backend != SelectedBackend::Hip {
        return Err(Error::Unsupported(
            "native numerical profile requires ROCm".into(),
        ));
    }

    if request.diagnostics.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "schema": "zerogravity.run.decision.v1",
                "model": resolved.target,
                "snapshot": resolved.snapshot,
                "backend": match backend { SelectedBackend::Cuda => "cuda", SelectedBackend::Hip => "rocm" },
                "numerical": if native { "hip-rocblas-bf16-mixed-v2" } else { "source" },
                "context": request.task.context,
                "max_tokens": request.task.max_tokens,
                "source": "canonical-cache-snapshot",
                "reservation": "none",
                "dry_run": true,
            }))
            .map_err(|error| Error::State(error.to_string()))?
        );
        return Ok(());
    }

    if request.task.teacher_ids.is_some()
        || request.diagnostics.audit_logits_jsonl.is_some()
        || request.diagnostics.audit_layers_jsonl.is_some()
    {
        return Err(Error::Unsupported(
            "source-bound GPU run does not expose CPU teacher/audit diagnostics".into(),
        ));
    }

    if request.diagnostics.explain {
        eprintln!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "schema": "zerogravity.run.decision.v1",
                "model": resolved.target,
                "snapshot": resolved.snapshot,
                "backend": match backend { SelectedBackend::Cuda => "cuda", SelectedBackend::Hip => "rocm" },
                "numerical": if native { "hip-rocblas-bf16-mixed-v2" } else { "source" },
                "context": request.task.context,
                "max_tokens": request.task.max_tokens,
                "source": "canonical-cache-snapshot",
            }))
            .map_err(|error| Error::State(error.to_string()))?
        );
    }

    let prompt = prompt(request.task.prompt.clone())?;
    let runtime = zerogravity::Runtime::builder().build()?;
    let result = (|| -> Result<()> {
        let profile = QwenServingProfile {
            active: 1,
            context: request.task.context,
            output: request.task.max_tokens,
            max_pending: Some(1),
            cohort_width: 0,
        };
        let load = match backend {
            SelectedBackend::Cuda => cuda_load(
                vec![resolved.alias.clone()],
                resolved.cache.clone(),
                resolved.snapshot.clone(),
                profile,
            )?,
            SelectedBackend::Hip => hip_load(
                vec![resolved.alias.clone()],
                resolved.cache.clone(),
                resolved.snapshot.clone(),
                profile,
                hip_budget(&request, &inventory)?,
                native,
            )?,
        };
        runtime
            .models()
            .load(load)
            .map_err(|error| error.into_core())?;

        let receipt = runtime
            .request_text(TextRequest {
                model: resolved.alias.clone(),
                input: prompt,
                chat: !request.task.raw,
                max_tokens: request.task.max_tokens,
                stream: false,
                token_metrics: None,
            })
            .map_err(|error| error.into_core())?;
        receive(receipt)
    })();
    let shutdown = runtime.shutdown();
    match (result, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

fn target(model: &zerogravity::pipeline::ModelInstance) -> Option<String> {
    if QWEN_TARGETS.contains(&model.target.as_str()) {
        return Some(model.target.clone());
    }
    let source = SourceSelector::parse(model.source.as_deref()?).ok()?;
    match source {
        SourceSelector::Catalog(name) if QWEN_TARGETS.contains(&name.as_str()) => Some(name),
        _ => None,
    }
}

fn resolve(model: &zerogravity::pipeline::ModelInstance, alias: &str) -> Result<Resolved> {
    let target =
        target(model).ok_or_else(|| Error::Unsupported("not a Qwen source target".into()))?;
    let source = model
        .source
        .as_deref()
        .ok_or_else(|| Error::Malformed("source-bound Qwen run requires a model source".into()))?;
    let selector = SourceSelector::parse(source)?;
    if matches!(selector, SourceSelector::Local(_)) {
        return Err(Error::Unsupported(
            "source-bound Qwen GPU run requires an acquired cache reference, not a local GGUF path"
                .into(),
        ));
    }
    let root = zerogravity::cache::paths::resolve_root(
        None,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    let metadata = CacheMetadata::open(&root)?;
    let snapshot = if let Some(requested) = model.snapshot.as_deref() {
        metadata.snapshot(requested)?
    } else if matches!(&selector, SourceSelector::Catalog(name) if name == &target) {
        let digest = zerogravity::models::qwen_source_snapshot(&target).ok_or_else(|| {
            Error::Unsupported("Qwen target has no compiled canonical source".into())
        })?;
        metadata.snapshot(digest).map_err(|_| {
            Error::Unsupported(format!(
                "{target} is not acquired; run `zerogravity pull {target}` first"
            ))
        })?
    } else {
        metadata.select(&selector, None)?
    };
    let disk_limit = metadata.disk_limit_bytes()?;
    Ok(Resolved {
        alias: alias.to_owned(),
        cache: CacheOptions::new(Some(root), disk_limit),
        snapshot: snapshot.digest,
        target,
    })
}

fn select_backend(
    request: &Request,
    inventory: &devices::Inventory,
    target: &str,
) -> Result<Option<SelectedBackend>> {
    let allowed = zerogravity::devices::allowed(request, inventory)?;
    match request.backend {
        Backend::Cuda => {
            require_cuda(&allowed, target)?;
            Ok(Some(SelectedBackend::Cuda))
        }
        Backend::Rocm => {
            require_hip(&allowed)?;
            Ok(Some(SelectedBackend::Hip))
        }
        Backend::Auto => {
            if cuda_candidate(&allowed, target) {
                Ok(Some(SelectedBackend::Cuda))
            } else if hip_candidate(&allowed) {
                Ok(Some(SelectedBackend::Hip))
            } else {
                Ok(None)
            }
        }
        Backend::Cpu | Backend::Vulkan => Ok(None),
    }
}

fn cuda_candidate(devices: &[devices::Observation], target: &str) -> bool {
    cfg!(feature = "cuda")
        && target != "qwen3.8-27b"
        && devices.iter().any(|device| {
            device.class == "gpu"
                && device.vendor == "nvidia"
                && device.cuda_cc == Some((8, 6))
                && device.available.is_some_and(|bytes| bytes >= 5 * GIB)
        })
}

fn hip_candidate(devices: &[devices::Observation]) -> bool {
    cfg!(feature = "hip")
        && devices
            .iter()
            .any(|device| device.class == "gpu" && device.vendor == "amd")
}

fn require_cuda(devices: &[devices::Observation], target: &str) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (devices, target);
        return Err(Error::Unsupported("CUDA backend is not compiled".into()));
    }
    #[cfg(feature = "cuda")]
    {
        if target == "qwen3.8-27b" {
            return Err(Error::Unsupported(
                "current resident Qwen CUDA envelope cannot fit qwen3.8-27b".into(),
            ));
        }
        let device = devices
            .iter()
            .find(|device| device.class == "gpu" && device.vendor == "nvidia")
            .ok_or_else(|| Error::Unsupported("no selected NVIDIA CUDA device".into()))?;
        if device.cuda_cc != Some((8, 6)) {
            return Err(Error::Unsupported(format!(
                "current Qwen CUDA source kernels are SM86-qualified; selected device reports compute capability {}",
                device
                    .cuda_cc
                    .map(|(major, minor)| format!("{major}.{minor}"))
                    .unwrap_or_else(|| "unknown".into())
            )));
        }
        if device.available.is_some_and(|bytes| bytes < 5 * GIB) {
            return Err(Error::Limit(
                "current Qwen CUDA source path requires at least 5 GiB available VRAM".into(),
            ));
        }
        Ok(())
    }
}

fn require_hip(devices: &[devices::Observation]) -> Result<()> {
    #[cfg(not(feature = "hip"))]
    {
        let _ = devices;
        return Err(Error::Unsupported(
            "ROCm/HIP backend is not compiled".into(),
        ));
    }
    #[cfg(feature = "hip")]
    {
        devices
            .iter()
            .any(|device| device.class == "gpu" && device.vendor == "amd")
            .then_some(())
            .ok_or_else(|| Error::Unsupported("no selected AMD HIP device".into()))
    }
}

#[cfg(feature = "cuda")]
fn cuda_load(
    aliases: Vec<String>,
    cache: CacheOptions,
    snapshot: String,
    profile: QwenServingProfile,
) -> Result<ModelLoadRequest> {
    Ok(ModelLoadRequest::qwen_cuda(
        aliases, cache, snapshot, profile,
    ))
}
#[cfg(not(feature = "cuda"))]
fn cuda_load(
    _: Vec<String>,
    _: CacheOptions,
    _: String,
    _: QwenServingProfile,
) -> Result<ModelLoadRequest> {
    Err(Error::Unsupported("CUDA backend is not compiled".into()))
}

#[cfg(feature = "hip")]
fn hip_load(
    aliases: Vec<String>,
    cache: CacheOptions,
    snapshot: String,
    profile: QwenServingProfile,
    budget: zerogravity::models::HipSourceBudget,
    native: bool,
) -> Result<ModelLoadRequest> {
    Ok(if native {
        ModelLoadRequest::qwen_hip_native(aliases, cache, snapshot, profile, budget)
    } else {
        ModelLoadRequest::qwen_hip(aliases, cache, snapshot, profile, budget)
    })
}
#[cfg(not(feature = "hip"))]
fn hip_load(
    _: Vec<String>,
    _: CacheOptions,
    _: String,
    _: QwenServingProfile,
    _: (),
    _: bool,
) -> Result<ModelLoadRequest> {
    Err(Error::Unsupported(
        "ROCm/HIP backend is not compiled".into(),
    ))
}

#[cfg(feature = "hip")]
fn hip_budget(
    request: &Request,
    inventory: &devices::Inventory,
) -> Result<zerogravity::models::HipSourceBudget> {
    let allowed = zerogravity::devices::allowed(request, inventory)?;
    let observed = zerogravity::gpu::observe_hip_device(Default::default())
        .ok_or_else(|| Error::Unsupported("HIP device 0 is unavailable".into()))?;
    let observed_pci = observed
        .physical
        .pci_address
        .map(|pci| pci.to_string())
        .and_then(|pci| zerogravity::devices::normalize_pci(&pci).ok());
    let selected = observed_pci.as_ref().is_some_and(|pci| {
        let id = format!("pci:{pci}");
        allowed
            .iter()
            .any(|device| device.class == "gpu" && device.vendor == "amd" && device.id == id)
    });
    if !selected {
        return Err(Error::Unsupported(
            "HIP device 0 is outside the selected device policy".into(),
        ));
    }

    // HIP's own allocator/runtime reports the allocatable pool, including UMA
    // on integrated GPUs. Do not infer capacity from DRM GTT addressability.
    let ceiling = observed.total_memory;
    let available = observed.free_memory.min(ceiling);
    let margin = request.headroom.bytes(ceiling, true)?;
    let usable = available
        .checked_sub(margin)
        .ok_or_else(|| Error::Limit("HIP availability is below configured headroom".into()))?;
    zerogravity::models::HipSourceBudget::new(usable)
}
#[cfg(not(feature = "hip"))]
fn hip_budget(_: &Request, _: &devices::Inventory) -> Result<()> {
    Err(Error::Unsupported(
        "ROCm/HIP backend is not compiled".into(),
    ))
}

fn prompt(value: Option<String>) -> Result<String> {
    if let Some(value) = value {
        return Ok(value);
    }
    use std::io::Read;
    let mut text = String::new();
    std::io::stdin()
        .take(1024 * 1024 + 1)
        .read_to_string(&mut text)?;
    if text.len() > 1024 * 1024 {
        return Err(Error::Limit("prompt stdin exceeds 1 MiB".into()));
    }
    Ok(text)
}

fn receive(receipt: zerogravity::runtime::text::RequestEvents) -> Result<()> {
    while let Ok(event) = receipt.events.recv() {
        match event {
            ResponseEvent::Done {
                output,
                completion_tokens,
                ..
            } => {
                println!("{output}");
                eprintln!("zerogravity: {completion_tokens} generated tokens");
                return Ok(());
            }
            ResponseEvent::Error(error) => return Err(error.into_core()),
            ResponseEvent::Text(_) | ResponseEvent::TokenText { .. } => {}
        }
    }
    Err(Error::State(
        "model worker disconnected before terminal response".into(),
    ))
}
