//! CLI adapter onto the same source-bound Qwen Runtime used by serving.
use super::*;
use zerogravity::cache::{CacheMetadata, CacheOptions};
use zerogravity::models::{ModelLoadRequest, QwenServingProfile};
use zerogravity::runtime::text::{ResponseEvent, SemanticRequest, TextInput};

#[cfg(feature = "cuda")]
const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SelectedBackend {
    Cuda,
    Hip,
}

struct Resolved {
    alias: String,
    cache: CacheOptions,
    snapshot: String,
    target: String,
}

fn candidate(model: &zerogravity::pipeline::ModelInstance) -> bool {
    if model.native_artifacts.is_some() {
        return false;
    }
    target(model).is_some()
}

pub(super) fn selected_backend(
    request: &Request,
    model: &zerogravity::pipeline::ModelInstance,
    inventory: &devices::Inventory,
) -> Result<Option<SelectedBackend>> {
    if !candidate(model) {
        return Ok(None);
    }
    zerogravity::devices::fresh(inventory)?;
    let target = target(model).expect("candidate has Qwen target");
    require_source_backend(request, inventory, &target).map(Some)
}

pub(super) fn run(
    request: Request,
    inventory: devices::Inventory,
    backend: SelectedBackend,
    interactive: bool,
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
    crate::output::emit!(
        Verbose,
        "Resolved {} on {:?}; numerical: {}",
        resolved.target,
        backend,
        if native { "native" } else { "source" }
    );
    crate::output::emit!(Diagnostic, "Snapshot: {}", resolved.snapshot);
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
        crate::output::emit!(Explicit,
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

    let runtime = zerogravity::Runtime::builder().build()?;
    let result = (|| -> Result<()> {
        let profile = QwenServingProfile {
            active: 1,
            context: request.task.context,
            output: request.task.max_tokens,
            max_pending: Some(1),
            cohort_width: 0,
        };
        let load = load_request(
            &request,
            &inventory,
            backend,
            vec![resolved.alias.clone()],
            resolved.cache.clone(),
            resolved.snapshot.clone(),
            SourceLoadOptions {
                profile,
                cohort_width: None,
                device_mib: None,
            },
        )?;
        runtime
            .control()
            .load(load)
            .map_err(|error| error.into_core())?;
        let device = runtime.models().snapshot().ok().and_then(|models| {
            let owner = models
                .iter()
                .find(|owner| owner.aliases.contains(&resolved.alias))?;
            let memory = owner.capabilities.as_ref()?.details.memory.as_ref()?;
            match memory {
                zerogravity::models::ModelMemory::Resident(memory) => {
                    Some(memory.device.device.clone())
                }
                zerogravity::models::ModelMemory::ResidentText { admission, .. } => {
                    Some(admission.device.device.clone())
                }
                _ => None,
            }
        });
        let backend_name = match backend {
            SelectedBackend::Cuda => "CUDA",
            SelectedBackend::Hip => "ROCm",
        };
        if let Some(device) = device {
            crate::output::emit!(
                Status,
                "{} · {backend_name} · {device} · context {}",
                resolved.target,
                request.task.context
            );
        } else {
            crate::output::emit!(
                Status,
                "{} · {backend_name} · context {}",
                resolved.target,
                request.task.context
            );
        }
        let invoke = |prompt| {
            let receipt = runtime
                .request_semantic(SemanticRequest {
                    model: resolved.alias.clone(),
                    input: TextInput::from_text(prompt, !request.task.raw),
                    max_tokens: request.task.max_tokens,
                    stream: false,
                    token_metrics: None,
                })
                .map_err(|error| error.into_core())?;
            receive(receipt)
        };
        if interactive {
            super::input::session(&resolved.target, invoke)
        } else {
            invoke(prompt(request.task.prompt.clone())?)
        }
    })();
    let shutdown = runtime.shutdown();
    match (result, shutdown) {
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
        (Ok(()), Ok(())) => Ok(()),
    }
}

/// Cold overrides only; the loaded model and scheduler retain the resolved
/// profile. No per-request policy, alternate allocator or extra worker.
pub(crate) struct SourceLoadOptions {
    pub profile: QwenServingProfile,
    pub cohort_width: Option<usize>,
    pub device_mib: Option<usize>,
}

pub(crate) fn load_request(
    request: &Request,
    inventory: &devices::Inventory,
    backend: SelectedBackend,
    aliases: Vec<String>,
    cache: CacheOptions,
    snapshot: String,
    options: SourceLoadOptions,
) -> Result<ModelLoadRequest> {
    zerogravity::devices::fresh(inventory)?;
    let SourceLoadOptions {
        profile,
        cohort_width,
        device_mib,
    } = options;
    match backend {
        SelectedBackend::Cuda => {
            if device_mib.is_some() {
                return Err(Error::Unsupported(
                    "canonical device-mib currently requires original ROCm source".into(),
                ));
            }
            cuda_load(aliases, cache, snapshot, profile, cohort_width)
        }
        SelectedBackend::Hip => hip_load(
            aliases,
            cache,
            snapshot,
            profile
                .with_cohort(cohort_width)
                .map_err(|e| e.into_core())?,
            hip_budget(request, inventory, device_mib)?,
            request.numerical.as_deref() == Some("native"),
        ),
    }
}

fn target(model: &zerogravity::pipeline::ModelInstance) -> Option<String> {
    let source = SourceSelector::parse(model.source.as_deref()?).ok()?;
    match source {
        SourceSelector::Snapshot(digest) => {
            zerogravity::models::qwen_target_for_snapshot(&digest).map(str::to_owned)
        }
        SourceSelector::Name(name)
            if zerogravity::models::qwen_source_snapshot(&name).is_some() =>
        {
            Some(name)
        }
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
    let snapshot =
        zerogravity::models::select_cached(&metadata, &selector, model.snapshot.as_deref())?;
    if zerogravity::models::qwen_target_for_snapshot(&snapshot.digest) != Some(target.as_str()) {
        return Err(Error::Unsupported(
            "source-bound GPU run requires an exact approved Qwen snapshot".into(),
        ));
    }
    if model.model_config.is_some() || model.tokenizer.is_some() {
        return Err(bad(
            "cached source-bound run requires original role-bound sidecars",
        ));
    }
    let disk_limit = metadata.disk_limit_bytes()?;
    Ok(Resolved {
        alias: alias.to_owned(),
        cache: CacheOptions::new(Some(root), disk_limit),
        snapshot: snapshot.digest,
        target,
    })
}

pub(crate) fn select_backend(
    request: &Request,
    inventory: &devices::Inventory,
    target: &str,
) -> Result<Option<SelectedBackend>> {
    if matches!(request.backend, Backend::Cpu | Backend::Vulkan) {
        return Ok(None);
    }
    require_source_backend(request, inventory, target).map(Some)
}

/// Check actual cold bindings before selecting. Auto may choose another
/// eligible backend here, but never after a route has been selected/loaded.
pub(crate) fn require_source_backend(
    request: &Request,
    inventory: &devices::Inventory,
    target: &str,
) -> Result<SelectedBackend> {
    let allowed = zerogravity::devices::allowed(request, inventory)?;
    // Native arithmetic is a route constraint, not permission to execute
    // the source-reference CUDA numerical profile under another label.
    if request.numerical.as_deref() == Some("native") {
        if !matches!(request.backend, Backend::Auto | Backend::Rocm) {
            return Err(Error::Unsupported(
                "native numerical profile requires ROCm".into(),
            ));
        }
        require_hip(&allowed)?;
        return Ok(SelectedBackend::Hip);
    }
    match request.backend {
        Backend::Cuda => {
            require_cuda(&allowed, target, &request.headroom)?;
            Ok(SelectedBackend::Cuda)
        }
        Backend::Rocm => {
            require_hip(&allowed)?;
            Ok(SelectedBackend::Hip)
        }
        Backend::Auto => {
            let cuda = match require_cuda(&allowed, target, &request.headroom) {
                Ok(()) => return Ok(SelectedBackend::Cuda),
                Err(error) => error,
            };
            let hip = match require_hip(&allowed) {
                Ok(()) => return Ok(SelectedBackend::Hip),
                Err(error) => error,
            };
            // Retain measured byte bounds if an otherwise eligible CUDA
            // route was refused by capacity and no HIP binding is usable.
            if matches!(cuda, Error::Resource(_)) {
                return Err(cuda);
            }
            Err(Error::Unsupported(format!(
                "No eligible original-source Qwen GPU route. CUDA: {cuda}; ROCm/HIP: {hip}. Check the runtime environment and selected devices, or use a supported GGUF for CPU."
            )))
        }
        Backend::Cpu | Backend::Vulkan => Err(Error::Unsupported(
            "Original-source Qwen execution currently requires CUDA or ROCm/HIP; CPU and Vulkan are not eligible for this source route. Select an eligible GPU or a supported GGUF for CPU.".into(),
        )),
    }
}

#[cfg(feature = "acquisition")]
pub(super) fn preflight_acquisition(
    request: &Request,
    inventory: &devices::Inventory,
    backend: SelectedBackend,
) -> Result<()> {
    zerogravity::devices::fresh(inventory)?;
    if request.numerical.as_deref() == Some("native") && backend != SelectedBackend::Hip {
        return Err(Error::Unsupported(
            "native numerical profile requires ROCm".into(),
        ));
    }
    if request.task.teacher_ids.is_some()
        || request.diagnostics.audit_logits_jsonl.is_some()
        || request.diagnostics.audit_layers_jsonl.is_some()
        || request.graph.models.iter().any(|model| {
            model.model_config.is_some()
                || model.tokenizer.is_some()
                || model.native_artifacts.is_some()
        })
    {
        return Err(Error::Unsupported("automatic original-source acquisition requires ordinary role-bound source execution without diagnostic overrides".into()));
    }
    if backend == SelectedBackend::Hip {
        let _budget = hip_budget(request, inventory, None)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn cuda_fit(device: &devices::Observation, headroom: &Headroom) -> Result<u64> {
    let ceiling = device
        .ceiling
        .ok_or_else(|| Error::State("CUDA capacity unknown".into()))?;
    let available = device
        .available
        .ok_or_else(|| Error::State("CUDA availability unknown".into()))?;
    Ok(available
        .min(ceiling)
        .saturating_sub(headroom.bytes(ceiling, false)?))
}

fn require_cuda(devices: &[devices::Observation], target: &str, headroom: &Headroom) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (devices, target, headroom);
        return Err(Error::Unsupported("CUDA backend is not compiled".into()));
    }
    #[cfg(feature = "cuda")]
    {
        if target == "qwen3.8-27b" {
            return Err(Error::Unsupported(
                "Original Qwen 27B CUDA execution is not yet qualified; select the eligible ROCm route".into(),
            ));
        }
        let device = devices
            .iter()
            .find(|device| {
                device.class == "gpu" && device.vendor == "nvidia" && device.cuda_ordinal == Some(0)
            })
            .ok_or_else(|| {
                let nvidia = devices.iter().filter(|device|
                    device.class == "gpu" && device.vendor == "nvidia");
                let observed = nvidia.clone().any(|device| device.cuda_ordinal.is_some());
                let present = nvidia.count() != 0;
                Error::Unsupported(if observed {
                    "Original Qwen CUDA requires selected CUDA device 0; another ordinal is not qualified"
                } else if present {
                    "Selected NVIDIA GPU has no usable CUDA runtime/driver binding; check installed driver and packaged runtime libraries"
                } else {
                    "No NVIDIA GPU in the selected device policy"
                }.into())
            })?;
        if device.cuda_cc != Some((8, 6)) {
            return Err(Error::Unsupported(format!(
                "current Qwen CUDA source kernels are SM86-qualified; selected device reports compute capability {}",
                device
                    .cuda_cc
                    .map(|(major, minor)| format!("{major}.{minor}"))
                    .unwrap_or_else(|| "unknown".into())
            )));
        }
        zerogravity::gpu::cuda::compiler::validate_environment()?;
        let available = cuda_fit(device, headroom)?;
        if available < 5 * GIB {
            return Err(Error::resource(
                "CUDA VRAM after headroom",
                "original-source Qwen admission",
                5 * GIB,
                available,
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
        observe_selected_hip(devices).map(|_| ())
    }
}

#[cfg(feature = "cuda")]
fn cuda_load(
    aliases: Vec<String>,
    cache: CacheOptions,
    snapshot: String,
    profile: QwenServingProfile,
    cohort_width: Option<usize>,
) -> Result<ModelLoadRequest> {
    Ok(match cohort_width {
        None => ModelLoadRequest::qwen_cuda(aliases, cache, snapshot, profile),
        Some(width) => ModelLoadRequest::qwen_cuda_with_smart_config(
            aliases,
            cache,
            snapshot,
            profile,
            zerogravity::models::SmartConfig::enabled(
                zerogravity::models::QwenCudaSmartOverrides {
                    cohort_width: Some(width),
                    ..Default::default()
                },
            ),
        ),
    })
}
#[cfg(not(feature = "cuda"))]
fn cuda_load(
    _: Vec<String>,
    _: CacheOptions,
    _: String,
    _: QwenServingProfile,
    _: Option<usize>,
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
fn observe_selected_hip(
    devices: &[devices::Observation],
) -> Result<zerogravity::gpu::HipDeviceMetadata> {
    if !devices
        .iter()
        .any(|device| device.class == "gpu" && device.vendor == "amd")
    {
        return Err(Error::Unsupported(
            "No AMD GPU in the selected device policy".into(),
        ));
    }
    if !zerogravity::gpu::hip_runtime_libraries_available() {
        return Err(Error::Unsupported(
            "ROCm/HIP runtime libraries unavailable; check the packaged runtime environment".into(),
        ));
    }
    let observed = zerogravity::gpu::observe_hip_device(Default::default())
        .ok_or_else(|| Error::Unsupported("HIP runtime cannot observe device 0; check driver/device access and runtime compatibility".into()))?;
    let selected = observed
        .physical
        .pci_address
        .and_then(|pci| zerogravity::devices::normalize_pci(&pci.to_string()).ok())
        .is_some_and(|pci| {
            let id = format!("pci:{pci}");
            devices
                .iter()
                .any(|device| device.class == "gpu" && device.vendor == "amd" && device.id == id)
        });
    if !selected {
        return Err(Error::Unsupported("HIP device 0 is outside the selected device policy or its physical identity is unavailable".into()));
    }
    Ok(observed)
}

#[cfg(feature = "hip")]
fn hip_budget(
    request: &Request,
    inventory: &devices::Inventory,
    device_mib: Option<usize>,
) -> Result<zerogravity::models::HipSourceBudget> {
    let allowed = zerogravity::devices::allowed(request, inventory)?;
    let observed = observe_selected_hip(&allowed)?;

    // HIP's own allocator/runtime reports the allocatable pool, including UMA
    // on integrated GPUs. Do not infer capacity from DRM GTT addressability.
    let ceiling = observed.total_memory;
    let available = observed.free_memory.min(ceiling);
    let margin = request.headroom.bytes(ceiling, true)?;
    let usable = available
        .checked_sub(margin)
        .ok_or_else(|| Error::Limit("HIP availability is below configured headroom".into()))?;
    let requested = device_mib
        .map(|mib| {
            u64::try_from(mib)
                .ok()
                .and_then(|n| n.checked_mul(1024 * 1024))
                .ok_or_else(|| Error::Limit("device-mib overflow".into()))
        })
        .transpose()?;
    zerogravity::models::HipSourceBudget::within_available(usable, requested)
}
#[cfg(not(feature = "hip"))]
fn hip_budget(_: &Request, _: &devices::Inventory, _: Option<usize>) -> Result<()> {
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
                crate::output::emit!(Status, "zerogravity: {completion_tokens} generated tokens");
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
