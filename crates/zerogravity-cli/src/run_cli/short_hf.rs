//! Finite original CUDA route, without an HTTP router or alternate engine.
use super::*;
pub(super) fn selected(model: &zerogravity::pipeline::ModelInstance) -> bool {
    model.native_artifacts.is_some()
        || model
            .source
            .as_deref()
            .and_then(|s| match SourceSelector::parse(s).ok()? {
                SourceSelector::Local(p) => Some(p.is_dir()),
                _ => None,
            })
            .unwrap_or(false)
}
#[cfg(not(all(
    any(feature = "short-hf-serving", feature = "short-hf-run"),
    target_os = "linux"
)))]
pub(super) fn run(_: Request, _: devices::Inventory, _: Vec<devices::Observation>) -> Result<()> {
    Err(Error::Unsupported("original HF CUDA direct run requires Linux and compiled short-hf-run; no source/model/client is loaded".into()))
}
#[cfg(all(
    any(feature = "short-hf-serving", feature = "short-hf-run"),
    target_os = "linux"
))]
pub(super) fn run(
    mut request: Request,
    inventory: devices::Inventory,
    _candidates: Vec<devices::Observation>,
) -> Result<()> {
    use zerogravity::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    };
    use zerogravity::short_hf as original;
    if !matches!(request.backend, Backend::Auto | Backend::Cuda) {
        return Err(Error::Unsupported("exact HF route has only qualified CUDA numerical/native profile; no CPU/ROCm/Vulkan substitution".into()));
    }
    if request.diagnostics.audit_logits_jsonl.is_some()
        || request.diagnostics.audit_layers_jsonl.is_some()
        || request.task.teacher_ids.is_some()
    {
        return Err(Error::Unsupported(
            "finite HF direct route has no audited/teacher adapter".into(),
        ));
    }
    if request.provenance.get("task.context") == Some(&Origin::Builtin) {
        request.task.context = 512;
    }
    if request.provenance.get("task.max_tokens") == Some(&Origin::Builtin) {
        request.task.max_tokens = 8;
    }
    if request.task.context != 512 || !(1..=8).contains(&request.task.max_tokens) {
        return Err(Error::Unsupported(
            "finite HF CUDA requires context512/output1..8; explicit bounds are never shrunk"
                .into(),
        ));
    }
    let Entrypoint::Model(id) = &request.entrypoint else {
        return Err(bad("finite HF requires model entrypoint"));
    };
    let m = request
        .graph
        .models
        .iter_mut()
        .find(|m| &m.id == id)
        .ok_or_else(|| bad("entrypoint absent"))?;
    if m.model_config.is_some() || m.tokenizer.is_some() {
        return Err(bad(
            "exact HF inventory owns config/tokenizer; separate sidecars refused",
        ));
    }
    let selector = SourceSelector::parse(
        m.source
            .as_deref()
            .ok_or_else(|| bad("HF source required"))?,
    )?;
    let cached = match &selector {
        SourceSelector::Local(_) => None,
        _ => {
            let root = zerogravity::cache::paths::resolve_root(
                None,
                std::env::var_os("XDG_CACHE_HOME").as_deref(),
                std::env::var_os("HOME").as_deref(),
            )?;
            let metadata = zerogravity::cache::CacheMetadata::open(&root)?;
            let snapshot = metadata.select(&selector, m.snapshot.as_deref())?;
            Some((root, metadata.disk_limit_bytes()?, snapshot))
        }
    };
    let directory = match &selector {
        SourceSelector::Local(path) => {
            if m.snapshot.is_some() {
                return Err(bad("--snapshot is for cached source selectors"));
            }
            path.clone()
        }
        _ => {
            let (root, _, snapshot) = cached.as_ref().expect("cached selector");
            root.join("snapshots").join(&snapshot.digest).join("files")
        }
    };
    let native = PathBuf::from(m.native_artifacts.as_deref().ok_or_else(|| {
        bad("finite HF requires --native-artifacts DIR (or model.native_artifacts)")
    })?);
    let meta = original::inspect_route(&directory, &native)?;
    let target = if meta.family == "0.8b" {
        "qwen3.5-0.8b"
    } else {
        "qwen3.5-2b"
    };
    if m.target.is_empty() {
        m.target = target.into();
    } else if m.target != target {
        return Err(bad("declared target differs from exact HF source profile"));
    }
    request.graph.validate_entrypoint(&request.entrypoint)?;
    let device_growth = (meta.provider_bytes as u64)
        .checked_add(512 * 1024 * 1024)
        .ok_or_else(|| Error::Limit("finite HF profile overflow".into()))?;
    let arbiter = ResourceArbiter::process_shared();
    let choose = |observed: &devices::Inventory| -> Result<devices::Observation> {
        let allowed = devices::allowed(&request, observed)?;
        let host = observed
            .devices
            .iter()
            .find(|d| d.id == "cpu:0")
            .ok_or_else(|| Error::State("current HOST observation absent".into()))?;
        let margin = host
            .ceiling
            .map(|c| request.headroom.bytes(c, true))
            .transpose()?
            .ok_or_else(|| Error::State("HOST ceiling unknown".into()))?;
        if host
            .available
            .and_then(|a| a.checked_sub(margin))
            .is_none_or(|a| a < 4096 * 1024 * 1024)
        {
            return Err(Error::Limit(
                "finite HF original HOST4GiB plus current routing headroom does not fit".into(),
            ));
        }
        for d in allowed {
            if d.class != "gpu"
                || d.vendor != "nvidia"
                || d.cuda_ordinal.is_none()
                || d.cuda_cc != Some((8, 6))
                || d.exclusion.is_some()
            {
                continue;
            }
            let Some((capacity, available)) = d.ceiling.zip(d.available) else {
                continue;
            };
            let margin = request.headroom.bytes(capacity, false)?;
            if capacity < 5120 * 1024 * 1024
                || available
                    .checked_sub(margin)
                    .is_none_or(|a| a < device_growth)
            {
                continue;
            }
            let key = PhysicalDeviceKey::new(
                "pci",
                d.id.strip_prefix("pci:")
                    .ok_or_else(|| bad("finite GPU requires full PCI identity"))?,
            )?;
            if let Ok(s) = arbiter.snapshot(&key) {
                if s.fenced || s.limit != 5120 * 1024 * 1024 || s.reserved != 0 {
                    continue;
                }
            }
            return Ok(d);
        }
        Err(Error::Limit("no allowed source-qualified sm86 CUDA card fits original provider+512MiB backend and current free-memory headroom; no split/backend/source fallback".into()))
    };
    let inventory = if devices::fresh(&inventory).is_err() {
        devices::discover()?
    } else {
        inventory
    };
    devices::fresh(&inventory)?;
    let chosen = choose(&inventory)?;
    let explain = serde_json::json!({"schema":"zerogravity.run.short-hf.decision.v1","source_repository":meta.repository,"source_revision":meta.revision,"source_manifest_sha256":meta.manifest_sha256,"target":target,"device":chosen.id,"cuda_ordinal":chosen.cuda_ordinal,"context":512,"max_tokens":request.task.max_tokens,"provider_bytes":meta.provider_bytes,"backend_device_bytes":512*1024*1024u64,"backend_host_bytes":384*1024*1024u64,"physical_operator_cap_bytes":5120*1024*1024u64,"source_host_cap_bytes":4096*1024*1024u64,"cold_native_source_runtime_authentication":"pending original loader","prompt_eligibility":"pending original tokenizer; finite47/57/62/67/72 rows, natural output only","pre_client_admission_gap":true,"hard_os_admission":false,"source_verified":false,"performance_qualified":false,"observed_unix_ms":inventory.observed_unix_ms,"headroom":format!("{:?}",request.headroom),"devices":format!("{:?}",request.devices),"provenance":request.provenance.iter().map(|(k,v)|(k,format!("{v:?}"))).collect::<BTreeMap<_,_>>()});
    if request.diagnostics.dry_run {
        println!(
            "{}",
            serde_json::to_string_pretty(&explain).map_err(|e| Error::State(e.to_string()))?
        );
        return Ok(());
    }
    if request.diagnostics.explain {
        eprintln!(
            "{}",
            serde_json::to_string(&explain).map_err(|e| Error::State(e.to_string()))?
        );
    }
    zerogravity::gpu::validate_cuda_compiler_environment()?;
    let host = original::host_admission(ResourceOwner::new(
        "short-hf-run",
        "original-source-loader",
    )?)?;
    if host.arbiter.snapshot(&host.resource)?.reserved != 0 {
        return Err(Error::State("finite direct route requires fresh original HOST scope; existing source/context owners cannot be guessed".into()));
    }
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 128 * 1024 * 1024,
        }],
    )?;
    // The same immutable cache selection is fully verified and pinned before
    // intentional Client bootstrap. Original source inventory hashing still
    // owns model authority; the lease stays alive through original close.
    let cached_lease = if let Some((root, disk, snapshot)) = &cached {
        let store = zerogravity::cache::CacheStore::open_existing(
            zerogravity::cache::CacheOptions::new(Some(root.clone()), *disk),
        )?;
        let lease = store.open_snapshot(&snapshot.digest)?;
        if lease.manifest() != &snapshot.manifest {
            return Err(Error::State("selected cached manifest changed".into()));
        }
        Some(lease)
    } else {
        None
    };
    let prompt = match request.task.prompt.clone() {
        Some(p) => p,
        None => {
            use std::io::Read;
            let mut p = String::new();
            std::io::stdin()
                .take(1024 * 1024 + 1)
                .read_to_string(&mut p)?;
            if p.len() > 1024 * 1024 {
                return Err(Error::Limit("prompt stdin1MiB".into()));
            }
            p
        }
    };
    let tokenizer = original::route_tokenizer(&directory)?;
    let ids = tokenizer.prompt_tokens(&prompt, !request.task.raw)?;
    if !matches!(ids.len(), 47 | 57 | 62 | 67 | 72)
        || ids.iter().any(|&id| id >= 248320)
        || request.task.prefill_chunk < ids.len()
    {
        return Err(Error::Unsupported(
            "finite HF direct prompt rows47/57/62/67/72 and whole source prefill only".into(),
        ));
    }
    drop(tokenizer);
    drop(control);
    let refreshed = devices::discover()?;
    devices::fresh(&refreshed)?;
    let fresh = choose(&refreshed)?;
    if fresh.id != chosen.id || fresh.cuda_ordinal != chosen.cuda_ordinal {
        return Err(Error::State(
            "selected CUDA physical binding/availability changed before bootstrap".into(),
        ));
    }
    let physical = PhysicalDeviceKey::new(
        "pci",
        fresh
            .id
            .strip_prefix("pci:")
            .ok_or_else(|| bad("selected PCI identity"))?,
    )?;
    arbiter.register_matching(physical.clone(), 5120 * 1024 * 1024)?;
    let device = zerogravity::gpu::Device::cuda(
        fresh
            .cuda_ordinal
            .ok_or_else(|| bad("CUDA ordinal unavailable"))?,
    )
    .map_err(|e| Error::Unsupported(e.to_string()))?;
    // Intentional original runtime bootstrap. Registration above is not a
    // reservation: the existing pre-Client admission gap remains explicit.
    let actual = zerogravity::gpu::initialize_cuda_physical_identity(&device)?;
    if actual != physical {
        return Err(Error::State("runtime CUDA Client differs from selected physical card; no source/compiler work started".into()));
    }
    let result = futures_lite::future::block_on(async {
        let loaded = original::load(
            device,
            &directory,
            &native,
            &host,
            meta.provider_bytes,
            None,
        )
        .await?;
        let observation = loaded.resident().observation();
        let weak = std::sync::Arc::downgrade(loaded.resident());
        let result: Result<Vec<u32>> = async {
            if loaded.source_metadata().manifest_sha256 != meta.manifest_sha256
                || loaded
                    .tokenizer()
                    .prompt_tokens(&prompt, !request.task.raw)?
                    != ids
            {
                return Err(Error::State(
                    "loaded source/tokenizer differs from planned finite input".into(),
                ));
            }
            loaded
                .resident()
                .validate_prompt(&ids, request.task.max_tokens)?;
            let mut session = loaded.resident().session(None).await?;
            let mut output = Vec::with_capacity(request.task.max_tokens);
            let mut receipt = session.infer_prompt(&ids, None).await?;
            for i in 0..request.task.max_tokens {
                let id = receipt.selected_id;
                if id == loaded.tokenizer().eos() {
                    break;
                }
                output.push(id);
                if i + 1 < request.task.max_tokens {
                    receipt = session.infer_next(None).await?;
                }
            }
            drop(session);
            Ok(output)
        }
        .await;
        let text = result.and_then(|ids| loaded.tokenizer().decode(&ids, true));
        let close = loaded.resident().clone().close(None).await;
        drop(loaded);
        match (text, close) {
            (Ok(text), Ok(_)) => {
                let state = observation.snapshot()?;
                drop(observation);
                let physical = host.arbiter.snapshot(&state.physical_key)?;
                let host_state = host.arbiter.snapshot(&host.resource)?;
                if weak.upgrade().is_some()
                    || state.pending_events != 0
                    || state.resident_bytes != 0
                    || state.root_pool_allocated_sets != 0
                    || state.root_pool_backing_bytes != 0
                    || state.backend_device_reserved_bytes != 512 * 1024 * 1024
                    || state.backend_host_reserved_bytes != 384 * 1024 * 1024
                    || physical.reserved != state.backend_device_reserved_bytes
                    || host_state.reserved != state.backend_host_reserved_bytes
                    || physical.fenced
                    || physical
                        .active_models
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        != ["backend/cuda"]
                    || host_state
                        .active_models
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                        != ["backend/cuda"]
                {
                    return Err(Error::State("finite HF known close did not retire original model to actual backend512/384 baseline".into()));
                }
                println!("{text}");
                Ok(())
            }
            (Err(error), Err(close)) => Err(Error::State(format!(
                "finite HF inference failed: {error}; original close/retirement also failed: {close}"
            ))),
            (Err(error), Ok(_)) => Err(error),
            (_, Err(error)) => Err(error),
        }
    });
    drop(cached_lease);
    result
}
