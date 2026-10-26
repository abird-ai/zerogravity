//! Finite original CUDA diagnostic route.
use super::*;
/// Finite route scalars the card proposal and refusal trace need; separated
/// from `ShortHfRouteMetadata` so the pure proposal is unit-testable without
/// a real source directory.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) struct FiniteRouteProfile {
    /// Selected finite CUDA HOST requirement (meta.host_bytes).
    pub host_bytes: usize,
    /// Original provider bytes (meta.provider_bytes).
    pub provider_bytes: usize,
}

/// Pure finite-card proposal over one inventory snapshot. Returns the first
/// allowed source-qualified sm86 CUDA card plus ordered exclusion reasons
/// for every other scanned device. A host-fit failure suppresses every
/// proposal (the route still needs the original HOST admission) while the
/// card scan continues, so one trace can hold both host and card reasons.
/// No arbiter mutation, no Client, no I/O beyond the caller-supplied
/// snapshot state.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) fn select_sm86_card(
    request: &Request,
    profile: &FiniteRouteProfile,
    inventory: &devices::Inventory,
    arbiter: &zerogravity::resource_arbiter::ResourceArbiter,
) -> Result<(Option<devices::Observation>, Vec<String>)> {
    use zerogravity::resource_arbiter::PhysicalDeviceKey;
    let allowed = devices::allowed(request, inventory)?;
    let mut exclusions = Vec::new();
    let host = inventory
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
        .is_none_or(|a| a < profile.host_bytes as u64)
    {
        exclusions.push(format!(
            "host: finite HF original HOST{}MiB plus current routing headroom does not fit",
            profile.host_bytes / (1024 * 1024)
        ));
    }
    let host_unfit = exclusions.iter().any(|e| e.starts_with("host:"));
    let device_growth = (profile.provider_bytes as u64)
        .checked_add(512 * 1024 * 1024)
        .ok_or_else(|| Error::Limit("finite HF profile overflow".into()))?;
    for d in &allowed {
        if d.class != "gpu" {
            continue;
        }
        if let Some(reason) = &d.exclusion {
            exclusions.push(format!("{}: {reason}", d.id));
            continue;
        }
        if d.vendor != "nvidia" || d.cuda_ordinal.is_none() || d.cuda_cc != Some((8, 6)) {
            exclusions.push(format!(
                "{}: finite Qwen CUDA route requires qualified sm86 NVIDIA CUDA card",
                d.id
            ));
            continue;
        }
        let Some((capacity, available)) = d.ceiling.zip(d.available) else {
            exclusions.push(format!(
                "{}: current capacity/free observation unavailable",
                d.id
            ));
            continue;
        };
        let margin = request.headroom.bytes(capacity, false)?;
        if capacity < 5120 * 1024 * 1024 {
            exclusions.push(format!(
                "{}: card capacity below the 5120MiB operator cap",
                d.id
            ));
            continue;
        }
        if available
            .checked_sub(margin)
            .is_none_or(|a| a < device_growth)
        {
            exclusions.push(format!(
                "{}: insufficient current free memory after routing headroom",
                d.id
            ));
            continue;
        }
        let key = PhysicalDeviceKey::new(
            "pci",
            d.id.strip_prefix("pci:")
                .ok_or_else(|| bad("finite GPU requires full PCI identity"))?,
        )?;
        if let Ok(s) = arbiter.snapshot(&key)
            && (s.fenced || s.limit != 5120 * 1024 * 1024 || s.reserved != 0)
        {
            exclusions.push(format!(
                "{}: card already serves an in-process route reservation",
                d.id
            ));
            continue;
        }
        if host_unfit {
            continue;
        }
        return Ok((Some(d.clone()), exclusions));
    }
    Ok((None, exclusions))
}

/// Exact refusal when nothing proposes; a host-fit failure keeps today's
/// host limit, otherwise the no-card limit stays verbatim.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) fn no_route_refusal(
    profile: &FiniteRouteProfile,
    exclusions: &[String],
) -> Error {
    if exclusions.iter().any(|e| e.starts_with("host:")) {
        Error::Limit(format!(
            "finite HF original HOST{}MiB plus current routing headroom does not fit",
            profile.host_bytes / (1024 * 1024)
        ))
    } else {
        Error::Limit("no allowed source-qualified sm86 CUDA card fits original provider+512MiB backend and current free-memory headroom; no split/backend/source fallback".into())
    }
}

/// Dry-run status fields shared with the generic decision trace: a dry-run
/// proposal is never a reservation.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) fn dry_run_trace_status(trace: &mut serde_json::Value) {
    trace["dry_run"] = serde_json::json!(true);
    trace["reservation"] = serde_json::json!("none");
}

/// A dry-run holds nothing; the finite adapter keeps reporting the explicit
/// pre-Client gap until a route actually boots a Client.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) fn dry_run_admission_trace(trace: &mut serde_json::Value) {
    trace["pre_client_admission_gap"] = serde_json::json!(true);
}

/// Executing --explain trace: the transactional seam (below) replaced the
/// open pre-Client gap; the boundary stays advisory exactly as before.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) fn executing_admission_trace(trace: &mut serde_json::Value) {
    trace["pre_client_admission"] =
        serde_json::json!("transactional card hold before Client; released at verified identity");
    trace["advisory_boundary"] = serde_json::json!(
        "other processes and driver allocations; post-release pre-backend window"
    );
}

/// Transactional pre-Client card admission for the finite CUDA route.
/// `register_matching` pins the immutable 5 GiB operator cap, then ONE
/// mutex-guarded `admit` reserves the entire finite card budget so a
/// concurrent in-process route cannot double-admit this card while the
/// primary context is bootstrapped. Hold the lease until
/// `initialize_cuda_physical_identity` reports the verified physical key,
/// then drop it; the native backend/compiler-context reservation owns the
/// ledger from its own admission onward. This is NOT an OS hard memory
/// guarantee: other processes and driver allocations stay advisory.
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(crate) fn admit_pre_client_card(
    arbiter: &zerogravity::resource_arbiter::ResourceArbiter,
    physical: &zerogravity::resource_arbiter::PhysicalDeviceKey,
) -> Result<zerogravity::resource_arbiter::GlobalResourceLease> {
    use zerogravity::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
    arbiter.register_matching(physical.clone(), 5120 * 1024 * 1024)?;
    arbiter.admit(
        physical,
        ResourceOwner::new("qwen35-cuda", "pre-client-card")?,
        &[AllocationRequest {
            kind: AllocationClass::ProviderEnvelope,
            bytes: 5120 * 1024 * 1024,
        }],
    )
}

#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn same_host(
    a: &zerogravity::resource_arbiter::DeviceBudgetSnapshot,
    b: &zerogravity::resource_arbiter::DeviceBudgetSnapshot,
) -> bool {
    a.limit == b.limit
        && a.reserved == b.reserved
        && a.remaining == b.remaining
        && a.reservations == b.reservations
        && a.per_class == b.per_class
        && a.active_models == b.active_models
        && a.fenced == b.fenced
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn fresh_host(
    snapshot: &zerogravity::resource_arbiter::DeviceBudgetSnapshot,
    limit: u64,
) -> Result<()> {
    if snapshot.limit != limit
        || snapshot.reserved != 0
        || snapshot.remaining != limit
        || snapshot.reservations != 0
        || !snapshot.per_class.is_empty()
        || !snapshot.active_models.is_empty()
        || snapshot.fenced
    {
        return Err(Error::State(
            "finite direct route requires fresh original HOST scope; existing source/context owners cannot be guessed".into(),
        ));
    }
    Ok(())
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn bootstrap_baseline(
    host: &zerogravity::cache::TransferAdmission,
    limit: u64,
) -> Result<zerogravity::resource_arbiter::DeviceBudgetSnapshot> {
    use zerogravity::resource_arbiter::{
        AllocationClass, DeviceBudgetSnapshot, ResourceOwner, lease_actual_inventory_bytes,
    };
    let bootstrap =
        zerogravity::tokenize_admission::approve_arbiter(host, "qwen35-cuda-bootstrap")?;
    bootstrap.validate_domain(host)?;
    let plan = bootstrap.plan();
    let owner = ResourceOwner::new(&host.owner.model, "qwen35-cuda-bootstrap")?;
    let inventory = lease_actual_inventory_bytes(&host.resource, &owner, 2)?;
    let reserved = plan
        .bootstrap_bytes
        .checked_add(inventory)
        .ok_or_else(|| Error::Limit("finite HF bootstrap baseline sum".into()))?;
    let mut per_class = std::collections::BTreeMap::new();
    per_class.insert(AllocationClass::ImmutableWeights, plan.bootstrap_bytes);
    per_class.insert(AllocationClass::Other, inventory);
    let mut active_models = std::collections::BTreeSet::new();
    active_models.insert(host.owner.model.clone());
    let expected = DeviceBudgetSnapshot {
        limit,
        reserved,
        remaining: limit
            .checked_sub(reserved)
            .ok_or_else(|| Error::Limit("finite HF bootstrap baseline remaining".into()))?,
        reservations: 1,
        fenced: false,
        per_class,
        active_models,
    };
    let actual = host.arbiter.snapshot(&host.resource)?;
    if !same_host(&expected, &actual) {
        return Err(Error::State(format!(
            "finite HF bootstrap baseline differs: expected {expected:?}, actual {actual:?}"
        )));
    }
    Ok(actual)
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn after_backend(
    baseline: &zerogravity::resource_arbiter::DeviceBudgetSnapshot,
    backend_bytes: u64,
) -> Result<zerogravity::resource_arbiter::DeviceBudgetSnapshot> {
    use zerogravity::resource_arbiter::AllocationClass;
    if backend_bytes != 384 * 1024 * 1024 {
        return Err(Error::Limit(
            "finite HF backend HOST reservation must be exactly 384 MiB".into(),
        ));
    }
    let mut expected = baseline.clone();
    expected.reserved = expected
        .reserved
        .checked_add(backend_bytes)
        .ok_or_else(|| Error::Limit("finite HF backend baseline sum".into()))?;
    expected.remaining = expected
        .limit
        .checked_sub(expected.reserved)
        .ok_or_else(|| Error::Limit("finite HF backend baseline remaining".into()))?;
    expected.reservations = expected
        .reservations
        .checked_add(1)
        .ok_or_else(|| Error::Limit("finite HF backend baseline reservation count".into()))?;
    *expected
        .per_class
        .entry(AllocationClass::Scratch)
        .or_default() += backend_bytes;
    expected.active_models.insert("backend/cuda".into());
    Ok(expected)
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn request_credit_parts(
    host: &zerogravity::cache::TransferAdmission,
    session: &str,
    bytes: u64,
) -> Result<(u64, u64)> {
    use zerogravity::resource_arbiter::{ResourceOwner, lease_actual_inventory_bytes};
    if bytes == 0 {
        return Err(Error::Limit("zero finite HF request payload".into()));
    }
    let owner = ResourceOwner::new(&host.owner.model, session)?;
    let inventory = lease_actual_inventory_bytes(&host.resource, &owner, 2)?;
    Ok((bytes, inventory))
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn admit_request_credit(
    host: &zerogravity::cache::TransferAdmission,
    session: &str,
    payload: u64,
    inventory: u64,
) -> Result<zerogravity::resource_arbiter::GlobalResourceLease> {
    use zerogravity::resource_arbiter::{AllocationClass, AllocationRequest, ResourceOwner};
    if payload == 0 || inventory == 0 {
        return Err(Error::Limit("zero finite HF request credit".into()));
    }
    host.arbiter.admit(
        &host.resource,
        ResourceOwner::new(&host.owner.model, session)?,
        &[
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: payload,
            },
            AllocationRequest {
                kind: AllocationClass::Other,
                bytes: inventory,
            },
        ],
    )
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn request_credit(
    host: &zerogravity::cache::TransferAdmission,
    session: &str,
    bytes: u64,
) -> Result<zerogravity::resource_arbiter::GlobalResourceLease> {
    let (payload, inventory) = request_credit_parts(host, session, bytes)?;
    admit_request_credit(host, session, payload, inventory)
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn prompt_payload_bytes(
    tokenizer: &zerogravity::qwen35::tokenize::QwenTokenizer,
    prompt: &str,
    prompt_capacity: usize,
    chat: bool,
) -> Result<(u64, usize)> {
    let id_capacity = tokenizer.prompt_token_bound(prompt.len(), chat);
    let ids =
        u64::try_from(id_capacity).map_err(|_| Error::Limit("planned prompt ID bound".into()))?;
    let ids = ids
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("planned prompt ID bytes".into()))?;
    let text = u64::try_from(prompt_capacity)
        .map_err(|_| Error::Limit("planned prompt text capacity".into()))?;
    let payload = text
        .checked_add(ids)
        .ok_or_else(|| Error::Limit("planned prompt payload sum".into()))?;
    Ok((payload, id_capacity))
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn decode_payload_bytes(
    tokenizer: &zerogravity::qwen35::tokenize::QwenTokenizer,
    tokens: usize,
) -> Result<u64> {
    let scratch = u64::try_from(tokenizer.decode_scratch_bound(tokens))
        .map_err(|_| Error::Limit("decoded text scratch".into()))?;
    let per_token = u64::try_from(tokenizer.max_token_bytes()?)
        .map_err(|_| Error::Limit("decoded token bytes".into()))?;
    let text = u64::try_from(tokens)
        .ok()
        .and_then(|n| n.checked_mul(per_token))
        .ok_or_else(|| Error::Limit("decoded text bytes".into()))?;
    scratch
        .checked_add(text)
        .ok_or_else(|| Error::Limit("decoded payload sum".into()))
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
fn with_request_credit(
    baseline: &zerogravity::resource_arbiter::DeviceBudgetSnapshot,
    model: &str,
    payload: u64,
    inventory: u64,
) -> Result<zerogravity::resource_arbiter::DeviceBudgetSnapshot> {
    use zerogravity::resource_arbiter::AllocationClass;
    if payload == 0 || inventory == 0 {
        return Err(Error::Limit("zero finite HF request credit".into()));
    }
    let mut expected = baseline.clone();
    expected.reserved = expected
        .reserved
        .checked_add(payload)
        .and_then(|n| n.checked_add(inventory))
        .ok_or_else(|| Error::Limit("finite HF request baseline sum".into()))?;
    expected.remaining = expected
        .limit
        .checked_sub(expected.reserved)
        .ok_or_else(|| Error::Limit("finite HF request baseline remaining".into()))?;
    expected.reservations = expected
        .reservations
        .checked_add(1)
        .ok_or_else(|| Error::Limit("finite HF request reservation count".into()))?;
    *expected
        .per_class
        .entry(AllocationClass::Scratch)
        .or_default() += payload;
    *expected
        .per_class
        .entry(AllocationClass::Other)
        .or_default() += inventory;
    expected.active_models.insert(model.into());
    Ok(expected)
}
#[cfg(not(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
)))]
pub(in crate::run_cli) fn run(
    _: Request,
    _: devices::Inventory,
    _: Vec<devices::Observation>,
) -> Result<()> {
    Err(Error::Unsupported("original HF CUDA direct run requires Linux and compiled qwen35-cuda; no source/model/client is loaded".into()))
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux"
))]
pub(in crate::run_cli) fn run(
    mut request: Request,
    inventory: devices::Inventory,
    _candidates: Vec<devices::Observation>,
) -> Result<()> {
    use zerogravity::qwen35_cuda as original;
    use zerogravity::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        lease_inventory_bytes,
    };
    // Payload allocations stay before the leases that price them on every exit.
    struct RequestHold {
        prompt: String,
        ids: Option<Vec<u32>>,
        output: Option<Vec<u32>>,
        tokenizer: Option<original::RouteTokenizer>,
        prompt_scratch: Option<zerogravity::resource_arbiter::GlobalResourceLease>,
        decode_payload: u64,
        decode_inventory: u64,
        decode_credit: Option<zerogravity::resource_arbiter::GlobalResourceLease>,
        input_credit: Option<zerogravity::resource_arbiter::GlobalResourceLease>,
        output_credit: Option<zerogravity::resource_arbiter::GlobalResourceLease>,
    }
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
    // Direct Qwen CUDA selects the explicit 8 GiB HOST profile. Check current
    // RAM/headroom before metadata/native parsing, prompt reads or Client work.
    qwen35_cuda_host_available(
        (original::HOST_MIB_MAX * 1024 * 1024) as u64,
        &request.headroom,
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
    let arbiter = ResourceArbiter::process_shared();
    let profile = FiniteRouteProfile {
        host_bytes: meta.host_bytes,
        provider_bytes: meta.provider_bytes,
    };
    let inventory = if devices::fresh(&inventory).is_err() {
        devices::discover()?
    } else {
        inventory
    };
    devices::fresh(&inventory)?;
    let (proposed, exclusions) = select_sm86_card(&request, &profile, &inventory, &arbiter)?;
    let decision = serde_json::json!({"schema":"zerogravity.run.qwen35-cuda.decision.v1","source_repository":meta.repository,"source_revision":meta.revision,"source_manifest_sha256":meta.manifest_sha256,"target":target,"device":proposed.as_ref().map(|d|d.id.clone()),"cuda_ordinal":proposed.as_ref().and_then(|d|d.cuda_ordinal),"context":512,"max_tokens":request.task.max_tokens,"provider_bytes":meta.provider_bytes,"backend_device_bytes":512*1024*1024u64,"backend_host_bytes":384*1024*1024u64,"physical_operator_cap_bytes":5120*1024*1024u64,"source_host_cap_bytes":meta.host_bytes,"cold_native_source_runtime_authentication":"pending original loader","prompt_eligibility":"pending original tokenizer; finite47/57/62/67/72 rows, natural output only","excluded":exclusions,"hard_os_admission":false,"source_verified":false,"performance_qualified":false,"observed_unix_ms":inventory.observed_unix_ms,"headroom":format!("{:?}",request.headroom),"devices":format!("{:?}",request.devices),"provenance":request.provenance.iter().map(|(k,v)|(k,format!("{v:?}"))).collect::<BTreeMap<_,_>>()});
    if request.diagnostics.dry_run {
        // Ordering invariant: no register/register_matching/admit/stdin read/
        // audit-file open may precede this return; NVML init and cuDevice
        // enumeration stay permitted (bounded read-only observation).
        let mut trace = decision.clone();
        dry_run_trace_status(&mut trace);
        dry_run_admission_trace(&mut trace);
        println!(
            "{}",
            serde_json::to_string_pretty(&trace).map_err(|e| Error::State(e.to_string()))?
        );
        return match proposed {
            Some(_) => Ok(()),
            None => Err(no_route_refusal(&profile, &exclusions)),
        };
    }
    if request.diagnostics.explain {
        let mut trace = decision.clone();
        executing_admission_trace(&mut trace);
        eprintln!(
            "{}",
            serde_json::to_string(&trace).map_err(|e| Error::State(e.to_string()))?
        );
    }
    let Some(chosen) = proposed else {
        return Err(no_route_refusal(&profile, &exclusions));
    };
    zerogravity::gpu::cuda::compiler::validate_environment()?;
    let host = original::host_admission_mib(
        meta.host_bytes / (1024 * 1024),
        ResourceOwner::new("qwen35-cuda", "original-source-loader")?,
    )?;
    let baseline = host.arbiter.snapshot(&host.resource)?;
    fresh_host(&baseline, meta.host_bytes as u64)?;
    let baseline = bootstrap_baseline(&host, meta.host_bytes as u64)?;
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 128 * 1024 * 1024,
            },
            AllocationRequest {
                kind: AllocationClass::Other,
                bytes: lease_inventory_bytes(&host.owner.model, &host.owner.session, 2)?,
            },
        ],
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
    let chat = !request.task.raw;
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
    // Payloads are declared before their pricing leases so every early return
    // destroys the prompt/ID text and vectors before releasing their credits.
    let mut held = RequestHold {
        prompt,
        ids: None,
        output: None,
        tokenizer: None,
        prompt_scratch: None,
        decode_payload: 0,
        decode_inventory: 0,
        decode_credit: None,
        input_credit: None,
        output_credit: None,
    };
    held.tokenizer = Some(original::route_tokenizer(&directory, &host)?);
    let prompt_scratch = request_credit(
        &host,
        "qwen35-cuda-prompt",
        held.tokenizer
            .as_ref()
            .expect("route tokenizer")
            .tokenizer()
            .admitted_prompt_scratch_bound(held.prompt.len(), chat)?,
    )?;
    held.prompt_scratch = Some(prompt_scratch);
    // Named input payload: the held prompt's actual String backing plus the
    // planned prompt-ID vector capacity. Admit it before `prompt_tokens`
    // allocates either owner.
    let (input_payload, prompt_id_capacity) = prompt_payload_bytes(
        held.tokenizer
            .as_ref()
            .expect("route tokenizer")
            .tokenizer(),
        &held.prompt,
        held.prompt.capacity(),
        chat,
    )?;
    let (input_scratch, input_inventory) =
        request_credit_parts(&host, "qwen35-cuda-input", input_payload)?;
    held.input_credit = Some(admit_request_credit(
        &host,
        "qwen35-cuda-input",
        input_scratch,
        input_inventory,
    )?);
    // Named output-ID payload, admitted before the generated-output vector.
    let output_scratch = u64::try_from(request.task.max_tokens)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Limit("finite HF output ID bytes".into()))?;
    let (output_payload, output_inventory) =
        request_credit_parts(&host, "qwen35-cuda-output", output_scratch)?;
    held.output_credit = Some(admit_request_credit(
        &host,
        "qwen35-cuda-output",
        output_payload,
        output_inventory,
    )?);
    let ids = held
        .tokenizer
        .as_ref()
        .expect("route tokenizer")
        .tokenizer()
        .prompt_tokens(&held.prompt, chat)?;
    // `to_vec` may retain capacity above its length; it must fit the pinned
    // construction bound before it leaves encoder scratch.
    if ids.len() > prompt_id_capacity || ids.capacity() > prompt_id_capacity {
        return Err(Error::Limit(
            "prompt ID vector capacity exceeds the admitted bound".into(),
        ));
    }
    if !matches!(ids.len(), 47 | 57 | 62 | 67 | 72)
        || ids.iter().any(|&id| id >= 248320)
        || request.task.prefill_chunk < ids.len()
    {
        return Err(Error::Unsupported(
            "finite HF direct prompt rows47/57/62/67/72 and whole source prefill only".into(),
        ));
    }
    held.ids = Some(ids);
    drop(held.tokenizer.take());
    drop(held.prompt_scratch.take());
    drop(control);
    let after_route = host.arbiter.snapshot(&host.resource)?;
    let expected_route = with_request_credit(
        &with_request_credit(&baseline, &host.owner.model, input_scratch, input_inventory)?,
        &host.owner.model,
        output_payload,
        output_inventory,
    )?;
    if !same_host(&expected_route, &after_route) {
        return Err(Error::State(format!(
            "finite HF route did not return to exact bootstrap+known request credits: expected {expected_route:?}, actual {after_route:?}"
        )));
    }
    let refreshed = devices::discover()?;
    devices::fresh(&refreshed)?;
    let (fresh, fresh_exclusions) = select_sm86_card(&request, &profile, &refreshed, &arbiter)?;
    let Some(fresh) = fresh else {
        return Err(no_route_refusal(&profile, &fresh_exclusions));
    };
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
    // Transactional check-and-admit BEFORE intentional Client bootstrap: the
    // whole finite card budget is held while the primary context is created,
    // so concurrent in-process routes cannot double-admit this card. Dropped
    // at verified identity, before `original::load`, so the exact rotary/close
    // equations stay untouched; the native backend reservation then owns the
    // ledger from its own admission onward. Called through the one shared
    // `run_cli` seam binding; the serving frontend uses the same seam.
    let pre_client = admit_pre_client_card(&arbiter, &physical)?;
    let device = zerogravity::gpu::Device::cuda(
        fresh
            .cuda_ordinal
            .ok_or_else(|| bad("CUDA ordinal unavailable"))?,
    )
    .map_err(|e| Error::Unsupported(e.to_string()))?;
    let actual = zerogravity::gpu::initialize_cuda_physical_identity(&device)?;
    drop(pre_client);
    if actual != physical {
        return Err(Error::State("runtime CUDA Client differs from selected physical card; no source/compiler work started".into()));
    }
    let result = futures_lite::future::block_on(async move {
        let loaded = original::qualification::load(
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
            if loaded.source_metadata().manifest_sha256 != meta.manifest_sha256 {
                return Err(Error::State(
                    "loaded source/tokenizer differs from planned finite input".into(),
                ));
            }
            let prompt_scratch = request_credit(
                &host,
                "qwen35-cuda-prompt",
                loaded
                    .tokenizer()
                    .admitted_prompt_scratch_bound(held.prompt.len(), chat)?,
            )?;
            let actual = loaded.tokenizer().prompt_tokens(&held.prompt, chat)?;
            let planned = held
                .ids
                .as_ref()
                .ok_or_else(|| Error::State("planned prompt IDs absent".into()))?;
            if actual != *planned {
                return Err(Error::State(
                    "loaded source/tokenizer differs from planned finite input".into(),
                ));
            }
            // The second-pass actual IDs are destroyed before their scratch
            // credit; the first-pass IDs and prompt remain named input payloads.
            drop(actual);
            drop(prompt_scratch);
            loaded.validate_prompt(planned, request.task.max_tokens)?;
            let mut session = loaded.session(None).await?;
            let mut output = Vec::with_capacity(request.task.max_tokens);
            if output.capacity() > request.task.max_tokens {
                return Err(Error::Limit(
                    "generated output ID capacity exceeds the admitted max_tokens".into(),
                ));
            }
            let mut receipt = session.infer_prompt(planned, None).await?;
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
        let text = match result {
            Ok(ids) => {
                held.output = Some(ids);
                let ids = held.output.as_ref().expect("generated output IDs");
                match decode_payload_bytes(loaded.tokenizer(), ids.len()).and_then(|payload| {
                    request_credit_parts(&host, "qwen35-cuda-decode", payload).and_then(
                        |(payload, inventory)| {
                            admit_request_credit(&host, "qwen35-cuda-decode", payload, inventory)
                                .map(|credit| (payload, inventory, credit))
                        },
                    )
                }) {
                    Ok((payload, inventory, credit)) => {
                        held.decode_payload = payload;
                        held.decode_inventory = inventory;
                        held.decode_credit = Some(credit);
                        loaded.tokenizer().decode(ids, true)
                    }
                    Err(error) => Err(error),
                }
            }
            Err(error) => Err(error),
        };
        let close = loaded.resident().clone().close(None).await;
        drop(loaded);
        let closed = match (text, close) {
            (Ok(text), Ok(_)) => (|| -> Result<()> {
                println!("{text}");
                drop(text);
                let state = observation.snapshot()?;
                drop(observation);
                let physical = host.arbiter.snapshot(&state.physical_key)?;
                let host_state = host.arbiter.snapshot(&host.resource)?;
                let expected_host = with_request_credit(
                    &with_request_credit(
                        &after_backend(&baseline, state.backend_host_reserved_bytes)?,
                        &host.owner.model,
                        input_scratch,
                        input_inventory,
                    )?,
                    &host.owner.model,
                    output_payload,
                    output_inventory,
                )?;
                let expected_host = match held.decode_credit.as_ref() {
                    Some(_) => with_request_credit(
                        &expected_host,
                        &host.owner.model,
                        held.decode_payload,
                        held.decode_inventory,
                    )?,
                    None => {
                        return Err(Error::State(
                            "finite HF decoded text without its payload credit".into(),
                        ));
                    }
                };
                if weak.upgrade().is_some()
                    || state.pending_events != 0
                    || state.resident_bytes != 0
                    || state.root_pool_allocated_sets != 0
                    || state.root_pool_backing_bytes != 0
                    || state.backend_device_reserved_bytes != 512 * 1024 * 1024
                    || state.backend_host_reserved_bytes != 384 * 1024 * 1024
                    || physical.reserved != state.backend_device_reserved_bytes
                    || !same_host(&expected_host, &host_state)
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
                        != ["backend/cuda", "qwen35-cuda"]
                {
                    return Err(Error::State(format!(
                        "finite HF known close did not retire original model to exact bootstrap+384MiB backend baseline: expected host {expected_host:?}, actual host {host_state:?}"
                    )));
                }
                Ok(())
            })(),
            (Err(error), Err(close)) => {
                drop(observation);
                Err(Error::State(format!(
                    "finite HF inference failed: {error}; original close/retirement also failed: {close}"
                )))
            }
            (Err(error), Ok(_)) => {
                drop(observation);
                Err(error)
            }
            (_, Err(error)) => {
                drop(observation);
                Err(error)
            }
        };
        // Destroy named request payloads before their pricing leases, then
        // require the exact post-output bootstrap+384MiB backend baseline.
        drop(held);
        let final_host = host.arbiter.snapshot(&host.resource)?;
        let expected_final = after_backend(&baseline, 384 * 1024 * 1024)?;
        match (closed, same_host(&expected_final, &final_host)) {
            (Ok(()), true) => Ok(()),
            (Err(error), true) => Err(error),
            (Ok(()), false) => Err(Error::State(format!(
                "finite HF final post-output host state differs from bootstrap+384MiB backend: expected {expected_final:?}, actual {final_host:?}"
            ))),
            (Err(error), false) => Err(Error::State(format!(
                "finite HF operational failure: {error}; final post-output host state differs from bootstrap+384MiB backend: expected {expected_final:?}, actual {final_host:?}"
            ))),
        }
    });
    drop(cached_lease);
    result
}
#[cfg(all(
    any(feature = "qwen35-cuda-serving", feature = "qwen35-cuda"),
    target_os = "linux",
    test
))]
mod baseline_tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use zerogravity::resource_arbiter::{AllocationClass, DeviceBudgetSnapshot};

    fn baseline() -> DeviceBudgetSnapshot {
        let mut per_class = BTreeMap::new();
        per_class.insert(AllocationClass::ImmutableWeights, 4096);
        per_class.insert(AllocationClass::Other, 128);
        let mut active_models = BTreeSet::new();
        active_models.insert("qwen35-cuda".to_string());
        let limit = 8 * 1024 * 1024 * 1024u64;
        let reserved = 4224;
        DeviceBudgetSnapshot {
            limit,
            reserved,
            remaining: limit - reserved,
            reservations: 1,
            fenced: false,
            per_class,
            active_models,
        }
    }

    #[test]
    fn exact_backend_and_request_credit_full_snapshot_equations() {
        let base = baseline();
        let backend = after_backend(&base, 384 * 1024 * 1024).unwrap();
        assert_eq!(backend.reserved, base.reserved + 384 * 1024 * 1024);
        assert_eq!(backend.reservations, 2);
        assert_eq!(
            backend.per_class.get(&AllocationClass::Scratch),
            Some(&(384 * 1024 * 1024))
        );
        assert!(backend.active_models.contains("backend/cuda"));
        let with = with_request_credit(&backend, "qwen35-cuda", 1000, 200).unwrap();
        assert_eq!(with.reserved, backend.reserved + 1200);
        assert_eq!(with.reservations, 3);
        assert_eq!(
            with.per_class.get(&AllocationClass::Scratch),
            Some(&(384 * 1024 * 1024 + 1000))
        );
        assert_eq!(with.per_class.get(&AllocationClass::Other), Some(&328));
        assert!(same_host(&with, &with.clone()));
    }

    #[test]
    fn full_snapshot_negatives_reject_wrong_class_count_model_fence_and_backend_size() {
        let base = baseline();
        let backend = after_backend(&base, 384 * 1024 * 1024).unwrap();
        let with = with_request_credit(&backend, "qwen35-cuda", 1000, 200).unwrap();
        let mut wrong_class = with.clone();
        wrong_class.per_class.insert(AllocationClass::KvCache, 1);
        assert!(!same_host(&with, &wrong_class));
        let mut wrong_count = with.clone();
        wrong_count.reservations += 1;
        assert!(!same_host(&with, &wrong_count));
        let mut wrong_model = with.clone();
        wrong_model.active_models.insert("other".into());
        assert!(!same_host(&with, &wrong_model));
        let mut fenced = with.clone();
        fenced.fenced = true;
        assert!(!same_host(&with, &fenced));
        assert!(after_backend(&base, 384 * 1024 * 1024 - 1).is_err());
        assert!(with_request_credit(&base, "qwen35-cuda", 0, 1).is_err());
        assert!(with_request_credit(&base, "qwen35-cuda", 1, 0).is_err());
    }
}
