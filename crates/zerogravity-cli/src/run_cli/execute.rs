use super::*;

pub(super) fn run(request: Request) -> Result<()> {
    request.validate_policy()?;
    if !matches!(request.speculation, Speculation::Auto | Speculation::Off) {
        return Err(Error::Unsupported(
            "required speculative execution has no qualified local adapter; use auto/off".into(),
        ));
    }
    if request.numerical.as_deref().is_some_and(|p| p != "source") {
        return Err(Error::Unsupported(
            "requested numerical profile has no qualified direct adapter; source is available"
                .into(),
        ));
    }
    let compiled = match request.backend {
        Backend::Auto => cfg!(any(
            feature = "cpu",
            feature = "cuda",
            feature = "hip",
            feature = "vulkan"
        )),
        Backend::Cpu => cfg!(all(
            feature = "cpu",
            feature = "model-qwen35",
            feature = "text"
        )),
        Backend::Cuda => cfg!(feature = "cuda"),
        Backend::Rocm => cfg!(feature = "hip"),
        Backend::Vulkan => cfg!(feature = "vulkan"),
    };
    if !compiled {
        return Err(Error::Unsupported(format!(
            "{} backend requires compiled features for its backend/model/text capabilities; its CPU adapter requires cpu, model-qwen35 and text",
            request.backend.label()
        )));
    }
    let inventory = devices::discover()?;
    let candidates = devices::allowed(&request, &inventory)?;
    let Entrypoint::Model(id) = &request.entrypoint else {
        request.graph.validate_entrypoint(&request.entrypoint)?;
        return Err(Error::Unsupported(
            "graph run adapter is not qualified; graph validation succeeded, routes remain inert"
                .into(),
        ));
    };
    let i = request
        .graph
        .models
        .iter()
        .position(|m| &m.id == id)
        .ok_or_else(|| bad("direct entrypoint model absent"))?;
    if qwen35_cuda::selected(&request.graph.models[i]) {
        return qwen35_cuda::run(request, inventory, candidates);
    }
    gguf(request, inventory, candidates, i)
}
#[cfg(not(all(feature = "cpu", feature = "model-qwen35", feature = "text")))]
fn gguf(_: Request, _: devices::Inventory, _: Vec<devices::Observation>, _: usize) -> Result<()> {
    Err(Error::Unsupported(
        "GGUF run requires compiled features cpu, model-qwen35 and text".into(),
    ))
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn gguf(
    mut request: Request,
    inventory: devices::Inventory,
    candidates: Vec<devices::Observation>,
    i: usize,
) -> Result<()> {
    if !candidates.iter().any(|d| d.class == "host-memory") {
        return Err(Error::Unsupported("generic GGUF GPU run lacks source-issued complete cold/compiler costs and runtime binding; exact finite HF source uses --native-artifacts".into()));
    }
    let selected = source::resolve(&request.graph.models[i], request.task.context)?;
    if request.graph.models[i].target.is_empty() {
        request.graph.models[i].target = selected.model_target.clone();
    } else if request.graph.models[i].target != selected.model_target {
        return Err(bad(
            "declared model target differs from actual source geometry",
        ));
    }
    request.graph.validate_entrypoint(&request.entrypoint)?;
    if request.task.context as u64 > selected.geometry.max_position_embeddings {
        return Err(Error::Limit(
            "requested context exceeds actual model geometry".into(),
        ));
    }

    let inventory = if devices::fresh(&inventory).is_err() {
        devices::discover()?
    } else {
        inventory
    };
    devices::fresh(&inventory)?;
    let candidates = devices::allowed(&request, &inventory)?;
    let mut decision = None;
    let mut exclusions = Vec::new();
    for d in &candidates {
        if d.class != "host-memory" {
            exclusions.push(format!(
                "{}: {}",
                d.id,
                d.exclusion
                    .as_deref()
                    .unwrap_or("unqualified source/runtime binding")
            ));
            continue;
        }
        if !cfg!(all(
            feature = "cpu",
            feature = "model-qwen35",
            feature = "text"
        )) {
            exclusions.push("cpu:0: CPU/text adapter not compiled".into());
            continue;
        }
        let Some((ceiling, available)) = d.ceiling.zip(d.available) else {
            exclusions.push("cpu:0: current availability unknown".into());
            continue;
        };
        let margin = request.headroom.bytes(ceiling, true)?;
        if available
            .checked_sub(margin)
            .is_some_and(|n| n >= selected.planning_bytes)
        {
            decision = Some((d.id.clone(), ceiling, available - margin));
            break;
        }
        exclusions
            .push("cpu:0: insufficient currently observed memory after routing headroom".into());
    }
    let explanation = serde_json::json!({"schema":"zerogravity.run.decision.v1","source":selected.source_label,"target":selected.model_target,"backend":request.backend.label(),"selected":decision.as_ref().map(|d|&d.0),"context":request.task.context,"resolved_device_policy":format!("{:?}",request.devices),"resolved_headroom":format!("{:?}",request.headroom),"numerical":request.numerical,"max_tokens":request.task.max_tokens,"raw":request.task.raw,"prompt_bytes":request.task.prompt.as_ref().map(String::len),"prefill_chunk":request.task.prefill_chunk,"planning_bytes":selected.planning_bytes,"planning_scope":"conservative CPU source/state/temporary planning; original source and logical-state admissions remain independent; no OS hard RAM guarantee","availability_observed_unix_ms":inventory.observed_unix_ms,"excluded":exclusions,"speculation":"off (no qualified measured speculative route)","source_verified":false,"model_execution_qualified_by_metadata":false,"provenance":request.provenance.iter().map(|(k,v)|(k,format!("{v:?}"))).collect::<BTreeMap<_,_>>()});
    if request.diagnostics.dry_run {
        // Ordering invariant: no register/admit/stdin read/audit-file open
        // may precede this return; NVML init and cuDevice enumeration stay
        // permitted (bounded read-only observation).
        let mut trace = explanation.clone();
        trace["dry_run"] = serde_json::json!(true);
        trace["reservation"] = serde_json::json!("none");
        println!(
            "{}",
            serde_json::to_string_pretty(&trace).map_err(|e| Error::State(e.to_string()))?
        );
        return if decision.is_some() {
            Ok(())
        } else {
            Err(Error::Unsupported(
                "no eligible local device; see dry-run exclusions".into(),
            ))
        };
    }
    if request.diagnostics.explain {
        eprintln!(
            "{}",
            serde_json::to_string(&explanation).map_err(|e| Error::State(e.to_string()))?
        );
    }
    let (_, ceiling, available) = decision.ok_or_else(|| {
        Error::Unsupported(format!(
            "no eligible local device: {}",
            exclusions.join("; ")
        ))
    })?;
    cpu(request, selected, ceiling, available)
}
#[cfg(all(feature = "cpu", feature = "model-qwen35", feature = "text"))]
fn cpu(
    request: Request,
    selected: source::PreparedSelection,
    fixed_ceiling: u64,
    available_after_margin: u64,
) -> Result<()> {
    use zerogravity::resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    };
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", "system-ram")?;
    let refreshed = devices::discover()?;
    devices::fresh(&refreshed)?;
    let host = refreshed
        .devices
        .iter()
        .find(|d| d.id == "cpu:0")
        .ok_or_else(|| Error::State("fresh host observation unavailable".into()))?;
    // Bootstrap only from the intersected fixed source-free capacity view;
    // current free bytes/margin never become an operator ceiling.
    if arbiter.snapshot(&key).is_err() {
        arbiter.register(
            key.clone(),
            fixed_ceiling.min(
                host.ceiling
                    .ok_or_else(|| Error::State("fresh HOST ceiling unavailable".into()))?,
            ),
        )?;
    }
    let fresh_available = host
        .available
        .zip(host.ceiling)
        .and_then(|(a, c)| a.checked_sub(request.headroom.bytes(c, true).ok()?))
        .ok_or_else(|| Error::Limit("fresh host availability after headroom unavailable".into()))?;
    let budget = arbiter.snapshot(&key)?;
    let available_after_margin = available_after_margin
        .min(fresh_available)
        .saturating_sub(budget.reserved);
    if budget.fenced || selected.planning_bytes > available_after_margin.min(budget.remaining) {
        return Err(Error::Limit(
            "existing HOST budget/current availability refuses CPU planning".into(),
        ));
    }
    let owner = ResourceOwner::new("cli-run", "direct-cpu")?;
    let envelope = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::ProviderEnvelope,
            bytes: selected
                .planning_bytes
                .checked_sub(selected.raw_ticket_peak)
                .ok_or_else(|| Error::State("raw tickets exceed planning envelope".into()))?,
        }],
    )?;
    let prepared = match selected.resolved {
        source::Resolved::Local {
            weights,
            config,
            mut tokenizer,
            tokenizer_bytes,
        } => {
            use std::io::Read;
            let mut bytes = Vec::new();
            (&mut tokenizer)
                .take(tokenizer_bytes + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() as u64 != tokenizer_bytes {
                return Err(Error::State(
                    "held tokenizer changed before preparation".into(),
                ));
            }
            zerogravity::prepare::PreparedQwen::from_inspected_gguf(weights, &config, &bytes)?
        }
        #[cfg(all(feature = "cache", target_os = "linux"))]
        source::Resolved::Cached {
            root,
            disk,
            snapshot,
            weights,
            config,
            tokenizer,
            layout,
        } => {
            let store = zerogravity::cache::CacheStore::open_existing(
                zerogravity::cache::CacheOptions::new(Some(root), disk),
            )?;
            let lease = store.open_snapshot(&snapshot)?;
            let admission = zerogravity::cache::TransferAdmission {
                arbiter: arbiter.clone(),
                resource: key.clone(),
                owner,
            };
            zerogravity::prepare::PreparedQwen::from_snapshots_with_layout(
                &lease,
                &weights,
                &lease,
                &config,
                &lease,
                &tokenizer,
                &admission,
                Some(&layout),
            )?
        }
    };
    let options = crate::InferenceOptions {
        prompt: request.task.prompt,
        max_tokens: request.task.max_tokens,
        raw: request.task.raw,
        context: request.task.context,
        config: None,
        tokenizer: None,
        audit_jsonl: request
            .diagnostics
            .audit_logits_jsonl
            .map(|p| p.to_string_lossy().into_owned()),
        audit_layers_jsonl: request
            .diagnostics
            .audit_layers_jsonl
            .map(|p| p.to_string_lossy().into_owned()),
        teacher_ids: request.task.teacher_ids,
        prefill_chunk: request.task.prefill_chunk,
    };
    let result = crate::infer_prepared(&selected.source_label, prepared, options);
    drop(envelope);
    result
}
