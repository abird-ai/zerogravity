//! Source-free host-policy controls plus separately ignored physical cases.

use super::*;
use crate::run_policy::*;
use std::{collections::BTreeMap, path::PathBuf};
fn bad(message: impl Into<String>) -> Error {
    Error::Malformed(message.into())
}
use crate::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

const MIB: u64 = 1024 * 1024;
const HOST_REFUSAL: &str =
    "finite HF original HOST8192MiB plus current routing headroom does not fit";
const NO_CARD_REFUSAL: &str = "no allowed source-qualified sm86 CUDA card fits original provider+512MiB backend and current free-memory headroom; no split/backend/source fallback";

fn route_request(source: &str) -> Request {
    Request {
        graph: crate::pipeline::Config {
            models: vec![crate::pipeline::ModelInstance {
                id: "main".into(),
                target: String::new(),
                source: Some(source.into()),
                model_config: None,
                tokenizer: None,
                snapshot: None,
                native_artifacts: None,
            }],
            pipelines: Vec::new(),
            routes: Vec::new(),
        },
        entrypoint: Entrypoint::Model("main".into()),
        task: Task::default(),
        backend: Backend::Auto,
        devices: Devices::Auto,
        headroom: Headroom::Default,
        numerical: None,
        speculation: Speculation::Auto,
        draft_device: None,
        diagnostics: Diagnostics::default(),
        vendor_filter: None,
        provenance: BTreeMap::from([
            ("task.context".into(), Origin::Builtin),
            ("task.max_tokens".into(), Origin::Builtin),
        ]),
    }
}

fn gpu(
    id: &str,
    vendor: &str,
    cc: Option<(i32, i32)>,
    ordinal: Option<usize>,
    ceiling: Option<u64>,
    available: Option<u64>,
    exclusion: Option<&str>,
) -> devices::Observation {
    devices::Observation {
        id: id.into(),
        class: "gpu".into(),
        vendor: vendor.into(),
        ceiling,
        available,
        backends: vec![],
        exclusion: exclusion.map(str::to_string),
        provenance: "pure metadata fixture".into(),
        cuda_ordinal: ordinal,
        cuda_cc: cc,
    }
}

fn host(ceiling: Option<u64>, available: Option<u64>) -> devices::Observation {
    devices::Observation {
        id: "cpu:0".into(),
        class: "host-memory".into(),
        vendor: "cpu".into(),
        ceiling,
        available,
        backends: vec![],
        exclusion: None,
        provenance: "pure metadata fixture".into(),
        cuda_ordinal: None,
        cuda_cc: None,
    }
}

fn inventory(devices: Vec<devices::Observation>) -> devices::Inventory {
    devices::Inventory {
        devices,
        observed_unix_ms: 0,
    }
}

fn profile(host_bytes: usize, provider_bytes: usize) -> qualification::FiniteRouteProfile {
    qualification::FiniteRouteProfile {
        host_bytes,
        provider_bytes,
    }
}

fn refusal_message(error: Error) -> String {
    match error {
        Error::Limit(m) => m,
        Error::Resource(limit) => limit.to_string(),
        other => panic!("expected limit refusal: {other:?}"),
    }
}

#[test]
fn native_artifacts_marker_keeps_qualified_auto_route() {
    let mut request = route_request("/source-not-opened/model");
    request.graph.models[0].native_artifacts = Some("/native-not-opened".into());
    assert_eq!(request.backend, Backend::Auto);
    assert!(request.graph.models[0].target.is_empty());
    assert!(selected(&request.backend, &request.graph.models[0]));
}

#[test]
fn sm86_proposal_keeps_ordered_exclusions_for_other_devices() {
    let request = route_request("m.gguf");
    let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
    let amd = gpu(
        "pci:0000:01:00.0",
        "amd",
        None,
        None,
        Some(6 * 1024 * MIB),
        Some(6 * 1024 * MIB),
        Some(
            "source-qualified cold/compiler cost and runtime physical binding unavailable in generic run",
        ),
    );
    let fit = gpu(
        "pci:0000:02:00.0",
        "nvidia",
        Some((8, 6)),
        Some(0),
        Some(6 * 1024 * MIB),
        Some(6 * 1024 * MIB),
        None,
    );
    let inv = inventory(vec![
        amd,
        fit.clone(),
        host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
    ]);
    let (proposed, exclusions) =
        qualification::select_sm86_card(&request, &profile, &inv, &ResourceArbiter::new()).unwrap();
    assert_eq!(proposed.expect("fit card").id, fit.id);
    assert_eq!(
        exclusions,
        vec![
            "pci:0000:01:00.0: source-qualified cold/compiler cost and runtime physical binding unavailable in generic run"
        ]
    );
}

#[test]
fn sm86_card_exclusion_reasons_mirror_every_skip_condition() {
    // Explicit headroom makes the free-memory exclusion deterministic.
    let mut request = route_request("m.gguf");
    request.headroom = Headroom::parse("3GiB").unwrap();
    let profile = profile(8192 * 1024 * 1024, 4 * 1024 * 1024 * 1024usize);
    let inv = inventory(vec![
        gpu(
            "pci:0000:03:00.0",
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            None,
            Some("NVML current free/capacity binding unavailable"),
        ),
        gpu(
            "pci:0000:04:00.0",
            "nvidia",
            Some((8, 6)),
            Some(1),
            None,
            None,
            Some("multiple CUDA views share one PCI identity (MIG/partition ambiguous)"),
        ),
        gpu(
            "pci:0000:05:00.0",
            "nvidia",
            Some((8, 6)),
            Some(2),
            Some(4 * 1024 * MIB),
            Some(4 * 1024 * MIB),
            None,
        ),
        gpu(
            "pci:0000:06:00.0",
            "nvidia",
            Some((8, 6)),
            Some(3),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        ),
        gpu(
            "pci:0000:07:00.0",
            "nvidia",
            Some((8, 9)),
            Some(4),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        ),
        gpu("pci:0000:08:00.0", "amd", None, None, None, None, None),
        host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
    ]);
    let (proposed, exclusions) =
        qualification::select_sm86_card(&request, &profile, &inv, &ResourceArbiter::new()).unwrap();
    assert!(proposed.is_none(), "no card passes every condition");
    assert_eq!(
        exclusions,
        vec![
            "pci:0000:03:00.0: NVML current free/capacity binding unavailable",
            "pci:0000:04:00.0: multiple CUDA views share one PCI identity (MIG/partition ambiguous)",
            "pci:0000:05:00.0: card capacity below the 5120MiB operator cap",
            "pci:0000:06:00.0: insufficient current free memory after routing headroom",
            "pci:0000:07:00.0: finite Qwen CUDA route requires qualified sm86 NVIDIA CUDA card",
            "pci:0000:08:00.0: finite Qwen CUDA route requires qualified sm86 NVIDIA CUDA card",
        ]
    );
    assert_eq!(
        refusal_message(qualification::no_route_refusal(&profile, &exclusions)),
        NO_CARD_REFUSAL
    );
}

#[test]
fn sm86_ledger_hold_fence_and_cap_conflict_exclude_the_card_exactly() {
    let request = route_request("m.gguf");
    let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
    let fit = |id: &str| {
        gpu(
            id,
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        )
    };
    let reserved = |arbiter: &ResourceArbiter,
                    key: &PhysicalDeviceKey|
     -> crate::resource_arbiter::GlobalResourceLease {
        arbiter.register_matching(key.clone(), 5120 * MIB).unwrap();
        arbiter
            .admit(
                key,
                ResourceOwner::new("other-route", "session").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 128 * MIB,
                }],
            )
            .unwrap()
    };
    for (arbiter, name) in [
        (ResourceArbiter::new(), "reserved"),
        (ResourceArbiter::new(), "fenced"),
        (ResourceArbiter::new(), "cap"),
    ] {
        let key = PhysicalDeviceKey::new("pci", "0000:09:00.0").expect("synthetic PCI identity");
        // The reserved case keeps its lease alive across the scan.
        let held = match name {
            "reserved" => Some(reserved(&arbiter, &key)),
            "fenced" => {
                arbiter.register_matching(key.clone(), 5120 * MIB).unwrap();
                arbiter.fence_unknown_device(&key).unwrap();
                None
            }
            _ => {
                arbiter.register(key.clone(), 4096 * MIB).unwrap();
                None
            }
        };
        let inv = inventory(vec![
            fit("pci:0000:09:00.0"),
            host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
        ]);
        let (proposed, exclusions) =
            qualification::select_sm86_card(&request, &profile, &inv, &arbiter).unwrap();
        assert!(proposed.is_none(), "{name}");
        assert_eq!(
            exclusions,
            vec!["pci:0000:09:00.0: card already serves an in-process route reservation"],
            "{name}"
        );
        drop(held);
    }
}

#[test]
fn sm86_host_unfit_suppresses_proposals_and_refuses_with_host_limit() {
    let request = route_request("m.gguf");
    let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
    let fit = gpu(
        "pci:0000:0a:00.0",
        "nvidia",
        Some((8, 6)),
        Some(0),
        Some(6 * 1024 * MIB),
        Some(6 * 1024 * MIB),
        None,
    );
    let amd = gpu(
        "pci:0000:0b:00.0",
        "amd",
        None,
        None,
        Some(6 * 1024 * MIB),
        Some(6 * 1024 * MIB),
        None,
    );
    let unfit_host = host(Some(64 * 1024 * MIB), Some(1));
    let inv = inventory(vec![amd, fit, unfit_host]);
    let (proposed, exclusions) =
        qualification::select_sm86_card(&request, &profile, &inv, &ResourceArbiter::new()).unwrap();
    assert!(proposed.is_none(), "host unfitness suppresses proposals");
    assert_eq!(
        exclusions.first().map(String::as_str),
        Some("host: finite HF original HOST8192MiB plus current routing headroom does not fit")
    );
    assert!(exclusions.len() >= 2, "card exclusions share the trace");
    assert_eq!(
        refusal_message(qualification::no_route_refusal(&profile, &exclusions)),
        HOST_REFUSAL
    );
}

#[test]
fn sm86_observation_gaps_stay_hard_errors_and_empty_traces_refuse_exactly() {
    let request = route_request("m.gguf");
    let profile = profile(8192 * 1024 * 1024, 512 * 1024 * 1024);
    let arbiter = ResourceArbiter::new();
    let fit = gpu(
        "pci:0000:0c:00.0",
        "nvidia",
        Some((8, 6)),
        Some(0),
        Some(6 * 1024 * MIB),
        Some(6 * 1024 * MIB),
        None,
    );
    // No HOST observation is a hard state error, never a silent exclusion.
    let inv = inventory(vec![fit]);
    assert!(qualification::select_sm86_card(&request, &profile, &inv, &arbiter).is_err());
    // An unknown HOST ceiling is a hard state error.
    let inv = inventory(vec![
        gpu(
            "pci:0000:0c:00.0",
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        ),
        host(None, None),
    ]);
    assert!(qualification::select_sm86_card(&request, &profile, &inv, &arbiter).is_err());
    // An unavailable explicit device is a policy error, not an exclusion.
    let mut request = route_request("m.gguf");
    request.devices = Devices::parse(&["gpu:5".into()]).unwrap();
    let inv = inventory(vec![
        gpu(
            "pci:0000:0c:00.0",
            "nvidia",
            Some((8, 6)),
            Some(0),
            Some(6 * 1024 * MIB),
            Some(6 * 1024 * MIB),
            None,
        ),
        host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB)),
    ]);
    assert!(qualification::select_sm86_card(&request, &profile, &inv, &arbiter).is_err());
    // A fitting host with no GPUs refuses with the exact no-card limit.
    let request = route_request("m.gguf");
    let inv = inventory(vec![host(Some(64 * 1024 * MIB), Some(64 * 1024 * MIB))]);
    let (proposed, exclusions) =
        qualification::select_sm86_card(&request, &profile, &inv, &arbiter).unwrap();
    assert!(proposed.is_none());
    assert!(exclusions.is_empty());
    assert_eq!(
        refusal_message(qualification::no_route_refusal(&profile, &exclusions)),
        NO_CARD_REFUSAL
    );
}

#[test]
fn decision_trace_fields_split_dry_run_from_executing_admission() {
    let mut dry = serde_json::json!({});
    qualification::dry_run_trace_status(&mut dry);
    qualification::dry_run_admission_trace(&mut dry);
    assert_eq!(dry["dry_run"], serde_json::json!(true));
    assert_eq!(dry["reservation"], serde_json::json!("none"));
    assert_eq!(dry["pre_client_admission_gap"], serde_json::json!(true));
    assert!(dry.get("pre_client_admission").is_none());
    assert!(dry.get("advisory_boundary").is_none());

    let mut executing = serde_json::json!({});
    qualification::executing_admission_trace(&mut executing);
    assert_eq!(
        executing["pre_client_admission"],
        serde_json::json!("transactional card hold before Client; released at verified identity")
    );
    assert_eq!(
        executing["advisory_boundary"],
        serde_json::json!(
            "other processes and driver allocations; post-release pre-backend window"
        )
    );
    assert!(executing.get("pre_client_admission_gap").is_none());
    assert!(executing.get("dry_run").is_none());
    assert!(executing.get("reservation").is_none());
}

#[test]
fn pre_client_card_admit_races_to_exactly_one_holder_then_retries() {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("pci", "0000:0d:00.0").unwrap();
    arbiter.register_matching(key.clone(), 5120 * MIB).unwrap();
    let fresh = arbiter.snapshot(&key).unwrap();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let (first, second) = std::thread::scope(|scope| {
        let a = scope.spawn({
            let arbiter = &arbiter;
            let key = &key;
            let barrier = barrier.clone();
            move || {
                barrier.wait();
                qualification::admit_pre_client_card(arbiter, key)
            }
        });
        let b = scope.spawn({
            let arbiter = &arbiter;
            let key = &key;
            move || {
                barrier.wait();
                qualification::admit_pre_client_card(arbiter, key)
            }
        });
        (a.join().unwrap(), b.join().unwrap())
    });
    let (winner, loser) = match (first, second) {
        (Ok(lease), Err(error)) => (lease, error),
        (Err(error), Ok(lease)) => (lease, error),
        (Ok(_), Ok(_)) => panic!("two pre-Client holds admitted one card"),
        (Err(_), Err(_)) => panic!("no pre-Client hold admitted the free card"),
    };
    assert!(matches!(&loser, Error::Resource(_)), "{loser}");
    let held = arbiter.snapshot(&key).unwrap();
    assert_eq!(held.reserved, 5120 * MIB);
    assert_eq!(held.reservations, 1);
    assert_eq!(
        held.per_class.get(&AllocationClass::ProviderEnvelope),
        Some(&(5120 * MIB))
    );
    assert_eq!(
        held.active_models
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["qwen35-cuda"]
    );
    // The immutable cap stays one shared budget while the hold is alive.
    assert!(arbiter.register(key.clone(), 1).is_err());
    drop(winner);
    // After the hold is released the loser retries successfully and the
    // budget returns to the exact pre-admission snapshot.
    let retried = qualification::admit_pre_client_card(&arbiter, &key).unwrap();
    drop(retried);
    let released = arbiter.snapshot(&key).unwrap();
    assert_eq!(released.reserved, fresh.reserved);
    assert_eq!(released.remaining, fresh.remaining);
    assert_eq!(released.reservations, fresh.reservations);
    assert_eq!(released.per_class, fresh.per_class);
    assert_eq!(released.active_models, fresh.active_models);
    assert_eq!(released.fenced, fresh.fenced);
}

#[test]
fn pre_client_card_admit_refuses_fenced_keys_and_conflicting_caps_exactly() {
    let key = PhysicalDeviceKey::new("pci", "0000:0e:00.0").unwrap();
    let fenced = ResourceArbiter::new();
    fenced.register_matching(key.clone(), 5120 * MIB).unwrap();
    fenced.fence_unknown_device(&key).unwrap();
    assert!(matches!(
        &qualification::admit_pre_client_card(&fenced, &key),
        Err(Error::State(m)) if m == "physical device completion uncertain: fenced"
    ));
    let conflicting = ResourceArbiter::new();
    conflicting.register(key.clone(), 4096 * MIB).unwrap();
    assert!(matches!(
        &qualification::admit_pre_client_card(&conflicting, &key),
        Err(Error::State(m)) if m == "conflicting physical device budget attestation"
    ));
    // Idempotent same-cap registration leaves exactly one budget.
    let idempotent = ResourceArbiter::new();
    idempotent
        .register_matching(key.clone(), 5120 * MIB)
        .unwrap();
    idempotent
        .register_matching(key.clone(), 5120 * MIB)
        .unwrap();
    assert_eq!(idempotent.snapshot(&key).unwrap().limit, 5120 * MIB);
    assert!(idempotent.register(key.clone(), 5120 * MIB).is_err());
    let hold = qualification::admit_pre_client_card(&idempotent, &key).unwrap();
    drop(hold);
    assert_eq!(idempotent.snapshot(&key).unwrap().reserved, 0);
}

fn absolute_env(name: &str) -> Result<PathBuf> {
    std::env::var_os(name)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| bad(format!("required absolute {name}")))
}

#[test]
#[ignore = "explicit physical Qwen CUDA dry-run; approved sm86 hardware and real source/native bundle, separate fresh owned process"]
fn physical_qwen35_cuda_dry_run_is_read_only_and_registers_nothing() -> Result<()> {
    let source = absolute_env("ZG_QWEN35_MODEL_DIR")?;
    let native = absolute_env("ZG_QWEN35_CUDA_KERNEL_DIR")?;
    let source = source.to_str().ok_or_else(|| bad("UTF-8 source path"))?;
    let native = native.to_str().ok_or_else(|| bad("UTF-8 native path"))?;
    let mut request = route_request(source);
    request.graph.models[0].native_artifacts = Some(native.into());
    request.diagnostics.dry_run = true;
    request.task.prompt = Some("a".into());
    let arbiter = ResourceArbiter::process_shared();
    let keys = devices::discover()?
        .devices
        .iter()
        .filter(|d| d.id.starts_with("pci:"))
        .filter_map(|d| {
            d.id.strip_prefix("pci:")
                .and_then(|pci| PhysicalDeviceKey::new("pci", pci).ok())
        })
        .collect::<Vec<_>>();
    assert!(!keys.is_empty(), "physical CUDA card observation");
    // A second dry-run in the same process observes the same ledger: a
    // dry-run may not register a card budget or admit anything.
    let before = keys.iter().map(|k| arbiter.snapshot(k)).collect::<Vec<_>>();
    for request in [request.clone(), request] {
        let inventory = devices::discover()?;
        run_route(
            request,
            inventory,
            || panic!("dry-run read stdin"),
            |_| Ok(()),
        )?;
    }
    let after = keys.iter().map(|k| arbiter.snapshot(k)).collect::<Vec<_>>();
    for (before, after) in before.iter().zip(&after) {
        match (before, after) {
            (Err(_), Err(_)) => {}
            (Ok(before), Ok(after)) => {
                assert_eq!(after.reserved, before.reserved);
                assert_eq!(after.reservations, before.reservations);
                assert_eq!(after.per_class, before.per_class);
                assert_eq!(after.active_models, before.active_models);
                assert_eq!(after.fenced, before.fenced);
            }
            _ => panic!("dry-run changed the physical-card ledger registration"),
        }
    }
    Ok(())
}

#[test]
#[ignore = "explicit physical pre-Client seam; approved sm86 hardware, separate fresh owned process"]
fn physical_pre_client_card_seam_holds_the_whole_card_budget() -> Result<()> {
    let inventory = devices::discover()?;
    devices::fresh(&inventory)?;
    let card = inventory
        .devices
        .iter()
        .find(|d| {
            d.class == "gpu"
                && d.vendor == "nvidia"
                && d.cuda_cc == Some((8, 6))
                && d.cuda_ordinal.is_some()
                && d.exclusion.is_none()
        })
        .ok_or_else(|| bad("no allowed source-qualified sm86 CUDA card"))?;
    let key = PhysicalDeviceKey::new(
        "pci",
        card.id
            .strip_prefix("pci:")
            .ok_or_else(|| bad("PCI identity"))?,
    )?;
    let arbiter = ResourceArbiter::process_shared();
    assert!(arbiter.snapshot(&key).is_err());
    let hold = qualification::admit_pre_client_card(&arbiter, &key)?;
    // A second route refuses with the arbiter's exact limit while the
    // first route holds the whole finite card budget.
    assert!(matches!(
        &qualification::admit_pre_client_card(&arbiter, &key),
        Err(Error::Resource(_))
    ));
    let held = arbiter.snapshot(&key)?;
    assert_eq!(held.reserved, 5120 * MIB);
    assert_eq!(held.reservations, 1);
    let ordinal = card.cuda_ordinal.ok_or_else(|| bad("CUDA ordinal"))?;
    let device =
        crate::gpu::Device::cuda(ordinal).map_err(|e| Error::Unsupported(e.to_string()))?;
    let actual = crate::gpu::initialize_cuda_physical_identity(&device)?;
    assert_eq!(actual, key);
    drop(hold);
    let released = arbiter.snapshot(&key)?;
    assert_eq!(released.reserved, 0);
    assert_eq!(released.reservations, 0);
    assert!(released.per_class.is_empty());
    assert!(released.active_models.is_empty());
    Ok(())
}

#[test]
fn unselected_sdk_route_refuses_before_inventory_source_or_callbacks() {
    let request = route_request("/must-not-open");
    let error = run_route(
        request,
        inventory(Vec::new()),
        || panic!("unselected stdin"),
        |_| panic!("unselected presentation"),
    )
    .unwrap_err();
    assert!(matches!(error, Error::Unsupported(ref message) if message.contains("not selected")));
}
