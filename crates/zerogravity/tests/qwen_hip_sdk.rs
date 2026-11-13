//! Public runtime-only source route. Each ignored case needs a fresh gfx1151 process.
#![cfg(all(feature = "qwen35-hip", target_os = "linux"))]

#[path = "support/qwen_hip_context.rs"]
mod contexts;
#[cfg(feature = "serve")]
#[path = "support/qwen_hip_http.rs"]
mod http;
#[path = "support/qwen_hip_population.rs"]
mod population;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    os::unix::fs::{MetadataExt, OpenOptionsExt},
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use zerogravity::{
    Runtime,
    cache::CacheOptions,
    models::{ModelLoadRequest, QwenHipProfile},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
    run::{TaskInput, TaskRequest},
    runtime::{
        SchedulerLimits, TextTokenizer,
        text::{GpuTokenObservation, Request, RequestEvents, ResponseEvent, TokenMetricsRequest},
    },
};

#[derive(Deserialize)]
struct Call {
    repeat: usize,
    case: usize,
    call: usize,
    rows: usize,
    input_ids: Vec<u32>,
    selected: u32,
}
#[derive(Deserialize)]
struct Reference {
    repository: String,
    status: String,
    numerical_profile: String,
    original_full_forward: bool,
    rust_or_gpu_parity: bool,
    active_context: usize,
    calls: Vec<Call>,
}
fn source_snapshot(label: &str) -> &'static str {
    match label {
        "08" => "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c",
        "2b" => "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        "27b" => "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
        _ => panic!("explicit approved model required"),
    }
}
fn stamp(file: &File) -> (u64, u64, u64, i64, i64, i64, i64) {
    let m = file.metadata().unwrap();
    assert!(m.is_file());
    (
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime(),
        m.mtime_nsec(),
        m.ctime(),
        m.ctime_nsec(),
    )
}
fn reference(path: &Path, label: &str) -> Reference {
    let (expected, repository) = match label {
        "08" => (
            "492fd2b361f79cec3b2443c5454850dc91aef6d881e3c41330e2d50f3c544d3d".to_owned(),
            "Qwen/Qwen3.5-0.8B",
        ),
        "2b" => (
            "1a299081a66d386c9f86e8b6e71e44fd653ae6424a1709a93184d3cda48e8c96".to_owned(),
            "Qwen/Qwen3.5-2B",
        ),
        "27b" => (
            std::env::var("ZG_QWEN_SOURCE_CALLS_MANIFEST_SHA256")
                .expect("separately audited original27B manifest digest"),
            "Qwen/Qwen3.8-27B",
        ),
        _ => panic!("explicit approved model required"),
    };
    assert!(expected.len() == 64 && expected.bytes().all(|b| b.is_ascii_hexdigit()));
    let bytes = manifest_bytes(
        path,
        &expected,
        if label == "27b" { 16 } else { 8 } * 1024 * 1024,
    );
    let result: Reference = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(result.repository, repository);
    assert_eq!(
        result.status,
        "CAPTURED_ORIGINAL_SOURCE_CALL_REFERENCE_ONLY"
    );
    assert_eq!(result.numerical_profile, "TorchCpuBf16Avx2MixedSourceV1");
    assert!(result.original_full_forward && !result.rust_or_gpu_parity);
    assert_eq!(result.active_context, 514);
    assert_eq!(result.calls.len(), 66);
    result
}
fn manifest_bytes(path: &Path, expected: &str, limit: u64) -> Vec<u8> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0o400000 | 0o2000000)
        .open(path)
        .unwrap();
    let before = stamp(&file);
    let mut bytes = Vec::new();
    (&file).take(limit + 1).read_to_end(&mut bytes).unwrap();
    assert!(bytes.len() as u64 <= limit);
    assert_eq!(stamp(&file), before);
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
    bytes
}
fn request(alias: &str, prompt: &str) -> Request {
    Request {
        model: alias.into(),
        input: prompt.into(),
        chat: false,
        max_tokens: 3,
        stream: true,
        token_metrics: Some(TokenMetricsRequest {
            started: Instant::now(),
            wire_request_id: None,
        }),
    }
}
fn next(events: &RequestEvents) -> ResponseEvent {
    events
        .events
        .recv_timeout(Duration::from_secs(1200))
        .expect("worker event deadline")
}
struct Observed {
    ids: Vec<u32>,
    roots: Vec<GpuTokenObservation>,
    output: String,
}
fn complete(events: RequestEvents, tokens: usize, tokenizer: &dyn TextTokenizer) -> Observed {
    let mut result = Observed {
        ids: Vec::new(),
        roots: Vec::new(),
        output: String::new(),
    };
    loop {
        match next(&events) {
            ResponseEvent::TokenText { text, token } => {
                assert_eq!(token.index, result.ids.len());
                let root = token.gpu.expect("actual source publication observation");
                assert_eq!(root.producer_frontier, (tokens + token.index) as u64);
                assert!(root.producer_root.is_some() && root.native_cohort.is_none());
                result.ids.push(token.id);
                result.roots.push(root);
                result.output.push_str(&text);
            }
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            } => {
                assert_eq!(prompt_tokens, tokens);
                assert_eq!(completion_tokens, result.ids.len());
                assert!(!result.ids.is_empty());
                assert_eq!(output, result.output);
                assert_eq!(output, tokenizer.decode(&result.ids, true).unwrap());
                assert!(matches!(finish_reason, "length" | "stop"));
                break;
            }
            ResponseEvent::Error(error) => panic!("source worker: {error}"),
            ResponseEvent::Text(_) => panic!("requested original token observations"),
        }
    }
    let first = result.roots[0].producer_root.unwrap();
    for pair in result.roots.windows(2) {
        let a = pair[0].producer_root.unwrap();
        let b = pair[1].producer_root.unwrap();
        assert_eq!(b.session_id, first.session_id);
        assert_eq!(b.session_epoch, first.session_epoch);
        assert_ne!(a.root, b.root);
    }
    result
}

#[test]
#[ignore = "approved canonical source/reference, gfx1151, isolated serial release process"]
fn original_source_public_sdk_completion_cancellation_and_retirement() {
    if cfg!(debug_assertions) {
        panic!("release physical target required");
    }
    assert_eq!(std::env::var("HIP_VISIBLE_DEVICES").as_deref(), Ok("0"));
    assert_eq!(std::env::var("ROCR_VISIBLE_DEVICES").as_deref(), Ok("0"));
    for (field, expected) in [("vendor", "0x1002"), ("device", "0x1586")] {
        assert_eq!(
            std::fs::read_to_string(format!("/sys/bus/pci/devices/0000:c6:00.0/{field}"))
                .unwrap()
                .trim(),
            expected
        );
    }
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").expect("explicit model");
    let mode = std::env::var("ZG_QWEN_HIP_SDK_MODE").expect("explicit mode");
    assert!(matches!(mode.as_str(), "complete" | "disconnect"));
    let snapshot = source_snapshot(&label);
    let arbiter = ResourceArbiter::process_shared();
    let host = PhysicalDeviceKey::new("host", "system-ram").unwrap();
    let device = PhysicalDeviceKey::new("pci", "0000:c6:00.0").unwrap();
    arbiter
        .register_matching(host.clone(), 16 * 1024 * 1024 * 1024)
        .unwrap();
    let support = arbiter
        .admit(
            &host,
            ResourceOwner::new("sdk-test", "reference").unwrap(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 64 * 1024 * 1024,
            }],
        )
        .unwrap();
    let manifest =
        std::env::var_os("ZG_QWEN_HIP_SDK_REFERENCE").expect("retained original manifest");
    let reference = reference(Path::new(&manifest), &label);
    let bootstrap = zerogravity::tokenize_admission::approve_arbiter(
        &zerogravity::resource_arbiter::TransferAdmission {
            arbiter: arbiter.clone(),
            resource: host.clone(),
            owner: ResourceOwner::new("sdk-test", "bootstrap").unwrap(),
        },
        "bootstrap",
    )
    .unwrap();
    // Original thread-root allowance and tokenizer globals are process-lifetime.
    let serving = zerogravity::runtime::text::pending::ServingHost::from_process_admission(
        &zerogravity::resource_arbiter::TransferAdmission {
            arbiter: arbiter.clone(),
            resource: host.clone(),
            owner: ResourceOwner::new("sdk-test", "host").unwrap(),
        },
    )
    .unwrap();
    let host_baseline = arbiter.snapshot(&host).unwrap().reserved;
    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 2,
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime
        .models()
        .load(ModelLoadRequest::QwenHip {
            aliases: vec!["source".into(), "alias".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: snapshot.into(),
            profile: QwenHipProfile {
                active: 2,
                context: 514,
                output: 3,
                max_pending: Some(4),
                native_rows: 0,
            },
        })
        .unwrap();
    let model = runtime.models().resolve("source").unwrap();
    let alias = runtime.models().resolve("alias").unwrap();
    assert!(model.shares_source(&alias));
    assert_eq!(runtime.status().execution_adapters, 1);
    let weak = Arc::downgrade(model.prepared());
    let tokenizer = model.prepared().tokenizer();
    let calls: Vec<_> = reference
        .calls
        .iter()
        .filter(|c| c.repeat == 0 && c.case == 0)
        .collect();
    assert_eq!(calls.len(), 3);
    for (i, call) in calls.iter().enumerate() {
        assert_eq!(call.call, i);
    }
    let prompt = tokenizer.decode(&calls[0].input_ids, false).unwrap();
    assert_eq!(
        tokenizer.prompt_tokens(&prompt, false).unwrap(),
        calls[0].input_ids
    );
    assert_eq!(calls[0].rows, 47);
    if mode == "disconnect" {
        let events = runtime.request_text(request("source", &prompt)).unwrap();
        match next(&events) {
            ResponseEvent::TokenText { token, .. } => {
                assert_eq!(token.id, calls[0].selected);
                assert_eq!(token.gpu.unwrap().producer_frontier, 47);
            }
            ResponseEvent::Error(e) => panic!("source before disconnect: {e}"),
            _ => panic!("first actual publication absent"),
        }
        events.cancel();
        drop(events); // Consumer failure: queued/in-flight work must still retire safely.
    }
    let first = complete(
        runtime.request_text(request("source", &prompt)).unwrap(),
        47,
        tokenizer.as_ref(),
    );
    assert_eq!(
        first.ids,
        calls.iter().map(|c| c.selected).collect::<Vec<_>>()
    );
    let second = complete(
        runtime.request_text(request("alias", &prompt)).unwrap(),
        47,
        tokenizer.as_ref(),
    );
    assert_eq!(first.ids, second.ids);
    assert_ne!(
        first.roots[0].producer_root.unwrap().session_id,
        second.roots[0].producer_root.unwrap().session_id
    );
    // Public generic run delegates to the same registry/worker, not a diagnostic route.
    let mut stream = runtime
        .run(&TaskRequest {
            model: "alias".into(),
            input: TaskInput::Text(prompt),
            max_output: 3.try_into().unwrap(),
        })
        .unwrap();
    let mut done = 0;
    while let Some(event) = stream.next().unwrap() {
        match event {
            ResponseEvent::Done {
                output,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                assert_eq!(
                    (output, prompt_tokens, completion_tokens),
                    (first.output.clone(), 47, 3)
                );
                done += 1;
            }
            ResponseEvent::Error(e) => panic!("public run: {e}"),
            _ => {}
        }
    }
    assert_eq!(done, 1);
    drop(stream);
    // Initialized one-row prompt tail after a real64-row, state-only prefix.
    let tail = " a".repeat(65);
    assert_eq!(tokenizer.prompt_tokens(&tail, false).unwrap().len(), 65);
    let tail_a = complete(
        runtime.request_text(request("source", &tail)).unwrap(),
        65,
        tokenizer.as_ref(),
    );
    let tail_b = complete(
        runtime.request_text(request("alias", &tail)).unwrap(),
        65,
        tokenizer.as_ref(),
    );
    assert_eq!(tail_a.ids, tail_b.ids);
    assert_eq!(tail_a.output, tail_b.output);
    runtime.shutdown().unwrap();
    assert!(runtime.status().closed);
    assert_eq!(runtime.status().execution_adapters, 0);
    let physical = arbiter.snapshot(&device).unwrap();
    assert_eq!(physical.reserved, 4 * 1024 * 1024 * 1024);
    assert!(!physical.fenced);
    drop((model, alias, runtime));
    assert!(weak.upgrade().is_none());
    let held = arbiter.snapshot(&host).unwrap();
    assert!(!held.fenced);
    assert!(held.reserved >= host_baseline + 2 * 1024 * 1024 * 1024 + 64 * 1024 * 1024);
    assert_eq!(tokenizer.prompt_tokens(&tail, false).unwrap().len(), 65);
    drop(tokenizer);
    let final_host = arbiter.snapshot(&host).unwrap();
    assert_eq!(final_host.reserved, host_baseline + 2 * 1024 * 1024 * 1024);
    assert!(!final_host.fenced);
    drop((reference, support, bootstrap, serving));
    println!(
        "SOURCE_PUBLIC_SDK_PASS model={label} mode={mode} original_ids=6 prompt_tail65=replayed independent_sessions=true native_cohorts=false device_context_bytes={} host_with_support_bytes={} full_float_comparisons=0",
        physical.reserved, final_host.reserved
    );
}
