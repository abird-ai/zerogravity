//! Real populated prompts through the ordinary SDK's state-only chunks and decode.
use super::{population::Harness, *};

#[derive(Deserialize)]
struct ContextCall {
    repeat: usize,
    case: usize,
    call: usize,
    phase: String,
    position: usize,
    rows: usize,
    input_ids: Vec<u32>,
    selected: u32,
    checkpoint: bool,
}
#[derive(Deserialize)]
pub(super) struct ContextReference {
    schema: String,
    status: String,
    repository: String,
    active_context: usize,
    original_full_forward: bool,
    all_original_source_and_mapped_bytes_unchanged: bool,
    rust_or_gpu_parity: bool,
    numerical_profile: String,
    prompts: Vec<String>,
    calls: Vec<ContextCall>,
}
impl ContextReference {
    pub(super) fn open(label: &str) -> Self {
        let path = std::env::var_os("ZG_QWEN_CONTEXT_REFERENCE").unwrap();
        // Supplied only from the separately verified original capture, never GPU output.
        let digest = std::env::var("ZG_QWEN_CONTEXT_MANIFEST_SHA256").unwrap();
        assert_eq!(digest.len(), 64);
        let bytes = manifest_bytes(Path::new(&path), &digest, 16 * 1024 * 1024);
        let r: Self = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(r.schema, "zg.original-hf-source-context-reference.v1");
        assert_eq!(r.status, "CAPTURED_ORIGINAL_SOURCE_CALL_REFERENCE_ONLY");
        assert_eq!(
            r.repository,
            match label {
                "08" => "Qwen/Qwen3.5-0.8B",
                "2b" => "Qwen/Qwen3.5-2B",
                _ => panic!("source model"),
            }
        );
        assert_eq!(r.numerical_profile, "TorchCpuBf16Avx2MixedSourceV1");
        assert!(
            r.original_full_forward
                && r.all_original_source_and_mapped_bytes_unchanged
                && !r.rust_or_gpu_parity
        );
        assert_eq!(r.active_context, 16384);
        assert_eq!(r.prompts.len(), 5);
        assert_eq!(r.calls.len(), 1012);
        r
    }
    pub(super) fn trajectory(
        &self,
        case: usize,
        capacity: usize,
        tokenizer: &dyn TextTokenizer,
    ) -> (String, Vec<u32>) {
        let prompt = &self.prompts[case];
        let mut outputs = Vec::new();
        for repeat in 0..2 {
            let calls: Vec<_> = self
                .calls
                .iter()
                .filter(|c| c.case == case && c.repeat == repeat)
                .collect();
            let prefill = (capacity - 2) / 64 + 1;
            assert_eq!(calls.len(), prefill + 2);
            let (mut position, mut ids, mut selected) = (0, Vec::new(), Vec::new());
            for (index, call) in calls.iter().enumerate() {
                assert_eq!((call.call, call.position), (index, position));
                assert_eq!(call.rows, call.input_ids.len());
                let decode = index >= prefill;
                assert_eq!(call.phase, if decode { "decode" } else { "prefill" });
                assert_eq!(
                    call.rows,
                    if decode {
                        1
                    } else if index == prefill - 1 {
                        62
                    } else {
                        64
                    }
                );
                if decode {
                    assert_eq!(call.input_ids, [calls[index - 1].selected]);
                } else {
                    ids.extend_from_slice(&call.input_ids);
                }
                assert_eq!(call.checkpoint, index >= prefill - 1);
                if call.checkpoint {
                    selected.push(call.selected);
                }
                position += call.rows;
            }
            assert_eq!(position, capacity);
            assert_eq!(tokenizer.prompt_tokens(prompt, false).unwrap(), ids);
            assert_eq!(ids.len(), capacity - 2);
            outputs.push(selected);
        }
        assert_eq!(outputs[0], outputs[1]);
        (prompt.clone(), outputs.remove(0))
    }
}
#[test]
#[ignore = "approved populated original reference, gfx1151, fresh isolated release process"]
fn public_source_populated_context_completion_cancel_and_replay() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let capacity: usize = std::env::var("ZG_QWEN_CONTEXT").unwrap().parse().unwrap();
    let case = [1024, 2048, 4096, 8192, 16384]
        .iter()
        .position(|n| *n == capacity)
        .unwrap();
    let mode = std::env::var("ZG_QWEN_HIP_POPULATION").unwrap();
    assert!(matches!(mode.as_str(), "complete" | "cancel"));
    let h = Harness::open(&label, 2, capacity, 3, 3);
    let r = ContextReference::open(&label);
    let (prompt, expected) = r.trajectory(case, capacity, h.tokenizer.as_ref());
    let first = h.runtime.request_text(request("source", &prompt)).unwrap();
    let peer = h.runtime.request_text(request("alias", &prompt)).unwrap();
    let first_session = if mode == "cancel" {
        let token = match next(&first) {
            ResponseEvent::TokenText { token, .. } => token,
            ResponseEvent::Error(error) => panic!("populated first token: {error}"),
            _ => panic!("actual populated publication absent"),
        };
        assert_eq!(token.id, expected[0]);
        let root = token.gpu.unwrap();
        assert_eq!(root.producer_frontier, (capacity - 2) as u64);
        assert!(root.native_cohort.is_none());
        first.cancel();
        drop(first);
        root.producer_root.unwrap().session_id
    } else {
        let observed = complete(first, capacity - 2, h.tokenizer.as_ref());
        assert_eq!(observed.ids, expected);
        observed.roots[0].producer_root.unwrap().session_id
    };
    let observed = complete(peer, capacity - 2, h.tokenizer.as_ref());
    assert_eq!(observed.ids, expected);
    let peer_session = observed.roots[0].producer_root.unwrap().session_id;
    assert_ne!(first_session, peer_session);
    let replay = complete(
        h.runtime.request_text(request("source", &prompt)).unwrap(),
        capacity - 2,
        h.tokenizer.as_ref(),
    );
    assert_eq!(replay.ids, expected);
    let replay_session = replay.roots[0].producer_root.unwrap().session_id;
    assert!(replay_session != first_session && replay_session != peer_session);
    let admission = h.admission();
    assert_eq!(admission["failed_gpu_quanta"], 0);
    assert_eq!(admission["queued_requests"], 0);
    assert_eq!(admission["gpu_fused_batches"], 0);
    assert!(admission["peak_active_sessions"].as_u64().unwrap() <= 2);
    println!(
        "SOURCE_CONTEXT_SDK_OUTCOME model={label} context={capacity} mode={mode} prompt_tokens={} original_ids=true independent_roots=true replay=true cancel_requested_after_populated_token={} native_cohorts=false full_float_comparisons=0 admission={admission}",
        capacity - 2,
        mode == "cancel"
    );
    drop(r);
    h.retire();
    println!("SOURCE_CONTEXT_SDK_PASS model={label} context={capacity} mode={mode}");
}

#[test]
#[ignore = "approved canonical source, fresh isolated process; no native context allowed"]
fn public_source_32k_capacity_refuses_before_native_creation() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let snapshot = match label.as_str() {
        "08" => "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c",
        "2b" => "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        _ => panic!("approved model"),
    };
    let arbiter = ResourceArbiter::process_shared();
    let host = PhysicalDeviceKey::new("host", "system-ram").unwrap();
    let device = PhysicalDeviceKey::new("pci", "0000:c6:00.0").unwrap();
    arbiter
        .register_matching(host.clone(), 16 * 1024 * 1024 * 1024)
        .unwrap();
    let admission = zerogravity::resource_arbiter::TransferAdmission {
        arbiter: arbiter.clone(),
        resource: host.clone(),
        owner: ResourceOwner::new("capacity-test", "bootstrap").unwrap(),
    };
    let _bootstrap =
        zerogravity::tokenize_admission::approve_arbiter(&admission, "bootstrap").unwrap();
    // Baseline includes the original process-lifetime thread-control root.
    let _serving =
        zerogravity::runtime::text::pending::ServingHost::from_process_admission(&admission)
            .unwrap();
    let baseline = arbiter.snapshot(&host).unwrap();
    assert!(
        matches!(arbiter.snapshot(&device),Err(zerogravity::Error::State(s)) if s == "unknown device budget")
    );
    let runtime = Runtime::builder().build().unwrap();
    let error = runtime
        .models()
        .load(ModelLoadRequest::QwenHip {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: snapshot.into(),
            profile: QwenHipProfile {
                active: 1,
                context: 32768,
                output: 3,
                max_pending: Some(4),
            },
        })
        .unwrap_err();
    assert_eq!(error.code, "inference_error");
    assert_eq!(
        error.message,
        "resource limit: source-call complete host preflight"
    );
    assert!(runtime.models().list().unwrap().is_empty());
    assert_eq!(runtime.status().execution_adapters, 0);
    runtime.shutdown().unwrap();
    assert!(
        matches!(arbiter.snapshot(&device),Err(zerogravity::Error::State(s)) if s == "unknown device budget")
    );
    drop(runtime);
    let terminal = arbiter.snapshot(&host).unwrap();
    assert_eq!(terminal.reserved, baseline.reserved);
    assert_eq!(terminal.reservations, baseline.reservations);
    assert_eq!(terminal.per_class, baseline.per_class);
    assert_eq!(terminal.active_models, baseline.active_models);
    assert!(!terminal.fenced);
    println!(
        "SOURCE_CONTEXT_CAPACITY_PASS model={label} context=32768 source_capacity_refused=true native_context_created=false source_readers_retired=true host_reserved={}",
        baseline.reserved
    );
}
