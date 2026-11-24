//! Real offered populations on the public source route; no native cohort claim.
use super::*;
#[path = "qwen_source_budget.rs"]
mod source_budget;
use std::collections::BTreeSet;
use zerogravity::{models::ModelInstance, runtime::text::pending::ServingHost};

pub(super) struct Harness {
    pub runtime: Runtime,
    pub budget: zerogravity::models::HipSourceBudget,
    model: ModelInstance,
    alias: ModelInstance,
    pub tokenizer: Arc<dyn TextTokenizer>,
    pub reference: Option<Reference>,
    arbiter: ResourceArbiter,
    host: PhysicalDeviceKey,
    device: PhysicalDeviceKey,
    baseline: u64,
    _support: zerogravity::resource_arbiter::GlobalResourceLease,
    _serving: ServingHost,
}
impl Harness {
    pub fn open(label: &str, active: usize, context: usize, output: usize, pending: usize) -> Self {
        Self::open_inner(label, active, context, output, pending, true)
    }
    pub fn open_benchmark(
        label: &str,
        active: usize,
        context: usize,
        output: usize,
        pending: usize,
    ) -> Self {
        Self::open_inner(label, active, context, output, pending, false)
    }
    fn open_inner(
        label: &str,
        active: usize,
        context: usize,
        output: usize,
        pending: usize,
        with_reference: bool,
    ) -> Self {
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
        let snapshot = source_snapshot(label);
        let arbiter = ResourceArbiter::process_shared();
        let host = PhysicalDeviceKey::new("host", "system-ram").unwrap();
        let device = PhysicalDeviceKey::new("pci", "0000:c6:00.0").unwrap();
        arbiter
            .register_matching(host.clone(), 16 * 1024 * 1024 * 1024)
            .unwrap();
        let support = arbiter
            .admit(
                &host,
                ResourceOwner::new("population-test", "reference").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 64 * 1024 * 1024,
                }],
            )
            .unwrap();
        let reference = with_reference.then(|| {
            let manifest =
                std::env::var_os("ZG_QWEN_HIP_SDK_REFERENCE").expect("retained original manifest");
            reference(Path::new(&manifest), label)
        });
        let admission = zerogravity::resource_arbiter::TransferAdmission {
            arbiter: arbiter.clone(),
            resource: host.clone(),
            owner: ResourceOwner::new("population-test", "host").unwrap(),
        };
        let bootstrap =
            zerogravity::tokenize_admission::approve_arbiter(&admission, "bootstrap").unwrap();
        let serving = ServingHost::from_process_admission(&admission).unwrap();
        drop(bootstrap);
        let baseline = arbiter.snapshot(&host).unwrap().reserved;
        let runtime = Runtime::builder()
            .limits(SchedulerLimits {
                max_active: active,
                max_queued: Some(pending),
                ..Default::default()
            })
            .build()
            .unwrap();
        let explicit = source_budget::explicit().unwrap();
        let budget = explicit.unwrap_or_default();
        let cohort_width = std::env::var("ZG_QWEN_HIP_COHORT_WIDTH")
            .ok()
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(0);
        let profile = QwenServingProfile {
            active,
            context,
            output,
            max_pending: Some(pending),
            cohort_width,
        };
        let aliases = vec!["source".into(), "alias".into()];
        let cache = CacheOptions::new(None, 96 * 1024 * 1024 * 1024);
        let request = if std::env::var_os("ZG_QWEN_HIP_NATIVE_GEMM").is_some() {
            ModelLoadRequest::qwen_hip_native(aliases, cache, snapshot.into(), profile, budget)
        } else if let Some(budget) = explicit {
            ModelLoadRequest::QwenHipWithBudget {
                aliases,
                cache,
                snapshot: snapshot.into(),
                profile,
                budget,
            }
        } else {
            ModelLoadRequest::QwenHip {
                aliases,
                cache,
                snapshot: snapshot.into(),
                profile,
            }
        };
        runtime.models().load(request).unwrap();
        let physical = arbiter.snapshot(&device).unwrap();
        assert_eq!(physical.reserved, budget.total_device_bytes() as u64);
        assert_eq!(physical.limit, budget.physical_admission_bytes());
        assert!(!physical.fenced);
        let model = runtime.models().resolve("source").unwrap();
        let alias = runtime.models().resolve("alias").unwrap();
        assert!(model.shares_source(&alias));
        let tokenizer = model.prepared().tokenizer();
        Self {
            runtime,
            budget,
            model,
            alias,
            tokenizer,
            reference,
            arbiter,
            host,
            device,
            baseline,
            _support: support,
            _serving: serving,
        }
    }
    pub fn canonical(&self) -> (String, Vec<u32>) {
        let calls: Vec<_> = self
            .reference
            .as_ref()
            .expect("independent reference required")
            .calls
            .iter()
            .filter(|c| c.repeat == 0 && c.case == 0)
            .collect();
        assert_eq!(calls.len(), 3);
        let prompt = self.tokenizer.decode(&calls[0].input_ids, false).unwrap();
        assert_eq!(
            self.tokenizer.prompt_tokens(&prompt, false).unwrap(),
            calls[0].input_ids
        );
        (prompt, calls.iter().map(|c| c.selected).collect())
    }
    pub fn admission(&self) -> serde_json::Value {
        let engine: zerogravity_engine::Runtime = self.runtime.clone().into();
        engine.with_text_router(|r| r.admission()).unwrap()
    }
    pub fn retire(self) {
        let weak = Arc::downgrade(self.model.prepared());
        if !self.runtime.status().closed {
            let report = self.admission();
            let models = report["source_models"].as_array().unwrap();
            assert!(!models.is_empty());
            for model in models {
                let fit = &model["preparation"];
                assert_eq!(fit["provider"], "Hip");
                assert_eq!(fit["device"]["namespace"], self.device.backend);
                assert_eq!(fit["device"]["identity"], self.device.device);
                assert_eq!(
                    fit["admitted_model_capacity_bytes"].as_u64().unwrap(),
                    self.budget.model_device_bytes() as u64
                );
                assert_eq!(
                    fit["context_device_budget_bytes"].as_u64().unwrap(),
                    self.budget.context_device_bytes() as u64
                );
                assert_eq!(
                    fit["physical_limit_bytes"].as_u64().unwrap(),
                    self.budget.physical_admission_bytes()
                );
                assert!(
                    fit["model_required_bytes"].as_u64().unwrap()
                        <= fit["admitted_model_capacity_bytes"].as_u64().unwrap()
                );
                assert!(fit["host_upper_bytes"].as_u64().unwrap() > 0);
                assert_eq!(
                    fit["state_reserve_bytes"].as_u64().unwrap(),
                    fit["root_bytes"].as_u64().unwrap() * fit["root_count"].as_u64().unwrap()
                );
            }
            assert!(
                models
                    .windows(2)
                    .all(|pair| pair[0]["preparation"] == pair[1]["preparation"])
            );
            assert!(self.runtime.models().aliases(&self.alias).unwrap().len() >= 2);
            self.runtime.models().unload(&self.alias).unwrap();
            assert!(!self.runtime.status().closed);
            assert!(self.runtime.models().list().unwrap().is_empty());
            assert_eq!(self.runtime.status().execution_adapters, 0);
            let retired = self.arbiter.snapshot(&self.device).unwrap();
            assert_eq!(retired.reserved, 4 * 1024 * 1024 * 1024);
            assert!(!retired.fenced);
        }
        self.runtime.shutdown().unwrap();
        assert!(self.runtime.status().closed);
        assert_eq!(self.runtime.status().execution_adapters, 0);
        let device = self.arbiter.snapshot(&self.device).unwrap();
        assert_eq!(device.reserved, 4 * 1024 * 1024 * 1024);
        assert!(!device.fenced);
        drop((self.model, self.alias, self.runtime));
        assert!(weak.upgrade().is_none());
        let held = self.arbiter.snapshot(&self.host).unwrap();
        assert!(
            !held.fenced
                && held.reserved >= self.baseline + 2 * 1024 * 1024 * 1024 + 64 * 1024 * 1024
        );
        assert!(
            !self
                .tokenizer
                .prompt_tokens(" a", false)
                .unwrap()
                .is_empty()
        );
        drop(self.tokenizer);
        let host = self.arbiter.snapshot(&self.host).unwrap();
        assert_eq!(host.reserved, self.baseline + 2 * 1024 * 1024 * 1024);
        assert!(!host.fenced);
        println!(
            "SOURCE_POPULATION_RETIRE device_context_bytes={} host_with_support_bytes={} native_retired=true tokenizer_final_reader=true fenced=false",
            device.reserved, host.reserved
        );
    }
}

#[test]
#[ignore = "approved canonical source/reference, gfx1151, fresh isolated release process"]
fn public_source_population_completion_cancel_and_shutdown() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let clients: usize = std::env::var("ZG_QWEN_HIP_CLIENTS")
        .unwrap()
        .parse()
        .unwrap();
    let mode = std::env::var("ZG_QWEN_HIP_POPULATION").unwrap();
    assert!([1, 4, 8, 16, 32, 64].contains(&clients));
    assert!(matches!(mode.as_str(), "complete" | "cancel" | "shutdown"));
    let active = clients.min(8);
    let h = Harness::open(&label, active, 514, 3, clients + 1);
    let (prompt, expected) = h.canonical();
    let mut requests: Vec<_> = (0..clients)
        .map(|i| {
            Some(
                h.runtime
                    .request_text(request(
                        if i % 2 == 0 { "source" } else { "alias" },
                        &prompt,
                    ))
                    .unwrap(),
            )
        })
        .collect();
    let offered = h.admission();
    assert_eq!(offered["active_sessions_limit"], active);
    assert_eq!(offered["native_batch_rows_max"], 1);
    let mut canceled = 0;
    if mode == "cancel" {
        // Observe a real publication before canceling both an active reader and queued peers.
        match next(requests[0].as_ref().unwrap()) {
            ResponseEvent::TokenText { token, .. } => {
                assert_eq!(token.id, expected[0]);
                assert_eq!(token.gpu.unwrap().producer_frontier, 47);
            }
            ResponseEvent::Error(error) => panic!("first population token: {error}"),
            _ => panic!("first actual population token absent"),
        }
        for i in (0..clients).step_by(3) {
            let events = requests[i].take().unwrap();
            events.cancel();
            drop(events);
            canceled += 1;
        }
    }
    if mode == "shutdown" {
        // Close admission while work remains; original shutdown drains accepted work.
        // Four events/request fit the bounded channel without a live consumer.
        h.runtime.shutdown().unwrap();
        assert!(h.runtime.request_text(request("source", &prompt)).is_err());
    }
    let mut sessions = BTreeSet::new();
    let mut completed = 0;
    for events in requests.into_iter().flatten() {
        let result = complete(events, 47, h.tokenizer.as_ref());
        assert_eq!(result.ids, expected);
        assert!(sessions.insert(result.roots[0].producer_root.unwrap().session_id));
        completed += 1;
    }
    assert_eq!(completed + canceled, clients);
    let terminal = if mode == "shutdown" {
        serde_json::Value::Null
    } else {
        // A healthy later request exercises reuse after the entire offered population.
        let result = complete(
            h.runtime.request_text(request("alias", &prompt)).unwrap(),
            47,
            h.tokenizer.as_ref(),
        );
        assert_eq!(result.ids, expected);
        assert!(sessions.insert(result.roots[0].producer_root.unwrap().session_id));
        let state = h.admission();
        assert_eq!(state["failed_gpu_quanta"], 0);
        assert_eq!(state["queued_requests"], 0);
        assert!(state["peak_active_sessions"].as_u64().unwrap() <= active as u64);
        assert!(state["peak_active_sessions"].as_u64().unwrap() > 0);
        assert_eq!(state["gpu_fused_batches"], 0);
        state
    };
    println!(
        "SOURCE_POPULATION_OUTCOME model={label} mode={mode} offered={clients} completed_readers={completed} cancel_requested={canceled} unique_observed_sessions={} active_limit={active} fused_cohorts=false original_ids=true full_float_comparisons=0 offered_state={offered} terminal_state={terminal}",
        sessions.len()
    );
    h.retire();
    println!("SOURCE_POPULATION_PASS model={label} mode={mode} offered={clients}");
}

#[test]
#[ignore = "approved source, gfx1151, fresh isolated release process"]
fn public_aligned_multichunk_prefill_matches_singletons() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let h = Harness::open_benchmark(&label, 4, 512, 3, 65);
    let prompts = [" a", " b"].map(|p| p.repeat(81));
    for prompt in &prompts {
        assert_eq!(h.tokenizer.prompt_tokens(prompt, false).unwrap().len(), 81);
    }
    let expected = prompts.each_ref().map(|prompt| {
        complete(
            h.runtime.request_text(request("source", prompt)).unwrap(),
            81,
            h.tokenizer.as_ref(),
        )
        .ids
    });
    let before = h.admission()["gpu_fused_state_only_batches"]
        .as_u64()
        .unwrap();
    let requests = (0..4)
        .map(|i| {
            h.runtime
                .request_text(request("source", &prompts[i % 2]))
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut sessions = BTreeSet::new();
    for (i, events) in requests.into_iter().enumerate() {
        let out = complete_with_fused(events, 81, h.tokenizer.as_ref(), true);
        assert_eq!(out.ids, expected[i % 2]);
        assert!(sessions.insert(out.roots[0].producer_root.unwrap().session_id));
    }
    let admission = h.admission();
    assert!(admission["gpu_fused_state_only_batches"].as_u64().unwrap() > before);
    assert_eq!(admission["failed_gpu_quanta"], 0);
    h.retire();
}

#[test]
#[ignore = "approved source, gfx1151, fresh isolated release process; benchmark replay only"]
fn public_source_population_benchmark() {
    let label = std::env::var("ZG_QWEN_HIP_SDK_MODEL").unwrap();
    let clients: Vec<usize> = std::env::var("ZG_QWEN_HIP_BENCH_CLIENTS")
        .unwrap_or_else(|_| "1,4,8,16,32,64".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert!(!clients.is_empty() && clients.iter().all(|c| [1, 4, 8, 16, 32, 64].contains(c)));
    let cohort_width = std::env::var("ZG_QWEN_HIP_COHORT_WIDTH")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(0);
    let load = Instant::now();
    let h = Harness::open_benchmark(&label, 8, 1024, 3, 65);
    println!(
        "HIP_BENCH_LOAD {}",
        serde_json::json!({"model":label,"load_ms":load.elapsed().as_secs_f64()*1000.0,
        "active_limit":8,"context":1024,"output":3,
        "total_device_budget_bytes":h.budget.total_device_bytes(),
        "context_device_bytes":h.budget.context_device_bytes(),
        "model_device_capacity_bytes":h.budget.model_device_bytes(),
        "physical_admission_bytes":h.budget.physical_admission_bytes(),
        "physical_reserved_bytes":h.arbiter.snapshot(&h.device).unwrap().reserved})
    );
    let prompt = "<|im_start|>user\nContinue this list of integers, separated by commas, for many more terms: 1, 2, 3, 4, 5, 6, 7, 8,<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids = h.tokenizer.prompt_tokens(prompt, false).unwrap();
    assert_eq!(prompt_ids.len(), 47);
    println!(
        "HIP_BENCH_INPUT {}",
        serde_json::json!({"model":label,"prompt":prompt,"prompt_token_ids":prompt_ids})
    );
    for clients in clients {
        let mut expected = None;
        for wave in 0..3 {
            let start = Instant::now();
            let requests: Vec<_> = (0..clients)
                .map(|_| h.runtime.request_text(request("source", prompt)).unwrap())
                .collect();
            let mut sessions = BTreeSet::new();
            let mut outputs = Vec::with_capacity(clients);
            for events in requests {
                let output = complete_with_fused(
                    events,
                    47,
                    h.tokenizer.as_ref(),
                    cohort_width != 0 && clients > 1,
                );
                assert!(sessions.insert(output.roots[0].producer_root.unwrap().session_id));
                outputs.push(output.ids);
            }
            if let Some(expected) = &expected {
                assert_eq!(&outputs, expected);
            } else {
                expected = Some(outputs.clone());
            }
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            let lane_invariant = outputs
                .first()
                .is_none_or(|first| outputs.iter().all(|ids| ids == first));
            let admission = h.admission();
            assert_eq!(admission["failed_gpu_quanta"], 0);
            assert_eq!(admission["queued_requests"], 0);
            if cohort_width == 0 || clients == 1 {
                assert_eq!(admission["gpu_fused_batches"], 0);
            } else {
                assert!(admission["gpu_fused_batches"].as_u64().unwrap() > 0);
            }
            println!(
                "HIP_BENCH_WAVE {}",
                serde_json::json!({"model":label,"clients":clients,"wave":wave,
                "warmup":wave==0,"elapsed_ms":elapsed_ms,"completed":outputs.len(),"output_tokens":3*outputs.len(),
                "token_ids":outputs,"expected":expected,"replay_match":true,"lane_invariant":lane_invariant,
                "native_batch_width":cohort_width,"active_sessions_limit":8,"admission":admission})
            );
        }
    }
    h.retire();
    println!("HIP_BENCH_CLOSED model={label} closed=true replay_match=true");
}
