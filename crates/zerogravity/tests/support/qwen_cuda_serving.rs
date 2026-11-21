//! Public source-backed CUDA SDK/HTTP; immutable reviewed inputs, no finite route.
use super::super::{MIB, env, json_file, number};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use zerogravity::{
    Runtime,
    cache::CacheOptions,
    models::{ModelLoadRequest, QwenServingProfile},
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        TransferAdmission,
    },
    runtime::{
        SchedulerLimits, TextTokenizer,
        text::{Request, RequestEvents, ResponseEvent, TokenMetricsRequest},
    },
};
#[path = "qwen_source_http.rs"]
mod http;

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
        .recv_timeout(Duration::from_secs(300))
        .expect("source event deadline")
}
fn complete(events: RequestEvents, expected: &[u32], tokenizer: &dyn TextTokenizer) -> u64 {
    observe(events, Some(expected), 47, tokenizer).0
}
fn observe(
    events: RequestEvents,
    expected: Option<&[u32]>,
    rows: usize,
    tokenizer: &dyn TextTokenizer,
) -> (u64, Vec<u32>) {
    let mut ids = Vec::new();
    let mut output = String::new();
    let mut root = None;
    let mut session = None;
    loop {
        match next(&events) {
            ResponseEvent::TokenText { text, token } => {
                assert_eq!(token.index, ids.len());
                if let Some(expected) = expected {
                    assert_eq!(token.id, expected[ids.len()]);
                }
                let gpu = token.gpu.unwrap();
                assert_eq!(gpu.producer_frontier, rows as u64 + ids.len() as u64);
                assert!(gpu.fused_cohort.is_none());
                let published = gpu.producer_root.unwrap();
                if let Some(session) = session {
                    assert_eq!(published.session_id, session);
                }
                if let Some(root) = root {
                    assert_ne!(published.root, root);
                }
                session = Some(published.session_id);
                root = Some(published.root);
                ids.push(token.id);
                output.push_str(&text);
            }
            ResponseEvent::Done {
                output: actual,
                prompt_tokens,
                completion_tokens,
                ..
            } => {
                if let Some(expected) = expected {
                    assert_eq!(ids, expected);
                }
                assert_eq!((prompt_tokens, completion_tokens), (rows, 3));
                assert_eq!(actual, output);
                assert_eq!(output, tokenizer.decode(&ids, true).unwrap());
                return (session.unwrap(), ids);
            }
            ResponseEvent::Error(error) => panic!("source worker: {error}"),
            ResponseEvent::Text(_) => panic!("actual publication observations required"),
        }
    }
}

#[test]
#[ignore = "owned release CUDA SDK/HTTP; canonical cache, reviewed references, private netns"]
fn public_completion_cancellation_and_retirement() {
    if cfg!(debug_assertions) {
        panic!("release physical target required");
    }
    let label = std::env::var("ZG_QWEN35_SERVING_MODEL").unwrap();
    let mode = std::env::var("ZG_QWEN35_SERVING_MODE").unwrap();
    assert!(matches!(mode.as_str(), "sdk" | "http"));
    let streamed = std::env::var("ZG_QWEN35_STREAM_WEIGHTS").as_deref() == Ok("1");
    let (snapshot, review_sha) = match label.as_str() {
        "08" => (
            "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c",
            "eefa1272e211f7f3ad949a24db136da844d90993c34fc009ca458a49d67016f8",
        ),
        "2b" => (
            "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
            "0c492b7bd63efb129422fca66ab7c5aa2a2ba05a81191d204ecb031957c03443",
        ),
        _ => panic!("explicit approved small source required"),
    };
    let arbiter = ResourceArbiter::process_shared();
    let host = PhysicalDeviceKey::new("host", "system-ram").unwrap();
    let device = PhysicalDeviceKey::new("pci", "0000:01:00.0").unwrap();
    arbiter
        .register_matching(host.clone(), 16 * 1024 * 1024 * 1024)
        .unwrap();
    let admission = TransferAdmission {
        arbiter: arbiter.clone(),
        resource: host.clone(),
        owner: ResourceOwner::new("cuda-serving-test", "support").unwrap(),
    };
    let support = arbiter
        .admit(
            &host,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 64 * MIB as u64,
            }],
        )
        .unwrap();
    let bootstrap =
        zerogravity::tokenize_admission::approve_arbiter(&admission, "bootstrap").unwrap();
    let serving =
        zerogravity::runtime::text::pending::ServingHost::from_process_admission(&admission)
            .unwrap();
    let baseline = arbiter.snapshot(&host).unwrap().reserved;
    let review = json_file(
        &env("ZG_QWEN35_REGENERATED_REVIEW").unwrap(),
        review_sha,
        MIB,
    )
    .unwrap();
    let manifest = json_file(
        &env("ZG_QWEN35_REGENERATED_REFERENCE")
            .unwrap()
            .join("full-vocabulary.json"),
        review["evidence_sha256"]["full-vocabulary.json"]
            .as_str()
            .unwrap(),
        MIB,
    )
    .unwrap();
    let calls = manifest["records"].as_array().unwrap();
    assert_eq!(calls.len(), 48);
    let ids = calls[0]["model_input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| u32::try_from(v.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>();
    let expected = calls[..3]
        .iter()
        .map(|v| u32::try_from(number(v, "selected").unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(ids.len(), 47);
    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 2,
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime
        .models()
        .load(ModelLoadRequest::QwenCuda {
            aliases: vec!["source".into(), "alias".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: snapshot.into(),
            profile: QwenServingProfile {
                active: 2,
                context: 1024,
                output: 3,
                max_pending: Some(4),
                cohort_width: 0,
            },
            provider_bytes: if streamed { 512 * MIB } else { 4608 * MIB },
            device_bytes: 5120 * MIB as u64,
        })
        .unwrap();
    let model = runtime.models().resolve("source").unwrap();
    let alias = runtime.models().resolve("alias").unwrap();
    assert!(model.shares_source(&alias));
    assert_eq!(runtime.status().execution_adapters, 1);
    let weak = Arc::downgrade(model.prepared());
    let tokenizer = model.prepared().tokenizer();
    let prompt = tokenizer.decode(&ids, false).unwrap();
    assert_eq!(tokenizer.prompt_tokens(&prompt, false).unwrap(), ids);
    if mode == "sdk" {
        let mut chat = request("alias", &prompt);
        chat.chat = true;
        assert!(runtime.request_text(chat).is_err());
        let first = runtime.request_text(request("source", &prompt)).unwrap();
        let second = runtime.request_text(request("alias", &prompt)).unwrap();
        assert_ne!(
            complete(first, &expected, tokenizer.as_ref()),
            complete(second, &expected, tokenizer.as_ref())
        );
        let cancelled = runtime.request_text(request("source", &prompt)).unwrap();
        match next(&cancelled) {
            ResponseEvent::TokenText { token, .. } => assert_eq!(token.id, expected[0]),
            ResponseEvent::Error(error) => panic!("first publication failed: {error}"),
            _ => panic!("first publication required"),
        }
        cancelled.cancel();
        drop(cancelled);
        complete(
            runtime.request_text(request("alias", &prompt)).unwrap(),
            &expected,
            tokenizer.as_ref(),
        );
        let mut run = runtime
            .run(&zerogravity::run::TaskRequest {
                model: "alias".into(),
                input: zerogravity::run::TaskInput::Text(prompt.clone()),
                max_output: 3.try_into().unwrap(),
            })
            .unwrap();
        let mut done = 0;
        while let Some(event) = run.next().unwrap() {
            match event {
                ResponseEvent::Done {
                    output,
                    prompt_tokens,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!((prompt_tokens, completion_tokens), (47, 3));
                    assert_eq!(output, tokenizer.decode(&expected, true).unwrap());
                    done += 1;
                }
                ResponseEvent::Error(error) => panic!("generic run: {error}"),
                _ => {}
            }
        }
        assert_eq!(done, 1);
        drop(run);
        let tail = " a".repeat(65);
        assert_eq!(tokenizer.prompt_tokens(&tail, false).unwrap().len(), 65);
        let first = observe(
            runtime.request_text(request("source", &tail)).unwrap(),
            None,
            65,
            tokenizer.as_ref(),
        );
        let second = observe(
            runtime.request_text(request("alias", &tail)).unwrap(),
            Some(&first.1),
            65,
            tokenizer.as_ref(),
        );
        assert_ne!(first.0, second.0);
        println!(
            "CUDA_SOURCE_CHUNK_TAIL_PASS prompt_rows=65 independent_replay=true generic_run=true independent_numerical_reference=false"
        );
    } else {
        assert_eq!(std::env::var("ZG_PRIVATE_LOOPBACK").as_deref(), Ok("1"));
        let server = runtime
            .server()
            .bind("127.0.0.1:0".parse().unwrap())
            .limits(zerogravity::serving::ingress::IngressLimits {
                connections: 2,
                body_bytes: 64 * 1024,
                output_bytes: 64 * 1024,
            })
            .enable_per_request_metrics(true)
            .build()
            .unwrap()
            .start()
            .unwrap();
        let address = server.local_addr();
        http::refuse_chat(address);
        let text = tokenizer.decode(&expected, true).unwrap();
        let mut sessions = std::collections::BTreeSet::new();
        for (id, stream, disconnect) in [
            ("json", false, false),
            ("sse", true, false),
            ("disconnect", true, true),
            ("replay", false, false),
        ] {
            let result = http::completion(
                address,
                serde_json::json!({"model":"alias", "prompt":prompt, "max_tokens":3, "stream":stream, "return_token_ids":true}),
                &expected,
                &text,
                id,
                disconnect,
                47,
            );
            if let Some((_, session)) = result {
                assert!(sessions.insert(session));
            } else {
                assert!(disconnect);
            }
        }
        server.shutdown().unwrap();
    }
    runtime.shutdown().unwrap();
    assert!(runtime.status().closed);
    assert_eq!(runtime.status().execution_adapters, 0);
    let physical = arbiter.snapshot(&device).unwrap();
    assert_eq!(physical.reserved, 512 * MIB as u64);
    assert!(!physical.fenced);
    drop((model, alias, runtime));
    assert!(weak.upgrade().is_none());
    assert!(arbiter.snapshot(&host).unwrap().reserved >= baseline + (384 + 64) * MIB as u64);
    assert_eq!(tokenizer.prompt_tokens(&prompt, false).unwrap(), ids);
    drop(tokenizer);
    let retired = arbiter.snapshot(&host).unwrap();
    assert_eq!(retired.reserved, baseline + 384 * MIB as u64);
    assert!(!retired.fenced);
    drop((manifest, review, support, bootstrap, serving));
    println!(
        "CUDA_SOURCE_SERVING_PASS model={label} mode={mode} streamed={streamed} independent_sessions=true alias_source=shared cancellation=replayed source_retired=true physical_batch_width=1 device_context_bytes={} qualification=false full_reference_acceptance=false",
        physical.reserved
    );
}
