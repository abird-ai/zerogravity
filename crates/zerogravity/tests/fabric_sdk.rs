//! External consumer uses the public facade alone.
use zerogravity::{
    Error, Runtime,
    run::{TaskInput, TaskRequest},
};
#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[path = "support/language_pipeline.rs"]
mod language_pipeline;

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[allow(dead_code)]
#[path = "support/qwen_source_http.rs"]
mod qwen_source_http;

#[cfg(all(
    feature = "model-qwen35",
    any(feature = "hip", feature = "cuda"),
    target_os = "linux"
))]
#[path = "support/qwen_prepared_observations.rs"]
mod qwen_prepared_observations;

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
fn qwen_cuda_native_runtime(paged: bool) -> Runtime {
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenCudaStatePolicy, QwenServingProfile},
        runtime::SchedulerLimits,
    };
    const MIB: usize = 1024 * 1024;
    const SNAPSHOT: &str = "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c";

    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 8,
            max_queued: Some(16),
            ..Default::default()
        })
        .build()
        .unwrap();
    let profile = QwenServingProfile {
        active: 8,
        context: 512,
        output: 3,
        max_pending: Some(16),
        cohort_width: 0,
    };
    let load = if paged {
        ModelLoadRequest::QwenCudaWithStatePolicy {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: SNAPSHOT.into(),
            profile,
            provider_bytes: 4608 * MIB,
            device_bytes: 5120 * MIB as u64,
            state_policy: QwenCudaStatePolicy::Paged {
                page_tokens: 4,
                max_append_tokens: 64,
            },
        }
    } else {
        ModelLoadRequest::QwenCuda {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: SNAPSHOT.into(),
            profile,
            provider_bytes: 4608 * MIB,
            device_bytes: 5120 * MIB as u64,
        }
    };
    runtime.models().load(load).unwrap();
    runtime
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
fn qwen_cuda_context_runtime(model: &str, context: usize, output: usize) -> Runtime {
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenCudaStatePolicy, QwenServingProfile},
        runtime::SchedulerLimits,
    };
    const MIB: usize = 1024 * 1024;
    let snapshot = match model {
        "08" => "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c",
        "2b" => "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        _ => panic!("explicit approved small CUDA source required"),
    };
    assert!(matches!(context, 1024 | 2048 | 4096 | 8192 | 16384 | 32768));
    assert!((1..=3).contains(&output) && output < context);

    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 1,
            max_queued: Some(1),
            ..Default::default()
        })
        .build()
        .unwrap();
    let profile = QwenServingProfile {
        active: 1,
        context,
        output,
        max_pending: Some(1),
        cohort_width: 0,
    };
    let request = if std::env::var_os("ZG_QWEN_CUDA_PAGED").is_some() {
        ModelLoadRequest::QwenCudaWithStatePolicy {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: snapshot.into(),
            profile,
            provider_bytes: 4608 * MIB,
            device_bytes: 5120 * MIB as u64,
            state_policy: QwenCudaStatePolicy::Paged {
                page_tokens: 64,
                max_append_tokens: 64,
            },
        }
    } else {
        ModelLoadRequest::QwenCuda {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: snapshot.into(),
            profile,
            provider_bytes: 4608 * MIB,
            device_bytes: 5120 * MIB as u64,
        }
    };
    runtime.models().load(request).unwrap();
    runtime
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
fn qwen_cuda_population_runtime(model: &str) -> Runtime {
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenServingProfile},
        runtime::SchedulerLimits,
    };
    const MIB: usize = 1024 * 1024;
    let snapshot = match model {
        "08" => "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c",
        "2b" => "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        "27b" => "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
        _ => panic!("explicit approved CUDA source required"),
    };
    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 8,
            max_queued: Some(128),
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime
        .models()
        .load(ModelLoadRequest::QwenCuda {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: snapshot.into(),
            profile: QwenServingProfile {
                active: 8,
                context: 512,
                output: 3,
                max_pending: Some(128),
                cohort_width: 0,
            },
            provider_bytes: 4608 * MIB,
            device_bytes: 5120 * MIB as u64,
        })
        .unwrap();
    runtime
}

#[test]
fn external_sdk_compiles_without_cli_or_concrete_model_features() {
    let runtime = Runtime::builder().build().unwrap();
    assert!(runtime.models().list().unwrap().is_empty());
    assert_eq!(runtime.status().execution_adapters, 0);
    // Compile the inventory API without probing the test host.
    let _inventory = |devices: &zerogravity::devices::Devices<'_>| devices.inventory();
    assert!(matches!(
        runtime.run(&TaskRequest {
            model: "unloaded".into(),
            input: TaskInput::Text("test".into()),
            max_output: 1.try_into().unwrap(),
        }),
        Err(Error::Unsupported(_))
    ));
    #[cfg(feature = "serve")]
    {
        let server = zerogravity::serving::Server::builder(runtime.clone())
            .build()
            .unwrap();
        let _also_builder = zerogravity::serving::ServerBuilder::new(runtime.clone());
        let _runtime_builder = runtime.server();
        assert!(server.address().ip().is_loopback());
        assert!(matches!(server.start(), Err(Error::Limit(_))));
    }
    runtime.shutdown().unwrap();
    assert!(matches!(
        runtime.devices().inventory(),
        Err(Error::State(_))
    ));
    assert!(
        runtime
            .artifacts()
            .inspect(std::path::Path::new("/not-opened"))
            .is_err()
    );
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[test]
#[ignore = "approved small CUDA snapshots; retained populated-context public SDK qualification"]
fn qwen_cuda_populated_context_sdk_replays() {
    use std::time::{Duration, Instant};
    use zerogravity::runtime::text::{Request, ResponseEvent, TokenMetricsRequest};

    let model = std::env::var("ZG_QWEN_CUDA_CONTEXT_MODEL").unwrap();
    let context: usize = std::env::var("ZG_QWEN_CUDA_CONTEXT")
        .unwrap()
        .parse()
        .unwrap();
    const OUTPUT: usize = 2;
    let prompt_tokens = std::env::var("ZG_QWEN_CUDA_PROMPT_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>().unwrap())
        .unwrap_or(context - OUTPUT);
    assert!(prompt_tokens > 0 && prompt_tokens + OUTPUT <= context);
    let runtime = qwen_cuda_context_runtime(&model, context, OUTPUT);
    let resolved = runtime.models().resolve("source").unwrap();
    let tokenizer = resolved.prepared().tokenizer();
    let prompt = " a".repeat(prompt_tokens);
    assert_eq!(
        tokenizer.prompt_tokens(&prompt, false).unwrap().len(),
        prompt_tokens
    );

    let run = || {
        let events = runtime
            .request_text(Request {
                model: "source".into(),
                input: prompt.clone(),
                chat: false,
                max_tokens: OUTPUT,
                stream: true,
                token_metrics: Some(TokenMetricsRequest {
                    started: Instant::now(),
                    wire_request_id: None,
                }),
            })
            .unwrap();
        let mut ids = Vec::with_capacity(OUTPUT);
        let mut roots = Vec::with_capacity(OUTPUT);
        loop {
            match events
                .events
                .recv_timeout(Duration::from_secs(600))
                .expect("populated CUDA context response deadline")
            {
                ResponseEvent::TokenText { token, .. } => {
                    let gpu = token.gpu.expect("populated CUDA token GPU metadata");
                    assert_eq!(gpu.producer_frontier, (prompt_tokens + ids.len()) as u64);
                    assert!(gpu.fused_cohort.is_none());
                    roots.push(gpu.producer_root.expect("populated CUDA producer root"));
                    ids.push(token.id);
                }
                ResponseEvent::Done {
                    prompt_tokens: actual_prompt,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!((actual_prompt, completion_tokens), (prompt_tokens, OUTPUT));
                    break;
                }
                ResponseEvent::Error(error) => panic!("populated CUDA context: {error}"),
                ResponseEvent::Text(_) => panic!("streaming token observations required"),
            }
        }
        assert_eq!(ids.len(), OUTPUT);
        assert_eq!(roots.len(), OUTPUT);
        assert_eq!(roots[0].session_id, roots[1].session_id);
        assert_ne!(roots[0].root, roots[1].root);
        (ids, roots[0].session_id)
    };

    let first = run();
    let replay = run();
    assert_eq!(first.0, replay.0);
    assert_ne!(first.1, replay.1);
    drop(resolved);
    runtime.shutdown().unwrap();
    assert!(runtime.status().closed);
    println!(
        "P12_CUDA_POPULATED_CONTEXT_PASS model={model} context={context} prompt_tokens={prompt_tokens} output_tokens={OUTPUT} replay=exact independent_sessions=true producer_frontiers=exact final_shutdown=true independent_all_state_domains_reference=false"
    );
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[test]
#[ignore = "approved CUDA snapshots; matched public-SDK population benchmark"]
fn qwen_cuda_public_sdk_population_benchmark() {
    use std::{
        collections::BTreeSet,
        time::{Duration, Instant},
    };
    use zerogravity::runtime::text::{Request, ResponseEvent, TokenMetricsRequest};

    let model = std::env::var("ZG_QWEN_CUDA_BENCH_MODEL").unwrap();
    let clients = std::env::var("ZG_QWEN_CUDA_BENCH_CLIENTS")
        .unwrap_or_else(|_| "1,4,8,16,32,64".into())
        .split(',')
        .map(|value| value.parse::<usize>().unwrap())
        .collect::<Vec<_>>();
    assert!(
        !clients.is_empty()
            && clients
                .iter()
                .all(|value| [1, 4, 8, 16, 32, 64].contains(value))
    );
    let runtime = qwen_cuda_population_runtime(&model);
    let resolved = runtime.models().resolve("source").unwrap();
    let tokenizer = resolved.prepared().tokenizer();
    let prompt = "<|im_start|>user\nContinue this list of integers, separated by commas, for many more terms: 1, 2, 3, 4, 5, 6, 7, 8,<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids = tokenizer.prompt_tokens(prompt, false).unwrap();
    assert_eq!(prompt_ids.len(), 47);
    println!(
        "CUDA_BENCH_INPUT {}",
        serde_json::json!({"model":model,"prompt":prompt,"prompt_token_ids":prompt_ids})
    );

    for clients in clients {
        let mut expected = None;
        for wave in 0..3 {
            let started = Instant::now();
            let events = (0..clients)
                .map(|_| {
                    runtime
                        .request_text(Request {
                            model: "source".into(),
                            input: prompt.into(),
                            chat: false,
                            max_tokens: 3,
                            stream: true,
                            token_metrics: Some(TokenMetricsRequest {
                                started,
                                wire_request_id: None,
                            }),
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let mut outputs = Vec::with_capacity(clients);
            let mut cohorts = BTreeSet::new();
            for events in events {
                let mut ids = Vec::with_capacity(3);
                loop {
                    match events
                        .events
                        .recv_timeout(Duration::from_secs(300))
                        .expect("CUDA benchmark response deadline")
                    {
                        ResponseEvent::TokenText { token, .. } => {
                            if let Some(cohort) = token.gpu.and_then(|gpu| gpu.fused_cohort) {
                                cohorts.insert(cohort.width);
                            }
                            ids.push(token.id);
                        }
                        ResponseEvent::Done {
                            prompt_tokens,
                            completion_tokens,
                            ..
                        } => {
                            assert_eq!((prompt_tokens, completion_tokens), (47, 3));
                            break;
                        }
                        ResponseEvent::Error(error) => panic!("CUDA benchmark worker: {error}"),
                        ResponseEvent::Text(_) => panic!("streaming token observations required"),
                    }
                }
                assert_eq!(ids.len(), 3);
                outputs.push(ids);
            }
            if let Some(expected) = &expected {
                assert_eq!(&outputs, expected);
            } else {
                expected = Some(outputs.clone());
            }
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let lane_invariant = outputs
                .first()
                .is_none_or(|first| outputs.iter().all(|ids| ids == first));
            println!(
                "CUDA_BENCH_WAVE {}",
                serde_json::json!({
                    "model":model,
                    "clients":clients,
                    "wave":wave,
                    "warmup":wave==0,
                    "elapsed_ms":elapsed_ms,
                    "completed":outputs.len(),
                    "output_tokens":3*outputs.len(),
                    "token_ids":outputs,
                    "expected":expected,
                    "cohort_widths":cohorts,
                    "replay_match":true,
                    "lane_invariant":lane_invariant
                })
            );
        }
    }
    runtime.models().unload(&resolved).unwrap();
    assert!(!runtime.status().closed);
    assert!(runtime.models().list().unwrap().is_empty());
    assert_eq!(runtime.status().execution_adapters, 0);
    assert_eq!(tokenizer.prompt_tokens(prompt, false).unwrap(), prompt_ids);
    drop(resolved);
    runtime.shutdown().unwrap();
    assert!(runtime.status().closed);
    println!("CUDA_BENCH_CLOSED model={model} closed=true replay_match=true");
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[test]
#[ignore = "approved original CUDA cache; physical public SDK contiguous-vs-paged native cohort; compare timings only under a matched optimized Cargo profile"]
fn qwen_cuda_paged_native_sdk_matches_contiguous() {
    use std::{
        collections::BTreeSet,
        time::{Duration, Instant},
    };
    use zerogravity::runtime::text::{Request, ResponseEvent, TokenMetricsRequest};

    let run = |paged: bool| {
        let runtime = qwen_cuda_native_runtime(paged);
        let engine: zerogravity_engine::Runtime = runtime.clone().into();
        let report = engine
            .with_text_router(|router| router.admission())
            .unwrap();
        use zerogravity::{
            models::ModelMemory,
            provider::{ObservationSupport, ProviderKind, ResidentStatePolicy},
        };
        let snapshot = runtime.snapshot().unwrap();
        assert_eq!(snapshot.status.loaded_owners, 1);
        let details = &snapshot.catalog[0].capabilities.as_ref().unwrap().details;
        let execution = details.execution.as_ref().unwrap();
        assert_eq!(execution.selected.provider, ProviderKind::Cuda);
        assert_eq!(
            execution.text_calls.as_ref().unwrap().observations,
            ObservationSupport::None
        );
        assert_eq!(
            matches!(
                details.state_policy,
                Some(ResidentStatePolicy::Paged { .. })
            ),
            paged
        );
        assert_eq!(details.prefix_sharing, Some(paged));
        let Some(ModelMemory::ResidentText {
            admission,
            resident_bytes,
        }) = &details.memory
        else {
            panic!("original native text admission absent");
        };
        assert!(*resident_bytes > 0 && *resident_bytes <= admission.admitted_model_capacity_bytes);
        assert_eq!(
            report["catalog"][0]["capabilities"]["details"]["memory"]["resident_text"]["admission"],
            serde_json::to_value(admission).unwrap()
        );
        let fit = &report["source_models"][0]["preparation"];
        assert_eq!(fit["provider"], "Cuda");
        assert_eq!(fit["device"]["namespace"], "pci");
        assert_eq!(
            fit["physical_limit_bytes"].as_u64().unwrap(),
            5120 * 1024 * 1024
        );
        assert_eq!(
            fit["context_device_budget_bytes"].as_u64().unwrap(),
            512 * 1024 * 1024
        );
        assert!(fit["host_upper_bytes"].is_null());
        assert!(
            fit["model_required_bytes"].as_u64().unwrap()
                <= fit["admitted_model_capacity_bytes"].as_u64().unwrap()
        );
        let full_roots = fit["root_bytes"].as_u64().unwrap() * fit["root_count"].as_u64().unwrap();
        let actual_state = fit["state_reserve_bytes"].as_u64().unwrap();
        if paged {
            assert!(actual_state <= full_roots);
        } else {
            assert_eq!(actual_state, full_roots);
        }
        drop(engine);
        let model = runtime.models().resolve("source").unwrap();
        let tokenizer = model.prepared().tokenizer();
        let prompt = " a".repeat(48);
        assert_eq!(tokenizer.prompt_tokens(&prompt, false).unwrap().len(), 48);
        let wave = || {
            let wave_started = Instant::now();
            let events = (0..8)
                .map(|_| {
                    runtime
                        .request_text(Request {
                            model: "source".into(),
                            input: prompt.clone(),
                            chat: false,
                            max_tokens: 3,
                            stream: true,
                            token_metrics: Some(TokenMetricsRequest {
                                started: wave_started,
                                wire_request_id: None,
                            }),
                        })
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let mut outputs = Vec::with_capacity(events.len());
            let mut cohort_widths = BTreeSet::new();
            let mut token_commit_ms = [0.0f64; 3];
            for events in events {
                let mut ids = Vec::new();
                loop {
                    match events
                        .events
                        .recv_timeout(Duration::from_secs(300))
                        .expect("paged SDK response deadline")
                    {
                        ResponseEvent::TokenText { token, .. } => {
                            if let Some(cohort) = token.gpu.and_then(|gpu| gpu.fused_cohort) {
                                cohort_widths.insert(cohort.width);
                            }
                            token_commit_ms[ids.len()] += token
                                .commit_observed_at
                                .duration_since(wave_started)
                                .as_secs_f64()
                                * 1000.0;
                            ids.push(token.id);
                        }
                        ResponseEvent::Done {
                            prompt_tokens,
                            completion_tokens,
                            ..
                        } => {
                            assert_eq!((prompt_tokens, completion_tokens), (48, 3));
                            break;
                        }
                        ResponseEvent::Error(error) => panic!("paged SDK worker: {error}"),
                        ResponseEvent::Text(_) => panic!("streaming token observations required"),
                    }
                }
                assert_eq!(ids.len(), 3);
                outputs.push(ids);
            }
            let wave_ms = wave_started.elapsed().as_secs_f64() * 1000.0;
            assert!(
                cohort_widths.iter().any(|&width| width > 1),
                "native cohort publication was not observed"
            );
            for value in &mut token_commit_ms {
                *value /= 8.0;
            }
            (outputs, cohort_widths, wave_ms, token_commit_ms)
        };
        let warm = wave();
        let measured = wave();
        assert_eq!(warm.0, measured.0);
        runtime.shutdown().unwrap();
        assert!(runtime.status().closed);
        measured
    };

    let contiguous = run(false);
    let paged = run(true);
    assert_eq!(paged.0, contiguous.0);
    println!(
        "P12_SOURCE_PAGED_SDK_PASS requests=8 prompt_tokens=48 output_tokens=3 contiguous_cohorts={:?} paged_cohorts={:?} contiguous_wave_ms={:.3} paged_wave_ms={:.3} contiguous_output_tok_s={:.3} paged_output_tok_s={:.3} contiguous_commit_ms={:?} paged_commit_ms={:?} tokens=exact final_shutdown=true",
        contiguous.1,
        paged.1,
        contiguous.2,
        paged.2,
        24_000.0 / contiguous.2,
        24_000.0 / paged.2,
        contiguous.3,
        paged.3,
    );
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[test]
#[ignore = "approved original CUDA cache; private loopback public HTTP contiguous-vs-paged native cohort"]
fn qwen_cuda_paged_native_http_matches_contiguous() {
    use serde_json::{Value, json};
    use std::{
        collections::BTreeSet,
        io::{BufReader, Read},
    };

    let run = |paged: bool| {
        let runtime = qwen_cuda_native_runtime(paged);
        let model = runtime.models().resolve("source").unwrap();
        let tokenizer = model.prepared().tokenizer();
        let prompt = " a".repeat(4);
        assert_eq!(tokenizer.prompt_tokens(&prompt, false).unwrap().len(), 4);
        let server = runtime
            .server()
            .bind("127.0.0.1:0".parse().unwrap())
            .limits(zerogravity::serving::ingress::IngressLimits {
                connections: 8,
                body_bytes: 64 * 1024,
                output_bytes: 64 * 1024,
            })
            .enable_per_request_metrics(true)
            .build()
            .unwrap()
            .start()
            .unwrap();
        let address = server.local_addr();
        let sockets = (0..8)
            .map(|index| {
                qwen_source_http::connect(
                    address,
                    "/v1/completions",
                    &json!({
                        "model":"source",
                        "prompt":prompt,
                        "max_tokens":2,
                        "stream":false,
                        "return_token_ids":true
                    }),
                    &format!("paged-http-{index}"),
                )
            })
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(sockets.len());
        let mut cohort_widths = BTreeSet::new();
        for socket in sockets {
            let mut reader = BufReader::new(socket);
            let head = qwen_source_http::headers(&mut reader);
            assert!(head.starts_with("HTTP/1.1 200"), "{head}");
            let mut raw = Vec::new();
            reader.take(64 * 1024 + 1).read_to_end(&mut raw).unwrap();
            assert!(raw.len() <= 64 * 1024);
            let document: Value = serde_json::from_slice(&raw).unwrap();
            assert!(document.get("error").is_none(), "{document}");
            assert_eq!(document["usage"]["prompt_tokens"], 4);
            assert_eq!(document["usage"]["completion_tokens"], 2);
            let ids = document["choices"][0]["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            assert_eq!(ids.len(), 2);
            for token in document["zerogravity_token_metrics"]["tokens"]
                .as_array()
                .unwrap()
            {
                if let Some(width) = token["gpu"]["fused_cohort"]["width"].as_u64() {
                    cohort_widths.insert(width as usize);
                }
            }
            outputs.push(ids);
        }
        assert!(
            cohort_widths.iter().any(|&width| width > 1),
            "native HTTP cohort publication was not observed"
        );
        server.shutdown().unwrap();
        runtime.shutdown().unwrap();
        assert!(runtime.status().closed);
        (outputs, cohort_widths)
    };

    let contiguous = run(false);
    let paged = run(true);
    assert_eq!(paged.0, contiguous.0);
    println!(
        "P12_SOURCE_PAGED_HTTP_PASS requests=8 prompt_tokens=4 output_tokens=2 contiguous_cohorts={:?} paged_cohorts={:?} tokens=exact final_shutdown=true",
        contiguous.1, paged.1
    );
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[test]
#[ignore = "approved original CUDA cache; trusted server-side scoped paged prefix reuse"]
fn qwen_cuda_paged_scoped_http_skips_replay() {
    use serde_json::{Value, json};
    use std::{
        io::{BufReader, Read, Write},
        net::{SocketAddr, TcpStream},
    };
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenCudaStatePolicy, QwenServingProfile},
        runtime::{PrefixSharingScope, SchedulerLimits},
    };

    const MIB: usize = 1024 * 1024;
    const SNAPSHOT: &str = "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c";

    let runtime = Runtime::builder()
        .limits(SchedulerLimits {
            max_active: 2,
            max_queued: Some(4),
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime
        .models()
        .load(ModelLoadRequest::QwenCudaWithStatePolicy {
            aliases: vec!["source".into()],
            cache: CacheOptions::new(None, 96 * 1024 * 1024 * 1024),
            snapshot: SNAPSHOT.into(),
            profile: QwenServingProfile {
                active: 2,
                context: 512,
                output: 1,
                max_pending: Some(4),
                cohort_width: 0,
            },
            provider_bytes: 4608 * MIB,
            device_bytes: 5120 * MIB as u64,
            state_policy: QwenCudaStatePolicy::Paged {
                page_tokens: 4,
                max_append_tokens: 64,
            },
        })
        .unwrap();
    let prompt = " a".repeat(65);
    assert_eq!(
        runtime
            .models()
            .resolve("source")
            .unwrap()
            .prepared()
            .tokenizer()
            .prompt_tokens(&prompt, false)
            .unwrap()
            .len(),
        65
    );

    let request = |address: SocketAddr, label: &str| {
        let socket = qwen_source_http::connect(
            address,
            "/v1/completions",
            &json!({
                "model":"source",
                "prompt":prompt,
                "max_tokens":1,
                "stream":false,
                "return_token_ids":true
            }),
            label,
        );
        let mut reader = BufReader::new(socket);
        let head = qwen_source_http::headers(&mut reader);
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        let mut raw = Vec::new();
        reader.take(64 * 1024 + 1).read_to_end(&mut raw).unwrap();
        assert!(raw.len() <= 64 * 1024);
        let document: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(document["usage"]["prompt_tokens"], 65);
        assert_eq!(document["usage"]["completion_tokens"], 1);
        document["choices"][0]["token_ids"][0].as_u64().unwrap() as u32
    };
    let model_tokens = |address: SocketAddr| {
        let mut socket = TcpStream::connect(address).unwrap();
        socket
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .unwrap();
        let mut reader = BufReader::new(socket);
        let head = qwen_source_http::headers(&mut reader);
        assert!(head.starts_with("HTTP/1.1 200"), "{head}");
        let mut raw = Vec::new();
        reader.read_to_end(&mut raw).unwrap();
        let document: Value = serde_json::from_slice(&raw).unwrap();
        document["admission"]["completed_model_token_steps"]
            .as_u64()
            .unwrap()
    };
    let server = |scope| {
        runtime
            .server()
            .bind("127.0.0.1:0".parse().unwrap())
            .limits(zerogravity::serving::ingress::IngressLimits {
                connections: 4,
                body_bytes: 64 * 1024,
                output_bytes: 64 * 1024,
            })
            .enable_per_request_metrics(true)
            .prefix_sharing_scope(Some(scope))
            .build()
            .unwrap()
            .start()
            .unwrap()
    };

    let scope_a = PrefixSharingScope::new(7).unwrap();
    let scope_b = PrefixSharingScope::new(8).unwrap();
    let server_a = server(scope_a);
    let address_a = server_a.local_addr();
    let before = model_tokens(address_a);
    let warm = request(address_a, "prefix-http-warm");
    let after_warm = model_tokens(address_a);
    let hit = request(address_a, "prefix-http-hit");
    let after_hit = model_tokens(address_a);
    server_a.shutdown().unwrap();

    let server_b = server(scope_b);
    let address_b = server_b.local_addr();
    let miss = request(address_b, "prefix-http-miss");
    let after_miss = model_tokens(address_b);
    server_b.shutdown().unwrap();

    assert_eq!(warm, hit);
    assert_eq!(warm, miss);
    assert_eq!(after_warm - before, 65);
    assert_eq!(after_hit - after_warm, 1);
    assert_eq!(after_miss - after_hit, 65);
    runtime.shutdown().unwrap();
    assert!(runtime.status().closed);
    println!(
        "P12_SOURCE_PAGED_PREFIX_HTTP_PASS prompt_tokens=65 cached_prefix=64 warm_model_tokens=65 same_scope_model_tokens=1 different_scope_model_tokens=65 output_id_exact=true server_scoped=true client_scope=false final_shutdown=true"
    );
}

#[cfg(all(feature = "qwen35-hip", target_os = "linux"))]
#[test]
fn canonical_hip_registration_checks_runtime_limits_and_aliases_before_source_io() {
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenServingProfile},
    };
    let runtime = Runtime::builder().build().unwrap();
    let request = |active, aliases| ModelLoadRequest::QwenHip {
        aliases,
        cache: CacheOptions::new(Some("/source-not-opened".into()), 96 * 1024 * 1024 * 1024),
        snapshot: "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52".into(),
        profile: QwenServingProfile {
            active,
            context: 514,
            output: 2,
            max_pending: Some(1),
            cohort_width: 0,
        },
    };
    let error = runtime
        .models()
        .load(request(2, vec!["source".into()]))
        .unwrap_err();
    assert_eq!(error.code, "invalid_config");
    let error = runtime
        .models()
        .load(request(1, vec!["source".into(), "source".into()]))
        .unwrap_err();
    assert_eq!(error.code, "invalid_model");
    let mut oversized = request(1, vec!["source".into()]);
    if let ModelLoadRequest::QwenHip { profile, .. } = &mut oversized {
        profile.context = 16384;
        profile.output = 513;
    }
    assert_eq!(
        runtime.models().load(oversized).unwrap_err().code,
        "invalid_source_profile"
    );
    assert!(runtime.models().list().unwrap().is_empty());
    assert_eq!(runtime.status().execution_adapters, 0);
    runtime.shutdown().unwrap();
    assert!(
        runtime
            .models()
            .load(request(1, vec!["source".into()]))
            .is_err()
    );
}

#[test]
fn semantic_api_is_the_original_program_type() {
    fn original(
        plan: &zerogravity::accelerator::program::ForwardPlan,
    ) -> &zerogravity::semantic::ForwardPlan {
        plan
    }
    let _same_type = original;
    let _family_contract: Option<&dyn zerogravity::models::ModelDefinition> = None;
    let _assembly: Option<zerogravity::models::ModelAssembly> = None;
    // Source-owned preparation is usable without naming an internal crate.
    #[cfg(feature = "model-qwen35")]
    let _record: fn(
        std::sync::Arc<zerogravity::qwen35::LoadedQwenModel>,
        zerogravity::models::ProgramRequest,
    ) -> zerogravity::Result<zerogravity::models::ModelProgram> =
        zerogravity::models::ModelProgram::record;
    let _source_contract: Option<&dyn zerogravity::models::TensorSource> = None;
    for component in zerogravity::builtins::components() {
        let has_cpu = cfg!(all(feature = "cpu", feature = "text", feature = "cache"));
        let has_gpu = cfg!(all(
            feature = "text",
            any(feature = "cuda", feature = "hip", feature = "vulkan")
        ));
        let expected = if (has_cpu && matches!(component.component, "cpu" | "qwen35"))
            || (has_gpu
                && component.compiled
                && matches!(component.component, "cuda" | "hip" | "vulkan" | "qwen35"))
        {
            zerogravity::diagnostics::MigrationStatus::TextAdapter
        } else {
            zerogravity::diagnostics::MigrationStatus::AwaitingAdapter
        };
        assert_eq!(component.fabric, expected);
    }
}

#[cfg(all(
    feature = "text",
    any(feature = "cuda", feature = "hip", feature = "vulkan")
))]
#[test]
fn ordinary_gpu_sdk_obeys_runtime_limits_and_close_before_source_or_device_lookup() {
    let runtime = Runtime::builder().build().unwrap();
    let mut profile = zerogravity::builtins::text::GpuProfile {
        workers: 1,
        active: 2,
        context: 512,
        output: 1,
        provider_mib: 256,
        device_mib: 256,
        cohort_width: 0,
        host_mib: 256,
        max_pending: Some(1),
    };
    let request = || {
        vec![(
            "not-opened".into(),
            std::path::PathBuf::from("/source-not-opened"),
        )]
    };
    let error = runtime
        .models()
        .load(zerogravity::models::ModelLoadRequest::GpuText {
            declared: request(),
            backend: zerogravity::devices::GpuBackend::Cuda,
            profile,
        })
        .unwrap_err();
    assert_eq!(error.code, "invalid_config");
    assert!(error.message.contains("runtime limits"));
    profile.active = 1;
    profile.max_pending = None;
    assert_eq!(
        runtime
            .load_gpu_text(request(), zerogravity::devices::GpuBackend::Cuda, profile)
            .unwrap_err()
            .code,
        "invalid_config"
    );
    assert_eq!(runtime.status().execution_adapters, 0);
    runtime.shutdown().unwrap();
    profile.max_pending = Some(1);
    assert_eq!(
        runtime
            .load_gpu_text(request(), zerogravity::devices::GpuBackend::Cuda, profile)
            .unwrap_err()
            .code,
        "runtime_closed"
    );
}

#[test]
fn catalog_registration_uses_original_limits_and_closed_lifecycle() {
    use zerogravity::{models::ModelLoadRequest, runtime::TextConfig};
    let runtime = Runtime::builder().build().unwrap();
    let mut config = TextConfig {
        active_per_worker: 2,
        ..Default::default()
    };
    let request = |config| ModelLoadRequest::CpuText {
        declared: vec![("never-opened".into(), "/fabric-source-not-opened".into())],
        config,
    };
    assert_eq!(
        runtime.models().load(request(config)).unwrap_err().code,
        "invalid_config"
    );
    config.active_per_worker = 1;
    config.max_pending = None;
    assert_eq!(
        runtime.models().load(request(config)).unwrap_err().code,
        "invalid_config"
    );
    assert_eq!(runtime.status().execution_adapters, 0);
    assert!(runtime.models().list().unwrap().is_empty());
    assert!(matches!(
        runtime.models().resolve("never-opened"),
        Err(Error::Unsupported(_))
    ));
    runtime.shutdown().unwrap();
    assert_eq!(
        runtime.models().load(request(config)).unwrap_err().code,
        "runtime_closed"
    );
    assert!(matches!(
        runtime.models().resolve("never-opened"),
        Err(Error::State(_))
    ));
}

#[cfg(not(all(feature = "cpu", feature = "text", feature = "cache")))]
#[test]
fn unavailable_catalog_adapter_refuses_before_source_or_worker_creation() {
    use zerogravity::{models::ModelLoadRequest, runtime::TextConfig};
    let runtime = Runtime::builder().build().unwrap();
    let error = runtime
        .models()
        .load(ModelLoadRequest::CpuText {
            declared: vec![("never-opened".into(), "/fabric-source-not-opened".into())],
            config: TextConfig::default(),
        })
        .unwrap_err();
    assert_eq!(error.code, "backend_not_compiled");
    assert_eq!(runtime.status().loaded_models, 0);
    assert_eq!(runtime.status().execution_adapters, 0);
    runtime.shutdown().unwrap();
}

#[cfg(all(feature = "qwen35-cuda-serving", target_os = "linux"))]
#[test]
#[ignore = "explicit local GGUF and physical CUDA device"]
fn qwen_gguf_cuda_sdk_unloads_with_retained_handle() {
    use std::{path::PathBuf, time::Duration};
    use zerogravity::{
        builtins::text::GpuProfile,
        devices::{GpuBackend, normalize_pci},
        models::ModelLoadRequest,
        resource_arbiter::{PhysicalDeviceKey, ResourceArbiter},
        runtime::text::{Request, ResponseEvent},
    };
    let path = PathBuf::from(std::env::var_os("ZG_QWEN_GGUF").expect("approved GGUF"));
    let weight_bytes = std::fs::metadata(&path).unwrap().len();
    let physical = zerogravity::gpu::observe_cuda_devices()
        .unwrap()
        .into_iter()
        .find(|d| d.ordinal == 0)
        .unwrap();
    let device =
        PhysicalDeviceKey::new("pci", &normalize_pci(&physical.pci_bus_id).unwrap()).unwrap();
    let host = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu").unwrap();
    struct OtherSource(PathBuf);
    impl Drop for OtherSource {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    let parent = path.parent().unwrap();
    let other_source = OtherSource(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tmp")
            .join(format!("_tmp_mixed_gguf_{}", std::process::id())),
    );
    std::fs::create_dir(&other_source.0).unwrap();
    let other_path = other_source.0.join("model.gguf");
    for (original, target) in [
        (path.clone(), other_path.clone()),
        (
            parent.join("config.json"),
            other_source.0.join("config.json"),
        ),
        (
            parent.join("tokenizer.json"),
            other_source.0.join("tokenizer.json"),
        ),
    ] {
        std::fs::hard_link(original, target).unwrap();
    }
    let arbiter = ResourceArbiter::process_shared();
    let runtime = Runtime::builder()
        .limits(zerogravity::runtime::SchedulerLimits {
            max_active: 2,
            ..Default::default()
        })
        .build()
        .unwrap();
    runtime
        .models()
        .load(ModelLoadRequest::GpuText {
            declared: vec![
                ("source".into(), path.clone()),
                ("alias".into(), path),
                ("other".into(), other_path),
            ],
            backend: GpuBackend::Cuda,
            profile: GpuProfile {
                workers: 1,
                active: 2,
                context: 512,
                output: 3,
                host_mib: 8192,
                provider_mib: 2048,
                device_mib: 5120,
                cohort_width: 0,
                max_pending: Some(4),
            },
        })
        .unwrap();
    let handle = runtime.models().resolve("alias").unwrap();
    assert_eq!(
        runtime.models().aliases(&handle).unwrap(),
        ["alias", "source"]
    );
    let other = runtime.models().resolve("other").unwrap();
    assert!(!handle.shares_source(&other));
    use zerogravity::{
        models::{ModelLifecycle, ModelMemory},
        provider::ProviderKind,
    };
    let loaded = runtime.snapshot().unwrap();
    assert_eq!(
        (loaded.status.loaded_models, loaded.status.loaded_owners),
        (3, 2)
    );
    assert_eq!(loaded.catalog.len(), 2);
    for model in &loaded.catalog {
        assert_eq!(model.lifecycle, ModelLifecycle::Ready);
        assert!(model.independent_retirement);
        let details = &model.capabilities.as_ref().unwrap().details;
        assert_eq!(
            details.execution.as_ref().unwrap().selected.provider,
            ProviderKind::Cuda
        );
        assert!(details.source_host_reserved_bytes.unwrap() >= weight_bytes);
        let Some(ModelMemory::Resident(memory)) = &details.memory else {
            panic!("original GGUF model memory absent");
        };
        assert_eq!(memory.device, device);
        assert_eq!(memory.model_capacity_bytes, 2048 * 1024 * 1024);
        assert!(
            memory.resident_bytes > 0
                && memory.resident_bytes as u64 <= memory.model_capacity_bytes
        );
    }
    assert!(
        loaded
            .resources
            .iter()
            .any(|r| r.key == device && r.budget.reserved == 4096 * 1024 * 1024)
    );
    let tokenizer = handle.prepared().tokenizer();
    let prompt = " a";
    let ids = tokenizer.prompt_tokens(prompt, false).unwrap();
    let submit = |alias: &str| {
        runtime
            .request_text(Request {
                model: alias.into(),
                input: prompt.into(),
                chat: false,
                max_tokens: 1,
                stream: false,
                token_metrics: None,
            })
            .unwrap()
    };
    let complete = |events: zerogravity::runtime::text::RequestEvents| match events
        .events
        .recv_timeout(Duration::from_secs(120))
        .unwrap()
    {
        ResponseEvent::Done {
            output,
            prompt_tokens,
            completion_tokens,
            ..
        } => {
            assert_eq!(prompt_tokens, ids.len());
            assert!(completion_tokens <= 1);
            output
        }
        ResponseEvent::Error(error) => panic!("GGUF worker: {error}"),
        _ => panic!("unexpected GGUF response"),
    };
    let expected = complete(submit("source"));
    assert_eq!(complete(submit("alias")), expected);
    assert_eq!(complete(submit("other")), expected);
    let plan =
        language_pipeline::compare(&runtime, "qwen3.5-0.8b", &["alias", "other"], "1, 2, 3,", 3);
    assert_ne!(plan.stages()[0].source(), plan.stages()[1].source());
    let before = arbiter.snapshot(&host).unwrap().reserved;
    let before_device = arbiter.snapshot(&device).unwrap();
    assert_eq!(before_device.reserved, 4096 * 1024 * 1024);
    let accepted_target = submit("alias");
    let accepted_other = submit("other");
    runtime.models().unload(&handle).unwrap();
    assert_eq!(complete(accepted_target), expected);
    assert_eq!(complete(accepted_other), expected);
    assert!(!runtime.status().closed);
    assert_eq!(runtime.status().loaded_models, 1);
    assert!(
        runtime
            .run_pipeline(&plan, prompt, 1.try_into().unwrap())
            .is_err()
    );
    drop(plan);
    assert!(runtime.models().resolve("source").is_err());
    assert!(runtime.models().resolve("alias").is_err());
    assert!(
        runtime
            .models()
            .resolve("other")
            .unwrap()
            .shares_source(&other)
    );
    let observed = runtime.snapshot().unwrap();
    assert_eq!(observed.status.loaded_owners, 1);
    assert_eq!(observed.catalog[0].lifecycle, ModelLifecycle::Retired);
    assert!(observed.catalog[0].capabilities.is_none());
    assert_eq!(observed.catalog[1].source, other.source());
    assert!(
        observed
            .resources
            .iter()
            .any(|r| r.key == device && r.budget.reserved == 2048 * 1024 * 1024)
    );
    let retired = arbiter.snapshot(&device).unwrap();
    assert_eq!(retired.reserved, 2048 * 1024 * 1024);
    assert!(!retired.fenced);
    assert!(before - arbiter.snapshot(&host).unwrap().reserved >= weight_bytes);
    assert_eq!(tokenizer.prompt_tokens(prompt, false).unwrap(), ids);
    assert_eq!(complete(submit("other")), expected);
    runtime.models().unload(&other).unwrap();
    assert_eq!(arbiter.snapshot(&device).unwrap().reserved, 0);
    assert!(runtime.models().list().unwrap().is_empty());
    drop((handle, tokenizer, other));
    runtime.shutdown().unwrap();
    println!("GGUF_CUDA_SELECTIVE_CATALOG_UNLOAD_PASS");
}
