//! External consumer uses the public facade alone.
use zerogravity::{
    Error, Runtime,
    run::{TaskInput, TaskRequest},
};
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
#[ignore = "approved original CUDA cache; physical public SDK contiguous-vs-paged native cohort"]
fn qwen_cuda_paged_native_sdk_matches_contiguous() {
    use std::{
        collections::BTreeSet,
        time::{Duration, Instant},
    };
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenCudaStatePolicy, QwenSourceProfile},
        runtime::{
            SchedulerLimits,
            text::{Request, ResponseEvent, TokenMetricsRequest},
        },
    };

    const MIB: usize = 1024 * 1024;
    const SNAPSHOT: &str = "d7c7bf56e176bd70a8190acb9873cc719e65b6b306181039f2896d6f9387b36c";

    let run = |paged: bool| {
        let runtime = Runtime::builder()
            .limits(SchedulerLimits {
                max_active: 8,
                max_queued: Some(16),
                ..Default::default()
            })
            .build()
            .unwrap();
        let profile = QwenSourceProfile {
            active: 8,
            context: 512,
            output: 2,
            max_pending: Some(16),
            native_rows: 8,
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
                    max_append_tokens: 4,
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
        let model = runtime.models().resolve("source").unwrap();
        let tokenizer = model.prepared().tokenizer();
        let prompt = " a".repeat(4);
        assert_eq!(tokenizer.prompt_tokens(&prompt, false).unwrap().len(), 4);

        let events = (0..8)
            .map(|_| {
                runtime
                    .request_text(Request {
                        model: "source".into(),
                        input: prompt.clone(),
                        chat: false,
                        max_tokens: 2,
                        stream: true,
                        token_metrics: Some(TokenMetricsRequest {
                            started: Instant::now(),
                            wire_request_id: None,
                        }),
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(events.len());
        let mut cohort_widths = BTreeSet::new();
        for events in events {
            let mut ids = Vec::new();
            loop {
                match events
                    .events
                    .recv_timeout(Duration::from_secs(300))
                    .expect("paged SDK response deadline")
                {
                    ResponseEvent::TokenText { token, .. } => {
                        if let Some(cohort) = token.gpu.and_then(|gpu| gpu.native_cohort) {
                            cohort_widths.insert(cohort.width);
                        }
                        ids.push(token.id);
                    }
                    ResponseEvent::Done {
                        prompt_tokens,
                        completion_tokens,
                        ..
                    } => {
                        assert_eq!((prompt_tokens, completion_tokens), (4, 2));
                        break;
                    }
                    ResponseEvent::Error(error) => panic!("paged SDK worker: {error}"),
                    ResponseEvent::Text(_) => panic!("streaming token observations required"),
                }
            }
            assert_eq!(ids.len(), 2);
            outputs.push(ids);
        }
        assert!(
            cohort_widths.iter().any(|&width| width > 1),
            "native cohort publication was not observed"
        );
        runtime.shutdown().unwrap();
        assert!(runtime.status().closed);
        (outputs, cohort_widths)
    };

    let contiguous = run(false);
    let paged = run(true);
    assert_eq!(paged.0, contiguous.0);
    println!(
        "P12_SOURCE_PAGED_SDK_PASS requests=8 prompt_tokens=4 output_tokens=2 contiguous_cohorts={:?} paged_cohorts={:?} tokens=exact final_shutdown=true",
        contiguous.1, paged.1
    );
}

#[cfg(all(feature = "qwen35-hip", target_os = "linux"))]
#[test]
fn canonical_hip_registration_checks_runtime_limits_and_aliases_before_source_io() {
    use zerogravity::{
        cache::CacheOptions,
        models::{ModelLoadRequest, QwenHipProfile},
    };
    let runtime = Runtime::builder().build().unwrap();
    let request = |active, aliases| ModelLoadRequest::QwenHip {
        aliases,
        cache: CacheOptions::new(Some("/source-not-opened".into()), 96 * 1024 * 1024 * 1024),
        snapshot: "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52".into(),
        profile: QwenHipProfile {
            active,
            context: 514,
            output: 2,
            max_pending: Some(1),
            native_rows: 0,
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
        native_rows: 0,
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
