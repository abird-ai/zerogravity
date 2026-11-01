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
