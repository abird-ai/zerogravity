use super::captured::*;
use super::*;
use crate::qwen35_cuda::fixed_prompt as source;
use crate::resource_arbiter::ResourceArbiter;
pub(super) fn validate_profile(
    backend: GpuBackend,
    profile: &GpuProfile,
    host_mib: usize,
) -> Result<source::FixedPromptLifecycleProfile, ServiceError> {
    if backend != GpuBackend::Cuda
        || profile.workers != 1
        || profile.context != 512
        || !(1..=8).contains(&profile.output)
        || profile.native_rows != 0
        || profile.device_mib != source::DEVICE_MIB
    {
        return Err(ServiceError::new(
            400,
            "unsupported_fixed-prompt_profile",
            "fixed-prompt requires CUDA,1worker,context512,output1..8,interleaved,device5120MiB and exact finite owner profile",
        ));
    }
    match (profile.provider_mib, host_mib, profile.active) {
        (source::PROVIDER_MIB, source::HOST_MIB..=16384, 1..=2) => {
            Ok(source::FixedPromptLifecycleProfile::FiniteTwo)
        }
        (4096, 4096, 1..=8) => Ok(source::FixedPromptLifecycleProfile::BenchmarkEight),
        _ => Err(ServiceError::new(
            400,
            "unsupported_fixed-prompt_profile",
            "use finite2 provider2304/host>=3072 or active8 exact4096/4096MiB",
        )),
    }
}

pub fn load_fixed_prompt(
    declared: Vec<(String, PathBuf)>,
    backend: GpuBackend,
    profile: GpuProfile,
    host_mib: usize,
    bundle: Option<&Path>,
) -> Result<Router, ServiceError> {
    let lifecycle = validate_profile(backend, &profile, host_mib)?;
    if declared.is_empty() || declared.len() > 16 {
        return Err(ServiceError::new(
            400,
            "invalid_models",
            "fixed-prompt requires1..16 aliases of one actual original source",
        ));
    }
    let mut aliases = BTreeMap::new();
    let mut directory = None;
    for (alias, path) in declared {
        if alias.is_empty()
            || alias.len() > 96
            || !alias
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
            || aliases.contains_key(&alias)
        {
            return Err(ServiceError::new(
                400,
                "invalid_model",
                "bad/duplicate fixed-prompt route name",
            ));
        }
        let path = std::fs::canonicalize(path).map_err(|_| {
            ServiceError::new(400, "missing_model", "original HF directory unavailable")
        })?;
        if !path.is_dir() || directory.as_ref().is_some_and(|d| d != &path) {
            return Err(ServiceError::new(
                400,
                "unsupported_source_set",
                "one immutable fixed-prompt HF source per loaded worker",
            ));
        }
        directory = Some(path);
        aliases.insert(alias, ());
    }
    let bundle = source::native_artifact_root(bundle).map_err(ServiceError::inference)?;
    preflight_cuda_toolkit(backend)?;
    // Borrow the already configured serving host ledger instead of
    // registering a second, ad-hoc host domain for the source loader. The
    // loader retains R + G + V for its actual source tokenizer.
    let serving_host = ServingHost::new(host_mib)?;
    let admitted =
        serving_host.transfer_admission("qwen35-cuda-serving", "original-source-loader")?;
    let loaded = block_on(source::load_with_profile(
        backend.device().map_err(ServiceError::inference)?,
        directory.as_deref().expect("nonempty source set"),
        &bundle,
        &admitted,
        lifecycle,
        None,
    ))
    .map_err(ServiceError::inference)?;
    let max_token_bytes = loaded
        .tokenizer()
        .max_token_bytes()
        .map_err(ServiceError::inference)?;
    let metadata = metadata_charge(&serving_host, "captured-model-adapter")?;
    let model = Arc::new(CapturedModel {
        tokenizer: crate::builtins::text::tokenizer_owner(
            loaded.clone(),
            |owner| owner.tokenizer(),
            &serving_host,
            "qwen35-cuda-serving",
        )?,
        engine: ModelEngine::FixedPrompt(loaded),
        group: 0,
        hidden_width: 1024,
        vocabulary_size: 248320,
        native_rows: 0,
        max_token_bytes,
        _metadata: metadata,
        readers: Arc::new(AtomicUsize::new(0)),
    });
    let models = aliases.into_keys().map(|id| (id, model.clone())).collect();
    start(
        models,
        profile.active,
        profile.output,
        0,
        serving_host,
        profile.max_pending,
    )
}

pub(super) fn resources(
    models: &BTreeMap<String, Arc<CapturedModel>>,
    host_snapshot: (u64, u64, u64, u64),
) -> Option<serde_json::Value> {
    let model = models.values().find_map(|model| match &model.engine {
        ModelEngine::FixedPrompt(source) => Some(source),
        _ => None,
    })?;
    let observed = model.resident().observation().snapshot().ok()?;
    let arbiter = ResourceArbiter::process_shared();
    let physical = arbiter.snapshot(&observed.physical_key).ok()?;
    let (host_limit, host_reserved) = (host_snapshot.0, host_snapshot.1);
    let serving_host_bytes = host_snapshot.3;
    Some(serde_json::json!({
        "lifecycle_profile":format!("{:?}",model.resident().lifecycle_profile()),
        "loaded_session_limit":model.resident().lifecycle_profile().session_limit(),
        "provider_capacity_bytes":observed.provider_capacity_bytes,
        "resident_bytes":observed.resident_bytes,
        "pending_events":observed.pending_events,
        "usable":observed.usable,
        "host_limit_bytes":host_limit,"host_reserved_bytes":host_reserved,
        "total_host_limit_bytes":host_limit,"total_host_reserved_bytes":host_reserved,
        "serving_host_reserved_bytes":serving_host_bytes,
        "backend_host_reserved_bytes":0,
        "model_host_reserved_bytes":host_reserved.checked_sub(serving_host_bytes),
        "device_limit_bytes":physical.limit,"physical_reserved_bytes":physical.reserved,
    }))
}

pub(super) fn inference_error(
    error: crate::Error,
    model: &source::Model,
    cancelled: &AtomicBool,
) -> ServiceError {
    let cancelled_message = matches!(&error,crate::Error::State(text)if matches!(text.as_str(),"native fragment cancelled"|"closed original cubin qualification cancelled"|"fixed_prompt positions cancelled"));
    if cancelled.load(Ordering::Acquire)
        && cancelled_message
        && model
            .resident()
            .observation()
            .snapshot()
            .is_ok_and(|s| s.usable)
    {
        ServiceError::new(499, "cancelled", "request cancelled")
    } else {
        ServiceError::inference(error)
    }
}
#[cfg(test)]
mod active8_tests {
    use super::*;
    use crate::builtins::text::tokenizer_tests::CHANNEL_TESTS as HOST_TEST_LOCK;
    use zerogravity_engine::router::{
        MAX_ACTIVE_PER_WORKER,
        pending::{PendingQueue, Pop},
    };
    fn profile(active: usize, provider_mib: usize) -> GpuProfile {
        GpuProfile {
            workers: 1,
            active,
            context: 512,
            output: 8,
            provider_mib,
            device_mib: 5120,
            native_rows: 0,
            host_mib: 4096,
            max_pending: None,
        }
    }
    #[test]
    fn router_passes_exact_lifecycle_to_real_source_loader() {
        assert_eq!(
            validate_profile(GpuBackend::Cuda, &profile(2, 2304), 3072).unwrap(),
            source::FixedPromptLifecycleProfile::FiniteTwo
        );
        for active in 1..=8 {
            assert_eq!(
                validate_profile(GpuBackend::Cuda, &profile(active, 4096), 4096).unwrap(),
                source::FixedPromptLifecycleProfile::BenchmarkEight
            );
        }
        for (active, provider, host) in [
            (3, 2304, 4096),
            (8, 2304, 4096),
            (8, 4096, 3072),
            (9, 4096, 4096),
        ] {
            assert!(validate_profile(GpuBackend::Cuda, &profile(active, provider), host).is_err());
        }
        let mut wrong = profile(8, 4096);
        wrong.native_rows = 8;
        assert!(validate_profile(GpuBackend::Cuda, &wrong, 4096).is_err());
        wrong.native_rows = 0;
        wrong.device_mib = 4096;
        assert!(validate_profile(GpuBackend::Cuda, &wrong, 4096).is_err());
    }
    #[test]
    fn shared_pending_queue_is_not_compiled_at_sixteen() {
        let _host = HOST_TEST_LOCK.lock().unwrap();
        assert_eq!(MAX_ACTIVE_PER_WORKER, 8);
        let queue = PendingQueue::new(ServingHost::new(256).unwrap(), None).unwrap();
        let _worker = queue.worker().unwrap();
        for n in 0..17 {
            queue.enqueue(queue.reserve().unwrap(), n).unwrap();
        }
        assert_eq!(queue.snapshot().pending, 17);
        assert!(matches!(queue.try_pop(|_| false), Pop::Item(0)));
    }
}
