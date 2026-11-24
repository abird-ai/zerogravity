use super::*;
fn check_cuda_toolkit(path: &Path) -> Result<(), ServiceError> {
    for header in ["include/cuda_runtime.h", "include/cuda_fp16.h"] {
        if !path.join(header).is_file() {
            return Err(ServiceError::new(
                503,
                "cuda_toolkit_missing",
                format!("CUDA_PATH={} does not contain {header}", path.display()),
            ));
        }
    }
    if ![
        path.join("include/cccl/cuda/std"),
        path.join("include/cuda/std"),
    ]
    .iter()
    .any(|p| p.is_dir())
    {
        return Err(ServiceError::new(
            503,
            "cuda_cccl_missing",
            format!(
                "CUDA_PATH={} has no CUDA CCCL device headers",
                path.display()
            ),
        ));
    }
    Ok(())
}
pub(crate) fn preflight_cuda_toolkit(backend: GpuBackend) -> Result<(), ServiceError> {
    if backend != GpuBackend::Cuda {
        return Ok(());
    }
    let installed = std::env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            ["/usr/local/cuda", "/opt/cuda"]
                .into_iter()
                .map(PathBuf::from)
                .find(|p| p.is_dir())
        })
        .or_else(|| {
            Path::new("/usr/bin/nvcc")
                .exists()
                .then(|| PathBuf::from("/usr"))
        })
        .ok_or_else(|| {
            ServiceError::new(
                503,
                "cuda_toolkit_missing",
                "CUDA NVRTC requires CUDA_PATH pointing to toolkit include headers",
            )
        })?;
    check_cuda_toolkit(&installed)
}

pub fn validate_profile(
    workers: usize,
    active: usize,
    max_context: usize,
    max_output: usize,
    provider_mib: usize,
    device_mib: usize,
) -> Result<(), ServiceError> {
    if workers != 1
        || !(1..=MAX_ACTIVE_PER_WORKER).contains(&active)
        || max_context != GPU_CAPACITY
        || !(1..=GPU_CAPACITY).contains(&max_output)
        || !(256..=65536).contains(&provider_mib)
        || !(256..=65536).contains(&device_mib)
        || provider_mib > device_mib
    {
        return Err(ServiceError::new(
            400,
            "invalid_gpu_profile",
            "GPU serve requires workers=1, context=512, active=1..8, \
                 output=1..512, provider/device 256..65536 MiB and provider<=device",
        ));
    }
    Ok(())
}
pub fn load_gpu_router(
    declared: Vec<(String, PathBuf)>,
    backend: GpuBackend,
    profile: GpuProfile,
) -> Result<Router, ServiceError> {
    let GpuProfile {
        workers,
        active,
        context,
        output,
        provider_mib,
        device_mib,
        cohort_width,
        host_mib,
        max_pending,
    } = profile;
    if !(256..=1_048_576).contains(&host_mib) {
        return Err(ServiceError::new(
            400,
            "invalid_host_budget",
            "host-mib must be 256..=1048576",
        ));
    }
    if max_pending == Some(0) {
        return Err(ServiceError::new(
            400,
            "invalid_config",
            "max-pending-requests must be a positive count or none",
        ));
    }
    validate_profile(workers, active, context, output, provider_mib, device_mib)?;
    if cohort_width != 0
        && (!matches!(cohort_width, 2 | 4 | 8)
            || backend != GpuBackend::Cuda
            || active < cohort_width)
    {
        return Err(ServiceError::new(
            400,
            "unsupported_batch_mode",
            "fused serving requires CUDA, rows=2/4/8 and active>=rows",
        ));
    }
    if !backend.compiled() {
        return Err(ServiceError::new(
            400,
            "backend_not_compiled",
            "selected GPU backend was not compiled into this library",
        ));
    }
    preflight_cuda_toolkit(backend)?;
    let declared = local::LocalRoutes::resolve(declared)?;
    // CPU fallback is forbidden: unlinked or absent provider must refuse
    // before a resident model is constructed or HTTP announces readiness.
    let provider_cap = provider_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| ServiceError::new(400, "provider_budget", "GPU provider cap overflow"))?;
    let device_cap = (device_mib as u64) * 1024 * 1024;
    // The host ledger exists before any source is parsed so the
    // process-global tokenizer bootstrap, the retained inventory and the
    // cold constructor are admitted ahead of the work they price.
    let host = ServingHost::new(host_mib)?;
    let models = declared.prepare(|alias, path, group| {
        let local::LocalModel {
            weights: source,
            tokenizer,
            max_token_bytes,
        } = local::load(alias, path, &host)?;
        let hidden_width =
            usize::try_from(source.model().geometry().hidden_size).map_err(|_| {
                ServiceError::new(400, "source_hidden", "invalid Qwen source hidden width")
            })?;
        let vocabulary_size = source.model().geometry().vocab_size;
        let source_identity = source.model().source_identity();
        let program = crate::models::ModelProgram::record(
            source,
            crate::models::ProgramRequest {
                task: crate::task::TaskKind::Language,
                context: GPU_CAPACITY,
                logits: true,
                observations: true,
            },
        )
        .map_err(ServiceError::inference)?;
        let device = backend.device().map_err(ServiceError::inference)?;
        let provider = crate::gpu::ResidentModelProvider::new(device, provider_cap, device_cap)
            .map_err(ServiceError::inference)?;
        let resident = block_on(crate::models::prepare_executable(program, &provider))
            .map_err(ServiceError::inference)?
            .into_prepared();
        let metadata = adapter_metadata::<Model>(&host, alias, "gpu-model-adapter")?;
        Ok(Arc::new(Model {
            resident: Mutex::new(Some(resident)),
            source: source_identity,
            tokenizer,
            group,
            hidden_width,
            vocabulary_size,
            cohort_width,
            max_token_bytes,
            readers: Arc::new(AtomicUsize::new(0)),
            _metadata: metadata,
        }))
    })?;
    start(models, active, output, cohort_width, host, max_pending)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_backend_refuses_before_tokenizer_source_or_device_lookup() {
        let Some(backend) = [GpuBackend::Cuda, GpuBackend::Hip, GpuBackend::VulkanAmd]
            .into_iter()
            .find(|b| !b.compiled())
        else {
            return;
        };
        let profile = GpuProfile {
            workers: 1,
            active: 1,
            context: 512,
            output: 1,
            provider_mib: 256,
            device_mib: 256,
            cohort_width: 0,
            host_mib: 256,
            max_pending: Some(1),
        };
        let error = load_gpu_router(
            vec![("not-opened".into(), PathBuf::from("/source-not-opened"))],
            backend,
            profile,
        )
        .err()
        .expect("absent backend must refuse");
        assert_eq!(error.code, "backend_not_compiled");
    }

    #[test]
    fn final_source_reader_retains_original_counter_and_its_admission() {
        let _serial = crate::builtins::text::tokenizer_tests::CHANNEL_TESTS
            .lock()
            .unwrap();
        let host = ServingHost::new(256).unwrap();
        let baseline = host.snapshot().unwrap();
        let metadata = adapter_metadata::<Model>(&host, "reader-test", "adapter").unwrap();
        let readers = Arc::new(AtomicUsize::new(0));
        let weak_counter = Arc::downgrade(&readers);
        let weak_charge = Arc::downgrade(&metadata);
        let last = SourceReaders::enter(&readers, &metadata);
        let reserved = host.snapshot().unwrap();
        drop((readers, metadata));
        assert_eq!(weak_counter.upgrade().unwrap().load(Ordering::Acquire), 1);
        assert!(weak_charge.upgrade().is_some());
        assert_eq!(host.snapshot().unwrap(), reserved);
        drop(last);
        assert!(weak_counter.upgrade().is_none());
        assert!(weak_charge.upgrade().is_none());
        assert_eq!(host.snapshot().unwrap(), baseline);
    }

    #[test]
    fn cuda_requires_runtime_and_cccl_headers_before_device_startup() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "zerogravity-cuda-prerequisite-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("include")).unwrap();
        assert_eq!(
            check_cuda_toolkit(&root).unwrap_err().code,
            "cuda_toolkit_missing"
        );
        std::fs::write(root.join("include/cuda_runtime.h"), b"").unwrap();
        std::fs::write(root.join("include/cuda_fp16.h"), b"").unwrap();
        assert_eq!(
            check_cuda_toolkit(&root).unwrap_err().code,
            "cuda_cccl_missing"
        );
        std::fs::create_dir_all(root.join("include/cccl/cuda/std")).unwrap();
        assert!(check_cuda_toolkit(&root).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn gpu_profile_is_bounded_before_any_device_lookup() {
        assert!(validate_profile(1, 2, 512, 16, 2048, 4096).is_ok());
        for bad in [
            (2, 2, 512, 16, 2048, 4096),
            (1, 0, 512, 16, 2048, 4096),
            (1, 9, 512, 16, 2048, 4096),
            (1, 2, 8192, 16, 2048, 4096),
            (1, 2, 512, 513, 2048, 4096),
            (1, 2, 512, 16, 4096, 2048),
            (1, 2, 512, 16, 0, 4096),
        ] {
            assert!(validate_profile(bad.0, bad.1, bad.2, bad.3, bad.4, bad.5).is_err());
        }
    }
}
