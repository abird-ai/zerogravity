//! Finite original-HF capability inside the existing fair worker/presentation.
use super::*;
use zerogravity::gpu::cuda::native::FixedPromptCallReceipt;
use zerogravity::qwen35_cuda::fixed_prompt as source;
use zerogravity::resource_arbiter::ResourceArbiter;

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
impl GpuRouter {
    pub(in crate::serve) fn new_canonical(
        declared: Vec<(String, PathBuf)>,
        backend: GpuBackend,
        profile: GpuProfile,
        host_mib: usize,
        bundle: Option<&Path>,
    ) -> Result<Self, ServiceError> {
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
        let model = Arc::new(GpuModel {
            tokenizer: TextTokenizer::FixedPrompt(loaded.clone()),
            engine: ModelEngine::FixedPrompt(loaded),
            group: 0,
            hidden_width: 1024,
            vocabulary_size: 248320,
            native_rows: 0,
            max_token_bytes,
            _inventory: None,
            readers: Arc::new(AtomicUsize::new(0)),
        });
        let models = aliases.into_keys().map(|id| (id, model.clone())).collect();
        Self::start(
            models,
            profile.active,
            profile.output,
            0,
            serving_host,
            profile.max_pending,
        )
    }
}
pub(super) fn resources(
    models: &BTreeMap<String, Arc<GpuModel>>,
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
    error: zerogravity::Error,
    model: &source::Model,
    cancelled: &AtomicBool,
) -> ServiceError {
    let cancelled_message = matches!(&error,zerogravity::Error::State(text)if matches!(text.as_str(),"native fragment cancelled"|"closed original cubin qualification cancelled"|"fixed_prompt positions cancelled"));
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
impl GpuActive {
    fn install_canonical(
        &mut self,
        receipt: FixedPromptCallReceipt,
        stats: &WaveStats,
    ) -> Result<(), ServiceError> {
        let RequestEngine::FixedPrompt {
            session, selected, ..
        } = &mut self.engine
        else {
            return Err(ServiceError::new(
                500,
                "source_route",
                "fixed_prompt result on legacy route",
            ));
        };
        if receipt.root != session.binding()
            || receipt.executed_nodes != 255
            || receipt.native_launches == 0
            || receipt.consumed_tokens != if self.cursor == 0 { 47 } else { 1 }
        {
            return Err(ServiceError::new(
                500,
                "canonical_receipt",
                "incomplete original source traversal/commit",
            ));
        }
        *selected = Some(receipt.selected_id);
        stats.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        stats
            .kernel_dispatches
            .fetch_add(receipt.dispatches as u64, Ordering::Relaxed);
        stats
            .model_tokens
            .fetch_add(receipt.consumed_tokens as u64, Ordering::Relaxed);
        stats.committed_roots.fetch_add(1, Ordering::Relaxed);
        self.token_gpu_observation = Some(GpuTokenObservation {
            producer_frontier: receipt.root.frontier(),
            producer_root: None,
            native_cohort: None,
        });
        self.token_commit_observed_at = Some(Instant::now());
        self.packet_presented = false;
        self.blocked_since = None;
        Ok(())
    }
    pub(super) fn step_canonical(&mut self, stats: &WaveStats) -> Result<MemberStep, ServiceError> {
        let prefill = self.cursor == 0;
        if !prefill {
            match self.present_decode(stats)? {
                Presentation::Finished => return Ok(MemberStep::Finished),
                Presentation::Blocked => return Ok(MemberStep::Parked),
                Presentation::Ready => {}
            }
        }
        if self.cancelled.load(Ordering::Acquire) {
            return Ok(MemberStep::Cancelled);
        }
        let RequestEngine::FixedPrompt { model, session, .. } = &mut self.engine else {
            return Err(ServiceError::new(
                500,
                "source_route",
                "fixed_prompt step on legacy route",
            ));
        };
        let result = if prefill {
            block_on(session.infer_prompt(&self.prompt, Some(&self.cancelled)))
        } else {
            block_on(session.infer_next(Some(&self.cancelled)))
        };
        let receipt = result.map_err(|e| inference_error(e, model, &self.cancelled))?;
        self.install_canonical(receipt, stats)?;
        if prefill {
            self.cursor = self.prompt.len();
        }
        Ok(MemberStep::Forwarded)
    }
}
pub(super) fn retire_sources(models: &BTreeMap<String, Arc<GpuModel>>, stats: &WaveStats) {
    // A model alias may share one resident source group. Any live reader in
    // the group keeps retirement closed, even if this particular alias has
    // already released its reader count.
    let busy: std::collections::BTreeSet<_> = models
        .values()
        .filter(|model| model.readers.load(Ordering::Acquire) != 0)
        .map(|model| model.group)
        .collect();
    let mut groups = std::collections::BTreeSet::new();
    for model in models.values() {
        if busy.contains(&model.group) {
            continue;
        }
        if groups.insert(model.group)
            && let ModelEngine::FixedPrompt(source) = &model.engine
            && let Err(error) = block_on(source.resident().clone().close(None))
        {
            stats.failed.fetch_add(1, Ordering::Relaxed);
            eprintln!("fixed-prompt native retirement failed: {error}");
        }
    }
}

#[cfg(test)]
mod active8_tests {
    use super::*;
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
        assert_eq!(MAX_ACTIVE_PER_WORKER, 8);
        let queue = PendingQueue::new(ServingHost::isolated(512).unwrap(), None).unwrap();
        let _worker = queue.worker().unwrap();
        for n in 0..17 {
            queue.enqueue(queue.reserve().unwrap(), n).unwrap();
        }
        assert_eq!(queue.snapshot().pending, 17);
        assert!(matches!(queue.try_pop(|_| false), Pop::Item(0)));
    }
}
