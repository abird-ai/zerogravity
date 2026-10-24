//! Finite original-HF capability inside the existing fair worker/presentation.
use super::*;
use zerogravity::canonical47 as source;
use zerogravity::gpu::vllm_cuda_qualification::Canonical47CallReceipt;
use zerogravity::{
    cache::TransferAdmission,
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner},
};

pub(super) fn validate_profile(
    backend: GpuBackend,
    profile: &GpuProfile,
    host_mib: usize,
) -> Result<source::Canonical47LifecycleProfile, ServiceError> {
    if backend != GpuBackend::Cuda
        || profile.workers != 1
        || profile.context != 512
        || !(1..=8).contains(&profile.output)
        || profile.native_rows != 0
        || profile.device_mib != source::DEVICE_MIB
    {
        return Err(ServiceError::new(
            400,
            "unsupported_canonical47_profile",
            "canonical47 requires CUDA,1worker,context512,output1..8,interleaved,device5120MiB and exact finite owner profile",
        ));
    }
    match (profile.provider_mib, host_mib, profile.active) {
        (source::PROVIDER_MIB, source::HOST_MIB..=16384, 1..=2) => {
            Ok(source::Canonical47LifecycleProfile::FiniteTwo)
        }
        (4096, 4096, 1..=8) => Ok(source::Canonical47LifecycleProfile::BenchmarkEight),
        _ => Err(ServiceError::new(
            400,
            "unsupported_canonical47_profile",
            "use finite2 provider2304/host>=3072 or active8 exact4096/4096MiB",
        )),
    }
}
fn host(host_mib: usize) -> Result<TransferAdmission, ServiceError> {
    let arbiter = ResourceArbiter::process_shared();
    // Same configured host ledger used by the existing CPU router.
    let resource =
        PhysicalDeviceKey::new("host", "zerogravity-serve-cpu").map_err(ServiceError::inference)?;
    let bytes = (host_mib as u64) * 1024 * 1024;
    match arbiter.snapshot(&resource) {
        Ok(s) if s.limit == bytes => {}
        Ok(_) => {
            return Err(ServiceError::new(
                409,
                "host_budget_conflict",
                "existing configured host limit differs",
            ));
        }
        Err(_) => arbiter
            .register(resource.clone(), bytes)
            .map_err(ServiceError::inference)?,
    }
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("canonical47-serving", "original-source-loader")
            .map_err(ServiceError::inference)?,
    })
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
                "canonical47 requires1..16 aliases of one actual original source",
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
                    "bad/duplicate canonical47 route name",
                ));
            }
            let path = std::fs::canonicalize(path).map_err(|_| {
                ServiceError::new(400, "missing_model", "original HF directory unavailable")
            })?;
            if !path.is_dir() || directory.as_ref().is_some_and(|d| d != &path) {
                return Err(ServiceError::new(
                    400,
                    "unsupported_source_set",
                    "one immutable canonical47 HF source per loaded worker",
                ));
            }
            directory = Some(path);
            aliases.insert(alias, ());
        }
        let bundle = source::native_artifact_root(bundle).map_err(ServiceError::inference)?;
        preflight_cuda_toolkit(backend)?;
        let admitted = host(host_mib)?;
        let loaded = block_on(source::load_with_profile(
            backend.device().map_err(ServiceError::inference)?,
            directory.as_deref().expect("nonempty source set"),
            &bundle,
            &admitted,
            lifecycle,
            None,
        ))
        .map_err(ServiceError::inference)?;
        let model = Arc::new(GpuModel {
            tokenizer: TextTokenizer::Canonical(loaded.clone()),
            engine: ModelEngine::Canonical(loaded),
            group: 0,
            hidden_width: 1024,
            vocabulary_size: 248320,
            native_rows: 0,
        });
        let models = aliases.into_keys().map(|id| (id, model.clone())).collect();
        Self::start(models, profile.active, profile.output, 0)
    }
}
pub(super) fn resources(models: &BTreeMap<String, Arc<GpuModel>>) -> Option<serde_json::Value> {
    let model = models.values().find_map(|model| match &model.engine {
        ModelEngine::Canonical(source) => Some(source),
        _ => None,
    })?;
    let observed = model.resident().observation().snapshot().ok()?;
    let arbiter = ResourceArbiter::process_shared();
    let host = arbiter
        .snapshot(&PhysicalDeviceKey::new("host", "zerogravity-serve-cpu").ok()?)
        .ok()?;
    let physical = arbiter.snapshot(&observed.physical_key).ok()?;
    Some(serde_json::json!({
        "lifecycle_profile":format!("{:?}",model.resident().lifecycle_profile()),
        "loaded_session_limit":model.resident().lifecycle_profile().session_limit(),
        "provider_capacity_bytes":observed.provider_capacity_bytes,
        "resident_bytes":observed.resident_bytes,
        "pending_events":observed.pending_events,
        "usable":observed.usable,
        "host_limit_bytes":host.limit,"host_reserved_bytes":host.reserved,
        "device_limit_bytes":physical.limit,"physical_reserved_bytes":physical.reserved,
    }))
}

pub(super) fn inference_error(
    error: zerogravity::Error,
    model: &source::Canonical47Model,
    cancelled: &AtomicBool,
) -> ServiceError {
    let cancelled_message = matches!(&error,zerogravity::Error::State(text)if matches!(text.as_str(),"native fragment cancelled"|"closed original cubin qualification cancelled"|"canonical positions cancelled"));
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
        receipt: Canonical47CallReceipt,
        stats: &WaveStats,
    ) -> Result<(), ServiceError> {
        let RequestEngine::Canonical {
            session, selected, ..
        } = &mut self.engine
        else {
            return Err(ServiceError::new(
                500,
                "source_route",
                "canonical result on legacy route",
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
        let RequestEngine::Canonical { model, session, .. } = &mut self.engine else {
            return Err(ServiceError::new(
                500,
                "source_route",
                "canonical step on legacy route",
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
    let mut groups = std::collections::BTreeSet::new();
    for model in models.values() {
        if groups.insert(model.group)
            && let ModelEngine::Canonical(source) = &model.engine
            && let Err(error) = block_on(source.resident().clone().close(None))
        {
            stats.failed.fetch_add(1, Ordering::Relaxed);
            eprintln!("canonical47 native retirement failed: {error}");
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
        }
    }
    #[test]
    fn router_passes_exact_lifecycle_to_real_source_loader() {
        assert_eq!(
            validate_profile(GpuBackend::Cuda, &profile(2, 2304), 3072).unwrap(),
            source::Canonical47LifecycleProfile::FiniteTwo
        );
        for active in 1..=8 {
            assert_eq!(
                validate_profile(GpuBackend::Cuda, &profile(active, 4096), 4096).unwrap(),
                source::Canonical47LifecycleProfile::BenchmarkEight
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
    fn worker_queue_has_real_sixteen_item_backpressure() {
        assert_eq!(QUEUED_PER_WORKER, 16);
        assert_eq!(MAX_ACTIVE_PER_WORKER, 8);
        let (tx, rx) = mpsc::sync_channel(QUEUED_PER_WORKER);
        for n in 0..16 {
            tx.try_send(n).unwrap();
        }
        assert!(matches!(tx.try_send(16), Err(mpsc::TrySendError::Full(16))));
        assert_eq!(rx.try_recv().unwrap(), 0);
        tx.try_send(16).unwrap();
    }
}
