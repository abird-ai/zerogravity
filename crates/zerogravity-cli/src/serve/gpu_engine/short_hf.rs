//! Genuine source-owned short-HF sessions on the original fair worker.
use super::super::engine::GpuRootObservation;
use super::*;
mod cohort;
pub(super) use cohort::advance_cohort;
use zerogravity::gpu::vllm_cuda_qualification::ShortHfCallReceipt;
use zerogravity::resource_arbiter::ResourceArbiter;
use zerogravity::short_hf as source;

fn validate_profile(
    backend: GpuBackend,
    p: &GpuProfile,
    host_mib: usize,
    bundle: Option<&Path>,
) -> Result<(), ServiceError> {
    if backend != GpuBackend::Cuda
        || p.workers != 1
        || p.context != 512
        || !(1..=8).contains(&p.output)
        || !(1..=8).contains(&p.active)
        || !matches!(p.native_rows, 0 | 4)
        || (p.native_rows != 0 && p.active < p.native_rows)
        || p.provider_mib == 0
        || p.provider_mib > p.device_mib
        || p.device_mib != 5120
        || p.host_mib != host_mib
        || !super::super::short_hf_host_profile(host_mib)
        || bundle.is_none()
    {
        return Err(ServiceError::new(
            400,
            "unsupported_short_hf_profile",
            "short-HF requires CUDA/1worker/context512/output1..8/active1..8/native0or4/4096or8192host/provider1..5120/5120device/native-artifacts",
        ));
    }
    Ok(())
}
impl GpuRouter {
    pub(in crate::serve) fn new_short_hf(
        declared: Vec<(String, PathBuf)>,
        backend: GpuBackend,
        profile: GpuProfile,
        host_mib: usize,
        bundle: Option<&Path>,
    ) -> Result<Self, ServiceError> {
        validate_profile(backend, &profile, host_mib, bundle)?;
        crate::run_cli::short_hf_host_available(
            (host_mib * 1024 * 1024) as u64,
            &zerogravity::run_policy::Headroom::Default,
        )
        .map_err(ServiceError::inference)?;
        if declared.is_empty() || declared.len() > 16 {
            return Err(ServiceError::new(
                400,
                "invalid_models",
                "one original HF source and1..16aliases",
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
                    "bad/duplicate short-HF route name",
                ));
            }
            let path = std::fs::canonicalize(path).map_err(|_| {
                ServiceError::new(400, "missing_model", "original HF directory unavailable")
            })?;
            if !path.is_dir() || directory.as_ref().is_some_and(|d| d != &path) {
                return Err(ServiceError::new(
                    400,
                    "unsupported_source_set",
                    "one immutable HF source per worker",
                ));
            }
            directory = Some(path);
            aliases.insert(alias, ());
        }
        let bundle = bundle.ok_or_else(|| {
            ServiceError::new(
                400,
                "missing_native_artifacts",
                "actual source native bundle required",
            )
        })?;
        preflight_cuda_toolkit(backend)?;
        // Borrow the configured serving host ledger; do not register another
        // host domain for the original source loader. The loader retains
        // R + G + V for its actual source tokenizer.
        let serving_host = ServingHost::new(host_mib)?;
        let admitted =
            serving_host.transfer_admission("short-hf-serving", "original-source-loader")?;
        let device = backend.device().map_err(ServiceError::inference)?;
        let directory = directory
            .as_deref()
            .ok_or_else(|| ServiceError::new(400, "missing_model", "no actual source"))?;
        let capacity = profile
            .provider_mib
            .checked_mul(1024 * 1024)
            .ok_or_else(|| {
                ServiceError::new(400, "invalid_provider_budget", "provider bytes overflow")
            })?;
        let loaded = block_on(async {
            if profile.native_rows == 4 {
                source::load_b4(device, directory, bundle, &admitted, capacity, None).await
            } else {
                source::load(device, directory, bundle, &admitted, capacity, None).await
            }
        })
        .map_err(ServiceError::inference)?;
        let meta = loaded.source_metadata();
        let model = Arc::new(GpuModel {
            hidden_width: meta.hidden_width,
            vocabulary_size: meta.vocabulary_size,
            tokenizer: TextTokenizer::ShortHf(loaded.clone()),
            engine: ModelEngine::ShortHf(loaded.clone()),
            group: 0,
            native_rows: profile.native_rows,
            max_token_bytes: loaded
                .tokenizer()
                .max_token_bytes()
                .map_err(ServiceError::inference)?,
            _inventory: None,
            readers: Arc::new(AtomicUsize::new(0)),
        });
        let models = aliases.into_keys().map(|id| (id, model.clone())).collect();
        Self::start(
            models,
            profile.active,
            profile.output,
            profile.native_rows,
            serving_host,
            profile.max_pending,
        )
    }
}

pub(super) fn models_present(models: &BTreeMap<String, Arc<GpuModel>>) -> bool {
    models
        .values()
        .any(|model| matches!(&model.engine, ModelEngine::ShortHf(_)))
}

pub(super) fn health(
    models: &BTreeMap<String, Arc<GpuModel>>,
    host_snapshot: (u64, u64, u64, u64),
) -> Option<(serde_json::Value, serde_json::Value)> {
    let model = models.values().find_map(|m| match &m.engine {
        ModelEngine::ShortHf(s) => Some(s),
        _ => None,
    })?;
    let meta = model.source_metadata();
    let observed = model.resident().observation().snapshot().ok()?;
    let arbiter = ResourceArbiter::process_shared();
    let physical = arbiter.snapshot(&observed.physical_key).ok()?;
    let (host_limit, host_reserved) = (host_snapshot.0, host_snapshot.1);
    let serving_host_bytes = host_snapshot.3;
    Some((
        serde_json::json!({
            "schema":"zerogravity.short-hf.serving-source.v1","model":meta.family,
            "repository":meta.repository,"revision":meta.revision,"index_sha256":meta.index_sha256,
            "source_manifest_sha256":meta.manifest_sha256,"prefill_rows":meta.prefill_rows,
            "cached_base_through":meta.cached_through,"native_cohort_width":model.resident().native_cohort_width(),"initial_cohort_width":model.resident().initial_cohort_width(),"max_output":meta.max_output,"context":meta.context,
        }),
        serde_json::json!({
            "provider_capacity_bytes":observed.provider_capacity_bytes,"resident_bytes":observed.resident_bytes,
            "pending_events":observed.pending_events,"usable":observed.usable,
            "host_limit_bytes":host_limit,"host_reserved_bytes":host_reserved,
            "total_host_limit_bytes":host_limit,"total_host_reserved_bytes":host_reserved,
            "serving_host_reserved_bytes":serving_host_bytes,
            "device_limit_bytes":physical.limit,"physical_reserved_bytes":physical.reserved,
            "backend_host_reserved_bytes":observed.backend_host_reserved_bytes,
            "backend_device_reserved_bytes":observed.backend_device_reserved_bytes,
            "model_host_reserved_bytes":host_reserved
                .checked_sub(observed.backend_host_reserved_bytes)
                .and_then(|bytes| bytes.checked_sub(serving_host_bytes)),
            "model_device_reserved_bytes":physical.reserved.checked_sub(observed.backend_device_reserved_bytes),
            "readiness_candidate_requests":model.resident().readiness_candidates(),
            "startup_readiness_compiler_attempts":model.resident().readiness_usage().attempts,
            "startup_readiness_key_source_charged_bytes":model.resident().readiness_usage().key_source_bytes,
            "startup_readiness_ptx_charged_bytes":model.resident().readiness_usage().ptx_bytes,
            "compiler_cumulative_attempts":observed.compiler_cumulative_usage.attempts,
            "compiler_cumulative_key_source_charged_bytes":observed.compiler_cumulative_usage.key_source_bytes,
            "compiler_cumulative_ptx_charged_bytes":observed.compiler_cumulative_usage.ptx_bytes,
            "readiness_n1_prompt_rows":model.resident().readiness_n1_prompt_rows(),"readiness_n1_cached_rows":model.resident().readiness_n1_cached_rows(),
            "readiness_b4":model.resident().native_cohort_width()==4,
            "readiness_finite_grouping_complete":model.resident().readiness_finite_grouping_complete(),
        }),
    ))
}

pub(super) fn inference_error(
    error: zerogravity::Error,
    model: &source::ShortHfModel,
    cancelled: &AtomicBool,
) -> ServiceError {
    let cancelled_message = matches!(&error,zerogravity::Error::State(text)if matches!(text.as_str(),
        "native fragment cancelled"|"closed original cubin qualification cancelled"|"canonical positions cancelled"|"short HF request cancelled"));
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
    fn install_short_hf(
        &mut self,
        r: ShortHfCallReceipt,
        stats: &WaveStats,
    ) -> Result<(), ServiceError> {
        let consumed = if self.cursor == 0 {
            self.prompt.len()
        } else {
            1
        };
        let RequestEngine::ShortHf {
            session, selected, ..
        } = &mut self.engine
        else {
            return Err(ServiceError::new(
                500,
                "source_route",
                "short-HF result on another route",
            ));
        };
        if r.root != session.binding()
            || r.executed_nodes != 255
            || r.native_launches == 0
            || r.consumed_tokens != consumed
        {
            return Err(ServiceError::new(
                500,
                "short_hf_receipt",
                "incomplete actual source traversal/commit",
            ));
        }
        *selected = Some(r.selected_id);
        stats.gpu_forwards.fetch_add(1, Ordering::Relaxed);
        stats
            .kernel_dispatches
            .fetch_add(r.dispatches as u64, Ordering::Relaxed);
        stats
            .model_tokens
            .fetch_add(r.consumed_tokens as u64, Ordering::Relaxed);
        stats.committed_roots.fetch_add(1, Ordering::Relaxed);
        self.token_gpu_observation = Some(GpuTokenObservation {
            producer_frontier: r.root.frontier(),
            producer_root: Some(GpuRootObservation::committed(r.root)),
            native_cohort: None,
        });
        self.token_commit_observed_at = Some(Instant::now());
        self.packet_presented = false;
        self.blocked_since = None;
        Ok(())
    }
    pub(super) fn step_short_hf(&mut self, stats: &WaveStats) -> Result<MemberStep, ServiceError> {
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
        let RequestEngine::ShortHf { model, session, .. } = &mut self.engine else {
            return Err(ServiceError::new(
                500,
                "source_route",
                "short-HF step on another route",
            ));
        };
        // Await original bounded quantum. Actual cancellation prevents commit;
        // never drop/race a submitted inference future on HTTP disconnect.
        let result = if prefill {
            block_on(session.infer_prompt(&self.prompt, Some(&self.cancelled)))
        } else {
            block_on(session.infer_next(Some(&self.cancelled)))
        };
        let receipt = result.map_err(|e| inference_error(e, model, &self.cancelled))?;
        self.install_short_hf(receipt, stats)?;
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
        .filter(|m| m.readers.load(Ordering::Acquire) != 0)
        .map(|m| m.group)
        .collect();
    let mut groups = std::collections::BTreeSet::new();
    for m in models.values() {
        if busy.contains(&m.group) {
            continue;
        }
        if groups.insert(m.group)
            && let ModelEngine::ShortHf(source) = &m.engine
            && let Err(error) = block_on(source.resident().clone().close(None))
        {
            stats.failed.fetch_add(1, Ordering::Relaxed);
            eprintln!("short-HF native retirement failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile() -> GpuProfile {
        GpuProfile {
            workers: 1,
            active: 8,
            context: 512,
            output: 8,
            provider_mib: 4096,
            device_mib: 5120,
            native_rows: 0,
            host_mib: 4096,
            max_pending: None,
        }
    }
    #[test]
    fn route_rejects_budget_fusion_and_context_before_source_device_or_native_load() {
        let mut eight = profile();
        eight.host_mib = 8192;
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &eight,
                8192,
                Some(Path::new("/actual-bundle"))
            )
            .is_ok()
        );
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &profile(),
                4096,
                Some(Path::new("/actual-bundle"))
            )
            .is_ok()
        );
        // Selected profile and configured ledger must agree before any RAM,
        // source or Client work.
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &profile(),
                8192,
                Some(Path::new("/actual-bundle"))
            )
            .is_err()
        );
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &eight,
                4096,
                Some(Path::new("/actual-bundle"))
            )
            .is_err()
        );
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &profile(),
                5120,
                Some(Path::new("/actual-bundle"))
            )
            .is_err()
        );
        let mut opted_in = profile();
        opted_in.native_rows = 4;
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &opted_in,
                4096,
                Some(Path::new("/actual-bundle"))
            )
            .is_ok()
        );
        opted_in.active = 3;
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &opted_in,
                4096,
                Some(Path::new("/actual-bundle"))
            )
            .is_err()
        );
        for backend in [
            GpuBackend::Hip,
            GpuBackend::VulkanNvidia,
            GpuBackend::VulkanAmd,
        ] {
            assert!(
                validate_profile(backend, &profile(), 4096, Some(Path::new("/actual-bundle")))
                    .is_err()
            );
        }
        for (field, value) in [
            ("active", 9),
            ("context", 511),
            ("output", 9),
            ("provider", 0),
            ("provider", 5121),
            ("device", 4096),
            ("native", 2),
            ("native", 8),
        ] {
            let mut p = profile();
            match field {
                "active" => p.active = value,
                "context" => p.context = value,
                "output" => p.output = value,
                "provider" => p.provider_mib = value,
                "device" => p.device_mib = value,
                "native" => p.native_rows = value,
                _ => unreachable!(),
            }
            assert!(
                validate_profile(
                    GpuBackend::Cuda,
                    &p,
                    4096,
                    Some(Path::new("/actual-bundle"))
                )
                .is_err()
            );
        }
        assert!(
            validate_profile(
                GpuBackend::Cuda,
                &profile(),
                3072,
                Some(Path::new("/actual-bundle"))
            )
            .is_err()
        );
        assert!(validate_profile(GpuBackend::Cuda, &profile(), 4096, None).is_err());
    }
}
