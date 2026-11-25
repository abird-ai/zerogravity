use super::*;
use crate::{
    accelerator::program::NumericalContract,
    cache::{CacheOptions, CacheStore},
    gpu::Device,
    qwen35_source::{self, Inputs},
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission},
    tokenize::QwenTokenizer,
};
use futures_lite::future::block_on;
use std::collections::BTreeMap;
use zerogravity_engine::{model_instance::execution::*, router::Router};

#[derive(Clone, Copy)]
pub(crate) enum SourceBackend {
    #[cfg(all(feature = "model-qwen35", feature = "hip"))]
    Hip(crate::gpu::HipSourceBudget, NumericalContract),
    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    Cuda {
        provider_bytes: usize,
        device_bytes: u64,
        smart_config: crate::models::SmartConfig<crate::models::QwenCudaSmartOverrides>,
    },
}
impl SourceBackend {
    fn validate(self, profile: QwenServingProfile) -> Result<(), ServiceError> {
        #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
        if let Self::Cuda {
            provider_bytes,
            device_bytes,
            ..
        } = self
            && (provider_bytes == 0
                || u64::try_from(provider_bytes).map_or(true, |n| n > device_bytes))
        {
            return Err(ServiceError::new(
                400,
                "invalid_source_budget",
                "CUDA requires a positive provider capacity within physical device admission",
            ));
        }
        #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
        if let Self::Cuda { smart_config, .. } = self
            && smart_config.mode != crate::smart_config::SmartConfigMode::Exclusive
        {
            if let Some(policy) = smart_config.overrides.state_policy {
                validate_cuda_state_policy(profile, policy)?;
            }
            if let Some(cohort_width) = smart_config.overrides.cohort_width {
                profile.validate_cohort_width(cohort_width)?;
            }
        }
        #[cfg(all(feature = "model-qwen35", feature = "hip"))]
        if matches!(self, Self::Hip(..)) {
            profile.validate_base()?;
            profile.validate_cohort_width(profile.cohort_width)?;
        } else {
            profile.validate_base()?;
        }
        #[cfg(not(all(feature = "model-qwen35", feature = "hip")))]
        profile.validate_base()?;
        Ok(())
    }
    fn contract(self) -> NumericalContract {
        match self {
            #[cfg(all(feature = "model-qwen35", feature = "hip"))]
            Self::Hip(_, numerical) => numerical,
            #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
            Self::Cuda { .. } => NumericalContract::CudaSm86Bf16ReferenceV1,
        }
    }
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
fn validate_cuda_state_policy(
    profile: QwenServingProfile,
    policy: crate::models::QwenCudaStatePolicy,
) -> Result<(), ServiceError> {
    if let crate::models::QwenCudaStatePolicy::Paged {
        page_tokens,
        max_append_tokens,
    } = policy
        && (page_tokens == 0
            || max_append_tokens == 0
            || page_tokens > profile.context
            || max_append_tokens > profile.context)
    {
        return Err(ServiceError::new(
            400,
            "invalid_source_state_policy",
            "paged CUDA state requires positive page/append tokens within context",
        ));
    }
    Ok(())
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
fn describe_cuda_state_policy(policy: crate::models::QwenCudaStatePolicy) -> String {
    match policy {
        crate::models::QwenCudaStatePolicy::Contiguous => "contiguous".into(),
        crate::models::QwenCudaStatePolicy::Paged {
            page_tokens,
            max_append_tokens,
        } => format!("paged(page_tokens={page_tokens},max_append_tokens={max_append_tokens})"),
    }
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
fn smart_cuda_provider_cap(
    configured_provider: usize,
    device_bytes: u64,
) -> crate::Result<(usize, Option<u128>, String)> {
    let budget =
        zerogravity_core::provider::ProviderMemoryBudget::new(configured_provider, device_bytes)?;
    let Ok(inventory) = crate::devices::discover() else {
        return Ok((
            configured_provider,
            None,
            "current CUDA observation unavailable; using configured provider envelope".into(),
        ));
    };
    if crate::devices::fresh(&inventory).is_err() {
        return Ok((
            configured_provider,
            Some(inventory.observed_unix_ms),
            "current CUDA observation stale; using configured provider envelope".into(),
        ));
    }
    let Some(device) = inventory.devices.iter().find(|device| {
        device.vendor == "nvidia" && device.cuda_ordinal == Some(0) && device.exclusion.is_none()
    }) else {
        return Ok((
            configured_provider,
            Some(inventory.observed_unix_ms),
            "CUDA ordinal0 current availability unavailable; using configured provider envelope"
                .into(),
        ));
    };
    let Some(free) = device
        .available
        .and_then(|bytes| usize::try_from(bytes).ok())
    else {
        return Ok((
            configured_provider,
            Some(inventory.observed_unix_ms),
            "CUDA ordinal0 free VRAM unavailable; using configured provider envelope".into(),
        ));
    };
    let effective = budget.fit_available(free as u64)?;
    Ok((
        effective,
        Some(inventory.observed_unix_ms),
        format!(
            "fresh CUDA ordinal0 available={free}B configured_provider={configured_provider}B device_headroom={}B",
            budget.headroom_bytes()
        ),
    ))
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
fn normalize_cuda_smart_input(
    mut profile: QwenServingProfile,
    backend: SourceBackend,
) -> (QwenServingProfile, SourceBackend) {
    match backend {
        SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            mut smart_config,
        } => {
            if smart_config.overrides.cohort_width.is_none() && profile.cohort_width != 0 {
                smart_config.overrides.cohort_width = Some(profile.cohort_width);
            }
            profile.cohort_width = 0;
            (
                profile,
                SourceBackend::Cuda {
                    provider_bytes,
                    device_bytes,
                    smart_config,
                },
            )
        }
        #[cfg(all(feature = "model-qwen35", feature = "hip"))]
        SourceBackend::Hip(..) => (profile, backend),
    }
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
#[derive(Clone)]
struct CudaSmartCandidate {
    state_policy: crate::models::QwenCudaStatePolicy,
    cohort_width: usize,
    inventory: crate::gpu::CudaForwardSourceInventory,
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
fn automatic_cohort_width(active: usize) -> &'static [usize] {
    match active {
        8.. => &[8, 4, 0],
        4..=7 => &[4, 0],
        _ => &[0],
    }
}

#[cfg(all(feature = "model-qwen35", feature = "cuda"))]
fn resolve_cuda_smart_config(
    model: &str,
    mut profile: QwenServingProfile,
    plan: std::sync::Arc<crate::accelerator::program::ForwardPlan>,
    backend: SourceBackend,
) -> Result<(SourceBackend, QwenServingProfile), ServiceError> {
    use crate::{
        gpu::{CudaForwardWeightStorage, ResidentModel},
        smart_config::{SmartConfigDecision, SmartConfigMode, SmartConfigSource},
    };

    const PAGE_TOKENS: usize = 64;
    const MAX_APPEND_TOKENS: usize = 64;
    let (provider_bytes, device_bytes, smart_config) = match backend {
        SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            smart_config,
        } => (provider_bytes, device_bytes, smart_config),
        #[cfg(all(feature = "model-qwen35", feature = "hip"))]
        SourceBackend::Hip(..) => return Ok((backend, profile)),
    };

    let mode = smart_config.mode;
    let state_user = smart_config.overrides.state_policy;
    let native_user = smart_config.overrides.cohort_width;

    if mode == SmartConfigMode::Disabled {
        let state = crate::smart_config::resolve(
            mode,
            state_user,
            crate::models::QwenCudaStatePolicy::Contiguous,
            || unreachable!("disabled smart config does not evaluate auto state policy"),
        )
        .map_err(ServiceError::inference)?;
        let native = crate::smart_config::resolve(mode, native_user, 0usize, || {
            unreachable!("disabled smart config does not evaluate auto cohort width")
        })
        .map_err(ServiceError::inference)?;
        validate_cuda_state_policy(profile, state.value)?;
        profile.validate_cohort_width(native.value)?;
        profile.cohort_width = native.value;

        let mut report = crate::smart_config::SmartConfigReport::new(model, "cuda", mode, None);
        report.push(
            "state_policy",
            describe_cuda_state_policy(state.value),
            state.source,
            state.reason,
        );
        report.push(
            "cohort_width",
            native.value.to_string(),
            native.source,
            native.reason,
        );
        report.emit();
        return Ok((
            SourceBackend::Cuda {
                provider_bytes,
                device_bytes,
                smart_config: crate::smart_config::SmartConfig::disabled(
                    crate::models::QwenCudaSmartOverrides {
                        state_policy: Some(state.value),
                        cohort_width: Some(native.value),
                    },
                ),
            },
            profile,
        ));
    }

    let state_values = if mode == SmartConfigMode::Enabled {
        state_user.map_or_else(
            || {
                vec![
                    crate::models::QwenCudaStatePolicy::Contiguous,
                    crate::models::QwenCudaStatePolicy::Paged {
                        page_tokens: PAGE_TOKENS,
                        max_append_tokens: MAX_APPEND_TOKENS,
                    },
                ]
            },
            |state| vec![state],
        )
    } else {
        vec![
            crate::models::QwenCudaStatePolicy::Contiguous,
            crate::models::QwenCudaStatePolicy::Paged {
                page_tokens: PAGE_TOKENS,
                max_append_tokens: MAX_APPEND_TOKENS,
            },
        ]
    };
    let native_values = if mode == SmartConfigMode::Enabled {
        native_user
            .map(|rows| vec![rows])
            .unwrap_or_else(|| automatic_cohort_width(profile.active).to_vec())
    } else {
        automatic_cohort_width(profile.active).to_vec()
    };

    for state in &state_values {
        validate_cuda_state_policy(profile, *state)?;
    }
    for rows in &native_values {
        profile.validate_cohort_width(*rows)?;
    }

    let (effective_cap, observed_unix_ms, observation) =
        smart_cuda_provider_cap(provider_bytes, device_bytes).map_err(ServiceError::inference)?;
    let mut candidates = Vec::with_capacity(state_values.len() * native_values.len());
    let mut refusals = Vec::new();
    let mut capacity_miss: Option<crate::ResourceLimit> = None;
    for cohort_width in native_values {
        for state_policy in &state_values {
            match ResidentModel::cuda_forward_source_preflight_for_state_policy(
                plan.clone(),
                profile.active,
                cohort_width,
                effective_cap,
                (*state_policy).into(),
            ) {
                Ok(inventory) => candidates.push(CudaSmartCandidate {
                    state_policy: *state_policy,
                    cohort_width,
                    inventory,
                }),
                Err(crate::Error::Resource(miss)) => {
                    if capacity_miss
                        .as_ref()
                        .is_none_or(|prior| miss.requested < prior.requested)
                    {
                        capacity_miss = Some(miss);
                    }
                }
                Err(error @ crate::Error::Unsupported(_)) => refusals.push(format!(
                    "state={} cohort_width={cohort_width}: {error}",
                    describe_cuda_state_policy(*state_policy)
                )),
                Err(error) => return Err(ServiceError::inference(error)),
            }
        }
    }

    let selected = candidates
        .into_iter()
        .max_by_key(|candidate| {
            (
                usize::from(
                    candidate.inventory.weight_storage == CudaForwardWeightStorage::Resident,
                ),
                candidate.cohort_width,
                usize::from(matches!(
                    candidate.state_policy,
                    crate::models::QwenCudaStatePolicy::Contiguous
                )),
            )
        })
        .ok_or_else(|| {
            ServiceError::inference(capacity_miss.map(crate::Error::Resource).unwrap_or_else(
                || {
                    crate::Error::Unsupported(format!(
                        "no qualified CUDA smart-config candidate: {}",
                        refusals.join("; ")
                    ))
                },
            ))
        })?;

    let common_reason = format!(
        "startup grid priority=resident_weights>native_width>contiguous_tie; {observation}; chosen={{storage:{:?},total:{}}}",
        selected.inventory.weight_storage, selected.inventory.total_admission_bytes
    );
    let state_source = if mode == SmartConfigMode::Enabled && state_user.is_some() {
        SmartConfigSource::User
    } else {
        SmartConfigSource::Auto
    };
    let native_source = if mode == SmartConfigMode::Enabled && native_user.is_some() {
        SmartConfigSource::User
    } else {
        SmartConfigSource::Auto
    };
    let state_reason = if state_source == SmartConfigSource::User {
        format!("user value constrains smart candidate grid; {common_reason}")
    } else if mode == SmartConfigMode::Exclusive && state_user.is_some() {
        format!("{common_reason}; user state_policy ignored by exclusive mode")
    } else {
        common_reason.clone()
    };
    let native_reason = if native_source == SmartConfigSource::User {
        format!("user value constrains smart candidate grid; {common_reason}")
    } else if mode == SmartConfigMode::Exclusive && native_user.is_some() {
        format!("{common_reason}; user cohort_width ignored by exclusive mode")
    } else {
        common_reason
    };
    let state = SmartConfigDecision {
        value: selected.state_policy,
        source: state_source,
        reason: state_reason,
    };
    let native = SmartConfigDecision {
        value: selected.cohort_width,
        source: native_source,
        reason: native_reason,
    };

    profile.cohort_width = native.value;
    let mut report =
        crate::smart_config::SmartConfigReport::new(model, "cuda", mode, observed_unix_ms);
    report.push(
        "state_policy",
        describe_cuda_state_policy(state.value),
        state.source,
        state.reason,
    );
    report.push(
        "cohort_width",
        native.value.to_string(),
        native.source,
        native.reason,
    );
    report.emit();

    Ok((
        SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            smart_config: crate::smart_config::SmartConfig::disabled(
                crate::models::QwenCudaSmartOverrides {
                    state_policy: Some(state.value),
                    cohort_width: Some(native.value),
                },
            ),
        },
        profile,
    ))
}

fn validate_aliases(aliases: &[String]) -> Result<(), ServiceError> {
    zerogravity_engine::router::validate_aliases(aliases.iter().map(String::as_str))
}
pub(crate) fn load_router(
    aliases: Vec<String>,
    cache: CacheOptions,
    snapshot: String,
    profile: QwenServingProfile,
    backend: SourceBackend,
) -> Result<Router, ServiceError> {
    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    let (profile, backend) = normalize_cuda_smart_input(profile, backend);
    backend.validate(profile)?;
    validate_aliases(&aliases)?;
    crate::cache::paths::validate_digest(&snapshot).map_err(ServiceError::inference)?;
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "system-ram").map_err(ServiceError::inference)?;
    arbiter
        .register_matching(resource.clone(), 16 * 1024 * 1024 * 1024)
        .map_err(ServiceError::inference)?;
    let admission = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new(&aliases[0], "original-source")
            .map_err(ServiceError::inference)?,
    };
    let host = ServingHost::from_process_admission(&admission)?;
    let SourceParts {
        source,
        tokenizer,
        max_token_bytes,
        metadata,
    } = open_source(
        &aliases[0],
        cache,
        &snapshot,
        profile.context,
        &host,
        backend.contract(),
    )?;
    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    let (backend, profile) =
        resolve_cuda_smart_config(&aliases[0], profile, source.plan.plan().clone(), backend)?;
    #[cfg(not(all(feature = "model-qwen35", feature = "cuda")))]
    let profile = profile;
    // The same source-owned preparation boundary as borrowed GGUF models.
    // Window access and the original family/cache lease stay with the provider.
    use crate::gpu::{ResidentTextPolicy, ResidentTextProvider};
    let (device, policy) = match backend {
        #[cfg(all(feature = "model-qwen35", feature = "hip"))]
        SourceBackend::Hip(budget, _) => (
            Device::Hip(Default::default()),
            ResidentTextPolicy::Hip(budget),
        ),
        #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
        SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            smart_config,
        } => (
            Device::Cuda(Default::default()),
            ResidentTextPolicy::Cuda {
                provider_bytes,
                device_bytes,
                state: smart_config
                    .overrides
                    .state_policy
                    .expect("resolved CUDA smart config")
                    .into(),
            },
        ),
    };
    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    let prefix_cache = if matches!(
        &policy,
        ResidentTextPolicy::Cuda {
            state: crate::gpu::CudaSourceStatePolicy::Paged { .. },
            ..
        }
    ) {
        // Existing validated profile bounds active <= 8 and context <= 32768.
        // Include one incoming token array before FIFO eviction. Snapshot
        // payloads separately retain their original resident reader credits.
        let max_entries = profile.active * 2;
        let max_weight = profile.context * profile.active;
        let bytes = TextPrefixCache::storage_bytes(max_entries, max_weight + profile.context)
            .map_err(ServiceError::inference)?;
        let storage = host.admit(
            &aliases[0],
            "prefix-cache-storage",
            AllocationClass::RecurrentState,
            bytes as u64,
        )?;
        Some(CachedPrefixes {
            cache: TextPrefixCache::new(max_entries, max_weight)
                .map_err(ServiceError::inference)?,
            _storage: storage,
        })
    } else {
        None
    };
    let provider = ResidentTextProvider {
        device,
        policy,
        source_owner: source.clone(),
        max_sessions: profile.active,
        cohort_width: profile.cohort_width,
        cancelled: None,
    };
    let program = source
        .plan
        .model_program()
        .map_err(ServiceError::inference)?;
    let resident = block_on(crate::models::prepare_executable(program, &provider))
        .map_err(ServiceError::inference)?
        .into_prepared();
    #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
    if resident.supports_shared_prefix() != prefix_cache.is_some() {
        // No new cache allocation after the physical owner is prepared.
        block_on(resident.shutdown()).map_err(ServiceError::inference)?;
        return Err(ServiceError::new(
            500,
            "source_owner",
            "prepared prefix policy differs",
        ));
    }
    let model = Arc::new(Model {
        resident: Mutex::new(Some(resident)),
        #[cfg(all(feature = "model-qwen35", feature = "cuda"))]
        prefix_cache: Mutex::new(prefix_cache),
        source,
        tokenizer,
        max_token_bytes,
        _metadata: metadata,
    });
    let models = aliases
        .into_iter()
        .map(|alias| (alias, model.clone()))
        .collect::<BTreeMap<_, _>>();
    Router::with_executor::<super::execute::SourceText>(
        models,
        host,
        WorkerConfig {
            workers: 1,
            active: profile.active,
            context: profile.context,
            output: profile.output,
            max_pending: profile.max_pending,
            policy: if profile.cohort_width == 0 {
                WorkPolicy::Sequential
            } else {
                WorkPolicy::Waves(WavePolicy {
                    max_members: profile.cohort_width,
                    max_tokens: profile.cohort_width * 64,
                    max_scratch: 256 * 1024 * 1024,
                    fairness: Fairness {
                        round_tokens: profile.cohort_width * 64 + profile.active,
                        prefill_age_rounds: 2,
                    },
                    prefill_quantum: 64,
                    unequal_prefill: false,
                    unequal_decode: true,
                    cohort_width: profile.cohort_width,
                    cohorts: Cohorts::Fused {
                        widths: if profile.cohort_width == 8 {
                            &[8, 4, 2, 1]
                        } else {
                            &[4, 2, 1]
                        },
                        split_prefix: true,
                        retry: RetryOrder::Regroup,
                    },
                })
            },
        },
    )
}

struct SourceParts {
    source: Arc<Source>,
    tokenizer: Arc<dyn TextTokenizer>,
    max_token_bytes: usize,
    metadata: Arc<zerogravity_engine::router::pending::ServingCharge>,
}
fn open_source(
    alias: &str,
    mut cache: CacheOptions,
    snapshot: &str,
    context: usize,
    host: &ServingHost,
    contract: NumericalContract,
) -> Result<SourceParts, ServiceError> {
    let admission = host.transfer_admission(alias, "original-source")?;
    // Bounds and real credit precede cache metadata, source parsing and plan.
    // A smaller caller cap remains smaller; never broaden cache authority.
    cache.limits.max_manifest_bytes = cache.limits.max_manifest_bytes.min(1024 * 1024);
    cache.limits.max_files = cache.limits.max_files.min(64);
    let control = host.admit(
        alias,
        "source-and-plan",
        AllocationClass::ImmutableWeights,
        64 * 1024 * 1024,
    )?;
    let store = CacheStore::open_existing(cache).map_err(ServiceError::inference)?;
    // The source loader authenticates every metadata payload and whole shard
    // from these same held descriptors. Avoid hashing the complete snapshot here
    // only to repeat the exact shard authentication in VerifiedWindowPlan::open.
    let snapshot = Arc::new(
        store
            .open_snapshot_descriptors(snapshot)
            .map_err(ServiceError::inference)?,
    );
    let mut opened = qwen35_source::open(Inputs::Snapshot(&snapshot), contract, &admission, None)
        .map_err(ServiceError::inference)?;
    let tokenizer_bytes = opened.tokenizer.take().ok_or_else(|| {
        ServiceError::new(
            500,
            "source_tokenizer",
            "authenticated tokenizer bytes absent",
        )
    })?;
    let bootstrap = crate::tokenize_admission::approve_arbiter(&admission, "bootstrap")?
        .for_source(&tokenizer_bytes)?;
    let retained = bootstrap.retain_arbiter(&admission, "tokenizer-inventory")?;
    let tokenizer = {
        let _constructor = bootstrap.constructor_of_arbiter(
            &admission,
            "tokenizer-constructor",
            &tokenizer_bytes,
        )?;
        QwenTokenizer::from_approved_bytes(&tokenizer_bytes).map_err(ServiceError::inference)?
    };
    drop(tokenizer_bytes);
    bootstrap.verify(&tokenizer)?;
    let max_token_bytes = tokenizer
        .max_token_bytes()
        .map_err(ServiceError::inference)?;
    let plan = opened
        .seal
        .build_plan(context)
        .map_err(ServiceError::inference)?;
    let source = Arc::new(Source {
        plan,
        metadata: opened.metadata,
        _snapshot: snapshot,
        _control: control,
    });
    let metadata =
        super::super::gpu::adapter_metadata::<Model>(host, alias, "source-model-adapter")?;
    let tokenizer = super::super::tokenizer_owner(
        (tokenizer, retained, source.clone()),
        |owner| &owner.0,
        host,
        alias,
    )?;
    Ok(SourceParts {
        source,
        tokenizer,
        max_token_bytes,
        metadata,
    })
}

#[cfg(all(test, all(feature = "model-qwen35", feature = "hip")))]
mod tests {
    use super::*;
    use crate::gpu::ResidentModel;
    #[test]
    #[ignore = "approved canonical cache snapshot; host-only, no GPU"]
    fn canonical_source_and_tokenizer_keep_original_cache_and_host_owners() {
        let snapshot =
            std::env::var("ZG_QWEN_SOURCE_SNAPSHOT").expect("explicit approved snapshot");
        let root = std::env::var_os("ZG_QWEN_CACHE_ROOT").map(std::path::PathBuf::from);
        let arbiter = ResourceArbiter::process_shared();
        let resource = PhysicalDeviceKey::new("host", "system-ram").unwrap();
        arbiter
            .register_matching(resource.clone(), 16 * 1024 * 1024 * 1024)
            .unwrap();
        let admission = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("source-host", "canonical").unwrap(),
        };
        let host = ServingHost::from_process_admission(&admission).unwrap();
        let _bootstrap =
            crate::tokenize_admission::approve_arbiter(&admission, "bootstrap").unwrap();
        let baseline = host.snapshot().unwrap();
        let parts = open_source(
            "source-host",
            CacheOptions::new(root, 96 * 1024 * 1024 * 1024),
            &snapshot,
            514,
            &host,
            NumericalContract::TorchCpuAvx2Bf16MixedV1,
        )
        .unwrap();
        let plan = &parts.source.plan;
        let program = plan.model_program().unwrap();
        assert!(program.source().same_issuer(plan.source()));
        assert!(Arc::ptr_eq(program.program(), plan.plan()));
        drop(program);
        assert_eq!(
            plan.source_seal().numerical_contract(),
            NumericalContract::TorchCpuAvx2Bf16MixedV1
        );
        assert_eq!(plan.context_capacity_tokens(), 514);
        assert_eq!(plan.max_chunk_rows(), 64);
        let inventory = ResidentModel::cpu_forward_source_preflight(
            plan.source().clone(),
            plan.plan().clone(),
            2,
        )
        .unwrap();
        assert_eq!(inventory.root_count, 6);
        let geometry = plan.model_profile();
        assert_eq!(plan.boundaries().len(), geometry.layer_count());
        assert_eq!(
            plan.plan().required_domains().len(),
            geometry.state_domains()
        );
        assert_eq!(
            plan.plan().proposed_domains().len(),
            geometry.state_domains()
        );
        assert!(inventory.source_weight_bytes > 0);
        assert!(
            inventory.resident_weight_bytes == 0
                || inventory.resident_weight_bytes == inventory.source_weight_bytes
        );
        assert_eq!(inventory.context_device_bytes, 4 * 1024 * 1024 * 1024);
        assert!(inventory.physical_device_upper_bytes <= 24 * 1024 * 1024 * 1024);
        assert!(inventory.host_upper_bytes <= 16 * 1024 * 1024 * 1024);
        if inventory.source_weight_bytes > 20 * 1024 * 1024 * 1024 {
            assert_eq!(inventory.resident_weight_bytes, 0);
        }
        eprintln!(
            "SOURCE_SDK_CANONICAL_HOST_INVENTORY geometry={geometry:?} inventory={inventory:?}"
        );
        assert_eq!(
            plan.source().source_identity(),
            plan.source_seal().source_identity()
        );
        let tokens = parts
            .tokenizer
            .prompt_tokens("Hello, world.", false)
            .unwrap();
        assert!(!tokens.is_empty());
        assert!(
            tokens
                .iter()
                .all(|&id| (id as usize) < plan.model_profile().vocabulary_size())
        );
        let call = plan
            .bind_invocation(
                crate::accelerator::program::InvocationPhase::Prefill,
                0,
                tokens.into(),
                true,
            )
            .unwrap();
        let materialized = call.materialize_plan().unwrap();
        assert_eq!(materialized.source(), plan.plan().source());
        drop((materialized, call));
        let weak_source = Arc::downgrade(&parts.source);
        let weak_cache = Arc::downgrade(&parts.source._snapshot);
        let final_tokenizer = parts.tokenizer.clone();
        assert!(host.snapshot().unwrap().1 > baseline.1);
        drop(parts);
        assert!(weak_source.upgrade().is_some());
        assert!(weak_cache.upgrade().is_some());
        drop(final_tokenizer);
        assert!(weak_source.upgrade().is_none());
        assert!(weak_cache.upgrade().is_none());
        assert_eq!(host.snapshot().unwrap(), baseline);
        eprintln!(
            "SOURCE_SDK_CANONICAL_HOST_PASS snapshot={snapshot} source_plan=true tokenizer=true final_reader=true gpu=false"
        );
    }

    #[test]
    #[ignore = "approved canonical cache snapshot, physical HIP, isolated process"]
    fn aligned_hip_prefill_matches_singleton_state_and_readout() {
        use crate::{
            accelerator::program::InvocationPhase,
            gpu::{
                ResidentOutput, ResidentTextCohortPreparation, ResidentTextPreparation,
                ResidentTextPublication,
            },
        };
        let snapshot =
            std::env::var("ZG_QWEN_SOURCE_SNAPSHOT").expect("explicit approved snapshot");
        let root = std::env::var_os("ZG_QWEN_CACHE_ROOT").map(std::path::PathBuf::from);
        let arbiter = ResourceArbiter::process_shared();
        let resource = PhysicalDeviceKey::new("host", "system-ram").unwrap();
        arbiter
            .register_matching(resource.clone(), 16 * 1024 * 1024 * 1024)
            .unwrap();
        let admission = TransferAdmission {
            arbiter: arbiter.clone(),
            resource,
            owner: ResourceOwner::new("source-prefill", "differential").unwrap(),
        };
        let host = ServingHost::from_process_admission(&admission).unwrap();
        let _bootstrap =
            crate::tokenize_admission::approve_arbiter(&admission, "bootstrap").unwrap();
        let parts = open_source(
            "source-prefill",
            CacheOptions::new(root, 96 * 1024 * 1024 * 1024),
            &snapshot,
            512,
            &host,
            NumericalContract::TorchCpuAvx2Bf16MixedV1,
        )
        .unwrap();
        let plan = &parts.source.plan;
        let tokens = [
            "A red bird follows the river. ",
            "The blue train climbs a mountain. ",
        ]
        .map(|p| {
            let mut ids = parts.tokenizer.prompt_tokens(&p.repeat(32), false).unwrap();
            assert!(ids.len() >= 81);
            ids.truncate(81);
            ids
        });
        assert_ne!(tokens[0], tokens[1]);
        block_on(async {
            let model = ResidentModel::load_cpu_forward_source_with_budget_and_cohort(
                Device::Hip(Default::default()),
                plan.source().clone(),
                parts.source.clone(),
                plan.plan().clone(),
                4,
                4,
                crate::gpu::HipSourceBudget::default(),
                None,
            )
            .await
            .unwrap();
            let mut singles = [
                model.session().await.unwrap(),
                model.session().await.unwrap(),
            ];
            let mut cohort = [
                model.session().await.unwrap(),
                model.session().await.unwrap(),
            ];
            let mut base = 0;
            let mut compared = 0usize;
            for (rows, emit) in [(64usize, false), (17, true)] {
                let calls = tokens
                    .iter()
                    .map(|ids| {
                        plan.bind_invocation(
                            InvocationPhase::Prefill,
                            base,
                            Arc::from(&ids[base..base + rows]),
                            emit,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let output = if emit {
                    ResidentOutput::Logits
                } else {
                    ResidentOutput::StateOnly
                };
                let mut expected = Vec::new();
                for (session, call) in singles.iter_mut().zip(&calls) {
                    let ResidentTextPreparation::Ready(pending) = session
                        .prepare(
                            call.bound_invocation(),
                            InvocationPhase::Prefill,
                            output,
                            None,
                        )
                        .await
                        .unwrap()
                    else {
                        panic!("singleton prefill refused")
                    };
                    let ResidentTextPublication::Committed(read) = session.commit(pending).unwrap()
                    else {
                        panic!("singleton prefill discarded")
                    };
                    expected.push(read);
                }
                if emit {
                    let before = cohort.each_ref().map(|session| session.binding());
                    let inputs = cohort
                        .iter()
                        .zip(&calls)
                        .map(|(session, call)| (session, call.bound_invocation()))
                        .collect::<Vec<_>>();
                    let ResidentTextCohortPreparation::Ready(discard) = model
                        .prepare_prefill_cohort_admitted(&inputs)
                        .await
                        .unwrap()
                    else {
                        panic!("cohort prefill refused")
                    };
                    assert!(discard.fused_dispatches() > 0);
                    drop(discard);
                    assert_eq!(cohort.each_ref().map(|session| session.binding()), before);
                }
                let inputs = cohort
                    .iter()
                    .zip(&calls)
                    .map(|(session, call)| (session, call.bound_invocation()))
                    .collect::<Vec<_>>();
                let ResidentTextCohortPreparation::Ready(pending) = model
                    .prepare_prefill_cohort_admitted(&inputs)
                    .await
                    .unwrap()
                else {
                    panic!("cohort prefill refused")
                };
                assert!(pending.fused_dispatches() > 0);
                let reads = pending
                    .commit(&mut cohort.iter_mut().collect::<Vec<_>>())
                    .unwrap();
                base += rows;
                for lane in 0..2 {
                    assert_eq!(cohort[lane].binding().frontier(), base as u64);
                    assert_eq!(singles[lane].binding().frontier(), base as u64);
                    assert_ne!(
                        cohort[lane].binding().session(),
                        singles[lane].binding().session()
                    );
                    assert_eq!(reads[lane].logit_steps, usize::from(emit));
                    assert_eq!(reads[lane].state_only_steps, usize::from(!emit));
                    let bits_equal = |a: &[f32], b: &[f32]| {
                        a.len() == b.len()
                            && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
                    };
                    assert!(
                        bits_equal(&reads[lane].logits, &expected[lane].logits),
                        "logits lane={lane} base={base}"
                    );
                    let actual = cohort[lane]
                        .snapshot()
                        .unwrap()
                        .read_domains()
                        .await
                        .unwrap();
                    let expected = singles[lane]
                        .snapshot()
                        .unwrap()
                        .read_domains()
                        .await
                        .unwrap();
                    assert_eq!(actual.len(), expected.len());
                    for ((slot, a), (expected_slot, b)) in actual.iter().zip(expected.iter()) {
                        assert_eq!(slot, expected_slot);
                        assert!(
                            bits_equal(a, b),
                            "state lane={lane} base={base} slot={slot:?}"
                        );
                        compared += a.len();
                    }
                    compared += reads[lane].logits.len();
                }
                assert_ne!(cohort[0].binding().session(), cohort[1].binding().session());
                println!(
                    "HIP_PREFILL_DIFFERENTIAL rows={rows} frontier={base} state_only={} words={compared}",
                    !emit
                );
            }
            drop((singles, cohort, model));
        });
        let device = arbiter
            .snapshot(&PhysicalDeviceKey::new("pci", "0000:c6:00.0").unwrap())
            .unwrap();
        assert_eq!(device.reserved, 4 * 1024 * 1024 * 1024);
        assert!(!device.fenced);
    }

    #[test]
    fn aliases_refuse_before_cache_or_device_access() {
        assert!(validate_aliases(&["a".into(), "b".into()]).is_ok());
        for aliases in [
            vec![],
            vec!["a".into(), "a".into()],
            vec!["../x".into()],
            vec!["x".repeat(97)],
        ] {
            assert!(validate_aliases(&aliases).is_err());
        }
    }
}

#[cfg(all(test, all(feature = "model-qwen35", feature = "cuda")))]
mod cuda_paged_tests {
    use super::*;
    use crate::{
        accelerator::program::InvocationPhase,
        gpu::{
            CudaSourceStatePolicy, ResidentModel, ResidentOutput, ResidentTextCohortPreparation,
            ResidentTextModel, ResidentTextPreparation, ResidentTextPublication,
        },
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    #[cfg(feature = "qwen35-cuda-paged-cancellation-qualification")]
    use std::{future::Future, task::Poll};

    const MIB: usize = 1024 * 1024;
    const PROVIDER_BYTES: usize = 4608 * MIB;
    const DEVICE_BYTES: u64 = 5120 * MIB as u64;

    #[derive(Debug)]
    struct Checkpoint {
        frontier: u64,
        logits: Vec<f32>,
        domains: Vec<(crate::accelerator::program::StateSlot, Vec<f32>)>,
    }

    #[derive(Debug)]
    struct NativeCheckpoint {
        token: (u32, u32),
        domains: Vec<(crate::accelerator::program::StateSlot, Vec<f32>)>,
    }

    fn exact_words(expected: &[f32], actual: &[f32], label: &str) {
        assert_eq!(expected.len(), actual.len(), "{label} length");
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "{label} differs at word {index}"
            );
        }
    }

    fn exact_checkpoint(expected: &Checkpoint, actual: &Checkpoint, label: &str) {
        assert_eq!(expected.frontier, actual.frontier, "{label} frontier");
        exact_words(&expected.logits, &actual.logits, label);
        assert_eq!(
            expected.domains.len(),
            42,
            "{label} expected all state domains"
        );
        assert_eq!(actual.domains.len(), 42, "{label} actual all state domains");
        for (slot, expected) in &expected.domains {
            let actual = actual
                .domains
                .iter()
                .find(|(actual, _)| actual == slot)
                .map(|(_, words)| words)
                .unwrap_or_else(|| panic!("{label} missing {slot:?}"));
            exact_words(expected, actual, label);
        }
    }

    fn exact_domains(
        expected: &[(crate::accelerator::program::StateSlot, Vec<f32>)],
        actual: &[(crate::accelerator::program::StateSlot, Vec<f32>)],
        label: &str,
    ) {
        assert_eq!(expected.len(), 42, "{label} expected all state domains");
        assert_eq!(actual.len(), 42, "{label} actual all state domains");
        for (slot, expected) in expected {
            let actual = actual
                .iter()
                .find(|(actual, _)| actual == slot)
                .map(|(_, words)| words)
                .unwrap_or_else(|| panic!("{label} missing {slot:?}"));
            exact_words(expected, actual, label);
        }
    }

    fn exact_restore_publication(
        before: crate::session_root::RootBinding,
        after: crate::session_root::RootBinding,
        saved: crate::session_root::RootBinding,
        label: &str,
    ) {
        assert_eq!(after.source(), before.source(), "{label} source");
        assert_eq!(after.session(), before.session(), "{label} session");
        assert_eq!(
            after.root(),
            before
                .root()
                .checked_add(1)
                .expect("restore root increment"),
            "{label} root"
        );
        assert_eq!(
            after.session_epoch(),
            before
                .session_epoch()
                .checked_add(1)
                .expect("restore epoch increment"),
            "{label} epoch"
        );
        assert_eq!(after.frontier(), saved.frontier(), "{label} frontier");
    }

    fn paged_source_fixture(label: &str) -> (SourceParts, usize, usize, Vec<u32>) {
        let snapshot =
            std::env::var("ZG_QWEN_SOURCE_SNAPSHOT").expect("explicit approved snapshot");
        let root = std::env::var_os("ZG_QWEN_CACHE_ROOT").map(std::path::PathBuf::from);
        let page_tokens = std::env::var("ZG_QWEN_PAGED_PAGE_TOKENS")
            .expect("explicit page_tokens")
            .parse::<usize>()
            .expect("page_tokens");
        let max_append_tokens = std::env::var("ZG_QWEN_PAGED_MAX_APPEND_TOKENS")
            .expect("explicit max_append_tokens")
            .parse::<usize>()
            .expect("max_append_tokens");
        assert!((2..=64).contains(&page_tokens));
        assert!(max_append_tokens >= page_tokens);
        let arbiter = ResourceArbiter::process_shared();
        let resource = PhysicalDeviceKey::new("host", "system-ram").unwrap();
        arbiter
            .register_matching(resource.clone(), 16 * 1024 * 1024 * 1024)
            .unwrap();
        let admission = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new(label, "original-source").unwrap(),
        };
        let host = ServingHost::from_process_admission(&admission).unwrap();
        let parts = open_source(
            label,
            CacheOptions::new(root, 96 * 1024 * 1024 * 1024),
            &snapshot,
            512,
            &host,
            NumericalContract::CudaSm86Bf16ReferenceV1,
        )
        .unwrap();
        let needed = page_tokens * 2 + 2;
        let prompt = " a".repeat(needed * 2);
        let mut tokens = parts
            .tokenizer
            .prompt_tokens(&prompt, false)
            .expect("source tokenizer");
        assert!(tokens.len() >= needed);
        tokens.truncate(needed);
        assert!(
            tokens
                .iter()
                .all(|&id| { (id as usize) < parts.source.plan.model_profile().vocabulary_size() })
        );
        (parts, page_tokens, max_append_tokens, tokens)
    }

    async fn trajectory(
        model: &Arc<ResidentTextModel>,
        plan: &crate::qwen35::plan::QwenForwardPlan,
        tokens: &[u32],
        page_tokens: usize,
    ) -> crate::Result<Vec<Checkpoint>> {
        let mut session = model.session().await?;
        let chunks = [page_tokens - 1, 1, 1, page_tokens];
        let mut offset = 0usize;
        let mut checkpoints = Vec::with_capacity(chunks.len());
        for rows in chunks {
            let ids = Arc::<[u32]>::from(tokens[offset..offset + rows].to_vec());
            let invocation = plan.bind_invocation(InvocationPhase::Prefill, offset, ids, true)?;
            let pending = session
                .prepare(
                    invocation.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await?;
            let ResidentTextPreparation::Ready(pending) = pending else {
                return Err(crate::Error::State(
                    "paged source differential candidate refused".into(),
                ));
            };
            let ResidentTextPublication::Committed(read) = session.commit(pending)? else {
                return Err(crate::Error::State(
                    "paged source differential candidate discarded".into(),
                ));
            };
            offset += rows;
            if session.binding().frontier() != offset as u64 {
                return Err(crate::Error::State(
                    "paged source differential frontier differs".into(),
                ));
            }
            let snapshot = session.snapshot()?;
            checkpoints.push(Checkpoint {
                frontier: session.binding().frontier(),
                logits: read.logits.clone(),
                domains: snapshot.read_domains().await?.to_vec(),
            });
            drop(read);
        }
        Ok(checkpoints)
    }

    async fn native_trajectory(
        model: &Arc<ResidentTextModel>,
        plan: &crate::qwen35::plan::QwenForwardPlan,
        tokens: &[u32],
        width: usize,
    ) -> crate::Result<Vec<(NativeCheckpoint, NativeCheckpoint)>> {
        let mut sessions = Vec::with_capacity(width);
        for _ in 0..width {
            sessions.push(model.session().await?);
        }
        let prefill = (0..width)
            .map(|row| {
                plan.bind_invocation(
                    InvocationPhase::Prefill,
                    0,
                    Arc::<[u32]>::from(vec![tokens[row]]),
                    true,
                )
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let inputs = sessions
            .iter()
            .zip(&prefill)
            .map(|(session, invocation)| (session, invocation.bound_invocation()))
            .collect::<Vec<_>>();
        let ResidentTextCohortPreparation::Ready(pending) =
            model.prepare_prefill_cohort_admitted(&inputs).await?
        else {
            return Err(crate::Error::State(
                "paged native prefill cohort was not admitted".into(),
            ));
        };
        let reads = pending.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        let mut prefill_checkpoints = Vec::with_capacity(width);
        let mut feedback = Vec::with_capacity(width);
        for (session, read) in sessions.iter().zip(&reads) {
            let token = read
                .device_token
                .as_ref()
                .ok_or_else(|| crate::Error::State("paged native prefill token absent".into()))?;
            feedback.push(token.clone());
            prefill_checkpoints.push(NativeCheckpoint {
                token: (token.id, token.max_bits),
                domains: session.snapshot()?.read_domains().await?.to_vec(),
            });
        }
        drop(reads);

        let decode = feedback
            .iter()
            .enumerate()
            .map(|(row, token)| {
                plan.bind_invocation(
                    InvocationPhase::Decode,
                    1,
                    Arc::<[u32]>::from(vec![token.id]),
                    true,
                )
                .map(|invocation| (row, invocation))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        let inputs = sessions
            .iter()
            .zip(&decode)
            .zip(&feedback)
            .map(|((session, (_, invocation)), token)| {
                (session, invocation.bound_invocation(), token)
            })
            .collect::<Vec<_>>();
        let ResidentTextCohortPreparation::Ready(pending) =
            model.prepare_greedy_decode_cohort_admitted(&inputs).await?
        else {
            return Err(crate::Error::State(
                "paged native decode cohort was not admitted".into(),
            ));
        };
        let reads = pending.commit(&mut sessions.iter_mut().collect::<Vec<_>>())?;
        let mut result = Vec::with_capacity(width);
        for ((session, read), prefill) in sessions.iter().zip(&reads).zip(prefill_checkpoints) {
            let token = read
                .device_token
                .as_ref()
                .ok_or_else(|| crate::Error::State("paged native decode token absent".into()))?;
            result.push((
                prefill,
                NativeCheckpoint {
                    token: (token.id, token.max_bits),
                    domains: session.snapshot()?.read_domains().await?.to_vec(),
                },
            ));
        }
        Ok(result)
    }

    #[test]
    #[ignore = "approved original HF cache; fused 2/4/8-row cohort contiguous-vs-paged state transport"]
    fn original_source_native_paged_state_transport_matches_contiguous() {
        let (parts, page_tokens, max_append_tokens, tokens) =
            paged_source_fixture("paged-source-native-transport");
        let run = |policy| {
            ResidentModel::load_cuda_forward_source_with_state_policy(
                Device::Cuda(Default::default()),
                parts.source.plan.source().clone(),
                parts.source.clone(),
                parts.source.plan.plan().clone(),
                8,
                8,
                PROVIDER_BYTES,
                DEVICE_BYTES,
                policy,
                None,
            )
        };
        let expected = block_on(async {
            let model = run(CudaSourceStatePolicy::Contiguous).await.unwrap();
            let mut widths = Vec::new();
            for width in [2, 4, 8] {
                widths.push(
                    native_trajectory(&model, &parts.source.plan, &tokens, width)
                        .await
                        .unwrap(),
                );
            }
            model.shutdown().await.unwrap();
            widths
        });
        let actual = block_on(async {
            let model = run(CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            })
            .await
            .unwrap();
            let mut widths = Vec::new();
            for width in [2, 4, 8] {
                widths.push(
                    native_trajectory(&model, &parts.source.plan, &tokens, width)
                        .await
                        .unwrap(),
                );
            }
            model.shutdown().await.unwrap();
            widths
        });
        for (width_index, width) in [2, 4, 8].into_iter().enumerate() {
            for row in 0..width {
                let (expected_prefill, expected_decode) = &expected[width_index][row];
                let (actual_prefill, actual_decode) = &actual[width_index][row];
                assert_eq!(actual_prefill.token, expected_prefill.token);
                assert_eq!(actual_decode.token, expected_decode.token);
                exact_domains(
                    &expected_prefill.domains,
                    &actual_prefill.domains,
                    &format!("native paged B{width} prefill row{row}"),
                );
                exact_domains(
                    &expected_decode.domains,
                    &actual_decode.domains,
                    &format!("native paged B{width} decode row{row}"),
                );
            }
        }
        println!(
            "P12_SOURCE_PAGED_NATIVE_TRANSPORT_PASS widths=2,4,8 prefill=exact decode=exact all_state_domains=exact contiguous_reference=true final_shutdown=true"
        );
    }

    #[test]
    #[ignore = "approved original HF cache; actual resident/streamed CUDA source transport"]
    fn original_source_resident_and_streamed_transport_match() {
        use crate::gpu::CudaForwardWeightStorage;

        let (parts, page_tokens, _, tokens) =
            paged_source_fixture("resident-streamed-source-transport");
        // Same source, plan, math, state policy and row limits. Use the
        // actual complete admission to choose a genuine streamed fit.
        let policy = CudaSourceStatePolicy::Contiguous;
        let fit = |cap| {
            ResidentModel::cuda_forward_source_preflight_for_state_policy(
                parts.source.plan.plan().clone(),
                4,
                4,
                cap,
                policy,
            )
            .unwrap()
        };
        let resident = fit(PROVIDER_BYTES);
        assert_eq!(resident.weight_storage, CudaForwardWeightStorage::Resident);
        let streamed_cap = resident.total_admission_bytes - 1;
        let streamed = fit(streamed_cap);
        assert_eq!(streamed.weight_storage, CudaForwardWeightStorage::Streamed);
        assert_eq!(streamed.resident_weight_bytes, 0);
        let run = |cap, resident_bytes| {
            block_on(async {
                let model = ResidentModel::load_cuda_forward_source_with_state_policy(
                    Device::Cuda(Default::default()),
                    parts.source.plan.source().clone(),
                    parts.source.clone(),
                    parts.source.plan.plan().clone(),
                    4,
                    4,
                    cap,
                    DEVICE_BYTES,
                    policy,
                    None,
                )
                .await
                .unwrap();
                assert_eq!(model.admission().resident_weight_bytes, resident_bytes);
                let singleton = trajectory(&model, &parts.source.plan, &tokens, page_tokens)
                    .await
                    .unwrap();
                let cohort = native_trajectory(&model, &parts.source.plan, &tokens, 4)
                    .await
                    .unwrap();
                if resident_bytes == 0 {
                    // Known physical completion but cancelled publication.
                    // The same streamed page/model remains reusable.
                    let mut session = model.session().await.unwrap();
                    let initial = session.binding();
                    let call = parts
                        .source
                        .plan
                        .bind_invocation(InvocationPhase::Prefill, 0, Arc::from(&tokens[..1]), true)
                        .unwrap();
                    let cancelled = Arc::new(AtomicBool::new(false));
                    let ResidentTextPreparation::Ready(pending) = session
                        .prepare(
                            call.bound_invocation(),
                            InvocationPhase::Prefill,
                            ResidentOutput::Logits,
                            Some(cancelled.clone()),
                        )
                        .await
                        .unwrap()
                    else {
                        panic!("streamed cancellation admission");
                    };
                    cancelled.store(true, Ordering::Release);
                    assert!(matches!(
                        session.commit(pending).unwrap(),
                        ResidentTextPublication::Discarded(_)
                    ));
                    assert_eq!(session.binding(), initial);
                    let ResidentTextPreparation::Ready(pending) = session
                        .prepare(
                            call.bound_invocation(),
                            InvocationPhase::Prefill,
                            ResidentOutput::Logits,
                            None,
                        )
                        .await
                        .unwrap()
                    else {
                        panic!("streamed retry admission");
                    };
                    assert!(matches!(
                        session.commit(pending).unwrap(),
                        ResidentTextPublication::Committed(_)
                    ));
                    assert_eq!(session.binding().frontier(), 1);
                }
                model.shutdown().await.unwrap();
                (singleton, cohort)
            })
        };
        let (expected, expected_cohort) = run(PROVIDER_BYTES, resident.resident_weight_bytes);
        let (actual, actual_cohort) = run(streamed_cap, 0);
        assert_eq!(actual.len(), expected.len());
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            exact_checkpoint(
                expected,
                actual,
                &format!("streamed singleton checkpoint {index}"),
            );
        }
        assert_eq!(actual_cohort.len(), expected_cohort.len());
        for (row, ((expected_prefill, expected_decode), (actual_prefill, actual_decode))) in
            expected_cohort.iter().zip(&actual_cohort).enumerate()
        {
            assert_eq!(actual_prefill.token, expected_prefill.token);
            assert_eq!(actual_decode.token, expected_decode.token);
            exact_domains(
                &expected_prefill.domains,
                &actual_prefill.domains,
                &format!("streamed cohort prefill row{row}"),
            );
            exact_domains(
                &expected_decode.domains,
                &actual_decode.domains,
                &format!("streamed cohort decode row{row}"),
            );
        }
        println!(
            "SOURCE_RESIDENT_STREAMED_TRANSPORT_PASS singleton=exact cohort4=exact full_logits=exact all_domains=exact actual_streamed_weights=true final_shutdown=true"
        );
    }

    #[test]
    #[ignore = "approved original HF cache; complete-domain paged prefix fork/COW"]
    fn original_source_paged_complete_prefix_fork_matches_replay() {
        let (parts, page_tokens, max_append_tokens, tokens) =
            paged_source_fixture("paged-source-prefix-fork");
        let model = block_on(ResidentModel::load_cuda_forward_source_with_state_policy(
            Device::Cuda(Default::default()),
            parts.source.plan.source().clone(),
            parts.source.clone(),
            parts.source.plan.plan().clone(),
            3,
            0,
            PROVIDER_BYTES,
            DEVICE_BYTES,
            CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            },
            None,
        ))
        .unwrap();
        block_on(async {
            let prefix = page_tokens - 1;
            let prefix_ids = Arc::<[u32]>::from(tokens[..prefix].to_vec());
            let prefix_call = parts
                .source
                .plan
                .bind_invocation(InvocationPhase::Prefill, 0, prefix_ids, false)
                .unwrap();

            let mut producer = model.session().await.unwrap();
            let ResidentTextPreparation::Ready(pending) = producer
                .prepare(
                    prefix_call.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::StateOnly,
                    None,
                )
                .await
                .unwrap()
            else {
                panic!("producer prefix admission");
            };
            let ResidentTextPublication::Committed(read) = producer.commit(pending).unwrap() else {
                panic!("producer prefix publication");
            };
            drop(read);
            let cached = producer.snapshot().unwrap();
            let state_identity = model.state_identity().unwrap();
            assert_eq!(cached.state_identity().unwrap(), state_identity);
            assert_eq!(producer.state_identity().unwrap(), state_identity);
            assert_eq!(
                state_identity.source(),
                parts.source.plan.source_seal().source_identity()
            );
            assert_eq!(state_identity.geometry().context_capacity(), 512);
            assert_eq!(
                state_identity.selected().provider,
                zerogravity_core::provider::ProviderKind::Cuda
            );
            assert_eq!(
                state_identity.numerical_profile(),
                NumericalContract::CudaSm86Bf16ReferenceV1.program_profile()
            );
            assert_eq!(
                state_identity.state_policy(),
                CudaSourceStatePolicy::Paged {
                    page_tokens,
                    max_append_tokens
                }
            );
            assert!(
                state_identity.invocation_ports().token_ids
                    && state_identity.invocation_ports().positions
            );
            let cached_binding = cached.binding();
            let cached_domains = cached.read_domains().await.unwrap().to_vec();
            assert_eq!(cached_binding.frontier(), prefix as u64);

            let mut fork = model.session().await.unwrap();
            let fresh = fork.binding();
            assert_eq!(
                (fresh.root(), fresh.session_epoch(), fresh.frontier()),
                (0, 0, 0)
            );
            assert_eq!(fork.state_identity().unwrap(), state_identity);
            assert!(
                fork.restore(&cached).is_err(),
                "same-session restore must reject a foreign root"
            );
            assert_eq!(fork.binding(), fresh);
            fork.fork_shared_prefix(&cached).unwrap();
            let forked = fork.binding();
            assert_eq!(forked.session(), fresh.session());
            assert_ne!(forked.session(), cached_binding.session());
            assert_eq!(forked.source(), cached_binding.source());
            assert_eq!(forked.root(), 1);
            assert_eq!(forked.session_epoch(), 1);
            assert_eq!(forked.frontier(), prefix as u64);
            exact_domains(
                &cached_domains,
                &fork.snapshot().unwrap().read_domains().await.unwrap(),
                "shared prefix immediate",
            );

            let mut replay = model.session().await.unwrap();
            let ResidentTextPreparation::Ready(pending) = replay
                .prepare(
                    prefix_call.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::StateOnly,
                    None,
                )
                .await
                .unwrap()
            else {
                panic!("replay prefix admission");
            };
            let ResidentTextPublication::Committed(read) = replay.commit(pending).unwrap() else {
                panic!("replay prefix publication");
            };
            drop(read);
            assert!(replay.fork_shared_prefix(&cached).is_err());
            exact_domains(
                &cached_domains,
                &replay.snapshot().unwrap().read_domains().await.unwrap(),
                "shared prefix replay",
            );

            let continuation = parts
                .source
                .plan
                .bind_invocation(
                    InvocationPhase::Prefill,
                    prefix,
                    Arc::<[u32]>::from(vec![tokens[prefix]]),
                    true,
                )
                .unwrap();
            let ResidentTextPreparation::Ready(fork_pending) = fork
                .prepare(
                    continuation.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap()
            else {
                panic!("fork continuation admission");
            };
            let ResidentTextPublication::Committed(fork_read) = fork.commit(fork_pending).unwrap()
            else {
                panic!("fork continuation publication");
            };
            let ResidentTextPreparation::Ready(replay_pending) = replay
                .prepare(
                    continuation.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap()
            else {
                panic!("replay continuation admission");
            };
            let ResidentTextPublication::Committed(replay_read) =
                replay.commit(replay_pending).unwrap()
            else {
                panic!("replay continuation publication");
            };
            exact_words(
                &replay_read.logits,
                &fork_read.logits,
                "shared prefix continuation logits",
            );
            let fork_domains = fork.snapshot().unwrap().read_domains().await.unwrap();
            let replay_domains = replay.snapshot().unwrap().read_domains().await.unwrap();
            exact_domains(
                &replay_domains,
                &fork_domains,
                "shared prefix continuation across all state domains",
            );
            exact_domains(
                &cached_domains,
                &cached.read_domains().await.unwrap(),
                "shared prefix cached snapshot immutable",
            );
            drop((fork_read, replay_read, fork_domains, replay_domains));
            let own = fork.snapshot().unwrap();
            let before = fork.binding();
            fork.restore(&own).unwrap();
            exact_restore_publication(
                before,
                fork.binding(),
                own.binding(),
                "same-session complete snapshot",
            );
            assert_eq!(
                own.state_identity().unwrap(),
                fork.state_identity().unwrap()
            );
            drop(own);

            drop((cached, producer, fork, replay));
            model.shutdown().await.unwrap();
        });
        println!(
            "P12_SOURCE_PAGED_PREFIX_FORK_PASS prefix_tokens={} append_tokens=1 root_independent=true no_replay=true logits=exact all_state_domains=exact cached_snapshot_immutable=true final_shutdown=true",
            page_tokens - 1
        );
    }

    #[test]
    #[ignore = "approved original HF cache; scoped worker prefix hit/miss"]
    fn original_source_paged_scoped_prefix_cache_skips_replay() {
        use std::time::{Duration, Instant};
        use zerogravity_engine::{
            router::{Request, ResponseEvent, TokenMetricsRequest},
            state_cache::PrefixSharingScope,
        };

        let snapshot =
            std::env::var("ZG_QWEN_SOURCE_SNAPSHOT").expect("explicit approved snapshot");
        let root = std::env::var_os("ZG_QWEN_CACHE_ROOT").map(std::path::PathBuf::from);
        let mut router = load_router(
            vec!["source".into()],
            CacheOptions::new(root, 96 * 1024 * 1024 * 1024),
            snapshot,
            QwenServingProfile {
                active: 2,
                context: 512,
                output: 1,
                max_pending: Some(4),
                cohort_width: 0,
            },
            SourceBackend::Cuda {
                provider_bytes: PROVIDER_BYTES,
                device_bytes: DEVICE_BYTES,
                smart_config: crate::models::SmartConfig::enabled(
                    crate::models::QwenCudaSmartOverrides {
                        state_policy: Some(crate::models::QwenCudaStatePolicy::Paged {
                            page_tokens: 4,
                            max_append_tokens: 64,
                        }),
                        cohort_width: None,
                    },
                ),
            },
        )
        .unwrap();
        let prompt = " a".repeat(65);
        assert_eq!(
            router
                .model("source")
                .unwrap()
                .prepared()
                .tokenizer()
                .prompt_tokens(&prompt, false)
                .unwrap()
                .len(),
            65
        );

        let model_tokens = |router: &Router| {
            router.admission().unwrap()["completed_model_token_steps"]
                .as_u64()
                .unwrap()
        };
        let run = |router: &Router, scope: PrefixSharingScope, prompt: &str, prompt_len: usize| {
            let events = router
                .request_scoped(
                    Request {
                        model: "source".into(),
                        input: prompt.into(),
                        chat: false,
                        max_tokens: 1,
                        stream: true,
                        token_metrics: Some(TokenMetricsRequest {
                            started: Instant::now(),
                            wire_request_id: None,
                        }),
                    },
                    scope,
                )
                .unwrap();
            let mut ids = Vec::new();
            loop {
                match events
                    .events
                    .recv_timeout(Duration::from_secs(300))
                    .expect("scoped prefix response deadline")
                {
                    ResponseEvent::TokenText { token, .. } => ids.push(token.id),
                    ResponseEvent::Done {
                        prompt_tokens,
                        completion_tokens,
                        ..
                    } => {
                        assert_eq!((prompt_tokens, completion_tokens), (prompt_len, 1));
                        break;
                    }
                    ResponseEvent::Error(error) => panic!("scoped prefix worker: {error}"),
                    ResponseEvent::Text(_) => panic!("streaming token observation required"),
                }
            }
            assert_eq!(ids.len(), 1);
            ids[0]
        };

        let scope_a = PrefixSharingScope::new(7).unwrap();
        let scope_b = PrefixSharingScope::new(8).unwrap();
        let before = model_tokens(&router);
        let warm = run(&router, scope_a, &prompt, 65);
        let after_warm = model_tokens(&router);
        let hit = run(&router, scope_a, &prompt, 65);
        let after_hit = model_tokens(&router);
        let miss = run(&router, scope_b, &prompt, 65);
        let after_miss = model_tokens(&router);

        assert_eq!(warm, hit);
        assert_eq!(warm, miss);
        assert_eq!(after_warm - before, 65);
        assert_eq!(after_hit - after_warm, 1);
        assert_eq!(after_miss - after_hit, 65);
        // Both scopes have a state-only prefix of length 64. A complete
        // match must leave final logit work, not restore a feedback-less root.
        let short = " a".repeat(64);
        let tokenizer = router.model("source").unwrap().prepared().tokenizer();
        assert_eq!(tokenizer.prompt_tokens(&short, false).unwrap().len(), 64);
        drop(tokenizer);
        let exact_length = run(&router, scope_a, &short, 64);
        let after_short = model_tokens(&router);
        let uncached = run(&router, PrefixSharingScope::new(9).unwrap(), &short, 64);
        assert_eq!(exact_length, uncached);
        assert_eq!(after_short - after_miss, 64);
        assert_eq!(model_tokens(&router) - after_short, 64);
        router.shutdown().unwrap();
        println!(
            "P12_SOURCE_PAGED_PREFIX_CACHE_PASS prompt_tokens=65 cached_prefix=64 warm_model_tokens=65 same_scope_model_tokens=1 different_scope_model_tokens=65 output_id_exact=true exact_length_replay=64 scoped=true final_shutdown=true"
        );
    }

    #[test]
    #[ignore = "approved original HF cache; live paged prefix FIFO/weight eviction and scope revocation"]
    fn original_source_paged_prefix_cache_evicts_and_revokes() {
        use std::time::Duration;
        use zerogravity_engine::{
            router::{Request, ResponseEvent},
            state_cache::{PrefixCacheEviction, PrefixSharingScope},
        };

        let snapshot =
            std::env::var("ZG_QWEN_SOURCE_SNAPSHOT").expect("explicit approved snapshot");
        let root = std::env::var_os("ZG_QWEN_CACHE_ROOT").map(std::path::PathBuf::from);
        let mut router = load_router(
            vec!["source".into()],
            CacheOptions::new(root, 96 * 1024 * 1024 * 1024),
            snapshot,
            QwenServingProfile {
                active: 1,
                context: 512,
                output: 1,
                max_pending: Some(2),
                cohort_width: 0,
            },
            SourceBackend::Cuda {
                provider_bytes: PROVIDER_BYTES,
                device_bytes: DEVICE_BYTES,
                smart_config: crate::models::SmartConfig::enabled(
                    crate::models::QwenCudaSmartOverrides {
                        state_policy: Some(crate::models::QwenCudaStatePolicy::Paged {
                            page_tokens: 4,
                            max_append_tokens: 64,
                        }),
                        cohort_width: None,
                    },
                ),
            },
        )
        .unwrap();
        let tokenizer = router.model("source").unwrap().prepared().tokenizer();
        let prompt_a = " a".repeat(65);
        let prompt_b = " b".repeat(65);
        let prompt_c = " c".repeat(65);
        let prompt_long = " a".repeat(321);
        let prompt_short = " a".repeat(257);
        for (prompt, expected) in [
            (&prompt_a, 65),
            (&prompt_b, 65),
            (&prompt_c, 65),
            (&prompt_long, 321),
            (&prompt_short, 257),
        ] {
            assert_eq!(
                tokenizer.prompt_tokens(prompt, false).unwrap().len(),
                expected
            );
        }

        let model_tokens = |router: &Router| {
            router.admission().unwrap()["completed_model_token_steps"]
                .as_u64()
                .unwrap()
        };
        let run = |router: &Router, prompt: &str, scope: PrefixSharingScope| {
            let events = router
                .request_scoped(
                    Request {
                        model: "source".into(),
                        input: prompt.into(),
                        chat: false,
                        max_tokens: 1,
                        stream: false,
                        token_metrics: None,
                    },
                    scope,
                )
                .unwrap();
            loop {
                match events
                    .events
                    .recv_timeout(Duration::from_secs(300))
                    .expect("prefix pressure response deadline")
                {
                    ResponseEvent::Done {
                        output,
                        prompt_tokens,
                        completion_tokens,
                        ..
                    } => {
                        assert_eq!(
                            prompt_tokens,
                            tokenizer.prompt_tokens(prompt, false).unwrap().len()
                        );
                        assert_eq!(completion_tokens, 1);
                        break output;
                    }
                    ResponseEvent::Error(error) => panic!("prefix pressure worker: {error}"),
                    ResponseEvent::Text(_) | ResponseEvent::TokenText { .. } => {}
                }
            }
        };

        let fifo_scope = PrefixSharingScope::new(11).unwrap();
        let mut before = model_tokens(&router);
        let out_a = run(&router, &prompt_a, fifo_scope);
        let mut after = model_tokens(&router);
        assert_eq!(after - before, 65);
        before = after;
        let _ = run(&router, &prompt_b, fifo_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 65);
        before = after;
        let _ = run(&router, &prompt_c, fifo_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 65);
        before = after;
        let _ = run(&router, &prompt_b, fifo_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 1, "newer FIFO entry must remain");
        before = after;
        let out_a_after = run(&router, &prompt_a, fifo_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 65, "oldest FIFO entry must be evicted");
        assert_eq!(out_a_after, out_a);
        assert_eq!(
            router.invalidate_prefix_scope(fifo_scope).unwrap(),
            PrefixCacheEviction {
                entries: 2,
                weight: 128,
            }
        );
        before = model_tokens(&router);
        let _ = run(&router, &prompt_c, fifo_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 65, "revoked scope must miss");
        assert_eq!(
            router.invalidate_prefix_scope(fifo_scope).unwrap(),
            PrefixCacheEviction {
                entries: 1,
                weight: 64,
            }
        );

        let weight_scope = PrefixSharingScope::new(12).unwrap();
        before = model_tokens(&router);
        let long = run(&router, &prompt_long, weight_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 321);
        before = after;
        let long_hit = run(&router, &prompt_long, weight_scope);
        after = model_tokens(&router);
        assert_eq!(after - before, 1);
        assert_eq!(long_hit, long);
        before = after;
        let _ = run(&router, &prompt_short, weight_scope);
        after = model_tokens(&router);
        assert_eq!(
            after - before,
            257,
            "256-token checkpoint must be evicted by semantic-weight pressure"
        );
        assert_eq!(
            router.invalidate_prefix_scope(weight_scope).unwrap(),
            PrefixCacheEviction {
                entries: 2,
                weight: 448,
            }
        );
        assert_eq!(
            router.invalidate_prefix_scope(weight_scope).unwrap(),
            PrefixCacheEviction::default()
        );
        router.shutdown().unwrap();
        println!(
            "P12_SOURCE_PAGED_PREFIX_EVICTION_PASS entry_bound=2 weight_bound=512 fifo_oldest_miss=true newer_hit=true revoke_entries=2 revoke_weight=128 revoked_scope_miss=true long_prompt_tokens=321 long_hit_model_tokens=1 weight_evicted_256=true final_revoke_entries=2 final_revoke_weight=448 final_shutdown=true"
        );
    }

    #[test]
    #[ignore = "approved original HF cache; exact contiguous-vs-paged CUDA singleton all-state-domain comparison"]
    fn original_source_b1_paged_matches_contiguous_at_page_boundaries() {
        let (parts, page_tokens, max_append_tokens, mut tokens) =
            paged_source_fixture("paged-source-differential");
        let needed = page_tokens * 2 + 1;
        tokens.truncate(needed);

        let run = |policy| {
            ResidentModel::load_cuda_forward_source_with_state_policy(
                Device::Cuda(Default::default()),
                parts.source.plan.source().clone(),
                parts.source.clone(),
                parts.source.plan.plan().clone(),
                1,
                0,
                PROVIDER_BYTES,
                DEVICE_BYTES,
                policy,
                None,
            )
        };
        let (expected, contiguous_identity) = block_on(async {
            let model = run(CudaSourceStatePolicy::Contiguous).await.unwrap();
            let checkpoints = trajectory(&model, &parts.source.plan, &tokens, page_tokens)
                .await
                .unwrap();
            let identity = model.state_identity().unwrap();
            model.shutdown().await.unwrap();
            (checkpoints, identity)
        });
        let (actual, paged_identity) = block_on(async {
            let model = run(CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            })
            .await
            .unwrap();
            let checkpoints = trajectory(&model, &parts.source.plan, &tokens, page_tokens)
                .await
                .unwrap();
            let identity = model.state_identity().unwrap();
            model.shutdown().await.unwrap();
            (checkpoints, identity)
        });
        assert_eq!(contiguous_identity.source(), paged_identity.source());
        assert_eq!(contiguous_identity.geometry(), paged_identity.geometry());
        assert_ne!(contiguous_identity, paged_identity);
        assert_eq!(expected.len(), 4);
        assert_eq!(actual.len(), 4);
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            exact_checkpoint(
                expected,
                actual,
                &format!("paged singleton checkpoint {index}"),
            );
        }
        assert_eq!(
            actual.iter().map(|c| c.frontier).collect::<Vec<_>>(),
            vec![
                (page_tokens - 1) as u64,
                page_tokens as u64,
                (page_tokens + 1) as u64,
                (page_tokens * 2 + 1) as u64,
            ]
        );
        println!(
            "P12_SOURCE_PAGED_B1_PASS page_tokens={page_tokens} max_append_tokens={max_append_tokens} frontiers={},{},{},{} full_vocab=exact all_state_domains=exact contiguous_reference=true final_shutdown=true",
            page_tokens - 1,
            page_tokens,
            page_tokens + 1,
            page_tokens * 2 + 1,
        );
    }

    #[test]
    #[ignore = "approved original HF cache; paged CUDA singleton cancel/discard/restore lifecycle"]
    fn original_source_b1_paged_lifecycle_rolls_back_exactly() {
        let (parts, page_tokens, max_append_tokens, tokens) =
            paged_source_fixture("paged-source-lifecycle");
        block_on(async {
            let model = ResidentModel::load_cuda_forward_source_with_state_policy(
                Device::Cuda(Default::default()),
                parts.source.plan.source().clone(),
                parts.source.clone(),
                parts.source.plan.plan().clone(),
                1,
                0,
                PROVIDER_BYTES,
                DEVICE_BYTES,
                CudaSourceStatePolicy::Paged {
                    page_tokens,
                    max_append_tokens,
                },
                None,
            )
            .await
            .unwrap();
            let mut session = model.session().await.unwrap();
            let root0 = session.binding();
            let initial = session.snapshot().unwrap();
            let initial_domains = initial.read_domains().await.unwrap().to_vec();

            let first_ids = Arc::<[u32]>::from(tokens[..page_tokens - 1].to_vec());
            let first = parts
                .source
                .plan
                .bind_invocation(InvocationPhase::Prefill, 0, first_ids, true)
                .unwrap();

            let pre_cancel = Arc::new(AtomicBool::new(true));
            assert!(matches!(
                session
                    .prepare(
                        first.bound_invocation(),
                        InvocationPhase::Prefill,
                        ResidentOutput::Logits,
                        Some(pre_cancel),
                    )
                    .await
                    .unwrap(),
                ResidentTextPreparation::CancelledBeforeLaunch
            ));
            assert_eq!(session.binding(), root0);
            exact_domains(
                &initial_domains,
                &session.snapshot().unwrap().read_domains().await.unwrap(),
                "paged prelaunch cancel",
            );

            let abandoned = session
                .prepare(
                    first.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let ResidentTextPreparation::Ready(abandoned) = abandoned else {
                panic!("paged abandoned candidate was not Ready");
            };
            assert_eq!(session.binding(), root0);
            drop(abandoned);
            assert_eq!(session.binding(), root0);
            exact_domains(
                &initial_domains,
                &session.snapshot().unwrap().read_domains().await.unwrap(),
                "paged abandoned Ready",
            );

            let cancel_after_ready = Arc::new(AtomicBool::new(false));
            let pending = session
                .prepare(
                    first.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    Some(cancel_after_ready.clone()),
                )
                .await
                .unwrap();
            let ResidentTextPreparation::Ready(pending) = pending else {
                panic!("paged cancel-after-Ready candidate was not Ready");
            };
            cancel_after_ready.store(true, Ordering::Release);
            assert!(matches!(
                session.commit(pending).unwrap(),
                ResidentTextPublication::Discarded(_)
            ));
            assert_eq!(session.binding(), root0);
            exact_domains(
                &initial_domains,
                &session.snapshot().unwrap().read_domains().await.unwrap(),
                "paged cancel after Ready",
            );

            let pending = session
                .prepare(
                    first.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let ResidentTextPreparation::Ready(pending) = pending else {
                panic!("paged first committed candidate was not Ready");
            };
            assert!(matches!(
                session.commit(pending).unwrap(),
                ResidentTextPublication::Committed(_)
            ));
            assert_eq!(session.binding().frontier(), (page_tokens - 1) as u64);
            let before_boundary = session.snapshot().unwrap();
            let before_boundary_domains = before_boundary.read_domains().await.unwrap().to_vec();

            let boundary = parts
                .source
                .plan
                .bind_invocation(
                    InvocationPhase::Prefill,
                    page_tokens - 1,
                    Arc::<[u32]>::from(vec![tokens[page_tokens - 1]]),
                    true,
                )
                .unwrap();
            let pending = session
                .prepare(
                    boundary.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let ResidentTextPreparation::Ready(pending) = pending else {
                panic!("paged boundary candidate was not Ready");
            };
            assert!(matches!(
                session.commit(pending).unwrap(),
                ResidentTextPublication::Committed(_)
            ));
            assert_eq!(session.binding().frontier(), page_tokens as u64);
            let at_boundary = session.snapshot().unwrap();
            let at_boundary_domains = at_boundary.read_domains().await.unwrap().to_vec();

            let beyond = parts
                .source
                .plan
                .bind_invocation(
                    InvocationPhase::Prefill,
                    page_tokens,
                    Arc::<[u32]>::from(vec![tokens[page_tokens]]),
                    true,
                )
                .unwrap();
            let pending = session
                .prepare(
                    beyond.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let ResidentTextPreparation::Ready(pending) = pending else {
                panic!("paged post-boundary candidate was not Ready");
            };
            assert!(matches!(
                session.commit(pending).unwrap(),
                ResidentTextPublication::Committed(_)
            ));
            assert_eq!(session.binding().frontier(), (page_tokens + 1) as u64);

            let before_restore = session.binding();
            session.restore(&before_boundary).unwrap();
            exact_restore_publication(
                before_restore,
                session.binding(),
                before_boundary.binding(),
                "paged restore before boundary",
            );
            exact_domains(
                &before_boundary_domains,
                &session.snapshot().unwrap().read_domains().await.unwrap(),
                "paged restore before boundary",
            );
            let before_restore = session.binding();
            session.restore(&at_boundary).unwrap();
            exact_restore_publication(
                before_restore,
                session.binding(),
                at_boundary.binding(),
                "paged restore at boundary",
            );
            exact_domains(
                &at_boundary_domains,
                &session.snapshot().unwrap().read_domains().await.unwrap(),
                "paged restore at boundary",
            );

            drop(initial);
            drop(before_boundary);
            drop(at_boundary);
            drop(session);
            model.shutdown().await.unwrap();
            println!(
                "P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens={page_tokens} cold_load=true prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true restore_before_boundary=true restore_at_boundary=true all_state_domains=exact final_shutdown=true"
            );
        });
    }

    #[cfg(feature = "qwen35-cuda-paged-cancellation-qualification")]
    #[test]
    #[ignore = "approved original HF cache; actual resident page-pool pressure around shared prefix"]
    fn original_source_paged_prefix_survives_external_page_pressure() {
        let (parts, page_tokens, max_append_tokens, _) =
            paged_source_fixture("paged-source-prefix-page-pressure");
        assert!(max_append_tokens >= 64);
        let tokens = parts
            .tokenizer
            .prompt_tokens(&" a".repeat(65), false)
            .unwrap();
        assert_eq!(tokens.len(), 65);
        let model = block_on(ResidentModel::load_cuda_forward_source_with_state_policy(
            Device::Cuda(Default::default()),
            parts.source.plan.source().clone(),
            parts.source.clone(),
            parts.source.plan.plan().clone(),
            2,
            0,
            PROVIDER_BYTES,
            DEVICE_BYTES,
            CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            },
            None,
        ))
        .unwrap();

        block_on(async {
            let prefix = 64usize;
            let prefix_call = parts
                .source
                .plan
                .bind_invocation(
                    InvocationPhase::Prefill,
                    0,
                    Arc::<[u32]>::from(tokens[..prefix].to_vec()),
                    false,
                )
                .unwrap();
            let tail_call = parts
                .source
                .plan
                .bind_invocation(
                    InvocationPhase::Prefill,
                    prefix,
                    Arc::<[u32]>::from(vec![tokens[prefix]]),
                    true,
                )
                .unwrap();

            let mut producer = model.session().await.unwrap();
            let ResidentTextPreparation::Ready(pending) = producer
                .prepare(
                    prefix_call.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::StateOnly,
                    None,
                )
                .await
                .unwrap()
            else {
                panic!("prefix producer admission");
            };
            let ResidentTextPublication::Committed(read) = producer.commit(pending).unwrap() else {
                panic!("prefix producer publication");
            };
            drop(read);
            let cached = producer.snapshot().unwrap();
            assert_eq!(cached.binding().frontier(), prefix as u64);
            drop(producer);

            let baseline_logits = {
                let mut baseline = model.session().await.unwrap();
                baseline.fork_shared_prefix(&cached).unwrap();
                let ResidentTextPreparation::Ready(pending) = baseline
                    .prepare(
                        tail_call.bound_invocation(),
                        InvocationPhase::Prefill,
                        ResidentOutput::Logits,
                        None,
                    )
                    .await
                    .unwrap()
                else {
                    panic!("baseline tail admission");
                };
                let ResidentTextPublication::Committed(read) = baseline.commit(pending).unwrap()
                else {
                    panic!("baseline tail publication");
                };
                let logits = read.logits.clone();
                drop(read);
                drop(baseline);
                logits
            };

            let mut pressured = model.session().await.unwrap();
            pressured.fork_shared_prefix(&cached).unwrap();
            let root = pressured.binding();
            let before = model.diagnostic_paged_pool_snapshot().unwrap();
            let occupied = |allocated: usize, idle: usize| {
                allocated.checked_sub(idle).expect("paged pool occupancy")
            };
            let before_occupied = occupied(before.allocated_pages, before.idle_pages);
            let available = before
                .max_pages
                .checked_sub(before_occupied)
                .expect("paged pool available pages");
            assert!(available > 0, "qualifier requires external page capacity");
            let pressure = model.diagnostic_hold_paged_pages(available).unwrap();
            let full = model.diagnostic_paged_pool_snapshot().unwrap();
            assert_eq!(full.allocated_pages, full.max_pages);
            assert_eq!(full.idle_pages, 0);
            assert_eq!(
                occupied(full.allocated_pages, full.idle_pages),
                full.max_pages
            );

            assert!(matches!(
                pressured
                    .prepare(
                        tail_call.bound_invocation(),
                        InvocationPhase::Prefill,
                        ResidentOutput::Logits,
                        None,
                    )
                    .await
                    .unwrap(),
                ResidentTextPreparation::Capacity
            ));
            assert_eq!(pressured.binding(), root, "page pressure mutated root");
            let refused = model.diagnostic_paged_pool_snapshot().unwrap();
            assert_eq!(
                occupied(refused.allocated_pages, refused.idle_pages),
                refused.max_pages,
                "pressure pages were released during refusal"
            );
            assert_eq!(
                refused.vector_allocated_sets - refused.vector_idle_sets,
                before.vector_allocated_sets - before.vector_idle_sets,
                "refused vector candidate leaked"
            );
            assert_eq!(
                refused.staging_allocated_sets - refused.staging_idle_sets,
                before.staging_allocated_sets - before.staging_idle_sets,
                "refused staging candidate leaked"
            );

            drop(pressure);
            let released = model.diagnostic_paged_pool_snapshot().unwrap();
            assert_eq!(
                occupied(released.allocated_pages, released.idle_pages),
                before_occupied,
                "external page pressure did not retire"
            );

            let ResidentTextPreparation::Ready(pending) = pressured
                .prepare(
                    tail_call.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap()
            else {
                panic!("tail did not recover after pressure release");
            };
            let ResidentTextPublication::Committed(read) = pressured.commit(pending).unwrap()
            else {
                panic!("tail publication after pressure release");
            };
            exact_words(
                &baseline_logits,
                &read.logits,
                "external page pressure recovered logits",
            );
            assert_eq!(pressured.binding().frontier(), 65);
            drop(read);
            drop(pressured);
            drop(cached);

            let final_pool = model.diagnostic_paged_pool_snapshot().unwrap();
            assert_eq!(
                final_pool.allocated_pages, final_pool.idle_pages,
                "page pool did not fully retire after pressure qualifier"
            );
            model.shutdown().await.unwrap();
            println!(
                "P12_SOURCE_PAGED_PREFIX_PAGE_PRESSURE_PASS prefix_tokens=64 tail_tokens=1 page_tokens={page_tokens} max_append_tokens={max_append_tokens} max_pages={} pressure_pages={available} prelaunch_capacity=true root_unchanged=true vector_staging_unwound=true cached_root_survived=true recovered_logits=exact final_pages_idle=true final_shutdown=true",
                before.max_pages
            );
        });
    }

    #[cfg(feature = "qwen35-cuda-paged-cancellation-qualification")]
    #[test]
    #[ignore = "approved original HF cache; real submitted paged CUDA cancellation before Ready"]
    fn original_source_b1_paged_inflight_cancel_retires_exactly() {
        let (parts, page_tokens, max_append_tokens, tokens) =
            paged_source_fixture("paged-source-inflight-cancel");
        block_on(async {
            let model = ResidentModel::load_cuda_forward_source_with_state_policy(
                Device::Cuda(Default::default()),
                parts.source.plan.source().clone(),
                parts.source.clone(),
                parts.source.plan.plan().clone(),
                1,
                0,
                PROVIDER_BYTES,
                DEVICE_BYTES,
                CudaSourceStatePolicy::Paged {
                    page_tokens,
                    max_append_tokens,
                },
                None,
            )
            .await
            .unwrap();
            let mut session = model.session().await.unwrap();
            let root0 = session.binding();
            let initial = session.snapshot().unwrap();
            let initial_domains = initial.read_domains().await.unwrap().to_vec();
            let before = model.diagnostic_paged_pool_snapshot().unwrap();
            let occupied = |allocated: usize, idle: usize| {
                allocated.checked_sub(idle).expect("pool occupancy")
            };

            let first_ids = Arc::<[u32]>::from(tokens[..page_tokens - 1].to_vec());
            let first = parts
                .source
                .plan
                .bind_invocation(InvocationPhase::Prefill, 0, first_ids, true)
                .unwrap();
            let cancelled = Arc::new(AtomicBool::new(false));
            model.diagnostic_pause_next_launch().unwrap();
            let mut future = Box::pin(session.prepare(
                first.bound_invocation(),
                InvocationPhase::Prefill,
                ResidentOutput::Logits,
                Some(cancelled.clone()),
            ));
            let submitted_handles = std::future::poll_fn(|cx| {
                if future.as_mut().poll(cx).is_ready() {
                    panic!("paged cancellation qualifier completed before launch cut");
                }
                let handles = model.diagnostic_launch_handles();
                if handles == 0 {
                    Poll::Pending
                } else {
                    Poll::Ready(handles)
                }
            })
            .await;
            let during = model.diagnostic_paged_pool_snapshot().unwrap();
            assert!(submitted_handles > 0, "no real CUDA submission observed");
            assert_eq!(model.diagnostic_pending_events().unwrap(), 1);
            assert_eq!(model.diagnostic_quarantined_handles().unwrap(), 0);
            assert_eq!(model.diagnostic_quarantined_completion_owners().unwrap(), 0);
            assert!(
                occupied(during.vector_allocated_sets, during.vector_idle_sets)
                    > occupied(before.vector_allocated_sets, before.vector_idle_sets),
                "submitted Vector candidate recycled before completion"
            );
            assert!(
                occupied(during.staging_allocated_sets, during.staging_idle_sets)
                    > occupied(before.staging_allocated_sets, before.staging_idle_sets),
                "submitted staging candidate recycled before completion"
            );
            assert!(
                occupied(during.allocated_pages, during.idle_pages)
                    > occupied(before.allocated_pages, before.idle_pages),
                "submitted page candidate recycled before completion"
            );

            cancelled.store(true, Ordering::Release);
            model.diagnostic_resume_launch();
            assert!(matches!(
                future.await,
                Err(crate::Error::State(reason)) if reason == "ordinary CUDA request cancelled"
            ));
            assert_eq!(
                session.binding(),
                root0,
                "in-flight cancel published a root"
            );
            let after_cancel = session.snapshot().unwrap();
            exact_domains(
                &initial_domains,
                &after_cancel.read_domains().await.unwrap(),
                "paged in-flight cancel",
            );
            drop(after_cancel);
            assert_eq!(model.diagnostic_pending_events().unwrap(), 0);
            assert!(model.diagnostic_service_usable());
            assert_eq!(model.diagnostic_quarantined_handles().unwrap(), 0);
            assert_eq!(model.diagnostic_quarantined_completion_owners().unwrap(), 0);
            let after = model.diagnostic_paged_pool_snapshot().unwrap();
            assert_eq!(
                occupied(after.vector_allocated_sets, after.vector_idle_sets),
                occupied(before.vector_allocated_sets, before.vector_idle_sets),
                "cancelled Vector candidate did not retire exactly"
            );
            assert_eq!(
                occupied(after.staging_allocated_sets, after.staging_idle_sets),
                occupied(before.staging_allocated_sets, before.staging_idle_sets),
                "cancelled staging candidate did not retire exactly"
            );
            assert_eq!(
                occupied(after.allocated_pages, after.idle_pages),
                occupied(before.allocated_pages, before.idle_pages),
                "cancelled page candidate did not retire exactly"
            );

            cancelled.store(false, Ordering::Release);
            let replay = session
                .prepare(
                    first.bound_invocation(),
                    InvocationPhase::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let ResidentTextPreparation::Ready(replay) = replay else {
                panic!("valid work after in-flight cancellation was not Ready");
            };
            let ResidentTextPublication::Committed(read) = session.commit(replay).unwrap() else {
                panic!("valid work after in-flight cancellation was not committed");
            };
            assert_eq!(session.binding().frontier(), (page_tokens - 1) as u64);
            assert_eq!(model.diagnostic_pending_events().unwrap(), 0);
            drop(read);
            drop(initial);
            drop(session);
            model.shutdown().await.unwrap();
            println!(
                "P12_SOURCE_PAGED_INFLIGHT_CANCEL_PASS page_tokens={page_tokens} max_append_tokens={max_append_tokens} submitted_handles={submitted_handles} cancelled_before_ready=true root_unchanged=true all_state_domains=exact vector_held=true staging_held=true pages_held=true completion=known retired_exact=true replay=true final_shutdown=true"
            );
        });
    }

    #[cfg(feature = "qwen35-cuda-paged-cancellation-qualification")]
    #[test]
    #[ignore = "approved original HF cache; dropped submitted paged CUDA work must quarantine"]
    fn original_source_b1_paged_unknown_completion_quarantines_exactly() {
        let (parts, page_tokens, max_append_tokens, tokens) =
            paged_source_fixture("paged-source-unknown-completion");
        block_on(async {
            let model = ResidentModel::load_cuda_forward_source_with_state_policy(
                Device::Cuda(Default::default()),
                parts.source.plan.source().clone(),
                parts.source.clone(),
                parts.source.plan.plan().clone(),
                1,
                0,
                PROVIDER_BYTES,
                DEVICE_BYTES,
                CudaSourceStatePolicy::Paged {
                    page_tokens,
                    max_append_tokens,
                },
                None,
            )
            .await
            .unwrap();
            let session = model.session().await.unwrap();
            let root0 = session.binding();
            let initial = session.snapshot().unwrap();
            let initial_domains = initial.read_domains().await.unwrap().to_vec();
            assert_eq!(initial_domains.len(), 42);
            drop(initial);
            let before = model.diagnostic_paged_pool_snapshot().unwrap();
            let occupied = |allocated: usize, idle: usize| {
                allocated.checked_sub(idle).expect("pool occupancy")
            };

            let first_ids = Arc::<[u32]>::from(tokens[..page_tokens - 1].to_vec());
            let first = parts
                .source
                .plan
                .bind_invocation(InvocationPhase::Prefill, 0, first_ids, true)
                .unwrap();
            model.diagnostic_pause_next_launch().unwrap();
            let mut future = Box::pin(session.prepare(
                first.bound_invocation(),
                InvocationPhase::Prefill,
                ResidentOutput::Logits,
                None,
            ));
            let submitted_handles = std::future::poll_fn(|cx| {
                if future.as_mut().poll(cx).is_ready() {
                    panic!("paged unknown qualifier completed before launch cut");
                }
                let handles = model.diagnostic_launch_handles();
                if handles == 0 {
                    Poll::Pending
                } else {
                    Poll::Ready(handles)
                }
            })
            .await;
            let during = model.diagnostic_paged_pool_snapshot().unwrap();
            assert!(submitted_handles > 0, "no real CUDA submission observed");
            assert_eq!(model.diagnostic_pending_events().unwrap(), 1);
            assert!(
                occupied(during.vector_allocated_sets, during.vector_idle_sets)
                    > occupied(before.vector_allocated_sets, before.vector_idle_sets)
                    && occupied(during.staging_allocated_sets, during.staging_idle_sets)
                        > occupied(before.staging_allocated_sets, before.staging_idle_sets)
                    && occupied(during.allocated_pages, during.idle_pages)
                        > occupied(before.allocated_pages, before.idle_pages),
                "submitted paged resources were not held at the launch cut"
            );

            drop(future);
            model.diagnostic_resume_launch();
            assert_eq!(
                session.binding(),
                root0,
                "unknown completion published a root"
            );
            assert!(!model.diagnostic_service_usable());
            assert_eq!(model.diagnostic_pending_events().unwrap(), 1);
            assert!(
                model.diagnostic_quarantined_handles().unwrap() >= submitted_handles,
                "unknown completion did not retain submitted handles"
            );
            assert!(
                model.diagnostic_quarantined_completion_owners().unwrap() > 0,
                "unknown completion lost source completion authority"
            );
            let after = model.diagnostic_paged_pool_snapshot().unwrap();
            assert!(
                occupied(after.vector_allocated_sets, after.vector_idle_sets)
                    >= occupied(during.vector_allocated_sets, during.vector_idle_sets)
                    && occupied(after.staging_allocated_sets, after.staging_idle_sets)
                        >= occupied(during.staging_allocated_sets, during.staging_idle_sets)
                    && occupied(after.allocated_pages, after.idle_pages)
                        >= occupied(during.allocated_pages, during.idle_pages),
                "unknown-completion resources were recycled"
            );
            println!(
                "P12_SOURCE_PAGED_UNKNOWN_COMPLETION_PASS page_tokens={page_tokens} max_append_tokens={max_append_tokens} submitted_handles={submitted_handles} root_binding_unchanged=true service_fenced=true pending_event=true handles_quarantined=true completion_owners_retained=true pools_not_recycled=true recovery=false"
            );
            drop(session);
            drop(model);
        });
    }
}
