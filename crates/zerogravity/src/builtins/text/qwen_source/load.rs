use super::*;
use crate::{
    accelerator::program::SourceNumericalContract,
    cache::{CacheOptions, CacheStore},
    gpu::{Device, ResidentModel},
    qwen35_source::{self, Inputs},
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission},
    tokenize::QwenTokenizer,
};
use futures_lite::future::block_on;
use std::collections::BTreeMap;
use zerogravity_engine::{model_instance::execution::*, router::Router};

#[derive(Clone, Copy)]
pub(crate) enum SourceBackend {
    #[cfg(feature = "qwen35-hip")]
    Hip(crate::gpu::HipSourceBudget),
    #[cfg(feature = "qwen35-cuda")]
    Cuda {
        provider_bytes: usize,
        device_bytes: u64,
        state_policy: crate::gpu::CudaSourceStatePolicy,
    },
}
impl SourceBackend {
    fn validate(self) -> Result<(), ServiceError> {
        #[cfg(feature = "qwen35-cuda")]
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
        Ok(())
    }
    fn contract(self) -> SourceNumericalContract {
        match self {
            #[cfg(feature = "qwen35-hip")]
            Self::Hip(_) => SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1,
            #[cfg(feature = "qwen35-cuda")]
            Self::Cuda { .. } => SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        }
    }
}

fn validate_aliases(aliases: &[String]) -> Result<(), ServiceError> {
    if aliases.is_empty()
        || aliases.len() > 16
        || aliases.iter().enumerate().any(|(i, alias)| {
            alias.is_empty()
                || alias.len() > 96
                || !alias
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                || aliases[..i].contains(alias)
        })
    {
        return Err(ServiceError::new(
            400,
            "invalid_model",
            "source model requires 1..16 unique valid aliases",
        ));
    }
    Ok(())
}
pub(crate) fn load_router(
    aliases: Vec<String>,
    cache: CacheOptions,
    snapshot: String,
    profile: QwenSourceProfile,
    backend: SourceBackend,
) -> Result<Router, ServiceError> {
    profile.validate()?;
    backend.validate()?;
    #[cfg(feature = "qwen35-cuda")]
    if let SourceBackend::Cuda {
        state_policy:
            crate::gpu::CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            },
        ..
    } = backend
        && (profile.native_rows != 0
            || page_tokens == 0
            || max_append_tokens == 0
            || page_tokens > profile.context
            || max_append_tokens > profile.context)
    {
        return Err(ServiceError::new(
            400,
            "invalid_source_state_policy",
            "paged CUDA state requires B1 and positive page/append tokens within context",
        ));
    }
    #[cfg(feature = "qwen35-hip")]
    if profile.native_rows != 0 && matches!(backend, SourceBackend::Hip(_)) {
        return Err(ServiceError::new(
            400,
            "unsupported_source_batch",
            "source native_rows currently requires CUDA",
        ));
    }
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
    // Exact original source/template/owner. No qualification loader or program
    // reconstruction and no alternate scheduler/publisher.
    let resident = match backend {
        #[cfg(feature = "qwen35-hip")]
        SourceBackend::Hip(budget) => {
            block_on(ResidentModel::load_ordinary_cpu_source_with_budget(
                Device::Hip(Default::default()),
                source.plan.source().clone(),
                source.clone(),
                source.plan.plan().clone(),
                profile.active,
                budget,
                None,
            ))
        }
        #[cfg(feature = "qwen35-cuda")]
        SourceBackend::Cuda {
            provider_bytes,
            device_bytes,
            state_policy,
        } => block_on(ResidentModel::load_ordinary_cuda_source_with_state_policy(
            Device::Cuda(Default::default()),
            source.plan.source().clone(),
            source.clone(),
            source.plan.plan().clone(),
            profile.active,
            profile.native_rows,
            provider_bytes,
            device_bytes,
            state_policy,
            None,
        )),
    }
    .map_err(ServiceError::inference)?;
    let model = Arc::new(Model {
        resident: Mutex::new(Some(resident)),
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
            policy: if profile.native_rows == 0 {
                WorkPolicy::Sequential
            } else {
                WorkPolicy::Waves(WavePolicy {
                    max_members: profile.native_rows,
                    max_tokens: profile.native_rows * 64,
                    max_scratch: 256 * 1024 * 1024,
                    fairness: Fairness {
                        round_tokens: profile.native_rows * 64 + profile.active,
                        prefill_age_rounds: 2,
                    },
                    prefill_quantum: 64,
                    unequal_prefill: false,
                    unequal_decode: true,
                    native_rows: profile.native_rows,
                    cohorts: Cohorts::Native {
                        widths: if profile.native_rows == 8 {
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
    contract: SourceNumericalContract,
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

#[cfg(all(test, feature = "qwen35-hip"))]
mod tests {
    use super::*;
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
            SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1,
        )
        .unwrap();
        let plan = &parts.source.plan;
        assert_eq!(
            plan.source_seal().numerical_contract(),
            SourceNumericalContract::TorchCpuBf16Avx2MixedSourceV1
        );
        assert_eq!(plan.context_capacity_tokens(), 514);
        assert_eq!(plan.max_chunk_rows(), 64);
        let inventory = ResidentModel::ordinary_cpu_source_preflight(
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
                crate::accelerator::program::SourceCallMode::Prefill,
                0,
                tokens.into(),
                true,
            )
            .unwrap();
        let materialized = call.materialize_source_plan().unwrap();
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

#[cfg(all(test, feature = "qwen35-cuda"))]
mod cuda_paged_tests {
    use super::*;
    use crate::{
        accelerator::program::SourceCallMode,
        gpu::{
            CudaSourceStatePolicy, ResidentOutput, SourceCallModel, SourceCallPreparation,
            SourceCallPublication,
        },
    };
    use std::sync::atomic::{AtomicBool, Ordering};

    const MIB: usize = 1024 * 1024;
    const PROVIDER_BYTES: usize = 4608 * MIB;
    const DEVICE_BYTES: u64 = 5120 * MIB as u64;

    #[derive(Debug)]
    struct Checkpoint {
        frontier: u64,
        logits: Vec<f32>,
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
        assert_eq!(expected.domains.len(), 42, "{label} expected all42");
        assert_eq!(actual.domains.len(), 42, "{label} actual all42");
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
        assert_eq!(expected.len(), 42, "{label} expected all42");
        assert_eq!(actual.len(), 42, "{label} actual all42");
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
            SourceNumericalContract::ReferenceCudaSm86Bf16V1,
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
        model: &Arc<SourceCallModel>,
        plan: &crate::qwen35::plan::OrdinaryQwenPlan,
        tokens: &[u32],
        page_tokens: usize,
    ) -> crate::Result<Vec<Checkpoint>> {
        let mut session = model.session().await?;
        let chunks = [page_tokens - 1, 1, 1, page_tokens];
        let mut offset = 0usize;
        let mut checkpoints = Vec::with_capacity(chunks.len());
        for rows in chunks {
            let ids = Arc::<[u32]>::from(tokens[offset..offset + rows].to_vec());
            let invocation = plan.bind_invocation(SourceCallMode::Prefill, offset, ids, true)?;
            let pending = session
                .prepare(
                    invocation.bound_invocation(),
                    SourceCallMode::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await?;
            let SourceCallPreparation::Ready(pending) = pending else {
                return Err(crate::Error::State(
                    "paged source differential candidate refused".into(),
                ));
            };
            let SourceCallPublication::Committed(read) = session.commit(pending)? else {
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

    #[test]
    #[ignore = "approved original HF cache; exact contiguous-vs-paged CUDA B1 all42"]
    fn original_source_b1_paged_matches_contiguous_at_page_boundaries() {
        let (parts, page_tokens, max_append_tokens, mut tokens) =
            paged_source_fixture("paged-source-differential");
        let needed = page_tokens * 2 + 1;
        tokens.truncate(needed);

        let run = |policy| {
            ResidentModel::load_ordinary_cuda_source_with_state_policy(
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
        let expected = block_on(async {
            let model = run(CudaSourceStatePolicy::Contiguous).await.unwrap();
            let checkpoints = trajectory(&model, &parts.source.plan, &tokens, page_tokens)
                .await
                .unwrap();
            model.shutdown().await.unwrap();
            checkpoints
        });
        let actual = block_on(async {
            let model = run(CudaSourceStatePolicy::Paged {
                page_tokens,
                max_append_tokens,
            })
            .await
            .unwrap();
            let checkpoints = trajectory(&model, &parts.source.plan, &tokens, page_tokens)
                .await
                .unwrap();
            model.shutdown().await.unwrap();
            checkpoints
        });
        assert_eq!(expected.len(), 4);
        assert_eq!(actual.len(), 4);
        for (index, (expected, actual)) in expected.iter().zip(&actual).enumerate() {
            exact_checkpoint(expected, actual, &format!("paged B1 checkpoint {index}"));
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
            "P12_SOURCE_PAGED_B1_PASS page_tokens={page_tokens} max_append_tokens={max_append_tokens} frontiers={},{},{},{} full_vocab=exact all42=exact contiguous_reference=true final_shutdown=true",
            page_tokens - 1,
            page_tokens,
            page_tokens + 1,
            page_tokens * 2 + 1,
        );
    }

    #[test]
    #[ignore = "approved original HF cache; paged CUDA B1 cancel/discard/restore lifecycle"]
    fn original_source_b1_paged_lifecycle_rolls_back_exactly() {
        let (parts, page_tokens, max_append_tokens, tokens) =
            paged_source_fixture("paged-source-lifecycle");
        eprintln!("PAGED_LIFECYCLE_STAGE source_ready page_tokens={page_tokens}");
        let contiguous_warmup =
            std::env::var("ZG_QWEN_PAGED_CONTIGUOUS_WARMUP").as_deref() == Ok("1");
        block_on(async {
            if contiguous_warmup {
                let warm = ResidentModel::load_ordinary_cuda_source_with_state_policy(
                    Device::Cuda(Default::default()),
                    parts.source.plan.source().clone(),
                    parts.source.clone(),
                    parts.source.plan.plan().clone(),
                    1,
                    0,
                    PROVIDER_BYTES,
                    DEVICE_BYTES,
                    CudaSourceStatePolicy::Contiguous,
                    None,
                )
                .await
                .unwrap();
                eprintln!("PAGED_LIFECYCLE_STAGE contiguous_warm_model_loaded");
                let mut warm_session = warm.session().await.unwrap();
                let warm_invocation = parts
                    .source
                    .plan
                    .bind_invocation(
                        SourceCallMode::Prefill,
                        0,
                        Arc::<[u32]>::from(vec![tokens[0]]),
                        true,
                    )
                    .unwrap();
                let warm_pending = warm_session
                    .prepare(
                        warm_invocation.bound_invocation(),
                        SourceCallMode::Prefill,
                        ResidentOutput::Logits,
                        None,
                    )
                    .await
                    .unwrap();
                let SourceCallPreparation::Ready(warm_pending) = warm_pending else {
                    panic!("contiguous warmup candidate was not Ready");
                };
                assert!(matches!(
                    warm_session.commit(warm_pending).unwrap(),
                    SourceCallPublication::Committed(_)
                ));
                drop(warm_session);
                warm.shutdown().await.unwrap();
                eprintln!("PAGED_LIFECYCLE_STAGE contiguous_warm_complete");
            }
            let model = ResidentModel::load_ordinary_cuda_source_with_state_policy(
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
            eprintln!("PAGED_LIFECYCLE_STAGE model_loaded");
            let mut session = model.session().await.unwrap();
            eprintln!("PAGED_LIFECYCLE_STAGE session_ready");
            let root0 = session.binding();
            let initial = session.snapshot().unwrap();
            eprintln!("PAGED_LIFECYCLE_STAGE initial_snapshot_ready");
            let initial_domains = initial.read_domains().await.unwrap().to_vec();
            eprintln!("PAGED_LIFECYCLE_STAGE initial_snapshot_read");

            let first_ids = Arc::<[u32]>::from(tokens[..page_tokens - 1].to_vec());
            let first = parts
                .source
                .plan
                .bind_invocation(SourceCallMode::Prefill, 0, first_ids, true)
                .unwrap();

            let pre_cancel = Arc::new(AtomicBool::new(true));
            assert!(matches!(
                session
                    .prepare(
                        first.bound_invocation(),
                        SourceCallMode::Prefill,
                        ResidentOutput::Logits,
                        Some(pre_cancel),
                    )
                    .await
                    .unwrap(),
                SourceCallPreparation::CancelledBeforeLaunch
            ));
            eprintln!("PAGED_LIFECYCLE_STAGE prelaunch_cancel_ready");
            assert_eq!(session.binding(), root0);
            exact_domains(
                &initial_domains,
                &session.snapshot().unwrap().read_domains().await.unwrap(),
                "paged prelaunch cancel",
            );

            let abandoned = session
                .prepare(
                    first.bound_invocation(),
                    SourceCallMode::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let SourceCallPreparation::Ready(abandoned) = abandoned else {
                panic!("paged abandoned candidate was not Ready");
            };
            eprintln!("PAGED_LIFECYCLE_STAGE abandoned_ready_completed");
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
                    SourceCallMode::Prefill,
                    ResidentOutput::Logits,
                    Some(cancel_after_ready.clone()),
                )
                .await
                .unwrap();
            let SourceCallPreparation::Ready(pending) = pending else {
                panic!("paged cancel-after-Ready candidate was not Ready");
            };
            eprintln!("PAGED_LIFECYCLE_STAGE cancel_after_ready_completed");
            cancel_after_ready.store(true, Ordering::Release);
            assert!(matches!(
                session.commit(pending).unwrap(),
                SourceCallPublication::Discarded(_)
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
                    SourceCallMode::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let SourceCallPreparation::Ready(pending) = pending else {
                panic!("paged first committed candidate was not Ready");
            };
            assert!(matches!(
                session.commit(pending).unwrap(),
                SourceCallPublication::Committed(_)
            ));
            assert_eq!(session.binding().frontier(), (page_tokens - 1) as u64);
            eprintln!("PAGED_LIFECYCLE_STAGE preboundary_committed");
            let before_boundary = session.snapshot().unwrap();
            let before_boundary_domains = before_boundary.read_domains().await.unwrap().to_vec();
            eprintln!("PAGED_LIFECYCLE_STAGE preboundary_snapshot_read");

            let boundary = parts
                .source
                .plan
                .bind_invocation(
                    SourceCallMode::Prefill,
                    page_tokens - 1,
                    Arc::<[u32]>::from(vec![tokens[page_tokens - 1]]),
                    true,
                )
                .unwrap();
            let pending = session
                .prepare(
                    boundary.bound_invocation(),
                    SourceCallMode::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let SourceCallPreparation::Ready(pending) = pending else {
                panic!("paged boundary candidate was not Ready");
            };
            assert!(matches!(
                session.commit(pending).unwrap(),
                SourceCallPublication::Committed(_)
            ));
            assert_eq!(session.binding().frontier(), page_tokens as u64);
            eprintln!("PAGED_LIFECYCLE_STAGE boundary_committed");
            let at_boundary = session.snapshot().unwrap();
            let at_boundary_domains = at_boundary.read_domains().await.unwrap().to_vec();
            eprintln!("PAGED_LIFECYCLE_STAGE boundary_snapshot_read");

            let beyond = parts
                .source
                .plan
                .bind_invocation(
                    SourceCallMode::Prefill,
                    page_tokens,
                    Arc::<[u32]>::from(vec![tokens[page_tokens]]),
                    true,
                )
                .unwrap();
            let pending = session
                .prepare(
                    beyond.bound_invocation(),
                    SourceCallMode::Prefill,
                    ResidentOutput::Logits,
                    None,
                )
                .await
                .unwrap();
            let SourceCallPreparation::Ready(pending) = pending else {
                panic!("paged post-boundary candidate was not Ready");
            };
            assert!(matches!(
                session.commit(pending).unwrap(),
                SourceCallPublication::Committed(_)
            ));
            assert_eq!(session.binding().frontier(), (page_tokens + 1) as u64);
            eprintln!("PAGED_LIFECYCLE_STAGE postboundary_committed");

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
            eprintln!("PAGED_LIFECYCLE_STAGE restore_preboundary_verified");
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
            eprintln!("PAGED_LIFECYCLE_STAGE restore_boundary_verified");
            drop(session);
            model.shutdown().await.unwrap();
            eprintln!("PAGED_LIFECYCLE_STAGE shutdown_complete");
            println!(
                "P12_SOURCE_PAGED_LIFECYCLE_PASS page_tokens={page_tokens} contiguous_warmup={contiguous_warmup} prelaunch_cancel=true abandoned_ready=true cancel_after_ready=true restore_before_boundary=true restore_at_boundary=true all42=exact final_shutdown=true"
            );
        });
    }
}
