//! Authenticated short-HF source and tokenizer; the original engine owns math.
use crate::gpu::vllm_cuda_qualification as native;
use crate::qwen35::tokenize::QwenTokenizer;
use crate::resource_arbiter::{GlobalResourceLease, lease_inventory_bytes};
use crate::{Error, Result};
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::cache::TransferAdmission;
mod manifest;
mod source;
pub use source::ShortHfRouteTokenizer;
const MIB: usize = 1024 * 1024;
pub const HOST_MIB: usize = 4096;
/// Explicit CUDA short-HF HOST profile bounds; CPU and other models stay 4096.
pub const HOST_MIB_MAX: usize = 8192;
pub fn supported_host_mib(host_mib: usize) -> bool {
    matches!(host_mib, HOST_MIB | HOST_MIB_MAX)
}
/// Default finite physical-test profile; serving passes its configured allowance.
pub const PROVIDER_MIB: usize = 4608;
pub const DEVICE_MIB: usize = 5120;

/// Compiled source inventory, checked against every actual runtime source file.
/// These fields describe identity/finite input geometry, never numerical results.
pub struct ShortHfSourceMetadata {
    pub family: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub index_sha256: &'static str,
    pub manifest_sha256: &'static str,
    pub hidden_width: usize,
    pub vocabulary_size: u64,
    pub prefill_rows: [usize; 5],
    pub max_output: usize,
    pub context: usize,
    pub cached_through: usize,
}
pub struct ShortHfModel {
    resident: Arc<native::ShortHfLoaded>,
    tokenizer: QwenTokenizer,
    metadata: ShortHfSourceMetadata,
    _tokenizer: crate::tokenize_admission::Retained,
    _loader: GlobalResourceLease,
}
impl ShortHfModel {
    pub fn resident(&self) -> &Arc<native::ShortHfLoaded> {
        &self.resident
    }
    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }
    pub fn source_metadata(&self) -> &ShortHfSourceMetadata {
        &self.metadata
    }
}
fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("short-HF source load cancelled".into()));
    }
    Ok(())
}
pub fn native_artifact_root(explicit: Option<&Path>) -> Result<PathBuf> {
    let root = crate::canonical47::paths::native_root(
        explicit,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )?;
    Ok(if explicit.is_some() {
        root
    } else {
        root.with_file_name("short-hf-v1")
    })
}
pub async fn load(
    device: crate::gpu::Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ShortHfModel>> {
    load_profile(
        false,
        device,
        hf_directory,
        artifact_directory,
        host,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}
pub async fn load_b4(
    device: crate::gpu::Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ShortHfModel>> {
    load_profile(
        true,
        device,
        hf_directory,
        artifact_directory,
        host,
        provider_capacity_bytes,
        cancelled,
    )
    .await
}
async fn load_profile(
    cohort_b4: bool,
    device: crate::gpu::Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    provider_capacity_bytes: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ShortHfModel>> {
    not_cancelled(cancelled)?;
    if provider_capacity_bytes == 0 || provider_capacity_bytes > DEVICE_MIB * MIB {
        return Err(Error::Limit(
            "short-HF provider allowance exceeds physical envelope".into(),
        ));
    }
    if !matches!(device, crate::gpu::Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "short-HF CUDA source profile only".into(),
        ));
    }
    let host_limit = host.arbiter.snapshot(&host.resource)?.limit;
    if host_limit != (HOST_MIB * MIB) as u64 && host_limit != (HOST_MIB_MAX * MIB) as u64 {
        return Err(Error::Limit(
            "short-HF actual4or8GiB original host owner required".into(),
        ));
    }
    let inventory = lease_inventory_bytes(&host.owner.model, &host.owner.session, 2)?;
    let loader = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[
            crate::resource_arbiter::AllocationRequest {
                kind: crate::resource_arbiter::AllocationClass::Scratch,
                bytes: (128 * MIB) as u64,
            },
            crate::resource_arbiter::AllocationRequest {
                kind: crate::resource_arbiter::AllocationClass::Other,
                bytes: inventory,
            },
        ],
    )?;
    let bootstrap = crate::tokenize_admission::approve_arbiter(host, "short-hf-inventory")?;
    let retained = bootstrap.retain_arbiter(host, "short-hf-inventory")?;
    let opened = source::open(hf_directory, host, cancelled)?;
    let bound = opened.source.bind_full_model_source()?;
    let bundle = manifest::Bundle::open(
        artifact_directory,
        bound.model_profile(),
        cohort_b4,
        cancelled,
    )?;
    let rotary = native::qualify_short_hf_rotary_cubins(
        device.clone(),
        opened.source.clone(),
        (DEVICE_MIB * MIB) as u64,
        Some(bundle.rotary(opened.source.source(), cancelled)?),
        native::RotaryMode::Complete,
        cancelled,
        |_| Ok(()),
    )
    .await?;
    let observed = &rotary.comparison;
    if observed.native_launches != 12
        || observed.completed_readback_flights != 2
        || observed.resident_bytes != 0
        || observed.pending_events != 0
        || observed.physical_reserved_bytes != observed.backend_reserved_bytes
        || observed.backend_reserved_bytes != 512 * 1024 * 1024
        || observed.physical_fenced
    {
        return Err(Error::State(
            "short-HF original source rotary did not complete/retire".into(),
        ));
    }
    let inputs = native::ShortHfQualificationInputs {
        native: bundle.model(opened.source.source(), cancelled)?,
        rotary: Some(rotary),
    };
    let resident = if cohort_b4 {
        native::prepare_short_hf_b4(device, bound, inputs, provider_capacity_bytes, cancelled)
            .await?
    } else {
        native::prepare_short_hf(device, bound, inputs, provider_capacity_bytes, cancelled).await?
    };
    if let Err(error) = not_cancelled(cancelled) {
        resident.clone().close(None).await?;
        return Err(error);
    }
    Ok(Arc::new(ShortHfModel {
        resident,
        tokenizer: opened.tokenizer,
        metadata: opened.metadata,
        _tokenizer: retained,
        _loader: loader,
    }))
}

/// A known finite source policy, distinct from actual full-source/native
/// authentication performed by load and its original ResourceInventory.
pub struct ShortHfRouteMetadata {
    pub family: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub manifest_sha256: &'static str,
    pub provider_bytes: usize,
    /// Selected finite CUDA HOST requirement, distinct from provider/device.
    pub host_bytes: usize,
    profile: zerogravity_core::accelerator::program::FullModelProfile,
}
pub fn inspect_route(directory: &Path, native_directory: &Path) -> Result<ShortHfRouteMetadata> {
    let metadata = source::route_metadata(directory)?;
    // Metadata parser only: image payload verification remains in load.
    drop(manifest::Bundle::open(
        native_directory,
        metadata.profile,
        false,
        None,
    )?);
    Ok(metadata)
}
pub fn route_tokenizer(
    directory: &Path,
    host: &TransferAdmission,
) -> Result<ShortHfRouteTokenizer> {
    source::route_tokenizer(directory, host)
}

/// Original scoped source HOST policy shared by direct and serving frontends.
/// It is not system RAM capacity or a second physical-memory allocator.
pub fn host_admission_mib(
    host_mib: usize,
    owner: crate::resource_arbiter::ResourceOwner,
) -> Result<TransferAdmission> {
    use crate::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};
    if !supported_host_mib(host_mib) {
        return Err(Error::Limit(
            "short-HF HOST profile supports 4096 or 8192 MiB only".into(),
        ));
    }
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
    arbiter.register_matching(resource.clone(), (host_mib * MIB) as u64)?;
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner,
    })
}
/// Legacy explicit 4 GiB short-HF HOST wrapper/default.
pub fn host_admission(owner: crate::resource_arbiter::ResourceOwner) -> Result<TransferAdmission> {
    host_admission_mib(HOST_MIB, owner)
}

#[cfg(test)]
mod b4_inventory_tests {
    use super::*;
    use crate::resource_arbiter::{
        DeviceBudgetSnapshot, ResourceArbiter, ResourceOwner, lease_actual_inventory_bytes,
    };

    fn same_budget(a: &DeviceBudgetSnapshot, b: &DeviceBudgetSnapshot) -> bool {
        a.limit == b.limit
            && a.reserved == b.reserved
            && a.remaining == b.remaining
            && a.reservations == b.reservations
            && a.per_class == b.per_class
            && a.active_models == b.active_models
            && a.fenced == b.fenced
    }

    fn host(mib: u64) -> Result<TransferAdmission> {
        // Original live process arbiter/key: the tokenizer bootstrap registry is
        // process-static. Run each permanent-owner case in a fresh test process
        // (exact case filter); never replace the arbiter or clear the registry.
        let arbiter = ResourceArbiter::process_shared();
        let resource =
            crate::resource_arbiter::PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
        arbiter.register_matching(resource.clone(), mib * MIB as u64)?;
        Ok(TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("short-hf-route-test", "source-only")?,
        })
    }

    fn route_credit(
        host: &TransferAdmission,
        bootstrap: &crate::tokenize_admission::Bootstrap,
    ) -> Result<u64> {
        let plan = bootstrap.plan();
        let owner = ResourceOwner::new(&host.owner.model, "short-hf-route-inventory")?;
        let inventory = lease_actual_inventory_bytes(&host.resource, &owner, 3)?;
        plan.retained_bytes
            .checked_add(plan.request_cache_growth_bytes)
            .and_then(|n| n.checked_add(plan.serialized_vm_bytes))
            .and_then(|n| n.checked_add(inventory))
            .ok_or_else(|| Error::Limit("route tokenizer credit sum".into()))
    }

    fn bootstrap_budget(
        host: &TransferAdmission,
        bootstrap: &crate::tokenize_admission::Bootstrap,
    ) -> Result<crate::resource_arbiter::DeviceBudgetSnapshot> {
        use crate::resource_arbiter::AllocationClass;
        let plan = bootstrap.plan();
        // Independently derive B from the known plan, the exact two-request
        // lease inventory and the declared host limit/owner. Never clone the
        // observed snapshot to manufacture agreement.
        let owner = ResourceOwner::new(&host.owner.model, "short-hf-route-test-bootstrap")?;
        let inventory = lease_actual_inventory_bytes(&host.resource, &owner, 2)?;
        let reserved = plan
            .bootstrap_bytes
            .checked_add(inventory)
            .ok_or_else(|| Error::Limit("route bootstrap budget sum".into()))?;
        let mut per_class = std::collections::BTreeMap::new();
        per_class.insert(AllocationClass::ImmutableWeights, plan.bootstrap_bytes);
        per_class.insert(AllocationClass::Other, inventory);
        let mut active_models = std::collections::BTreeSet::new();
        active_models.insert(host.owner.model.clone());
        let limit = host.arbiter.snapshot(&host.resource)?.limit;
        Ok(crate::resource_arbiter::DeviceBudgetSnapshot {
            limit,
            reserved,
            remaining: limit
                .checked_sub(reserved)
                .ok_or_else(|| Error::Limit("route bootstrap budget remaining".into()))?,
            reservations: 1,
            fenced: false,
            per_class,
            active_models,
        })
    }

    #[test]
    fn short_hf_host_profile_is_exactly_4_or_8_gib() {
        assert!(supported_host_mib(4096));
        assert!(supported_host_mib(8192));
        for unsupported in [0, 4095, 4097, 5120, 8191, 8193, 16_384] {
            assert!(!supported_host_mib(unsupported));
        }
        assert!(
            host_admission_mib(
                5120,
                ResourceOwner::new("short-hf-route-test", "source-only").unwrap()
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "fresh owned process: CPU-only missing-source refusal; run this case alone"]
    fn route_failure_releases_the_admitted_retained_credit() -> Result<()> {
        let host = host(4096)?;
        let bootstrap =
            crate::tokenize_admission::approve_arbiter(&host, "short-hf-route-test-bootstrap")?;
        bootstrap.validate_domain(&host)?;
        let baseline = host.arbiter.snapshot(&host.resource)?;
        let error = route_tokenizer(Path::new("/nonexistent-short-hf-route"), &host)
            .err()
            .ok_or_else(|| Error::State("missing route directory was admitted".into()))?;
        drop(error);
        assert!(
            same_budget(&baseline, &host.arbiter.snapshot(&host.resource)?),
            "failed route must release R+G+V and its inventory"
        );
        Ok(())
    }

    #[test]
    #[ignore = "fresh owned process: actual approved source tokenizer; set ZG_SHORT_HF_SOURCE (CPU-only accounting, no model/GPU); run this case alone"]
    fn actual_route_tokenizers_own_distinct_credit_and_release_on_drop() -> Result<()> {
        let directory = std::env::var_os("ZG_SHORT_HF_SOURCE")
            .map(std::path::PathBuf::from)
            .filter(|p| p.is_absolute())
            .ok_or_else(|| Error::State("required absolute ZG_SHORT_HF_SOURCE".into()))?;
        let host = host(8192)?;
        let bootstrap =
            crate::tokenize_admission::approve_arbiter(&host, "short-hf-route-test-bootstrap")?;
        bootstrap.validate_domain(&host)?;
        let baseline = host.arbiter.snapshot(&host.resource)?;
        let credit = route_credit(&host, &bootstrap)?;
        let zero = bootstrap_budget(&host, &bootstrap)?;
        assert!(
            same_budget(&baseline, &zero),
            "bootstrap plan and actual baseline must agree"
        );
        let first = route_tokenizer(&directory, &host)?;
        let one = host.arbiter.snapshot(&host.resource)?;
        assert_eq!(one.reserved, zero.reserved + credit);
        let second = route_tokenizer(&directory, &host)?;
        let two = host.arbiter.snapshot(&host.resource)?;
        assert_eq!(two.reserved, zero.reserved + 2 * credit);
        // Arc aliases share the one wrapper/lease; last alias drop releases it.
        let shared = std::sync::Arc::new(first);
        let alias = std::sync::Arc::clone(&shared);
        drop(shared);
        assert!(!alias.tokenizer().prompt_tokens("Hello", true)?.is_empty());
        assert_eq!(
            host.arbiter.snapshot(&host.resource)?.reserved,
            zero.reserved + 2 * credit
        );
        drop(alias);
        assert_eq!(
            host.arbiter.snapshot(&host.resource)?.reserved,
            zero.reserved + credit
        );
        // A refused route leaves the known-lease budget unchanged.
        assert!(route_tokenizer(Path::new("/nonexistent-short-hf-route"), &host).is_err());
        assert_eq!(
            host.arbiter.snapshot(&host.resource)?.reserved,
            zero.reserved + credit
        );
        drop(second);
        assert!(
            same_budget(&baseline, &host.arbiter.snapshot(&host.resource)?),
            "repeated route owners must each release their own credit"
        );
        println!(
            "SHORT_HF_ROUTE_TOKENIZER_CREDIT retained_credit={credit} repeated_distinct=true borrowed_only=true"
        );
        Ok(())
    }

    fn env_path(name: &str) -> Result<PathBuf> {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .ok_or_else(|| Error::State(format!("required absolute {name}")))
    }

    enum SourceCostOutcome {
        Complete,
        Refused(Error),
    }

    fn actual_source_inventory_at(
        source_path: &Path,
        native_path: &Path,
        host_mib: usize,
        cohort_b4: bool,
    ) -> Result<()> {
        match actual_source_inventory_outcome_at(source_path, native_path, host_mib, cohort_b4)? {
            SourceCostOutcome::Complete => Ok(()),
            SourceCostOutcome::Refused(error) => Err(error),
        }
    }

    fn actual_source_inventory_outcome_at(
        source_path: &Path,
        native_path: &Path,
        host_mib: usize,
        cohort_b4: bool,
    ) -> Result<SourceCostOutcome> {
        use crate::resource_arbiter::{
            AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };
        let host_limit = host_mib
            .checked_mul(MIB)
            .ok_or_else(|| Error::Limit("actual source HOST limit bytes".into()))?;
        let host_limit_u64 = u64::try_from(host_limit)
            .map_err(|_| Error::Limit("actual source HOST limit extent".into()))?;
        // Original live process arbiter/key: the tokenizer bootstrap registry is
        // process-static. Run each permanent-owner case in a fresh test process
        // (exact case filter); never replace the arbiter or clear the registry.
        let arbiter = ResourceArbiter::process_shared();
        let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
        arbiter.register(resource.clone(), host_limit_u64)?;
        let host = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("short-hf-b4-cost", "source-only")?,
        };
        let bootstrap = crate::tokenize_admission::approve_arbiter(&host, "short-hf-b4-bootstrap")?;
        let label = if cohort_b4 { "B4" } else { "N1" };
        bootstrap.validate_domain(&host)?;
        let plan = bootstrap.plan();
        let bootstrap_owner = ResourceOwner::new(&host.owner.model, "short-hf-b4-bootstrap")?;
        let bootstrap_inventory =
            lease_actual_inventory_bytes(&host.resource, &bootstrap_owner, 2)?;
        let bootstrap_reserved = plan
            .bootstrap_bytes
            .checked_add(bootstrap_inventory)
            .ok_or_else(|| Error::Limit(format!("{label} bootstrap baseline sum")))?;
        let mut bootstrap_per_class = std::collections::BTreeMap::new();
        bootstrap_per_class.insert(AllocationClass::ImmutableWeights, plan.bootstrap_bytes);
        bootstrap_per_class.insert(AllocationClass::Other, bootstrap_inventory);
        let mut bootstrap_models = std::collections::BTreeSet::new();
        bootstrap_models.insert(host.owner.model.clone());
        let expected_bootstrap = DeviceBudgetSnapshot {
            limit: host_limit_u64,
            reserved: bootstrap_reserved,
            remaining: host_limit_u64
                .checked_sub(bootstrap_reserved)
                .ok_or_else(|| Error::Limit(format!("{label} bootstrap baseline remaining")))?,
            reservations: 1,
            fenced: false,
            per_class: bootstrap_per_class,
            active_models: bootstrap_models,
        };
        let baseline = host.arbiter.snapshot(&host.resource)?;
        if !same_budget(&expected_bootstrap, &baseline) {
            return Err(Error::State(format!(
                "{label} independently validated bootstrap baseline differs"
            )));
        }
        let charge = |bytes| {
            host.arbiter.admit(
                &host.resource,
                host.owner.clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes,
                }],
            )
        };
        let loader = charge((128 * MIB) as u64)?;
        // Later physical qualifier's real retained index; cost-only reservation.
        let reference_index = charge((16 * MIB) as u64)?;
        // R + G + V is retained around EACH opened tokenizer, so the cost
        // harness counts the same source credit the loader owns. The prior
        // owner is dropped before the independent reopen.
        let attempt = (|| -> Result<SourceCostOutcome> {
            let tokenizer = bootstrap.retain_arbiter(&host, "short-hf-b4-tokenizer")?;
            let opened = source::open(source_path, &host, None)?;
            let bound = opened.source.bind_full_model_source()?;
            let bundle =
                manifest::Bundle::open(native_path, bound.model_profile(), cohort_b4, None)?;
            let inputs = bundle.model(opened.source.source(), None)?;
            // Keep actual Ok/Err from the existing cost inspection; only this
            // authenticated invocation is tagged as a cost-stage refusal.
            let inventory = match if cohort_b4 {
                native::ShortHfResourceInventory::inspect_b4(&bound, &inputs)
            } else {
                native::ShortHfResourceInventory::inspect_n1(&bound, &inputs)
            } {
                Ok(inventory) => inventory,
                Err(error) => return Ok(SourceCostOutcome::Refused(error)),
            };
            let provider = match bound.model_profile() {
                zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => 4096,
                zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => 4608,
            };
            let (roots, cohort_values) = if cohort_b4 {
                match bound.model_profile() {
                    zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                        (22, 316_002_304)
                    }
                    zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
                        (18, 11_342_848)
                    }
                }
            } else {
                (14, 0)
            };
            if inventory.root_count != roots
                || inventory.provider_upper_bytes > provider * MIB
                || inventory.host_transient_bytes < 320 * MIB
            {
                return Err(Error::Limit(format!(
                    "complete actual {label} source/provider/transient geometry"
                )));
            }
            println!(
                "SHORT_HF_{label}_ACTUAL_SOURCE_INVENTORY {}",
                serde_json::json!({
                    "model":opened.metadata.family,"repository":opened.metadata.repository,"revision":opened.metadata.revision,
                    "source_manifest_sha256":opened.metadata.manifest_sha256,"index_sha256":opened.metadata.index_sha256,
                    "managed_allocations":inventory.managed_allocations,"managed_aligned_bytes":inventory.managed_aligned_bytes,
                    "native_provider_bytes":inventory.native_provider_bytes,"provider_upper_bytes":inventory.provider_upper_bytes,
                    "host_existing_bytes":inventory.host_existing_bytes,"host_retained_additional_bytes":inventory.host_retained_additional_bytes,
                    "host_transient_bytes":inventory.host_transient_bytes,"host_upper_bytes":inventory.host_upper_bytes,
                    "initial_b4_qualified":false,"initial_b4_costed":roots==22,"root_count":inventory.root_count,"root_initialization_host_bytes":inventory.root_initialization_host_bytes,
                    "readback_existing_bytes":inventory.readback_reservation_bytes,"readback_b4_total_bytes":if cohort_b4 {640*MIB} else {0},
                    "resident_head_bytes":inventory.resident_head_bytes,"source_chunk_upper":inventory.source_chunk_upper,
                    "value_rms_existing_shared_bytes":inventory.value_rms_shared_aligned_bytes,"b4_value_aligned_bytes":cohort_values,
                    "provider_cap_bytes":provider*MIB,"host_cap_bytes":host_limit,"physical_cap_bytes":5120*MIB,
                    "reference_index_charge_bytes":16*MIB,"tokenizer_credit_held":true,"cohort_b4":cohort_b4,"native_math_qualified":false,"gpu_executed":false
                })
            );
            let content = (
                opened.metadata.manifest_sha256,
                opened.metadata.index_sha256,
                opened.metadata.revision,
            );
            drop((inputs, bundle, bound, opened, tokenizer));
            // Full authenticated source cost, compared with the selected finite
            // host domain after the actual inventory print. Do not turn a
            // refusal into an 8 GiB fit claim, nor a success into a refusal.
            let inventory_host_upper = u64::try_from(inventory.host_upper_bytes)
                .map_err(|_| Error::Limit(format!("{label} actual source host upper extent")))?;
            if inventory_host_upper > host_limit_u64 {
                return Ok(SourceCostOutcome::Refused(Error::Limit(format!(
                    "actual authenticated {label} source host cost exceeds the selected finite host domain; refusal propagates, no fit claim"
                ))));
            }
            // Independent complete payload/member reopen checks the SAME approved
            // immutable inventories after analysis. Source issuer IDs remain fresh.
            let tokenizer = bootstrap.retain_arbiter(&host, "short-hf-b4-tokenizer")?;
            let after = source::open(source_path, &host, None)?;
            let after_bound = after.source.bind_full_model_source()?;
            let after_bundle =
                manifest::Bundle::open(native_path, after_bound.model_profile(), cohort_b4, None)?;
            let after_inputs = after_bundle.model(after.source.source(), None)?;
            if content
                != (
                    after.metadata.manifest_sha256,
                    after.metadata.index_sha256,
                    after.metadata.revision,
                )
            {
                return Err(Error::State(format!(
                    "{label} actual source before-after content identity"
                )));
            }
            drop((after_inputs, after_bundle, after_bound, after, tokenizer));
            Ok(SourceCostOutcome::Complete)
        })();
        drop((reference_index, loader));
        let final_host = host.arbiter.snapshot(&host.resource)?;
        let clean = same_budget(&baseline, &final_host);
        match attempt {
            Ok(SourceCostOutcome::Complete) => {
                if !clean {
                    return Err(Error::State(
                        "actual source cost success retained a source credit".into(),
                    ));
                }
                println!(
                    "SHORT_HF_{label}_SOURCE_COST_OK host_reserved=bootstrap host_mib={host_mib} cohort_b4={cohort_b4} gpu_executed=false no_fit_claim=true"
                );
                Ok(SourceCostOutcome::Complete)
            }
            Ok(SourceCostOutcome::Refused(error)) => {
                if !clean {
                    return Err(Error::State(
                        "actual source cost refusal retained a source credit".into(),
                    ));
                }
                let message = match &error {
                    Error::Limit(message) => message.clone(),
                    _ => error.to_string(),
                };
                println!(
                    "SHORT_HF_{label}_SOURCE_COST_REFUSED message={message} host_reserved=bootstrap host_mib={host_mib} cohort_b4={cohort_b4} gpu_executed=false no_fit_claim=true"
                );
                Ok(SourceCostOutcome::Refused(error))
            }
            Err(error) => {
                if !clean {
                    return Err(Error::State(format!(
                        "actual source failure {error} retained a source credit"
                    )));
                }
                Err(error)
            }
        }
    }

    #[test]
    #[ignore = "fresh owned process: approved original source + native full-image revalidation; run this case alone"]
    fn actual_b4_complete_source_inventory() -> Result<()> {
        actual_source_inventory_at(
            &env_path("ZG_SHORT_HF_SOURCE")?,
            &env_path("ZG_SHORT_HF_NATIVE")?,
            8192,
            true,
        )
    }

    #[test]
    #[ignore = "fresh owned process: explicit legacy 4GiB authenticated-cost refusal; run this case alone"]
    fn actual_b4_legacy_4gib_refusal_is_tied_to_authenticated_cost() -> Result<()> {
        match actual_source_inventory_outcome_at(
            &env_path("ZG_SHORT_HF_SOURCE")?,
            &env_path("ZG_SHORT_HF_NATIVE")?,
            4096,
            true,
        )? {
            SourceCostOutcome::Refused(Error::Limit(_)) => Ok(()),
            SourceCostOutcome::Refused(error) => Err(Error::State(format!(
                "B4 legacy4GiB refusal was not the authenticated cost-stage Limit: {error:?}"
            ))),
            SourceCostOutcome::Complete => Err(Error::State(
                "B4 legacy4GiB authenticated source cost unexpectedly returned Ok".into(),
            )),
        }
    }

    #[test]
    #[ignore = "fresh owned process: approved original source + matching N1 native full-image revalidation; run this case alone"]
    fn actual_n1_complete_source_inventory() -> Result<()> {
        actual_source_inventory_at(
            &env_path("ZG_SHORT_HF_SOURCE")?,
            &env_path("ZG_SHORT_HF_NATIVE")?,
            8192,
            false,
        )
    }

    #[test]
    #[ignore = "fresh owned process: missing-source refusal registers the process tokenizer domain; run this case alone"]
    fn b4_missing_inputs_are_refused_before_authenticated_cost() -> Result<()> {
        match actual_source_inventory_outcome_at(
            Path::new("/nonexistent-short-hf-source"),
            Path::new("/nonexistent-short-hf-native"),
            8192,
            true,
        ) {
            // Only the original source-open I/O NotFound refusal is expected;
            // every other Err, including cleanup failures, propagates.
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
            Ok(SourceCostOutcome::Complete) => {
                Err(Error::State("missing input was admitted".into()))
            }
            Ok(SourceCostOutcome::Refused(error)) => Err(Error::State(format!(
                "missing input reached the authenticated cost stage: {error:?}"
            ))),
        }
    }

    #[test]
    #[ignore = "fresh owned process: corrupt-source refusal registers the process tokenizer domain; run this case alone"]
    fn b4_corrupt_inputs_are_refused_cleanly() -> Result<()> {
        match actual_source_inventory_outcome_at(
            Path::new("/dev/null"),
            Path::new("/dev/null"),
            8192,
            true,
        ) {
            // Scope is invalid-directory refusal only (Io NotADirectory), not
            // corrupt JSON content; every other Err propagates.
            Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::NotADirectory => Ok(()),
            Err(error) => Err(error),
            Ok(SourceCostOutcome::Complete) => {
                Err(Error::State("corrupt input was admitted".into()))
            }
            Ok(SourceCostOutcome::Refused(error)) => Err(Error::State(format!(
                "corrupt input reached the authenticated cost stage: {error:?}"
            ))),
        }
    }
}
