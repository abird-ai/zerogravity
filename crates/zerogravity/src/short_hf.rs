//! Authenticated short-HF source and tokenizer; the original engine owns math.
use crate::gpu::vllm_cuda_qualification as native;
use crate::qwen35::tokenize::QwenTokenizer;
use crate::resource_arbiter::GlobalResourceLease;
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
const MIB: usize = 1024 * 1024;
pub const HOST_MIB: usize = 4096;
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
    if host.arbiter.snapshot(&host.resource)?.limit != (HOST_MIB * MIB) as u64 {
        return Err(Error::Limit(
            "short-HF actual4GiB original host owner required".into(),
        ));
    }
    let loader = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[crate::resource_arbiter::AllocationRequest {
            kind: crate::resource_arbiter::AllocationClass::Scratch,
            bytes: (128 * MIB) as u64,
        }],
    )?;
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
pub fn route_tokenizer(directory: &Path) -> Result<QwenTokenizer> {
    source::route_tokenizer(directory)
}

/// Original scoped source HOST policy shared by direct and serving frontends.
/// It is not system RAM capacity or a second physical-memory allocator.
pub fn host_admission(owner: crate::resource_arbiter::ResourceOwner) -> Result<TransferAdmission> {
    use crate::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter};
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
    arbiter.register_matching(resource.clone(), (HOST_MIB * MIB) as u64)?;
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner,
    })
}

#[cfg(test)]
mod b4_inventory_tests {
    use super::*;
    #[test]
    #[ignore = "CPU-only approved original source +native full-image revalidation; no model/GPU execution"]
    fn actual_b4_complete_source_inventory() -> Result<()> {
        use crate::resource_arbiter::{
            AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
        };
        let path = |name: &str| {
            std::env::var_os(name)
                .map(std::path::PathBuf::from)
                .filter(|p| p.is_absolute())
                .ok_or_else(|| Error::State(format!("required absolute {name}")))
        };
        let arbiter = ResourceArbiter::process_shared();
        let resource = PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?;
        arbiter.register(resource.clone(), (4096 * MIB) as u64)?;
        let host = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("short-hf-b4-cost", "source-only")?,
        };
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
        let opened = source::open(&path("ZG_SHORT_HF_SOURCE")?, &host, None)?;
        let bound = opened.source.bind_full_model_source()?;
        let bundle = manifest::Bundle::open(
            &path("ZG_SHORT_HF_NATIVE")?,
            bound.model_profile(),
            true,
            None,
        )?;
        let inputs = bundle.model(opened.source.source(), None)?;
        let inventory = native::ShortHfResourceInventory::inspect_b4(&bound, &inputs)?;
        let (roots, provider, cohort_values) = match bound.model_profile() {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08 => {
                (22, 4096, 316_002_304)
            }
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B => {
                (18, 4608, 11_342_848)
            }
        };
        if inventory.root_count != roots
            || inventory.provider_upper_bytes > provider * MIB
            || inventory.host_upper_bytes > 4096 * MIB
            || inventory.host_transient_bytes < 320 * MIB
        {
            return Err(Error::Limit(
                "complete actual B4 source/cold/live/readback bound".into(),
            ));
        }
        println!(
            "SHORT_HF_B4_ACTUAL_SOURCE_INVENTORY {}",
            serde_json::json!({
                "model":opened.metadata.family,"repository":opened.metadata.repository,"revision":opened.metadata.revision,
                "source_manifest_sha256":opened.metadata.manifest_sha256,"index_sha256":opened.metadata.index_sha256,
                "managed_allocations":inventory.managed_allocations,"managed_aligned_bytes":inventory.managed_aligned_bytes,
                "native_provider_bytes":inventory.native_provider_bytes,"provider_upper_bytes":inventory.provider_upper_bytes,
                "host_existing_bytes":inventory.host_existing_bytes,"host_retained_additional_bytes":inventory.host_retained_additional_bytes,
                "host_transient_bytes":inventory.host_transient_bytes,"host_upper_bytes":inventory.host_upper_bytes,
                "initial_b4_qualified":false,"initial_b4_costed":roots==22,"root_count":inventory.root_count,"root_initialization_host_bytes":inventory.root_initialization_host_bytes,
                "readback_existing_bytes":inventory.readback_reservation_bytes,"readback_b4_total_bytes":640*MIB,
                "resident_head_bytes":inventory.resident_head_bytes,"source_chunk_upper":inventory.source_chunk_upper,
                "value_rms_existing_shared_bytes":inventory.value_rms_shared_aligned_bytes,"b4_value_aligned_bytes":cohort_values,
                "provider_cap_bytes":provider*MIB,"host_cap_bytes":4096*MIB,"physical_cap_bytes":5120*MIB,
                "reference_index_charge_bytes":16*MIB,"native_math_qualified":false,"gpu_executed":false
            })
        );
        let content = (
            opened.metadata.manifest_sha256,
            opened.metadata.index_sha256,
            opened.metadata.revision,
        );
        drop((inputs, bundle, bound, opened));
        // Independent complete payload/member reopen checks the SAME approved
        // immutable inventories after analysis. Source issuer IDs remain fresh.
        let after = source::open(&path("ZG_SHORT_HF_SOURCE")?, &host, None)?;
        let after_bound = after.source.bind_full_model_source()?;
        let after_bundle = manifest::Bundle::open(
            &path("ZG_SHORT_HF_NATIVE")?,
            after_bound.model_profile(),
            true,
            None,
        )?;
        let after_inputs = after_bundle.model(after.source.source(), None)?;
        if content
            != (
                after.metadata.manifest_sha256,
                after.metadata.index_sha256,
                after.metadata.revision,
            )
        {
            return Err(Error::State(
                "B4 actual source before-after content identity".into(),
            ));
        }
        drop((
            after_inputs,
            after_bundle,
            after_bound,
            after,
            reference_index,
            loader,
        ));
        if host.arbiter.snapshot(&host.resource)?.reserved != 0 {
            return Err(Error::State(
                "B4 source-only final host reservation nonzero".into(),
            ));
        }
        println!(
            "SHORT_HF_B4_INVENTORY_CLOSED source_before_after=true native_images_before_after=true host_reserved=0 gpu_executed=false"
        );
        Ok(())
    }
}
