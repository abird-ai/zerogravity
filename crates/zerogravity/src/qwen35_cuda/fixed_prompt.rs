//! Finite original-HF source/native loader for the existing fixed-prompt engine.
//! No downloader, expected archive, alternate evaluator or eligibility inference.
use crate::gpu::cuda::native;
use crate::qwen35::{source::binding::OriginalWindowModelSource, tokenize::QwenTokenizer};
use crate::resource_arbiter::{GlobalResourceLease, lease_inventory_bytes};
use crate::{Error, Result};
pub use native::FixedPromptLifecycleProfile;
use native::{FixedPromptLoaded, prepare_fixed_prompt_natural_with_profile};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_artifacts::cache::TransferAdmission;
#[cfg(test)]
mod active8_tests;
mod bindings;
mod manifest;
pub(crate) mod paths;
mod registry;
mod session;
mod source;
pub use session::Session;
#[cfg(test)]
mod tests;

pub const HOST_MIB: usize = 3072;
pub const PROVIDER_MIB: usize = 2304;
pub const DEVICE_MIB: usize = 5120;
const MIB: usize = 1024 * 1024;
const LOADER_BYTES: usize = 128 * MIB;

/// Tokenizer allocations remain charged while any model/request owner survives.
pub struct Model {
    resident: Arc<FixedPromptLoaded>,
    templates: Arc<bindings::Templates>,
    tokenizer: Arc<QwenTokenizer>,
    _tokenizer: crate::tokenize_admission::Retained,
    _loader: GlobalResourceLease,
}
impl Model {
    pub async fn session(&self, cancelled: Option<&AtomicBool>) -> Result<Session> {
        Ok(Session::new(
            self.resident.session(cancelled).await?,
            self.templates.clone(),
        ))
    }
    pub fn validate_prompt(&self, ids: &[u32], max_tokens: usize) -> Result<()> {
        OriginalWindowModelSource::validate_fixed_prompt_request(ids, max_tokens)
    }
    pub fn resident(&self) -> &Arc<FixedPromptLoaded> {
        &self.resident
    }
    pub fn tokenizer(&self) -> &QwenTokenizer {
        &self.tokenizer
    }
}

pub fn native_artifact_root(explicit: Option<&Path>) -> Result<PathBuf> {
    paths::native_root(
        explicit,
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    )
}
fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        Err(Error::State("fixed-prompt source load cancelled".into()))
    } else {
        Ok(())
    }
}

fn require_host_profile(
    host: &TransferAdmission,
    profile: FixedPromptLifecycleProfile,
) -> Result<()> {
    if profile == FixedPromptLifecycleProfile::BenchmarkEight
        && host.arbiter.snapshot(&host.resource)?.limit != profile.host_bytes() as u64
    {
        return Err(Error::Limit(
            "canonical eight-session actual4GiB host ledger required".into(),
        ));
    }
    Ok(())
}

/// Source bytes are authenticated before constructing the first device provider.
pub async fn load(
    device: crate::gpu::Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<Model>> {
    load_with_profile(
        device,
        hf_directory,
        artifact_directory,
        host,
        FixedPromptLifecycleProfile::FiniteTwo,
        cancelled,
    )
    .await
}

/// Selects the actual prepaid owner envelope; source/math eligibility is unchanged.
pub async fn load_with_profile(
    device: crate::gpu::Device,
    hf_directory: &Path,
    artifact_directory: &Path,
    host: &TransferAdmission,
    profile: FixedPromptLifecycleProfile,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<Model>> {
    not_cancelled(cancelled)?;
    if !matches!(device, crate::gpu::Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "fixed-prompt CUDA source profile only".into(),
        ));
    }
    require_host_profile(host, profile)?;
    // Source opening has its original64MiB metadata +1MiB scan admissions.
    // This independent128MiB envelope precedes manifest/tokenizer parsing and
    // carries C's checked lease-inventory overhead in the same reservation.
    let inventory = lease_inventory_bytes(&host.owner.model, &host.owner.session, 2)?;
    let loader = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[
            crate::resource_arbiter::AllocationRequest {
                kind: crate::resource_arbiter::AllocationClass::Scratch,
                bytes: LOADER_BYTES as u64,
            },
            crate::resource_arbiter::AllocationRequest {
                kind: crate::resource_arbiter::AllocationClass::Other,
                bytes: inventory,
            },
        ],
    )?;
    let source = source::open(hf_directory, host, cancelled)?;
    let bundle = manifest::Bundle::open(artifact_directory, cancelled)?;
    let bootstrap = crate::tokenize_admission::approve_arbiter(host, "fixed-prompt-inventory")?;
    let retained = bootstrap.retain_arbiter(host, "fixed-prompt-inventory")?;
    let tokenizer = Arc::new(source::tokenizer(hf_directory, host, cancelled)?);
    let templates = Arc::new(bindings::Templates::prepare(source.clone())?);
    let rotary = super::qualification::qualify_source_rotary_cubins(
        device.clone(),
        source.clone(),
        (DEVICE_MIB * MIB) as u64,
        Some(bundle.rotary(source.source(), cancelled)?),
        native::RotaryMode::Complete,
        cancelled,
        |_| Ok(()),
    )
    .await?;
    // All real rotary kernels/images retire before the full model image pack.
    let comparison = rotary.comparison;
    if comparison.native_launches != 12
        || comparison.completed_readback_flights != 2
        || comparison.resident_bytes != 0
        || comparison.pending_events != 0
        || comparison.physical_reserved_bytes != 0
        || comparison.physical_fenced
    {
        return Err(Error::State(
            "fixed-prompt source rotary did not complete/retire".into(),
        ));
    }
    let inputs = native::FixedPromptQkKvQualificationInputs {
        native: bundle.model(source.source(), cancelled)?,
        rotary: Some(rotary),
    };
    let resident = prepare_fixed_prompt_natural_with_profile(
        device,
        source.source().clone(),
        templates.physical_scopes(),
        inputs,
        profile,
        cancelled,
    )
    .await?;
    if let Err(error) = not_cancelled(cancelled) {
        resident.clone().close(None).await?;
        return Err(error);
    }
    Ok(Arc::new(Model {
        resident,
        templates,
        tokenizer,
        _tokenizer: retained,
        _loader: loader,
    }))
}
