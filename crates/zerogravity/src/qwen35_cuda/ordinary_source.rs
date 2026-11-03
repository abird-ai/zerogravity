//! Production ordinary-Qwen CUDA source authentication.
//!
//! This opens only the exact approved original HF source and constructs the
//! source-semantic seal used by the call-free ordinary plan. It owns no
//! tokenizer, native artifact, capture table, GPU allocation or root authority.

use super::held_directory::Directory;
use crate::accelerator::program::SourceNumericalContract;
use crate::qwen35_source::Opened;
use crate::{
    Error, Result,
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission},
};
use std::{path::Path, sync::atomic::AtomicBool};

const MIB: usize = 1024 * 1024;
pub(crate) const HOST_MIB: usize = 4096;
pub(crate) const HOST_MIB_MAX: usize = 8192;

pub(crate) fn host_admission_mib(
    host_mib: usize,
    owner: ResourceOwner,
) -> Result<TransferAdmission> {
    if !matches!(host_mib, HOST_MIB | HOST_MIB_MAX) {
        return Err(Error::Limit(
            "ordinary Qwen CUDA HOST profile supports 4096 or 8192 MiB only".into(),
        ));
    }
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "zerogravity-qwen35-cuda-ordinary")?;
    arbiter.register_matching(resource.clone(), (host_mib * MIB) as u64)?;
    Ok(TransferAdmission {
        arbiter,
        resource,
        owner,
    })
}

pub(crate) fn open(
    directory: &Path,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<Opened> {
    crate::qwen35_source::not_cancelled(cancelled)?;
    let dir = Directory::open(directory)?;
    crate::qwen35_source::open(
        crate::qwen35_source::Inputs::Directory(&dir),
        SourceNumericalContract::ReferenceCudaSm86Bf16V1,
        host,
        cancelled,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_host_profiles_are_explicit_only() {
        let owner = || ResourceOwner::new("ordinary-source-test", "host-profile").unwrap();
        assert!(host_admission_mib(0, owner()).is_err());
        assert!(host_admission_mib(4095, owner()).is_err());
        assert!(host_admission_mib(8193, owner()).is_err());
    }
}
