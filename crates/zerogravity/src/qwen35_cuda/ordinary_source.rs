//! Production ordinary-Qwen CUDA source authentication.
//!
//! This opens only the exact approved original HF source and constructs the
//! source-semantic seal used by the call-free ordinary plan. It owns no
//! tokenizer, native artifact, capture table, GPU allocation or root authority.

use super::{
    held_directory::Directory,
    source_inventory::{digest, member, select},
};
use crate::{
    Error, Result,
    qwen35::source::{
        binding::{OrdinaryCudaSourceSeal, OriginalWindowModelSource},
        owned::QwenOriginalWindowText,
    },
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner, TransferAdmission},
};
use sha2::{Digest, Sha256};
use std::{
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::{
    artifact::Artifact,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};

const MIB: usize = 1024 * 1024;
pub(crate) const HOST_MIB: usize = 4096;
pub(crate) const HOST_MIB_MAX: usize = 8192;

#[derive(Clone, Debug)]
pub(crate) struct Metadata {
    pub(crate) family: &'static str,
    pub(crate) repository: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) index_sha256: &'static str,
    pub(crate) hidden_width: usize,
}

pub(crate) struct Opened {
    pub(crate) seal: OrdinaryCudaSourceSeal,
    pub(crate) metadata: Metadata,
}

fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Qwen CUDA source load cancelled".into()));
    }
    Ok(())
}

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
    not_cancelled(cancelled)?;
    let dir = Directory::open(directory)?;
    let config = dir.bounded("config.json", 4096, cancelled)?;
    let spec = select(&config)?;
    if Sha256::digest(spec.inventory.as_bytes())[..] != digest(spec.manifest_sha) {
        return Err(Error::State(
            "compiled original inventory identity differs".into(),
        ));
    }
    let inventory: serde_json::Value = serde_json::from_str(spec.inventory)
        .map_err(|error| Error::Malformed(format!("compiled source inventory: {error}")))?;
    let small = spec.family == "0.8b";
    let files = inventory["files"]
        .as_array()
        .filter(|rows| rows.len() == if small { 11 } else { 12 })
        .ok_or_else(|| Error::Malformed("complete compiled source membership".into()))?;
    if small {
        drop(dir.read(
            "source-receipt.json",
            spec.inventory.len(),
            Some(&digest(spec.manifest_sha)),
            cancelled,
        )?);
    }

    let mut index = None;
    let mut shard = None;
    let mut names = std::collections::BTreeSet::new();
    for row in files {
        let (name, bytes, sha) = member(row, small)?;
        if !names.insert(name) {
            return Err(Error::Malformed("duplicate original source member".into()));
        }
        if name.ends_with(".safetensors") {
            if shard.is_some() || name != "model.safetensors-00001-of-00001.safetensors" {
                return Err(Error::Unsupported(
                    "ordinary Qwen CUDA exact one-shard source".into(),
                ));
            }
            shard = Some(ExpectedSourceShard::new(
                name.into(),
                Artifact::from_file(dir.file(name)?, Path::new(name))?,
                bytes as u64,
                sha.into(),
            )?);
            continue;
        }
        let raw = dir.read(name, bytes, Some(&digest(sha)), cancelled)?;
        match name {
            "config.json" if raw != config => {
                return Err(Error::State("held selected config changed".into()));
            }
            "model.safetensors.index.json" => index = Some(raw),
            _ => {}
        }
    }
    if ![
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ]
    .iter()
    .all(|name| names.contains(name))
    {
        return Err(Error::Malformed(
            "complete actual runtime source members".into(),
        ));
    }

    let index = index.ok_or_else(|| Error::Malformed("original source index absent".into()))?;
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        spec.index_sha,
        vec![shard.ok_or_else(|| Error::Malformed("original source shard absent".into()))?],
        WindowOpenLimits {
            max_source_bytes: spec.max_source,
            max_metadata_bytes: (64 * MIB) as u64,
            max_shards: 1,
            max_tensors: 1024,
            scan_chunk_bytes: MIB,
        },
        host,
    )?;
    not_cancelled(cancelled)?;
    let verified = if let Some(flag) = cancelled {
        plan.open_cancellable(flag)?
    } else {
        plan.open()?
    };
    let owner = Arc::new(QwenOriginalWindowText::bind(
        verified,
        &config,
        spec.config_sha,
        spec.index_sha,
    )?);
    let model = OriginalWindowModelSource::from_owner(owner)?;
    let seal = model.bind_ordinary_cuda_source()?;
    if seal.model_profile().hidden_size() != spec.hidden {
        return Err(Error::State(
            "ordinary Qwen actual source/profile differs from inventory".into(),
        ));
    }
    Ok(Opened {
        seal,
        metadata: Metadata {
            family: spec.family,
            repository: spec.repository,
            revision: spec.revision,
            index_sha256: spec.index_sha,
            hidden_width: spec.hidden,
        },
    })
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
