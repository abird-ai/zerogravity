//! Original Qwen source authentication shared by runtime and diagnostic callers.
//!
//! This opens only the exact approved original HF source and constructs the
//! source-semantic seal used by the call-free forward plan. It owns no
//! tokenizer, native artifact, capture table, GPU allocation or root authority.

use self::source_inventory::{digest, member, select};
use crate::{
    Error, Result,
    accelerator::program::NumericalContract,
    qwen35::source::{
        binding::{ForwardSourceSeal, OriginalWindowModelSource},
        owned::QwenOriginalWindowText,
    },
    resource_arbiter::TransferAdmission,
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
#[cfg(feature = "qwen35-cuda-qualification")]
pub(crate) mod held_directory;
pub(crate) mod held_file;
mod inputs;
pub(crate) mod source_inventory;
pub(crate) use inputs::Inputs;

#[derive(Clone, Debug)]
pub(crate) struct Metadata {
    pub(crate) family: &'static str,
    pub(crate) repository: &'static str,
    pub(crate) revision: &'static str,
    pub(crate) hidden_width: usize,
}

pub(crate) struct Opened {
    pub(crate) seal: ForwardSourceSeal,
    pub(crate) metadata: Metadata,
    #[cfg(any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    ))]
    pub(crate) tokenizer: Option<Vec<u8>>,
}

pub(crate) fn not_cancelled(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Qwen CUDA source load cancelled".into()));
    }
    Ok(())
}

pub(crate) fn open(
    dir: Inputs<'_>,
    numerical: NumericalContract,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<Opened> {
    not_cancelled(cancelled)?;
    let config = dir.bounded("config.json", 64 * 1024, cancelled)?;
    let spec = select(&config)?;
    if Sha256::digest(spec.inventory.as_bytes())[..] != digest(spec.manifest_sha) {
        return Err(Error::State(
            "compiled original inventory identity differs".into(),
        ));
    }
    let inventory: serde_json::Value = serde_json::from_str(spec.inventory)
        .map_err(|error| Error::Malformed(format!("compiled source inventory: {error}")))?;
    let files = inventory["files"]
        .as_array()
        .filter(|rows| rows.len() == spec.member_count)
        .ok_or_else(|| Error::Malformed("complete compiled source membership".into()))?;
    dir.require_inventory(spec, files.len(), cancelled)?;

    #[cfg(any(
        all(feature = "model-qwen35", feature = "hip"),
        all(feature = "model-qwen35", feature = "cuda")
    ))]
    let mut tokenizer = None;
    let mut index = None;
    let mut shards = Vec::with_capacity(spec.shard_count);
    let mut names = std::collections::BTreeSet::new();
    for row in files {
        let (name, bytes, sha) = member(row, spec.named_members)?;
        if !names.insert(name) {
            return Err(Error::Malformed("duplicate original source member".into()));
        }
        if name.ends_with(".safetensors") {
            if shards.len() >= spec.shard_count {
                return Err(Error::Malformed("Qwen forward original shard count".into()));
            }
            shards.push(ExpectedSourceShard::new(
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
            #[cfg(any(
                all(feature = "model-qwen35", feature = "hip"),
                all(feature = "model-qwen35", feature = "cuda")
            ))]
            "tokenizer.json" if dir.retains_tokenizer() => tokenizer = Some(raw),
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

    if shards.len() != spec.shard_count {
        return Err(Error::Malformed(
            "Qwen forward original shards absent".into(),
        ));
    }
    let index = index.ok_or_else(|| Error::Malformed("original source index absent".into()))?;
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        spec.index_sha,
        shards,
        WindowOpenLimits {
            max_source_bytes: spec.max_source,
            max_metadata_bytes: (64 * MIB) as u64,
            max_shards: spec.shard_count,
            max_tensors: spec.tensor_limit,
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
    let seal = match numerical {
        NumericalContract::CudaSm86Bf16ReferenceV1 => model.bind_cuda_forward_source()?,
        NumericalContract::TorchCpuAvx2Bf16MixedV1 => model.bind_cpu_forward_source()?,
        _ => {
            return Err(Error::Unsupported(
                "forward source numerical contract".into(),
            ));
        }
    };
    if seal.model_profile().hidden_size() != spec.hidden {
        return Err(Error::State(
            "Qwen forward actual source/profile differs from inventory".into(),
        ));
    }
    Ok(Opened {
        seal,
        #[cfg(any(
            all(feature = "model-qwen35", feature = "hip"),
            all(feature = "model-qwen35", feature = "cuda")
        ))]
        tokenizer,
        metadata: Metadata {
            family: spec.family,
            repository: spec.repository,
            revision: spec.revision,
            hidden_width: spec.hidden,
        },
    })
}
