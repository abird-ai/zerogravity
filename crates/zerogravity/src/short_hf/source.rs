//! Exact approved inventories; paths and directory names never choose a model.
use super::*;
use crate::canonical47::paths::{Directory, digest};
use crate::qwen35::original_binding::OriginalWindowModelSource;
use sha2::{Digest, Sha256};
use zerogravity_artifacts::{
    artifact::Artifact,
    model::qwen_owned::QwenOriginalWindowText,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};

pub(super) struct Opened {
    pub source: Arc<OriginalWindowModelSource>,
    pub tokenizer: QwenTokenizer,
    pub metadata: ShortHfSourceMetadata,
}
struct Spec {
    family: &'static str,
    repository: &'static str,
    revision: &'static str,
    config_sha: &'static str,
    index_sha: &'static str,
    manifest_sha: &'static str,
    inventory: &'static str,
    hidden: usize,
    max_source: u64,
}
static SPECS: [Spec; 2] = [
    Spec {
        family: "0.8b",
        repository: "Qwen/Qwen3.5-0.8B",
        revision: "2fc06364715b967f1860aea9cf38778875588b17",
        config_sha: "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        index_sha: "d8a08838a613b025eb7952ed9db11696213e57e76a375661ef5c12f9dd5dcf4e",
        manifest_sha: "c4c1d349743c0e0ba51ab07f769348e6b13f9896c354303c7af5e839c11dbf09",
        inventory: include_str!("source08.json"),
        hidden: 1024,
        max_source: 2 * 1024 * MIB as u64,
    },
    Spec {
        family: "2b",
        repository: "Qwen/Qwen3.5-2B",
        revision: "15852e8c16360a2fea060d615a32b45270f8a8fc",
        config_sha: "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
        index_sha: "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
        manifest_sha: "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
        inventory: include_str!("source2b.json"),
        hidden: 2048,
        max_source: 5 * 1024 * MIB as u64,
    },
];
fn select(config: &[u8]) -> Result<&'static Spec> {
    let sha = Sha256::digest(config);
    SPECS
        .iter()
        .find(|s| sha[..] == digest(s.config_sha))
        .ok_or_else(|| Error::Unsupported("short-HF exact approved config/source required".into()))
}
fn member(row: &serde_json::Value, small: bool) -> Result<(&str, usize, &str)> {
    let text = |key| {
        row[key]
            .as_str()
            .ok_or_else(|| Error::Malformed("compiled source member".into()))
    };
    let name = text(if small { "name" } else { "path" })?;
    let sha = text(if small { "sha256" } else { "digest" })?;
    let bytes = row["bytes"]
        .as_u64()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| Error::Limit("compiled source member bytes".into()))?;
    Ok((name, bytes, sha))
}
pub(super) fn open(
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
        .map_err(|e| Error::Malformed(format!("compiled source inventory: {e}")))?;
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
    let mut tokenizer = None;
    let mut shard = None;
    let mut names = std::collections::BTreeSet::new();
    for row in files {
        let (name, bytes, sha) = member(row, small)?;
        if !names.insert(name) {
            return Err(Error::Malformed("duplicate original source member".into()));
        }
        if name.ends_with(".safetensors") {
            if shard.is_some() || name != "model.safetensors-00001-of-00001.safetensors" {
                return Err(Error::Unsupported("short-HF exact one-shard source".into()));
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
            "tokenizer.json" => tokenizer = Some(QwenTokenizer::from_bytes(&raw)?),
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
            "complete actual runtime source/tokenizer/template members".into(),
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
    let source = OriginalWindowModelSource::from_owner(owner)?;
    if source
        .bind_full_model_source()?
        .model_profile()
        .hidden_size()
        != spec.hidden
    {
        return Err(Error::State(
            "actual full source/profile differs from inventory".into(),
        ));
    }
    Ok(Opened {
        source,
        tokenizer: tokenizer.ok_or_else(|| Error::Malformed("original tokenizer absent".into()))?,
        metadata: ShortHfSourceMetadata {
            family: spec.family,
            repository: spec.repository,
            revision: spec.revision,
            index_sha256: spec.index_sha,
            manifest_sha256: spec.manifest_sha,
            hidden_width: spec.hidden,
            vocabulary_size: 248320,
            prefill_rows: [47, 57, 62, 67, 72],
            max_output: 8,
            context: 512,
            cached_through: 78,
        },
    })
}

/// Source-bound route metadata only; no source allocation or execution grant.
pub(super) fn route_metadata(directory: &Path) -> Result<super::ShortHfRouteMetadata> {
    let dir = Directory::open(directory)?;
    let config = dir.bounded("config.json", 4096, None)?;
    let spec = select(&config)?;
    Ok(super::ShortHfRouteMetadata {
        family: spec.family,
        repository: spec.repository,
        revision: spec.revision,
        manifest_sha256: spec.manifest_sha,
        provider_bytes: if spec.family == "0.8b" {
            4096 * MIB
        } else {
            4608 * MIB
        },
        profile: if spec.family == "0.8b" {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen08
        } else {
            zerogravity_core::accelerator::program::FullModelProfile::Qwen2B
        },
    })
}
pub(super) fn route_tokenizer(directory: &Path) -> Result<QwenTokenizer> {
    let dir = Directory::open(directory)?;
    let config = dir.bounded("config.json", 4096, None)?;
    let spec = select(&config)?;
    let inventory: serde_json::Value = serde_json::from_str(spec.inventory)
        .map_err(|_| Error::State("compiled source inventory".into()))?;
    let rows = inventory["files"]
        .as_array()
        .ok_or_else(|| Error::State("compiled source members".into()))?;
    let row = rows
        .iter()
        .find(|row| {
            member(row, spec.family == "0.8b").is_ok_and(|(name, _, _)| name == "tokenizer.json")
        })
        .ok_or_else(|| Error::State("compiled tokenizer absent".into()))?;
    let (name, bytes, sha) = member(row, spec.family == "0.8b")?;
    QwenTokenizer::from_bytes(&dir.read(name, bytes, Some(&digest(sha)), None)?)
}
