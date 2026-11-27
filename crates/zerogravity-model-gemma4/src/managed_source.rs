//! Original managed source only; no expected archives or runtime run paths.
use crate::text::{Gemma4CpuInventory, OriginalGemma4Text};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::cache::{SnapshotLease, TransferAdmission};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};

fn cancelled_check(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        Err(Error::State("Gemma source preparation cancelled".into()))
    } else {
        Ok(())
    }
}
use crate::text::Gemma4SourcePins;
use serde_json::Value;
#[cfg(feature = "execution")]
use sha2::{Digest, Sha256};
#[cfg(feature = "execution")]
use std::os::unix::fs::FileExt;
use std::{fs::File, os::unix::fs::MetadataExt, path::Path};
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::FileRole;
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits};
fn pins() -> Result<Value> {
    serde_json::from_slice(include_bytes!("execution/source.json"))
        .map_err(|e| Error::Malformed(format!("compiled Gemma source registry: {e}")))
}
fn field<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value[name]
        .as_str()
        .ok_or_else(|| Error::Malformed("compiled source field".into()))
}
fn stamp(file: &File) -> Result<[u64; 7]> {
    let m = file.metadata()?;
    if !m.is_file() {
        return Err(Error::Malformed("held regular Gemma source".into()));
    }
    Ok([
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ])
}
pub(crate) fn require(snapshot: &SnapshotLease) -> Result<()> {
    let pins = pins()?;
    let manifest = snapshot.manifest();
    let source = manifest.source();
    if manifest.digest()? != field(&pins, "snapshot")?
        || source.provider() != "huggingface"
        || source.repository() != field(&pins, "repository")?
        || source.revision() != field(&pins, "revision")?
        || manifest.entries().len() != 8
    {
        return Err(Error::State(
            "authentic E2B managed snapshot/source pin".into(),
        ));
    }
    for row in pins["files"]
        .as_array()
        .ok_or_else(|| Error::Malformed("source registry".into()))?
    {
        let name = field(row, "name")?;
        let entry = manifest
            .entries()
            .iter()
            .find(|entry| entry.path() == name)
            .ok_or_else(|| Error::Malformed("missing source member".into()))?;
        let role = match name {
            "model.safetensors" => FileRole::Weights,
            "config.json" => FileRole::Config,
            "tokenizer.json" => FileRole::Tokenizer,
            _ => entry.role(),
        };
        if entry.role() != role
            || entry.digest() != field(row, "sha256")?
            || Some(entry.bytes()) != row["bytes"].as_u64()
            || stamp(snapshot.file(name)?)?[2] != entry.bytes()
        {
            return Err(Error::State(
                "exact managed Gemma source roles/extents/digests".into(),
            ));
        }
    }
    Ok(())
}
pub(crate) fn open(
    snapshot: &SnapshotLease,
    host: &TransferAdmission,
    maximum: u64,
    cancelled: Option<&AtomicBool>,
) -> Result<VerifiedTensorSource> {
    require(snapshot)?;
    cancelled_check(cancelled)?;
    let pins = pins()?;
    let row = pins["files"]
        .as_array()
        .ok_or_else(|| Error::Malformed("source registry".into()))?
        .iter()
        .find(|r| r["name"] == "model.safetensors")
        .ok_or_else(|| Error::Malformed("source checkpoint".into()))?;
    let parser = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 4 * 1024 * 1024,
        }],
    )?;
    let shard = ExpectedSourceShard::new(
        "model.safetensors".into(),
        Artifact::from_file(
            snapshot.file("model.safetensors")?.try_clone()?,
            Path::new("model.safetensors"),
        )?,
        row["bytes"]
            .as_u64()
            .ok_or_else(|| Error::Malformed("source extent".into()))?,
        field(row, "sha256")?.into(),
    )?;
    let plan = SourceWeights::prepare_verified_windows(
        include_bytes!("execution/header-index.json"),
        field(&pins, "index_sha256")?,
        vec![shard],
        WindowOpenLimits {
            max_source_bytes: maximum.min(12 * 1024 * 1024 * 1024),
            max_metadata_bytes: 64 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 2011,
            scan_chunk_bytes: 1024 * 1024,
        },
        host,
    )?;
    drop(parser); // VerifiedWindowPlan now owns the admitted carried metadata.
    match cancelled {
        Some(flag) => plan.open_cancellable(flag),
        None => plan.open(),
    }
}
#[cfg(feature = "execution")]
pub(crate) fn load(
    snapshot: &SnapshotLease,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<(OriginalGemma4Text, Gemma4CpuInventory)> {
    require(snapshot)?;
    let config = snapshot.read_verified("config.json", 65536)?;
    OriginalGemma4Text::cpu_declared_inventory(&config)?;
    let source = open(snapshot, host, 12 * 1024 * 1024 * 1024, cancelled)?;
    bind_text(source, &config, cancelled)
}
pub(crate) fn bind_text(
    source: VerifiedTensorSource,
    config: &[u8],
    cancelled: Option<&AtomicBool>,
) -> Result<(OriginalGemma4Text, Gemma4CpuInventory)> {
    let pins = pins()?;
    let inventory = OriginalGemma4Text::cpu_inventory(&source, config)?;
    let model = OriginalGemma4Text::bind(
        source,
        config,
        Gemma4SourcePins {
            repository: field(&pins, "repository")?,
            revision: field(&pins, "revision")?,
            config_sha256: field(&pins, "config_sha256")?,
            source_index_sha256: field(&pins, "index_sha256")?,
            source_bundle_sha256: field(&pins, "bundle_sha256")?,
        },
        cancelled,
    )?;
    Ok((model, inventory))
}
#[cfg(feature = "execution")]
pub(crate) fn after(snapshot: &SnapshotLease, cancelled: Option<&AtomicBool>) -> Result<()> {
    require(snapshot)?;
    for entry in snapshot.manifest().entries() {
        cancelled_check(cancelled)?;
        let file = snapshot.file(entry.path())?;
        let before = stamp(file)?;
        let mut hash = Sha256::new();
        let mut at = 0;
        let mut buffer = [0; 65536];
        while at < entry.bytes() {
            cancelled_check(cancelled)?;
            let n = (entry.bytes() - at).min(buffer.len() as u64) as usize;
            file.read_exact_at(&mut buffer[..n], at)?;
            hash.update(&buffer[..n]);
            at += n as u64;
        }
        if stamp(file)? != before || format!("{:x}", hash.finalize()) != entry.digest() {
            return Err(Error::State(
                "same held Gemma source after ordinary execution".into(),
            ));
        }
    }
    Ok(())
}
