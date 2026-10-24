//! SAME original managed31 members and18 shard FDs; no alternate weights/path.
use super::*;
use crate::{artifact::Artifact, cache::FileRole};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    os::unix::fs::{FileExt, MetadataExt},
    path::Path,
};
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits};
use zerogravity_model_qwen35::original_binding::QwenNumericalProfile;
fn pins() -> Result<Value> {
    serde_json::from_slice(include_bytes!("source.json"))
        .map_err(|e| Error::Malformed(format!("compiled Qwen source pins: {e}")))
}
fn field<'a>(row: &'a Value, key: &str) -> Result<&'a str> {
    row[key]
        .as_str()
        .ok_or_else(|| Error::Malformed("compiled source field".into()))
}
fn stamp(f: &File) -> Result<[u64; 7]> {
    let m = f.metadata()?;
    if !m.is_file() {
        return Err(Error::State("held original regular source".into()));
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
pub(super) fn require(snapshot: &SnapshotLease) -> Result<()> {
    let pins = pins()?;
    let manifest = snapshot.manifest();
    let source = manifest.source();
    if manifest.digest()? != field(&pins, "snapshot")?
        || source.provider() != "huggingface"
        || source.repository() != field(&pins, "repository")?
        || source.revision() != field(&pins, "revision")?
        || manifest.entries().len() != 31
    {
        return Err(Error::State("exact managed original27B snapshot".into()));
    }
    for row in pins["files"]
        .as_array()
        .ok_or_else(|| Error::Malformed("source member pins".into()))?
    {
        let name = field(row, "name")?;
        let entry = manifest
            .entries()
            .iter()
            .find(|e| e.path() == name)
            .ok_or_else(|| Error::State("original source member absent".into()))?;
        let role = if name.ends_with(".safetensors") {
            FileRole::Weights
        } else if name == "config.json" {
            FileRole::Config
        } else if name == "tokenizer.json" {
            FileRole::Tokenizer
        } else {
            entry.role()
        };
        if entry.role() != role
            || entry.digest() != field(row, "sha256")?
            || Some(entry.bytes()) != row["bytes"].as_u64()
            || stamp(snapshot.file(name)?)?[2] != entry.bytes()
        {
            return Err(Error::State(
                "exact original managed source role/digest/extent".into(),
            ));
        }
    }
    Ok(())
}
pub(super) fn load(
    snapshot: &SnapshotLease,
    host: &TransferAdmission,
    cancelled: Option<&AtomicBool>,
) -> Result<OriginalWindowOperators> {
    check_cancel(cancelled)?;
    require(snapshot)?;
    let pins = pins()?;
    let config = snapshot.read_verified("config.json", 65536)?;
    let index = snapshot.read_verified("model.safetensors.index.json", 256 * 1024)?;
    let parser = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 16 * 1024 * 1024,
        }],
    )?;
    let mut shards = Vec::with_capacity(18);
    let mut bytes = 0u64;
    for row in pins["files"]
        .as_array()
        .ok_or_else(|| Error::Malformed("source shards".into()))?
    {
        let name = field(row, "name")?;
        if !name.ends_with(".safetensors") {
            continue;
        }
        check_cancel(cancelled)?;
        let n = row["bytes"]
            .as_u64()
            .ok_or_else(|| Error::Malformed("source shard extent".into()))?;
        bytes = bytes
            .checked_add(n)
            .ok_or_else(|| Error::Limit("source shard total".into()))?;
        shards.push(ExpectedSourceShard::new(
            name.into(),
            Artifact::from_file(snapshot.file(name)?.try_clone()?, Path::new(name))?,
            n,
            field(row, "sha256")?.into(),
        )?);
    }
    if shards.len() != 18 || bytes != 55_563_006_776 {
        return Err(Error::State("all18 original shard bytes".into()));
    }
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        field(&pins, "index_sha256")?,
        shards,
        WindowOpenLimits {
            max_source_bytes: 64 * 1024 * 1024 * 1024,
            max_metadata_bytes: 64 * 1024 * 1024,
            max_shards: 18,
            max_tensors: 1199,
            scan_chunk_bytes: 4 * 1024 * 1024,
        },
        host,
    )?;
    drop(parser);
    let source = match cancelled {
        Some(f) => plan.open_cancellable(f)?,
        None => plan.open()?,
    };
    if source.verified_bundle_sha256() != field(&pins, "bundle_sha256")? {
        return Err(Error::State(
            "same original whole18-shard source bundle".into(),
        ));
    }
    let model = OriginalWindowOperators::bind(
        source,
        &config,
        field(&pins, "config_sha256")?,
        field(&pins, "index_sha256")?,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )?;
    let g = model.binding().config().geometry();
    if g.hidden_size != 5120 || g.vocab_size != 248320 || g.layers.len() != 64 {
        return Err(Error::Unsupported(
            "original27B finite full64 geometry".into(),
        ));
    }
    Ok(model)
}
pub(super) fn after(snapshot: &SnapshotLease, cancelled: Option<&AtomicBool>) -> Result<()> {
    require(snapshot)?;
    for entry in snapshot.manifest().entries() {
        check_cancel(cancelled)?;
        let f = snapshot.file(entry.path())?;
        let before = stamp(f)?;
        let mut hash = Sha256::new();
        let mut at = 0;
        let mut buffer = [0u8; 65536];
        while at < entry.bytes() {
            check_cancel(cancelled)?;
            let n = (entry.bytes() - at).min(buffer.len() as u64) as usize;
            f.read_exact_at(&mut buffer[..n], at)?;
            hash.update(&buffer[..n]);
            at += n as u64;
        }
        if stamp(f)? != before || format!("{:x}", hash.finalize()) != entry.digest() {
            return Err(Error::State(
                "same held original source SHA after execution".into(),
            ));
        }
    }
    Ok(())
}
