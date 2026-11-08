//! Approved source opening for explicit whole-model diagnostics; no run/serve edge.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::Path,
    sync::Arc,
};
use zerogravity::cache::{SnapshotLease, TransferAdmission};
use zerogravity::qwen35::source::binding::{OriginalWindowOperators, QwenNumericalProfile};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity::{Error, Result};
use zerogravity_artifacts::{
    artifact::Artifact,
    source_weights::{ExpectedSourceShard, SourceWeights, WindowOpenLimits},
};

pub struct Member {
    pub name: String,
    pub bytes: u64,
    pub sha: String,
}
pub struct Loaded {
    pub model: Arc<OriginalWindowOperators>,
    pub members: Vec<Member>,
    pub host: TransferAdmission,
    files: BTreeMap<String, File>,
    stamps: BTreeMap<String, [u64; 7]>,
    snapshot: Option<Arc<SnapshotLease>>,
    _control: GlobalResourceLease,
}
pub fn need(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::State(message.into()))
    }
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key]
        .as_str()
        .ok_or_else(|| Error::Malformed(format!("source fixture field {key}")))
}
fn stamp(file: &File) -> Result<[u64; 7]> {
    let m = file.metadata()?;
    need(m.is_file(), "source held regular file")?;
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
fn hash(file: &File, bytes: u64) -> Result<String> {
    let mut h = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut at = 0;
    while at < bytes {
        let n = (bytes - at).min(buffer.len() as u64) as usize;
        file.read_exact_at(&mut buffer[..n], at)?;
        h.update(&buffer[..n]);
        at += n as u64;
    }
    let mut tail = [0u8; 1];
    need(file.read_at(&mut tail, bytes)? == 0, "source actual EOF")?;
    Ok(format!("{:x}", h.finalize()))
}
impl Loaded {
    pub fn open(label: &str, directory: &Path) -> Result<Arc<Self>> {
        Self::open_inner(label, directory, false)
    }
    pub fn open_native(label: &str, directory: &Path) -> Result<Arc<Self>> {
        Self::open_inner(label, directory, true)
    }
    fn open_inner(label: &str, directory: &Path, native: bool) -> Result<Arc<Self>> {
        Self::open_with_chunk(label, directory, native, 4 * 1024 * 1024)
    }
    pub fn open_with_chunk(
        label: &str,
        directory: &Path,
        native: bool,
        chunk: usize,
    ) -> Result<Arc<Self>> {
        let named = matches!(label, "08" | "27b");
        let raw: &[u8] = match label {
            "08" => include_bytes!("../../src/qwen35_cuda/source08.json"),
            "2b" => include_bytes!("../../src/qwen35_cuda/source2b.json"),
            "27b" => include_bytes!("../../src/qwen27b_gpu/source.json"),
            _ => {
                return Err(Error::Unsupported(
                    "explicit original model source fixture".into(),
                ));
            }
        };
        need(
            directory.is_absolute(),
            "explicit approved source directory",
        )?;
        let host = TransferAdmission {
            arbiter: ResourceArbiter::process_shared(),
            resource: PhysicalDeviceKey::new("host", "system-ram")?,
            owner: ResourceOwner::new("source-text-diagnostic", "original-source-control")?,
        };
        host.arbiter
            .register_matching(host.resource.clone(), 16 * 1024 * 1024 * 1024)?;
        let control = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 32 * 1024 * 1024,
            }],
        )?;
        let snapshot = if native {
            Some(super::source_import::open_native(label, directory)?)
        } else {
            None
        };
        let pins: Value =
            serde_json::from_slice(raw).map_err(|e| Error::Malformed(e.to_string()))?;
        let mut members = Vec::new();
        let mut files = BTreeMap::new();
        let mut stamps = BTreeMap::new();
        for row in pins["files"]
            .as_array()
            .ok_or_else(|| Error::Malformed("source fixture members".into()))?
        {
            let name = text(row, if named { "name" } else { "path" })?.to_owned();
            zerogravity::cache::paths::validate_relative_path(&name)?;
            let bytes = row["bytes"]
                .as_u64()
                .ok_or_else(|| Error::Malformed("source member bytes".into()))?;
            let sha = text(row, if named { "sha256" } else { "digest" })?.to_owned();
            let file = if let Some(snapshot) = &snapshot {
                // Dup the original pinned description; do not reopen its pathname.
                snapshot.file(&name)?.try_clone()?
            } else {
                std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(0o400000 | 0o2000000)
                    .open(directory.join(&name))?
            };
            let before = stamp(&file)?;
            need(
                before[2] == bytes && hash(&file, bytes)? == sha && stamp(&file)? == before,
                "exact stable approved source member",
            )?;
            stamps.insert(name.clone(), before);
            files.insert(name.clone(), file);
            members.push(Member { name, bytes, sha });
        }
        let bounded = |name: &str, limit: u64| -> Result<(Vec<u8>, String)> {
            let member = members
                .iter()
                .find(|m| m.name == name)
                .ok_or_else(|| Error::State("source member absent".into()))?;
            need(member.bytes <= limit, "bounded source metadata")?;
            let mut b = vec![0u8; member.bytes as usize];
            files[name].read_exact_at(&mut b, 0)?;
            need(
                format!("{:x}", Sha256::digest(&b)) == member.sha,
                "same held source metadata",
            )?;
            Ok((b, member.sha.clone()))
        };
        let (config, config_sha) = bounded("config.json", 65536)?;
        let (index, index_sha) = bounded("model.safetensors.index.json", 256 * 1024)?;
        let shards = members
            .iter()
            .filter(|m| m.name.ends_with(".safetensors"))
            .map(|m| {
                ExpectedSourceShard::new(
                    m.name.clone(),
                    Artifact::from_file(files[&m.name].try_clone()?, Path::new(&m.name))?,
                    m.bytes,
                    m.sha.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let prepared = SourceWeights::prepare_verified_windows(
            &index,
            &index_sha,
            shards,
            WindowOpenLimits {
                max_source_bytes: if label == "27b" { 64 } else { 8 } * 1024 * 1024 * 1024,
                max_metadata_bytes: 64 * 1024 * 1024,
                max_shards: if label == "27b" { 18 } else { 2 },
                max_tensors: 2048,
                scan_chunk_bytes: chunk,
            },
            &host,
        )?;
        let source = prepared.open()?;
        let model = Arc::new(OriginalWindowOperators::bind(
            source,
            &config,
            &config_sha,
            &index_sha,
            QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
        )?);
        Ok(Arc::new(Self {
            model,
            members,
            host,
            files,
            stamps,
            snapshot,
            _control: control,
        }))
    }
    pub fn cache_witness(&self) -> Result<std::sync::Weak<SnapshotLease>> {
        self.snapshot
            .as_ref()
            .map(Arc::downgrade)
            .ok_or_else(|| Error::State("native managed snapshot absent".into()))
    }
    pub fn close(self: Arc<Self>) -> Result<()> {
        need(
            Arc::strong_count(&self) == 1,
            "original source retains native owners",
        )?;
        self.verify_after()
    }
    pub fn verify_after(&self) -> Result<()> {
        for member in &self.members {
            let f = &self.files[&member.name];
            let before = self.stamps[&member.name];
            need(
                stamp(f)? == before && hash(f, member.bytes)? == member.sha && stamp(f)? == before,
                "unchanged original source through diagnostic",
            )?;
        }
        Ok(())
    }
}
