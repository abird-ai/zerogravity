//! Retained original multi-row calls. Reference bytes are read-only assertions.
use super::source::{Loaded, need};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::Path,
};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity::{Error, Result};
#[path = "qwen_source_calls_reference/checkpoint.rs"]
mod checkpoint;
#[path = "qwen_source_calls_reference/observer.rs"]
mod observation;
pub use observation::{ObservationMap, Observer};

#[derive(Deserialize)]
struct Record {
    repeat: usize,
    case: usize,
    call: usize,
    position: usize,
    rows: usize,
    phase: String,
    label: String,
    layer: Option<usize>,
    shape: Vec<usize>,
    dtype: String,
    offset: u64,
    bytes: usize,
    sha256: String,
}
#[derive(Clone, Deserialize)]
pub struct Call {
    pub repeat: usize,
    pub case: usize,
    pub call: usize,
    pub phase: String,
    pub position: usize,
    pub rows: usize,
    pub input_ids: Vec<u32>,
    pub selected: u32,
    pub max_bits: u32,
}
impl Call {
    pub fn key(&self) -> (usize, usize, usize) {
        (self.repeat, self.case, self.call)
    }
}
#[derive(Deserialize)]
struct Member {
    name: String,
    bytes: u64,
    sha256: String,
}
#[derive(Deserialize)]
struct Document {
    schema: String,
    status: String,
    repository: String,
    members: Vec<Member>,
    records: Vec<Record>,
    calls: Vec<Call>,
    raw_bytes: u64,
    raw_sha256: String,
    active_context: usize,
    repeats: usize,
    rust_or_gpu_parity: bool,
    numerical_profile: String,
    original_full_forward: bool,
    all_original_source_and_mapped_bytes_unchanged: bool,
}
type Key = (usize, usize, usize, String, Option<usize>);
pub struct Expected {
    doc: Document,
    index: BTreeMap<Key, usize>,
    raw: File,
    stamp: [u64; 7],
    _host: GlobalResourceLease,
}
fn stamp(f: &File) -> Result<[u64; 7]> {
    let m = f.metadata()?;
    need(m.is_file(), "source-call reference regular file")?;
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
fn open(path: &Path) -> Result<File> {
    Ok(std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(0o400000 | 0o2000000)
        .open(path)?)
}
impl Expected {
    pub fn open(directory: &Path, label: &str) -> Result<Self> {
        let (manifest_sha, raw_sha, raw_bytes, repository) = match label {
            "08" => (
                "492fd2b361f79cec3b2443c5454850dc91aef6d881e3c41330e2d50f3c544d3d",
                "534a0313fb15a970e5c140b83fba4a916877d6d7aad111e297166cf0d4cd6fbd",
                2_879_819_424,
                "Qwen/Qwen3.5-0.8B",
            ),
            "2b" => (
                "1a299081a66d386c9f86e8b6e71e44fd653ae6424a1709a93184d3cda48e8c96",
                "595d139b771855ab10279a750538023536ce2fdb3f62b39501510422484a1b26",
                3_010_809_504,
                "Qwen/Qwen3.5-2B",
            ),
            _ => return Err(Error::Unsupported("source-call reference model".into())),
        };
        let arbiter = ResourceArbiter::process_shared();
        let key = PhysicalDeviceKey::new("host", "system-ram")?;
        arbiter.register_matching(key.clone(), 16 * 1024 * 1024 * 1024)?;
        let host = arbiter.admit(
            &key,
            ResourceOwner::new("source-call-diagnostic", "independent-reference")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 64 * 1024 * 1024,
            }],
        )?;
        let mut bytes = Vec::new();
        let manifest = open(&directory.join("reference.json"))?;
        let manifest_stamp = stamp(&manifest)?;
        (&manifest)
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        need(
            bytes.len() <= 16 * 1024 * 1024
                && stamp(&manifest)? == manifest_stamp
                && format!("{:x}", Sha256::digest(&bytes)) == manifest_sha,
            "source-call manifest identity/stable inode",
        )?;
        let doc: Document =
            serde_json::from_slice(&bytes).map_err(|e| Error::Malformed(e.to_string()))?;
        need(
            doc.schema == "zg.original-hf-source-call-reference.v1"
                && doc.status == "CAPTURED_ORIGINAL_SOURCE_CALL_REFERENCE_ONLY"
                && doc.repository == repository
                && doc.active_context == 514
                && doc.repeats == 2
                && !doc.rust_or_gpu_parity
                && doc.raw_bytes == raw_bytes
                && doc.raw_sha256 == raw_sha
                && doc.calls.len() == 66
                && doc.records.len() == 7608
                && doc.numerical_profile == "TorchCpuBf16Avx2MixedSourceV1"
                && doc.original_full_forward
                && doc.all_original_source_and_mapped_bytes_unchanged,
            "source-call original scope",
        )?;
        let raw = open(&directory.join("raw.bin"))?;
        let before = stamp(&raw)?;
        need(before[2] == raw_bytes, "source-call raw extent")?;
        let mut h = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        let mut at = 0;
        while at < raw_bytes {
            let n = (raw_bytes - at).min(buffer.len() as u64) as usize;
            raw.read_exact_at(&mut buffer[..n], at)?;
            h.update(&buffer[..n]);
            at += n as u64;
        }
        need(
            format!("{:x}", h.finalize()) == raw_sha && stamp(&raw)? == before,
            "source-call raw full digest/stable inode",
        )?;
        let mut calls = BTreeMap::new();
        let mut frontiers = BTreeMap::new();
        for c in &doc.calls {
            let frontier = frontiers.entry((c.repeat, c.case)).or_insert((0, 0));
            need(
                c.repeat < 2
                    && c.case < 6
                    && c.call == frontier.0
                    && c.position == frontier.1
                    && c.rows == c.input_ids.len()
                    && (1..=64).contains(&c.rows)
                    && c.position + c.rows <= 514
                    && c.input_ids.iter().all(|id| *id < 248320)
                    && c.selected < 248320
                    && matches!(c.phase.as_str(), "prefill" | "decode")
                    && (c.phase != "decode" || c.rows == 1 && c.position > 0),
                "source-call schedule and actual frontiers",
            )?;
            need(calls.insert(c.key(), c).is_none(), "unique source call")?;
            *frontier = (c.call + 1, c.position + c.rows);
        }
        need(
            frontiers.len() == 12,
            "source-call independent repeat/case groups",
        )?;
        let mut index = BTreeMap::new();
        let mut end = 0;
        for (i, r) in doc.records.iter().enumerate() {
            let c = calls
                .get(&(r.repeat, r.case, r.call))
                .ok_or_else(|| Error::Malformed("record without original call".into()))?;
            let n = r
                .shape
                .iter()
                .try_fold(1usize, |a, b| a.checked_mul(*b))
                .ok_or_else(|| Error::Limit("reference shape overflow".into()))?;
            let width = match r.dtype.as_str() {
                "BF16" => 2,
                "F32" => 4,
                "I64" => 8,
                _ => return Err(Error::Malformed("reference codec".into())),
            };
            need(
                r.offset == end
                    && n.checked_mul(width) == Some(r.bytes)
                    && r.bytes <= 16 * 1024 * 1024
                    && r.position == c.position
                    && r.rows == c.rows
                    && r.phase == c.phase,
                "source-call exact record extent/identity",
            )?;
            need(
                index
                    .insert((r.repeat, r.case, r.call, r.label.clone(), r.layer), i)
                    .is_none(),
                "unique source-call record",
            )?;
            end += r.bytes as u64;
        }
        need(end == raw_bytes, "source-call entire archive coverage")?;
        Ok(Self {
            doc,
            index,
            raw,
            stamp: before,
            _host: host,
        })
    }
    pub fn calls(&self) -> &[Call] {
        &self.doc.calls
    }
    pub fn require_source(&self, loaded: &Loaded) -> Result<()> {
        need(
            self.doc.members.len() == loaded.members.len()
                && self
                    .doc
                    .members
                    .iter()
                    .zip(&loaded.members)
                    .all(|(a, b)| a.name == b.name && a.bytes == b.bytes && a.sha256 == b.sha),
            "source-call same approved source members",
        )
    }
    pub fn unchanged(&self) -> Result<()> {
        need(
            stamp(&self.raw)? == self.stamp,
            "source-call reference held inode unchanged",
        )
    }
    fn find(&self, c: &Call, label: &str, layer: Option<usize>) -> Result<usize> {
        self.index
            .get(&(c.repeat, c.case, c.call, label.into(), layer))
            .copied()
            .ok_or_else(|| Error::State(format!("missing source-call record {label}/{layer:?}")))
    }
    fn load(&self, i: usize) -> Result<Vec<u8>> {
        self.unchanged()?;
        let r = &self.doc.records[i];
        let mut b = vec![0; r.bytes];
        self.raw.read_exact_at(&mut b, r.offset)?;
        need(
            format!("{:x}", Sha256::digest(&b)) == r.sha256,
            "source-call individual record digest",
        )?;
        self.unchanged()?;
        Ok(b)
    }
}
