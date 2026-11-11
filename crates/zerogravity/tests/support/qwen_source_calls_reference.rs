//! Retained original multi-row calls. Reference bytes are read-only assertions.
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
#[allow(dead_code)]
#[path = "qwen_source_calls_reference/checkpoint.rs"]
mod checkpoint;
#[path = "qwen_source_calls_reference/geometry.rs"]
mod geometry;
#[path = "qwen_source_calls_reference/observer.rs"]
mod observation;
#[path = "qwen_source_reference_scope.rs"]
mod scope;
pub use observation::{ObservationMap, Observer};
pub use scope::ReferenceScope;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::State(message.into()))
    }
}

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
    #[serde(default = "captured")]
    pub checkpoint: bool,
    #[serde(default)]
    pub logits_sha256: Option<String>,
}
fn captured() -> bool {
    true
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
    #[serde(default)]
    prompts: Vec<String>,
}
type Key = (usize, usize, usize, String, Option<usize>);
pub struct Expected {
    doc: Document,
    index: BTreeMap<Key, usize>,
    raw: File,
    stamp: [u64; 7],
    layers: usize,
    domains: usize,
    planes: usize,
    _host: GlobalResourceLease,
}
pub(super) fn stamp(f: &File) -> Result<[u64; 7]> {
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
pub(super) fn open(path: &Path) -> Result<File> {
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
            "27b" => {
                let digest =
                    std::env::var("ZG_QWEN_SOURCE_CALLS_MANIFEST_SHA256").map_err(|_| {
                        Error::Malformed(
                            "separately audited original27B manifest digest required".into(),
                        )
                    })?;
                return Self::open_pinned(directory, "Qwen/Qwen3.8-27B", &digest, None, false);
            }
            _ => return Err(Error::Unsupported("source-call reference model".into())),
        };
        Self::open_pinned(
            directory,
            repository,
            manifest_sha,
            Some((raw_sha, raw_bytes)),
            false,
        )
    }
    /// New original captures supply a separately verified immutable manifest digest.
    /// Historical 66-call pins and strict scope above remain unchanged.
    pub fn open_pinned(
        directory: &Path,
        repository: &str,
        manifest_sha: &str,
        raw_pin: Option<(&str, u64)>,
        populated: bool,
    ) -> Result<Self> {
        let scope = ReferenceScope::for_repository(repository)?;
        let layers = scope.layers;
        let planes = 2 * layers;
        let domains = layers / 4 * 7;
        let archive_cap = if layers == 64 { 64 } else { 16 } * 1024 * 1024 * 1024u64;
        need(
            manifest_sha.len() == 64 && manifest_sha.bytes().all(|b| b.is_ascii_hexdigit()),
            "explicit original reference digest",
        )?;
        let (schema, capacity, cases, calls_count, records_count) = if populated {
            let (calls, records) = scope.populated_counts();
            (
                "zg.original-hf-source-context-reference.v1",
                scope.active_context(),
                scope.contexts.len(),
                calls,
                records,
            )
        } else {
            (
                "zg.original-hf-source-call-reference.v1",
                514,
                6,
                66,
                2 * (6 * (layers + 4 + planes) + 27 * (layers + 4 + 2 * planes)),
            )
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
            doc.schema == schema
                && doc.status == "CAPTURED_ORIGINAL_SOURCE_CALL_REFERENCE_ONLY"
                && doc.repository == repository
                && doc.active_context == capacity
                && doc.repeats == 2
                && !doc.rust_or_gpu_parity
                && (0 < doc.raw_bytes && doc.raw_bytes <= archive_cap)
                && raw_pin
                    .is_none_or(|(sha, bytes)| doc.raw_bytes == bytes && doc.raw_sha256 == sha)
                && doc.calls.len() == calls_count
                && doc.records.len() == records_count
                && doc.prompts.len() == if populated { cases } else { 0 }
                && doc.numerical_profile == "TorchCpuBf16Avx2MixedSourceV1"
                && doc.original_full_forward
                && doc.all_original_source_and_mapped_bytes_unchanged,
            "source-call original scope",
        )?;
        let raw_bytes = doc.raw_bytes;
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
            format!("{:x}", h.finalize()) == doc.raw_sha256 && stamp(&raw)? == before,
            "source-call raw full digest/stable inode",
        )?;
        let mut calls = BTreeMap::new();
        let mut frontiers = BTreeMap::new();
        for c in &doc.calls {
            let frontier = frontiers.entry((c.repeat, c.case)).or_insert((0, 0));
            need(
                c.repeat < 2
                    && c.case < cases
                    && c.call == frontier.0
                    && c.position == frontier.1
                    && c.rows == c.input_ids.len()
                    && (1..=64).contains(&c.rows)
                    && c.position
                        .checked_add(c.rows)
                        .is_some_and(|n| n <= capacity)
                    && c.input_ids.iter().all(|id| *id < 248320)
                    && c.selected < 248320
                    && (!populated
                        || c.logits_sha256.as_ref().is_some_and(|s| {
                            s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
                        }))
                    && matches!(c.phase.as_str(), "prefill" | "decode")
                    && (c.phase != "decode" || c.rows == 1 && c.position > 0),
                "source-call schedule and actual frontiers",
            )?;
            need(calls.insert(c.key(), c).is_none(), "unique source call")?;
            *frontier = (c.call + 1, c.position + c.rows);
        }
        need(
            frontiers.len() == 2 * cases,
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
                c.checkpoint
                    && r.offset == end
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
        let result = Self {
            doc,
            index,
            raw,
            stamp: before,
            layers,
            domains,
            planes,
            _host: host,
        };
        if populated {
            for (case, &capacity) in scope.contexts.iter().enumerate() {
                need(
                    result.prompt(case).is_some_and(|p| !p.is_empty()),
                    "original context text present",
                )?;
                for repeat in 0..2 {
                    let group = result
                        .calls()
                        .iter()
                        .filter(|c| c.case == case && c.repeat == repeat)
                        .collect::<Vec<_>>();
                    let prefill = (capacity - 2) / 64 + 1;
                    need(
                        group.len() == prefill + 2,
                        "complete original populated trajectory",
                    )?;
                    for (i, c) in group.into_iter().enumerate() {
                        let last_prefill = i == prefill - 1;
                        let (phase, rows) = if i >= prefill {
                            ("decode", 1)
                        } else {
                            ("prefill", if last_prefill { 62 } else { 64 })
                        };
                        need(
                            c.phase == phase
                                && c.rows == rows
                                && c.checkpoint == (i >= prefill - 1),
                            "original64-row population and final three checkpoints",
                        )?;
                    }
                    need(
                        frontiers[&(repeat, case)].1 == capacity,
                        "actual populated final context",
                    )?;
                }
            }
        }
        Ok(result)
    }
    pub fn prompt(&self, case: usize) -> Option<&str> {
        self.doc.prompts.get(case).map(String::as_str)
    }
    pub fn domain_count(&self) -> usize {
        self.domains
    }
    pub fn plane_count(&self) -> usize {
        self.planes
    }
    pub fn calls(&self) -> &[Call] {
        &self.doc.calls
    }
    pub fn require_source<'a, I>(
        &self,
        members: I,
        config: &zerogravity::qwen35::source::safetensors::QwenTextConfig,
    ) -> Result<()>
    where
        I: IntoIterator<Item = (&'a str, u64, &'a str)>,
    {
        let members = members.into_iter().collect::<Vec<_>>();
        need(
            self.doc.members.len() == members.len()
                && self
                    .doc
                    .members
                    .iter()
                    .zip(&members)
                    .all(|(a, b)| a.name == b.0 && a.bytes == b.1 && a.sha256 == b.2),
            "source-call same approved source members",
        )?;
        self.require_geometry(config)
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
