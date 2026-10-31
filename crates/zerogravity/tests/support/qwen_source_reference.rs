//! Authenticated original-HF records; expected bytes never become operands.
use super::source::{Loaded, need};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt},
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};
use zerogravity::accelerator::program::{ConvDomain, RecurrentDomain, StateSlot};
use zerogravity::gpu::{TextSourceFrame, TextSourceFrameKind, TextSourceObserver};
use zerogravity::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity::{Error, Result};

#[derive(Deserialize)]
struct Record {
    repeat: usize,
    position: usize,
    token: u32,
    label: String,
    layer: Option<usize>,
    shape: Vec<usize>,
    dtype: String,
    offset: u64,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
pub struct Call {
    pub repeat: usize,
    pub position: usize,
    pub token: u32,
    pub selected: u32,
    pub max_bits: u32,
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
}
type Key = (usize, usize, String, Option<usize>);
pub struct Expected {
    doc: Document,
    layers: usize,
    index: BTreeMap<Key, usize>,
    raw: File,
    stamp: [u64; 7],
    _host: GlobalResourceLease,
}
fn stamp(f: &File) -> Result<[u64; 7]> {
    let m = f.metadata()?;
    need(m.is_file(), "original reference regular file")?;
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
                "0d6baff2a278bc26a3ecba4cf1c8d70c2192320b062594ee7329213a4f2d1644",
                "961504e63717488859eeffa46d0b9feccbc17c47ec332a3990ef43b692027271",
                5_189_864_448,
                "Qwen/Qwen3.5-0.8B",
            ),
            "2b" => (
                "6d0be96e734595b075a2f907f0b6342462a65b84cd18b3648d7cfb871f5d2ebf",
                "5783accbccefd5a96f10711c854efaeab50a4696cfad3c3c24f4bf7a711955b9",
                5_196_680_192,
                "Qwen/Qwen3.5-2B",
            ),
            _ => return Err(Error::Unsupported("original reference fixture".into())),
        };
        let arbiter = ResourceArbiter::process_shared();
        let key = PhysicalDeviceKey::new("host", "system-ram")?;
        arbiter.register_matching(key.clone(), 16 * 1024 * 1024 * 1024)?;
        let host = arbiter.admit(
            &key,
            ResourceOwner::new("source-text-diagnostic", "independent-reference")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 64 * 1024 * 1024,
            }],
        )?;
        let mut bytes = Vec::new();
        open(&directory.join("reference.json"))?
            .take(16 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)?;
        need(
            bytes.len() <= 16 * 1024 * 1024
                && format!("{:x}", Sha256::digest(&bytes)) == manifest_sha,
            "independent original reference manifest identity",
        )?;
        let doc: Document =
            serde_json::from_slice(&bytes).map_err(|e| Error::Malformed(e.to_string()))?;
        need(
            doc.schema == "zg.original-hf-source-token-reference.v1"
                && doc.status == "CAPTURED_ORIGINAL_SOURCE_REFERENCE_ONLY"
                && doc.repository == repository
                && doc.active_context == 64
                && doc.repeats == 2
                && !doc.rust_or_gpu_parity
                && doc.raw_sha256 == raw_sha
                && doc.raw_bytes == raw_bytes
                && doc.records.len() == 15776
                && doc.calls.len() == 128,
            "original complete trained-model scope",
        )?;
        let raw = open(&directory.join("raw.bin"))?;
        let before = stamp(&raw)?;
        need(before[2] == raw_bytes, "original raw complete extent")?;
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
            "original raw full digest/stable held inode",
        )?;
        let mut index = BTreeMap::new();
        let mut end = 0;
        for (i, r) in doc.records.iter().enumerate() {
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
                r.repeat < 2
                    && r.position < 64
                    && r.offset == end
                    && r.bytes == n * width
                    && r.bytes <= 16 * 1024 * 1024,
                "reference exact typed record extents",
            )?;
            need(
                index
                    .insert((r.repeat, r.position, r.label.clone(), r.layer), i)
                    .is_none(),
                "unique original record",
            )?;
            end += r.bytes as u64;
        }
        need(end == raw_bytes, "reference covers entire original archive")?;
        Ok(Self {
            layers: doc
                .records
                .iter()
                .filter(|r| r.label == "layer_output")
                .filter_map(|r| r.layer)
                .max()
                .ok_or_else(|| Error::State("reference layer outputs absent".into()))?
                + 1,
            doc,
            index,
            raw,
            stamp: before,
            _host: host,
        })
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
            "original reference same approved source members",
        )?;
        need(
            self.layers == loaded.model.binding().config().geometry().layers.len(),
            "same complete source layer extent",
        )
    }
    pub fn call(&self, repeat: usize, position: usize) -> Result<&Call> {
        let call = self
            .doc
            .calls
            .get(repeat * 64 + position)
            .ok_or_else(|| Error::Limit("reference call".into()))?;
        need(
            call.repeat == repeat && call.position == position,
            "exact reference call",
        )?;
        Ok(call)
    }
    pub fn observe<'a>(
        &'a mut self,
        repeat: usize,
        position: usize,
        token: u32,
        late_cancel: Option<&'a AtomicBool>,
    ) -> Result<Observer<'a>> {
        need(
            self.call(repeat, position)?.token == token,
            "actual natural input equals independent original trajectory",
        )?;
        let expected = self
            .doc
            .records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| (r.repeat == repeat && r.position == position).then_some(i))
            .collect::<BTreeSet<_>>();
        let mut zero_required = BTreeMap::new();
        if position == 0 {
            for &i in &expected {
                let r = &self.doc.records[i];
                let slot = match r.label.as_str() {
                    "after.conv4" => {
                        Some(StateSlot::Conv(ConvDomain(r.layer.ok_or_else(|| {
                            Error::State("reference Conv4 layer".into())
                        })?)))
                    }
                    "after.recurrent" => Some(StateSlot::Recurrent(RecurrentDomain(
                        r.layer
                            .ok_or_else(|| Error::State("reference recurrent layer".into()))?,
                    ))),
                    _ => None,
                };
                if let Some(slot) = slot {
                    zero_required.insert(slot, r.bytes / if r.dtype == "BF16" { 2 } else { 4 });
                }
            }
        }
        let mut observer = Observer {
            zero_required,
            zero_seen: BTreeMap::new(),
            reference: self,
            repeat,
            position,
            token,
            expected,
            done: BTreeSet::new(),
            active: None,
            raw: Vec::new(),
            next: 0,
            words: 0,
            initial_zero_words: 0,
            mismatches: 0,
            mismatch_fields: BTreeMap::new(),
            first: None,
            late_cancel,
        };
        let i = observer.record("positions", None)?;
        observer.load(i)?;
        need(
            observer.raw.len() == 24
                && observer
                    .raw
                    .chunks_exact(8)
                    .all(|b| i64::from_le_bytes(b.try_into().unwrap()) == position as i64),
            "original actual rotary positions",
        )?;
        observer.done.insert(i);
        observer.active = None;
        Ok(observer)
    }
}
pub struct Observer<'a> {
    zero_required: BTreeMap<StateSlot, usize>,
    zero_seen: BTreeMap<StateSlot, usize>,
    reference: &'a mut Expected,
    repeat: usize,
    position: usize,
    token: u32,
    expected: BTreeSet<usize>,
    done: BTreeSet<usize>,
    active: Option<usize>,
    raw: Vec<u8>,
    next: usize,
    pub words: usize,
    pub initial_zero_words: usize,
    pub mismatches: usize,
    mismatch_fields: BTreeMap<(String, Option<usize>), (usize, usize)>,
    first: Option<String>,
    late_cancel: Option<&'a AtomicBool>,
}
impl Observer<'_> {
    fn record(&self, label: &str, layer: Option<usize>) -> Result<usize> {
        self.reference
            .index
            .get(&(self.repeat, self.position, label.to_owned(), layer))
            .copied()
            .ok_or_else(|| Error::State(format!("original record absent {label}/{layer:?}")))
    }
    fn load(&mut self, i: usize) -> Result<()> {
        let r = &self.reference.doc.records[i];
        need(
            r.token == self.token && stamp(&self.reference.raw)? == self.reference.stamp,
            "same source call/reference inode",
        )?;
        self.raw.resize(r.bytes, 0);
        self.reference.raw.read_exact_at(&mut self.raw, r.offset)?;
        need(
            format!("{:x}", Sha256::digest(&self.raw)) == r.sha256
                && stamp(&self.reference.raw)? == self.reference.stamp,
            "complete original record digest",
        )?;
        Ok(())
    }
    pub fn finish(&self) -> Result<()> {
        need(
            self.active.is_none()
                && self.done == self.expected
                && self.zero_seen == self.zero_required,
            "all original producer/state/position/logit records observed",
        )?;
        if self.mismatches != 0 {
            for ((label, layer), (words, signed_zeros)) in &self.mismatch_fields {
                eprintln!(
                    "SOURCE_MODEL_MISMATCH repeat={} position={} label={label} layer={layer:?} words={words} signed_zeros={signed_zeros}",
                    self.repeat, self.position
                );
            }
            return Err(Error::State(format!(
                "source numerical mismatches={} first={:?}",
                self.mismatches, self.first
            )));
        }
        Ok(())
    }
}
impl TextSourceObserver for Observer<'_> {
    fn completed(&mut self, frame: TextSourceFrame, actual: &[u8]) -> Result<()> {
        need(
            frame.position == self.position && actual.len().is_multiple_of(4),
            "actual source frame position/codec",
        )?;
        let (label, layer, kv) = match frame.kind {
            TextSourceFrameKind::Value { layer, stage } => match stage {
                "embedding" => {
                    need(layer == 0, "source embedding coordinate")?;
                    ("embedding".to_owned(), None, false)
                }
                "layer_output" => ("layer_output".to_owned(), Some(layer), false),
                "final_norm" => {
                    need(
                        layer + 1 == self.reference.layers,
                        "actual final source layer",
                    )?;
                    ("normalized".to_owned(), None, false)
                }
                "logits" => {
                    need(layer == usize::MAX, "source model-level output coordinate")?;
                    ("logits".to_owned(), None, false)
                }
                _ => return Err(Error::State("unexpected source value observation".into())),
            },
            TextSourceFrameKind::State { slot, plane, prior } => {
                if prior && self.position == 0 {
                    let total = self.zero_required.get(&slot).copied().ok_or_else(|| {
                        Error::State("unexpected initial zero-control domain".into())
                    })?;
                    let next = self.zero_seen.entry(slot).or_default();
                    need(
                        plane == 0
                            && frame.total_words == total
                            && frame.offset_words == *next
                            && !actual.is_empty()
                            && *next + actual.len() / 4 <= total
                            && actual.iter().all(|b| *b == 0),
                        "complete canonical initial root zero control (not an original populated-state record)",
                    )?;
                    *next += actual.len() / 4;
                    self.initial_zero_words += actual.len() / 4;
                    return Ok(());
                }
                let (name, layer, kv) = match slot {
                    StateSlot::Conv(d) => {
                        need(plane == 0, "source Conv4 plane")?;
                        ("conv4", d.0, false)
                    }
                    StateSlot::Recurrent(d) => {
                        need(plane == 0, "source recurrent plane")?;
                        ("recurrent", d.0, false)
                    }
                    StateSlot::Kv(d) => {
                        need(plane < 2, "source KV plane")?;
                        (if plane == 0 { "key" } else { "value" }, d.0, true)
                    }
                };
                (
                    format!("{}.{name}", if prior { "before" } else { "after" }),
                    Some(layer),
                    kv,
                )
            }
        };
        let i = self.record(&label, layer)?;
        if self.active != Some(i) {
            need(
                self.active.is_none() && !self.done.contains(&i) && frame.offset_words == 0,
                "source observation producer/chunk sequence",
            )?;
            self.load(i)?;
            self.active = Some(i);
            self.next = 0;
        }
        let r = &self.reference.doc.records[i];
        let width = if r.dtype == "BF16" {
            2
        } else {
            need(r.dtype == "F32", "source floating record")?;
            4
        };
        need(
            frame.total_words == r.bytes / width
                && frame.offset_words == self.next
                && self.next + actual.len() / 4 <= frame.total_words,
            "source full extent and contiguous completed chunks",
        )?;
        for (offset, b) in actual.chunks_exact(4).enumerate() {
            let word = self.next + offset;
            let original = if kv {
                let heads = r.shape[1];
                let active = r.shape[2];
                let width = r.shape[3];
                ((word / width) % heads * active + word / (heads * width)) * width + word % width
            } else {
                word
            };
            let expected = if width == 2 {
                u32::from(u16::from_le_bytes(
                    self.raw[original * 2..original * 2 + 2].try_into().unwrap(),
                )) << 16
            } else {
                u32::from_le_bytes(self.raw[original * 4..original * 4 + 4].try_into().unwrap())
            };
            let actual = u32::from_le_bytes(b.try_into().unwrap());
            if actual != expected {
                self.mismatches += 1;
                let counts = self
                    .mismatch_fields
                    .entry((label.clone(), layer))
                    .or_default();
                counts.0 += 1;
                counts.1 += usize::from((actual | expected) & 0x7fff_ffff == 0);
                self.first.get_or_insert_with(||format!("repeat={} position={} {label}/{layer:?} word={word} actual={actual:08x} expected={expected:08x}",self.repeat,self.position));
            }
        }
        self.next += actual.len() / 4;
        self.words += actual.len() / 4;
        if self.next == frame.total_words {
            self.done.insert(i);
            self.active = None;
            if label == "logits"
                && let Some(flag) = self.late_cancel
            {
                flag.store(true, Ordering::Release);
            }
        }
        Ok(())
    }
}
