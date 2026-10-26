//! Immutable expected bytes are readable only by borrowed output comparisons.
use super::{comparison::Difference, io::*};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};
use zerogravity::{
    Error, Result,
    accelerator::{SourceIdentity, program::KvDomain},
};

const EXPECTED: Option<&str> =
    Some("be800bc6e97015290b12bb8f1b756ff9490db34ee905b920937236a0b2f4dbe3");
#[derive(Deserialize)]
pub struct Request {
    pub cases: Vec<Case>,
}
#[derive(Deserialize, Clone)]
pub struct Case {
    pub name: String,
    pub text: String,
    pub rows: usize,
}
#[derive(Deserialize)]
pub struct Call {
    pub call: usize,
    pub input_ids: Vec<u32>,
    pub base: usize,
    pub frontier: usize,
    pub selected_id: u32,
    pub eos: bool,
}
#[derive(Deserialize)]
pub struct Record {
    pub call: usize,
    pub label: String,
    pub shape: Vec<usize>,
    pub bytes: usize,
    pub offset: u64,
    pub sha256: String,
    pub dtype: String,
    pub layer: Option<usize>,
    pub base: Option<usize>,
    pub frontier: Option<usize>,
}
#[derive(Deserialize)]
pub struct Archive {
    pub name: String,
    pub bytes: u64,
    pub sha256: String,
    pub records: Vec<Record>,
}
#[derive(Deserialize)]
pub struct Cell {
    pub schema: String,
    pub request: Case,
    pub prompt_ids: Vec<u32>,
    pub calls: Vec<Call>,
    pub raw: Archive,
    pub boundaries: Archive,
    pub repetitions: usize,
    pub repetition_equal: bool,
    pub boundary_repetitions: usize,
}
struct Held {
    file: File,
    before: [u64; 7],
    sha: String,
}
impl Held {
    fn new(parent: &Path, archive: &Archive, limit: u64) -> Proof<Self> {
        need(
            archive.bytes > 0
                && archive.bytes <= limit
                && Path::new(&archive.name)
                    .file_name()
                    .and_then(|x| x.to_str())
                    == Some(archive.name.as_str()),
            "contained bounded archive",
        )?;
        let file = open(&parent.join(&archive.name))?;
        let before = stamp(&file)?;
        need(
            before[2] == archive.bytes && hash(&file, archive.bytes)? == archive.sha256,
            "complete immutable original archive",
        )?;
        let mut at = 0;
        for r in &archive.records {
            need(
                r.offset == at
                    && r.bytes > 0
                    && r.bytes <= 524288
                    && r.dtype == "bf16-le"
                    && r.shape
                        .iter()
                        .try_fold(1usize, |n, x| n.checked_mul(*x))
                        .and_then(|n| n.checked_mul(2))
                        == Some(r.bytes),
                "whole BF16 record bounds/no gaps",
            )?;
            let mut bytes = vec![0; r.bytes];
            file.read_exact_at(&mut bytes, at)?;
            need(sha(&bytes) == r.sha256, "actual whole record SHA")?;
            for w in bytes.chunks_exact(2) {
                need(
                    u16::from_le_bytes(w.try_into()?) & 0x7f80 != 0x7f80,
                    "finite original BF16",
                )?;
            }
            at += r.bytes as u64;
        }
        need(
            at == archive.bytes && stamp(&file)? == before,
            "complete record union/identity",
        )?;
        Ok(Self {
            file,
            before,
            sha: archive.sha256.clone(),
        })
    }
    fn compare(&self, r: &Record, row: Option<usize>, actual: &[f32]) -> Proof<Difference> {
        let (offset, bytes) = match row {
            Some(i) => {
                need(
                    r.shape == [r.shape[0], 1536] && i < r.shape[0] && actual.len() == 1536,
                    "actual source row",
                )?;
                (r.offset + (i * 3072) as u64, 3072)
            }
            None => {
                need(actual.len() * 2 == r.bytes, "whole output extent")?;
                (r.offset, r.bytes)
            }
        };
        let mut raw = vec![0; bytes];
        self.file.read_exact_at(&mut raw, offset)?;
        super::comparison::scan(&raw, actual)
    }
    fn after(&self) -> Proof<()> {
        need(
            stamp(&self.file)? == self.before && hash(&self.file, self.before[2])? == self.sha,
            "reference after completed callbacks",
        )
    }
}
pub struct Matrix {
    parent: PathBuf,
    pub request: Request,
    capture: Value,
}
impl Matrix {
    pub fn open(path: &Path) -> Proof<Self> {
        let raw = read(path, 65536, None)?;
        need(
            Some(sha(&raw).as_str()) == EXPECTED,
            "fresh explicit healthy matrix expected-only pin",
        )?;
        let receipt: Value = serde_json::from_slice(&raw)?;
        need(
            receipt["status"] == "EXECUTED_ORIGINAL_GEMMA4_MATRIX_EXPECTED_ONLY"
                && receipt["exit_code"] == 0
                && receipt["source_before"] == receipt["source_after"]
                && receipt["source_FDs_closed"] == true
                && receipt["cleanup_errors"] == json!([])
                && receipt["source_FD_cleanup_errors"] == json!([])
                && receipt["log_cleanup"]["all_closed"] == true
                && receipt["log_cleanup"]["pipe_eof"] == true
                && receipt["foreground_cleanup"]["leader_reaped"] == true
                && receipt["foreground_cleanup"]["group_absent"] == true
                && receipt["owned_container_absent_after_reap"] == true,
            "whole actual original source/owned closure",
        )?;
        need(
            receipt["image_id"]
                == "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe"
                && receipt["image_digest"]
                    == "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
            "approved original image",
        )?;
        let pins: Value = serde_json::from_slice(include_bytes!(
            "../../../zerogravity-model-gemma4/src/execution/source.json"
        ))?;
        need(receipt["source"] == pins, "same complete managed source")?;
        let parent = path.parent().ok_or("matrix parent")?.to_owned();
        let capture = json(
            &parent.join("capture.json"),
            65536,
            Some(&receipt["capture"]),
        )?;
        let request: Value = serde_json::from_slice(include_bytes!(
            "../../../../tools/gemma4-e2b-ordinary-reference/request.json"
        ))?;
        need(
            capture["schema"] == "zg.gemma4-e2b.ordinary-matrix.v1"
                && capture["source"] == pins
                && capture["request"] == request
                && capture["profile"] == "HfTorchCpuBf16Avx2Eager5.17.0"
                && capture["transformers"] == "5.17.0"
                && capture["torch_git"] == "cf30153c4c131c8164ee7798e5022d810682e2cb"
                && capture["eos_ids"] == json!([1, 106, 50])
                && capture["plain_vs_observed_equal"] == true
                && capture["trained_rust_qualified"] == false
                && capture["native_serving_chat_or_performance_qualified"] == false
                && capture["cells"].as_array().is_some_and(|x| x.len() == 7),
            "entire independent original matrix, no authority widening",
        )?;
        Ok(Self {
            parent,
            request: serde_json::from_value(request)?,
            capture,
        })
    }
    pub fn cell(&self, index: usize, source: SourceIdentity) -> Proof<Expected> {
        let case = &self.request.cases[index];
        let pin = &self.capture["cells"][index]["metadata"];
        need(
            pin["name"] == format!("{}.json", case.name),
            "exact cell metadata",
        )?;
        let bytes = read(
            &self.parent.join(format!("{}.json", case.name)),
            300000,
            Some(pin),
        )?;
        let cell: Cell = serde_json::from_slice(&bytes)?;
        drop(bytes);
        need(
            cell.schema == "zg.gemma4-e2b.ordinary-cell.v1"
                && cell.request.name == case.name
                && cell.request.text == case.text
                && cell.request.rows == case.rows
                && cell.prompt_ids.len() == case.rows
                && cell.repetitions == 2
                && cell.repetition_equal
                && cell.boundary_repetitions == 1
                && cell.calls.len() <= 8
                && !cell.calls.is_empty(),
            "complete actual source cell",
        )?;
        let mut frontier = 0;
        for (n, c) in cell.calls.iter().enumerate() {
            let inputs = if n == 0 {
                cell.prompt_ids.clone()
            } else {
                vec![cell.calls[n - 1].selected_id]
            };
            need(
                c.call == n
                    && c.input_ids == inputs
                    && c.base == frontier
                    && c.frontier == frontier + inputs.len()
                    && c.frontier <= 64
                    && c.selected_id < 262144
                    && c.eos == [1, 106, 50].contains(&c.selected_id)
                    && (!c.eos || n + 1 == cell.calls.len()),
                "genuine natural source chain",
            )?;
            frontier = c.frontier;
        }
        need(
            cell.calls
                .last()
                .is_some_and(|c| c.eos || cell.calls.len() == 8)
                && cell.raw.records.len() == 31 * cell.calls.len()
                && cell.boundaries.records.len() == 70 * cell.calls.len(),
            "whole natural stop/all output records",
        )?;
        for (n, c) in cell.calls.iter().enumerate() {
            for (j, r) in cell.raw.records[n * 31..(n + 1) * 31].iter().enumerate() {
                let (label, shape) = if j == 0 {
                    ("logits".to_owned(), vec![262144])
                } else {
                    let o = (j - 1) / 2;
                    (
                        format!("kv.{o}.{}", if j % 2 == 1 { "key" } else { "value" }),
                        vec![c.frontier, if o % 5 == 4 { 512 } else { 256 }],
                    )
                };
                need(
                    r.call == n
                        && r.label == label
                        && r.shape == shape
                        && r.frontier == Some(c.frontier)
                        && r.layer.is_none(),
                    "all source KV/fullV roles",
                )?;
            }
            for (j, r) in cell.boundaries.records[n * 70..(n + 1) * 70]
                .iter()
                .enumerate()
            {
                need(
                    r.call == n
                        && r.layer == Some(j / 2)
                        && r.label
                            == if j % 2 == 0 {
                                "block.input"
                            } else {
                                "block.output"
                            }
                        && r.shape == [c.input_ids.len(), 1536]
                        && r.base == Some(c.base),
                    "all source35 layer input/output roles",
                )?;
            }
        }
        let raw = Held::new(&self.parent, &cell.raw, 16 * 1024 * 1024)?;
        let boundaries = Held::new(&self.parent, &cell.boundaries, 32 * 1024 * 1024)?;
        // Actual complete previous producer bytes must be the next call's prefix.
        for c in 1..cell.calls.len() {
            for j in 1..31 {
                let a = &cell.raw.records[(c - 1) * 31 + j];
                let b = &cell.raw.records[c * 31 + j];
                let mut x = vec![0; a.bytes];
                let mut y = vec![0; a.bytes];
                raw.file.read_exact_at(&mut x, a.offset)?;
                raw.file.read_exact_at(&mut y, b.offset)?;
                need(x == y, "all original consumed-prior continuity")?;
            }
        }
        Ok(Expected {
            cell,
            raw,
            boundaries,
            source,
            trace: Trace::default(),
        })
    }
}
// Expected bytes are compared only with completed original producer readbacks.
#[derive(Default)]
struct Trace {
    call: usize,
    seen: Vec<bool>,
    final_kv: [bool; 30],
    logits: bool,
    differences: usize,
    events: usize,
    first: Option<Value>,
    best: Option<(u32, u32)>,
}
pub struct Expected {
    pub cell: Cell,
    raw: Held,
    boundaries: Held,
    source: SourceIdentity,
    trace: Trace,
}
impl Expected {
    pub fn begin(&mut self, call: usize) {
        self.trace = Trace {
            call,
            seen: vec![false; self.cell.calls[call].input_ids.len() * 70],
            ..Trace::default()
        };
    }
    fn keep(&mut self, label: &str, difference: Difference) {
        self.trace.differences += difference.words;
        if difference.words > 0 && self.trace.first.is_none() {
            self.trace.first = Some(json!({"label":label,"difference":difference.report()}));
        }
    }
    pub fn completed(
        &self,
        actual: &zerogravity_gpu::InferenceReadback,
        before: zerogravity::session_root::RootBinding,
    ) -> Proof<Value> {
        let c = &self.cell.calls[self.trace.call];
        need(
            before.frontier() == c.base as u64 && self.source.is_bound(),
            "same source-bound completed proposal frontier",
        )?;
        need(
            self.trace.seen.iter().all(|x| *x)
                && self.trace.final_kv.iter().all(|x| *x)
                && self.trace.logits,
            "ALL35 boundaries/fullV/ALL15 canonical KV domains observed",
        )?;
        let packet = actual
            .device_token
            .as_ref()
            .ok_or("actual original completed greedy packet")?;
        need(
            self.trace.best == Some((packet.id, packet.max_bits)),
            "actual fullV-derived first greedy ID/maxbits equals original private packet",
        )?;
        Ok(
            json!({"call":self.trace.call,"exact":self.trace.differences==0&&packet.id==c.selected_id,"mismatched_words":self.trace.differences,"boundary_events":self.trace.events,"first_difference":self.trace.first,"actual_selected_id":packet.id,"actual_max_bits":format!("{:08x}",packet.max_bits),"expected_selected_id":c.selected_id,"frontier":c.frontier,"all_block_kv_and_output_comparisons":true}),
        )
    }
    pub fn normal_packet(
        &self,
        call: usize,
        actual: &zerogravity_gpu::InferenceReadback,
        before: zerogravity::session_root::RootBinding,
    ) -> Proof<()> {
        let c = &self.cell.calls[call];
        let packet = actual
            .device_token
            .as_ref()
            .ok_or("normal actual private packet")?;
        let r = &self.cell.raw.records[call * 31];
        let mut expected = [0u8; 2];
        self.raw
            .file
            .read_exact_at(&mut expected, r.offset + u64::from(c.selected_id) * 2)?;
        need(
            before.frontier() == c.base as u64
                && packet.id == c.selected_id
                && packet.max_bits == (u16::from_le_bytes(expected) as u32) << 16
                && actual.logits.is_empty()
                && actual.domains.is_empty()
                && actual.observations.is_empty()
                && actual.chunk_bytes == 4 * c.input_ids.len() + 12,
            "normal exact packet/status only; no producer/logits/KV D2H",
        )
    }
    pub fn normal_state(
        &self,
        call: usize,
        domains: &[(zerogravity::accelerator::program::StateSlot, Vec<f32>)],
    ) -> Proof<()> {
        need(
            domains.len() == 15,
            "normal independent complete state domain union",
        )?;
        let c = &self.cell.calls[call];
        for (index, (slot, actual)) in domains.iter().enumerate() {
            need(
                *slot == zerogravity::accelerator::program::StateSlot::Kv(KvDomain(index)),
                "normal actual state slot order",
            )?;
            let words = c.frontier * if index % 5 == 4 { 512 } else { 256 };
            need(actual.len() == 2 * words, "normal canonical KV words")?;
            for (plane, values) in actual.chunks_exact(words).enumerate() {
                let r = &self.cell.raw.records[call * 31 + 1 + 2 * index + plane];
                need(
                    self.raw.compare(r, None, values)?.words == 0,
                    "normal independent exact full-state words",
                )?;
            }
        }
        Ok(())
    }
    pub fn after(&self) -> Proof<()> {
        self.raw.after()?;
        self.boundaries.after()
    }
}
fn values(actual: &[u8]) -> Result<Vec<f32>> {
    if actual.is_empty() || !actual.len().is_multiple_of(4) || actual.len() > 1024 * 1024 {
        return Err(Error::State("bounded actual completed F32 frame".into()));
    }
    Ok(actual
        .chunks_exact(4)
        .map(|w| f32::from_bits(u32::from_ne_bytes(w.try_into().expect("whole F32 word"))))
        .collect())
}
impl zerogravity_gpu::TextObserver for Expected {
    fn completed_producer(
        &mut self,
        position: usize,
        observation: &zerogravity::accelerator::program::TextObservation,
        actual: &[u8],
    ) -> Result<()> {
        let checked = (|| -> Proof<()> {
            if !matches!(observation.stage, "block.input" | "block.output" | "logits") {
                return Ok(());
            }
            let c = &self.cell.calls[self.trace.call];
            need(
                position >= c.base && position < c.frontier,
                "actual current source row",
            )?;
            let values = values(actual)?;
            if observation.stage == "logits" {
                need(
                    position + 1 == c.frontier && !self.trace.logits && values.len() == 262144,
                    "one last-row whole vocabulary producer",
                )?;
                self.trace.logits = true;
                let r = &self.cell.raw.records[self.trace.call * 31];
                let diff = self.raw.compare(r, None, &values)?;
                let best = values
                    .iter()
                    .enumerate()
                    .fold(0, |b, (i, x)| if *x > values[b] { i } else { b });
                self.trace.best = Some((best as u32, values[best].to_bits()));
                self.keep("logits", diff);
            } else {
                need(
                    observation.layer < 35 && values.len() == 1536,
                    "actual complete block row",
                )?;
                let index =
                    observation.layer * 2 + usize::from(observation.stage == "block.output");
                let row = position - c.base;
                let seen = row * 70 + index;
                need(!self.trace.seen[seen], "unique actual block/source row")?;
                self.trace.seen[seen] = true;
                self.trace.events += 1;
                let r = &self.cell.boundaries.records[self.trace.call * 70 + index];
                let diff = self.boundaries.compare(r, Some(row), &values)?;
                self.keep(observation.stage, diff);
            }
            Ok(())
        })();
        checked.map_err(|e| Error::State(e.to_string()))
    }
    fn completed_kv(
        &mut self,
        prior: bool,
        domain: KvDomain,
        value: bool,
        active: usize,
        actual: &[u8],
    ) -> Result<()> {
        let checked = (|| -> Proof<()> {
            let c = &self.cell.calls[self.trace.call];
            need(
                domain.0 < 15 && active <= c.frontier,
                "actual complete KV prefix/domain",
            )?;
            if active == 0 {
                need(actual.is_empty(), "empty actual prior prefix")?;
                return Ok(());
            }
            let index = domain.0 * 2 + usize::from(value);
            let r = &self.cell.raw.records[self.trace.call * 31 + 1 + index];
            let width = if domain.0 % 5 == 4 { 512 } else { 256 };
            need(
                actual.len() == active * width * 4,
                "actual canonical KV prefix extent",
            )?;
            let v = values(actual)?;
            let label = r.label.clone();
            let mut raw = vec![0; active * width * 2];
            self.raw.file.read_exact_at(&mut raw, r.offset)?;
            let diff = super::comparison::scan(&raw, &v)?;
            if !prior && active == c.frontier {
                need(
                    !self.trace.final_kv[index],
                    "unique final actual canonical KV owner",
                )?;
                self.trace.final_kv[index] = true;
            }
            self.keep(&label, diff);
            Ok(())
        })();
        checked.map_err(|e| Error::State(e.to_string()))
    }
}
