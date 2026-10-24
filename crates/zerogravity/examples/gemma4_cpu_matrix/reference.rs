//! Immutable expected bytes are readable only by borrowed output comparisons.
use super::{comparison::Difference, io::*};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};
use zerogravity::gemma4::text::Gemma4ReferenceReadback;
use zerogravity::{
    Error, Result,
    accelerator::{SourceIdentity, program::KvDomain},
};
use zerogravity_cpu::reference_text::{
    ReferenceTextArithmetic, ReferenceTextObservation, ReferenceTextObserver,
    ReferenceTextOperation,
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
        let pins: Value =
            serde_json::from_slice(include_bytes!("../../src/gemma4_cpu/source.json"))?;
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
            trace: RefCell::new(Trace::default()),
        })
    }
}
#[derive(Default)]
struct Trace {
    call: usize,
    seen: Vec<bool>,
    words: usize,
    first: Option<Value>,
    events: usize,
}
pub struct Expected {
    pub cell: Cell,
    raw: Held,
    boundaries: Held,
    source: SourceIdentity,
    trace: RefCell<Trace>,
}
impl Expected {
    pub fn begin(&self, call: usize) {
        let count = self.cell.calls[call].input_ids.len() * 70;
        *self.trace.borrow_mut() = Trace {
            call,
            seen: vec![false; count],
            ..Trace::default()
        };
    }
    pub fn compare(&self, call: usize, actual: Gemma4ReferenceReadback<'_>) -> Proof<Value> {
        let c = &self.cell.calls[call];
        need(
            actual.source_identity() == self.source
                && actual.frontier() == c.frontier
                && actual.root_before().frontier() == c.base as u64,
            "same actual source/root completed proposal",
        )?;
        let trace = self.trace.borrow();
        need(
            trace.call == call && trace.seen.iter().all(|x| *x),
            "every actual35 input/output source row observed once",
        )?;
        let mut words = trace.words;
        let mut records = Vec::with_capacity(31);
        for (j, r) in self.cell.raw.records[call * 31..(call + 1) * 31]
            .iter()
            .enumerate()
        {
            let diff = if j == 0 {
                self.raw
                    .compare(r, None, actual.logits()?.ok_or("whole vocabulary")?)?
            } else {
                let o = (j - 1) / 2;
                need(
                    actual.domain_tokens(KvDomain(o))? == c.frontier,
                    "all real KV owners",
                )?;
                let mut values = Vec::with_capacity(r.bytes / 2);
                for t in 0..c.frontier {
                    let (k, v) = actual.kv_row(KvDomain(o), t)?;
                    values.extend_from_slice(if j % 2 == 1 { k } else { v });
                }
                self.raw.compare(r, None, &values)?
            };
            words += diff.words;
            if diff.words > 0 {
                records.push(json!({"label":r.label,"difference":diff.report()}));
            }
        }
        let v = actual.logits()?.ok_or("actual logits")?;
        let best = v
            .iter()
            .enumerate()
            .fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
        Ok(
            json!({"call":call,"exact":words==0&&best as u32==c.selected_id,"mismatched_words":words,"boundary_events":trace.events,"first_boundary_difference":trace.first,"output_differences":records,"actual_selected_id":best,"expected_selected_id":c.selected_id,"frontier":actual.frontier()}),
        )
    }
    pub fn after(&self) -> Proof<()> {
        self.raw.after()?;
        self.boundaries.after()
    }
}
impl ReferenceTextObserver for Expected {
    fn output(&self, event: ReferenceTextObservation<'_>) -> Result<()> {
        let checked = (|| -> Proof<()> {
            need(
                event.source == self.source
                    && event.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate,
                "same source/arithmetic observer",
            )?;
            if event.operation != ReferenceTextOperation::Boundary
                || !matches!(event.label, "block.input" | "block.output")
            {
                return Ok(());
            }
            let mut t = self.trace.borrow_mut();
            let c = &self.cell.calls[t.call];
            let layer = event.layer.ok_or("actual block layer")?;
            need(
                layer < 35 && event.position >= c.base && event.position < c.frontier,
                "current layer/source row",
            )?;
            let row = event.position - c.base;
            let index = layer * 2 + usize::from(event.label == "block.output");
            let seen = row * 70 + index;
            need(!t.seen[seen], "unique actual source boundary")?;
            t.seen[seen] = true;
            t.events += 1;
            let diff = self.boundaries.compare(
                &self.cell.boundaries.records[t.call * 70 + index],
                Some(row),
                event.values,
            )?;
            t.words += diff.words;
            if diff.words > 0 && t.first.is_none() {
                t.first = Some(
                    json!({"position":event.position,"layer":layer,"label":event.label,"difference":diff.report()}),
                );
            }
            Ok(())
        })();
        checked.map_err(|e| Error::State(e.to_string()))
    }
}
