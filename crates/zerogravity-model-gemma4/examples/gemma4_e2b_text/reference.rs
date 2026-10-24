use super::{
    comparison::{Difference, Encoding, scan},
    io::*,
    source,
};
use serde_json::{Value, json};
use std::{cell::RefCell, fs::File, os::unix::fs::FileExt, path::Path};
use zerogravity_core::accelerator::program::KvDomain;
use zerogravity_model_gemma4::text::Gemma4ReferenceReadback;
pub(super) struct Reference {
    pub capture: Value,
    raw: File,
    before: [u64; 7],
    raw_sha: String,
    raw_bytes: u64,
    comparisons: RefCell<Vec<Value>>,
}
impl Reference {
    pub fn open(invocation: &Path, expected_sha: &str) -> Proof<Self> {
        let raw = read(invocation, 1024 * 1024, None)?;
        need(
            sha(&raw) == expected_sha,
            "explicit expected-only original invocation SHA",
        )?;
        let receipt: Value = serde_json::from_slice(&raw)?;
        need(
            receipt["status"] == "EXECUTED_ORIGINAL_GEMMA4_TEXT_EXPECTED_ONLY"
                && receipt["source_before"] == receipt["source_after"]
                && receipt["source_FDs_closed"] == true
                && receipt["cleanup_errors"] == serde_json::json!([])
                && receipt["exit_code"] == 0
                && receipt["log_cleanup"]["pipe_eof"] == true
                && receipt["log_cleanup"]["all_closed"] == true
                && receipt["foreground_cleanup"]["leader_reaped"] == true
                && receipt["foreground_cleanup"]["group_absent"] == true
                && receipt["foreground_cleanup"]["all_nonzero_signals_before_reap"] == true
                && receipt["owned_container_absent_after_reap"] == true,
            "whole healthy original source/lifetime closure; no failed parent",
        )?;
        need(
            receipt["image_id"]
                == "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe"
                && receipt["image_digest"]
                    == "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
            "approved original CPU image",
        )?;
        let pins = source::pins()?;
        need(
            receipt["source"] == pins,
            "actual original complete source registry",
        )?;
        let files = pins["files"].as_array().ok_or("source registry")?;
        let before = receipt["source_before"]
            .as_array()
            .ok_or("actual full source owners")?;
        need(
            before.len() == 8 && files.len() == 8,
            "all eight actual original members",
        )?;
        for (row, pin) in before.iter().zip(files) {
            let fd = row["fd_identity"]
                .as_array()
                .ok_or("held source FD identity")?;
            need(
                row["name"] == pin["name"]
                    && row["bytes"] == pin["bytes"]
                    && row["sha256"] == pin["sha256"]
                    && fd.len() == 5
                    && fd[2] == pin["bytes"],
                "complete held original source bytes/SHA/identity",
            )?;
        }
        let parent = invocation.parent().ok_or("original reference directory")?;
        let capture = json(
            &parent.join("capture.json"),
            1024 * 1024,
            Some(&receipt["capture"]),
        )?;
        need(
            capture["schema"] == "zg.gemma4-e2b.original-text.v1"
                && capture["source"] == source::pins()?
                && capture["profile"] == "HfTorchCpuBf16Avx2Eager5.17.0"
                && capture["repetitions"] == 2
                && capture["repetition_equal"] == true
                && capture["trained_rust_qualified"] == false,
            "actual full original E2B reference scope",
        )?;
        validate(&capture)?;
        let raw = open(&parent.join("raw.bin"))?;
        let before = stamp(&raw)?;
        let raw_sha = string(&capture["raw"], "sha256")?.to_owned();
        let raw_bytes = capture["raw"]["bytes"]
            .as_u64()
            .ok_or("complete raw size")?;
        need(
            before[2] == raw_bytes && hash(&raw, raw_bytes)? == raw_sha,
            "whole actual original raw SHA/EOF",
        )?;
        Ok(Self {
            capture,
            raw,
            before,
            raw_sha,
            raw_bytes,
            comparisons: RefCell::new(Vec::with_capacity(3)),
        })
    }
    fn record(&self, call: usize, label: &str, actual: &[f32]) -> Proof<Difference> {
        let rows = self.capture["records"].as_array().ok_or("records")?;
        let row = rows
            .iter()
            .find(|r| r["call"] == call && r["label"] == label)
            .ok_or("missing actual record")?;
        let count = row["bytes"].as_u64().ok_or("record bytes")? as usize;
        need(
            count == actual.len() * 4,
            "complete actual expected/producer extent",
        )?;
        let mut bytes = vec![0u8; count];
        self.raw
            .read_exact_at(&mut bytes, row["offset"].as_u64().ok_or("raw offset")?)?;
        need(
            sha(&bytes) == string(row, "sha256")?,
            "exact record content",
        )?;
        scan(&bytes, actual, Encoding::WidenedBf16)
    }
    pub fn compare(&self, call: usize, completed: Gemma4ReferenceReadback<'_>) -> Proof<bool> {
        need(
            call == self.comparisons.borrow().len() && call < 3,
            "one original completed comparison per actual call",
        )?;
        self.compare_completed_attempt(call, completed)
    }
    /// A rejected completed proposal can be recomputed at the SAME root. Keep
    /// every comparison; no expected record becomes a producer or state input.
    pub fn compare_completed_attempt(
        &self,
        call: usize,
        completed: Gemma4ReferenceReadback<'_>,
    ) -> Proof<bool> {
        need(call < 3, "finite original completed attempt")?;
        let frontier = self.capture["calls"][call]["frontier"]
            .as_u64()
            .ok_or("expected frontier")? as usize;
        need(
            completed.frontier() == frontier,
            "actual completed current root frontier",
        )?;
        let logits = completed
            .logits()?
            .ok_or("actual selected-last full vocabulary")?;
        let difference = self.record(call, "logits", logits)?;
        let mut mismatched_words = difference.words;
        let mut records = vec![
            json!({"label":"logits","elements":logits.len(),"difference":difference.report()}),
        ];
        let selected = logits
            .iter()
            .enumerate()
            .fold(0, |best, (i, x)| if *x > logits[best] { i } else { best });
        let greedy_equal = self.capture["calls"][call]["selected_id"] == selected;
        for owner in 0..15 {
            need(
                completed.domain_tokens(KvDomain(owner))? == frontier,
                "all actual independent KV owners",
            )?;
            for (component, label) in [(0, "key"), (1, "value")] {
                let width = if owner % 5 == 4 { 512 } else { 256 };
                let mut all = Vec::with_capacity(frontier * width);
                for token in 0..frontier {
                    let pair = completed.kv_row(KvDomain(owner), token)?;
                    let row = if component == 0 { pair.0 } else { pair.1 };
                    need(row.len() == width, "exact owner head width")?;
                    all.extend_from_slice(row);
                }
                let label = format!("kv.{owner}.{label}");
                let difference = self.record(call, &label, &all)?;
                mismatched_words += difference.words;
                records.push(
                    json!({"label":label,"elements":all.len(),"difference":difference.report()}),
                );
            }
        }
        let exact = mismatched_words == 0 && greedy_equal;
        self.comparisons.borrow_mut().push(json!({"call":call,"exact":exact,
            "compared_records":records.len(),"mismatched_words":mismatched_words,
            "actual_selected_id":selected,"expected_selected_id":self.capture["calls"][call]["selected_id"],
            "greedy_equal":greedy_equal,"records":records}));
        Ok(exact)
    }
    pub fn comparison_report(&self) -> Value {
        json!({"scope":"complete fullV plus all15 complete owner KV pairs per completed callback",
            "calls":&*self.comparisons.borrow(),"trained_rust_qualified":false})
    }
    pub fn after(&self) -> Proof<()> {
        need(
            stamp(&self.raw)? == self.before && hash(&self.raw, self.raw_bytes)? == self.raw_sha,
            "same full original reference after completed callbacks",
        )
    }
}
fn validate(c: &Value) -> Proof<()> {
    let ids = c["prompt_ids"]
        .as_array()
        .ok_or("actual original tokenizer IDs")?;
    need(
        !ids.is_empty()
            && ids.len() <= 64
            && ids.iter().all(|x| x.as_u64().is_some_and(|n| n < 262144)),
        "bounded genuine tokenized prompt",
    )?;
    let calls = c["calls"].as_array().ok_or("actual calls")?;
    need(
        !calls.is_empty()
            && calls.len() <= 3
            && c["request"]
                == serde_json::json!({"text":"One plus one is","add_special_tokens":true,"max_output":3}),
        "exact whole source workload",
    )?;
    let records = c["records"].as_array().ok_or("records")?;
    need(
        records.len() == 31 * calls.len(),
        "fullV plus all15 KV pairs per call",
    )?;
    let mut frontier = 0;
    let mut at = 0;
    for (call, row) in calls.iter().enumerate() {
        let inputs = if call == 0 {
            c["prompt_ids"].clone()
        } else {
            serde_json::json!([calls[call - 1]["selected_id"]])
        };
        need(
            row["call"] == call
                && row["input_ids"] == inputs
                && row["selected_id"].as_u64().is_some_and(|n| n < 262144),
            "actual original private feedback chain",
        )?;
        frontier += if call == 0 { ids.len() } else { 1 };
        need(row["frontier"] == frontier, "true source frontier")?;
        let eos = c["eos_ids"]
            .as_array()
            .ok_or("original EOS IDs")?
            .contains(&row["selected_id"]);
        need(
            row["eos"] == eos
                && (!eos || call + 1 == calls.len())
                && (call + 1 < calls.len() || eos || calls.len() == 3),
            "natural original EOS disposition",
        )?;
        for index in 0..31 {
            let (label, shape, count) = if index == 0 {
                ("logits".to_string(), serde_json::json!([262144]), 262144)
            } else {
                let owner = (index - 1) / 2;
                let name = if index % 2 == 1 { "key" } else { "value" };
                let width = if owner % 5 == 4 { 512 } else { 256 };
                (
                    format!("kv.{owner}.{name}"),
                    serde_json::json!([frontier, width]),
                    frontier * width,
                )
            };
            let r = &records[call * 31 + index];
            need(
                r["call"] == call
                    && r["label"] == label
                    && r["shape"] == shape
                    && r["offset"] == at
                    && r["bytes"] == count * 4
                    && r["dtype"] == "original-bf16-widened-f32-le",
                "all original complete record labels/extents/order",
            )?;
            at += count * 4;
        }
    }
    need(
        at <= 16 * 1024 * 1024 && c["raw"]["name"] == "raw.bin" && c["raw"]["bytes"] == at,
        "whole original archive no padding/unknown records",
    )
}
