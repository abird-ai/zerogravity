//! Expected-only borrowed producer comparison. No operands or state constructors.
use super::{
    comparison::{Difference, Encoding, scan},
    io::*,
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs::File,
    os::unix::fs::FileExt,
    path::Path,
};
use zerogravity_core::{Error, Result, accelerator::SourceIdentity};
use zerogravity_cpu::reference_text::{
    ReferenceTextArithmetic, ReferenceTextObservation, ReferenceTextObserver,
    ReferenceTextOperation,
};
type Key = (usize, Option<usize>, String);
struct Row {
    key: Key,
    offset: u64,
    elements: usize,
    sha: String,
}
pub(super) struct Producers {
    source: SourceIdentity,
    raw: File,
    before: [u64; 7],
    raw_sha: String,
    legacy: bool,
    call_count: usize,
    rows: Vec<Row>,
    lookup: BTreeMap<Key, usize>,
    seen: RefCell<BTreeSet<usize>>,
    differences: RefCell<Vec<Option<Difference>>>,
}
fn call_for_position(position: usize) -> usize {
    position.saturating_sub(3)
}
fn expected(call_count: usize) -> BTreeMap<Key, usize> {
    let mut out = BTreeMap::new();
    let positions: Vec<_> = [0, 3].into_iter().chain(4..3 + call_count).collect();
    let mut add = |layer, label: &str, n, positions: &[usize]| {
        for position in positions {
            out.insert((*position, layer, label.to_owned()), n);
        }
    };
    for (label, n) in [
        ("embedding", 1536),
        ("ple.embedding", 8960),
        ("ple.projection", 8960),
        ("ple.norm", 8960),
    ] {
        add(None, label, n, &positions);
    }
    for layer in 0..35 {
        let width = if layer % 5 == 4 { 512 } else { 256 };
        let ffn = if layer < 15 { 6144 } else { 12288 };
        for (label, n) in [
            ("block.input", 1536),
            ("input.norm", 1536),
            ("query.projection", 8 * width),
            ("query.norm", 8 * width),
            ("query.rotary", 8 * width),
            ("attention.pv", 8 * width),
            ("attention.projection", 1536),
            ("attention.norm", 1536),
            ("attention.residual", 1536),
            ("ffn.norm", 1536),
            ("ffn.gate", ffn),
            ("ffn.up", ffn),
            ("ffn.gelu", ffn),
            ("ffn.activation", ffn),
            ("ffn.down", 1536),
            ("ffn.post_norm", 1536),
            ("ffn.residual", 1536),
            ("ple.input", 256),
            ("ple.gate", 256),
            ("ple.gelu", 256),
            ("ple.activation", 256),
            ("ple.projection", 1536),
            ("ple.post_norm", 1536),
            ("block.output", 1536),
        ] {
            add(Some(layer), label, n, &positions);
        }
        if layer < 15 {
            for label in ["key.projection", "key.norm", "key.rotary", "value.norm"] {
                add(Some(layer), label, width, &positions);
            }
            // E2B source explicitly has attention_k_eq_v=false, including full attention.
            add(Some(layer), "value.projection", width, &positions);
        }
        for &position in &positions {
            add(Some(layer), "attention.qk", 8 * (position + 1), &[position]);
            add(
                Some(layer),
                "attention.softmax",
                8 * (position + 1),
                &[position],
            );
        }
    }
    for (label, n) in [("final_norm", 1536), ("head", 262144), ("logits", 262144)] {
        add(None, label, n, &(3..3 + call_count).collect::<Vec<_>>());
    }
    out
}
impl Producers {
    pub fn open(
        directory: &Path,
        pin: &Value,
        pins: &Value,
        source: SourceIdentity,
    ) -> Proof<Self> {
        need(
            pin["name"] == "producers.json",
            "contained original producer metadata",
        )?;
        let mut data = json(
            &directory.join("producers.json"),
            2 * 1024 * 1024,
            Some(pin),
        )?;
        let legacy = data["schema"] == "zg.gemma4-e2b.producers.v1";
        need(
            legacy || data["schema"] == "zg.gemma4-e2b.producers.v2",
            "known original producer schema",
        )?;
        let call_count = if legacy {
            1
        } else {
            usize::try_from(data["call_count"].as_u64().ok_or("producer call count")?)?
        };
        need(
            (1..=3).contains(&call_count),
            "finite actual producer trajectory",
        )?;
        let positions: Vec<_> = [0, 3].into_iter().chain(4..3 + call_count).collect();
        need(
            data["source"] == *pins
                && data["prompt_rows"] == 4
                && data["positions"] == json!(positions)
                && data["original_results_unchanged"] == true
                && data["trained_rust_qualified"] == false,
            "actual original whole35 producer scope",
        )?;
        if !legacy {
            let calls: Vec<_> = (0..call_count)
                .map(|call| {
                    json!({"call":call,
                "base":if call==0 {0}else{call+3},"input_rows":if call==0 {4}else{1},
                "positions":if call==0 {vec![0,3]}else{vec![call+3]}})
                })
                .collect();
            need(
                data["calls"] == json!(calls)
                    && pin["bytes"].as_u64().is_some_and(|n| n <= 1024 * 1024),
                "actual current-root producer contexts and compact metadata bound",
            )?;
        }
        let records = match data["records"].take() {
            Value::Array(records) => records,
            _ => return Err("actual producer records".into()),
        };
        let record_count = records.len();
        let expected = expected(call_count);
        need(
            records.len() == expected.len() && records.len() <= 4096,
            "complete bounded source producer roles",
        )?;
        let mut rows = Vec::with_capacity(records.len());
        let mut lookup = BTreeMap::new();
        let mut at = 0;
        // Consume parsed records as compact rows are built; do not retain both full metadata copies.
        for (index, row) in records.into_iter().enumerate() {
            let layer = if row["layer"].is_null() {
                None
            } else {
                Some(usize::try_from(row["layer"].as_u64().ok_or("layer")?)?)
            };
            let key = (
                usize::try_from(row["position"].as_u64().ok_or("position")?)?,
                layer,
                string(&row, "label")?.to_owned(),
            );
            let n = *expected.get(&key).ok_or("unknown original producer role")?;
            need(
                row["elements"] == n
                    && row["bytes"] == n * 2
                    && row["offset"] == at
                    && lookup.insert(key.clone(), index).is_none(),
                "exact original producer layout/no duplicate",
            )?;
            rows.push(Row {
                key,
                offset: at,
                elements: n,
                sha: string(&row, "sha256")?.to_owned(),
            });
            at += (n * 2) as u64;
        }
        need(
            data["raw"]["name"] == "producers.bin"
                && data["raw"]["bytes"] == at
                && at <= 32 * 1024 * 1024,
            "complete bounded producer EOF",
        )?;
        let raw = open(&directory.join("producers.bin"))?;
        let before = stamp(&raw)?;
        let raw_sha = string(&data["raw"], "sha256")?.to_owned();
        need(
            before[2] == at && hash(&raw, at)? == raw_sha,
            "whole actual producer raw SHA/EOF",
        )?;
        Ok(Self {
            source,
            raw,
            before,
            raw_sha,
            legacy,
            call_count,
            rows,
            lookup,
            seen: RefCell::new(BTreeSet::new()),
            differences: RefCell::new(vec![None; record_count]),
        })
    }
    pub fn require_call_count(&self, actual: usize) -> Proof<()> {
        need(
            self.legacy || self.call_count == actual,
            "producer trajectory matches all actual original natural calls",
        )
    }
    pub fn require_lifecycle_scope(&self) -> Proof<()> {
        need(
            !self.legacy && self.call_count == 3,
            "lifecycle requires unchanged healthy all-call v2 producer scope",
        )
    }
    pub fn finish_call(&self, call: usize) -> Proof<()> {
        if call >= self.call_count {
            return Ok(());
        } // Legacy reference2 has no cached producer scope.
        let seen = self.seen.borrow();
        need(
            self.rows
                .iter()
                .enumerate()
                .filter(|(_, row)| call_for_position(row.key.0) == call)
                .all(|(index, _)| seen.contains(&index)),
            "all actual current-call producer roles observed before publication",
        )
    }
    pub fn exact_call(&self, call: usize) -> bool {
        self.rows
            .iter()
            .zip(self.differences.borrow().iter())
            .all(|(row, difference)| call_for_position(row.key.0) != call || difference.is_none())
    }
    pub fn report_call(&self, call: usize) -> Value {
        if call >= self.call_count {
            return Value::Null;
        }
        let seen = self.seen.borrow();
        let differences = self.differences.borrow();
        let mut expected = 0;
        let mut observed = 0;
        let mut words = 0;
        let mut records = Vec::new();
        let mut first = None;
        for (index, row) in self
            .rows
            .iter()
            .enumerate()
            .filter(|(_, row)| call_for_position(row.key.0) == call)
        {
            expected += 1;
            observed += usize::from(seen.contains(&index));
            if let Some(difference) = differences[index] {
                words += difference.words;
                records.push(difference.indexed(index));
                if first.is_none() {
                    first = Some(
                        json!({"position":row.key.0,"layer":row.key.1,"label":row.key.2,
                        "difference":difference.report()}),
                    );
                }
            }
        }
        json!({"call":call,"expected":expected,"observed":observed,"mismatched_producers":records.len(),
            "mismatched_words":words,"first_original_order":first,"records":records})
    }
    pub fn report(&self) -> Value {
        let calls: Vec<_> = (0..self.call_count)
            .map(|call| self.report_call(call))
            .collect();
        json!({"scope":if self.call_count==1 {"initial rows0/3 all35; no cached producers"}
            else {"initial rows0/3 plus each actual cached call, all35"},
            "call_count":self.call_count,"calls":calls,"observed":self.seen.borrow().len(),
            "expected":self.rows.len(),"record_difference_encoding":"[record_index,word_count,first_word,expected_bits,actual_bits]",
            "trained_rust_qualified":false})
    }
    pub fn after(&self) -> Proof<()> {
        need(
            stamp(&self.raw)? == self.before && hash(&self.raw, self.before[2])? == self.raw_sha,
            "same whole producer owner after callbacks",
        )
    }
    fn compare(&self, event: ReferenceTextObservation<'_>) -> Proof<()> {
        if event.operation != ReferenceTextOperation::Boundary
            || ![0, 3]
                .into_iter()
                .chain(4..3 + self.call_count)
                .any(|p| p == event.position)
        {
            return Ok(());
        }
        need(
            event.source == self.source
                && event.arithmetic == ReferenceTextArithmetic::Bf16EagerCandidate,
            "producer callback actual immutable source/arithmetic",
        )?;
        let key = (event.position, event.layer, event.label.to_owned());
        let index = *self
            .lookup
            .get(&key)
            .ok_or("unknown actual selected producer boundary")?;
        let row = &self.rows[index];
        need(
            row.elements == event.values.len() && self.seen.borrow_mut().insert(index),
            "complete producer extent/one observation",
        )?;
        // Same8MiB control; at most512KiB expected record, never the whole22MiB archive.
        let mut raw = vec![0u8; row.elements * 2];
        self.raw.read_exact_at(&mut raw, row.offset)?;
        need(sha(&raw) == row.sha, "complete expected producer words")?;
        let difference = scan(&raw, event.values, Encoding::Bf16)?;
        if difference.words != 0 {
            self.differences.borrow_mut()[index] = Some(difference);
        }
        // Numeric mismatches never alter operands or skip the remaining current-call boundaries.
        Ok(())
    }
}
impl ReferenceTextObserver for Producers {
    fn output(&self, event: ReferenceTextObservation<'_>) -> Result<()> {
        self.compare(event).map_err(|e| Error::State(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn whole_trajectory_has_exact_source_roles_and_fits_unchanged_limits() {
        let one = expected(1);
        let three = expected(3);
        assert_eq!(one.len(), 1981);
        assert_eq!(three.len(), 3965);
        assert_eq!(one.values().sum::<usize>() * 2, 10769888);
        assert_eq!(three.values().sum::<usize>() * 2, 22598144);
        assert_eq!(three[&(5, Some(34), "attention.qk".into())], 48);
        assert!(!three.contains_key(&(5, Some(15), "key.projection".into())));
        assert!(three.len() < 4096 && three.values().sum::<usize>() * 2 < 32 * 1024 * 1024);
    }

    #[test]
    fn every_differing_record_fits_the_existing_report_bound() {
        let rows: Vec<_> = expected(3)
            .into_iter()
            .map(|(key, elements)| Row {
                key,
                elements,
                offset: 0,
                sha: String::new(),
            })
            .collect();
        let differences = rows
            .iter()
            .map(|row| {
                Some(Difference {
                    words: row.elements,
                    first: Some((row.elements - 1, 0xffff0000, 0x7fff0000)),
                })
            })
            .collect();
        // Pure report fixture; no source/model read or mathematical execution.
        let seen = (0..rows.len()).collect();
        let producers = Producers {
            source: SourceIdentity::for_new_allocation().unwrap(),
            raw: File::open("/dev/null").unwrap(),
            before: [0; 7],
            raw_sha: String::new(),
            legacy: false,
            call_count: 3,
            rows,
            lookup: BTreeMap::new(),
            seen: RefCell::new(seen),
            differences: RefCell::new(differences),
        };
        let raw = serde_json::to_vec_pretty(&producers.report()).unwrap();
        //64KiB covers the93 complete-output summaries and unchanged source/root report.
        assert!(raw.len() + 65536 < 1024 * 1024);
    }
}
