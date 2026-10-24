//! Original accepted CPU-only oracle; read only after actual GPU producer.
use super::io::*;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};
use zerogravity::accelerator::program::StateSlot;
use zerogravity_gpu::{QwenSourceFrame, QwenSourceFrameKind, QwenSourceObserver};
fn string<'a>(value: &'a Value, key: &str) -> Proof<&'a str> {
    value[key]
        .as_str()
        .ok_or_else(|| format!("original reference string {key}").into())
}
const RAW_BYTES: u64 = 934_776_928;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Codec {
    Bf16,
    F32,
    I64,
}
impl Codec {
    fn width(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
            Self::I64 => 8,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
            Self::I64 => "I64",
        }
    }
}
#[derive(Clone, Debug)]
struct Spec {
    repeat: usize,
    call: usize,
    layer: Option<usize>,
    label: String,
    shape: Vec<usize>,
    codec: Codec,
}
impl Spec {
    fn elements(&self) -> usize {
        self.shape.iter().product()
    }
    fn bytes(&self) -> usize {
        self.elements() * self.codec.width()
    }
}
fn states(out: &mut Vec<Spec>, repeat: usize, call: usize, before: bool) {
    if before && call == 0 {
        return;
    }
    let moment = if before { "before" } else { "after" };
    let rows = call + usize::from(!before);
    for layer in 0..64 {
        let fields = if layer % 4 == 3 {
            vec![
                ("key", vec![1, 4, rows, 256], Codec::Bf16),
                ("value", vec![1, 4, rows, 256], Codec::Bf16),
            ]
        } else {
            vec![
                ("conv4", vec![1, 10240, 4], Codec::Bf16),
                ("recurrent", vec![1, 48, 128, 128], Codec::F32),
            ]
        };
        for (label, shape, codec) in fields {
            out.push(Spec {
                repeat,
                call,
                layer: Some(layer),
                label: format!("{moment}.{label}"),
                shape,
                codec,
            });
        }
    }
}
fn specs() -> Vec<Spec> {
    let mut out = Vec::with_capacity(1040);
    for repeat in 0..2 {
        for call in 0..2 {
            states(&mut out, repeat, call, true);
            for (label, shape, codec) in [
                ("embedding", vec![1, 1, 5120], Codec::Bf16),
                ("positions", vec![3, 1, 1], Codec::I64),
            ] {
                out.push(Spec {
                    repeat,
                    call,
                    layer: None,
                    label: label.into(),
                    shape,
                    codec,
                });
            }
            for layer in 0..64 {
                out.push(Spec {
                    repeat,
                    call,
                    layer: Some(layer),
                    label: "layer_output".into(),
                    shape: vec![1, 1, 5120],
                    codec: Codec::Bf16,
                });
            }
            for (label, count) in [("normalized", 5120), ("logits", 248320)] {
                out.push(Spec {
                    repeat,
                    call,
                    layer: None,
                    label: label.into(),
                    shape: vec![1, 1, count],
                    codec: Codec::Bf16,
                });
            }
            states(&mut out, repeat, call, false);
        }
    }
    out
}
fn validate_records(manifest: &Value) -> Proof<Vec<Spec>> {
    let specs = specs();
    let records = manifest["records"].as_array().ok_or("full64 record list")?;
    need(
        specs.len() == 1040 && records.len() == specs.len(),
        "exact full64 record cohort",
    )?;
    let mut offset = 0u64;
    for (r, s) in records.iter().zip(&specs) {
        let shape: Value = s.shape.clone().into();
        need(
            r["repeat"] == s.repeat
                && r["call"] == s.call
                && r["position"] == s.call
                && r["teacher"] == [11, 353][s.call]
                && r["layer"] == s.layer.map_or(Value::Null, |n| n.into())
                && r["label"] == s.label
                && r["shape"] == shape
                && r["dtype"] == s.codec.name()
                && r["device"] == "cpu"
                && r["offset"] == offset
                && r["bytes"] == s.bytes(),
            format!(
                "full producer repeat{}/call{}/layer{:?}/{} metadata",
                s.repeat, s.call, s.layer, s.label
            ),
        )?;
        let strides = r["original_stride"].as_array().ok_or("original strides")?;
        need(strides.len() == s.shape.len(), "exact original stride rank")?;
        let mut span = 0u64;
        for (d, stride) in s.shape.iter().zip(strides) {
            span = span
                .checked_add(
                    ((*d - 1) as u64)
                        .checked_mul(stride.as_u64().ok_or("nonnegative stride")?)
                        .ok_or("stride multiplication")?,
                )
                .ok_or("stride sum")?;
        }
        need(
            span < s.elements() as u64 && string(r, "sha256")?.len() == 64,
            "bounded actual stride in elements/full-record SHA",
        )?;
        offset = offset
            .checked_add(s.bytes() as u64)
            .ok_or("record extent")?;
    }
    need(
        offset == RAW_BYTES
            && manifest["raw"]["bytes"] == RAW_BYTES
            && manifest["raw"]["file"] == "raw.bin",
        "all raw intervals/EOF/no gaps",
    )?;
    Ok(specs)
}
#[allow(dead_code)]
struct Scope {
    declared: BTreeSet<(usize, usize)>,
    required: BTreeSet<usize>,
    total: usize,
    floats: usize,
    words: usize,
    positions: usize,
}
#[allow(dead_code)]
fn scope_totals(specs: &[Spec], proposals: &[(usize, usize)]) -> Proof<Scope> {
    need(
        !proposals.is_empty(),
        "nonempty declared shared proposal scope",
    )?;
    let mut declared = BTreeSet::new();
    for &(repeat, call) in proposals {
        need(
            repeat < 2 && call < 2 && declared.insert((repeat, call)),
            "bounded unique declared shared proposal scope",
        )?;
    }
    let mut required = BTreeSet::new();
    let (mut total, mut floats, mut words, mut positions) = (0, 0, 0, 0);
    for (index, spec) in specs.iter().enumerate() {
        if !declared.contains(&(spec.repeat, spec.call)) {
            continue;
        }
        total += 1;
        if spec.codec == Codec::I64 {
            positions += spec.elements();
        } else {
            required.insert(index);
            floats += 1;
            words += spec.elements();
        }
    }
    need(
        !required.is_empty() && floats == required.len(),
        "declared shared proposal has original float producers",
    )?;
    Ok(Scope {
        declared,
        required,
        total,
        floats,
        words,
        positions,
    })
}
#[allow(dead_code)]
fn verify_scope(
    specs: &[Spec],
    scope: &Scope,
    records: &[Value],
    pending: usize,
    words: usize,
    positions: usize,
    mismatches: usize,
) -> Proof<()> {
    need(
        pending == 0,
        "no in-progress producer chunks outside completed proposals",
    )?;
    let mut seen = BTreeSet::new();
    for record in records {
        let index = record["original_record_index"]
            .as_u64()
            .ok_or("original record index")? as usize;
        let spec = specs
            .get(index)
            .ok_or("original record index in complete cohort")?;
        need(
            spec.codec != Codec::I64
                && record["repeat"] == spec.repeat
                && record["call"] == spec.call
                && record["layer"] == spec.layer.map_or(Value::Null, |n| n.into())
                && record["label"] == spec.label
                && record["words"] == spec.elements()
                && record["mismatches"] == 0,
            "completed original proposal/record identity/exact extent",
        )?;
        need(
            scope.required.contains(&index) && seen.insert(index),
            "completed records exactly declared proposal set once",
        )?;
    }
    need(
        seen == scope.required && records.len() == scope.floats,
        "declared shared proposal record set complete/no extra/duplicate",
    )?;
    need(
        words == scope.words && positions == scope.positions && mismatches == 0,
        "declared shared proposal word/position totals exact",
    )
}
struct Held {
    path: PathBuf,
    file: File,
    stamp: [u64; 7],
    bytes: u64,
    sha: String,
}
impl Held {
    fn new(root: &Path, pin: &Value, cap: u64) -> Proof<Self> {
        let path = Path::new(string(pin, "path")?);
        let name = path.file_name().ok_or("expected basename")?;
        let path = root.join(name);
        let bytes = pin["bytes"].as_u64().ok_or("expected extent")?;
        need(bytes > 0 && bytes <= cap, "bounded expected file extent")?;
        let file = open(&path)?;
        let before = stamp(&file)?;
        let held = Self {
            path,
            file,
            stamp: before,
            bytes,
            sha: string(pin, "sha256")?.into(),
        };
        held.verify()?;
        Ok(held)
    }
    fn verify(&self) -> Proof<()> {
        need(
            stamp(&self.file)? == self.stamp
                && stamp(&open(&self.path)?)? == self.stamp
                && hash(&self.file, self.bytes)? == self.sha,
            "same original held expected FD/path/fullSHA/EOF",
        )
    }
    fn read(&self, offset: u64, bytes: usize) -> Proof<Vec<u8>> {
        need(
            bytes > 0
                && bytes <= 4 * 1024 * 1024
                && offset
                    .checked_add(bytes as u64)
                    .is_some_and(|n| n <= self.bytes),
            "complete bounded original expected record",
        )?;
        need(
            stamp(&self.file)? == self.stamp,
            "same expected FD before record",
        )?;
        let mut out = vec![0u8; bytes];
        self.file.read_exact_at(&mut out, offset)?;
        need(
            stamp(&self.file)? == self.stamp,
            "same expected FD after record",
        )?;
        Ok(out)
    }
}
struct Reference {
    manifest: Value,
    specs: Vec<Spec>,
    manifest_owner: Held,
    raw: Held,
}
impl Reference {
    fn open(root: &Path) -> Proof<Self> {
        let accepted = json(
            &root.join("accepted-original-full-window-1.json"),
            1024 * 1024,
            Some(
                &json!({"bytes":13214,"sha256":"2a81216ede102c8b0c22c082332f62f38bf0bb2f326298e519f34ed20fc3e3d7"}),
            ),
        )?;
        need(
            accepted["schema"] == "zg.actual-original27b-full-window-reference.v1"
                && accepted["status"] == "ACTUAL_ORIGINAL27B_FULL_WINDOW_REFERENCE_ONLY"
                && accepted["snapshot"]
                    == "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e"
                && accepted["records"] == 1040
                && accepted["raw_bytes"] == RAW_BYTES
                && accepted["original_source_and_owned_lifetime_closed"] == true,
            "accepted complete original CPU source-only reference",
        )?;
        let manifest_owner = Held::new(root, &accepted["manifest"], 16 * 1024 * 1024)?;
        let raw_manifest = manifest_owner.read(0, manifest_owner.bytes as usize)?;
        need(
            sha(&raw_manifest) == manifest_owner.sha,
            "same whole original manifest SHA",
        )?;
        let manifest: Value = serde_json::from_slice(&raw_manifest)?;
        need(
            manifest["schema"] == "zg.original27b-full-window-capture.v1"
                && manifest["snapshot"] == accepted["snapshot"]
                && manifest["scope"] == accepted["scope"]
                && manifest["original_full_forward"] == true
                && manifest["cache_domains"] == 112
                && manifest["full_hf_conv_width"] == 4
                && manifest["all_851_source_parameters_raw"] == true,
            "whole64 original manifest and actual source cache schema",
        )?;
        let specs = validate_records(&manifest)?;
        let raw = Held::new(root, &accepted["raw"], RAW_BYTES)?;
        need(
            raw.bytes == RAW_BYTES && manifest["raw"]["sha256"] == raw.sha,
            "same complete original raw artifact",
        )?;
        Ok(Self {
            manifest,
            specs,
            manifest_owner,
            raw,
        })
    }
}
#[derive(Default)]
struct Progress {
    words: usize,
    mismatches: usize,
    first: Option<Value>,
}
pub(super) struct Expected {
    root: PathBuf,
    reference: Option<Reference>,
    repeat: usize,
    call: usize,
    progress: BTreeMap<usize, Progress>,
    records: Vec<Value>,
    words: usize,
    mismatches: usize,
    positions: usize,
    initial_zero_words: usize,
}
impl Expected {
    pub(super) fn new(root: PathBuf) -> Proof<Self> {
        need(root.is_absolute(), "explicit original expected directory")?;
        Ok(Self {
            root,
            reference: None,
            repeat: 0,
            call: 0,
            progress: BTreeMap::new(),
            records: Vec::new(),
            words: 0,
            mismatches: 0,
            positions: 0,
            initial_zero_words: 0,
        })
    }
    pub(super) fn begin(&mut self, repeat: usize, call: usize) -> Proof<()> {
        need(
            repeat < 2 && call < 2 && self.progress.is_empty(),
            "original expected session/call boundary",
        )?;
        self.repeat = repeat;
        self.call = call;
        Ok(())
    }
    fn compare(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> Proof<()> {
        need(
            frame.position == self.call
                && actual.len().is_multiple_of(4)
                && !actual.is_empty()
                && frame.offset_words + actual.len() / 4 <= frame.total_words,
            "actual completed frame geometry",
        )?;
        if let QwenSourceFrameKind::State { prior: true, .. } = frame.kind
            && self.call == 0
        {
            need(
                actual.iter().all(|b| *b == 0),
                "actual original initial state is positive zero",
            )?;
            self.initial_zero_words += actual.len() / 4;
            return Ok(());
        }
        // Nothing in this object is called from a GPU operand/parameter path.
        if self.reference.is_none() {
            self.reference = Some(Reference::open(&self.root)?);
        }
        let reference = self.reference.as_ref().unwrap();
        let (layer, label, kv) = match frame.kind {
            QwenSourceFrameKind::Value { layer, stage } => match stage {
                "embedding" => (None, "embedding".into(), false),
                "layer_output" => (Some(layer), "layer_output".into(), false),
                "final_norm" => (None, "normalized".into(), false),
                "logits" => (None, "logits".into(), false),
                _ => return Err("unknown original producer boundary".into()),
            },
            QwenSourceFrameKind::State { slot, plane, prior } => {
                let (layer, kind, kv) = match slot {
                    StateSlot::Conv(d) => (d.0, "conv4", false),
                    StateSlot::Recurrent(d) => (d.0, "recurrent", false),
                    StateSlot::Kv(d) => (d.0, if plane == 0 { "key" } else { "value" }, true),
                };
                need(
                    plane == 0 || kv && plane == 1,
                    "original state component index",
                )?;
                (
                    Some(layer),
                    format!("{}.{}", if prior { "before" } else { "after" }, kind),
                    kv,
                )
            }
        };
        let index = reference
            .specs
            .iter()
            .position(|s| {
                s.repeat == self.repeat
                    && s.call == self.call
                    && s.layer == layer
                    && s.label == label
            })
            .ok_or("actual producer not in accepted complete cohort")?;
        let spec = &reference.specs[index];
        let row = &reference.manifest["records"][index];
        need(
            spec.elements() == frame.total_words && spec.codec != Codec::I64,
            "complete original producer extent",
        )?;
        let expected = reference
            .raw
            .read(row["offset"].as_u64().ok_or("record offset")?, spec.bytes())?;
        need(
            sha(&expected) == string(row, "sha256")?,
            "whole original record SHA before comparison",
        )?;
        let progress = self.progress.entry(index).or_default();
        need(
            progress.words == frame.offset_words,
            "original producer chunk order/no duplicate",
        )?;
        let active = if kv { frame.total_words / 1024 } else { 0 };
        for (j, word) in actual.chunks_exact(4).enumerate() {
            let at = frame.offset_words + j;
            let original = if kv {
                let token = at / 1024;
                let head = (at / 256) % 4;
                let lane = at % 256;
                head * active * 256 + token * 256 + lane
            } else {
                at
            };
            let actual = u32::from_le_bytes(word.try_into().unwrap());
            let expected = match spec.codec {
                Codec::Bf16 => {
                    u32::from(u16::from_le_bytes(
                        expected[original * 2..original * 2 + 2].try_into().unwrap(),
                    )) << 16
                }
                Codec::F32 => {
                    u32::from_le_bytes(expected[original * 4..original * 4 + 4].try_into().unwrap())
                }
                Codec::I64 => return Err("position record is not a float producer".into()),
            };
            if actual != expected {
                progress.mismatches += 1;
                if progress.first.is_none() {
                    progress.first = Some(
                        json!({"actual_index":at,"original_index":original,"actual_bits":format!("{actual:08x}"),
                    "expected_bits":format!("{expected:08x}")}),
                    );
                }
            }
        }
        progress.words += actual.len() / 4;
        Ok(())
    }
    pub(super) fn finish(&mut self) -> Proof<Value> {
        let reference = self
            .reference
            .as_ref()
            .ok_or("expected only opens after real computation")?;
        let mut count = 0;
        let (mut words, mut mismatches, mut positions) = (0, 0, 0);
        for (index, spec) in reference.specs.iter().enumerate() {
            if spec.repeat != self.repeat || spec.call != self.call {
                continue;
            }
            count += 1;
            if spec.codec == Codec::I64 {
                let row = &reference.manifest["records"][index];
                let actual = reference.raw.read(
                    row["offset"].as_u64().ok_or("position offset")?,
                    spec.bytes(),
                )?;
                need(
                    sha(&actual) == string(row, "sha256")?
                        && actual
                            .chunks_exact(8)
                            .all(|b| i64::from_le_bytes(b.try_into().unwrap()) == self.call as i64),
                    "three original current positions",
                )?;
                positions += 3;
                continue;
            }
            let p = self
                .progress
                .get(&index)
                .ok_or("missing actual full64 producer/state record")?;
            need(
                p.words == spec.elements(),
                "incomplete actual producer chunks",
            )?;
            words += p.words;
            mismatches += p.mismatches;
            self.records.push(json!({"repeat":self.repeat,"call":self.call,"original_record_index":index,
                "layer":spec.layer,"label":spec.label,"words":p.words,"mismatches":p.mismatches,"first":p.first}));
        }
        need(
            count == if self.call == 0 { 196 } else { 324 },
            "complete original proposal record count",
        )?;
        self.progress.clear();
        self.words += words;
        self.mismatches += mismatches;
        self.positions += positions;
        Ok(
            json!({"repeat":self.repeat,"call":self.call,"records":count,"float_words":words,
            "positions":positions,"mismatches":mismatches,"exact":mismatches==0}),
        )
    }
    #[allow(dead_code)]
    pub(super) fn finish_scoped(&mut self, proposals: &[(usize, usize)]) -> Proof<Value> {
        let r = self
            .reference
            .as_ref()
            .ok_or("original comparison never opened")?;
        r.manifest_owner.verify()?;
        r.raw.verify()?;
        let scope = scope_totals(&r.specs, proposals)?;
        for (index, spec) in r.specs.iter().enumerate() {
            if spec.codec != Codec::I64 || !scope.declared.contains(&(spec.repeat, spec.call)) {
                continue;
            }
            let row = &r.manifest["records"][index];
            let actual = r.raw.read(
                row["offset"].as_u64().ok_or("position offset")?,
                spec.bytes(),
            )?;
            need(
                sha(&actual) == string(row, "sha256")?
                    && actual
                        .chunks_exact(8)
                        .all(|b| i64::from_le_bytes(b.try_into().unwrap()) == spec.call as i64),
                "declared original current positions",
            )?;
        }
        verify_scope(
            &r.specs,
            &scope,
            &self.records,
            self.progress.len(),
            self.words,
            self.positions,
            self.mismatches,
        )?;
        r.manifest_owner.verify()?;
        r.raw.verify()?;
        Ok(
            json!({"declared_proposals":scope.declared.iter().map(|&(repeat,call)|json!([repeat,call])).collect::<Vec<_>>(),"records":scope.total,"float_records":scope.floats,"float_words":scope.words,"positions":scope.positions,"mismatches":self.mismatches,"exact":true}),
        )
    }
    #[allow(dead_code)]
    pub(super) fn in_progress(&self) -> (usize, usize) {
        (
            self.progress.len(),
            self.progress.values().map(|p| p.words).sum(),
        )
    }
    pub(super) fn after(&self) -> Proof<()> {
        let r = self
            .reference
            .as_ref()
            .ok_or("original comparison never opened")?;
        r.manifest_owner.verify()?;
        r.raw.verify()?;
        need(
            self.words == 240896000
                && self.positions == 12
                && self.mismatches == 0
                && self.records.len() == 1036,
            "all1040 records/240896000float+12position words exact",
        )
    }
    pub(super) fn summary(&self) -> Value {
        json!({"records":self.records,"float_words":self.words,"positions":self.positions,
            "mismatches":self.mismatches,"initial_zero_words":self.initial_zero_words,
            "expected_opened_only_after_actual_producer":self.reference.is_some()})
    }
}
impl QwenSourceObserver for Expected {
    fn completed(&mut self, frame: QwenSourceFrame, actual: &[u8]) -> zerogravity::Result<()> {
        self.compare(frame, actual)
            .map_err(|e| zerogravity::Error::State(format!("original full64 comparison: {e}")))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_specs_keep_full4_and_rectangular_kv_mapping() {
        let s = specs();
        assert_eq!(s.len(), 1040);
        assert_eq!(
            s.iter()
                .filter(|s| s.codec != Codec::I64)
                .map(Spec::elements)
                .sum::<usize>(),
            240896000
        );
        for active in [1, 2] {
            let mut seen = vec![false; active * 1024];
            for at in 0..active * 1024 {
                let original = ((at / 256) % 4) * active * 256 + (at / 1024) * 256 + at % 256;
                assert!(!seen[original]);
                seen[original] = true;
            }
            assert!(seen.into_iter().all(|v| v));
        }
        assert!(
            s.iter()
                .filter(|s| s.label.ends_with("conv4"))
                .all(|s| s.elements() == 40960)
        );
    }
    fn completed_records(specs: &[Spec], proposals: &[(usize, usize)]) -> Vec<Value> {
        let scope = scope_totals(specs, proposals).unwrap();
        scope
            .required
            .iter()
            .map(|&index| {
                let s = &specs[index];
                json!({"repeat":s.repeat,"call":s.call,"original_record_index":index,
                    "layer":s.layer,"label":s.label,"words":s.elements(),
                    "mismatches":0,"first":null})
            })
            .collect()
    }
    #[test]
    fn shared_scope_derives_known_and_unknown_from_original_specs() {
        let specs = specs();
        for (proposals, total, floats, words, positions) in [
            (
                vec![(0usize, 0usize), (0, 1), (1, 0)],
                716usize,
                713usize,
                160_781_824usize,
                9usize,
            ),
            (vec![(0usize, 0usize)], 196, 195, 40_333_824, 3),
        ] {
            let scope = scope_totals(&specs, &proposals).unwrap();
            assert_eq!(scope.floats, floats);
            assert_eq!(scope.required.len(), floats);
            assert_eq!(scope.words, words);
            assert_eq!(scope.positions, positions);
            assert_eq!(scope.total, total);
            let records = completed_records(&specs, &proposals);
            verify_scope(&specs, &scope, &records, 0, words, positions, 0).unwrap();
        }
    }
    #[test]
    fn shared_scope_full4_matches_strict_original_after_totals() {
        let specs = specs();
        let proposals = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)];
        let scope = scope_totals(&specs, &proposals).unwrap();
        assert_eq!(scope.floats, 1036);
        assert_eq!(scope.total, 1040);
        assert_eq!(scope.words, 240896000);
        assert_eq!(scope.positions, 12);
        let records = completed_records(&specs, &proposals);
        verify_scope(&specs, &scope, &records, 0, scope.words, scope.positions, 0).unwrap();
    }
    #[test]
    fn shared_scope_refuses_bad_declarations_and_wrong_mode() {
        let specs = specs();
        for proposals in [
            Vec::<(usize, usize)>::new(),
            vec![(0, 0), (0, 0)],
            vec![(2, 0)],
            vec![(0, 2)],
            vec![(0, 0), (0, 2)],
        ] {
            assert!(scope_totals(&specs, &proposals).is_err());
        }
        let known = [(0usize, 0usize), (0, 1), (1, 0)];
        let unknown = [(0usize, 0usize)];
        let known_scope = scope_totals(&specs, &known).unwrap();
        let unknown_scope = scope_totals(&specs, &unknown).unwrap();
        let known_records = completed_records(&specs, &known);
        let unknown_records = completed_records(&specs, &unknown);
        assert!(
            verify_scope(
                &specs,
                &unknown_scope,
                &known_records,
                0,
                known_scope.words,
                known_scope.positions,
                0
            )
            .is_err()
        );
        assert!(
            verify_scope(
                &specs,
                &known_scope,
                &unknown_records,
                0,
                unknown_scope.words,
                unknown_scope.positions,
                0
            )
            .is_err()
        );
    }
    #[test]
    fn shared_scope_refuses_missing_extra_duplicate_bad_record_or_progress() {
        let specs = specs();
        let proposals = [(0usize, 0usize), (0, 1), (1, 0)];
        let scope = scope_totals(&specs, &proposals).unwrap();
        let good = completed_records(&specs, &proposals);
        let check = |records: &[Value], pending, words, positions, mismatches| {
            verify_scope(
                &specs, &scope, records, pending, words, positions, mismatches,
            )
        };
        assert!(check(&good, 0, scope.words, scope.positions, 0).is_ok());
        let mut missing = good.clone();
        missing.pop();
        assert!(check(&missing, 0, scope.words, scope.positions, 0).is_err());
        let unselected = specs
            .iter()
            .position(|s| s.repeat == 1 && s.call == 1 && s.codec != Codec::I64)
            .unwrap();
        let mut extra = good.clone();
        extra.push(
            json!({"repeat":1,"call":1,"original_record_index":unselected,
            "layer":specs[unselected].layer,"label":specs[unselected].label,
            "words":specs[unselected].elements(),"mismatches":0,"first":null}),
        );
        assert!(check(&extra, 0, scope.words, scope.positions, 0).is_err());
        let mut duplicate = good.clone();
        duplicate.push(good[0].clone());
        assert!(check(&duplicate, 0, scope.words, scope.positions, 0).is_err());
        let mut wrong_index = good.clone();
        wrong_index[0]["original_record_index"] = json!(unselected);
        assert!(check(&wrong_index, 0, scope.words, scope.positions, 0).is_err());
        let mut wrong_proposal = good.clone();
        wrong_proposal[0]["call"] = json!(1);
        assert!(check(&wrong_proposal, 0, scope.words, scope.positions, 0).is_err());
        let first = good[0]["original_record_index"].as_u64().unwrap() as usize;
        let mut wrong_extent = good.clone();
        wrong_extent[0]["words"] = json!(specs[first].elements() - 1);
        assert!(check(&wrong_extent, 0, scope.words, scope.positions, 0).is_err());
        let mut mismatch = good.clone();
        mismatch[0]["mismatches"] = json!(1);
        assert!(check(&mismatch, 0, scope.words, scope.positions, 0).is_err());
        assert!(check(&good, 0, scope.words, scope.positions, 1).is_err());
        assert!(check(&good, 1, scope.words, scope.positions, 0).is_err());
        assert!(check(&good, 0, scope.words - 1, scope.positions, 0).is_err());
        assert!(check(&good, 0, scope.words, scope.positions - 1, 0).is_err());
    }
}
