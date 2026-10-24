#[path = "initial_core.rs"]
pub(super) mod initial_core;
#[path = "layer9.rs"]
pub(super) mod layer9;
use super::{io::*, reference_pin::Pin, source};
use serde_json::{Value, json};
use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};
use zerogravity_core::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain, StateSlot};
use zerogravity_model_qwen35::original_binding::{
    WindowDiagnosticBoundary as B, WindowDiagnosticReadback,
};

pub(super) const RAW_BYTES: u64 = 934_776_928;
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
pub(super) fn accepted(pin: Option<&Pin>) -> Proof<Value> {
    let pin = pin.ok_or("ROOT original full64 reference pin is absent; no learned replay")?;
    need(
        Path::new(pin.path).is_absolute()
            && pin.bytes > 0
            && pin.bytes <= 1024 * 1024
            && pin.sha256.len() == 64,
        "closed accepted reference identity",
    )?;
    let value = json(
        Path::new(pin.path),
        1024 * 1024,
        Some(&json!({"bytes":pin.bytes,"sha256":pin.sha256})),
    )?;
    need(
        value["schema"] == "zg.actual-original27b-full-window-reference.v1"
            && value["status"] == "ACTUAL_ORIGINAL27B_FULL_WINDOW_REFERENCE_ONLY"
            && value["snapshot"] == source::SNAPSHOT
            && value["config_sha256"] == source::CONFIG
            && value["index_sha256"] == source::INDEX
            && value["repository"] == "Qwen/Qwen3.8-27B"
            && value["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
            && value["parameter_mode"] == "raw-source-assign-no-cast"
            && value["original_source_and_owned_lifetime_closed"] == true,
        "actual original full64/source/owned lifetime",
    )?;
    need(
        value["calls"] == 2
            && value["original_repetitions"] == 2
            && value["text_roles"] == 851
            && value["layers"] == 64
            && value["cache_domains"] == 112
            && value["full_hf_conv_width"] == 4
            && value["records"] == 1040
            && value["raw_bytes"] == RAW_BYTES
            && value["rust_numerical_parity_qualified"] == false
            && value["mkl_exp_producer_qualified"] == false
            && value["state_root_native_serving_performance_qualified"] == false,
        "original-only expected cohort is no Rust/model permit",
    )?;
    Ok(value)
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
struct Held {
    path: PathBuf,
    file: File,
    stamp: [u64; 7],
    bytes: u64,
    sha: String,
}
impl Held {
    fn pinned(pin: &Value, cap: u64) -> Proof<Self> {
        let path = PathBuf::from(string(pin, "path")?);
        let bytes = pin["bytes"].as_u64().ok_or("held byte extent")?;
        need(
            path.is_absolute() && bytes > 0 && bytes <= cap,
            "bounded absolute held reference",
        )?;
        let file = open(&path)?;
        let stamp = stamp(&file)?;
        let value = Self {
            path,
            file,
            stamp,
            bytes,
            sha: string(pin, "sha256")?.into(),
        };
        value.verify()?;
        Ok(value)
    }
    fn verify(&self) -> Proof<()> {
        need(
            stamp(&self.file)? == self.stamp && stamp(&open(&self.path)?)? == self.stamp,
            "same reference FD/current path before full scan",
        )?;
        need(
            hash(&self.file, self.bytes)? == self.sha
                && stamp(&self.file)? == self.stamp
                && stamp(&open(&self.path)?)? == self.stamp,
            "same reference full SHA/EOF/FD/path after scan",
        )
    }
    fn read(&self, offset: u64, bytes: usize) -> Proof<Vec<u8>> {
        need(
            bytes > 0
                && bytes <= 4 * 1024 * 1024
                && offset
                    .checked_add(bytes as u64)
                    .is_some_and(|n| n <= self.bytes),
            "bounded defined reference read",
        )?;
        let mut raw = Vec::new();
        raw.try_reserve_exact(bytes)?;
        raw.resize(bytes, 0);
        self.file.read_exact_at(&mut raw, offset)?;
        need(stamp(&self.file)? == self.stamp, "same held record FD")?;
        Ok(raw)
    }
}
pub(super) struct Reference {
    manifest: Value,
    specs: Vec<Spec>,
    manifest_owner: Held,
    raw: Held,
}
pub(super) struct Comparison {
    repeat: usize,
    call: usize,
    words: usize,
    mismatches: usize,
    positions: usize,
    records: Vec<Value>,
    first_failure: Option<String>,
}
impl Comparison {
    pub(super) fn words(&self) -> usize {
        self.words
    }
    pub(super) fn refusal(&self) -> Option<&str> {
        self.first_failure.as_deref()
    }
    pub(super) fn into_report(self) -> Value {
        let first = self
            .records
            .iter()
            .find(|r| r["mismatched_words"].as_u64().unwrap_or(0) > 0)
            .cloned();
        let state = self
            .records
            .iter()
            .find(|r| r["mismatched_words"].as_u64().unwrap_or(0) > 0 && r["state_domain"] == true)
            .cloned();
        json!({"repeat":self.repeat,"call":self.call,"compared_float_words":self.words,
            "mismatched_float_words":self.mismatches,"position_metadata_words_checked":self.positions,
            "first_original_order_difference":first,"first_state_difference":state,
            "all_available_diagnostic_records_compared":true,"records":self.records,
            "original_conv4_complete_equivalence":self.records.iter().filter(|r|r["label"].as_str().is_some_and(|v|v.ends_with(".conv4"))).all(|r|r["mismatched_words"]==0&&r["unobserved_original_elements"]==0)})
    }
}
impl Reference {
    /// Called only by the completed callback; expected bytes never enter math.
    pub(super) fn open(accepted: &Value) -> Proof<Self> {
        let manifest_owner = Held::pinned(&accepted["manifest"], 16 * 1024 * 1024)?;
        let raw = manifest_owner.read(0, usize::try_from(manifest_owner.bytes)?)?;
        let manifest: Value = serde_json::from_slice(&raw)?;
        need(
            sha(&raw) == manifest_owner.sha
                && manifest["schema"] == "zg.original27b-full-window-capture.v1"
                && manifest["status"] == "CAPTURED_ORIGINAL27B_FULL_WINDOW_REQUIRES_VALIDATION"
                && manifest["snapshot"] == source::SNAPSHOT
                && manifest["scope"] == accepted["scope"]
                && manifest["original_full_forward"] == true
                && manifest["cache_domains"] == 112
                && manifest["full_hf_conv_width"] == 4
                && manifest["all_851_source_parameters_raw"] == true,
            "authenticated original source/full64 manifest",
        )?;
        let specs = validate_records(&manifest)?;
        let raw = Held::pinned(&accepted["raw"], RAW_BYTES)?;
        need(
            raw.bytes == RAW_BYTES && manifest["raw"]["sha256"] == raw.sha,
            "same accepted whole raw identity",
        )?;
        Ok(Self {
            manifest,
            specs,
            manifest_owner,
            raw,
        })
    }
    pub(super) fn compare(
        &self,
        repeat: usize,
        call: usize,
        actual: &WindowDiagnosticReadback<'_>,
    ) -> Proof<Comparison> {
        need(
            repeat < 2 && call < 2 && actual.base().frontier() == call as u64,
            "same actual session/current frontier/reference call",
        )?;
        let count = self
            .specs
            .iter()
            .filter(|s| s.repeat == repeat && s.call == call)
            .count();
        need(
            count == if call == 0 { 196 } else { 324 },
            "complete finite callback record cohort",
        )?;
        let mut compared = Comparison {
            repeat,
            call,
            words: 0,
            mismatches: 0,
            positions: 0,
            records: Vec::new(),
            first_failure: None,
        };
        compared.records.try_reserve_exact(count)?;
        for (index, (r, s)) in self.manifest["records"]
            .as_array()
            .ok_or("records")?
            .iter()
            .zip(&self.specs)
            .enumerate()
        {
            if s.repeat != repeat || s.call != call {
                continue;
            }
            let raw = self
                .raw
                .read(r["offset"].as_u64().ok_or("record offset")?, s.bytes())?;
            need(
                sha(&raw) == string(r, "sha256")?,
                "full original record integrity",
            )?;
            if s.codec == Codec::I64 {
                let mismatches = raw
                    .chunks_exact(8)
                    .filter(|b| i64::from_le_bytes((*b).try_into().unwrap()) != call as i64)
                    .count();
                let first=raw.chunks_exact(8).enumerate()
                    .find(|(_,b)|i64::from_le_bytes((*b).try_into().unwrap())!=call as i64)
                    .map(|(index,b)|json!({"actual_index":index,"original_index":index,
                        "actual_bits":format!("{:016x}",call as u64),
                        "expected_bits":format!("{:016x}",u64::from_le_bytes(b.try_into().unwrap()))}));
                if mismatches > 0 && compared.first_failure.is_none() {
                    compared.first_failure = Some(format!(
                        "repeat{repeat} call{call} original three-axis current position"
                    ));
                }
                compared.positions += 3;
                compared.records.push(
                    json!({"original_record_index":index,"layer":s.layer,"label":s.label,
                    "state_domain":false,"actual_kind":"source_frontier_metadata","dtype":"I64",
                    "compared_words":3,"mismatched_words":mismatches,"first":first}),
                );
                continue;
            }
            let layer = s.layer.unwrap_or(0);
            let values = match s.label.as_str() {
                "embedding" => actual.boundary(B::Embedding)?,
                "layer_output" => actual.boundary(B::Layer(layer))?,
                "normalized" => actual.boundary(B::FinalNorm)?,
                "logits" => actual.logits()?,
                label => {
                    let (moment, kind) = label.split_once('.').ok_or("state selector")?;
                    let slot = match kind {
                        "conv4" => StateSlot::Conv(ConvDomain(layer)),
                        "recurrent" => StateSlot::Recurrent(RecurrentDomain(layer)),
                        "key" | "value" => StateSlot::Kv(KvDomain(layer)),
                        _ => return Err("closed state selector".into()),
                    };
                    let parts = if moment == "before" {
                        actual.prior_state(slot)?
                    } else {
                        actual.state(slot)?
                    };
                    parts[usize::from(kind == "value")]
                }
            };
            let words = compare_words(s, values, &raw)?;
            if compared.first_failure.is_none()
                && let Some(first) = &words.first
            {
                compared.first_failure = Some(format!(
                    "repeat{} call{} layer{:?} {} word{}/{} actual={:08x} expected={:08x}",
                    s.repeat,
                    s.call,
                    s.layer,
                    s.label,
                    first.actual_index,
                    words.compared,
                    first.actual_bits,
                    first.expected_bits
                ));
            }
            compared.words += words.compared;
            compared.mismatches += words.mismatches;
            compared.records.push(words.report(
                index,
                r["offset"].as_u64().ok_or("record offset")?,
                s,
            ));
        }
        need(
            compared.records.len() == count && compared.positions == 3,
            "all available proposal observations checked",
        )?;
        Ok(compared)
    }
    pub(super) fn verify_after(&self) -> Proof<()> {
        self.manifest_owner.verify()?;
        self.raw.verify()
    }
}
fn expected_index(s: &Spec, i: usize) -> usize {
    if s.label.ends_with(".key") || s.label.ends_with(".value") {
        let rows = s.shape[2];
        let token = i / 1024;
        let head = i % 1024 / 256;
        let dim = i % 256;
        (head * rows + token) * 256 + dim
    } else {
        i
    }
}
struct Difference {
    actual_index: usize,
    original_index: usize,
    actual_bits: u32,
    expected_bits: u32,
}
struct WordComparison {
    compared: usize,
    mismatches: usize,
    first: Option<Difference>,
}
impl WordComparison {
    fn report(&self, index: usize, offset: u64, s: &Spec) -> Value {
        let first=self.first.as_ref().map(|f|json!({"actual_index":f.actual_index,
            "original_index":f.original_index,"original_byte_offset":offset+(f.original_index*s.codec.width()) as u64,
            "actual_bits":format!("{:08x}",f.actual_bits),"expected_bits":format!("{:08x}",f.expected_bits)}));
        json!({"original_record_index":index,"layer":s.layer,"label":s.label,"dtype":s.codec.name(),
            "state_domain":s.label.starts_with("before.")||s.label.starts_with("after."),
            "original_elements":s.elements(),"compared_words":self.compared,
            "unobserved_original_elements":s.elements()-self.compared,
            "mismatched_words":self.mismatches,"first":first})
    }
}
fn compare_words(s: &Spec, actual: &[f32], raw: &[u8]) -> Proof<WordComparison> {
    let count = s.elements();
    need(
        raw.len() == s.bytes() && actual.len() == count,
        "full defined diagnostic boundary extent",
    )?;
    let mut out = WordComparison {
        compared: count,
        mismatches: 0,
        first: None,
    };
    for (i, a) in actual.iter().enumerate() {
        let at = expected_index(s, i) * s.codec.width();
        let bits = match s.codec {
            Codec::Bf16 => u32::from(u16::from_le_bytes(raw[at..at + 2].try_into()?)) << 16,
            Codec::F32 => u32::from_le_bytes(raw[at..at + 4].try_into()?),
            Codec::I64 => return Err("positions are not model float operands".into()),
        };
        if !a.is_finite() || a.to_bits() != bits {
            out.mismatches += 1;
            if out.first.is_none() {
                out.first = Some(Difference {
                    actual_index: i,
                    original_index: expected_index(s, i),
                    actual_bits: a.to_bits(),
                    expected_bits: bits,
                });
            }
        }
    }
    Ok(out)
}
pub(super) fn comparison_words() -> usize {
    specs()
        .iter()
        .filter(|s| s.codec != Codec::I64)
        .map(Spec::elements)
        .sum()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_original_cohort_has_full_coverage_including_oldest_conv_frame() {
        let s = specs();
        assert_eq!(s.len(), 1040);
        assert_eq!(s.iter().map(|r| r.bytes() as u64).sum::<u64>(), RAW_BYTES);
        let conv = s.iter().find(|r| r.label == "after.conv4").unwrap();
        assert_eq!(
            [
                expected_index(conv, 0),
                expected_index(conv, 2),
                expected_index(conv, 3),
                expected_index(conv, 40959)
            ],
            [0, 2, 3, 40959]
        );
        assert!(
            !s.iter()
                .any(|r| r.call == 0 && r.label.starts_with("before."))
        );
        assert!(accepted(None).is_err());
    }
    #[test]
    fn cached_kv_transpose_keeps_token_head_and_row_identity() {
        let s = specs()
            .into_iter()
            .find(|r| r.call == 1 && r.label == "after.key")
            .unwrap();
        assert_eq!(
            [
                expected_index(&s, 0),
                expected_index(&s, 256),
                expected_index(&s, 1024),
                expected_index(&s, 2047)
            ],
            [0, 512, 256, 2047]
        );
    }
    #[test]
    fn exact_words_refuse_signed_zero_and_ulp_changes_without_tolerance() {
        let s = Spec {
            repeat: 0,
            call: 0,
            layer: None,
            label: "fixture".into(),
            shape: vec![1],
            codec: Codec::Bf16,
        };
        assert_eq!(
            compare_words(&s, &[0.], &0u16.to_le_bytes())
                .unwrap()
                .mismatches,
            0
        );
        assert_eq!(
            compare_words(&s, &[-0.], &0u16.to_le_bytes())
                .unwrap()
                .mismatches,
            1
        );
        assert_eq!(
            compare_words(&s, &[f32::from_bits(0x3f800001)], &0x3f80u16.to_le_bytes())
                .unwrap()
                .mismatches,
            1
        );
    }
    #[test]
    fn complete_word_scan_retains_later_errors_and_exact_original_mapping() {
        let s = specs()
            .into_iter()
            .find(|s| s.label == "after.conv4")
            .unwrap();
        let mut raw = vec![0u8; s.bytes()];
        // The oldest frame is now an authentic committed owner and observed.
        raw[..2].copy_from_slice(&0x3f80u16.to_le_bytes());
        let mut actual = vec![0.; 10240 * 4];
        actual[0] = 1.;
        let equal = compare_words(&s, &actual, &raw).unwrap();
        assert_eq!(equal.mismatches, 0);
        actual[0] = -0.;
        actual[40959] = 1.;
        let observed = compare_words(&s, &actual, &raw).unwrap();
        assert_eq!(observed.mismatches, 2);
        assert_eq!(observed.compared, 40960);
        let first = observed.first.as_ref().unwrap();
        assert_eq!(
            (first.actual_index, first.original_index, first.actual_bits),
            (0, 0, 0x80000000)
        );
        let report = observed.report(99, 100, &s);
        assert_eq!(report["unobserved_original_elements"], 0);
        assert_eq!(report["first"]["original_byte_offset"], 100);
        assert!(compare_words(&s, &actual[..30720], &raw).is_err());
    }
    #[test]
    fn summary_retains_original_order_and_separate_earlier_layer_state_evidence() {
        let s = specs();
        let layer = s
            .iter()
            .find(|s| s.layer == Some(9) && s.label == "layer_output")
            .unwrap();
        let state = s
            .iter()
            .find(|s| s.layer == Some(0) && s.label == "after.recurrent")
            .unwrap();
        let make = |s: &Spec, index| {
            WordComparison {
                compared: s.elements(),
                mismatches: 1,
                first: Some(Difference {
                    actual_index: 0,
                    original_index: 0,
                    actual_bits: 0x3f800000,
                    expected_bits: 0,
                }),
            }
            .report(index, 0, s)
        };
        let proposal = Comparison {
            repeat: 0,
            call: 0,
            words: layer.elements() + state.elements(),
            mismatches: 2,
            positions: 3,
            records: vec![make(layer, 11), make(state, 69)],
            first_failure: Some("original boundary refusal".into()),
        };
        assert_eq!(proposal.refusal(), Some("original boundary refusal"));
        let report = proposal.into_report();
        assert_eq!(report["first_original_order_difference"]["layer"], 9);
        assert_eq!(report["first_state_difference"]["layer"], 0);
        assert_eq!(report["records"].as_array().unwrap().len(), 2);
        assert_eq!(report["mismatched_float_words"], 2);
    }
    #[test]
    fn whole_record_summary_bound_fits_existing_control_and_report_envelopes() {
        let rows: Vec<_> = specs()
            .iter()
            .enumerate()
            .filter(|(_, s)| s.codec != Codec::I64)
            .map(|(index, s)| {
                WordComparison {
                    compared: s.elements(),
                    mismatches: 1,
                    first: Some(Difference {
                        actual_index: 0,
                        original_index: expected_index(s, 0),
                        actual_bits: u32::MAX,
                        expected_bits: u32::MAX,
                    }),
                }
                .report(index, RAW_BYTES, s)
            })
            .collect();
        assert_eq!(rows.len(), 1036);
        assert!(serde_json::to_vec(&rows).unwrap().len() + 64 * 1024 < 1024 * 1024);
        assert_eq!(
            specs()
                .iter()
                .filter(|s| s.repeat == 0 && s.call == 0)
                .count(),
            196
        );
        assert_eq!(
            specs()
                .iter()
                .filter(|s| s.repeat == 0 && s.call == 1)
                .count(),
            324
        );
    }
    #[test]
    fn producer_membership_and_element_strides_refuse_changed_call_shape_or_coverage() {
        let mut offset = 0u64;
        let records:Vec<_>=specs().iter().map(|s| {
            let mut stride=vec![1;s.shape.len()];
            for i in (0..stride.len()-1).rev() {stride[i]=stride[i+1]*s.shape[i+1];}
            let row=json!({"repeat":s.repeat,"call":s.call,"position":s.call,"teacher":([11,353][s.call]),
                "layer":s.layer,"label":s.label,"shape":s.shape,"dtype":s.codec.name(),"device":"cpu",
                "original_stride":stride,"offset":offset,"bytes":s.bytes(),"sha256":"0".repeat(64)});
            offset+=s.bytes() as u64; row
        }).collect();
        let good = json!({"records":records,"raw":{"bytes":RAW_BYTES,"file":"raw.bin"}});
        assert!(validate_records(&good).is_ok());
        for (index, key, value) in [
            (0, "call", json!(1)),
            (3, "shape", json!([1, 1, 5119])),
            (0, "offset", json!(1)),
            (1, "original_stride", json!([100, 0, 0])),
        ] {
            let mut bad = good.clone();
            bad["records"][index][key] = value;
            assert!(validate_records(&bad).is_err(), "{index}/{key}");
        }
    }
}
