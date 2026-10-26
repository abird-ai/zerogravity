//! Distinct initial prefix expectation; never an operand or full-model permit.
use super::*;
use zerogravity_model_qwen35::source::binding::WindowDiagnosticLayer9Stage as S;
#[path = "layer9_pin.rs"]
mod pin;
const STAGES: [(&str, usize, Option<S>); 16] = [
    ("input", 5120, None),
    ("input_norm", 5120, Some(S::InputNorm)),
    ("qkv", 10240, Some(S::Qkv)),
    ("z", 6144, Some(S::Z)),
    ("b", 48, Some(S::B)),
    ("a", 48, Some(S::A)),
    ("core", 6144, Some(S::Core)),
    ("norm_gate", 6144, Some(S::NormGate)),
    ("gated", 6144, Some(S::Gated)),
    ("gdn_out", 5120, Some(S::GdnOut)),
    ("post_attention", 5120, Some(S::PostAttention)),
    ("post_norm", 5120, Some(S::PostNorm)),
    ("ffn_gate", 17408, Some(S::FfnGate)),
    ("ffn_up", 17408, Some(S::FfnUp)),
    ("ffn_gated", 17408, Some(S::FfnGated)),
    ("ffn_down", 5120, Some(S::FfnDown)),
];
pub(super) fn prefix_specs() -> Vec<Spec> {
    let mut out = Vec::with_capacity(96);
    for repeat in 0..2 {
        let mut add = |layer, label: String, shape, codec| {
            out.push(Spec {
                repeat,
                call: 0,
                layer,
                label,
                shape,
                codec,
            })
        };
        add(None, "embedding".into(), vec![1, 1, 5120], Codec::Bf16);
        add(None, "positions".into(), vec![3, 1, 1], Codec::I64);
        for i in 0..9 {
            add(
                Some(i),
                "layer_output".into(),
                vec![1, 1, 5120],
                Codec::Bf16,
            );
        }
        for (label, n, _) in STAGES {
            let shape = if matches!(label, "core" | "norm_gate" | "gated") {
                vec![48, 128]
            } else {
                vec![1, 1, n]
            };
            add(Some(9), format!("layer9.{label}"), shape, Codec::Bf16);
        }
        add(
            Some(9),
            "layer_output".into(),
            vec![1, 1, 5120],
            Codec::Bf16,
        );
        let mut all = Vec::new();
        states(&mut all, repeat, 0, false);
        out.extend(all.into_iter().filter(|s| s.layer.is_some_and(|l| l <= 9)));
    }
    out
}
fn raw_bytes() -> u64 {
    prefix_specs().iter().map(|s| s.bytes() as u64).sum()
}
pub(in super::super) fn accepted() -> Proof<Value> {
    accepted_pin(pin::ACTUAL_REFERENCE.as_ref())
}
fn accepted_pin(p: Option<&Pin>) -> Proof<Value> {
    let p = p.ok_or("ROOT layer9 prefix reference pin absent")?;
    let v = json(
        Path::new(p.path),
        1024 * 1024,
        Some(&json!({"bytes":p.bytes,"sha256":p.sha256})),
    )?;
    need(
        v["schema"] == "zg.actual-original27b-layer9-prefix-reference.v1"
            && v["status"] == "ACTUAL_ORIGINAL27B_LAYER9_PREFIX_REFERENCE_ONLY"
            && v["snapshot"] == source::SNAPSHOT
            && v["config_sha256"] == source::CONFIG
            && v["index_sha256"] == source::INDEX
            && v["text_roles"] == 135
            && v["layers"] == 10
            && v["original_object_layers"] == 64
            && v["cache_domains"] == 18
            && v["calls"] == 1
            && v["original_repetitions"] == 2
            && v["records"] == 96
            && v["raw_bytes"] == raw_bytes()
            && v["full_hf_conv_width"] == 4
            && v["original_source_and_owned_lifetime_closed"] == true
            && v["full_reference_join"]["overlapping_records"] == 66
            && v["full_reference_join"]["all_overlapping_full_records_exact"] == true
            && v["rust_numerical_parity_qualified"] == false
            && v["mkl_exp_producer_qualified"] == false
            && v["state_root_native_serving_performance_qualified"] == false,
        "original135-role/18-domain prefix-only authenticated expectation",
    )?;
    Ok(v)
}
pub(in super::super) struct Prefix {
    manifest: Value,
    specs: Vec<Spec>,
    manifest_owner: Held,
    raw: Held,
}
impl Prefix {
    pub(in super::super) fn open(v: &Value) -> Proof<Self> {
        let manifest_owner = Held::pinned(&v["manifest"], 16 * 1024 * 1024)?;
        let m: Value = serde_json::from_slice(
            &manifest_owner.read(0, usize::try_from(manifest_owner.bytes)?)?,
        )?;
        need(
            m["schema"] == "zg.original27b-layer9-prefix-capture.v1"
                && m["status"] == "CAPTURED_ORIGINAL27B_LAYER9_PREFIX_REQUIRES_VALIDATION"
                && m["snapshot"] == source::SNAPSHOT
                && m["scope"] == v["scope"]
                && m["original_full_forward"] == false
                && m["cache_domains"] == 18
                && m["used_source_roles"] == 135,
            "same authentic original prefix manifest",
        )?;
        let specs = prefix_specs();
        validate_prefix_records(&m, &specs)?;
        let raw = Held::pinned(&v["raw"], raw_bytes())?;
        need(
            raw.bytes == raw_bytes() && m["raw"]["sha256"] == raw.sha,
            "same whole prefix raw",
        )?;
        Ok(Self {
            manifest: m,
            specs,
            manifest_owner,
            raw,
        })
    }
    pub(in super::super) fn compare(
        &self,
        repeat: usize,
        a: &WindowDiagnosticReadback<'_>,
    ) -> Proof<Comparison> {
        need(
            repeat < 2 && a.base().frontier() == 0,
            "initial-one prefix same current root",
        )?;
        let mut c = Comparison {
            repeat,
            call: 0,
            words: 0,
            mismatches: 0,
            positions: 0,
            records: Vec::with_capacity(48),
            first_failure: None,
        };
        for (i, (r, s)) in self.manifest["records"]
            .as_array()
            .ok_or("prefix records")?
            .iter()
            .zip(&self.specs)
            .enumerate()
        {
            if s.repeat != repeat {
                continue;
            }
            let offset = r["offset"].as_u64().ok_or("prefix offset")?;
            let raw = self.raw.read(offset, s.bytes())?;
            need(
                sha(&raw) == string(r, "sha256")?,
                "whole prefix record content",
            )?;
            if s.codec == Codec::I64 {
                need(raw == [0u8; 24], "prefix actual source positions0")?;
                c.positions += 3;
                c.records.push(json!({"original_record_index":i,"label":"positions","compared_words":3,"mismatched_words":0}));
                continue;
            }
            let l = s.layer.unwrap_or(0);
            let values = match s.label.as_str() {
                "embedding" => a.boundary(B::Embedding)?,
                "layer_output" => a.boundary(B::Layer(l))?,
                "layer9.input" => a.boundary(B::Layer(8))?,
                label if label.starts_with("layer9.") => {
                    let stage = STAGES
                        .iter()
                        .find(|(name, _, _)| label.strip_prefix("layer9.") == Some(*name))
                        .and_then(|(_, _, s)| *s)
                        .ok_or("closed layer9 producer")?;
                    a.boundary(B::Layer9(stage))?
                }
                label => {
                    let k = label.strip_prefix("after.").ok_or("prefix current state")?;
                    let slot = match k {
                        "conv4" => StateSlot::Conv(ConvDomain(l)),
                        "recurrent" => StateSlot::Recurrent(RecurrentDomain(l)),
                        "key" | "value" => StateSlot::Kv(KvDomain(l)),
                        _ => return Err("closed prefix state".into()),
                    };
                    a.state(slot)?[usize::from(k == "value")]
                }
            };
            let w = compare_words(s, values, &raw)?;
            if c.first_failure.is_none()
                && let Some(f) = &w.first
            {
                c.first_failure = Some(format!(
                    "layer9 prefix {} word{}/{} actual={:08x} expected={:08x}",
                    s.label, f.actual_index, w.compared, f.actual_bits, f.expected_bits
                ));
            }
            c.words += w.compared;
            c.mismatches += w.mismatches;
            c.records.push(w.report(i, offset, s));
        }
        need(
            c.records.len() == 48 && c.positions == 3,
            "all prefix producer/state records compared before refusal",
        )?;
        Ok(c)
    }
    pub(in super::super) fn verify_after(&self) -> Proof<()> {
        self.manifest_owner.verify()?;
        self.raw.verify()
    }
}
pub(super) fn validate_prefix_records(m: &Value, specs: &[Spec]) -> Proof<()> {
    let records = m["records"].as_array().ok_or("prefix record array")?;
    need(
        records.len() == specs.len() && specs.len() == 96,
        "complete ordered prefix record cohort",
    )?;
    let mut offset = 0u64;
    for (r, s) in records.iter().zip(specs) {
        need(
            r["repeat"] == s.repeat
                && r["call"] == 0
                && r["position"] == 0
                && r["teacher"] == 11
                && r["layer"] == s.layer.map_or(Value::Null, Value::from)
                && r["label"] == s.label
                && r["shape"] == json!(s.shape)
                && r["dtype"] == s.codec.name()
                && r["device"] == "cpu"
                && r["offset"] == offset
                && r["bytes"] == s.bytes()
                && string(r, "sha256")?.len() == 64,
            "exact prefix producer/order/shape/coverage",
        )?;
        let strides = r["original_stride"].as_array().ok_or("prefix strides")?;
        need(strides.len() == s.shape.len(), "exact prefix stride rank")?;
        let span = s.shape.iter().zip(strides).try_fold(0u64, |n, (d, s)| {
            n.checked_add(
                ((*d - 1) as u64)
                    .checked_mul(s.as_u64().ok_or("prefix nonnegative stride")?)
                    .ok_or("prefix stride product")?,
            )
            .ok_or("prefix stride span")
        })?;
        need(span < s.elements() as u64, "defined prefix stride extent")?;
        offset = offset
            .checked_add(s.bytes() as u64)
            .ok_or("prefix raw sum")?;
    }
    need(
        offset == raw_bytes() && m["raw"]["bytes"] == offset && m["raw"]["file"] == "raw.bin",
        "prefix exact raw coverage/EOF",
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prefix_has_true_original_order_and_full_18_domain_state() {
        let specs = prefix_specs();
        assert_eq!(specs.len(), 96);
        assert_eq!(
            specs
                .iter()
                .filter(|s| s.label.starts_with("after."))
                .count(),
            40
        );
        let labels: Vec<_> = specs
            .iter()
            .filter(|s| s.repeat == 0 && s.label.starts_with("layer9."))
            .map(|s| s.label.as_str())
            .collect();
        assert_eq!(labels[4..6], ["layer9.b", "layer9.a"]);
        assert_eq!(labels.len(), 16);
        assert!(accepted_pin(None).is_err());
        assert!(
            !specs
                .iter()
                .any(|s| s.label == "normalized" || s.label == "logits" || s.call != 0)
        );
    }
}
