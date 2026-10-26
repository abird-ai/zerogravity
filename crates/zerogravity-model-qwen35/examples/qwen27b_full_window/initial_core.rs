//! Initial-core only; expected payloads open after genuine source-prefix math.
use super::*;
use zerogravity_model_qwen35::source::binding::InitialCoreStage as S;
#[path = "initial_core_pin.rs"]
mod pin;
const FIELDS: [(&str, S); 10] = [
    ("convolution", S::Convolution),
    ("activated", S::Activated),
    ("query", S::Query),
    ("key", S::Key),
    ("normalized_query", S::NormalizedQuery),
    ("normalized_key", S::NormalizedKey),
    ("scaled_query", S::ScaledQuery),
    ("beta_value", S::BetaValue),
    ("query_key", S::QueryKey),
    ("core_f32", S::CoreF32),
];
fn inner_specs() -> Vec<Spec> {
    let mut rows = Vec::with_capacity(116);
    for s in super::layer9::prefix_specs() {
        let insert = s.label == "layer9.a";
        let repeat = s.repeat;
        rows.push(s);
        if insert {
            for (name, _) in FIELDS {
                let (shape, codec) = match name {
                    "convolution" => (vec![1, 10240, 7], Codec::Bf16),
                    "activated" => (vec![1, 10240, 4], Codec::Bf16),
                    "beta_value" | "core_f32" => (vec![1, 48, 64, 128], Codec::F32),
                    "query_key" => (vec![1, 48, 1, 64, 64], Codec::F32),
                    _ => (vec![1, 48, 1, 128], Codec::F32),
                };
                rows.push(Spec {
                    repeat,
                    call: 0,
                    layer: Some(9),
                    label: format!("inner.{name}"),
                    shape,
                    codec,
                });
            }
        }
    }
    rows
}
fn bytes() -> u64 {
    inner_specs().iter().map(|s| s.bytes() as u64).sum()
}
pub(in super::super) fn accepted() -> Proof<Value> {
    accepted_pin(pin::ACTUAL_REFERENCE.as_ref())
}
fn accepted_pin(p: Option<&Pin>) -> Proof<Value> {
    let p = p.ok_or("ROOT original initial-core prefix reference absent")?;
    let v = json(
        Path::new(p.path),
        1024 * 1024,
        Some(&json!({"bytes":p.bytes,"sha256":p.sha256})),
    )?;
    need(
        v["schema"] == "zg.actual-original27b-initial-core-reference.v1"
            && v["status"] == "ACTUAL_ORIGINAL27B_INITIAL_CORE_PREFIX_REFERENCE_ONLY"
            && v["snapshot"] == source::SNAPSHOT
            && v["config_sha256"] == source::CONFIG
            && v["index_sha256"] == source::INDEX
            && v["records"] == 116
            && v["raw_bytes"] == bytes()
            && v["text_roles"] == 135
            && v["cache_domains"] == 18
            && v["original_object_layers"] == 64
            && v["layers"] == 10
            && v["calls"] == 1
            && v["original_repetitions"] == 2
            && v["original_source_and_owned_lifetime_closed"] == true
            && v["prefix_reference_join"]["overlapping_records"] == 96
            && v["prefix_reference_join"]["all_exact"] == true
            && v["full_reference_join"]["overlapping_records"] == 66
            && v["full_reference_join"]["all_overlapping_full_records_exact"] == true
            && v["rust_numerical_parity_qualified"] == false
            && v["mkl_exp_producer_qualified"] == false
            && v["state_root_native_serving_performance_qualified"] == false,
        "actual original-only inner prefix/old overlap scope",
    )?;
    Ok(v)
}
pub(in super::super) struct Inner {
    manifest: Value,
    specs: Vec<Spec>,
    manifest_owner: Held,
    raw: Held,
}
impl Inner {
    pub(in super::super) fn open(v: &Value) -> Proof<Self> {
        let manifest_owner = Held::pinned(&v["manifest"], 16 * 1024 * 1024)?;
        let m: Value = serde_json::from_slice(
            &manifest_owner.read(0, usize::try_from(manifest_owner.bytes)?)?,
        )?;
        need(
            m["schema"] == "zg.original27b-initial-core-capture.v1"
                && m["status"] == "CAPTURED_ORIGINAL27B_INITIAL_CORE_REQUIRES_VALIDATION"
                && m["snapshot"] == source::SNAPSHOT
                && m["scope"] == v["scope"]
                && m["original_full_forward"] == false,
            "same original inner prefix manifest",
        )?;
        let specs = inner_specs();
        let records = m["records"].as_array().ok_or("inner records")?;
        need(
            records.len() == 116 && records.len() == specs.len(),
            "exact complete inner cohort",
        )?;
        let mut offset = 0u64;
        for (r, s) in records.iter().zip(&specs) {
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
                "true original inner producer/layout/coverage",
            )?;
            let strides = r["original_stride"].as_array().ok_or("inner strides")?;
            need(strides.len() == s.shape.len(), "inner exact stride rank")?;
            let span = s.shape.iter().zip(strides).try_fold(0u64, |n, (d, s)| {
                n.checked_add(
                    ((*d - 1) as u64)
                        .checked_mul(s.as_u64().ok_or("inner stride")?)
                        .ok_or("inner stride product")?,
                )
                .ok_or("inner stride span")
            })?;
            need(span < s.elements() as u64, "inner defined actual stride")?;
            offset += s.bytes() as u64;
        }
        need(
            offset == bytes() && m["raw"]["bytes"] == offset && m["raw"]["file"] == "raw.bin",
            "whole exact inner archive",
        )?;
        let raw = Held::pinned(&v["raw"], bytes())?;
        need(
            raw.bytes == bytes() && raw.sha == m["raw"]["sha256"],
            "same whole inner raw",
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
        a: &WindowDiagnosticReadback<'_>,
        source: zerogravity_core::accelerator::SourceIdentity,
    ) -> Proof<Value> {
        need(
            a.completed_prefix_layers() == Some(10)
                && a.base().frontier() == 0
                && a.logits().is_err(),
            "unpublished genuine prefix; no final/Head",
        )?;
        // Counterfactual primitive uses only genuine computed Rust tape operands.
        // Original payloads remain comparison-only; model math is unchanged.
        let fma_qk = zerogravity_cpu::numerics::torch_cpu::reduction::padded64_qk_fma128(
            source,
            a.initial_core(S::ScaledQuery)?,
            a.initial_core(S::NormalizedKey)?,
        )?;
        let beta_value = a.initial_core(S::BetaValue)?;
        let fma_core = beta_value
            .iter()
            .enumerate()
            .map(|(i, v)| 0.0 + fma_qk[i / 128] * v)
            .collect::<Vec<_>>();
        let mut candidate_records = Vec::with_capacity(2);
        let mut records = Vec::with_capacity(10);
        let mut total = 0;
        let mut mismatched = 0;
        for (r, s) in self.manifest["records"]
            .as_array()
            .ok_or("inner records")?
            .iter()
            .zip(&self.specs)
        {
            if s.repeat != 0 || !s.label.starts_with("inner.") {
                continue;
            }
            let name = s.label.strip_prefix("inner.").ok_or("inner label")?;
            let stage = FIELDS
                .iter()
                .find(|(n, _)| *n == name)
                .ok_or("closed inner stage")?
                .1;
            let actual = a.initial_core(stage)?;
            let count = match name {
                "convolution" | "activated" => 10240,
                "query_key" => 48,
                _ => 6144,
            };
            need(actual.len() == count, "same source active initial extent")?;
            let raw = self
                .raw
                .read(r["offset"].as_u64().ok_or("inner offset")?, s.bytes())?;
            need(
                sha(&raw) == string(r, "sha256")?,
                "complete genuine inner record",
            )?;
            let mut differences = 0;
            let mut first = None;
            for (i, x) in actual.iter().enumerate() {
                let j = original_index(name, i);
                let at = j * s.codec.width();
                let expected = if s.codec == Codec::Bf16 {
                    u32::from(u16::from_le_bytes(raw[at..at + 2].try_into()?)) << 16
                } else {
                    u32::from_le_bytes(raw[at..at + 4].try_into()?)
                };
                if !x.is_finite() || x.to_bits() != expected {
                    differences += 1;
                    if first.is_none() {
                        first = Some(
                            json!({"actual_index":i,"original_index":j,"actual_bits":format!("{:08x}",x.to_bits()),"expected_bits":format!("{:08x}",expected)}),
                        );
                    }
                }
            }
            if let Some(candidate) = match name {
                "query_key" => Some(fma_qk.as_slice()),
                "core_f32" => Some(fma_core.as_slice()),
                _ => None,
            } {
                let mut bad = 0;
                let mut first = None;
                for (i, x) in candidate.iter().enumerate() {
                    let j = original_index(name, i);
                    let at = j * 4;
                    let expected = u32::from_le_bytes(raw[at..at + 4].try_into()?);
                    if x.to_bits() != expected {
                        bad += 1;
                        if first.is_none() {
                            first = Some(
                                json!({"actual_index":i,"original_index":j,"actual_bits":format!("{:08x}",x.to_bits()),"expected_bits":format!("{:08x}",expected)}),
                            );
                        }
                    }
                }
                candidate_records.push(json!({"label":s.label,"compared_words":candidate.len(),"mismatched_words":bad,"first":first}));
            }
            total += count;
            mismatched += differences;
            records.push(json!({"label":s.label,"dtype":s.codec.name(),"compared_words":count,"mismatched_words":differences,
                "original_elements":s.elements(),"unobserved_original_elements":s.elements()-count,"first":first}));
        }
        need(
            records.len() == 10,
            "all available inner producers compared before refusal",
        )?;
        Ok(
            json!({"compared_words":total,"mismatched_words":mismatched,"first":records.iter().find(|r|r["mismatched_words"]!=0).cloned(),"records":records,"candidate_fma_from_computed_operands_only":candidate_records}),
        )
    }
    pub(in super::super) fn verify_after(&self) -> Proof<()> {
        self.manifest_owner.verify()?;
        self.raw.verify()
    }
}
fn original_index(name: &str, i: usize) -> usize {
    match name {
        "convolution" => i * 7 + 3,
        "activated" => i * 4 + 3,
        "query_key" => i * 64 * 64,
        "beta_value" | "core_f32" => i / 128 * 64 * 128 + i % 128,
        _ => i,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_inner_coverage_and_active_padded_coordinates() {
        assert_eq!(inner_specs().len(), 116);
        assert!(accepted_pin(None).is_err());
        assert_eq!(original_index("query_key", 47), 47 * 4096);
        assert_eq!(original_index("core_f32", 6143), 47 * 8192 + 127);
        assert_eq!(original_index("convolution", 10239), 71676);
    }
}
