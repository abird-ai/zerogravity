use super::{io::*, reference_pin::Pin, source};
use serde_json::Value;
use std::{fs::File, os::unix::fs::FileExt, path::Path};
use zerogravity_model_qwen35::operator_slice::output_tail::WindowOutputTail;
pub(super) const STAGES: [(&str, usize); 9] = [
    ("input", 5120),
    ("normalized_ffn", 5120),
    ("gate", 17408),
    ("up", 17408),
    ("gated", 17408),
    ("branch", 5120),
    ("residual", 5120),
    ("normalized", 5120),
    ("logits", 248320),
];
pub(super) struct Reference {
    manifest: Value,
    raw: File,
    before: [u64; 7],
    raw_sha: String,
    raw_bytes: u64,
}
pub(super) fn accepted(pin: Option<&Pin>) -> Proof<Value> {
    let p =
        pin.ok_or("ROOT original final-tail reference is not accepted; no learned execution")?;
    need(
        Path::new(p.path).is_absolute()
            && p.bytes > 0
            && p.bytes <= 1024 * 1024
            && p.sha256.len() == 64,
        "finite accepted reference identity",
    )?;
    let value = json(
        Path::new(p.path),
        1024 * 1024,
        Some(&serde_json::json!({"bytes":p.bytes,"sha256":p.sha256})),
    )?;
    need(
        value["schema"] == "zg.actual-original27b-final-tail-reference.v1"
            && value["status"] == "ACTUAL_ORIGINAL27B_FINAL_TAIL_REFERENCE_ONLY"
            && value["snapshot"] == source::SNAPSHOT
            && value["config_sha256"] == source::CONFIG
            && value["index_sha256"] == source::INDEX
            && value["original_source_and_owned_lifetime_closed"] == true,
        "accepted actual original final63 source/reference",
    )?;
    need(
        value["cases"] == 4
            && value["records"] == 72
            && value["independent_cohorts"] == serde_json::json!([1, 3])
            && value["original_repetitions"] == 2
            && value["compared_bf16_words"] == 2_609_152
            && value["input_trained_full_model_producer_qualified"] == false
            && value["rust_numerical_parity_qualified"] == false
            && value["whole_model_state_gpu_serving_or_performance_qualified"] == false,
        "complete explicit expected-only row cohort/no permit",
    )?;
    Ok(value)
}
pub(super) fn validate_records(m: &Value) -> Proof<()> {
    let rows = m["records"]
        .as_array()
        .ok_or("actual original stage records")?;
    need(rows.len() == 72, "all original nine stages/eight calls")?;
    let mut offset = 0;
    let cases = [(0, 0, 11), (1, 0, 353), (1, 1, 977), (1, 2, 42)];
    for (case, (cohort, row, seed)) in cases.into_iter().enumerate() {
        for repeat in 0..2 {
            for (stage, (label, count)) in STAGES.iter().enumerate() {
                let r = &rows[case * 18 + repeat * 9 + stage];
                need(
                    r["cohort"] == cohort
                        && r["row"] == row
                        && r["seed"] == seed
                        && r["repeat"] == repeat
                        && r["layer"] == 63
                        && r["label"] == *label
                        && r["offset"] == offset
                        && r["bytes"] == count * 2
                        && r["shape"] == serde_json::json!([1, count])
                        && r["stride"] == serde_json::json!([count, 1])
                        && r["dtype"] == "torch.bfloat16"
                        && r["device"] == "cpu",
                    "actual case/repeat/stage shape/dtype/order/full extent",
                )?;
                offset += count * 2;
            }
        }
    }
    need(
        offset == 5_218_304 && m["raw"]["bytes"] == offset && m["raw"]["file"] == "raw.bin",
        "complete raw coverage/EOF",
    )
}
impl Reference {
    pub(super) fn open(receipt: &Value) -> Proof<Self> {
        let manifest = json(
            Path::new(string(&receipt["manifest"], "path")?),
            1024 * 1024,
            Some(&receipt["manifest"]),
        )?;
        need(
            manifest["schema"] == "zg.original27b-final-tail-capture.v1"
                && manifest["status"] == "CAPTURED_ORIGINAL27B_FINAL_TAIL_REQUIRES_VALIDATION"
                && manifest["snapshot"] == source::SNAPSHOT,
            "actual captured original final63",
        )?;
        validate_records(&manifest)?;
        let raw = open(Path::new(string(&receipt["raw"], "path")?))?;
        let before = stamp(&raw)?;
        let bytes = receipt["raw"]["bytes"].as_u64().ok_or("raw length")?;
        let raw_sha = string(&receipt["raw"], "sha256")?.to_owned();
        need(
            bytes == 5_218_304
                && hash(&raw, bytes)? == raw_sha
                && manifest["raw"]["sha256"] == raw_sha,
            "all held original raw words SHA/EOF",
        )?;
        Ok(Self {
            manifest,
            raw,
            before,
            raw_sha,
            raw_bytes: bytes,
        })
    }
    fn payload(&self, case: usize, repeat: usize, stage: usize) -> Proof<Vec<u8>> {
        need(
            case < 4 && repeat < 2 && stage < 9,
            "closed reference record selector",
        )?;
        let r = &self.manifest["records"][case * 18 + repeat * 9 + stage];
        let n = r["bytes"].as_u64().ok_or("record byte count")? as usize;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(n)?;
        bytes.resize(n, 0);
        self.raw
            .read_exact_at(&mut bytes, r["offset"].as_u64().ok_or("record offset")?)?;
        need(
            sha(&bytes) == string(r, "sha256")? && stamp(&self.raw)? == self.before,
            "same actual complete record bytes/source owner",
        )?;
        Ok(bytes)
    }
    pub(super) fn input(&self, case: usize, repeat: usize) -> Proof<Vec<u8>> {
        self.payload(case, repeat, 0)
    }
    pub(super) fn compare(
        &self,
        case: usize,
        repeat: usize,
        out: &WindowOutputTail,
    ) -> Proof<usize> {
        need(out.layer() == 63, "genuine final layer63 source binding")?;
        let values = [
            out.input()?,
            out.normalized_ffn()?,
            out.gate()?,
            out.up()?,
            out.gated()?,
            out.branch()?,
            out.residual()?,
            out.normalized()?,
            out.full_vocabulary()?,
        ];
        let mut total = 0;
        for (stage, actual) in values.iter().enumerate() {
            let expected = self.payload(case, repeat, stage)?;
            need(
                actual.len() * 2 == expected.len(),
                "full original defined stage extent",
            )?;
            for (i, (v, b)) in actual.iter().zip(expected.chunks_exact(2)).enumerate() {
                let bits = u32::from(u16::from_le_bytes([b[0], b[1]])) << 16;
                need(
                    v.to_bits() == bits,
                    format!(
                        "cohort/row case{case} repeat{repeat} stage{} word{i}/{} actual={:08x} expected={bits:08x}",
                        STAGES[stage].0,
                        actual.len(),
                        v.to_bits()
                    ),
                )?;
            }
            total += actual.len();
        }
        Ok(total)
    }
    pub(super) fn verify_after(&self) -> Proof<()> {
        need(
            stamp(&self.raw)? == self.before && hash(&self.raw, self.raw_bytes)? == self.raw_sha,
            "same whole original raw after computation/caller drops",
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unsealed_reference_refuses_before_any_source_or_raw_open() {
        assert!(
            accepted(None)
                .unwrap_err()
                .to_string()
                .contains("not accepted")
        );
    }
    #[test]
    fn exact_stage_schema_rejects_partial_reordered_and_foreign_rows() {
        let mut rows = Vec::new();
        let mut at = 0;
        for (cohort, row, seed) in [(0, 0, 11), (1, 0, 353), (1, 1, 977), (1, 2, 42)] {
            for repeat in 0..2 {
                for (label, count) in STAGES {
                    rows.push(serde_json::json!({"cohort":cohort,"row":row,"seed":seed,"repeat":repeat,"layer":63,"label":label,"offset":at,"bytes":count*2,"shape":[1,count],"stride":[count,1],"dtype":"torch.bfloat16","device":"cpu"}));
                    at += count * 2;
                }
            }
        }
        let good = serde_json::json!({"records":rows,"raw":{"file":"raw.bin","bytes":at}});
        validate_records(&good).unwrap();
        for k in 0..5 {
            let mut bad = good.clone();
            match k {
                0 => {
                    bad["records"].as_array_mut().unwrap().pop();
                }
                1 => bad["records"][2]["bytes"] = 2.into(),
                2 => bad["records"][18]["cohort"] = 9.into(),
                3 => bad["records"][4]["shape"] = serde_json::json!([3, 17408]),
                4 => bad["records"][0]["offset"] = 1.into(),
                _ => unreachable!(),
            }
            assert!(validate_records(&bad).is_err());
        }
    }
}
