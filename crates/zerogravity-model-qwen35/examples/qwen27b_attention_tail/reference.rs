use super::{io::*, reference_pin::Pin, source};
use serde_json::Value;
use std::{fs::File, os::unix::fs::FileExt, path::Path};
use zerogravity_model_qwen35::operator_slice::attention_tail::WindowAttentionTail;
fn stages(position: usize) -> [(&'static str, usize); 28] {
    [
        ("input", 5120),
        ("normalized_input", 5120),
        ("query_gate_projection", 12288),
        ("key_projection", 1024),
        ("value_projection", 1024),
        ("query", 6144),
        ("gate", 6144),
        ("normalized_query", 6144),
        ("normalized_key", 1024),
        ("rotated_query", 6144),
        ("rotated_key", 1024),
        ("keys", (position + 1) * 1024),
        ("values", (position + 1) * 1024),
        ("dots", (position + 1) * 24),
        ("scores", (position + 1) * 24),
        ("probabilities", (position + 1) * 24),
        ("context", 6144),
        ("gated_attention", 6144),
        ("attention_branch", 5120),
        ("post_attention", 5120),
        ("normalized_ffn", 5120),
        ("ffn_gate", 17408),
        ("ffn_up", 17408),
        ("ffn_gated", 17408),
        ("ffn_branch", 5120),
        ("residual", 5120),
        ("normalized", 5120),
        ("logits", 248320),
    ]
}
const WORDS: usize = 3187696;
pub(super) struct Reference {
    manifest: Value,
    raw: File,
    before: [u64; 7],
    raw_sha: String,
    raw_bytes: u64,
}
pub(super) fn accepted(pin: Option<&Pin>) -> Proof<Value> {
    let p =
        pin.ok_or("ROOT original attention-tail reference is not accepted; no learned execution")?;
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
        value["schema"] == "zg.actual-original27b-attention-tail-reference.v1"
            && value["status"] == "ACTUAL_ORIGINAL27B_ATTENTION_TAIL_REFERENCE_ONLY"
            && value["snapshot"] == source::SNAPSHOT
            && value["config_sha256"] == source::CONFIG
            && value["index_sha256"] == source::INDEX
            && value["original_source_and_owned_lifetime_closed"] == true,
        "accepted actual original final63 source/reference",
    )?;
    need(
        value["cases"] == 4
            && value["records"] == 224
            && value["causal_cohorts"] == serde_json::json!([1, 3])
            && value["original_repetitions"] == 2
            && value["compared_bf16_words"] == WORDS
            && value["input_trained_full_model_producer_qualified"] == false
            && value["rust_numerical_parity_qualified"] == false
            && value["whole_model_state_gpu_serving_or_performance_qualified"] == false,
        "complete explicit expected-only row cohort/no permit",
    )?;
    need(
        value["scope"]["external_boundary_only"] == true
            && value["scope"]["diagnostic_causal_kv_only"] == true
            && value["scope"]["attention_math_numerically_qualified"] == false
            && value["scope"]["whole_model_qualified"] == false,
        "external causal-prefix reference is no numerical/model permit",
    )?;
    Ok(value)
}
pub(super) fn validate_records(m: &Value) -> Proof<()> {
    let rows = m["records"]
        .as_array()
        .ok_or("actual original stage records")?;
    need(rows.len() == 224, "all original nine stages/eight calls")?;
    let mut offset = 0;
    let cases = [(0, 0, 11), (1, 0, 353), (1, 1, 977), (1, 2, 42)];
    for (case, (cohort, row, seed)) in cases.into_iter().enumerate() {
        for repeat in 0..2 {
            for (stage, (label, count)) in stages(row).iter().enumerate() {
                let r = &rows[case * 56 + repeat * 28 + stage];
                need(
                    r["cohort"] == cohort
                        && r["row"] == row
                        && r["position"] == row
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
        offset == WORDS * 2 && m["raw"]["bytes"] == offset && m["raw"]["file"] == "raw.bin",
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
            manifest["schema"] == "zg.original27b-attention-tail-capture.v1"
                && manifest["status"] == "CAPTURED_ORIGINAL27B_ATTENTION_TAIL_REQUIRES_VALIDATION"
                && manifest["snapshot"] == source::SNAPSHOT
                && manifest["scope"] == receipt["scope"],
            "actual captured original final63",
        )?;
        validate_records(&manifest)?;
        let raw = open(Path::new(string(&receipt["raw"], "path")?))?;
        let before = stamp(&raw)?;
        let bytes = receipt["raw"]["bytes"].as_u64().ok_or("raw length")?;
        let raw_sha = string(&receipt["raw"], "sha256")?.to_owned();
        need(
            bytes == (WORDS * 2) as u64
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
            case < 4 && repeat < 2 && stage < 28,
            "closed reference record selector",
        )?;
        let r = &self.manifest["records"][case * 56 + repeat * 28 + stage];
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
        out: &WindowAttentionTail,
    ) -> Proof<usize> {
        need(out.layer() == 63, "genuine final layer63 source binding")?;
        let row = if case < 2 { 0 } else { case - 1 };
        let mut total = 0;
        for (stage, (label, _)) in stages(row).iter().enumerate() {
            let actual = out.stage(label)?;
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
                        stages(row)[stage].0,
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
    fn unsealed_reference_refuses_before_source_open() {
        assert!(accepted(None).is_err());
    }
    #[test]
    fn causal_stage_extents_distinguish_single_row_and_retained_history() {
        assert_eq!(stages(0)[11], ("keys", 1024));
        assert_eq!(stages(2)[11], ("keys", 3072));
        assert_eq!(stages(2)[15], ("probabilities", 72));
        assert_eq!(stages(0)[27], ("logits", 248320));
    }
}
