//! Post-commit observation only; no checkpoint bytes supply runtime operands.
use super::*;
use std::collections::BTreeSet;
use zerogravity::accelerator::program::StateSlot;

fn word(bytes: &[u8], dtype: &str, index: usize) -> Result<u32> {
    match dtype {
        "BF16" => Ok(u32::from(u16::from_le_bytes(
            bytes[index * 2..index * 2 + 2].try_into().unwrap(),
        )) << 16),
        "F32" => Ok(u32::from_le_bytes(
            bytes[index * 4..index * 4 + 4].try_into().unwrap(),
        )),
        _ => Err(Error::Malformed("checkpoint reference float codec".into())),
    }
}
impl Expected {
    pub fn compare_logits(&self, call: &Call, actual: &[f32]) -> Result<usize> {
        self.compare_logit_digest(call, actual)?;
        let i = self.find(call, "logits", None)?;
        let r = &self.doc.records[i];
        let raw = self.load(i)?;
        need(
            r.dtype == "BF16" && r.shape == [1, 1, 248320] && actual.len() == 248320,
            "unobserved complete vocabulary extent",
        )?;
        for (i, value) in actual.iter().enumerate() {
            if value.to_bits() != word(&raw, &r.dtype, i)? {
                return Err(Error::State(format!("unobserved logits word={i}")));
            }
        }
        Ok(actual.len())
    }
    /// Every populated original forward records the complete BF16 output digest.
    /// Sparse checkpoints additionally compare individual archived float words.
    pub fn compare_logit_digest(&self, call: &Call, actual: &[f32]) -> Result<usize> {
        compare_digest(call, actual)
    }
    pub fn compare_domains(
        &self,
        call: &Call,
        actual: &[(StateSlot, Vec<f32>)],
        capacity: usize,
    ) -> Result<(usize, usize)> {
        let mut seen = BTreeSet::new();
        let mut words = 0;
        let mut tails = 0;
        let mut planes = 0;
        for (slot, values) in actual {
            need(seen.insert(*slot), "unique post-commit domain")?;
            let (layer, names): (_, &[&str]) = match slot {
                StateSlot::Conv(d) => (d.0, &["after.conv4"]),
                StateSlot::Recurrent(d) => (d.0, &["after.recurrent"]),
                StateSlot::Kv(d) => (d.0, &["after.key", "after.value"]),
            };
            need(
                values.len().is_multiple_of(names.len()),
                "whole original state planes",
            )?;
            let plane_words = values.len() / names.len();
            for (plane, &name) in names.iter().enumerate() {
                let i = self.find(call, name, Some(layer))?;
                let r = &self.doc.records[i];
                let raw = self.load(i)?;
                let kv = matches!(slot, StateSlot::Kv(_));
                if kv {
                    need(
                        r.shape.len() == 4
                            && r.shape[0] == 1
                            && r.shape[2] == call.position + call.rows
                            && r.shape[2] <= capacity
                            && plane_words == r.shape[2] * r.shape[1] * r.shape[3],
                        "post-commit original KV geometry",
                    )?;
                    // Original snapshot readback checks the entire physical tail
                    // before returning the active token-major plane.
                    tails += (capacity - r.shape[2]) * r.shape[1] * r.shape[3];
                } else {
                    need(
                        plane_words * if r.dtype == "BF16" { 2 } else { 4 } == raw.len(),
                        "post-commit state extent",
                    )?;
                }
                for (j, value) in values[plane * plane_words..(plane + 1) * plane_words]
                    .iter()
                    .enumerate()
                {
                    let index = if kv {
                        let (heads, tokens, dimension) = (r.shape[1], r.shape[2], r.shape[3]);
                        let token = j / (heads * dimension);
                        let head = (j / dimension) % heads;
                        (head * tokens + token) * dimension + j % dimension
                    } else {
                        j
                    };
                    if value.to_bits() != word(&raw, &r.dtype, index)? {
                        return Err(Error::State(format!("post-commit {name}/{layer} word={j}")));
                    }
                    words += 1;
                }
                planes += 1;
            }
        }
        need(
            seen.len() == self.domains && planes == self.planes,
            "all original post-commit domains/planes",
        )?;
        self.unchanged()?;
        Ok((words, tails))
    }
}

fn compare_digest(call: &Call, actual: &[f32]) -> Result<usize> {
    let Some(expected) = &call.logits_sha256 else {
        return Ok(0);
    };
    need(actual.len() == 248320, "complete digest vocabulary")?;
    let mut h = Sha256::new();
    let mut bytes = [0u8; 2048];
    for values in actual.chunks(1024) {
        for (i, value) in values.iter().enumerate() {
            need(
                value.is_finite() && value.to_bits() & 65535 == 0,
                "actual BF16 digest words",
            )?;
            bytes[i * 2..i * 2 + 2]
                .copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        h.update(&bytes[..values.len() * 2]);
    }
    let actual_digest = format!("{:x}", h.finalize());
    if actual_digest != *expected {
        return Err(Error::State(format!(
            "all original BF16 logit bits digest: repeat={} case={} call={} position={} rows={} expected={expected} actual={actual_digest}",
            call.repeat, call.case, call.call, call.position, call.rows
        )));
    }
    Ok(actual.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Call, Vec<f32>) {
        let mut actual = vec![0.0; 248320];
        actual[0] = -0.0;
        actual[248319] = 1.0;
        let call = Call {
            repeat: 0,
            case: 4,
            call: 216,
            phase: "prefill".into(),
            position: 13824,
            rows: 64,
            input_ids: vec![11; 64],
            selected: 248319,
            max_bits: 1.0f32.to_bits(),
            checkpoint: false,
            logits_sha256: Some(
                "f09f226b02fb56ab91a48728078b9a96b399167c31a793983ca67264bcdf286a".into(),
            ),
        };
        (call, actual)
    }

    #[test]
    fn digest_coverage_counts_vocabulary_words_not_hash_characters() {
        let (call, actual) = fixture();
        let words = compare_digest(&call, &actual).unwrap();
        assert_eq!(words, 248320);
        assert_eq!(words * 516, 128133120);
    }

    #[test]
    fn digest_checks_all_chunks_and_preserves_failure_coordinates() {
        let (call, mut actual) = fixture();
        for index in [0, 1023, 1024, 248319] {
            let before = actual[index];
            actual[index] = f32::from_bits(before.to_bits() ^ 0x10000);
            let error = compare_digest(&call, &actual).unwrap_err().to_string();
            assert!(error.contains("repeat=0 case=4 call=216 position=13824 rows=64"));
            assert!(error.contains("expected=f09f226b"));
            assert!(error.contains(" actual="));
            actual[index] = before;
        }
    }

    #[test]
    fn digest_refuses_wrong_extent_non_bf16_and_nonfinite_words() {
        let (call, mut actual) = fixture();
        assert!(compare_digest(&call, &actual[..248319]).is_err());
        for value in [
            f32::from_bits(1),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            actual[1024] = value;
            assert!(compare_digest(&call, &actual).is_err());
        }
    }

    #[test]
    fn historical_absent_digest_adds_no_coverage() {
        let (mut call, actual) = fixture();
        call.logits_sha256 = None;
        assert_eq!(compare_digest(&call, &actual).unwrap(), 0);
    }
}
