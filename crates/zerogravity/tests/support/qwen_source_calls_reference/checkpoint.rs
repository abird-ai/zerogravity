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
            seen.len() == 42 && planes == 48,
            "all original post-commit domains/planes",
        )?;
        self.unchanged()?;
        Ok((words, tails))
    }
}
