//! Independent Torch2.13/cf30153 + HF5.17 eager attention; no ZG oracle.
use sha2::{Digest, Sha256};
use std::io::Read;
use zerogravity_core::{Error, Result, accelerator::program::TorchCpuAttentionGeometry};

const BYTES: usize = 13_488_016;
const SHA: &str = "0b215c2fe121ddd8fc50908e4aa7342a8251df84a9fbea40c581a54d6a9e17af";
pub(super) struct Case {
    pub geometry: TorchCpuAttentionGeometry,
    pub position: usize,
    pub inputs: Vec<Vec<u8>>,
    pub expected: Vec<Vec<u8>>,
}
pub(super) fn read() -> Result<Vec<Case>> {
    let path = std::env::var_os("ZG_SOURCE_ATTENTION_REFERENCE")
        .ok_or_else(|| Error::State("missing original attention archive".into()))?;
    let mut raw = Vec::with_capacity(BYTES + 1);
    std::fs::File::open(path)?
        .take((BYTES + 1) as u64)
        .read_to_end(&mut raw)?;
    if raw.len() != BYTES || format!("{:x}", Sha256::digest(&raw)) != SHA {
        return Err(Error::State("original attention archive identity".into()));
    }
    let mut cursor = raw.as_slice();
    if take(&mut cursor, 8)? != b"ZGATT001" || number(&mut cursor)? != 40 {
        return Err(Error::State("original attention archive schema".into()));
    }
    let mut cases = Vec::with_capacity(40);
    for _ in 0..40 {
        let tokens = number(&mut cursor)?;
        let geometry = TorchCpuAttentionGeometry {
            query_heads: number(&mut cursor)?,
            kv_heads: number(&mut cursor)?,
            width: number(&mut cursor)?,
            rotary: 64,
            theta: 10_000_000.,
        };
        let position = tokens
            .checked_sub(1)
            .ok_or_else(|| Error::State("empty attention case".into()))?;
        let (q, kv, _) = geometry.validate(position)?;
        let mut input_extents = vec![q, kv, kv, q];
        if position > 0 {
            input_extents.extend([position * kv; 2]);
        }
        let output_extents = [
            q,
            kv,
            tokens * kv,
            tokens * kv,
            geometry.query_heads * tokens,
            geometry.query_heads * tokens,
            geometry.query_heads * tokens,
            q,
            q,
        ];
        let mut fields = |extents: &[usize]| -> Result<Vec<Vec<u8>>> {
            extents
                .iter()
                .map(|n| {
                    let bytes = take(&mut cursor, n * 4)?;
                    for word in bytes.chunks_exact(4) {
                        let bits = u32::from_le_bytes(word.try_into().expect("four bytes"));
                        if bits & 0xffff != 0 || !f32::from_bits(bits).is_finite() {
                            return Err(Error::State(
                                "original attention finite widened BF16".into(),
                            ));
                        }
                    }
                    Ok(bytes.to_vec())
                })
                .collect()
        };
        let inputs = fields(&input_extents)?;
        let expected = fields(&output_extents)?;
        cases.push(Case {
            geometry,
            position,
            inputs,
            expected,
        });
    }
    if !cursor.is_empty() {
        return Err(Error::State("original attention trailing bytes".into()));
    }
    Ok(cases)
}
fn take<'a>(cursor: &mut &'a [u8], n: usize) -> Result<&'a [u8]> {
    let (left, right) = cursor
        .split_at_checked(n)
        .ok_or_else(|| Error::State("short original attention field".into()))?;
    *cursor = right;
    Ok(left)
}
fn number(cursor: &mut &[u8]) -> Result<usize> {
    usize::try_from(u64::from_le_bytes(
        take(cursor, 8)?.try_into().expect("eight bytes"),
    ))
    .map_err(|_| Error::Limit("attention archive integer".into()))
}
