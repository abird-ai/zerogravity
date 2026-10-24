use super::io::*;
use serde_json::{Value, json};
#[derive(Default)]
pub struct Difference {
    pub words: usize,
    first: Option<(usize, u32, u32)>,
}
impl Difference {
    pub fn report(&self) -> Value {
        json!({"words":self.words,"first":self.first.map(|(i,e,a)|json!([i,format!("{e:08x}"),format!("{a:08x}")]))})
    }
}
pub fn scan(raw: &[u8], actual: &[f32]) -> Proof<Difference> {
    need(
        raw.len() == actual.len() * 2 && !actual.is_empty() && actual.len() <= 262144,
        "whole BF16 extent",
    )?;
    let mut out = Difference::default();
    for (i, (w, a)) in raw.chunks_exact(2).zip(actual).enumerate() {
        let e = u32::from(u16::from_le_bytes(w.try_into()?)) << 16;
        need(
            f32::from_bits(e).is_finite() && a.is_finite(),
            "finite authentic BF16 output",
        )?;
        if e != a.to_bits() {
            out.words += 1;
            out.first.get_or_insert((i, e, a.to_bits()));
        }
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_bf16_signed_zero_and_all_mismatches() {
        let d = scan(&[0x80, 0x3f, 0, 0x80, 0, 0x40], &[0.0, 0.0, 2.0]).unwrap();
        assert_eq!(d.words, 2);
        assert!(scan(&[0x80, 0x7f], &[0.0]).is_err());
    }
}
