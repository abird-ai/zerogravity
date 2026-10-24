//! Numeric diagnostics scan every word; malformed expected/source data still refuses.
use super::io::*;
use serde_json::{Value, json};

#[derive(Clone, Copy)]
pub(super) enum Encoding {
    Bf16,
    WidenedBf16,
}
#[derive(Clone, Copy, Default)]
pub(super) struct Difference {
    pub words: usize,
    pub first: Option<(usize, u32, u32)>,
}
impl Difference {
    pub fn report(self) -> Value {
        json!({"mismatched_words":self.words,"first":self.first.map(|(word,expected,actual)|
            json!({"word":word,"expected":format!("{expected:08x}"),"actual":format!("{actual:08x}")}))})
    }
    /// Compact bounded report: the metadata owner identifies each actual record by index.
    pub fn indexed(self, index: usize) -> Value {
        let (word, expected, actual) = self.first.expect("nonzero numeric difference");
        json!([
            index,
            self.words,
            word,
            format!("{expected:08x}"),
            format!("{actual:08x}")
        ])
    }
}
pub(super) fn scan(raw: &[u8], actual: &[f32], encoding: Encoding) -> Proof<Difference> {
    let width = match encoding {
        Encoding::Bf16 => 2,
        Encoding::WidenedBf16 => 4,
    };
    need(
        !actual.is_empty()
            && actual.len() <= 262144
            && actual.len().checked_mul(width) == Some(raw.len()),
        "complete bounded expected/actual record extent",
    )?;
    let mut difference = Difference::default();
    for (word, (bytes, actual)) in raw.chunks_exact(width).zip(actual).enumerate() {
        let expected = match encoding {
            Encoding::Bf16 => u32::from(u16::from_le_bytes(bytes.try_into()?)) << 16,
            Encoding::WidenedBf16 => u32::from_le_bytes(bytes.try_into()?),
        };
        need(
            expected & 0xffff == 0 && f32::from_bits(expected).is_finite() && actual.is_finite(),
            "finite original BF16/actual result; no malformed oracle fallback",
        )?;
        if expected != actual.to_bits() {
            difference.words += 1;
            difference
                .first
                .get_or_insert((word, expected, actual.to_bits()));
        }
    }
    Ok(difference)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_words_are_scanned_after_the_first_numeric_difference() {
        let expected = [0.0f32, 1.0, -0.0, 2.0];
        let raw: Vec<_> = expected.iter().flat_map(|x| x.to_le_bytes()).collect();
        let difference = scan(&raw, &[1.0, 1.0, 0.0, 3.0], Encoding::WidenedBf16).unwrap();
        assert_eq!(difference.words, 3);
        assert_eq!(difference.first, Some((0, 0, 1.0f32.to_bits())));
        let exact = scan(&raw, &expected, Encoding::WidenedBf16).unwrap();
        assert_eq!(exact.words, 0);
        assert!(exact.first.is_none());
    }
    #[test]
    fn bf16_and_widened_source_words_have_the_same_strict_comparison() {
        let raw = [0x80u8, 0x3f, 0, 0x80]; //1,-0
        let difference = scan(&raw, &[1.0, 0.0], Encoding::Bf16).unwrap();
        assert_eq!(difference.words, 1);
        assert_eq!(difference.first, Some((1, 0x80000000, 0)));
        assert_eq!(
            difference.indexed(7),
            json!([7, 1, 1, "80000000", "00000000"])
        );
    }
    #[test]
    fn malformed_words_after_a_numeric_difference_still_refuse() {
        let raw: Vec<_> = [0u32, 0x3f800001]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert!(scan(&raw, &[1.0, 1.0], Encoding::WidenedBf16).is_err());
        assert!(scan(&[0, 0], &[f32::NAN], Encoding::Bf16).is_err());
        assert!(scan(&[0], &[0.0], Encoding::Bf16).is_err());
    }
}
