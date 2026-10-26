//! Bounded readback comparison only; never changes numerical acceptance.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Difference {
    actual_bytes: usize,
    expected_bytes: usize,
    different_bytes: usize,
    different_words: usize,
    first_byte: Option<usize>,
    // Word index, actual BF16 bits, independent N1 BF16 bits.
    first_word: Option<(usize, u16, u16)>,
}
impl Difference {
    pub(super) fn exact(&self) -> bool {
        self.actual_bytes == self.expected_bytes && self.different_bytes == 0
    }
}
pub(super) fn difference(actual: &[u8], expected: &[u8]) -> Difference {
    let first_byte = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .or_else(|| (actual.len() != expected.len()).then_some(actual.len().min(expected.len())));
    let first_word = first_byte.and_then(|i| {
        let word = i / 2;
        let a = actual.get(word * 2..word * 2 + 2)?;
        let b = expected.get(word * 2..word * 2 + 2)?;
        Some((
            word,
            u16::from_le_bytes([a[0], a[1]]),
            u16::from_le_bytes([b[0], b[1]]),
        ))
    });
    Difference {
        actual_bytes: actual.len(),
        expected_bytes: expected.len(),
        different_bytes: actual.iter().zip(expected).filter(|(a, b)| a != b).count()
            + actual.len().abs_diff(expected.len()),
        different_words: actual
            .chunks(2)
            .zip(expected.chunks(2))
            .filter(|(a, b)| a != b)
            .count()
            + actual
                .len()
                .div_ceil(2)
                .abs_diff(expected.len().div_ceil(2)),
        first_byte,
        first_word,
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_and_empty() {
        assert!(difference(&[], &[]).exact());
        assert!(difference(&[0, 128, 255, 127], &[0, 128, 255, 127]).exact());
    }
    #[test]
    fn every_difference_and_first_complete_word() {
        let d = difference(&[0, 128, 1, 63, 3, 64], &[0, 128, 2, 63, 4, 64]);
        assert!(!d.exact());
        assert_eq!(d.different_bytes, 2);
        assert_eq!(d.different_words, 2);
        assert_eq!(d.first_byte, Some(2));
        assert_eq!(d.first_word, Some((1, 0x3f01, 0x3f02)));
    }
    #[test]
    fn length_and_odd_tail_cannot_appear_exact() {
        let d = difference(&[1, 2, 3], &[1, 2]);
        assert!(!d.exact());
        assert_eq!(d.first_byte, Some(2));
        assert_eq!(d.first_word, None);
        assert_eq!(d.different_words, 1);
        assert_eq!(difference(&[1], &[2]).different_bytes, 1);
        assert_eq!(difference(&[1], &[2]).first_word, None);
    }
}
