//! Exact retained original source-rotary headers; full SHA seals before parsing.
pub(super) fn matches(bytes: &[u8]) -> bool {
    HEADERS
        .iter()
        .any(|(n, h)| bytes.len() == *n && bytes.get(..64) == Some(h.as_slice()))
}
const HEADERS: &[(usize, [u8; 64])] = &[
    (
        638536,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 104, 189, 9, 0, 0, 0, 0, 0, 40, 126, 9, 0, 0, 0, 0, 0, 86, 5, 86, 0,
            64, 0, 56, 0, 4, 0, 64, 0, 253, 0, 1, 0,
        ],
    ),
    (
        4066320,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 48, 11, 62, 0, 0, 0, 0, 0, 240, 137, 61, 0, 0, 0, 0, 0, 86, 5, 86, 0,
            64, 0, 56, 0, 4, 0, 64, 0, 5, 2, 1, 0,
        ],
    ),
    (
        1942400,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 160, 162, 29, 0, 0, 0, 0, 0, 32, 49, 29, 0, 0, 0, 0, 0, 86, 5, 86, 0,
            64, 0, 56, 0, 4, 0, 64, 0, 198, 1, 1, 0,
        ],
    ),
    (
        16041664,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 224, 197, 244, 0, 0, 0, 0, 0, 160, 111, 243, 0, 0, 0, 0, 0, 86, 5,
            86, 0, 64, 0, 56, 0, 4, 0, 64, 0, 89, 5, 1, 0,
        ],
    ),
    (
        4246208,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 224, 201, 64, 0, 0, 0, 0, 0, 32, 41, 64, 0, 0, 0, 0, 0, 86, 5, 86, 0,
            64, 0, 56, 0, 4, 0, 64, 0, 131, 2, 1, 0,
        ],
    ),
    (
        850848,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 192, 250, 12, 0, 0, 0, 0, 0, 128, 220, 12, 0, 0, 0, 0, 0, 86, 5, 86,
            0, 64, 0, 56, 0, 4, 0, 64, 0, 121, 0, 1, 0,
        ],
    ),
    (
        848032,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 192, 239, 12, 0, 0, 0, 0, 0, 128, 209, 12, 0, 0, 0, 0, 0, 86, 5, 86,
            0, 64, 0, 56, 0, 4, 0, 64, 0, 121, 0, 1, 0,
        ],
    ),
    (
        2103392,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 128, 23, 32, 0, 0, 0, 0, 0, 128, 76, 31, 0, 0, 0, 0, 0, 86, 5, 86, 0,
            64, 0, 56, 0, 4, 0, 64, 0, 44, 3, 1, 0,
        ],
    ),
    (
        4557464,
        [
            127, 69, 76, 70, 2, 1, 1, 51, 7, 0, 0, 0, 0, 0, 0, 0, 2, 0, 190, 0, 129, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 184, 137, 69, 0, 0, 0, 0, 0, 184, 167, 68, 0, 0, 0, 0, 0, 86, 5, 86,
            0, 64, 0, 56, 0, 4, 0, 64, 0, 136, 3, 1, 0,
        ],
    ),
];
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nine_exact_headers_keep_every_remaining_elf_guard() {
        assert_eq!(HEADERS.len(), 9);
        for (n, h) in HEADERS {
            // Structural fixture only; actual captured admission separately
            // requires the complete original image SHA before this parser.
            let mut b = vec![0u8; *n];
            b[..64].copy_from_slice(h);
            let table = u64::from_le_bytes(h[40..48].try_into().unwrap()) as usize;
            let names = u16::from_le_bytes(h[62..64].try_into().unwrap()) as usize;
            let at = table + names * 64;
            b[at + 4..at + 8].copy_from_slice(&3u32.to_le_bytes());
            b[at + 24..at + 32].copy_from_slice(&128u64.to_le_bytes());
            b[at + 32..at + 40].copy_from_slice(&1u64.to_le_bytes());
            assert!(matches(&b));
            assert_eq!(super::super::validate(&b).unwrap(), 86);
            for offset in 0..64 {
                let mut bad = b.clone();
                bad[offset] ^= 1;
                assert!(!matches(&bad), "header{n}/byte{offset}");
                if offset != 20 {
                    assert!(
                        super::super::validate(&bad).is_err(),
                        "remaining header{n}/byte{offset}"
                    );
                }
            }
            assert!(!matches(&b[..b.len() - 1]));
            assert!(super::super::validate(&b[..b.len() - 1]).is_err());
            let mut extra = b.clone();
            extra.push(0);
            assert!(!matches(&extra));
            assert!(super::super::validate(&extra).is_err());
            b[at + 24..at + 32].copy_from_slice(&u64::MAX.to_le_bytes());
            assert!(
                super::super::validate(&b).is_err(),
                "original section range guard retained"
            );
        }
    }
}
