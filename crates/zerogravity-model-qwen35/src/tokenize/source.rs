//! Exact source identities sharing the same NFC/Split/ByteLevel/BPE semantics.
//! Added-token inventories remain source data, not model-size capabilities.

use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Source {
    pub bytes: u64,
    pub sha256: &'static str,
    pub added_tokens: u64,
    pub max_added_bytes: u64,
}

pub(super) const BASE: Source = Source {
    bytes: 12_807_982,
    sha256: "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    added_tokens: 26,
    max_added_bytes: 20,
};

const SOURCES: [Source; 2] = [
    BASE,
    Source {
        bytes: 12_809_320,
        sha256: "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        added_tokens: 33,
        max_added_bytes: 21,
    },
];

pub(super) fn recognize(bytes: &[u8]) -> Option<&'static Source> {
    let len = bytes.len() as u64;
    if !SOURCES.iter().any(|source| source.bytes == len) {
        return None;
    }
    let digest = format!("{:x}", Sha256::digest(bytes));
    SOURCES
        .iter()
        .find(|source| source.bytes == len && source.sha256 == digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_inventories_preserve_shared_owners_and_default_pricing() {
        use super::super::bounds;
        for stacks in [1, 2, 4, 16, 64] {
            let base = bounds::plan(stacks).unwrap();
            assert_eq!(base, bounds::plan_for_source(stacks, &BASE).unwrap());
            let extended = bounds::plan_for_source(stacks, &SOURCES[1]).unwrap();
            assert!(extended.bpe_retained_bytes > base.bpe_retained_bytes);
            assert!(extended.bpe_constructor_static_bytes > base.bpe_constructor_static_bytes);
            assert!(extended.bpe_constructor_peak_bytes > base.bpe_constructor_peak_bytes);
            assert_eq!(extended.bootstrap_bytes, base.bootstrap_bytes);
            assert_eq!(extended.bootstrap_peak_bytes, base.bootstrap_peak_bytes);
            assert_eq!(
                extended.byte_level_retained_bytes,
                base.byte_level_retained_bytes
            );
            assert_eq!(extended.split_retained_bytes, base.split_retained_bytes);
            assert_eq!(
                extended.request_cache_growth_bytes,
                base.request_cache_growth_bytes
            );
            assert_eq!(extended.serialized_vm_bytes, base.serialized_vm_bytes);
        }
    }

    #[test]
    fn source_identity_cannot_be_inferred_from_length() {
        for source in SOURCES {
            assert!(recognize(&vec![b' '; source.bytes as usize]).is_none());
        }
        assert!(recognize(b"{}").is_none());
    }
}
