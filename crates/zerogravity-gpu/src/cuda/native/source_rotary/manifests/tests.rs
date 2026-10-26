impl RotaryOutput {
    pub const fn prefix_bytes(self) -> usize {
        match self {
            Self::Range32 => 128,
            Self::Scaled32 => 128,
            Self::PowBase => 8,
            Self::Pow32 => 128,
            Self::Reciprocal32 => 128,
            Self::InvFreq32 => 128,
            Self::Range1048576 => 268,
            Self::Frequencies => 8576,
            Self::Cos => 8576,
            Self::Sin => 8576,
            Self::Cat => 17152,
            Self::Cache => 8576,
        }
    }
}

use super::*;
#[test]
fn all_source_factories_have_exact_complete_abis_and_full_extents() {
    let all = launches();
    assert_eq!(all.len(), 12);
    assert_eq!(required_images().len(), 10);
    assert_eq!(
        RotaryOutput::ALL.iter().map(|r| r.bytes()).sum::<usize>(),
        809501320
    );
    for m in all {
        assert_eq!(m.manifest.binary_sm, 86);
        assert!(
            m.manifest
                .grid
                .iter()
                .chain(&m.manifest.block)
                .all(|&n| n > 0)
        );
        assert_eq!(m.roles.len(), m.manifest.bindings.len());
        for (r, b) in m.roles.iter().zip(m.manifest.bindings.iter()) {
            assert_eq!(r.bytes() as u64, b.bytes);
        }
    }
}
#[test]
fn full_cache_and_each_original_defined_prefix_stay_distinct() {
    assert_eq!(RotaryOutput::Cache.bytes(), 134217728);
    assert_eq!(RotaryOutput::Cache.prefix_bytes(), 8576);
    assert_eq!(
        RotaryOutput::ALL
            .iter()
            .map(|r| r.prefix_bytes())
            .sum::<usize>(),
        52372
    );
    assert_eq!(RotaryOutput::Cat.bytes(), 268435456);
    assert_eq!(RotaryOutput::Range1048576.bytes(), 4194304);
}
