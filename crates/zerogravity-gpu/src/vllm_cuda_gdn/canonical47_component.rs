//! Exact CURRENT owned source tuple; never an original-selection/acceptance permit.
//! Cumsum/KKT/H/Output differ from the original OBS code selections. Numerical
//! compatibility requires real connected output comparisons, never name equality.
use super::{GdnArithmetic, GdnCacheSlot, GdnFamilyIdentity, VllmGdnPhase, assets};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonical47GdnArithmetic {
    OwnedCanonical47Solve4,
    /// Original PLAIN solve2 trajectory has a separate finite solve leaf only.
    OriginalPlainSolve2,
}
impl Canonical47GdnArithmetic {
    pub(crate) fn require_owned(self) -> Result<()> {
        if self != Self::OwnedCanonical47Solve4 {
            return Err(Error::Unsupported(
                "connected47 original PLAIN tuple is not implemented".into(),
            ));
        }
        let expected = [
            (
                "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
                128,
                [6, 24, 1],
                0,
            ),
            (
                "22dd4697695aa897b26a5ed794164c6dcc204ef11429d5d37786b240d12e6670",
                128,
                [3, 32, 1],
                0,
            ),
            (
                "b4c97f73509b35ae7f3281b5621b44ca354e08b65a496575e95670537a0783ab",
                64,
                [1, 16, 1],
                8,
            ),
            (
                "d65e28bad710a3e78a38583d14ec26a8c838eb75e8becc0da40a9a6c17da7639",
                256,
                [1, 16, 1],
                16384,
            ),
            (
                "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135",
                128,
                [1, 16, 1],
                10240,
            ),
            (
                "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a",
                256,
                [1, 16, 1],
                40960,
            ),
            (
                "2c05861ae7051b9f267b683d7a411686edb18d8ab34598c6f9dac9c8e39dcb38",
                128,
                [4, 16, 1],
                41220,
            ),
            (
                "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                128,
                [4, 1, 16],
                12288,
            ),
            (
                "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea",
                32,
                [188, 1, 1],
                0,
            ),
        ];
        let stages = VllmGdnPhase::Initial47.stages();
        if stages.len() != expected.len() {
            return Err(Error::State("connected47 complete nine-stage tuple".into()));
        }
        for (&stage, (sha, threads, grid, shared)) in stages.iter().zip(expected) {
            let actual = assets::select(VllmGdnPhase::Initial47, stage)?;
            if actual.ptx_sha256 != sha
                || actual.threads != threads
                || actual.grid != grid
                || actual.shared != shared
            {
                return Err(Error::Unsupported(
                    "connected47 selected owned GDN tuple changed".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn require_prepared(self, actual: &GdnArithmetic) -> Result<()> {
        self.require_owned()?;
        GdnFamilyIdentity::original08().require_same(actual.family())?;
        actual.require_instance(VllmGdnPhase::Initial47, GdnCacheSlot::One, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn connected47_requires_explicit_whole_current_owned_tuple() {
        assert!(
            Canonical47GdnArithmetic::OwnedCanonical47Solve4
                .require_owned()
                .is_ok()
        );
        assert!(
            Canonical47GdnArithmetic::OriginalPlainSolve2
                .require_owned()
                .is_err()
        );
        let solve = assets::select(VllmGdnPhase::Initial47, super::super::GdnStage::Solve).unwrap();
        let recompute =
            assets::select(VllmGdnPhase::Initial47, super::super::GdnStage::Recompute).unwrap();
        assert_eq!(solve.threads, 128);
        assert_eq!(recompute.shared, 40960);
        assert_ne!(
            solve.ptx_sha256,
            "8201e61290e057badac428ff441391ceaf45c075c67daee4ffe24e816f86bb02"
        );
        assert_ne!(recompute.shared, 32768);
    }
}

/// Defined CURRENT owned stage storage, visible only through trusted readback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonical47GdnIntermediate {
    GRaw,
    GCumulative,
    Kkt,
    Inverse,
    W,
    U,
    FinalState,
    Core,
}
impl Canonical47GdnIntermediate {
    pub(crate) const ALL: [Self; 8] = [
        Self::GRaw,
        Self::GCumulative,
        Self::Kkt,
        Self::Inverse,
        Self::W,
        Self::U,
        Self::FinalState,
        Self::Core,
    ];
    pub(crate) fn role(self) -> super::GdnBuffer {
        use super::GdnBuffer as B;
        match self {
            Self::GRaw => B::GRaw,
            Self::GCumulative => B::GCumulative,
            Self::Kkt => B::Kkt,
            Self::Inverse => B::Inverse,
            Self::W => B::W,
            Self::U => B::U,
            Self::FinalState => B::FinalState,
            Self::Core => B::Core,
        }
    }
    /// Dense physical axes, outer batch is fixed1. FinalState is H/K/V.
    pub fn shape(self) -> &'static [usize] {
        match self {
            Self::GRaw | Self::GCumulative => &[47, 16],
            Self::Kkt | Self::Inverse => &[47, 16, 64],
            Self::W | Self::U | Self::Core => &[47, 16, 128],
            Self::FinalState => &[16, 128, 128],
        }
    }
    /// Two-byte storage is BF16; four-byte storage is F32.
    pub fn word_bytes(self) -> usize {
        match self {
            Self::Inverse | Self::W | Self::U | Self::Core => 2,
            _ => 4,
        }
    }
    pub fn bytes(self) -> usize {
        self.role().bytes(VllmGdnPhase::Initial47)
    }
}

#[cfg(test)]
mod diagnostic_tests {
    use super::*;
    #[test]
    fn exact47_diagnostic_extents_stay_bounded_and_do_not_pad_to64() {
        let mut total = 0;
        for key in Canonical47GdnIntermediate::ALL {
            assert_eq!(
                key.shape().iter().product::<usize>() * key.word_bytes(),
                key.bytes()
            );
            total += key.bytes();
        }
        assert_eq!(
            Canonical47GdnIntermediate::Inverse.bytes(),
            47 * 16 * 64 * 2
        );
        assert_eq!(Canonical47GdnIntermediate::Kkt.bytes(), 47 * 16 * 64 * 4);
        assert_eq!(total, 1_920_896);
        assert_eq!(84_391_204 + total, 86_312_100);
        assert!(84_391_204 + total < 128 * 1024 * 1024);
    }
}
