//! Explicit CURRENT owned2B choice, separate from original Autotuner selections.
//! Code equality is metadata; actual leaf/connected completion still qualifies math.
use super::{GdnArithmetic, GdnCacheSlot, GdnFamilyIdentity, GdnStage, VllmGdnPhase, assets};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen2bGdnArithmetic {
    OwnedSolve4Recompute3,
}

/// Actual retired original reference choices; neither selects an owned kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen2bOriginalGdnReference {
    ObservedSolve4Recompute4,
    PlainSolve2Recompute3,
}
impl Qwen2bGdnArithmetic {
    pub fn solve_ptx_sha256(self) -> &'static str {
        "fcae68be445ab7ec2b7dcc127d3d4c0851340ce3d237be00e909b4391e67c135"
    }
    pub fn recompute_ptx_sha256(self) -> &'static str {
        "d9c9fff7acb0fc04ad9b7ef177f8889ead5c5ce2df8b5d5a81a85958ae191763"
    }
    pub(crate) fn require_owned(self, phase: VllmGdnPhase) -> Result<()> {
        if !matches!(phase, VllmGdnPhase::InitialOne | VllmGdnPhase::Initial64) {
            return Err(Error::Unsupported(
                "owned2B explicit initial1/64 tuple only".into(),
            ));
        }
        let one = phase == VllmGdnPhase::InitialOne;
        let expected = [
            (
                "7e9600b09ee757c1d786657d7edfa5b3629cb5b1b00480de1fdb5a761e4a0a5c",
                128,
                [if one { 1 } else { 8 }, 24, 1],
                0,
            ),
            (
                if one {
                    "ca887254afcfa3b02b6a604ac3670ef4cca3cc72e05826c1fc9e1256cd85909e"
                } else {
                    "26138ddbd8249c06511453384e6877aef58ad39d23eb9d8733b3eb8369e2467a"
                },
                128,
                [if one { 1 } else { 4 }, 32, 1],
                0,
            ),
            (
                "b4c97f73509b35ae7f3281b5621b44ca354e08b65a496575e95670537a0783ab",
                64,
                [1, 16, 1],
                8,
            ),
            (
                "e8ba065941c5dbf148d98ee28eba369a987d176c264ce8e05e9cb697498bad44",
                128,
                [1, 16, 1],
                8192,
            ),
            (self.solve_ptx_sha256(), 128, [1, 16, 1], 10240),
            (self.recompute_ptx_sha256(), 256, [1, 16, 1], 32768),
            (
                "e191a9ba73a5a661175f81e823457c920c7792166e3f37e80a74fec1fcca1d1c",
                128,
                [2, 16, 1],
                49412,
            ),
            (
                "284a22e9904079006d593e67082d6d898cd35becf07e7aebf8086ec9c070df0e",
                128,
                [4, 1, 16],
                12288,
            ),
            (
                if one {
                    "a97d127b124f14d226c2b24cff907e8c7aa260545e41743f84581acf6d510bbb"
                } else {
                    "3900a7b7b673b1b398d39a8bb7d3e4ab4d87ea8798e9a2214be7c5b9d29beeea"
                },
                32,
                [if one { 16 } else { 256 }, 1, 1],
                0,
            ),
        ];
        if phase.stages().len() != expected.len() {
            return Err(Error::State("owned2B complete nine-stage tuple".into()));
        }
        for (&stage, (sha, threads, grid, shared)) in phase.stages().iter().zip(expected) {
            let actual = assets::select_qwen2b(phase, stage)?;
            if actual.ptx_sha256 != sha
                || actual.threads != threads
                || actual.grid != grid
                || actual.shared != shared
            {
                return Err(Error::Unsupported(
                    "owned2B complete source stage tuple changed".into(),
                ));
            }
        }
        // Family equality also includes complete immutable source, binding and
        // copy catalogues. Every StageArithmetic is factory-checked against it.
        for (stage, source, sha, threads, shared) in [
            (
                GdnStage::Solve,
                include_str!("solve.cu"),
                self.solve_ptx_sha256(),
                128,
                10240,
            ),
            (
                GdnStage::Recompute,
                include_str!("qwen2b_recompute.cu"),
                self.recompute_ptx_sha256(),
                256,
                32768,
            ),
        ] {
            let actual = assets::select_qwen2b(phase, stage)?;
            if actual.source != source
                || actual.ptx_sha256 != sha
                || actual.threads != threads
                || actual.shared != shared
                || actual.grid != [1, 16, 1]
            {
                return Err(Error::Unsupported(
                    "owned2B solve4/recompute3 tuple changed".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn require_prepared(
        self,
        actual: &GdnArithmetic,
        phase: VllmGdnPhase,
    ) -> Result<()> {
        self.require_owned(phase)?;
        GdnFamilyIdentity::original2b().require_same(actual.family())?;
        actual.require_instance(
            phase,
            if phase == VllmGdnPhase::InitialOne {
                GdnCacheSlot::One
            } else {
                GdnCacheSlot::Four
            },
            true,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_owned_tuple_does_not_admit_other_phase_or_original_choice() {
        let owned = Qwen2bGdnArithmetic::OwnedSolve4Recompute3;
        for phase in [VllmGdnPhase::InitialOne, VllmGdnPhase::Initial64] {
            owned.require_owned(phase).unwrap();
            for stage in phase.stages() {
                let spec = super::super::kernel_qwen2b_component(phase, *stage).unwrap();
                spec.arithmetic.require_stage(phase, *stage).unwrap();
                GdnFamilyIdentity::original2b()
                    .require_same(spec.arithmetic.family())
                    .unwrap();
            }
        }
        for phase in [
            VllmGdnPhase::Initial47,
            VllmGdnPhase::ContinuedTwo,
            VllmGdnPhase::CachedDecode,
        ] {
            assert!(owned.require_owned(phase).is_err());
        }
        assert_ne!(
            owned.solve_ptx_sha256(),
            "8201e61290e057badac428ff441391ceaf45c075c67daee4ffe24e816f86bb02"
        );
        assert_ne!(
            owned.recompute_ptx_sha256(),
            "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a"
        );
        assert_ne!(
            Qwen2bOriginalGdnReference::ObservedSolve4Recompute4,
            Qwen2bOriginalGdnReference::PlainSolve2Recompute3
        );
    }
}
