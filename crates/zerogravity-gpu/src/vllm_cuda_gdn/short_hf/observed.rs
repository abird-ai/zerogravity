//! Actual original per-cell source tuples; whole-model numerics remain to be executed.
use super::*;

#[path = "qwen08_47/catalog.rs"]
mod qwen08_47;
#[path = "qwen08_57/catalog.rs"]
mod qwen08_57;
#[path = "qwen08_62/catalog.rs"]
mod qwen08_62;
#[path = "qwen08_67/catalog.rs"]
mod qwen08_67;
#[path = "qwen08_72/catalog.rs"]
mod qwen08_72;
#[path = "qwen08_cached/catalog.rs"]
mod qwen08_cached;
#[path = "qwen2b_47/catalog.rs"]
mod qwen2b_47;
#[path = "qwen2b_57/catalog.rs"]
mod qwen2b_57;
#[path = "qwen2b_62/catalog.rs"]
mod qwen2b_62;
#[path = "qwen2b_67/catalog.rs"]
mod qwen2b_67;
#[path = "qwen2b_72/catalog.rs"]
mod qwen2b_72;
#[path = "qwen2b_cached/catalog.rs"]
mod qwen2b_cached;

pub(super) fn select(
    profile: FullModelProfile,
    phase: VllmGdnPhase,
    stage: GdnStage,
) -> Result<assets::Asset> {
    use zerogravity_core::accelerator::program::ShortHfWorkload::*;
    match (profile, phase) {
        (FullModelProfile::Qwen08, VllmGdnPhase::ShortHfPrefill(Prompt47)) => {
            qwen08_47::select(stage)
        }
        (FullModelProfile::Qwen08, VllmGdnPhase::ShortHfPrefill(Prompt57)) => {
            qwen08_57::select(stage)
        }
        (FullModelProfile::Qwen08, VllmGdnPhase::ShortHfPrefill(Prompt62)) => {
            qwen08_62::select(stage)
        }
        (FullModelProfile::Qwen08, VllmGdnPhase::ShortHfPrefill(Prompt67)) => {
            qwen08_67::select(stage)
        }
        (FullModelProfile::Qwen08, VllmGdnPhase::ShortHfPrefill(Prompt72)) => {
            qwen08_72::select(stage)
        }
        (FullModelProfile::Qwen08, VllmGdnPhase::CachedDecode) => qwen08_cached::select(stage),
        (FullModelProfile::Qwen2B, VllmGdnPhase::ShortHfPrefill(Prompt47)) => {
            qwen2b_47::select(stage)
        }
        (FullModelProfile::Qwen2B, VllmGdnPhase::ShortHfPrefill(Prompt57)) => {
            qwen2b_57::select(stage)
        }
        (FullModelProfile::Qwen2B, VllmGdnPhase::ShortHfPrefill(Prompt62)) => {
            qwen2b_62::select(stage)
        }
        (FullModelProfile::Qwen2B, VllmGdnPhase::ShortHfPrefill(Prompt67)) => {
            qwen2b_67::select(stage)
        }
        (FullModelProfile::Qwen2B, VllmGdnPhase::ShortHfPrefill(Prompt72)) => {
            qwen2b_72::select(stage)
        }
        (FullModelProfile::Qwen2B, VllmGdnPhase::CachedDecode) => qwen2b_cached::select(stage),
        _ => Err(Error::Unsupported(
            "actual original short-HF source tuple not present".into(),
        )),
    }
}

pub(super) const CATALOGUE: &[&str] = &[
    include_str!("qwen08_47/catalog.rs"),
    include_str!("qwen08_47/conv.cu"),
    include_str!("qwen08_47/prep.cu"),
    include_str!("qwen08_47/cumsum.cu"),
    include_str!("qwen08_47/kkt.cu"),
    include_str!("qwen08_47/solve.cu"),
    include_str!("qwen08_47/recompute.cu"),
    include_str!("qwen08_47/h.cu"),
    include_str!("qwen08_47/output.cu"),
    include_str!("qwen08_47/gated.cu"),
    include_str!("qwen08_57/catalog.rs"),
    include_str!("qwen08_57/conv.cu"),
    include_str!("qwen08_57/prep.cu"),
    include_str!("qwen08_57/cumsum.cu"),
    include_str!("qwen08_57/kkt.cu"),
    include_str!("qwen08_57/solve.cu"),
    include_str!("qwen08_57/recompute.cu"),
    include_str!("qwen08_57/h.cu"),
    include_str!("qwen08_57/output.cu"),
    include_str!("qwen08_57/gated.cu"),
    include_str!("qwen08_62/catalog.rs"),
    include_str!("qwen08_62/conv.cu"),
    include_str!("qwen08_62/prep.cu"),
    include_str!("qwen08_62/cumsum.cu"),
    include_str!("qwen08_62/kkt.cu"),
    include_str!("qwen08_62/solve.cu"),
    include_str!("qwen08_62/recompute.cu"),
    include_str!("qwen08_62/h.cu"),
    include_str!("qwen08_62/output.cu"),
    include_str!("qwen08_62/gated.cu"),
    include_str!("qwen08_67/catalog.rs"),
    include_str!("qwen08_67/conv.cu"),
    include_str!("qwen08_67/prep.cu"),
    include_str!("qwen08_67/cumsum.cu"),
    include_str!("qwen08_67/kkt.cu"),
    include_str!("qwen08_67/solve.cu"),
    include_str!("qwen08_67/recompute.cu"),
    include_str!("qwen08_67/h.cu"),
    include_str!("qwen08_67/output.cu"),
    include_str!("qwen08_67/gated.cu"),
    include_str!("qwen08_72/catalog.rs"),
    include_str!("qwen08_72/conv.cu"),
    include_str!("qwen08_72/prep.cu"),
    include_str!("qwen08_72/cumsum.cu"),
    include_str!("qwen08_72/kkt.cu"),
    include_str!("qwen08_72/solve.cu"),
    include_str!("qwen08_72/recompute.cu"),
    include_str!("qwen08_72/h.cu"),
    include_str!("qwen08_72/output.cu"),
    include_str!("qwen08_72/gated.cu"),
    include_str!("qwen08_cached/catalog.rs"),
    include_str!("qwen08_cached/conv.cu"),
    include_str!("qwen08_cached/decode.cu"),
    include_str!("qwen08_cached/gated.cu"),
    include_str!("qwen2b_47/catalog.rs"),
    include_str!("qwen2b_47/conv.cu"),
    include_str!("qwen2b_47/prep.cu"),
    include_str!("qwen2b_47/cumsum.cu"),
    include_str!("qwen2b_47/kkt.cu"),
    include_str!("qwen2b_47/solve.cu"),
    include_str!("qwen2b_47/recompute.cu"),
    include_str!("qwen2b_47/h.cu"),
    include_str!("qwen2b_47/output.cu"),
    include_str!("qwen2b_47/gated.cu"),
    include_str!("qwen2b_57/catalog.rs"),
    include_str!("qwen2b_57/conv.cu"),
    include_str!("qwen2b_57/prep.cu"),
    include_str!("qwen2b_57/cumsum.cu"),
    include_str!("qwen2b_57/kkt.cu"),
    include_str!("qwen2b_57/solve.cu"),
    include_str!("qwen2b_57/recompute.cu"),
    include_str!("qwen2b_57/h.cu"),
    include_str!("qwen2b_57/output.cu"),
    include_str!("qwen2b_57/gated.cu"),
    include_str!("qwen2b_62/catalog.rs"),
    include_str!("qwen2b_62/conv.cu"),
    include_str!("qwen2b_62/prep.cu"),
    include_str!("qwen2b_62/cumsum.cu"),
    include_str!("qwen2b_62/kkt.cu"),
    include_str!("qwen2b_62/solve.cu"),
    include_str!("qwen2b_62/recompute.cu"),
    include_str!("qwen2b_62/h.cu"),
    include_str!("qwen2b_62/output.cu"),
    include_str!("qwen2b_62/gated.cu"),
    include_str!("qwen2b_67/catalog.rs"),
    include_str!("qwen2b_67/conv.cu"),
    include_str!("qwen2b_67/prep.cu"),
    include_str!("qwen2b_67/cumsum.cu"),
    include_str!("qwen2b_67/kkt.cu"),
    include_str!("qwen2b_67/solve.cu"),
    include_str!("qwen2b_67/recompute.cu"),
    include_str!("qwen2b_67/h.cu"),
    include_str!("qwen2b_67/output.cu"),
    include_str!("qwen2b_67/gated.cu"),
    include_str!("qwen2b_72/catalog.rs"),
    include_str!("qwen2b_72/conv.cu"),
    include_str!("qwen2b_72/prep.cu"),
    include_str!("qwen2b_72/cumsum.cu"),
    include_str!("qwen2b_72/kkt.cu"),
    include_str!("qwen2b_72/solve.cu"),
    include_str!("qwen2b_72/recompute.cu"),
    include_str!("qwen2b_72/h.cu"),
    include_str!("qwen2b_72/output.cu"),
    include_str!("qwen2b_72/gated.cu"),
    include_str!("qwen2b_cached/catalog.rs"),
    include_str!("qwen2b_cached/conv.cu"),
    include_str!("qwen2b_cached/decode.cu"),
    include_str!("qwen2b_cached/gated.cu"),
];
