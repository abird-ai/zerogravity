//! Selected full-image identity comes from the existing compiled native registry.
use super::native::CubinImageKind as K;
pub(super) const MODEL: [K; 14] = [
    K::AmpereBf16Gemm128x64Sm80,
    K::AmpereBf16Gemm64x64Stages64x5Sm80,
    K::Bf16SplitKReduceSm86,
    K::Canonical47CutlassBf16Sm80,
    K::AttentionAmpere256x128Sm80,
    K::AttentionQkSm86,
    K::AttentionKvSm86,
    K::AttentionFaCausalCombineSm80,
    K::AttentionSigmoidSm86,
    K::AttentionMultiplySm86,
    K::Bf16Gemv2TSm86,
    K::CutlassWmmaSm80,
    K::Bf16GemvxSm86,
    K::AttentionFaSplitSm80,
];
pub(super) const ROTARY: [K; 9] = [
    K::RotaryArangeSm86,
    K::RotaryDivideSm86,
    K::RotaryFillSm86,
    K::RotaryPowSm86,
    K::RotaryReciprocalSm86,
    K::RotaryCosSm86,
    K::RotarySinSm86,
    K::RotaryCatSm86,
    K::RotaryBf16CastSm86,
];
pub(super) fn image_cap(kind: K) -> usize {
    use super::MIB;
    match kind {
        K::AttentionQkSm86 | K::AttentionKvSm86 => 2 * MIB,
        K::AttentionFaCausalCombineSm80 | K::AttentionSigmoidSm86 | K::AttentionMultiplySm86 => {
            4 * MIB
        }
        K::AttentionFaSplitSm80 => 8 * MIB,
        _ => 16 * MIB,
    }
}
pub(super) const ACTIVATION_BYTES: usize = 901_664;
pub(super) const ACTIVATION_SHA: &str =
    "7ec9414283b09942de90011eb5bdbaeb63b352cd0dc500f2c7ba3854f660fadf";
