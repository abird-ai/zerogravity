//! Closed whole-image catalogue; neither source discovery nor kind grants an ABI permit.
use super::super::cubin_manifests::ProjectionImage;
use zerogravity_core::{Result, accelerator::program::FullModelProfile};

/// Actual union across five prompt spans and seven natural continuations.
/// Activation is separately admitted by Mlp47QualificationInputs.
pub fn short_hf_required_images(profile: FullModelProfile) -> Result<Vec<ProjectionImage>> {
    let images = match profile {
        FullModelProfile::Qwen08 => vec![
            ProjectionImage::AmpereBf16Gemm128x64Sm80,
            ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
            ProjectionImage::AttentionAmpere256x128Sm80,
            ProjectionImage::AttentionFaCausalCombineSm80,
            ProjectionImage::AttentionFaSplitSm80,
            ProjectionImage::AttentionKvSm86,
            ProjectionImage::AttentionMultiplySm86,
            ProjectionImage::AttentionQkSm86,
            ProjectionImage::AttentionSigmoidSm86,
            ProjectionImage::Bf16Gemv2TSm86,
            ProjectionImage::Bf16GemvxSm86,
            ProjectionImage::Bf16SplitKReduceSm86,
            ProjectionImage::Canonical47CutlassBf16Sm80,
            ProjectionImage::CutlassWmmaSm80,
            ProjectionImage::ShortHfFaCausalCombineSm80,
            ProjectionImage::ShortHfFaSplitCombineSm80,
        ],
        FullModelProfile::Qwen2B => vec![
            ProjectionImage::AmpereBf16Gemm128x64Sm80,
            ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
            ProjectionImage::AttentionAmpere256x128Sm80,
            ProjectionImage::AttentionFaCausalCombineSm80,
            ProjectionImage::AttentionFaSplitSm80,
            ProjectionImage::AttentionKvSm86,
            ProjectionImage::AttentionMultiplySm86,
            ProjectionImage::AttentionQkSm86,
            ProjectionImage::AttentionSigmoidSm86,
            ProjectionImage::Bf16Gemv2TSm86,
            ProjectionImage::Bf16GemvxSm86,
            ProjectionImage::Bf16SplitKReduceSm86,
            ProjectionImage::Canonical47CutlassBf16Sm80,
            ProjectionImage::CutlassWmmaSm80,
            ProjectionImage::ShortHfCutlassBf16Sm80,
            ProjectionImage::ShortHfFaCausalCombineSm80,
            ProjectionImage::ShortHfFaSplitCombineSm80,
        ],
    };
    Ok(images)
}

/// Selected source phase union; N1 callers retain the original exact image set.
/// The cached2B addition is a distinct full capture, never a renamed N1 image.
pub fn short_hf_required_images_for_cohort(
    profile: FullModelProfile,
    cohort_b4: bool,
) -> Result<Vec<ProjectionImage>> {
    let mut images = short_hf_required_images(profile)?;
    if cohort_b4 && profile == FullModelProfile::Qwen2B {
        images.push(ProjectionImage::Cohort2BAmpere64SlicedSm80);
    }
    Ok(images)
}

/// Declared code/global-section alignment upper + original opaque64MiB owner.
/// On-chip shared sections and host image bytes are separate resources.
/// Actual module allocation/whole-unit physical fit requires ROOT execution.
pub fn short_hf_native_provider_bytes(profile: FullModelProfile) -> usize {
    let sections = match profile {
        FullModelProfile::Qwen08 => 63_985_828usize,
        FullModelProfile::Qwen2B => 67_743_743usize,
    };
    sections.div_ceil(1024 * 1024) * (1024 * 1024) + 64 * 1024 * 1024
}

/// Extra exact ed95 image:288 SHF_ALLOC sections,5,622,112B. Sum each
/// size+max(256,section_alignment)-1 =5,695,552B; maximum actual alignment128.
/// The original finite opaque64MiB policy remains unchanged, shared by the set.
pub fn short_hf_native_provider_bytes_for_cohort(
    profile: FullModelProfile,
    cohort_b4: bool,
) -> usize {
    if cohort_b4 && profile == FullModelProfile::Qwen2B {
        (67_743_743usize + 5_695_552).div_ceil(1024 * 1024) * (1024 * 1024) + 64 * 1024 * 1024
    } else {
        short_hf_native_provider_bytes(profile)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached2b_adds_exact_whole_image_and_prices_its_complete_section_upper() {
        let old = short_hf_required_images(FullModelProfile::Qwen2B).unwrap();
        let cached = short_hf_required_images_for_cohort(FullModelProfile::Qwen2B, true).unwrap();
        assert_eq!(cached.len(), 18);
        assert_eq!(
            cached
                .iter()
                .filter(|kind| !old.contains(kind))
                .copied()
                .collect::<Vec<_>>(),
            [ProjectionImage::Cohort2BAmpere64SlicedSm80]
        );
        assert_eq!(
            short_hf_required_images_for_cohort(FullModelProfile::Qwen08, true).unwrap(),
            short_hf_required_images(FullModelProfile::Qwen08).unwrap()
        );
        assert_eq!(
            short_hf_required_images_for_cohort(FullModelProfile::Qwen2B, false).unwrap(),
            old
        );
        assert_eq!(
            short_hf_native_provider_bytes_for_cohort(FullModelProfile::Qwen2B, true),
            141_557_760
        );
        assert_eq!(
            short_hf_native_provider_bytes_for_cohort(FullModelProfile::Qwen2B, false),
            135_266_304
        );
    }
    #[test]
    fn source_profile_unions_are_exact_sorted_distinct_and_exclude_activation() {
        let a = short_hf_required_images(FullModelProfile::Qwen08).unwrap();
        let b = short_hf_required_images(FullModelProfile::Qwen2B).unwrap();
        assert_eq!(a.len(), 16);
        assert_eq!(b.len(), 17);
        for images in [&a, &b] {
            let names = images
                .iter()
                .map(|kind| format!("{kind:?}"))
                .collect::<Vec<_>>();
            assert!(names.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(
                !images
                    .iter()
                    .any(|k| format!("{k:?}").starts_with("Rotary"))
            );
        }
        assert!(!a.contains(&ProjectionImage::ShortHfCutlassBf16Sm80));
        assert!(b.contains(&ProjectionImage::ShortHfCutlassBf16Sm80));
        assert_eq!(
            a.iter().map(|k| k.bytes()).sum::<usize>() + 901_664,
            69_577_200
        );
        assert_eq!(
            b.iter().map(|k| k.bytes()).sum::<usize>() + 901_664,
            73_542_672
        );
        assert_eq!(b.iter().filter(|k| !a.contains(k)).count(), 1);
        assert!(a.iter().all(|k| b.contains(k)));
    }
}
