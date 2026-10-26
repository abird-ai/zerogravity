//! Closed whole-image catalogue; neither source discovery nor kind grants an ABI permit.
use super::super::cubin_manifests::ProjectionImage;
use super::{NativeCatalogue, TextKernelCapture};
use zerogravity_core::Result;

/// Actual union across five prompt spans and seven natural continuations.
/// Activation is separately admitted by FixedPromptMlpQualificationInputs.
pub fn text_required_images(capture: TextKernelCapture) -> Result<Vec<ProjectionImage>> {
    let mut images = match capture.catalogue() {
        NativeCatalogue::Source1024 => vec![
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
            ProjectionImage::FixedPromptCutlassBf16Sm80,
            ProjectionImage::CutlassWmmaSm80,
            ProjectionImage::TextFaCausalCombineSm80,
            ProjectionImage::TextFaSplitCombineSm80,
        ],
        NativeCatalogue::Source2048 => vec![
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
            ProjectionImage::FixedPromptCutlassBf16Sm80,
            ProjectionImage::CutlassWmmaSm80,
            ProjectionImage::TextCutlassBf16Sm80,
            ProjectionImage::TextFaCausalCombineSm80,
            ProjectionImage::TextFaSplitCombineSm80,
        ],
    };
    images.sort_by_cached_key(|kind| format!("{kind:?}"));
    images.dedup();
    Ok(images)
}

/// Selected source phase union; N1 callers retain the original exact image set.
/// The cached2B addition is a distinct full capture, never a renamed N1 image.
pub fn text_required_images_for_cohort(
    capture: TextKernelCapture,
    cohort_b4: bool,
) -> Result<Vec<ProjectionImage>> {
    let mut images = text_required_images(capture)?;
    if cohort_b4 && capture.wide() {
        images.push(ProjectionImage::Cohort2BAmpere64SlicedSm80);
    }
    images.sort_by_cached_key(|kind| format!("{kind:?}"));
    images.dedup();
    Ok(images)
}

/// Declared code/global-section alignment upper + original opaque64MiB owner.
/// On-chip shared sections and host image bytes are separate resources.
/// Actual module allocation/whole-unit physical fit requires ROOT execution.
pub fn text_native_provider_bytes(capture: TextKernelCapture) -> usize {
    let sections = match capture.catalogue() {
        NativeCatalogue::Source1024 => 63_985_828usize,
        NativeCatalogue::Source2048 => 67_743_743usize,
    };
    sections.div_ceil(1024 * 1024) * (1024 * 1024) + 64 * 1024 * 1024
}

/// Extra exact ed95 image:288 SHF_ALLOC sections,5,622,112B. Sum each
/// size+max(256,section_alignment)-1 =5,695,552B; maximum actual alignment128.
/// The original finite opaque64MiB policy remains unchanged, shared by the set.
pub fn text_native_provider_bytes_for_cohort(capture: TextKernelCapture, cohort_b4: bool) -> usize {
    if cohort_b4 && capture.wide() {
        (67_743_743usize + 5_695_552).div_ceil(1024 * 1024) * (1024 * 1024) + 64 * 1024 * 1024
    } else {
        text_native_provider_bytes(capture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cached2b_adds_exact_whole_image_and_prices_its_complete_section_upper() {
        let old = text_required_images(
            TextKernelCapture::bind(
                "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0",
            )
            .unwrap(),
        )
        .unwrap();
        let cached = text_required_images_for_cohort(
            TextKernelCapture::bind(
                "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0",
            )
            .unwrap(),
            true,
        )
        .unwrap();
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
            text_required_images_for_cohort(
                TextKernelCapture::bind(
                    "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d"
                )
                .unwrap(),
                true
            )
            .unwrap(),
            text_required_images(
                TextKernelCapture::bind(
                    "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d"
                )
                .unwrap()
            )
            .unwrap()
        );
        assert_eq!(
            text_required_images_for_cohort(
                TextKernelCapture::bind(
                    "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0"
                )
                .unwrap(),
                false
            )
            .unwrap(),
            old
        );
        assert_eq!(
            text_native_provider_bytes_for_cohort(
                TextKernelCapture::bind(
                    "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0"
                )
                .unwrap(),
                true
            ),
            141_557_760
        );
        assert_eq!(
            text_native_provider_bytes_for_cohort(
                TextKernelCapture::bind(
                    "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0"
                )
                .unwrap(),
                false
            ),
            135_266_304
        );
    }
    #[test]
    fn capture_unions_are_exact_sorted_distinct_and_exclude_activation() {
        let a = text_required_images(
            TextKernelCapture::bind(
                "39b9a6180ca95f95510aac6471e0255ba62100200bf6daeac6f5a4fc8d2bb30d",
            )
            .unwrap(),
        )
        .unwrap();
        let b = text_required_images(
            TextKernelCapture::bind(
                "0938d4353dc49579daa5978a6dc78a70f23c8e8d7cb21130c95a914c0a82c4b0",
            )
            .unwrap(),
        )
        .unwrap();
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
        assert!(!a.contains(&ProjectionImage::TextCutlassBf16Sm80));
        assert!(b.contains(&ProjectionImage::TextCutlassBf16Sm80));
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
