//! Pure geometry/admission guards; no source/device/math qualification.
use super::*;
#[test]
fn closed_cut_additions_fit_the_separate_provider_envelope() {
    assert_eq!(ComponentCut::Gated.node_end(), 6);
    assert_eq!(ComponentCut::Gated.value_end(), 7);
    assert_eq!(ComponentCut::PostAttention.node_end(), 8);
    assert_eq!(ComponentCut::PostAttention.value_end(), 10);
    assert_eq!(ComponentCut::Gated.provider_bytes(), 192 * 1024 * 1024);
    assert_eq!(
        ComponentCut::PostAttention.provider_bytes(),
        224 * 1024 * 1024
    );
    for (rows, hidden, added) in [
        (1, 2048, 8_454_156),
        (64, 2048, 11_293_440),
        (47, 1024, 5_263_924),
    ] {
        let dense = rows * hidden;
        let dual = 20 * dense + 10 * hidden + 12 * rows;
        assert_eq!(hidden * 4096 + 2 * dense + dual, added);
        assert!(
            ComponentCut::Gated.provider_bytes() + added + 14 * 511
                < ComponentCut::PostAttention.provider_bytes()
        );
    }
}
#[test]
fn fixed47_windows_are_charged_independently_at_every_tensor_alignment() {
    let chunk = 1024 * 1024;
    let stride = 2048;
    let mut multiplicity = [0; 512];
    for id in recipe::CANONICAL47_PROMPT {
        multiplicity[id as usize % 512] += 1;
    }
    assert_eq!(multiplicity.into_iter().max(), Some(10));
    // Exact alignment has zero extra chunks. Every positive sub-row offset
    // selects one residue; no source offset or duplicate-window sharing assumed.
    for residue in 0..512 {
        for subrow in [0, 1, 2047] {
            let start = residue * stride + subrow;
            let chunks = recipe::CANONICAL47_PROMPT
                .into_iter()
                .map(|id| {
                    let begin = start + id as usize * stride;
                    (begin + stride - 1) / chunk - begin / chunk + 1
                })
                .sum::<usize>();
            assert!(chunks <= 57);
        }
    }
}
