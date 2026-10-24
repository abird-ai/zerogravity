use super::*;
fn values() -> [Vec<u8>; 7] {
    let mut v = std::array::from_fn(|i| vec![0; [262144, 262144, 262144, 4096, 4096, 8, 8][i]]);
    v[5][4] = 64;
    v
}
fn check(v: &[Vec<u8>; 7]) -> Result<()> {
    validate_values(std::array::from_fn(|i| v[i].as_slice()))
}
#[test]
fn t64_controls_reject47_other_span_and_second_chunk() {
    let mut v = values();
    assert!(check(&v).is_ok());
    for n in [1, 47, 63, 65] {
        v[5][4] = n;
        assert!(check(&v).is_err());
    }
    v[5][4] = 64;
    for i in 0..8 {
        v[6][i] = 1;
        assert!(check(&v).is_err());
        v[6][i] = 0;
    }
    v[5][0] = 1;
    assert!(check(&v).is_err());
}
#[test]
fn every_full_operand_type_extent_and_nonfinite_word_refuses() {
    for i in 0..7 {
        let mut v = values();
        v[i].pop();
        assert!(check(&v).is_err());
    }
    for i in 0..5 {
        let mut v = values();
        let bad = if matches!(i, 1 | 2) {
            vec![0xc1, 0x7f]
        } else {
            vec![1, 0, 0x80, 0x7f]
        };
        let at = v[i].len() - bad.len();
        v[i][at..].copy_from_slice(&bad);
        assert!(check(&v).is_err());
    }
}
#[test]
fn full_t64_zero_ai_checks_all_unwritten_blocks_and_final_real_row() {
    let mut ai = vec![0; 131072];
    assert!(validate_unwritten_ai(&ai).is_ok());
    for row in [0, 15, 16, 31, 32, 47] {
        for head in 0..16 {
            let col = (row / 16 + 1) * 16;
            let at = ((row * 16 + head) * 64 + col) * 2;
            ai[at + 1] = 0x80;
            assert!(validate_unwritten_ai(&ai).is_err());
            ai[at + 1] = 0;
        }
    }
    ai[((63 * 16 + 15) * 64 + 63) * 2] = 7;
    assert!(validate_unwritten_ai(&ai).is_ok());
    ai.push(0);
    assert!(validate_unwritten_ai(&ai).is_err());
}
#[test]
fn no_variant_alias_or_new_math_source_and_two_fresh_cohorts() {
    assert!(include_str!("../../vllm_cuda_gdn/solve.cu").contains(OWNED.solve_ptx_sha256()));
    assert!(
        include_str!("../../vllm_cuda_gdn/qwen2b_recompute.cu")
            .contains(OWNED.recompute_ptx_sha256())
    );
    assert_ne!(
        OWNED.recompute_ptx_sha256(),
        "5731fd146617cf5b6cc13f28b2622c31135df4a35907567ffba83ff9b0b10c9a"
    );
    assert_eq!(
        REFERENCE,
        Qwen2bOriginalGdnReference::ObservedSolve4Recompute4
    );
    assert_eq!(INPUT_BYTES + 2 * OUTPUT_BYTES, 2_105_360);
    assert!(resident::host_bound().unwrap() <= READBACK_HOST);
}
