use super::*;
fn values() -> [Vec<u8>; 7] {
    let mut v = std::array::from_fn(|i| vec![0; [192512, 192512, 192512, 3008, 3008, 8, 8][i]]);
    v[5][4] = 47;
    v
}
fn check(v: &[Vec<u8>; 7]) -> Result<()> {
    validate_values(std::array::from_fn(|i| v[i].as_slice()))
}
#[test]
fn original47_controls_cannot_be_retagged64_or_second_chunk() {
    let mut v = values();
    assert!(check(&v).is_ok());
    for n in [1, 46, 48, 64] {
        v[5][4] = n;
        assert!(check(&v).is_err());
    }
    v[5][4] = 47;
    for i in 0..8 {
        v[6][i] = 1;
        assert!(check(&v).is_err());
        v[6][i] = 0;
    }
    v[5][0] = 1;
    assert!(check(&v).is_err());
}
#[test]
fn exact_physical_types_extents_and_finite_values() {
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
        v[i][..bad.len()].copy_from_slice(&bad);
        assert!(check(&v).is_err());
    }
}
#[test]
fn every_head_and_last_real_row_preserves_unwritten_zeros() {
    let mut ai = vec![0; 96256];
    assert!(validate_unwritten_ai(&ai).is_ok());
    for row in [0, 15, 16, 31, 32, 46] {
        for head in 0..16 {
            let column = (row / 16 + 1) * 16;
            let at = ((row * 16 + head) * 64 + column) * 2;
            ai[at + 1] = 0x80;
            assert!(validate_unwritten_ai(&ai).is_err());
            ai[at + 1] = 0;
        }
    }
    ai[((46 * 16 + 15) * 64 + 46) * 2] = 7;
    assert!(validate_unwritten_ai(&ai).is_ok());
    ai.push(0);
    assert!(validate_unwritten_ai(&ai).is_err());
}
#[test]
fn original_two_source_variants_are_distinct_complete_code() {
    let source = include_str!("solve2.cu");
    assert!(source.contains(FixedPromptSolveVariant::Warp2Stage2.solve_ptx_sha256()));
    assert!(source.contains("__launch_bounds__(64)"));
    assert!(source.contains(
        "zg_vllm_8f235650d676e07a(void* zg_arg0, void* zg_arg1, void* zg_arg2, void* zg_arg3)"
    ));
    assert!(source.contains("// cubecl.cuda.dynamic_shared_bytes=10240"));
    assert_ne!(
        FixedPromptSolveVariant::Warp2Stage2.solve_ptx_sha256(),
        FixedPromptSolveVariant::Warp4Stage4.solve_ptx_sha256()
    );
    assert_eq!(INPUT_BYTES + 4 * OUTPUT_BYTES, 2_508_688);
    assert!(resident::host_bound().unwrap() <= READBACK_HOST);
}
