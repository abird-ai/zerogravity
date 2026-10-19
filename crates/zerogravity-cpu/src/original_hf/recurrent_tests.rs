//! Expected words are actual untouched pinned source outputs, not this kernel.

use super::*;

fn source() -> SourceIdentity {
    SourceIdentity::for_new_allocation().unwrap()
}
fn exact(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "word {index}");
    }
}
fn fixture() -> Vec<Vec<f32>> {
    // Original Qwen3.5 .8B source36F32-restored; tokens11,353; layer0 step1.
    // Archive c8b7c72d...36c0fa; modeling762feb6c...412c3b; pinned Torch CPU
    // cf30153c... AVX2. Head0, exact key-major state and rawF32/BF16 core.
    // Lossless observation extraction, no reconstructed numerical goldens.
    // Fixture SHA256 78b2664bd9018b419ed1d1ea2d36f6a7fc01d615a0653cb962b2176fb9900065.
    let bytes = include_bytes!("recurrent/pinned_cached_head0.f32le.bin");
    assert_eq!(&bytes[..8], b"ZGCG128\0");
    let mut cursor = 8;
    let word = |cursor: &mut usize| {
        let value = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        value
    };
    assert_eq!(word(&mut cursor), 1);
    assert_eq!(word(&mut cursor), 1);
    assert_eq!(word(&mut cursor), 14);
    let mut fields = Vec::new();
    for expected in [
        128, 128, 128, 1, 1, 16384, 128, 128, 128, 128, 128, 128, 128, 16384,
    ] {
        let count = word(&mut cursor) as usize;
        assert_eq!(count, expected);
        fields.push(
            (0..count)
                .map(|_| f32::from_bits(word(&mut cursor)))
                .collect(),
        );
    }
    assert_eq!(cursor, bytes.len());
    fields
}

#[test]
fn actual_original_cached_head0_matches_every_state_and_output_word() {
    let fields = fixture();
    let source = source();
    let q = l2_norm_128(source, &fields[0], 1e-6).unwrap();
    let k = l2_norm_128(source, &fields[1], 1e-6).unwrap();
    exact(&q, &fields[6]);
    exact(&k, &fields[7]);
    let scaled = scale_cached_queries_128(source, &q).unwrap();
    exact(&scaled, &fields[8]);
    let input = CachedGdn128Inputs {
        source,
        heads: 1,
        query: &fields[0],
        key: &fields[1],
        value: &fields[2],
        beta: &fields[3],
        decay: &fields[4],
        previous_state: &fields[5],
    };
    let before = input.previous_state.to_vec();
    let result = cached_step_128(input).unwrap();
    exact(&result.core_f32, &fields[11]);
    exact(&result.output_bf16, &fields[12]);
    exact(&result.state, &fields[13]);
    exact(input.previous_state, &before);
    assert!(result.output_bf16.iter().all(|v| v.to_bits() & 0xffff == 0));
}

#[test]
fn actual_cached_division_and_separate_update_refuse_algebraic_substitutions() {
    let fields = fixture();
    let scaled = scale_cached_queries_128(source(), &fields[6]).unwrap();
    exact(&scaled, &fields[8]);
    let inverse = (WIDTH as f32).sqrt().recip();
    let changed = fields[6]
        .iter()
        .zip(&fields[8])
        .filter(|(q, expected)| (**q * inverse).to_bits() != expected.to_bits())
        .count();
    assert_eq!(changed, 48);
    // Actual observed proposed-state word3: an FMA changes the source result.
    let key = f32::from_bits(0x3c15_9309);
    let delta = f32::from_bits(0xbb95_c7a3);
    let decayed = f32::from_bits(0xb2cc_742e);
    let update = key * delta;
    assert_eq!((decayed + update).to_bits(), fields[13][3].to_bits());
    assert_eq!((decayed + update).to_bits(), 0xb82f_2001);
    assert_eq!(key.mul_add(delta, decayed).to_bits(), 0xb82f_2000);
}

#[test]
fn cached_zero_state_signs_and_bf16_final_rounding_stay_explicit() {
    let zero = vec![0.0; WIDTH];
    let negative = vec![-0.0; WIDTH];
    let state = vec![-0.0; WIDTH * WIDTH];
    let result = cached_step_128(CachedGdn128Inputs {
        source: source(),
        heads: 1,
        query: &zero,
        key: &zero,
        value: &negative,
        beta: &[1.0],
        decay: &[1.0],
        previous_state: &state,
    })
    .unwrap();
    assert!(
        result
            .state
            .iter()
            .all(|v| v.to_bits() == (-0.0f32).to_bits())
    );
    assert!(result.core_f32.iter().all(|v| v.to_bits() == 0));
    assert!(result.output_bf16.iter().all(|v| v.to_bits() == 0));
    for (input, expected) in [
        (0x3f80_8000, 0x3f80_0000),
        (0x3f81_8000, 0x3f82_0000),
        (0xbf80_8000, 0xbf80_0000),
        (0xbf81_8000, 0xbf82_0000),
        (0x8000_0000, 0x8000_0000),
        (0x0000_8000, 0),
    ] {
        assert_eq!(
            bf16_output(f32::from_bits(input)).unwrap().to_bits(),
            expected
        );
    }
    for value in [f32::MAX, f32::INFINITY, f32::NAN] {
        assert!(bf16_output(value).is_err());
    }
}

#[test]
fn malformed_source_layout_dtype_coefficients_and_overflow_never_mutate_prior_state() {
    let zero = vec![0.0; WIDTH];
    let state = vec![1.0; WIDTH * WIDTH];
    let good = CachedGdn128Inputs {
        source: source(),
        heads: 1,
        query: &zero,
        key: &zero,
        value: &zero,
        beta: &[0.5],
        decay: &[0.5],
        previous_state: &state,
    };
    let before = state.clone();
    assert!(
        cached_step_128(CachedGdn128Inputs {
            source: SourceIdentity::unbound(),
            ..good
        })
        .is_err()
    );
    for heads in [0, usize::MAX, MAX_ELEMENTS / (WIDTH * WIDTH) + 1] {
        assert!(cached_step_128(CachedGdn128Inputs { heads, ..good }).is_err());
    }
    assert!(
        cached_step_128(CachedGdn128Inputs {
            query: &zero[..127],
            ..good
        })
        .is_err()
    );
    assert!(
        cached_step_128(CachedGdn128Inputs {
            previous_state: &state[..state.len() - 1],
            ..good
        })
        .is_err()
    );
    let mut wrong_dtype = zero.clone();
    wrong_dtype[0] = f32::from_bits(0x3f80_0001);
    assert!(
        cached_step_128(CachedGdn128Inputs {
            query: &wrong_dtype,
            ..good
        })
        .is_err()
    );
    assert!(
        cached_step_128(CachedGdn128Inputs {
            key: &wrong_dtype,
            ..good
        })
        .is_err()
    );
    assert!(
        cached_step_128(CachedGdn128Inputs {
            value: &wrong_dtype,
            ..good
        })
        .is_err()
    );
    for coefficient in [-0.5, 2.0, f32::NAN, f32::INFINITY] {
        assert!(
            cached_step_128(CachedGdn128Inputs {
                beta: &[coefficient],
                ..good
            })
            .is_err()
        );
        assert!(
            cached_step_128(CachedGdn128Inputs {
                decay: &[coefficient],
                ..good
            })
            .is_err()
        );
    }
    assert!(
        cached_step_128(CachedGdn128Inputs {
            beta: &[0.50001],
            ..good
        })
        .is_err()
    );
    let mut nonfinite = state.clone();
    nonfinite[3] = f32::NAN;
    assert!(
        cached_step_128(CachedGdn128Inputs {
            previous_state: &nonfinite,
            ..good
        })
        .is_err()
    );
    // All source inputs finite/exact BF16; the materialized state read overflows.
    let ones = vec![1.0; WIDTH];
    let huge = vec![f32::MAX; WIDTH * WIDTH];
    assert!(
        cached_step_128(CachedGdn128Inputs {
            key: &ones,
            previous_state: &huge,
            decay: &[1.0],
            ..good
        })
        .is_err()
    );
    exact(&state, &before);
    for query in [&[][..], &zero[..127]] {
        assert!(scale_cached_queries_128(good.source, query).is_err());
    }
    assert!(scale_cached_queries_128(SourceIdentity::unbound(), &zero).is_err());
    assert!(
        scale_cached_queries_128(good.source, &vec![0.0; MAX_ELEMENTS / WIDTH + WIDTH]).is_err()
    );
}

#[test]
fn cached_heads_remain_independent_under_combination_and_reordering() {
    let mut query = vec![0.0; 2 * WIDTH];
    let mut key = vec![0.0; 2 * WIDTH];
    let mut value = vec![0.0; 2 * WIDTH];
    let mut previous = vec![0.0; 2 * WIDTH * WIDTH];
    query[0] = 1.0;
    key[1] = 1.0;
    value[2] = 2.0;
    previous[7 * WIDTH + 3] = 0.25;
    query[WIDTH + 3] = -2.0;
    key[WIDTH + 4] = 0.5;
    value[WIDTH + 5] = -3.0;
    previous[WIDTH * WIDTH + 9 * WIDTH + 6] = -0.5;
    let beta = [0.5, 0.25];
    let decay = [0.5, 0.75];
    let source = source();
    let input = CachedGdn128Inputs {
        source,
        heads: 2,
        query: &query,
        key: &key,
        value: &value,
        beta: &beta,
        decay: &decay,
        previous_state: &previous,
    };
    let together = cached_step_128(input).unwrap();
    for head in 0..2 {
        let range = head * WIDTH..(head + 1) * WIDTH;
        let state = head * WIDTH * WIDTH..(head + 1) * WIDTH * WIDTH;
        let alone = cached_step_128(CachedGdn128Inputs {
            source,
            heads: 1,
            query: &query[range.clone()],
            key: &key[range.clone()],
            value: &value[range.clone()],
            beta: &beta[head..head + 1],
            decay: &decay[head..head + 1],
            previous_state: &previous[state.clone()],
        })
        .unwrap();
        exact(&alone.state, &together.state[state]);
        exact(&alone.core_f32, &together.core_f32[range.clone()]);
        exact(&alone.output_bf16, &together.output_bf16[range]);
    }
    let reorder = |values: &[f32], width: usize| {
        values[width..]
            .iter()
            .chain(&values[..width])
            .copied()
            .collect::<Vec<_>>()
    };
    let q = reorder(&query, WIDTH);
    let k = reorder(&key, WIDTH);
    let v = reorder(&value, WIDTH);
    let state = reorder(&previous, WIDTH * WIDTH);
    let swapped = cached_step_128(CachedGdn128Inputs {
        source,
        heads: 2,
        query: &q,
        key: &k,
        value: &v,
        beta: &[beta[1], beta[0]],
        decay: &[decay[1], decay[0]],
        previous_state: &state,
    })
    .unwrap();
    exact(&swapped.state, &reorder(&together.state, WIDTH * WIDTH));
    exact(&swapped.core_f32, &reorder(&together.core_f32, WIDTH));
    exact(&swapped.output_bf16, &reorder(&together.output_bf16, WIDTH));
}
