use super::*;
use crate::accelerator::program::TorchCpuAttentionGeometry;

fn geometry() -> TorchCpuAttentionGeometry {
    TorchCpuAttentionGeometry {
        query_heads: 2,
        kv_heads: 1,
        width: 256,
        rotary: 64,
        theta: 10_000_000.0,
    }
}
#[test]
fn analytic_gqa_cached_step_preserves_full_prior_and_staged_bf16_values() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let q = vec![0.0; 512];
    let k = vec![0.0; 256];
    let v = vec![3.0; 256];
    let prior_k = vec![0.0; 256];
    let prior_v = vec![1.0; 256];
    let result = eager_bf16(
        source,
        TorchCpuAttentionInputs {
            query: &q[..],
            key: &k[..],
            value: &v[..],
            gate: &q[..],
            prior: Some((&prior_k[..], &prior_v[..])),
            geometry: geometry(),
            position: 1,
        },
        None,
    )
    .unwrap();
    assert_eq!(result.dots, vec![0.0; 4]);
    assert_eq!(result.probabilities, vec![0.5; 4]);
    assert_eq!(result.context, vec![2.0; 512]);
    assert_eq!(result.gated, vec![1.0; 512]);
    assert_eq!(result.values, [prior_v.clone(), v].concat());
    assert_eq!(prior_v, vec![1.0; 256]);
    assert_eq!(&result.rotated_query[64..256], &q[64..256]);
}
#[test]
fn partial_rotary_keeps_unrotated_lanes_and_materializes_each_bf16_product() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut q = vec![0.0; 512];
    q[0] = 1.0;
    q[32] = 2.0;
    q[64] = -3.0;
    let k = vec![0.0; 256];
    let v = vec![1.0; 256];
    let gate = vec![0.0; 512];
    let result = eager_bf16(
        source,
        TorchCpuAttentionInputs {
            query: &q[..],
            key: &k[..],
            value: &v[..],
            gate: &gate[..],
            prior: Some((&k[..], &v[..])),
            geometry: geometry(),
            position: 1,
        },
        None,
    )
    .unwrap();
    assert_eq!(result.rotated_query[0], -1.140625);
    assert_eq!(result.rotated_query[32], 1.921875);
    assert_eq!(result.rotated_query[64].to_bits(), (-3.0f32).to_bits());
}
#[test]
fn malformed_nonfinite_foreign_source_and_cancel_fail_without_prefix_changes() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let q = vec![0.0; 512];
    let k = vec![0.0; 256];
    let v = vec![1.0; 256];
    let input = || TorchCpuAttentionInputs {
        query: &q[..],
        key: &k[..],
        value: &v[..],
        gate: &q[..],
        prior: None,
        geometry: geometry(),
        position: 0,
    };
    assert!(eager_bf16(source, input(), Some(&AtomicBool::new(true))).is_err());
    assert!(eager_bf16(SourceIdentity::unbound(), input(), None).is_err());
    let mut bad = input();
    bad.position = 1;
    assert!(eager_bf16(source, bad, None).is_err());
    let mut bad = input();
    bad.geometry.query_heads = 3;
    bad.geometry.kv_heads = 2;
    assert!(eager_bf16(source, bad, None).is_err());
    assert!(geometry().validate(64).is_err());
    let mut nan = q.clone();
    nan[0] = f32::NAN;
    let mut bad = input();
    bad.query = &nan;
    assert!(eager_bf16(source, bad, None).is_err());
    assert_eq!(v, vec![1.0; 256]);
}

#[test]
fn eager_cached_pv_uses_original_bmm_order_not_the_projection_dot() {
    // Independent Torch2.13.0+cu129 cf30153c4, AVX2, original Qwen repeat_kv/matmul.
    // Raw32-case artifact SHA256:9d43819ebae00ec8f00ad8d4840507b37108ca0ee971c0903c7618163283bc3b.
    let expected = [
        (1, 0x4b800000),
        (2, 0x4b000000),
        (3, 0x3f000000),
        (4, 0x3e800000),
        (7, 0x3f240000),
        (8, 0x3e000000),
        (15, 0x3ea20000),
        (16, 0x00000000),
        (31, 0x3e900000),
        (32, 0x00000000),
        (47, 0x3e8b0000),
        (48, 0x3e800000),
        (49, 0x48a70000),
        (50, 0x48a40000),
        (63, 0x3e880000),
        (64, 0x00000000),
    ];
    let pattern = [
        16_777_216.0,
        1.0,
        -16_777_216.0,
        1.0,
        1.0 / 1024.0,
        -1.0 / 1024.0,
        1.0,
        -1.0,
    ];
    let source = SourceIdentity::for_new_allocation().unwrap();
    let g = TorchCpuAttentionGeometry {
        query_heads: 8,
        kv_heads: 2,
        ..geometry()
    };
    let q = vec![0.0; 8 * 256];
    let k = vec![0.0; 2 * 256];
    for (tokens, word) in expected {
        let values = (0..tokens)
            .flat_map(|t| std::iter::repeat_n(pattern[t % 8], 512))
            .collect::<Vec<_>>();
        let prior_keys = vec![0.0; (tokens - 1) * 512];
        let prior = if tokens == 1 {
            None
        } else {
            Some((prior_keys.as_slice(), &values[..(tokens - 1) * 512]))
        };
        let result = eager_bf16(
            source,
            TorchCpuAttentionInputs {
                query: &q,
                key: &k,
                value: &values[(tokens - 1) * 512..],
                gate: &q,
                prior,
                geometry: g,
                position: tokens - 1,
            },
            None,
        )
        .unwrap();
        assert!(
            result.context.iter().all(|v| v.to_bits() == word),
            "original BMM context at{tokens}: {:08x} != {word:08x}",
            result.context[0].to_bits()
        );
        assert_eq!(result.values, values);
        assert_eq!(prior_keys, vec![0.0; (tokens - 1) * 512]);
    }
}

#[path = "../../../../../../tests/support/torch_cpu_attention.rs"]
mod original_archive;

#[test]
#[ignore = "requires independently captured pinned original HF attention archive"]
fn original_hf_attention_all_stages() {
    let cases = original_archive::read().unwrap();
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut words = 0;
    let mut mismatches = 0;
    for (index, case) in cases.iter().enumerate() {
        let inputs = case
            .inputs
            .iter()
            .map(|bytes| {
                bytes
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let result = eager_bf16(
            source,
            TorchCpuAttentionInputs {
                query: &inputs[0],
                key: &inputs[1],
                value: &inputs[2],
                gate: &inputs[3],
                prior: if case.position == 0 {
                    None
                } else {
                    Some((&inputs[4], &inputs[5]))
                },
                geometry: case.geometry,
                position: case.position,
            },
            None,
        )
        .unwrap();
        for (stage, (actual, expected)) in [
            result.rotated_query,
            result.rotated_key,
            result.keys,
            result.values,
            result.dots,
            result.scores,
            result.probabilities,
            result.context,
            result.gated,
        ]
        .iter()
        .zip(&case.expected)
        .enumerate()
        {
            assert_eq!(actual.len() * 4, expected.len());
            let mut differences = 0;
            for (word, (a, e)) in actual.iter().zip(expected.chunks_exact(4)).enumerate() {
                let e = u32::from_le_bytes(e.try_into().unwrap());
                if a.to_bits() != e {
                    if differences == 0 {
                        eprintln!(
                            "SOURCE_ATTENTION_CPU_DIFFERENCE case={index} stage={stage} word={word} actual={:08x} expected={e:08x}",
                            a.to_bits()
                        );
                    }
                    differences += 1;
                }
            }
            words += actual.len();
            mismatches += differences;
            eprintln!(
                "SOURCE_ATTENTION_CPU_STAGE case={index} stage={stage} words={} mismatches={differences}",
                actual.len()
            );
        }
    }
    eprintln!(
        "SOURCE_ATTENTION_CPU_RESULT cases={} words={words} mismatches={mismatches}",
        cases.len()
    );
    assert_eq!(mismatches, 0);
    assert_eq!(words, 1_774_240);
}
