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
