use super::super::policy::CubinSetPolicy as P;
use super::*;
#[test]
fn distinct_bounded_policy_preserves_old_capacities() {
    assert!(P::Qwen2bMlp.limits(5, 7, 6, false, false).is_ok());
    assert!(P::Qwen2bMlp.limits(5, 7, 7, false, false).is_err());
    assert!(P::Qwen2bMlp.limits(5, 7, 6, true, false).is_err());
    assert!(P::Qwen2bMlp.limits(5, 7, 6, false, true).is_err());
    assert!(P::Canonical47Mlp.limits(5, 7, 7, false, false).is_ok());
    assert!(P::Canonical47Qkv.limits(6, 9, 9, false, false).is_ok());
    assert!(P::ProjectionSix.limits(6, 8, 18, false, false).is_ok());
}
#[test]
fn every_sealed_tuple_dimension_is_required() {
    let record = ([1; 32], 901664, 86, [2; 32]);
    let rules = [record];
    assert!(matches_rule(&rules, record));
    assert!(!matches_rule(&[], record));
    for changed in [
        ([0; 32], record.1, record.2, record.3),
        (record.0, record.1 + 1, record.2, record.3),
        (record.0, record.1, 80, record.3),
        (record.0, record.1, record.2, [3; 32]),
    ] {
        assert!(!matches_rule(&rules, changed));
    }
}
