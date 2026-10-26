//! Host-only selector/accounting guards, no new provider or completion permit.
use super::*;
#[test]
fn optional_qkv_budget_never_inherits_the_old_scope_envelope() {
    assert_eq!(
        provider_bytes(FixedPromptPrefixKind::FirstThree).unwrap(),
        320 * 1024 * 1024
    );
    match QKV_PROVIDER_BYTES {
        None => assert!(provider_bytes(FixedPromptPrefixKind::ThroughAttentionQkv).is_err()),
        Some(bytes) => assert_eq!(
            provider_bytes(FixedPromptPrefixKind::ThroughAttentionQkv).unwrap(),
            bytes
        ),
    }
}
#[test]
fn sum_selector_uses_original_fused_readback_order_and_refuses_future_nodes() {
    let old = [8, 11, 18, 21, 28];
    assert!(rms_sum_index(39, &old, 31).is_err());
    assert_eq!(rms_sum_index(39, &old, 28).unwrap(), 63);
    let new = [8, 11, 18, 21, 28, 31];
    assert_eq!(rms_sum_index(42, &new, 31).unwrap(), 70);
    for wrong in [0, 1, 30, 32, 33] {
        assert!(rms_sum_index(42, &new, wrong).is_err());
    }
}

#[test]
fn kv_budget_and_positions_read_cohort_never_inherit_qkv() {
    let cap = provider_bytes(FixedPromptPrefixKind::ThroughAttentionKv).unwrap();
    assert_eq!(QK_KV_PROVIDER_BYTES, Some(320 * 1024 * 1024));
    assert_eq!(cap, 335544320);
    // Independent whole-graph bound includes every managed alignment tail.
    let bound = 332709031 + 331 * (512 - 1);
    assert_eq!(bound, 332878172);
    assert_eq!(cap - bound, 2666148);
    let duals = [8, 11, 18, 21, 28, 31];
    assert_eq!(rms_sum_index(45, &duals, 31).unwrap(), 73);
    assert_eq!(7 + 45 + duals.len() * 4, 76);
}

#[test]
fn fa_budget_is_independent_and_readback_stops_at46() {
    assert_eq!(FA_PROVIDER_BYTES, Some(320 * 1024 * 1024));
    assert_eq!(FA_HOST_BOUND, Some(733_214_412));
    assert_eq!(
        provider_bytes(FixedPromptPrefixKind::ThroughAttentionFa).unwrap(),
        320 * 1024 * 1024
    );
    assert!(333_075_290 < provider_bytes(FixedPromptPrefixKind::ThroughAttentionFa).unwrap());
    assert!(FA_HOST_BOUND.unwrap() < 704 * 1024 * 1024);
    assert_eq!(7 + 46 + 6 * 4, 77);
    assert_eq!(77 + 2 + 96 + 19, 194);
    assert_eq!(79_323_916 + 192_512 + 36, 79_516_464);
    assert_eq!(332_709_031 + 192_512 + 1504 + 36, 332_903_083);
    assert_eq!(332_903_083 + 337 * 511, 333_075_290);
}

#[test]
fn full_layer3_accounting_and_envelopes_do_not_inherit_fa() {
    let cap = provider_bytes(FixedPromptPrefixKind::ThroughLayer3).unwrap();
    assert_eq!(cap, 384 * 1024 * 1024);
    assert_eq!(LAYER3_HOST_BOUND, Some(750_271_896));
    assert!(361_862_009 < cap);
    assert!(LAYER3_HOST_BOUND.unwrap() < 768 * 1024 * 1024);
    let duals = [8, 11, 18, 21, 28, 31, 39];
    assert_eq!(rms_sum_index(54, &duals, 39).unwrap(), 86);
    assert_eq!(7 + 54 + duals.len() * 4, 89);
    assert_eq!(89 + 2 + 96 + 19, 206);
    assert_eq!(79_516_464 + 1_780_736 + 387_072, 81_684_272);
    assert_eq!(332_903_083 + 1_780_736 + 780_852 + 26_214_400, 361_679_071);
    assert!(361_679_071 > provider_bytes(FixedPromptPrefixKind::ThroughAttentionFa).unwrap());
    assert_eq!(361_679_071 + 358 * 511, 361_862_009);
    assert_eq!(142 + 4 + 14 + 8, 168);
}
