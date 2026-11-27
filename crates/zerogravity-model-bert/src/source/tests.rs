use super::*;
use serde_json::json;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
#[path = "../../tests/support/mod.rs"]
mod support;
use support::{Fixture, config, weights};

fn admission() -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host", "bert").unwrap();
    arbiter
        .register(resource.clone(), 256 * 1024 * 1024)
        .unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "encoder").unwrap(),
    }
}
fn reserved(a: &TransferAdmission) -> u64 {
    a.arbiter.snapshot(&a.resource).unwrap().reserved
}

#[test]
fn complete_source_bound_forward_and_result_retirement() {
    let a = admission();
    let f = Fixture::new();
    let (_store, snapshot) = f.snapshot(config(), weights(false, false), &a);
    let baseline = reserved(&a);
    let model = BertSentenceEncoder::load(&snapshot, &a).unwrap();
    assert_eq!(model.config().dimensions(), 4);
    assert!(model.source().is_bound());
    assert!(model.source.verified_single_sha256().is_some());
    assert!(model.source.verified_bundle_sha256().is_none());
    assert!(model.source.verified_shard_index_sha256().is_none());
    let retained = reserved(&a);
    let text = model.embed_text("HÉLLOS", None).unwrap();
    let ids = model.embed_tokens(&[2, 5, 6, 3], None).unwrap();
    assert_eq!(text.tokens(), 4);
    assert_eq!(text.values(), ids.values());
    drop((text, ids));
    assert_eq!(reserved(&a), retained);
    assert!(model.embed_text("hello hello hello hello", None).is_err());
    assert!(model.embed_text(" ", None).is_err());
    let first = model.embed_tokens(&[1, 2, 3], None).unwrap();
    let second = model.embed_tokens(&[1, 2, 4], None).unwrap();
    assert_eq!(first.tokens(), 3);
    assert_eq!(first.source(), model.source());
    assert_eq!(first.values().len(), 4);
    // Independent numeric oracle, generated using Python's scalar
    // math/erf implementation of the original BERT forward graph.
    let expected1 = [0.33089855f64, -0.59064984, 0.56863701, -0.46721610];
    let expected2 = [0.69847023f64, -0.07389471, -0.71059227, 0.04168299];
    for (actual, expected) in [(first.values(), expected1), (second.values(), expected2)] {
        for (&a, b) in actual.iter().zip(expected) {
            assert!(
                (f64::from(a) - b).abs() < 2e-5,
                "{actual:?} expected {expected:?}"
            );
        }
        assert!((actual.iter().map(|v| v * v).sum::<f32>() - 1.).abs() < 1e-6);
    }
    assert_ne!(first.values(), second.values());
    assert_eq!(reserved(&a), retained + 2 * 4 * 4);
    for input in [&[][..], &[7][..], &[1, 2, 3, 4, 5, 6][..]] {
        assert!(model.embed_tokens(input, None).is_err());
    }
    let cancelled = AtomicBool::new(true);
    assert!(model.embed_tokens(&[1], Some(&cancelled)).is_err());
    assert_eq!(reserved(&a), retained + 2 * 4 * 4);
    let pressure = reserve(
        &a,
        AllocationClass::Scratch,
        a.arbiter.snapshot(&a.resource).unwrap().remaining,
    )
    .unwrap();
    assert!(matches!(
        model.embed_tokens(&[1], None),
        Err(Error::Resource(_))
    ));
    drop(pressure);
    drop(model);
    assert_eq!(reserved(&a), baseline + 2 * 4 * 4);
    assert_eq!(first.source(), second.source());
    drop((first, second));
    assert_eq!(reserved(&a), baseline);
}

#[test]
fn refuse_bad_source_and_pooling_without_retained_charges() {
    for (bad_index, missing, bad_pool) in [
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        let a = admission();
        let f = Fixture::new();
        let mut files = config();
        if bad_pool {
            let p = files.last_mut().unwrap();
            let mut value: serde_json::Value = serde_json::from_slice(&p.1).unwrap();
            value["pooling_mode_mean_tokens"] = json!(false);
            value["pooling_mode_cls_token"] = json!(true);
            p.1 = value.to_string().into_bytes();
        }
        let (_store, snapshot) = f.snapshot(files, weights(bad_index, missing), &a);
        let baseline = reserved(&a);
        assert!(BertSentenceEncoder::load(&snapshot, &a).is_err());
        assert_eq!(reserved(&a), baseline);
    }
}
