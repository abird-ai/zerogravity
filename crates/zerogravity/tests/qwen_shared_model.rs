#![cfg(all(feature = "cpu", feature = "model-qwen35"))]

//! Source-real model-load, multi-session and registry concurrency proof.
//! Ignored by default because it requires the original local Qwen checkpoints.
//! No foreign inference engine, copied GGUF, model download or GPU required.
use std::{sync::Arc, thread};
use zerogravity::{
    model_catalog::ModelCatalog,
    qwen35::{LoadedQwenModel, QwenDenseCpu},
};

mod common;

#[test]
#[ignore = "requires original SHA-pinned Qwen3.5 0.8B and 2B GGUF + geometry"]
fn concurrent_model_catalog_loads_once_per_source_and_sessions_are_isolated() {
    let catalog = ModelCatalog::new();
    for (name, model) in [
        ("small", "qwen35-0.8b"),
        ("small-alias", "qwen35-0.8b"),
        ("large", "qwen35-2b"),
    ] {
        let root = common::workspace_path("target/models").join(model);
        catalog
            .register(name, &root.join("model.gguf"), &root.join("config.json"))
            .unwrap();
    }
    assert_eq!(catalog.list().unwrap().len(), 3);
    assert!(catalog.list().unwrap().iter().all(|x| !x.loaded));

    // Simultaneous aliases and unrelated models: one load per unique source.
    // Original source identities must not be accidentally regenerated on
    // per-session creation or conflated between different model checkpoints.
    let (small, duplicate, large) = thread::scope(|scope| {
        let a = scope.spawn(|| catalog.load("small").unwrap());
        let b = scope.spawn(|| catalog.load("small-alias").unwrap());
        let c = scope.spawn(|| catalog.load("large").unwrap());
        (a.join().unwrap(), b.join().unwrap(), c.join().unwrap())
    });
    assert!(Arc::ptr_eq(&small, &duplicate));
    assert!(!Arc::ptr_eq(&small, &large));
    assert_eq!(small.source_identity(), duplicate.source_identity());
    assert_ne!(small.source_identity(), large.source_identity());
    assert_eq!(
        small.source_bytes() as u64,
        std::fs::metadata(common::workspace_path(
            "target/models/qwen35-0.8b/model.gguf"
        ))
        .unwrap()
        .len(),
    );
    assert_eq!(
        large.source_bytes() as u64,
        std::fs::metadata(common::workspace_path("target/models/qwen35-2b/model.gguf"))
            .unwrap()
            .len(),
    );
    assert!(catalog.list().unwrap().iter().all(|x| x.loaded));
    assert_eq!(Arc::strong_count(&small), 3); // loaded slot, first alias, duplicate alias
    // aliases contain the same ModelSlot, not separate Arc<LoadedQwenModel>s.
    // Strong count includes this variable, 'duplicate', and loaded source slot.

    let one = small.new_session().unwrap();
    let mut another = catalog.new_session("small-alias").unwrap();
    let mut third = QwenDenseCpu::from_shared(small.clone()).unwrap();
    assert!(Arc::ptr_eq(one.loaded_model(), another.loaded_model()));
    assert!(Arc::ptr_eq(another.loaded_model(), third.loaded_model()));
    assert_eq!(one.position(), 0);
    assert_eq!(one.root_id(), 0);
    let original_second_state = another.state_fingerprint();
    let original_third_state = third.state_fingerprint();

    let (one, logits) = thread::spawn(move || {
        let mut session = one;
        let logits = session.forward(11).unwrap();
        (session, logits)
    })
    .join()
    .unwrap();
    assert_eq!(one.root_id(), 1);
    assert_eq!(another.position(), 0);
    assert_eq!(another.root_id(), 0);
    assert_eq!(another.state_fingerprint(), original_second_state);
    assert_eq!(third.state_fingerprint(), original_third_state);
    assert!(another.restore(&one.snapshot()).is_err());

    // Independent sessions of the same immutable model compute identical
    // numerical results when given an identical input/empty frontier.
    let logits_b = another.forward(11).unwrap();
    assert_eq!(logits, logits_b);
    assert_eq!(one.state_fingerprint(), another.state_fingerprint());
    assert_eq!(one.root_id(), another.root_id());
    assert_eq!(third.position(), 0);
    third.forward(353).unwrap();
    assert_eq!(third.root_id(), 1);
    assert_ne!(third.state_fingerprint(), one.state_fingerprint());

    // Unregister prevents NEW lookups but never forcibly detaches a running
    // session, resets an active StateRoot or changes original source ID.
    assert!(catalog.unregister("small").unwrap());
    assert!(catalog.load("small").is_err());
    assert!(catalog.unregister("small-alias").unwrap());
    assert!(catalog.load("small-alias").is_err());
    assert_eq!(one.root_id(), 1);
    let later = QwenDenseCpu::from_shared(small.clone()).unwrap();
    assert!(Arc::ptr_eq(later.loaded_model(), one.loaded_model()));
    assert_eq!(later.position(), 0);
    assert!(
        catalog
            .list()
            .unwrap()
            .iter()
            .all(|item| item.alias != "small")
    );
    assert!(
        catalog
            .list()
            .unwrap()
            .iter()
            .any(|item| item.alias == "large")
    );
    println!(
        "P07_MODEL_CATALOG_REAL_CHECKPOINTS_PASS unique_models=2 aliases=3 source_copies_per_model=1 sessions=4 independently_committed_roots=3"
    );
}

#[test]
#[ignore = "requires original Qwen3.5 0.8B source"]
fn loaded_source_multi_session_matches_legacy_direct_open() {
    let root = common::workspace_path("target/models/qwen35-0.8b");
    let original =
        LoadedQwenModel::open(&root.join("model.gguf"), &root.join("config.json")).unwrap();
    let mut shared = original.new_session().unwrap();
    let mut old = QwenDenseCpu::open(&root.join("model.gguf"), &root.join("config.json")).unwrap();
    let ids = [11u32, 353, 1044];
    let expected = old.prefill_transaction(&ids).unwrap();
    let actual = shared.prefill_transaction(&ids).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(shared.state_fingerprint(), old.state_fingerprint());
    assert_eq!(shared.position(), old.position());
    assert_eq!(shared.root_id(), old.root_id());
    assert_ne!(
        shared.loaded_model().source_identity(),
        old.loaded_model().source_identity()
    );
}
