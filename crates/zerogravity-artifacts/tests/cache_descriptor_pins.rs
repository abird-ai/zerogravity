//! Fresh policy and cooperative descriptor ownership, using synthetic roots.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use std::fs;
use std::os::unix::fs::DirBuilderExt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::Error;
use zerogravity_artifacts::cache::{CacheStore, ReclaimMode, ReclaimOutcome};

#[test]
fn new_empty_root_is_protected_and_concurrent_first_open_agrees() {
    let fixture = Fixture::new();
    std::thread::scope(|scope| {
        let first = scope.spawn(|| CacheStore::open(options(&fixture.root)).unwrap());
        let second = scope.spawn(|| CacheStore::open(options(&fixture.root)).unwrap());
        assert_eq!(
            first.join().unwrap().root_identity(),
            second.join().unwrap().root_identity()
        );
    });
    let policy: serde_json::Value =
        serde_json::from_slice(&fs::read(fixture.root.join("policy.json")).unwrap()).unwrap();
    assert_eq!(policy["schema"], "zerogravity.cache.policy.v2");
    assert!(fixture.root.join("retirement").is_dir());
    CacheStore::open_existing(options(&fixture.root)).unwrap();
}

#[test]
fn existing_opener_never_creates_or_adopts_missing_roots_and_policies() {
    let fixture = Fixture::new();
    assert_not_found(CacheStore::open_existing(options(&fixture.root)));
    assert!(!fixture.root.exists());
    drop(fixture.open());
    fs::remove_file(fixture.root.join("policy.json")).unwrap();
    assert_not_found(CacheStore::open_existing(options(&fixture.root)));
    assert!(!fixture.root.join("policy.json").exists());
    fs::write(fixture.root.join("unqualified-payload"), NEW).unwrap();
    assert!(matches!(
        CacheStore::open(options(&fixture.root)),
        Err(Error::State(_))
    ));
    assert!(!fixture.root.join("policy.json").exists());
    assert_eq!(
        fs::read(fixture.root.join("unqualified-payload")).unwrap(),
        NEW
    );
}

#[test]
fn existing_opener_does_not_recreate_required_or_protected_directories() {
    let fixture = Fixture::new();
    drop(fixture.open());
    let policy = fs::read(fixture.root.join("policy.json")).unwrap();
    fs::remove_dir(fixture.root.join("retirement")).unwrap();
    assert_not_found(CacheStore::open_existing(options(&fixture.root)));
    assert!(!fixture.root.join("retirement").exists());
    assert_eq!(fs::read(fixture.root.join("policy.json")).unwrap(), policy);
    fs::remove_dir(fixture.root.join("snapshots")).unwrap();
    assert_not_found(CacheStore::open_existing(options(&fixture.root)));
    assert!(!fixture.root.join("snapshots").exists());
}

#[test]
fn pending_retirement_marker_refuses_publication_before_changing_old_reference() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    publish(&store, &memory, OLD, Some("fixture")).unwrap();
    let pending = fixture
        .root
        .join("retirement")
        .join(payload_manifest(NEW).digest().unwrap());
    fs::DirBuilder::new().mode(0o700).create(&pending).unwrap();
    assert!(matches!(
        publish(&store, &memory, NEW, Some("fixture")),
        Err(Error::State(_))
    ));
    assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    fs::remove_dir(pending).unwrap();
    publish(&store, &memory, NEW, Some("fixture")).unwrap();
    assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
    assert_no_host_reservation(&memory);
}

#[test]
fn legacy_policy_bytes_and_payloads_remain_unchanged_and_removal_refuses() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &memory, NEW, Some("fixture")).unwrap());
    drop(store);
    let policy_path = fixture.root.join("policy.json");
    let mut policy: serde_json::Value =
        serde_json::from_slice(&fs::read(&policy_path).unwrap()).unwrap();
    policy["schema"] = "zerogravity.cache.policy.v1".into();
    let legacy_bytes = serde_json::to_vec(&policy).unwrap();
    fs::write(&policy_path, &legacy_bytes).unwrap();
    fs::remove_dir(fixture.root.join("retirement")).unwrap();
    fs::remove_file(fixture.root.join("locks/activity")).unwrap();
    let legacy = CacheStore::open_existing(options(&fixture.root)).unwrap();
    assert_payload(&legacy.resolve_ref("fixture").unwrap(), NEW);
    // Ordinary old-root publication/reopen/read stays compatible without
    // allocating the v2-only maintenance barrier or changing policy bytes.
    publish(&legacy, &memory, OLD, None).unwrap();
    CacheStore::open(options(&fixture.root)).unwrap();
    assert_payload(&legacy.open_snapshot(&snapshot).unwrap(), NEW);
    assert!(matches!(
        legacy.reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false)),
        Err(Error::Unsupported(_))
    ));
    assert_eq!(fs::read(&policy_path).unwrap(), legacy_bytes);
    assert!(!fixture.root.join("retirement").exists());
    assert!(!fixture.root.join("locks/activity").exists());
    assert_no_host_reservation(&memory);
}

#[test]
fn escaped_payload_clone_blocks_reclamation_after_snapshot_wrapper_drops() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &memory, NEW, None).unwrap());
    let lease = store.open_snapshot(&snapshot).unwrap();
    let file = lease.file(PAYLOAD_PATH).unwrap().try_clone().unwrap();
    drop(lease);
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Busy
    ));
    assert_eq!(file_bytes(&file), NEW);
    drop(file);
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Durable(_)
    ));
    assert_not_found(store.open_snapshot(&snapshot));
    assert_no_host_reservation(&memory);
}

#[test]
fn unrelated_live_first_seen_transaction_excludes_reclamation_until_drop() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    let snapshot = durable(publish(&store, &memory, OLD, None).unwrap());
    let mut transaction = store
        .begin_discovered(
            payload_manifest(NEW).source().clone(),
            vec![
                zerogravity_artifacts::cache::DiscoveredEntry::new(
                    PAYLOAD_PATH,
                    NEW.len() as u64,
                    zerogravity_artifacts::cache::FileRole::Auxiliary,
                    None,
                )
                .unwrap(),
            ],
            &memory,
            Arc::new(AtomicBool::new(false)),
        )
        .unwrap();
    let mut bytes = NEW;
    transaction.import_object(PAYLOAD_PATH, &mut bytes).unwrap();
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Busy
    ));
    drop(transaction);
    assert!(matches!(
        store
            .reclaim_snapshot(&snapshot, ReclaimMode::Apply, &AtomicBool::new(false))
            .unwrap(),
        ReclaimOutcome::Durable(_)
    ));
    assert_no_host_reservation(&memory);
}
