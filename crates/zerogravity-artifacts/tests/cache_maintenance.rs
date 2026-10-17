//! Existing cache admission/publication, with synthetic sources only.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::Error;
use zerogravity_artifacts::cache::{
    DiscoveredEntry, DiscoveredObject, DiscoveredTransaction, FileRole,
};

fn begin(
    store: &zerogravity_artifacts::cache::CacheStore,
    admission: &zerogravity_artifacts::cache::TransferAdmission,
    cancel: Arc<AtomicBool>,
) -> DiscoveredTransaction {
    store
        .begin_discovered(
            payload_manifest(NEW).source().clone(),
            vec![
                DiscoveredEntry::new(
                    PAYLOAD_PATH,
                    NEW.len() as u64,
                    FileRole::Auxiliary,
                    Some(digest(NEW)),
                )
                .unwrap(),
            ],
            admission,
            cancel,
        )
        .unwrap()
}

#[test]
fn fill_after_initial_miss_becomes_verified_hit_without_private_payload() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
    assert!(transaction.try_cached(PAYLOAD_PATH).unwrap().is_none());
    publish(&store, &admission, NEW, None).unwrap();
    let actual = match transaction
        .cached_or_partial_object(PAYLOAD_PATH, "immutable-object")
        .unwrap()
    {
        DiscoveredObject::Cached(entry) => entry,
        DiscoveredObject::Partial(_) => {
            panic!("verified fill must not allocate another private payload")
        }
    };
    assert_eq!(actual.digest(), digest(NEW));
    assert_eq!(transaction.manifest().unwrap().entries(), &[actual]);
    let mut prefix = [0; 128];
    let count = transaction.read_prefix(PAYLOAD_PATH, &mut prefix).unwrap();
    assert_eq!(&prefix[..count], NEW);
    for attempt in fs::read_dir(fixture.root.join("staging")).unwrap() {
        assert!(!attempt.unwrap().path().join("object-0").exists());
    }
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    transaction.commit(Some("fixture")).unwrap();
    assert_payload(&store.resolve_ref("fixture").unwrap(), NEW);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn legacy_partial_refusal_and_corrupt_fill_preserve_transaction_and_reference() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    publish(&store, &admission, NEW, None).unwrap();
    {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        assert!(matches!(
            transaction.partial_object(PAYLOAD_PATH, "immutable-object"),
            Err(Error::State(_))
        ));
        assert!(transaction.try_cached(PAYLOAD_PATH).unwrap().is_some());
    }
    fs::write(
        fixture.root.join(format!("blobs/sha256/{}", digest(NEW))),
        vec![b'x'; NEW.len()],
    )
    .unwrap();
    {
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        assert!(matches!(
            transaction.cached_or_partial_object(PAYLOAD_PATH, "immutable-object"),
            Err(Error::Malformed(_))
        ));
        assert!(transaction.manifest().is_err());
    }
    assert_eq!(
        store
            .resolve_ref("fixture")
            .unwrap()
            .manifest()
            .digest()
            .unwrap(),
        previous
    );
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[test]
fn canceled_transaction_and_snapshot_lookups_do_not_publish_or_mask_legacy_apis() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let admission = admission(256 * 1024 * 1024);
    let previous = durable(publish(&store, &admission, OLD, Some("fixture")).unwrap());
    let cancel = Arc::new(AtomicBool::new(false));
    let mut transaction = begin(&store, &admission, cancel.clone());
    cancel.store(true, Ordering::Release);
    assert!(matches!(
        transaction.try_cached(PAYLOAD_PATH),
        Err(Error::State(_))
    ));
    assert!(matches!(
        transaction.cached_or_partial_object(PAYLOAD_PATH, "immutable-object"),
        Err(Error::State(_))
    ));
    drop(transaction);
    assert!(matches!(
        store.open_snapshot_cancellable(&previous, &cancel),
        Err(Error::State(_))
    ));
    assert!(matches!(
        store.resolve_ref_cancellable("fixture", &cancel),
        Err(Error::State(_))
    ));
    assert_payload(&store.open_snapshot(&previous).unwrap(), OLD);
    assert_payload(&store.resolve_ref("fixture").unwrap(), OLD);
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&admission);
}

#[cfg(feature = "diagnostic-cache")]
mod concurrent {
    use super::*;
    use std::sync::{Mutex, mpsc};
    use std::thread;
    use std::time::Duration;
    use zerogravity_artifacts::Result;
    use zerogravity_artifacts::cache::{CacheHook, CacheMoment, CacheOperation, CachePoint};

    struct Pause {
        ready: mpsc::Sender<()>,
        release: Mutex<mpsc::Receiver<()>>,
    }
    impl CacheHook for Pause {
        fn point(&self, point: CachePoint) -> Result<()> {
            if point.operation == CacheOperation::ObjectFileSync
                && point.moment == CacheMoment::Before
            {
                self.ready.send(()).unwrap();
                self.release
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))
                    .unwrap();
            }
            Ok(())
        }
    }

    #[test]
    fn waited_known_publisher_returns_cached_result_and_canceled_waiter_leaves_owner_live() {
        let fixture = Fixture::new();
        let store = fixture.open();
        let admission = admission(384 * 1024 * 1024);
        let mut transaction = begin(&store, &admission, Arc::new(AtomicBool::new(false)));
        assert!(transaction.try_cached(PAYLOAD_PATH).unwrap().is_none());
        let (ready, receiver) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let owner = store.clone().with_hook(Arc::new(Pause {
            ready,
            release: Mutex::new(released),
        }));
        let publisher_admission = admission.clone();
        thread::scope(|scope| {
            let publisher =
                scope.spawn(|| publish(&owner, &publisher_admission, NEW, None).unwrap());
            receiver.recv_timeout(Duration::from_secs(10)).unwrap();
            let cancel = Arc::new(AtomicBool::new(false));
            let mut waiter = begin(&store, &admission, cancel.clone());
            let waiter = scope.spawn(move || {
                assert!(matches!(
                    waiter.cached_or_partial_object(PAYLOAD_PATH, "other-immutable-object"),
                    Err(Error::State(_))
                ));
            });
            thread::sleep(Duration::from_millis(20));
            cancel.store(true, Ordering::Release);
            waiter.join().unwrap();
            assert!(store.usage().unwrap().reserved_bytes > 0);
            let releaser = scope.spawn(move || {
                thread::sleep(Duration::from_millis(20));
                release.send(()).unwrap();
            });
            let result = transaction
                .cached_or_partial_object(PAYLOAD_PATH, "immutable-object")
                .unwrap();
            assert!(matches!(result, DiscoveredObject::Cached(_)));
            releaser.join().unwrap();
            publisher.join().unwrap();
        });
        transaction.commit(None).unwrap();
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&admission);
    }
}
