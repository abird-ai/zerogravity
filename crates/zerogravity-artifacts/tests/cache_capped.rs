//! Capped data uses the original admitted attempt and exact final publisher.
#![cfg(all(feature = "cache", target_os = "linux"))]
#[allow(dead_code)]
mod cache_transaction_support;

use cache_transaction_support::*;
use std::fs;
use std::io::{self, Cursor, Read};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::Error;
use zerogravity_artifacts::cache::{
    ByteLength, CacheStore, DigestProvenance, DiscoveredEntry, DiscoveredObject,
    DiscoveredTransaction, FileRole, Manifest, ManifestEntry, TransferAdmission,
};

fn entry(path: &str, length: ByteLength, hash: Option<String>) -> DiscoveredEntry {
    DiscoveredEntry::new_with_length(path, length, FileRole::Auxiliary, hash).unwrap()
}

fn begin(
    store: &CacheStore,
    admission: &TransferAdmission,
    entries: Vec<DiscoveredEntry>,
    cancel: Arc<AtomicBool>,
) -> DiscoveredTransaction {
    store
        .begin_discovered(
            payload_manifest(NEW).source().clone(),
            entries,
            admission,
            cancel,
        )
        .unwrap()
}

struct ObservedReader<'a> {
    bytes: Cursor<&'a [u8]>,
    read_bytes: usize,
    eof_reads: usize,
    cancel: Option<&'a AtomicBool>,
}

impl<'a> ObservedReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes: Cursor::new(bytes),
            read_bytes: 0,
            eof_reads: 0,
            cancel: None,
        }
    }
}

impl Read for ObservedReader<'_> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        let count = self.bytes.read(output)?;
        self.read_bytes += count;
        self.eof_reads += usize::from(count == 0);
        if let Some(cancel) = self.cancel {
            cancel.store(true, Ordering::Release);
        }
        Ok(count)
    }
}

#[test]
fn actual_eof_lengths_and_hashes_replace_caps_including_empty_and_exact_cap() {
    for (bytes, length) in [
        (&b""[..], ByteLength::Exact(0)),
        (&b""[..], ByteLength::AtMost(1000)),
        (&b"x"[..], ByteLength::AtMost(1)),
        (NEW, ByteLength::AtMost(1000)),
    ] {
        let fixture = Fixture::new();
        let store = fixture.open();
        let memory = admission(256 * 1024 * 1024);
        let mut tx = begin(
            &store,
            &memory,
            vec![entry(PAYLOAD_PATH, length, None)],
            Arc::new(AtomicBool::new(false)),
        );
        let mut reader = ObservedReader::new(bytes);
        let actual = tx.import_object(PAYLOAD_PATH, &mut reader).unwrap();
        assert_eq!(actual.bytes(), bytes.len() as u64);
        assert_eq!(actual.digest(), digest(bytes));
        assert_eq!(actual.digest_provenance(), DigestProvenance::FirstSeen);
        assert_eq!(reader.read_bytes, bytes.len());
        assert_eq!(reader.eof_reads, 1);
        assert_eq!(tx.manifest().unwrap().total_bytes(), bytes.len() as u64);
        let snapshot = durable(tx.commit(Some("actual")).unwrap());
        assert_payload(&store.open_snapshot(&snapshot).unwrap(), bytes);
        assert_eq!(store.usage().unwrap().reserved_bytes, 0);
        assert_no_host_reservation(&memory);
    }
}

#[test]
fn overflow_exact_truncation_wrong_digest_and_cancel_never_replace_old_reference() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    let previous = durable(publish(&store, &memory, OLD, Some("fixture")).unwrap());
    for (length, hash, canceled) in [
        (ByteLength::AtMost(2), None, false),
        (ByteLength::Exact(100), None, false),
        (ByteLength::AtMost(100), Some(digest(b"wrong")), false),
        (ByteLength::AtMost(100), None, true),
    ] {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut tx = begin(
            &store,
            &memory,
            vec![entry(PAYLOAD_PATH, length, hash)],
            cancel.clone(),
        );
        let mut reader = ObservedReader::new(NEW);
        reader.cancel = canceled.then_some(&cancel);
        assert!(tx.import_object(PAYLOAD_PATH, &mut reader).is_err());
        if length == ByteLength::AtMost(2) {
            assert_eq!(reader.read_bytes, 3);
        }
        assert!(tx.manifest().is_err());
        drop(tx);
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
        assert_no_host_reservation(&memory);
    }
}

#[test]
fn capped_guard_requires_eof_attestation_and_never_saves_or_recovers_a_partial() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    {
        let mut tx = begin(
            &store,
            &memory,
            vec![entry(PAYLOAD_PATH, ByteLength::Exact(64), None)],
            Arc::new(AtomicBool::new(false)),
        );
        let mut guard = tx
            .partial_object(PAYLOAD_PATH, "same-immutable-object")
            .unwrap();
        guard.set_validator("\"stable\"").unwrap();
        guard.write_chunk(b"old prefix").unwrap();
        guard.checkpoint().unwrap();
    }
    let retained = store.usage().unwrap().used_bytes;
    {
        let mut tx = begin(
            &store,
            &memory,
            vec![entry(PAYLOAD_PATH, ByteLength::AtMost(64), None)],
            Arc::new(AtomicBool::new(false)),
        );
        let mut guard = tx
            .partial_object(PAYLOAD_PATH, "same-immutable-object")
            .unwrap();
        assert_eq!(guard.resume_bytes(), 0);
        assert!(guard.validator().is_none());
        assert!(matches!(
            guard.set_validator("\"stable\""),
            Err(Error::Unsupported(_))
        ));
        guard.write_chunk(NEW).unwrap();
        assert!(matches!(guard.checkpoint(), Err(Error::Unsupported(_))));
        assert!(matches!(guard.finish(), Err(Error::Unsupported(_))));
        assert!(tx.manifest().is_err());
    }
    // Stable attempt-lock inodes remain charged even after ephemeral cleanup.
    assert!((retained..retained + 64 * 1024).contains(&store.usage().unwrap().used_bytes));
    let partials = fs::read_dir(fixture.root.join("staging"))
        .unwrap()
        .flat_map(|attempt| fs::read_dir(attempt.unwrap().path()).unwrap())
        .filter(|file| file.as_ref().unwrap().file_name() == "partial-0.json")
        .count();
    assert_eq!(partials, 1);
    let mut tx = begin(
        &store,
        &memory,
        vec![entry(PAYLOAD_PATH, ByteLength::AtMost(64), None)],
        Arc::new(AtomicBool::new(false)),
    );
    let mut guard = tx
        .partial_object(PAYLOAD_PATH, "same-immutable-object")
        .unwrap();
    assert_eq!(guard.resume_bytes(), 0);
    let mut body = Cursor::new(NEW);
    let mut body_bytes = [0; NEW.len()];
    body.read_exact(&mut body_bytes).unwrap();
    guard.write_chunk(&body_bytes).unwrap();
    // A real executor performs this check on its original terminal body reader.
    assert_eq!(body.read(&mut [0]).unwrap(), 0);
    let actual = guard.finish_after_eof().unwrap();
    assert_eq!(actual.bytes(), NEW.len() as u64);
    tx.commit(None).unwrap();
    assert_no_host_reservation(&memory);
}

#[test]
fn known_cache_hits_verify_actual_length_and_hash_without_reading_another_body() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    publish(&store, &memory, NEW, None).unwrap();
    for bound in [
        ByteLength::AtMost(1000),
        ByteLength::AtMost(NEW.len() as u64),
    ] {
        let mut tx = begin(
            &store,
            &memory,
            vec![entry(PAYLOAD_PATH, bound, Some(digest(NEW)))],
            Arc::new(AtomicBool::new(false)),
        );
        let cached = tx.try_cached(PAYLOAD_PATH).unwrap().unwrap();
        assert_eq!(cached.bytes(), NEW.len() as u64);
        assert_eq!(cached.digest_provenance(), DigestProvenance::TrustedSha256);
        tx.commit(None).unwrap();
    }
    let mut tx = begin(
        &store,
        &memory,
        vec![entry(
            PAYLOAD_PATH,
            ByteLength::AtMost(1),
            Some(digest(NEW)),
        )],
        Arc::new(AtomicBool::new(false)),
    );
    let mut reader = ObservedReader::new(NEW);
    assert!(matches!(
        tx.import_object(PAYLOAD_PATH, &mut reader),
        Err(Error::Malformed(_))
    ));
    assert_eq!(reader.read_bytes, 0);
    drop(tx);
    fs::write(
        fixture.root.join("blobs/sha256").join(digest(NEW)),
        vec![b'x'; NEW.len()],
    )
    .unwrap();
    let mut tx = begin(
        &store,
        &memory,
        vec![entry(
            PAYLOAD_PATH,
            ByteLength::AtMost(1000),
            Some(digest(NEW)),
        )],
        Arc::new(AtomicBool::new(false)),
    );
    assert!(matches!(
        tx.try_cached(PAYLOAD_PATH),
        Err(Error::Malformed(_))
    ));
    drop(tx);
    assert_no_host_reservation(&memory);
}

#[test]
fn unequal_trusted_caps_reserve_the_maximum_and_unknown_hashes_remain_independent() {
    const MIB: u64 = 1024 * 1024;
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * MIB);
    let before = store.usage().unwrap().used_bytes;
    let tx = begin(
        &store,
        &memory,
        vec![
            entry("a-small", ByteLength::AtMost(MIB), Some(digest(NEW))),
            entry("z-large", ByteLength::AtMost(5 * MIB), Some(digest(NEW))),
        ],
        Arc::new(AtomicBool::new(false)),
    );
    let usage = store.usage().unwrap();
    let peak = usage.used_bytes + usage.reserved_bytes - before;
    assert!(
        (5 * MIB..5 * MIB + 512 * 1024).contains(&peak),
        "wrong grouped peak {peak}"
    );
    let other = Manifest::new(
        payload_manifest(NEW).source().clone(),
        vec![ManifestEntry::new("other", digest(b"other"), 4 * MIB, FileRole::Auxiliary).unwrap()],
        &Default::default(),
    )
    .unwrap();
    assert!(matches!(
        store.begin(other.clone(), &memory, Arc::new(AtomicBool::new(false))),
        Err(Error::Limit(_))
    ));
    drop(tx);
    drop(
        store
            .begin(other, &memory, Arc::new(AtomicBool::new(false)))
            .unwrap(),
    );
    assert!(matches!(
        store.begin_discovered(
            payload_manifest(NEW).source().clone(),
            vec![
                entry("first", ByteLength::AtMost(4 * MIB), None),
                entry("second", ByteLength::AtMost(4 * MIB), None)
            ],
            &memory,
            Arc::new(AtomicBool::new(false))
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(store.usage().unwrap().reserved_bytes, 0);
    assert_no_host_reservation(&memory);
}

#[test]
fn conflicting_trusted_exact_lengths_and_caps_refuse_before_staging() {
    assert!(matches!(
        DiscoveredEntry::new_with_length(
            "empty-cap",
            ByteLength::AtMost(0),
            FileRole::Auxiliary,
            None
        ),
        Err(Error::Limit(_))
    ));
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    for second in [ByteLength::Exact(7), ByteLength::AtMost(7)] {
        assert!(matches!(
            store.begin_discovered(
                payload_manifest(NEW).source().clone(),
                vec![
                    entry("a", ByteLength::Exact(8), Some(digest(NEW))),
                    entry("b", second, Some(digest(NEW)))
                ],
                &memory,
                Arc::new(AtomicBool::new(false))
            ),
            Err(Error::Malformed(_))
        ));
        assert_eq!(
            fs::read_dir(fixture.root.join("staging")).unwrap().count(),
            0
        );
        assert_no_host_reservation(&memory);
    }
}

#[test]
fn trusted_guard_fill_uses_actual_length_for_another_cap_and_legacy_exact_stays_strict() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let memory = admission(256 * 1024 * 1024);
    let mut tx = begin(
        &store,
        &memory,
        vec![
            entry("a", ByteLength::AtMost(1000), Some(digest(NEW))),
            entry("b", ByteLength::AtMost(2000), Some(digest(NEW))),
        ],
        Arc::new(AtomicBool::new(false)),
    );
    let mut first = tx.partial_object("a", "immutable-object").unwrap();
    first.write_chunk(NEW).unwrap();
    first.finish_after_eof().unwrap();
    assert!(
        matches!(tx.cached_or_partial_object("b", "immutable-object").unwrap(), DiscoveredObject::Cached(actual) if actual.bytes() == NEW.len() as u64)
    );
    assert_eq!(tx.manifest().unwrap().total_bytes(), 2 * NEW.len() as u64);
    tx.commit(None).unwrap();
    let mut exact = begin(
        &store,
        &memory,
        vec![entry(
            PAYLOAD_PATH,
            ByteLength::Exact(1000),
            Some(digest(NEW)),
        )],
        Arc::new(AtomicBool::new(false)),
    );
    assert!(matches!(
        exact.try_cached(PAYLOAD_PATH),
        Err(Error::Malformed(_))
    ));
    drop(exact);
    assert_no_host_reservation(&memory);
}
