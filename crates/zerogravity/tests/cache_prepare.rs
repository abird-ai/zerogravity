#![cfg(all(
    feature = "cache",
    feature = "cpu",
    feature = "model-qwen35",
    feature = "text",
    target_os = "linux"
))]
//! Tiny ordinary gates plus explicitly ignored, original-checkpoint A3 parity.
use std::fs::{File, OpenOptions};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use zerogravity::cache::{
    CacheLimits, CacheOptions, CacheStore, FileRole, Manifest, ManifestEntry, ManifestSource,
    TransferAdmission,
};
use zerogravity::prepare::PreparedQwen;
use zerogravity::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
fn admission(bytes: u64) -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host", "cache-prepare").unwrap();
    arbiter.register(resource.clone(), bytes).unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "prepare").unwrap(),
    }
}
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::temp_dir().join(format!(
            "zg-a3-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        assert!(!root.exists());
        Self(root)
    }
    fn store(&self) -> CacheStore {
        CacheStore::open(CacheOptions::new(
            Some(self.0.clone()),
            16 * 1024 * 1024 * 1024,
        ))
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn role_and_staging_admission_fail_without_model_source_allocation() {
    let fixture = Fixture::new();
    let store = fixture.store();
    let memory = admission(128 * 1024 * 1024);
    let entries = [
        ("model.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
    ]
    .into_iter()
    .map(|(path, role)| ManifestEntry::new(path, ABC, 3, role).unwrap())
    .collect();
    let manifest = Manifest::new(
        ManifestSource::new("local", "fixture", "v1").unwrap(),
        entries,
        &CacheLimits::default(),
    )
    .unwrap();
    let mut tx = store
        .begin(manifest, &memory, Arc::new(AtomicBool::new(false)))
        .unwrap();
    for path in ["model.gguf", "config.json", "tokenizer.json"] {
        tx.import_object(path, &mut Cursor::new(b"abc")).unwrap();
    }
    tx.commit(Some("fixture")).unwrap();
    let snapshot = store.resolve_ref("fixture").unwrap();
    let denied = admission(1);
    assert!(matches!(
        PreparedQwen::from_snapshots(
            &snapshot,
            "model.gguf",
            &snapshot,
            "config.json",
            &snapshot,
            "tokenizer.json",
            &denied
        ),
        Err(zerogravity::Error::Limit(_))
    ));
    assert_eq!(
        denied.arbiter.snapshot(&denied.resource).unwrap().reserved,
        0
    );
    assert!(matches!(
        PreparedQwen::from_snapshots(
            &snapshot,
            "config.json",
            &snapshot,
            "config.json",
            &snapshot,
            "tokenizer.json",
            &memory
        ),
        Err(zerogravity::Error::Malformed(_))
    ));
    assert!(
        PreparedQwen::from_snapshots(
            &snapshot,
            "model.gguf",
            &snapshot,
            "config.json",
            &snapshot,
            "tokenizer.json",
            &memory
        )
        .is_err()
    );
    assert_eq!(
        memory.arbiter.snapshot(&memory.resource).unwrap().reserved,
        0
    );
}

fn import_real(
    store: &CacheStore,
    original: &Path,
    hashes: [&str; 3],
    memory: &TransferAdmission,
) -> String {
    let paths = [
        ("model.gguf", FileRole::Weights),
        ("config.json", FileRole::Config),
        ("tokenizer.json", FileRole::Tokenizer),
    ];
    let entries = paths
        .iter()
        .zip(hashes)
        .map(|((path, role), hash)| {
            ManifestEntry::new(
                *path,
                hash,
                std::fs::metadata(original.join(path)).unwrap().len(),
                *role,
            )
            .unwrap()
        })
        .collect();
    let manifest = Manifest::new(
        ManifestSource::new("local", "approved/qwen35", hashes[0]).unwrap(),
        entries,
        &CacheLimits::default(),
    )
    .unwrap();
    let digest = manifest.digest().unwrap();
    let mut tx = store
        .begin(manifest, memory, Arc::new(AtomicBool::new(false)))
        .unwrap();
    for (path, _) in paths {
        tx.import_object(path, &mut File::open(original.join(path)).unwrap())
            .unwrap();
    }
    tx.commit(Some("model")).unwrap();
    digest
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "requires the approved original Qwen3.5 .8B/2B GGUF/config/tokenizers; sequential CPU only"]
fn approved_cached_models_match_direct_source_exactly() {
    for (name, model_hash, config_hash) in [
        (
            "qwen35-0.8b",
            "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
            "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        ),
        (
            "qwen35-2b",
            "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
            "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
        ),
    ] {
        let original = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/models")
            .join(name);
        let fixture = Fixture::new();
        let store = fixture.store();
        let memory = admission(5 * 1024 * 1024 * 1024);
        let digest = import_real(
            &store,
            &original,
            [
                model_hash,
                config_hash,
                "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
            ],
            &memory,
        );
        let lease = store.resolve_ref("model").unwrap();
        for path in ["config.json", "tokenizer.json"] {
            let entry = lease
                .manifest()
                .entries()
                .iter()
                .find(|entry| entry.path() == path)
                .unwrap()
                .clone();
            let sidecar = Manifest::new(
                ManifestSource::new("local", format!("approved/{path}"), entry.digest()).unwrap(),
                vec![entry],
                &CacheLimits::default(),
            )
            .unwrap();
            let mut tx = store
                .begin(sidecar, &memory, Arc::new(AtomicBool::new(false)))
                .unwrap();
            tx.import_object(path, &mut File::open(original.join(path)).unwrap())
                .unwrap();
            tx.commit(Some(path)).unwrap();
        }
        let config_lease = store.resolve_ref("config.json").unwrap();
        let tokenizer_lease = store.resolve_ref("tokenizer.json").unwrap();
        assert_ne!(
            config_lease.manifest().digest().unwrap(),
            lease.manifest().digest().unwrap()
        );
        assert_ne!(
            tokenizer_lease.manifest().digest().unwrap(),
            config_lease.manifest().digest().unwrap()
        );
        let denied = admission(64 * 1024 * 1024);
        assert!(matches!(
            PreparedQwen::from_snapshots(
                &lease,
                "model.gguf",
                &config_lease,
                "config.json",
                &tokenizer_lease,
                "tokenizer.json",
                &denied
            ),
            Err(zerogravity::Error::Limit(_))
        ));
        assert_eq!(
            denied.arbiter.snapshot(&denied.resource).unwrap().reserved,
            0
        );
        let cached = PreparedQwen::from_snapshots(
            &lease,
            "model.gguf",
            &config_lease,
            "config.json",
            &tokenizer_lease,
            "tokenizer.json",
            &memory,
        )
        .unwrap();
        let direct = PreparedQwen::from_paths(
            &original.join("model.gguf"),
            &original.join("config.json"),
            &original.join("tokenizer.json"),
        )
        .unwrap();
        assert_eq!(cached.model().geometry().vocab_size, 248320);
        assert_eq!(
            cached.model().source_bytes() as u64,
            std::fs::metadata(original.join("model.gguf"))
                .unwrap()
                .len()
        );
        assert_ne!(
            cached.model().source_identity(),
            direct.model().source_identity()
        );
        assert_eq!(
            cached.tokenizer().prompt_tokens("Hi", false).unwrap(),
            direct.tokenizer().prompt_tokens("Hi", false).unwrap()
        );
        assert_eq!(
            memory.arbiter.snapshot(&memory.resource).unwrap().reserved,
            cached.model().source_bytes() as u64
        );
        let retained_model = cached.model().clone();
        let mut actual = cached.model().new_session().unwrap();
        let sibling = cached.model().new_session().unwrap();
        let sibling_state = sibling.state_fingerprint();
        let mut expected = direct.model().new_session().unwrap();
        // Retarget the cache alias to independent metadata-only data; opened
        // lease and prepared immutable source remain the original identities.
        let sidecar = Manifest::new(
            ManifestSource::new("local", "approved/sidecar", config_hash).unwrap(),
            vec![
                lease
                    .manifest()
                    .entries()
                    .iter()
                    .find(|e| e.path() == "config.json")
                    .unwrap()
                    .clone(),
            ],
            &CacheLimits::default(),
        )
        .unwrap();
        let mut update = store
            .begin(sidecar, &memory, Arc::new(AtomicBool::new(false)))
            .unwrap();
        update
            .import_object(
                "config.json",
                &mut File::open(original.join("config.json")).unwrap(),
            )
            .unwrap();
        update.commit(Some("model")).unwrap();
        assert_ne!(
            store
                .resolve_ref("model")
                .unwrap()
                .manifest()
                .digest()
                .unwrap(),
            digest
        );
        assert_eq!(lease.manifest().digest().unwrap(), digest);
        drop(lease);
        drop(config_lease);
        drop(tokenizer_lease);
        // Mutate only the newly imported PRIVATE cache copy, never originals.
        // Opened immutable model bytes must remain independent of cache inodes.
        OpenOptions::new()
            .write(true)
            .open(fixture.0.join("blobs/sha256").join(model_hash))
            .unwrap()
            .write_all(b"X")
            .unwrap();
        assert!(store.open_snapshot(&digest).is_err());
        drop(cached);
        assert_eq!(
            memory.arbiter.snapshot(&memory.resource).unwrap().reserved,
            retained_model.source_bytes() as u64
        );
        for (position, token) in [11, 353].into_iter().enumerate() {
            let actual_logits = actual.forward(token).unwrap();
            let expected_logits = expected.forward(token).unwrap();
            assert_eq!(actual_logits.len(), 248320);
            assert_eq!(expected_logits.len(), 248320);
            assert_eq!(
                bits(&actual_logits),
                bits(&expected_logits),
                "{name} token{token} full vocabulary"
            );
            assert_eq!(actual.position(), position + 1);
            assert_eq!(expected.position(), position + 1);
            let actual_domains = actual.snapshot().domain_values();
            let expected_domains = expected.snapshot().domain_values();
            assert_eq!(actual_domains.len(), 42);
            assert_eq!(actual_domains.len(), expected_domains.len());
            for ((a_slot, a), (e_slot, e)) in actual_domains.iter().zip(&expected_domains) {
                assert_eq!(a_slot, e_slot);
                assert_eq!(bits(a), bits(e), "{name} {a_slot:?}");
            }
            assert_eq!(sibling.position(), 0);
            assert_eq!(sibling.root_id(), 0);
            assert_eq!(sibling.state_fingerprint(), sibling_state);
        }
        assert!(
            actual
                .snapshot()
                .domain_values()
                .iter()
                .any(|(_, v)| v.iter().any(|v| *v != 0.0))
        );
        assert!(expected.restore(&actual.snapshot()).is_err());
        drop(actual);
        drop(sibling);
        assert_eq!(
            memory.arbiter.snapshot(&memory.resource).unwrap().reserved,
            retained_model.source_bytes() as u64
        );
        drop(retained_model);
        assert_eq!(
            memory.arbiter.snapshot(&memory.resource).unwrap().reserved,
            0
        );
        println!(
            "A3_CACHE_CPU_PASS model={name} tokens=2 full_vocab=exact all42domains=exact independent_sessions=1 rawlease_lifetime=exact mutation_isolated=1"
        );
    }
}
