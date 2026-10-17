//! Synthetic verified cache-to-owned-source boundary. No model/provider/device.
#![cfg(all(feature = "cache", target_os = "linux"))]

use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use zerogravity_artifacts::artifact::{Artifact, ArtifactFormat};
use zerogravity_artifacts::cache::{
    CacheOptions, CacheStore, CommitOutcome, FileRole, Manifest, ManifestEntry, ManifestSource,
    SnapshotLease, TransferAdmission,
};
use zerogravity_artifacts::{Error, SourceWeights};
use zerogravity_core::resource_arbiter::{
    AllocationClass, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

const MODEL: &str = "original/model.gguf";
const CONFIG: &str = "original/config.json";
const TOKENIZER: &str = "original/tokenizer.json";
const CONFIG_BYTES: &[u8] = b"{\"unchanged\":true}\n";
const TOKENIZER_BYTES: &[u8] = b"{\"tokens\":[\"a\",\"b\"]}\n";

struct Fixture {
    base: PathBuf,
    root: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let base = std::env::temp_dir().join(format!(
            "zerogravity-cache-source-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::DirBuilder::new().mode(0o700).create(&base).unwrap();
        Self {
            root: base.join("cache"),
            base,
        }
    }
    fn open(&self) -> CacheStore {
        CacheStore::open(CacheOptions::new(Some(self.root.clone()), 8 * 1024 * 1024)).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.base);
    }
}
fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn gguf(name: &str, values: [f32; 2]) -> Vec<u8> {
    fn string(output: &mut Vec<u8>, value: &str) {
        output.extend((value.len() as u64).to_le_bytes());
        output.extend(value.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    bytes.extend(1u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8u32.to_le_bytes());
    string(&mut bytes, "fixture");
    string(&mut bytes, name);
    bytes.extend(1u32.to_le_bytes());
    bytes.extend(2u64.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    for value in values {
        bytes.extend(value.to_le_bytes());
    }
    bytes
}
fn admission() -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host", "source-bridge").unwrap();
    arbiter
        .register(resource.clone(), 128 * 1024 * 1024)
        .unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "source-bridge").unwrap(),
    }
}
fn denied_admission() -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("test-host", "source-denied").unwrap();
    arbiter.register(resource.clone(), 1).unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("fixture", "source-denied").unwrap(),
    }
}
fn publish(store: &CacheStore, model: &[u8], role: FileRole) -> (String, SnapshotLease) {
    let files = [
        (MODEL, model, role),
        (CONFIG, CONFIG_BYTES, FileRole::Config),
        (TOKENIZER, TOKENIZER_BYTES, FileRole::Tokenizer),
    ];
    let manifest = Manifest::new(
        ManifestSource::new("local", "synthetic-original", "immutable-v1").unwrap(),
        files
            .iter()
            .map(|(path, bytes, role)| {
                ManifestEntry::new(*path, hash(bytes), bytes.len() as u64, *role).unwrap()
            })
            .collect(),
        &Default::default(),
    )
    .unwrap();
    let digest = manifest.digest().unwrap();
    let mut transaction = store
        .begin(manifest, &admission(), Arc::new(AtomicBool::new(false)))
        .unwrap();
    for (path, mut bytes, _) in files {
        transaction.import_object(path, &mut bytes).unwrap();
    }
    assert!(matches!(
        transaction.commit(Some("source")).unwrap(),
        CommitOutcome::Durable(_)
    ));
    (digest, store.resolve_ref("source").unwrap())
}

#[test]
fn original_logical_format_and_actual_owned_bytes_reach_only_existing_source_issuer() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let bytes = gguf("value", [1., 2.]);
    let (_, lease) = publish(&store, &bytes, FileRole::Weights);
    let blob = fixture.root.join(format!("blobs/sha256/{}", hash(&bytes)));
    assert!(matches!(Artifact::open(&blob), Err(Error::Unsupported(_))));
    let source = lease.source_weights(MODEL, &admission()).unwrap();
    assert_eq!(source.architecture(), "fixture");
    assert_eq!(source.source_len(), bytes.len());
    assert_eq!(source.vector("value").unwrap(), [1., 2.]);
    let second = lease.source_weights(MODEL, &admission()).unwrap();
    assert_ne!(source.source_identity(), second.source_identity());
    assert_eq!(lease.read_verified(CONFIG, 4096).unwrap(), CONFIG_BYTES);
    assert_eq!(
        lease.read_verified(TOKENIZER, 4096).unwrap(),
        TOKENIZER_BYTES
    );
    assert!(matches!(
        lease.read_verified(CONFIG, 1),
        Err(Error::Limit(_))
    ));
    assert!(matches!(
        lease.source_weights(MODEL, &denied_admission()),
        Err(Error::Limit(_))
    ));
    assert!(lease.source_weights(CONFIG, &admission()).is_err());
    assert!(lease.read_verified("absent", 4096).is_err());
}

#[test]
fn original_open_descriptors_survive_object_and_root_path_replacement() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let original = gguf("value", [1., 2.]);
    let (snapshot, lease) = publish(&store, &original, FileRole::Weights);
    let replacement = gguf("value", [3., 4.]);
    let object = fixture
        .root
        .join(format!("snapshots/{snapshot}/files/{MODEL}"));
    let inode = lease.file(MODEL).unwrap().metadata().unwrap().ino();
    fs::rename(&object, fixture.base.join("retained-original")).unwrap();
    fs::write(&object, replacement).unwrap();
    fs::set_permissions(&object, fs::Permissions::from_mode(0o600)).unwrap();
    assert_ne!(fs::metadata(&object).unwrap().ino(), inode);
    assert_eq!(
        lease
            .source_weights(MODEL, &admission())
            .unwrap()
            .vector("value")
            .unwrap(),
        [1., 2.]
    );
    fs::rename(&fixture.root, fixture.base.join("pinned-root")).unwrap();
    fs::DirBuilder::new()
        .mode(0o700)
        .create(&fixture.root)
        .unwrap();
    assert_eq!(
        lease
            .source_weights(MODEL, &admission())
            .unwrap()
            .vector("value")
            .unwrap(),
        [1., 2.]
    );
    assert_eq!(lease.read_verified(CONFIG, 4096).unwrap(), CONFIG_BYTES);
}

#[test]
fn same_inode_changes_after_lease_verification_refuse_and_owned_copy_stays_immutable() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let original = gguf("value", [1., 2.]);
    let (_, lease) = publish(&store, &original, FileRole::Weights);
    let owned = lease.source_weights(MODEL, &admission()).unwrap();
    let config = lease.read_verified(CONFIG, 4096).unwrap();
    let model_blob = fixture
        .root
        .join(format!("blobs/sha256/{}", hash(&original)));
    fs::write(&model_blob, gguf("value", [3., 4.])).unwrap();
    assert!(matches!(
        lease.source_weights(MODEL, &admission()),
        Err(Error::Malformed(_))
    ));
    assert_eq!(owned.vector("value").unwrap(), [1., 2.]);
    let config_blob = fixture
        .root
        .join(format!("blobs/sha256/{}", hash(CONFIG_BYTES)));
    let mut changed = CONFIG_BYTES.to_vec();
    changed[2] = b'x';
    fs::write(&config_blob, changed).unwrap();
    assert!(matches!(
        lease.read_verified(CONFIG, 4096),
        Err(Error::Malformed(_))
    ));
    assert_eq!(config, CONFIG_BYTES);
    let tokenizer_blob = fixture
        .root
        .join(format!("blobs/sha256/{}", hash(TOKENIZER_BYTES)));
    fs::write(&tokenizer_blob, b"short").unwrap();
    assert!(matches!(
        lease.read_verified(TOKENIZER, 4096),
        Err(Error::Malformed(_))
    ));
}

#[test]
fn verified_copy_reparses_actual_bytes_instead_of_original_inspection_metadata() {
    let fixture = Fixture::new();
    let path = fixture.base.join("opened");
    let before = gguf("first", [1., 2.]);
    let after = gguf("other", [3., 4.]);
    assert_eq!(before.len(), after.len());
    fs::write(&path, &before).unwrap();
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("original.gguf")).unwrap();
    assert_eq!(artifact.info.tensors[0].name, "first");
    fs::write(&path, &after).unwrap();
    let source = SourceWeights::from_verified_artifact(
        artifact,
        after.len() as u64,
        &hash(&after),
        &admission(),
    )
    .unwrap();
    assert!(!source.contains("first"));
    assert_eq!(source.vector("other").unwrap(), [3., 4.]);
}

#[test]
fn positioned_parse_and_copy_do_not_share_cursor_even_during_concurrent_seeks() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let (_, lease) = publish(&store, &gguf("value", [1., 2.]), FileRole::Weights);
    let mut descriptor = lease.file(MODEL).unwrap().try_clone().unwrap();
    descriptor.seek(SeekFrom::Start(7)).unwrap();
    lease.source_weights(MODEL, &admission()).unwrap();
    assert_eq!(descriptor.stream_position().unwrap(), 7);
    let mut config_descriptor = lease.file(CONFIG).unwrap().try_clone().unwrap();
    config_descriptor.seek(SeekFrom::Start(3)).unwrap();
    lease.read_verified(CONFIG, 4096).unwrap();
    assert_eq!(config_descriptor.stream_position().unwrap(), 3);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            for offset in 0..2000 {
                descriptor.seek(SeekFrom::Start(offset % 13)).unwrap();
            }
        });
        scope.spawn(|| {
            for offset in 0..2000 {
                config_descriptor.seek(SeekFrom::Start(offset % 7)).unwrap();
            }
        });
        for _ in 0..4 {
            scope.spawn(|| {
                for _ in 0..50 {
                    assert_eq!(
                        lease
                            .source_weights(MODEL, &admission())
                            .unwrap()
                            .vector("value")
                            .unwrap(),
                        [1., 2.]
                    );
                    assert_eq!(lease.read_verified(CONFIG, 4096).unwrap(), CONFIG_BYTES);
                }
            });
        }
    });
}

#[test]
fn source_caps_format_roles_length_and_digest_never_gain_loading_support() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let original = gguf("value", [1., 2.]);
    let (_, lease) = publish(&store, &original, FileRole::Auxiliary);
    assert!(matches!(
        lease.source_weights(MODEL, &admission()),
        Err(Error::Unsupported(_))
    ));
    let path = fixture.base.join("source");
    fs::write(&path, &original).unwrap();
    for (length, digest) in [
        (original.len() as u64 + 1, hash(&original)),
        (original.len() as u64, "0".repeat(64)),
        (original.len() as u64, "invalid".into()),
    ] {
        let artifact =
            Artifact::from_file(File::open(&path).unwrap(), Path::new("model.gguf")).unwrap();
        assert!(
            SourceWeights::from_verified_artifact(artifact, length, &digest, &admission()).is_err()
        );
    }
    assert!(
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).is_err()
    );
    assert!(
        Artifact::from_file(File::open(&fixture.base).unwrap(), Path::new("model.gguf")).is_err()
    );
    let mut large = fs::OpenOptions::new().write(true).open(&path).unwrap();
    large.set_len(4 * 1024 * 1024 * 1024 + 1).unwrap();
    large.flush().unwrap();
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.gguf")).unwrap();
    assert_eq!(artifact.info.format, ArtifactFormat::Gguf);
    assert!(matches!(
        SourceWeights::from_verified_artifact(
            artifact,
            4 * 1024 * 1024 * 1024 + 1,
            &"0".repeat(64),
            &admission()
        ),
        Err(Error::Limit(_))
    ));
}

#[test]
fn raw_owned_source_reservation_precedes_copy_and_lasts_until_all_model_owners_drop() {
    let fixture = Fixture::new();
    let store = fixture.open();
    let bytes = gguf("value", [1., 2.]);
    let (_, lease) = publish(&store, &bytes, FileRole::Weights);
    let admission = admission();
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .reserved,
        0
    );
    let source = Arc::new(lease.source_weights(MODEL, &admission).unwrap());
    let reservation = admission.arbiter.snapshot(&admission.resource).unwrap();
    assert_eq!(reservation.reserved, bytes.len() as u64);
    assert_eq!(reservation.reservations, 1);
    assert_eq!(
        reservation.per_class[&AllocationClass::ImmutableWeights],
        bytes.len() as u64
    );
    assert!(!reservation.fenced);
    drop(lease);
    let retained = source.clone();
    drop(source);
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .reserved,
        bytes.len() as u64
    );
    assert_eq!(retained.vector("value").unwrap(), [1., 2.]);
    drop(retained);
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .reserved,
        0
    );
    let lease = store.resolve_ref("source").unwrap();
    let denied = denied_admission();
    let mut descriptor = lease.file(MODEL).unwrap().try_clone().unwrap();
    descriptor.seek(SeekFrom::Start(11)).unwrap();
    assert!(matches!(
        lease.source_weights(MODEL, &denied),
        Err(Error::Limit(_))
    ));
    assert_eq!(descriptor.stream_position().unwrap(), 11);
    assert_eq!(
        denied.arbiter.snapshot(&denied.resource).unwrap().reserved,
        0
    );
    let blob = fixture.root.join(format!("blobs/sha256/{}", hash(&bytes)));
    fs::write(blob, gguf("value", [3., 4.])).unwrap();
    assert!(lease.source_weights(MODEL, &admission).is_err());
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .reserved,
        0
    );
}
