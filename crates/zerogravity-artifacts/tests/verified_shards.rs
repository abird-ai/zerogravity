//! Source-only multi-shard loading / physical reservation contract.
//! NO trained weights, network, GPU or family-specific math.
#![cfg(all(feature = "cache", target_os = "linux"))]

use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::source_weights::{ExpectedSourceShard, SourceOpenLimits};
use zerogravity_artifacts::{Error, SourceWeights};
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
use zerogravity_core::source::TensorSource;

const A: &str = "model-00001-of-00002.safetensors";
const B: &str = "model-00002-of-00002.safetensors";

struct Fixture {
    path: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-source-shards-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self { path }
    }
    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.path.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn safetensors(name: &str, dtype: &str, shape: &[u64], payload: &[u8]) -> Vec<u8> {
    let header = serde_json::to_vec(&serde_json::json!({
        name:{"dtype":dtype,"shape":shape,"data_offsets":[0,payload.len() as u64]}
    }))
    .unwrap();
    let mut out = Vec::new();
    out.extend((header.len() as u64).to_le_bytes());
    out.extend(header);
    out.extend(payload);
    out
}
fn index(first_name: &str, second_name: &str, total: u64) -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({
        "metadata":{"total_size":total},
        "weight_map": {first_name:A,second_name:B}
    }))
    .unwrap()
}
fn artifact(path: &Path, basename: &str) -> Artifact {
    Artifact::from_file(File::open(path).unwrap(), Path::new(basename)).unwrap()
}
fn listed(name: &str, opened: Artifact, expected: &[u8]) -> ExpectedSourceShard {
    ExpectedSourceShard::new(name.into(), opened, expected.len() as u64, sha(expected)).unwrap()
}
fn admission(limit: u64) -> (TransferAdmission, PhysicalDeviceKey) {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host-memory-test", "indexed-owned-source").unwrap();
    arbiter.register(key.clone(), limit).unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource: key.clone(),
        owner: ResourceOwner::new("safetensors", "cold-bundle").unwrap(),
    };
    (admission, key)
}
fn limits(max: u64) -> SourceOpenLimits {
    SourceOpenLimits {
        max_owned_bytes: max,
        max_metadata_bytes: 32 * 1024 * 1024,
        max_shards: 4,
        max_tensors: 8,
    }
}
struct Two {
    fx: Fixture,
    first: Vec<u8>,
    second: Vec<u8>,
    idx: Vec<u8>,
}
impl Two {
    fn new() -> Self {
        let fx = Fixture::new();
        let first = safetensors("a", "F32", &[2], &[0, 0, 128, 63, 0, 0, 0, 64]);
        let second = safetensors("b", "BF16", &[2], &[0x80, 0x3f, 0x00, 0x40]);
        let idx = index("a", "b", 12);
        fx.file(A, &first);
        fx.file(B, &second);
        Self {
            fx,
            first,
            second,
            idx,
        }
    }
    fn shards(&self) -> Vec<ExpectedSourceShard> {
        vec![
            listed(B, artifact(&self.fx.path.join(B), B), &self.second),
            listed(A, artifact(&self.fx.path.join(A), A), &self.first),
        ] // intentionally reverse index shard order
    }
    fn prepare(
        &self,
        admit: &TransferAdmission,
        max: u64,
    ) -> zerogravity_artifacts::Result<zerogravity_artifacts::verified_shards::VerifiedShardPlan>
    {
        SourceWeights::prepare_verified_shards(
            &self.idx,
            &sha(&self.idx),
            self.shards(),
            limits(max),
            admit,
        )
    }
}

#[test]
fn indexed_two_shard_owned_header_bytes_dtype_and_reference_vectors_are_real() {
    let fx = Two::new();
    let (admit, key) = admission(48 * 1024 * 1024);
    let plan = fx.prepare(&admit, 1024 * 1024).unwrap();
    assert_eq!(plan.shard_count(), 2);
    assert_eq!(plan.declared_payload_bytes(), 12);
    let charged = admit.arbiter.snapshot(&key).unwrap().reserved;
    assert!(charged > fx.first.len() as u64 + fx.second.len() as u64);
    let source = Arc::new(plan.materialize().unwrap());
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, charged);
    assert_eq!(source.source_len(), fx.first.len() + fx.second.len());
    assert_eq!(
        source.verified_shard_index_sha256(),
        Some(sha(&fx.idx).as_str())
    );
    assert_eq!(source.tensor_names().count(), 2);
    assert_eq!(source.architecture(), ""); // no invented original-model identity
    assert_eq!(source.vector("a").unwrap(), [1.0, 2.0]);
    assert_eq!(source.vector("b").unwrap(), [1.0, 2.0]);
    assert_eq!(
        source.tensor_bytes("a").unwrap(),
        &fx.first[fx.first.len() - 8..]
    );
    assert_eq!(
        source.tensor_bytes("b").unwrap(),
        &fx.second[fx.second.len() - 4..]
    );
    let t = TensorSource::tensor(source.as_ref(), "b").unwrap();
    assert_eq!(t.shape, &[2]);
    assert_eq!(t.kind, zerogravity_core::quant::QuantKind::Bf16);
    assert_eq!(t.packed_bytes, &fx.second[fx.second.len() - 4..]);
    let cloned = Arc::clone(&source);
    assert_eq!(cloned.source_identity(), source.source_identity());
    drop(source);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, charged);
    drop(cloned);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn distinct_bundle_identity_and_foreign_tensor_metadata_do_not_select_another_shard() {
    let fx = Two::new();
    let (admit, _) = admission(128 * 1024 * 1024);
    let a = fx
        .prepare(&admit, 1024 * 1024)
        .unwrap()
        .materialize()
        .unwrap();
    let b = fx
        .prepare(&admit, 1024 * 1024)
        .unwrap()
        .materialize()
        .unwrap();
    assert_ne!(a.source_identity(), b.source_identity());
    let mut forged = a.tensor("a").unwrap().clone();
    forged.byte_offset = a.tensor("b").unwrap().byte_offset;
    assert!(a.bytes(&forged).is_err());
    forged = a.tensor("a").unwrap().clone();
    forged.kind = zerogravity_core::quant::QuantKind::Bf16;
    assert!(a.bytes(&forged).is_err());
    forged = a.tensor("a").unwrap().clone();
    forged.shape = vec![1, 2];
    assert!(a.bytes(&forged).is_err());
    forged.name = "missing".into();
    assert!(a.bytes(&forged).is_err());
    assert!(a.tensor_bytes("nonexistent").is_err());
}
#[test]
fn source_index_header_bijection_and_caller_expected_digests_fail_closed() {
    let fx = Two::new();
    let (admit, key) = admission(128 * 1024 * 1024);
    let invalid = vec![
        index("a", "b", 11),              // wrong payload total
        index("a", "strange-tensor", 12), // indexed name not in owned header
        index("a", "b", u64::MAX),        // bad original declared payload
    ];
    for raw in invalid {
        assert!(
            SourceWeights::prepare_verified_shards(
                &raw,
                &sha(&raw),
                fx.shards(),
                limits(1024 * 1024),
                &admit
            )
            .is_err()
        );
        assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    assert!(
        SourceWeights::prepare_verified_shards(
            &fx.idx,
            &"0".repeat(64),
            fx.shards(),
            limits(1024 * 1024),
            &admit
        )
        .is_err()
    );
    assert!(
        SourceWeights::prepare_verified_shards(
            &fx.idx,
            &sha(&fx.idx),
            vec![fx.shards().remove(0)],
            limits(1024 * 1024),
            &admit
        )
        .is_err()
    );
    let mut dup = fx.shards();
    dup.push(listed(A, artifact(&fx.fx.path.join(A), A), &fx.first));
    assert!(
        SourceWeights::prepare_verified_shards(
            &fx.idx,
            &sha(&fx.idx),
            dup,
            limits(1024 * 1024),
            &admit
        )
        .is_err()
    );
    let bad_shard = ExpectedSourceShard::new(
        A.into(),
        artifact(&fx.fx.path.join(A), A),
        fx.first.len() as u64,
        "0".repeat(64),
    )
    .unwrap();
    let good = listed(B, artifact(&fx.fx.path.join(B), B), &fx.second);
    let plan = SourceWeights::prepare_verified_shards(
        &fx.idx,
        &sha(&fx.idx),
        vec![bad_shard, good],
        limits(1024 * 1024),
        &admit,
    )
    .unwrap();
    assert!(matches!(plan.materialize(), Err(Error::Malformed(_))));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn wrong_shard_assignment_or_header_before_copy_never_publishes_source() {
    let fx = Two::new();
    let (admit, key) = admission(128 * 1024 * 1024);
    let mut swapped = fx.shards();
    swapped.swap(0, 1);
    let first = swapped.remove(0);
    let second = swapped.remove(0);
    // Names swapped without changing actual bytes: index/header disagrees.
    let a = ExpectedSourceShard::new(
        B.into(),
        artifact(&fx.fx.path.join(A), A),
        fx.first.len() as u64,
        sha(&fx.first),
    )
    .unwrap();
    let b = ExpectedSourceShard::new(
        A.into(),
        artifact(&fx.fx.path.join(B), B),
        fx.second.len() as u64,
        sha(&fx.second),
    )
    .unwrap();
    assert!(
        SourceWeights::prepare_verified_shards(
            &fx.idx,
            &sha(&fx.idx),
            vec![a, b],
            limits(1024 * 1024),
            &admit
        )
        .is_err()
    );
    drop(first);
    drop(second);
    let after = safetensors("z", "F32", &[2], &[0, 0, 0, 0, 0, 0, 0, 0]);
    fs::write(fx.fx.path.join(A), after).unwrap();
    assert!(
        SourceWeights::prepare_verified_shards(
            &fx.idx,
            &sha(&fx.idx),
            fx.shards(),
            limits(1024 * 1024),
            &admit
        )
        .is_err()
    );
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn descriptor_path_replacement_and_mutation_bound_to_exact_owned_bytes() {
    let fx = Two::new();
    let (admit, key) = admission(128 * 1024 * 1024);
    let plan = fx.prepare(&admit, 1024 * 1024).unwrap();
    // Original opened descriptor survives pathname replacement before copy.
    fs::rename(fx.fx.path.join(A), fx.fx.path.join("retained-original")).unwrap();
    fs::write(fx.fx.path.join(A), safetensors("new", "F32", &[2], &[0; 8])).unwrap();
    let original = plan.materialize().unwrap();
    assert_eq!(original.vector("a").unwrap(), [1.0, 2.0]);
    assert!(admit.arbiter.snapshot(&key).unwrap().reserved > 0);
    // Mutating the original inode after the accepted copy cannot change it.
    fs::write(
        fx.fx.path.join("retained-original"),
        vec![0u8; fx.first.len()],
    )
    .unwrap();
    assert_eq!(original.vector("a").unwrap(), [1.0, 2.0]);
    drop(original);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    // Original descriptor inspected first then in-place altered: hash must fail.
    let original_path = fx.fx.path.join(B);
    let opened = artifact(&original_path, B);
    let altered = safetensors("b", "BF16", &[2], &[0, 0, 0, 0]);
    fs::write(&original_path, altered).unwrap();
    let other = listed(B, opened, &fx.second);
    // Restore the legitimate original A under its current pathname;
    // the retained inode was deliberately corrupted above for the
    // post-publication immutability check.
    fs::write(fx.fx.path.join(A), &fx.first).unwrap();
    let first = listed(A, artifact(&fx.fx.path.join(A), A), &fx.first);
    let plan = SourceWeights::prepare_verified_shards(
        &fx.idx,
        &sha(&fx.idx),
        vec![first, other],
        limits(1024 * 1024),
        &admit,
    );
    // Corruption may be detected by the preliminary metadata, or final SHA.
    if let Ok(plan) = plan {
        assert!(plan.materialize().is_err());
    }
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn insufficient_host_budget_and_default_4gib_source_cap_refuse_before_any_materialization() {
    let fx = Two::new();
    let (admit, key) = admission(1024);
    assert!(matches!(
        fx.prepare(&admit, 1024 * 1024),
        Err(Error::Limit(_))
    ));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    let (admit, key) = admission(128 * 1024 * 1024);
    assert!(matches!(fx.prepare(&admit, 1), Err(Error::Limit(_))));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    assert!(matches!(
        SourceWeights::prepare_verified_shards(
            &fx.idx,
            &sha(&fx.idx),
            fx.shards(),
            SourceOpenLimits {
                max_owned_bytes: 1024 * 1024,
                max_metadata_bytes: 1024,
                max_shards: 4,
                max_tensors: 8
            },
            &admit
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn multiple_cold_transactions_share_one_admitted_host_budget_and_rollback_exactly() {
    let fx = Two::new();
    let (admit, key) = admission(18 * 1024 * 1024);
    let plan_a = fx.prepare(&admit, 1024 * 1024).unwrap();
    let plan_b = fx.prepare(&admit, 1024 * 1024).unwrap();
    assert!(matches!(
        fx.prepare(&admit, 1024 * 1024),
        Err(Error::Limit(_))
    ));
    let charged = admit.arbiter.snapshot(&key).unwrap().reserved;
    let owned = Arc::new(plan_a.materialize().unwrap());
    drop(plan_b);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, charged / 2);
    drop(owned);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn sparse_large_original_file_header_must_refuse_pre_admission_not_read_5gib() {
    let fixture = Fixture::new();
    let path = fixture.path.join("huge.safetensors");
    let mut file_len = 5 * 1024 * 1024 * 1024u64 + 4096;
    // JSON offset length depends on header decimal digits. Re-solve
    // after choosing an even BF16 payload length; never allocate data.
    let (header, data_len) = loop {
        let mut data_len = file_len - 256;
        let mut header = Vec::new();
        for _ in 0..8 {
            header = serde_json::to_vec(&serde_json::json!({
                "verylarge":{"dtype":"BF16","shape":[data_len/2],"data_offsets":[0,data_len]}
            }))
            .unwrap();
            let next = file_len - 8 - header.len() as u64;
            if next == data_len {
                break;
            }
            data_len = next;
        }
        assert_eq!(file_len - 8 - header.len() as u64, data_len);
        if data_len.is_multiple_of(2) {
            break (header, data_len);
        }
        file_len += 1;
    };
    let mut file = File::create(&path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.set_len(file_len).unwrap(); // sparse, no 5GiB actual bytes touched
    file.seek(SeekFrom::Start(0)).unwrap();
    drop(file);
    let opened = artifact(&path, "huge.safetensors");
    let huge =
        ExpectedSourceShard::new("huge.safetensors".into(), opened, file_len, "0".repeat(64))
            .unwrap();
    let idx = serde_json::to_vec(&serde_json::json!({
        "metadata":{"total_size":data_len},
        "weight_map":{"verylarge":"huge.safetensors"}
    }))
    .unwrap();
    let (admit, key) = admission(1024 * 1024);
    let result = SourceWeights::prepare_verified_shards(
        &idx,
        &sha(&idx),
        vec![huge],
        SourceOpenLimits::default(),
        &admit,
    );
    assert!(matches!(result, Err(Error::Limit(_))));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    // Explicitly admitting >4GiB as a logical source limit still needs
    // capacity from the ORIGINAL physical host RAM arbiter. No sparse
    // payload is read when physical admission cannot cover it.
    let huge = ExpectedSourceShard::new(
        "huge.safetensors".into(),
        artifact(&path, "huge.safetensors"),
        file_len,
        "0".repeat(64),
    )
    .unwrap();
    let bigger = SourceOpenLimits {
        max_owned_bytes: 6 * 1024 * 1024 * 1024,
        ..SourceOpenLimits::default()
    };
    let physical =
        SourceWeights::prepare_verified_shards(&idx, &sha(&idx), vec![huge], bigger, &admit);
    assert!(matches!(physical, Err(Error::Limit(_))));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    // No huge file read, data download or 5GiB transient allocation.
}

#[test]
fn same_length_changed_header_with_matching_new_digest_refuses_stale_preflight_geometry() {
    let fx = Two::new();
    let (admit, key) = admission(128 * 1024 * 1024);
    // Inspect old F32[2]. Then change to F16[4] with identical byte
    // length and a caller-matching SHA256. The *inspected* geometry was
    // used for the index/admission, so a changed captured header refuses.
    let old_descriptor = artifact(&fx.fx.path.join(A), A);
    let changed = safetensors("a", "F16", &[4], &[0x00, 0x3c, 0x00, 0x40, 0, 0, 0, 0]);
    assert_eq!(changed.len(), fx.first.len());
    fs::write(fx.fx.path.join(A), &changed).unwrap();
    let mutated = listed(A, old_descriptor, &changed);
    let original = listed(B, artifact(&fx.fx.path.join(B), B), &fx.second);
    let plan = SourceWeights::prepare_verified_shards(
        &fx.idx,
        &sha(&fx.idx),
        vec![mutated, original],
        limits(1024 * 1024),
        &admit,
    )
    .unwrap();
    assert!(matches!(plan.materialize(), Err(Error::Malformed(_))));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
fn exact_single_shard_f16_header_payload_and_borrowed_tensor_source() {
    let fixture = Fixture::new();
    let file = "model.safetensors";
    let raw = safetensors("c", "F16", &[2], &[0x00, 0x3c, 0x00, 0x40]);
    fixture.file(file, &raw);
    let idx = serde_json::to_vec(&serde_json::json!({
        "metadata":{"total_size":4},
        "weight_map":{"c":file}
    }))
    .unwrap();
    let (admit, key) = admission(64 * 1024 * 1024);
    let plan = SourceWeights::prepare_verified_shards(
        &idx,
        &sha(&idx),
        vec![listed(file, artifact(&fixture.path.join(file), file), &raw)],
        limits(1024 * 1024),
        &admit,
    )
    .unwrap();
    assert_eq!(plan.declared_payload_bytes(), 4);
    let source = plan.materialize().unwrap();
    assert_eq!(source.source_len(), raw.len());
    assert_eq!(source.vector("c").unwrap(), [1.0, 2.0]);
    assert_eq!(
        TensorSource::tensor(&source, "c").unwrap().kind,
        zerogravity_core::quant::QuantKind::F16
    );
    drop(source);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
fn authenticated_bundle_digest_binds_actual_learned_bytes_and_is_order_independent() {
    let fx = Two::new();
    let (admit, key) = admission(128 * 1024 * 1024);
    let old = fx
        .prepare(&admit, 1024 * 1024)
        .unwrap()
        .materialize()
        .unwrap();
    let old_bundle = old.verified_bundle_sha256().unwrap().to_owned();
    assert_eq!(old_bundle.len(), 64);
    assert_eq!(
        old.verified_shard_index_sha256(),
        Some(sha(&fx.idx).as_str())
    );

    // Shard order does not change the canonical source-bundle content digest.
    let mut reverse = fx.shards();
    reverse.reverse();
    let reordered = SourceWeights::prepare_verified_shards(
        &fx.idx,
        &sha(&fx.idx),
        reverse,
        limits(1024 * 1024),
        &admit,
    )
    .unwrap()
    .materialize()
    .unwrap();
    assert_eq!(
        reordered.verified_bundle_sha256(),
        Some(old_bundle.as_str())
    );
    assert_ne!(reordered.source_identity(), old.source_identity());

    // Same source index, same names/headers/shapes, different actual BF16
    // weight content: source bundle MUST change even though index doesn't.
    let changed = safetensors("b", "BF16", &[2], &[0x00, 0x40, 0x80, 0x3f]);
    assert_eq!(changed.len(), fx.second.len());
    fs::write(fx.fx.path.join(B), &changed).unwrap();
    let new = SourceWeights::prepare_verified_shards(
        &fx.idx,
        &sha(&fx.idx),
        vec![
            listed(B, artifact(&fx.fx.path.join(B), B), &changed),
            listed(A, artifact(&fx.fx.path.join(A), A), &fx.first),
        ],
        limits(1024 * 1024),
        &admit,
    )
    .unwrap()
    .materialize()
    .unwrap();
    assert_eq!(
        new.verified_shard_index_sha256(),
        Some(sha(&fx.idx).as_str())
    );
    assert_ne!(new.verified_bundle_sha256(), Some(old_bundle.as_str()));
    assert_eq!(old.vector("b").unwrap(), [1.0, 2.0]);
    assert_eq!(new.vector("b").unwrap(), [2.0, 1.0]);
    drop(new);
    drop(reordered);
    drop(old);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
