//! Synthetic file/byte ownership tests, not trained streaming or GPU evidence.
#![cfg(all(feature = "cache", target_os = "linux"))]

use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use zerogravity_artifacts::Error;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::source_weights::{
    ExpectedSourceShard, SourceOpenLimits, SourceWeights, VerifiedWindowPlan, WindowOpenLimits,
};
use zerogravity_core::accelerator::program::MatrixAxes;
use zerogravity_core::resource_arbiter::{
    AllocationClass, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

const A: &str = "model-00001-of-00002.safetensors";
const B: &str = "model-00002-of-00002.safetensors";
const META: u64 = 16 * 1024 * 1024;
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn safetensors(name: &str, dtype: &str, shape: &[u64], raw: &[u8]) -> Vec<u8> {
    let header = serde_json::to_vec(
        &json!({name:{"dtype":dtype,"shape":shape,"data_offsets":[0,raw.len()]}}),
    )
    .unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(raw);
    bytes
}
struct Fixture {
    path: PathBuf,
    first: Vec<u8>,
    second: Vec<u8>,
    index: Vec<u8>,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-owned-windows-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        let first = safetensors(
            "matrix",
            "F32",
            &[4, 4],
            &(1..=16)
                .flat_map(|n| (n as f32).to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let second = safetensors(
            "bf16",
            "BF16",
            &[2, 3],
            &[0x3f80u16, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        );
        let index = serde_json::to_vec(
            &json!({"metadata":{"total_size":76},"weight_map":{"matrix":A,"bf16":B}}),
        )
        .unwrap();
        fs::write(path.join(A), &first).unwrap();
        fs::write(path.join(B), &second).unwrap();
        Self {
            path,
            first,
            second,
            index,
        }
    }
    fn shards(&self) -> Vec<ExpectedSourceShard> {
        [(B, &self.second), (A, &self.first)]
            .into_iter()
            .map(|(name, bytes)| {
                ExpectedSourceShard::new(
                    name.into(),
                    Artifact::from_file(
                        File::open(self.path.join(name)).unwrap(),
                        std::path::Path::new(name),
                    )
                    .unwrap(),
                    bytes.len() as u64,
                    sha(bytes),
                )
                .unwrap()
            })
            .collect()
    }
    fn plan(
        &self,
        admission: &TransferAdmission,
    ) -> zerogravity_artifacts::Result<VerifiedWindowPlan> {
        SourceWeights::prepare_verified_windows(
            &self.index,
            &sha(&self.index),
            self.shards(),
            limits(),
            admission,
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
fn limits() -> WindowOpenLimits {
    WindowOpenLimits {
        max_source_bytes: 1024 * 1024,
        max_metadata_bytes: META,
        max_shards: 2,
        max_tensors: 2,
        scan_chunk_bytes: 64,
    }
}
fn admission(capacity: u64, backend: &str) -> TransferAdmission {
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new(backend, "original-window-tests").unwrap();
    arbiter.register(resource.clone(), capacity).unwrap();
    TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original-fixture", "window-io").unwrap(),
    }
}
fn reserved(admission: &TransferAdmission) -> u64 {
    admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .reserved
}

#[test]
fn original_index_whole_hash_header_axes_dtype_and_bundle_are_exact() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let plan = fixture.plan(&host).unwrap();
    assert_eq!(
        plan.source_bytes(),
        (fixture.first.len() + fixture.second.len()) as u64
    );
    assert_eq!(plan.metadata_reserved_bytes(), META);
    assert_eq!(reserved(&host), META);
    let source = plan.open().unwrap();
    assert!(source.source_identity().is_bound());
    assert_eq!(source.verified_shard_index_sha256(), sha(&fixture.index));
    assert_eq!(
        source.source_bytes(),
        (fixture.first.len() + fixture.second.len()) as u64
    );
    assert_eq!(source.metadata_reserved_bytes(), META);
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    assert_eq!((matrix.rows(), matrix.columns()), (4, 4));
    assert_eq!(matrix.axes(), MatrixAxes::OutputInput);
    assert_eq!(
        matrix.tensor().kind,
        zerogravity_core::quant::QuantKind::F32
    );
    let tile = source.read_rows(&matrix, 1..3).unwrap();
    assert_eq!(tile.row_range(), 1..3);
    assert_eq!(tile.matrix().source_identity(), source.source_identity());
    assert_eq!(
        tile.as_slice(),
        &(5..=12)
            .flat_map(|n| (n as f32).to_le_bytes())
            .collect::<Vec<_>>()
    );
    assert_eq!(tile.source_offset(), matrix.tensor().byte_offset + 16);
    assert_eq!(tile.shard_basename(), A);
    assert_eq!(tile.shard_sha256(), sha(&fixture.first));
    assert_eq!(tile.extent_index_sha256().len(), 64);
    assert_eq!(reserved(&host), META + tile.charged_staging_bytes() as u64);
    let other = source.matrix("bf16", MatrixAxes::OutputInput).unwrap();
    let bf16 = source.read_rows(&other, 1..2).unwrap();
    assert_eq!((other.rows(), other.columns()), (2, 3));
    assert_eq!(bf16.as_slice(), &[0x80, 0x40, 0xa0, 0x40, 0xc0, 0x40]);
    assert_eq!(
        other.tensor().kind,
        zerogravity_core::quant::QuantKind::Bf16
    );
    // Existing complete materialization independently uses the same bundle hash.
    let owned = SourceWeights::prepare_verified_shards(
        &fixture.index,
        &sha(&fixture.index),
        fixture.shards(),
        SourceOpenLimits::default(),
        &host,
    )
    .unwrap()
    .materialize()
    .unwrap();
    assert_eq!(
        Some(source.verified_bundle_sha256()),
        owned.verified_bundle_sha256()
    );
    assert_eq!(
        tile.verified_bundle_sha256(),
        source.verified_bundle_sha256()
    );
    assert_ne!(source.source_identity(), owned.source_identity());
    drop((tile, bf16, matrix, other, source, owned));
    assert_eq!(reserved(&host), 0);
    assert_eq!(
        WindowOpenLimits::default().max_source_bytes,
        4 * 1024 * 1024 * 1024
    );
}

#[test]
fn private_owners_refuse_foreign_matrix_wrong_axes_and_invalid_rows_before_io() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let first = fixture.plan(&host).unwrap().open().unwrap();
    let second = fixture.plan(&host).unwrap().open().unwrap();
    let matrix = first.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    assert_ne!(first.source_identity(), second.source_identity());
    assert!(matches!(
        second.read_rows(&matrix, 0..1),
        Err(Error::State(_))
    ));
    assert!(first.matrix("matrix", MatrixAxes::InputOutput).is_err());
    assert!(first.matrix("foreign", MatrixAxes::OutputInput).is_err());
    for rows in [
        0..0,
        std::ops::Range { start: 3, end: 2 },
        0..5,
        usize::MAX..usize::MAX,
    ] {
        assert!(first.read_rows(&matrix, rows).is_err());
    }
    assert_eq!(reserved(&host), 2 * META);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
    drop((matrix, first, second));
    assert_eq!(reserved(&host), 0);
    let gpu = admission(64 * 1024 * 1024, "pci");
    assert!(matches!(fixture.plan(&gpu), Err(Error::State(_))));
    assert_eq!(reserved(&gpu), 0);
}

#[test]
fn window_and_descriptor_hold_source_metadata_and_actual_staging_until_last_drop() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let descriptor = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let window = source.read_rows(&descriptor, 3..4).unwrap();
    let bytes = window.as_slice().to_vec();
    let staging = window.charged_staging_bytes() as u64;
    assert_eq!(
        host.arbiter.snapshot(&host.resource).unwrap().per_class[&AllocationClass::TransferStaging],
        staging
    );
    drop(source);
    drop(descriptor);
    assert_eq!(reserved(&host), META + staging);
    assert_eq!(window.as_slice(), bytes);
    drop(window);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn mutation_or_short_read_never_exposes_new_bytes_and_keeps_prior_window_immutable() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let before = source.read_rows(&matrix, 0..2).unwrap();
    let original = before.as_slice().to_vec();
    let charge = reserved(&host);
    let mut changed = fixture.first.clone();
    *changed.last_mut().unwrap() ^= 1;
    fs::write(fixture.path.join(A), &changed).unwrap();
    assert!(source.read_rows(&matrix, 3..4).is_err());
    assert_eq!(reserved(&host), charge);
    assert_eq!(before.as_slice(), original);
    File::create(fixture.path.join(A))
        .unwrap()
        .set_len(12)
        .unwrap();
    assert!(source.read_rows(&matrix, 0..1).is_err());
    assert_eq!(reserved(&host), charge);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
    drop((source, matrix, before));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn opened_descriptor_survives_path_replacement_and_public_info_cannot_forge_geometry() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let mut first = Artifact::from_file(
        File::open(fixture.path.join(A)).unwrap(),
        std::path::Path::new(A),
    )
    .unwrap();
    first.info.tensors[0].shape = vec![900, 900];
    first.info.tensors[0].byte_len = 1;
    first.info.tensors[0].byte_offset = 0;
    let mut shards = fixture.shards();
    shards.pop();
    shards.push(
        ExpectedSourceShard::new(
            A.into(),
            first,
            fixture.first.len() as u64,
            sha(&fixture.first),
        )
        .unwrap(),
    );
    let plan = SourceWeights::prepare_verified_windows(
        &fixture.index,
        &sha(&fixture.index),
        shards,
        limits(),
        &host,
    )
    .unwrap();
    fs::rename(fixture.path.join(A), fixture.path.join("retained")).unwrap();
    fs::write(fixture.path.join(A), vec![0; fixture.first.len()]).unwrap();
    let source = plan.open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    assert_eq!(
        (matrix.rows(), matrix.columns(), matrix.tensor().byte_len),
        (4, 4, 64)
    );
    let window = source.read_rows(&matrix, 0..1).unwrap();
    assert_eq!(
        window.as_slice(),
        &(1..=4)
            .flat_map(|n| (n as f32).to_le_bytes())
            .collect::<Vec<_>>()
    );
    drop((source, matrix, window));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn whole_hash_index_payload_and_actual_header_closure_fail_without_source_publication() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let plan = fixture.plan(&host).unwrap();
    let mut changed = fixture.second.clone();
    *changed.last_mut().unwrap() ^= 1;
    fs::write(fixture.path.join(B), &changed).unwrap();
    assert!(plan.open().is_err());
    assert_eq!(reserved(&host), 0);
    fs::write(fixture.path.join(B), &fixture.second).unwrap();
    let mut index: serde_json::Value = serde_json::from_slice(&fixture.index).unwrap();
    index["metadata"]["total_size"] = json!(77);
    let index = serde_json::to_vec(&index).unwrap();
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        &sha(&index),
        fixture.shards(),
        limits(),
        &host,
    )
    .unwrap();
    assert!(plan.open().is_err());
    assert_eq!(reserved(&host), 0);
    let mut index: serde_json::Value = serde_json::from_slice(&fixture.index).unwrap();
    index["weight_map"]["matrix"] = json!(B);
    index["weight_map"]["bf16"] = json!(A);
    let index = serde_json::to_vec(&index).unwrap();
    let plan = SourceWeights::prepare_verified_windows(
        &index,
        &sha(&index),
        fixture.shards(),
        limits(),
        &host,
    )
    .unwrap();
    assert!(plan.open().is_err());
    assert_eq!(reserved(&host), 0);
}

#[test]
fn metadata_scan_and_window_capacity_refusals_are_real_and_do_not_fence() {
    let fixture = Fixture::new();
    let too_small = admission(META - 1, "host");
    assert!(matches!(fixture.plan(&too_small), Err(Error::Limit(_))));
    assert_eq!(reserved(&too_small), 0);
    let no_scan = admission(META, "host");
    let plan = fixture.plan(&no_scan).unwrap();
    assert!(matches!(plan.open(), Err(Error::Limit(_))));
    assert_eq!(reserved(&no_scan), 0);
    let host = admission(META + 256, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let mut windows = Vec::new();
    while let Ok(window) = source.read_rows(&matrix, 0..4) {
        windows.push(window);
        assert!(windows.len() <= 16);
    }
    assert!(!windows.is_empty());
    assert!(matches!(
        source.read_rows(&matrix, 0..4),
        Err(Error::Limit(_))
    ));
    assert!(reserved(&host) <= META + 256);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
    drop((windows, matrix, source));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn cancellation_and_dropped_plans_release_only_owned_completed_host_work() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let plan = fixture.plan(&host).unwrap();
    assert_eq!(reserved(&host), META);
    drop(plan);
    assert_eq!(reserved(&host), 0);
    let cancelled = AtomicBool::new(true);
    assert!(
        fixture
            .plan(&host)
            .unwrap()
            .open_cancellable(&cancelled)
            .is_err()
    );
    assert_eq!(reserved(&host), 0);
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    assert!(
        source
            .read_rows_cancellable(&matrix, 0..1, &cancelled)
            .is_err()
    );
    assert_eq!(reserved(&host), META);
    cancelled.store(false, Ordering::Release);
    let window = source
        .read_rows_cancellable(&matrix, 0..1, &cancelled)
        .unwrap();
    cancelled.store(true, Ordering::Release);
    let bytes = window.as_slice().to_vec();
    let charge = reserved(&host);
    assert!(
        source
            .read_rows_cancellable(&matrix, 0..1, &cancelled)
            .is_err()
    );
    assert_eq!(reserved(&host), charge);
    assert_eq!(window.as_slice(), bytes);
    drop((source, matrix, window));
    assert_eq!(reserved(&host), 0);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
}

#[test]
fn spare_shard_capacity_and_unbounded_host_labels_refuse_before_admission() {
    let fixture = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let capacity = META as usize / std::mem::size_of::<ExpectedSourceShard>() + 1;
    let mut shards = Vec::with_capacity(capacity);
    shards.extend(fixture.shards());
    assert_eq!(shards.len(), 2);
    assert!(shards.capacity() * std::mem::size_of::<ExpectedSourceShard>() > META as usize);
    assert!(matches!(
        SourceWeights::prepare_verified_windows(
            &fixture.index,
            &sha(&fixture.index),
            shards,
            limits(),
            &host
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(reserved(&host), 0);
    let mut long = host.clone();
    long.owner.model = "m".repeat(257);
    assert!(matches!(fixture.plan(&long), Err(Error::State(_))));
    long.owner.model = "original-fixture".into();
    long.owner.session = "s".repeat(257);
    assert!(matches!(fixture.plan(&long), Err(Error::State(_))));
    assert_eq!(reserved(&host), 0);
}
