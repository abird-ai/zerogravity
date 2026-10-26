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
fn observation_charge_does_not_satisfy_backing_retention_itself() {
    let fixture = Fixture::new();
    let host = admission(META + 16 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let witness = source.weak_owner();
    let alias = source.clone();
    assert!(witness.same_issuer(&alias));
    let foreign_host = admission(META + 4096, "host");
    let foreign = fixture.plan(&foreign_host).unwrap().open().unwrap();
    assert!(!witness.same_issuer(&foreign));
    drop(foreign);
    assert_eq!(reserved(&foreign_host), 0);
    let observation = source
        .reserve_workspace(4096)
        .unwrap()
        .into_observation_workspace()
        .unwrap();
    let retained = source
        .reserve_workspace(4096)
        .unwrap()
        .into_host_workspace()
        .unwrap();
    drop(source);
    drop(alias);
    assert!(witness.retained());
    assert_eq!(reserved(&host), META + 8192);
    drop(retained);
    assert!(!witness.retained());
    assert_eq!(reserved(&host), 4096);
    drop(observation);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn shared_small_window_reuse_spends_only_prepaid_capacity_and_keeps_source_alive() {
    use zerogravity_core::resource_arbiter::AllocationRequest;
    let f = Fixture::new();
    let host = admission(META + 4096, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let first = source.read_rows(&matrix, 1..3).unwrap();
    let expected = first.as_slice().to_vec();
    drop(first);
    assert_eq!(reserved(&host), META);
    let occupied = host
        .arbiter
        .admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 4096,
            }],
        )
        .unwrap();
    let hot = source.read_rows(&matrix, 1..3).unwrap();
    assert_eq!(hot.as_slice(), expected);
    assert_eq!(reserved(&host), META + 4096);
    assert!(matches!(
        source.read_rows(&matrix, 0..1),
        Err(Error::Limit(_))
    ));
    assert!(
        source
            .read_rows_cancellable(&matrix, 1..3, &AtomicBool::new(true))
            .is_err()
    );
    let stats = source.window_cache_stats().unwrap();
    assert_eq!(stats.hits, 1);
    assert!(stats.live_payload_bytes > 0 && stats.live_payload_bytes <= stats.budget_bytes);
    drop((source, matrix, occupied));
    assert_eq!(reserved(&host), META);
    assert_eq!(hot.as_slice(), expected);
    drop(hot);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn disabled_reuse_keeps_original_staging_charges_and_bytes() {
    let f = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = f
        .plan(&host)
        .unwrap()
        .without_window_cache()
        .open()
        .unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let first = source.read_rows(&matrix, 1..3).unwrap();
    let one = reserved(&host);
    let second = source.read_rows(&matrix, 1..3).unwrap();
    assert_eq!(first.as_slice(), second.as_slice());
    assert!(one > META && reserved(&host) > one);
    let stats = source.window_cache_stats().unwrap();
    assert_eq!(stats.budget_bytes, 0);
    assert_eq!(stats.live_payload_bytes, 0);
    assert_eq!(stats.hits, 0);
    drop((first, second, matrix, source));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn warm_reuse_preserves_held_source_when_path_is_replaced() {
    let f = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    let original = source.read_rows(&matrix, 0..1).unwrap().as_slice().to_vec();
    let retained = f.path.join("original-held");
    fs::rename(f.path.join(A), &retained).unwrap();
    fs::write(f.path.join(A), vec![0xff; f.first.len()]).unwrap();
    let held = source.read_rows(&matrix, 0..1).unwrap();
    assert_eq!(held.as_slice(), original);
    assert_eq!(source.window_cache_stats().unwrap().entries, 0);
    fs::write(retained, vec![0; f.first.len()]).unwrap();
    assert!(source.read_rows(&matrix, 0..1).is_err());
    assert_eq!(held.as_slice(), original);
    drop((held, matrix, source));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn warm_cache_never_masks_source_mutation_or_foreign_issuer() {
    use std::io::{Seek, SeekFrom, Write};
    let f = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let other = f.plan(&host).unwrap().open().unwrap();
    let matrix = source.matrix("matrix", MatrixAxes::OutputInput).unwrap();
    drop(source.read_rows(&matrix, 0..1).unwrap());
    let before = source.read_rows(&matrix, 0..1).unwrap();
    assert!(matches!(
        other.read_rows(&matrix, 0..1),
        Err(Error::State(_))
    ));
    let mut changed = fs::OpenOptions::new()
        .write(true)
        .open(f.path.join(A))
        .unwrap();
    changed
        .seek(SeekFrom::Start(matrix.tensor().byte_offset))
        .unwrap();
    changed.write_all(&[0xff]).unwrap();
    changed.sync_all().unwrap();
    assert!(source.read_rows(&matrix, 0..1).is_err());
    assert_eq!(
        before.as_slice(),
        (1..=4)
            .flat_map(|n| (n as f32).to_le_bytes())
            .collect::<Vec<_>>()
    );
    drop((before, matrix, other, source));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn split_output_and_scratch_keep_classes_and_release_only_completed_scratch() {
    let f = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let before = host.arbiter.snapshot(&host.resource).unwrap();
    let (output, scratch) = source.reserve_operator_buffers_split(512, 2048).unwrap();
    assert!(source.same_issuer(output.source()));
    assert!(source.same_issuer(scratch.as_ref().unwrap().source()));
    let during = host.arbiter.snapshot(&host.resource).unwrap();
    for (class, added) in [
        (AllocationClass::Output, 512),
        (AllocationClass::Scratch, 2048),
    ] {
        assert_eq!(
            during.per_class.get(&class).copied().unwrap_or(0),
            before.per_class.get(&class).copied().unwrap_or(0) + added
        );
    }
    let output = std::sync::Arc::new(output);
    let retained = output.clone();
    drop(scratch);
    assert_eq!(reserved(&host), before.reserved + 512);
    drop(output);
    assert_eq!(reserved(&host), before.reserved + 512);
    drop(retained);
    assert_eq!(reserved(&host), before.reserved);
    let (zero_output, absent_scratch) = source.reserve_operator_buffers_split(8, 0).unwrap();
    assert!(absent_scratch.is_none());
    assert_eq!(reserved(&host), before.reserved + 8);
    drop(zero_output);
    drop(source);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn split_second_admission_failure_unwinds_first_and_cannot_expand_envelope() {
    let f = Fixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = f.plan(&host).unwrap().open().unwrap();
    let mut pressure = Vec::new();
    loop {
        let left = host.arbiter.snapshot(&host.resource).unwrap().remaining;
        if left <= 1024 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace((left - 1024).min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let before = reserved(&host);
    assert!(source.reserve_operator_buffers_split(512, 2048).is_err());
    assert_eq!(reserved(&host), before);
    for (out, scratch) in [(0, 1), (32 * 1024 * 1024, 1), (usize::MAX, 1)] {
        assert!(source.reserve_operator_buffers_split(out, scratch).is_err());
        assert_eq!(reserved(&host), before);
    }
    drop(pressure);
    drop(source);
    assert_eq!(reserved(&host), 0);
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
    // Exercise actual staging pressure; cached views share prepaid bytes.
    let source = fixture
        .plan(&host)
        .unwrap()
        .without_window_cache()
        .open()
        .unwrap();
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

const PARAMETERS: &str = "parameters.safetensors";
struct ParameterFixture {
    path: PathBuf,
    bytes: Vec<u8>,
    index: Vec<u8>,
    norm: Vec<u8>,
    bias: Vec<u8>,
    half: Vec<u8>,
}
impl ParameterFixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-parameter-windows-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let norm = (0..40u16)
            .flat_map(|n| (0x3f00 + n).to_le_bytes())
            .collect::<Vec<_>>();
        let bias = [0x80000000u32, 0x3f000000, 0x3f800000, 0xbf800000, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        let half = [0x3c00u16, 0xbc00, 0, 0x8000]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let matrix = vec![0u8; 32];
        let not_vector = vec![0u8; 8];
        let mut header = serde_json::Map::new();
        let mut weight_map = serde_json::Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, raw) in [
            ("norm", "BF16", vec![40u64], norm.as_slice()),
            ("bias", "F32", vec![5], bias.as_slice()),
            ("half", "F16", vec![4], half.as_slice()),
            ("matrix", "F32", vec![2, 4], matrix.as_slice()),
            ("not_vector", "BF16", vec![1, 4], not_vector.as_slice()),
        ] {
            let start = payload.len();
            payload.extend_from_slice(raw);
            header.insert(
                name.into(),
                json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
            );
            weight_map.insert(name.into(), json!(PARAMETERS));
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(&payload);
        let index = serde_json::to_vec(
            &json!({"metadata":{"total_size":payload.len()},"weight_map":weight_map}),
        )
        .unwrap();
        fs::write(path.join(PARAMETERS), &bytes).unwrap();
        Self {
            path,
            bytes,
            index,
            norm,
            bias,
            half,
        }
    }
    fn artifact(&self) -> Artifact {
        Artifact::from_file(
            File::open(self.path.join(PARAMETERS)).unwrap(),
            std::path::Path::new(PARAMETERS),
        )
        .unwrap()
    }
    fn plan(
        &self,
        admission: &TransferAdmission,
    ) -> zerogravity_artifacts::Result<VerifiedWindowPlan> {
        self.plan_with_artifact(self.artifact(), admission)
    }
    fn plan_with_artifact(
        &self,
        artifact: Artifact,
        admission: &TransferAdmission,
    ) -> zerogravity_artifacts::Result<VerifiedWindowPlan> {
        self.plan_with_chunk_bytes(artifact, admission, 64)
    }
    fn plan_with_chunk_bytes(
        &self,
        artifact: Artifact,
        admission: &TransferAdmission,
        scan_chunk_bytes: usize,
    ) -> zerogravity_artifacts::Result<VerifiedWindowPlan> {
        let shard = ExpectedSourceShard::new(
            PARAMETERS.into(),
            artifact,
            self.bytes.len() as u64,
            sha(&self.bytes),
        )?;
        SourceWeights::prepare_verified_windows(
            &self.index,
            &sha(&self.index),
            vec![shard],
            WindowOpenLimits {
                max_source_bytes: 1024 * 1024,
                max_metadata_bytes: META,
                max_shards: 1,
                max_tensors: 5,
                scan_chunk_bytes,
            },
            admission,
        )
    }
}
impl Drop for ParameterFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn raw_rank1_parameter_kind_original_descriptor_and_contiguous_ranges_are_exact() {
    use zerogravity_core::quant::QuantKind;
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    assert_eq!(reserved(&host), META);
    for (name, kind, shape, raw) in [
        (
            "norm",
            QuantKind::Bf16,
            vec![40u64],
            fixture.norm.as_slice(),
        ),
        ("bias", QuantKind::F32, vec![5], fixture.bias.as_slice()),
        ("half", QuantKind::F16, vec![4], fixture.half.as_slice()),
    ] {
        let parameter = source.parameter(name).unwrap();
        let descriptor = parameter.source_parameter();
        assert_eq!(descriptor.source, source.source_identity());
        assert_eq!(descriptor.name, name);
        assert_eq!(descriptor.kind, kind);
        assert_eq!(descriptor.shape, shape);
        assert_eq!(descriptor.bytes, raw.len());
        parameter.validate_source_parameter(&descriptor).unwrap();
        assert_eq!(parameter.tensor().kind, kind);
        assert_eq!(parameter.tensor().shape, shape);
        let window = source
            .read_parameter(&parameter, 0..parameter.elements())
            .unwrap();
        assert!(window.is_complete());
        assert_eq!(window.element_range(), 0..parameter.elements());
        assert_eq!(window.parameter().source_parameter(), descriptor);
        assert_eq!(window.as_slice(), raw);
        assert_eq!(window.source_offset(), parameter.tensor().byte_offset);
        assert_eq!(window.shard_basename(), PARAMETERS);
        assert_eq!(window.shard_sha256(), sha(&fixture.bytes));
        assert_eq!(
            window.verified_bundle_sha256(),
            source.verified_bundle_sha256()
        );
        assert_eq!(window.extent_index_sha256().len(), 64);
        assert_eq!(
            reserved(&host),
            META + window.charged_staging_bytes() as u64
        );
        drop(window);
        assert_eq!(reserved(&host), META);
    }
    let parameter = source.parameter("norm").unwrap();
    let partial = source.read_parameter(&parameter, 2..37).unwrap();
    assert!(!partial.is_complete());
    assert_eq!(partial.element_range(), 2..37);
    assert_eq!(partial.as_slice(), &fixture.norm[4..74]);
    assert_eq!(partial.source_offset(), parameter.tensor().byte_offset + 4);
    // Original shape/bytes remain full even for a partial physical byte range.
    assert_eq!(partial.parameter().source_parameter().shape, [40u64]);
    assert_eq!(partial.parameter().source_parameter().bytes, 80);
    let start = partial.source_offset() as usize;
    let end = start + partial.as_slice().len();
    let chunks = start / 64..=(end - 1) / 64;
    let data = chunks
        .clone()
        .map(|index| 64.min(fixture.bytes.len() - index * 64))
        .sum::<usize>();
    let offsets = chunks.count() * std::mem::size_of::<(usize, std::ops::Range<usize>)>();
    assert_eq!(partial.charged_staging_bytes(), data + offsets);
    assert!(partial.charged_staging_bytes() > partial.as_slice().len());
    assert_eq!(
        host.arbiter.snapshot(&host.resource).unwrap().per_class[&AllocationClass::TransferStaging],
        (data + offsets) as u64
    );
    drop((partial, parameter, source));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn rank1_parameter_sealed_issuer_rejects_foreign_owner_changed_typed_metadata_and_bad_ranges() {
    use zerogravity_core::accelerator::program::SourceParameter;
    use zerogravity_core::quant::QuantKind;
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let first = fixture.plan(&host).unwrap().open().unwrap();
    let second = fixture.plan(&host).unwrap().open().unwrap();
    let parameter = first.parameter("norm").unwrap();
    let descriptor = parameter.source_parameter();
    assert_ne!(first.source_identity(), second.source_identity());
    assert!(matches!(
        second.read_parameter(&parameter, 0..1),
        Err(Error::State(_))
    ));
    for changed in [
        second.parameter("norm").unwrap().source_parameter(),
        SourceParameter {
            name: "bias".into(),
            ..descriptor.clone()
        },
        SourceParameter {
            kind: QuantKind::F32,
            ..descriptor.clone()
        },
        SourceParameter {
            kind: QuantKind::Q4_0,
            ..descriptor.clone()
        },
        SourceParameter {
            shape: vec![1, 40],
            ..descriptor.clone()
        },
        SourceParameter {
            shape: vec![39],
            ..descriptor.clone()
        },
        SourceParameter {
            bytes: 79,
            ..descriptor.clone()
        },
    ] {
        assert!(matches!(
            parameter.validate_source_parameter(&changed),
            Err(Error::State(_))
        ));
    }
    for range in [
        0..0,
        std::ops::Range { start: 3, end: 2 },
        0..41,
        usize::MAX..usize::MAX,
    ] {
        assert!(first.read_parameter(&parameter, range).is_err());
    }
    assert!(first.parameter("foreign").is_err());
    assert!(first.parameter("matrix").is_err());
    assert!(first.parameter("not_vector").is_err());
    assert!(first.matrix("norm", MatrixAxes::OutputInput).is_err());
    assert_eq!(reserved(&host), 2 * META);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
    drop((parameter, first, second));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn parameter_window_keeps_raw_bytes_and_metadata_live_after_all_caller_owners_drop() {
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let parameter = source.parameter("norm").unwrap();
    let identity = source.source_identity();
    let window = source
        .read_parameter(&parameter, 0..parameter.elements())
        .unwrap();
    let staging = window.charged_staging_bytes() as u64;
    drop((source, parameter));
    assert_eq!(reserved(&host), META + staging);
    assert_eq!(window.parameter().source_identity(), identity);
    assert_eq!(window.parameter().source_parameter().shape, [40u64]);
    assert_eq!(window.as_slice(), fixture.norm);
    drop(window);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn parameter_actual_header_and_retained_fd_ignore_public_info_and_path_replacement() {
    use zerogravity_core::quant::QuantKind;
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let mut artifact = fixture.artifact();
    let untrusted = artifact
        .info
        .tensors
        .iter_mut()
        .find(|tensor| tensor.name == "norm")
        .unwrap();
    untrusted.kind = QuantKind::F32;
    untrusted.shape = vec![1, 40];
    untrusted.byte_len = 1;
    untrusted.byte_offset = 0;
    let plan = fixture.plan_with_artifact(artifact, &host).unwrap();
    fs::rename(
        fixture.path.join(PARAMETERS),
        fixture.path.join("original-retained"),
    )
    .unwrap();
    fs::write(fixture.path.join(PARAMETERS), vec![0; fixture.bytes.len()]).unwrap();
    let source = plan.open().unwrap();
    let parameter = source.parameter("norm").unwrap();
    assert_eq!(parameter.tensor().kind, QuantKind::Bf16);
    assert_eq!(parameter.tensor().shape, [40u64]);
    assert_eq!(parameter.source_parameter().bytes, 80);
    let window = source.read_parameter(&parameter, 0..40).unwrap();
    assert_eq!(window.as_slice(), fixture.norm);
    drop((source, parameter, window));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn changed_parameter_source_or_short_read_never_publishes_mutated_bytes() {
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let parameter = source.parameter("norm").unwrap();
    let before = source.read_parameter(&parameter, 0..40).unwrap();
    let charge = reserved(&host);
    let mut changed = fixture.bytes.clone();
    changed[parameter.tensor().byte_offset as usize + 79] ^= 1;
    fs::write(fixture.path.join(PARAMETERS), changed).unwrap();
    assert!(source.read_parameter(&parameter, 20..40).is_err());
    assert_eq!(reserved(&host), charge);
    assert_eq!(before.as_slice(), fixture.norm);
    File::create(fixture.path.join(PARAMETERS))
        .unwrap()
        .set_len(12)
        .unwrap();
    assert!(source.read_parameter(&parameter, 0..1).is_err());
    assert_eq!(reserved(&host), charge);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
    drop((source, parameter, before));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn parameter_cancel_and_real_pool_exhaustion_preserve_prior_stage_and_release_all_charges() {
    use zerogravity_core::resource_arbiter::AllocationRequest;
    let fixture = ParameterFixture::new();
    let host = admission(META + 4096, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    let parameter = source.parameter("norm").unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(
        source
            .read_parameter_cancellable(&parameter, 0..40, &cancelled)
            .is_err()
    );
    assert_eq!(reserved(&host), META);
    let occupied = host
        .arbiter
        .admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: 4096,
            }],
        )
        .unwrap();
    assert!(matches!(
        source.read_parameter(&parameter, 0..40),
        Err(Error::Limit(_))
    ));
    assert_eq!(reserved(&host), META + 4096);
    drop(occupied);
    cancelled.store(false, Ordering::Release);
    let before = source
        .read_parameter_cancellable(&parameter, 0..40, &cancelled)
        .unwrap();
    let charge = reserved(&host);
    cancelled.store(true, Ordering::Release);
    assert!(
        source
            .read_parameter_cancellable(&parameter, 0..40, &cancelled)
            .is_err()
    );
    assert_eq!(reserved(&host), charge);
    assert_eq!(before.as_slice(), fixture.norm);
    drop((source, parameter, before));
    assert_eq!(reserved(&host), 0);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
}

#[test]
fn parameter_range_inherits_bounded_authenticated_chunk_count_before_staging() {
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let source = fixture
        .plan_with_chunk_bytes(fixture.artifact(), &host, 1)
        .unwrap()
        .open()
        .unwrap();
    let parameter = source.parameter("norm").unwrap();
    assert!(matches!(
        source.read_parameter(&parameter, 0..40),
        Err(Error::Limit(_))
    ));
    assert_eq!(reserved(&host), META);
    let small = source.read_parameter(&parameter, 1..2).unwrap();
    assert_eq!(small.as_slice(), &fixture.norm[2..4]);
    assert!(!small.is_complete());
    assert_eq!(
        small.charged_staging_bytes(),
        2 + 2 * std::mem::size_of::<(usize, std::ops::Range<usize>)>()
    );
    drop((source, parameter, small));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn sealed_headers_and_same_owner_workspace_retain_metadata_and_refuse_exhaustion() {
    let fixture = Fixture::new();
    let host = admission(32 * 1024 * 1024, "host");
    let source = fixture.plan(&host).unwrap().open().unwrap();
    assert!(source.same_issuer(&source.clone()));
    let other = fixture
        .plan(&admission(32 * 1024 * 1024, "host"))
        .unwrap()
        .open()
        .unwrap();
    assert!(!source.same_issuer(&other));
    assert_eq!(source.tensor_info("matrix").unwrap().shape, [4, 4]);
    assert_eq!(source.tensor_names().count(), 2);
    assert!(source.tensor_info("foreign").is_err());
    for bytes in [0, 32 * 1024 * 1024 + 1] {
        assert!(source.reserve_workspace(bytes).is_err());
    }
    assert!(source.reserve_operator_buffers(0, 4).is_err());
    let retained = source.reserve_operator_buffers(4096, 8192).unwrap();
    assert_eq!(retained.bytes(), 12288);
    assert!(retained.source().same_issuer(&source));
    assert_eq!(reserved(&host), META + 12288);
    assert_eq!(
        host.arbiter.snapshot(&host.resource).unwrap().per_class[&AllocationClass::Output],
        4096
    );
    let pressure = source
        .reserve_workspace((32 * 1024 * 1024 - META - 12288) as usize)
        .unwrap();
    assert!(source.reserve_workspace(1).is_err());
    drop(pressure);
    drop(source);
    assert_eq!(reserved(&host), META + 12288);
    assert_eq!(
        retained.source().tensor_info("bf16").unwrap().kind,
        zerogravity_artifacts::quant::QuantKind::Bf16
    );
    drop(retained);
    assert_eq!(reserved(&host), 0);
    assert!(!host.arbiter.snapshot(&host.resource).unwrap().fenced);
}

#[path = "tensor_window/conv4.rs"]
mod conv4;

#[test]
fn successful_source_owner_retains_its_actual_authenticated_staging_chunk_geometry() {
    let fixture = ParameterFixture::new();
    let host = admission(64 * 1024 * 1024, "host");
    let first = fixture
        .plan_with_chunk_bytes(fixture.artifact(), &host, 64)
        .unwrap()
        .open()
        .unwrap();
    let second = fixture
        .plan_with_chunk_bytes(fixture.artifact(), &host, 1)
        .unwrap()
        .open()
        .unwrap();
    assert_eq!(first.staging_chunk_bytes(), 64);
    assert_eq!(second.staging_chunk_bytes(), 1);
    assert_eq!(first.clone().staging_chunk_bytes(), 64);
    assert!(!first.same_issuer(&second));
    let parameter = second.parameter("norm").unwrap();
    let window = second.read_parameter(&parameter, 1..2).unwrap();
    assert_eq!(
        window.charged_staging_bytes(),
        2 * (1 + std::mem::size_of::<(usize, std::ops::Range<usize>)>())
    );
    drop((first, second, parameter, window));
    assert_eq!(reserved(&host), 0);
}

#[test]
fn workspace_limit_observes_same_original_host_owner_without_new_credit() {
    for bytes in [64 * 1024 * 1024, 4096 * 1024 * 1024u64] {
        let fixture = Fixture::new();
        let host = admission(bytes, "host");
        let source = fixture.plan(&host).unwrap().open().unwrap();
        let charged = reserved(&host);
        assert_eq!(source.workspace_limit_bytes().unwrap(), bytes);
        assert_eq!(reserved(&host), charged);
        let retained = source.clone();
        drop(source);
        assert_eq!(retained.workspace_limit_bytes().unwrap(), bytes);
        assert_eq!(reserved(&host), charged);
        drop(retained);
        assert_eq!(reserved(&host), 0);
    }
}

#[test]
fn compiler_context_detached_host_lease_retains_only_original_domain_charge() {
    let f = Fixture::new();
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", f.path.to_str().unwrap()).unwrap();
    let other_key = PhysicalDeviceKey::new("host", &format!("{}-other", f.path.display())).unwrap();
    let limit = 64 * 1024 * 1024;
    for k in [&key, &other_key] {
        arbiter.register(k.clone(), limit).unwrap();
    }
    let host = TransferAdmission {
        arbiter: arbiter.clone(),
        resource: key.clone(),
        owner: ResourceOwner::new("detached-fixture", "source").unwrap(),
    };
    let other = TransferAdmission {
        arbiter: arbiter.clone(),
        resource: other_key.clone(),
        owner: host.owner.clone(),
    };
    let source = f.plan(&host).unwrap().open().unwrap();
    let alternate = f.plan(&other).unwrap().open().unwrap();
    let owner = || ResourceOwner::new("backend/test", "detached").unwrap();
    assert!(
        source
            .reserve_detached_host_workspace(limit + 1, owner(), 4096)
            .is_err()
    );
    assert!(
        source
            .reserve_detached_host_workspace(limit, owner(), 0)
            .is_err()
    );
    let lease = source
        .reserve_detached_host_workspace(limit, owner(), 4096)
        .unwrap();
    assert_eq!(
        source.detached_host_workspace_bytes(&lease, limit).unwrap(),
        4096
    );
    assert!(
        alternate
            .detached_host_workspace_bytes(&lease, limit)
            .is_err()
    );
    let isolated = f.plan(&admission(limit, "host")).unwrap().open().unwrap();
    assert!(
        isolated
            .reserve_detached_host_workspace(limit, owner(), 4096)
            .is_err()
    );
    drop((source, alternate, isolated));
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 4096);
    assert_eq!(arbiter.snapshot(&other_key).unwrap().reserved, 0);
    drop(lease);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}
