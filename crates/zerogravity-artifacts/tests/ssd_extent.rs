//! SSD extent foundation. Source-only: no trained tensor geometry or GPU.
#![cfg(all(feature = "cache", target_os = "linux"))]

use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use zerogravity_artifacts::Error;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::ssd_extent::ExtentSource;
use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let folder = std::env::temp_dir().join(format!(
            "zg-ssd-extents-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&folder).unwrap();
        Self(folder)
    }
    fn write(&self, bytes: &[u8]) -> PathBuf {
        let path = self.0.join("original.bin");
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn manifest(bytes: &[u8], chunk: usize) -> Vec<u8> {
    assert!(bytes.len().is_multiple_of(chunk));
    let chunks = bytes
        .chunks(chunk)
        .enumerate()
        .map(|(i, part)| {
            json!({
                "offset":i*chunk,"bytes":chunk,"sha256":sha(part)
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_vec(&json!({
        "schema":"zerogravity.extents.v1",
        "file_bytes":bytes.len(),
        "whole_sha256":sha(bytes),"chunks":chunks
    }))
    .unwrap()
}
fn admission(bytes: u64) -> (TransferAdmission, PhysicalDeviceKey) {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host-pool", "ssd-test").unwrap();
    arbiter.register(key.clone(), bytes).unwrap();
    let admission = TransferAdmission {
        arbiter,
        resource: key.clone(),
        owner: ResourceOwner::new("immutable-model", "ssd-reads").unwrap(),
    };
    (admission, key)
}
#[test]
fn exact_source_chunks_are_verified_coalesced_and_pinned_through_drop() {
    let fx = Fixture::new();
    let raw = (0..4096u32).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let path = fx.write(&raw);
    let index = manifest(&raw, 4096);
    let source =
        ExtentSource::from_open_file(File::open(&path).unwrap(), &index, &sha(&index)).unwrap();
    assert_eq!(source.chunk_count(), 4);
    assert_eq!(source.file_bytes(), raw.len() as u64);
    assert_eq!(source.index_sha256(), sha(&index));
    assert_eq!(source.claimed_whole_sha256(), sha(&raw));
    let (admit, key) = admission(16 * 1024);
    let first = source.read_verified(&[1, 2], &admit).unwrap();
    assert_eq!(first.count(), 2);
    assert_eq!(first.len(), 8192);
    assert_eq!(first.bytes_for(1).unwrap(), &raw[4096..8192]);
    assert_eq!(first.bytes_for(2).unwrap(), &raw[8192..12288]);
    assert!(first.bytes_for(0).is_none());
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 8192);
    let second = source.read_verified(&[0, 3], &admit).unwrap();
    assert_eq!(second.len(), 8192);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 16384);
    assert_eq!(second.bytes_for(3).unwrap(), &raw[12288..]);
    drop(second);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 8192);
    drop(first);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn pinned_index_and_exact_size_must_match_before_reads() {
    let fx = Fixture::new();
    let raw = vec![7u8; 8192];
    let path = fx.write(&raw);
    let index = manifest(&raw, 4096);
    assert!(
        ExtentSource::from_open_file(File::open(&path).unwrap(), &index, &"0".repeat(64)).is_err()
    );
    let mut obj: serde_json::Value = serde_json::from_slice(&index).unwrap();
    obj["unknown"] = json!("execute-me");
    let changed = serde_json::to_vec(&obj).unwrap();
    assert!(
        ExtentSource::from_open_file(File::open(&path).unwrap(), &changed, &sha(&changed)).is_err()
    );
    obj = serde_json::from_slice(&index).unwrap();
    obj["chunks"][0]["offset"] = json!(1);
    let changed = serde_json::to_vec(&obj).unwrap();
    assert!(
        ExtentSource::from_open_file(File::open(&path).unwrap(), &changed, &sha(&changed)).is_err()
    );
    obj = serde_json::from_slice(&index).unwrap();
    obj["chunks"].as_array_mut().unwrap().remove(1);
    let changed = serde_json::to_vec(&obj).unwrap();
    assert!(
        ExtentSource::from_open_file(File::open(&path).unwrap(), &changed, &sha(&changed)).is_err()
    );
    obj = serde_json::from_slice(&index).unwrap();
    obj["whole_sha256"] = json!("bad");
    let changed = serde_json::to_vec(&obj).unwrap();
    assert!(
        ExtentSource::from_open_file(File::open(&path).unwrap(), &changed, &sha(&changed)).is_err()
    );
}
#[test]
fn mutation_short_io_and_physical_budget_fail_closed() {
    let fx = Fixture::new();
    let raw = vec![55u8; 8192];
    let path = fx.write(&raw);
    let index = manifest(&raw, 4096);
    let source =
        ExtentSource::from_open_file(File::open(&path).unwrap(), &index, &sha(&index)).unwrap();
    let (small, key) = admission(4096);
    assert!(matches!(
        source.read_verified(&[0, 1], &small),
        Err(Error::Limit(_))
    ));
    assert_eq!(small.arbiter.snapshot(&key).unwrap().reserved, 0);
    let (large, key) = admission(8192);
    for bad in [&[][..], &[0, 0][..], &[1, 0][..], &[4][..]] {
        assert!(source.read_verified(bad, &large).is_err());
        assert_eq!(large.arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    let changed = vec![56u8; 8192];
    fs::write(&path, changed).unwrap();
    assert!(source.read_verified(&[0, 1], &large).is_err());
    assert_eq!(large.arbiter.snapshot(&key).unwrap().reserved, 0);
    File::create(&path).unwrap().set_len(100).unwrap();
    assert!(source.read_verified(&[0], &large).is_err());
    assert_eq!(large.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn pathname_replacement_keeps_opened_descriptor_identity() {
    let fx = Fixture::new();
    let raw = vec![88u8; 8192];
    let path = fx.write(&raw);
    let index = manifest(&raw, 4096);
    let source =
        ExtentSource::from_open_file(File::open(&path).unwrap(), &index, &sha(&index)).unwrap();
    let retained = fx.0.join("former");
    fs::rename(&path, &retained).unwrap();
    fs::write(&path, vec![99u8; 8192]).unwrap();
    let (admit, key) = admission(8192);
    let staged = source.read_verified(&[0, 1], &admit).unwrap();
    assert_eq!(staged.bytes_for(0).unwrap(), &raw[..4096]);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 8192);
    // Mutation of the original descriptor after a staged copy cannot
    // rewrite the still-referenced immutable bytes.
    fs::write(&retained, vec![0u8; 8192]).unwrap();
    assert_eq!(staged.bytes_for(0).unwrap(), &raw[..4096]);
    assert!(source.read_verified(&[0], &admit).is_err());
    drop(staged);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
#[test]
fn sparse_multi_gib_source_manifest_never_requires_multi_gib_residency() {
    let fx = Fixture::new();
    let path = fx.0.join("huge.bin");
    let size = 5u64 * 1024 * 1024 * 1024;
    let mut file = File::create(&path).unwrap();
    file.set_len(size).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(&[0u8; 32]).unwrap();
    drop(file);
    let chunks = (0..(size / (4 * 1024 * 1024)))
        .map(|i| {
            json!({
                "offset":i*4*1024*1024u64,
                "bytes":4*1024*1024u64,
                "sha256":"0".repeat(64),
            })
        })
        .collect::<Vec<_>>();
    let index = serde_json::to_vec(&json!({
        "schema":"zerogravity.extents.v1","file_bytes":size,
        "whole_sha256":"0".repeat(64),"chunks":chunks,
    }))
    .unwrap();
    let source =
        ExtentSource::from_open_file(File::open(&path).unwrap(), &index, &sha(&index)).unwrap();
    assert_eq!(source.file_bytes(), size);
    let (admit, key) = admission(1024);
    assert!(matches!(
        source.read_verified(&[0], &admit),
        Err(Error::Limit(_))
    ));
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
    // No sparse payload read; metadata-only source admission.
}

#[test]
fn bounded_tensor_windows_copy_only_verified_requested_chunks() {
    let fx = Fixture::new();
    let raw = (0..4096u32).flat_map(u32::to_le_bytes).collect::<Vec<_>>();
    let path = fx.write(&raw);
    let index = manifest(&raw, 4096);
    let source =
        ExtentSource::from_open_file(File::open(&path).unwrap(), &index, &sha(&index)).unwrap();
    let (admit, key) = admission(12 * 1024);
    let window = source.read_window(4096 - 16, 64, &admit).unwrap();
    assert_eq!(window.as_slice(), &raw[4096 - 16..4096 + 48]);
    assert_eq!(window.source_offset(), 4080);
    assert_eq!(window.charged_bytes(), 8192);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 8192);
    assert!(source.read_window(0, 0, &admit).is_err());
    assert!(source.read_window(u64::MAX, 1, &admit).is_err());
    assert!(source.read_window(raw.len() as u64 - 1, 2, &admit).is_err());
    assert!(source.read_window(1, 4 * 4096, &admit).is_err()); // requires >12K
    drop(window);
    assert_eq!(admit.arbiter.snapshot(&key).unwrap().reserved, 0);
}
