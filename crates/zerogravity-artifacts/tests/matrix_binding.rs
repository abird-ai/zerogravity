//! Exact sealed resident operand joins; tiny real authenticated source, no GPU.
#![cfg(all(feature = "cache", target_os = "linux"))]
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits},
};
use zerogravity_core::{
    accelerator::program::MatrixAxes,
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner},
};
const META: u64 = 16 * 1024 * 1024;
const SHARD: &str = "model.safetensors";
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
struct Fixture {
    path: PathBuf,
    file: Vec<u8>,
    index: Vec<u8>,
    host: TransferAdmission,
}
impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let path = std::env::temp_dir().join(format!(
            "zg-exact-matrix-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        let header = serde_json::to_vec(&json!({
            "matrix_a":{"dtype":"BF16","shape":[2,8],"data_offsets":[0,32]},
            "matrix_b":{"dtype":"BF16","shape":[2,8],"data_offsets":[32,64]},
            "expert":{"dtype":"BF16","shape":[2,2,8],"data_offsets":[64,128]}
        }))
        .unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        for value in [1u16, 2, 3, 4] {
            for _ in 0..16 {
                file.extend(value.to_le_bytes());
            }
        }
        fs::write(path.join(SHARD), &file).unwrap();
        let index = serde_json::to_vec(&json!({"metadata":{"total_size":128},
            "weight_map":{"matrix_a":SHARD,"matrix_b":SHARD,"expert":SHARD}}))
        .unwrap();
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "exact-matrix-source").unwrap();
        arbiter.register(resource.clone(), 4 * META).unwrap();
        let host = TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("authenticated-matrix", "exact-member-range").unwrap(),
        };
        Self {
            path,
            file,
            index,
            host,
        }
    }
    fn open(&self, metadata: u64) -> VerifiedTensorSource {
        let shard = ExpectedSourceShard::new(
            SHARD.into(),
            Artifact::from_file(
                File::open(self.path.join(SHARD)).unwrap(),
                std::path::Path::new(SHARD),
            )
            .unwrap(),
            self.file.len() as u64,
            sha(&self.file),
        )
        .unwrap();
        SourceWeights::prepare_verified_windows(
            &self.index,
            &sha(&self.index),
            vec![shard],
            WindowOpenLimits {
                max_source_bytes: 4096,
                max_metadata_bytes: metadata,
                max_shards: 1,
                max_tensors: 3,
                scan_chunk_bytes: 64,
            },
            &self.host,
        )
        .unwrap()
        .open()
        .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}
#[test]
fn resident_operand_rejects_same_source_same_shape_wrong_member_or_range() {
    let fixture = Fixture::new();
    let source = fixture.open(META);
    let a = source.matrix("matrix_a", MatrixAxes::OutputInput).unwrap();
    let same = source
        .clone()
        .matrix("matrix_a", MatrixAxes::OutputInput)
        .unwrap();
    let b = source.matrix("matrix_b", MatrixAxes::OutputInput).unwrap();
    assert!(a.same_binding(&a.clone()));
    assert!(a.same_binding(&same));
    assert_eq!(a.source_identity(), b.source_identity());
    assert_eq!(
        (a.rows(), a.columns(), a.tensor().kind),
        (b.rows(), b.columns(), b.tensor().kind)
    );
    assert_ne!(a.tensor().byte_offset, b.tensor().byte_offset);
    assert!(!a.same_binding(&b));
    assert_ne!(
        source.read_rows(&a, 0..1).unwrap().as_slice(),
        source.read_rows(&b, 0..1).unwrap().as_slice()
    );
    // Same issuer AND exact tensor member; only the authenticated plane differs.
    let first = source.expert_matrix("expert", 0).unwrap();
    let second = source.expert_matrix("expert", 1).unwrap();
    assert_eq!(first.source_identity(), second.source_identity());
    assert_eq!(
        (first.rows(), first.columns()),
        (second.rows(), second.columns())
    );
    assert_eq!(first.tensor().name, second.tensor().name);
    assert_eq!(first.tensor().byte_offset, second.tensor().byte_offset);
    assert!(first.same_binding(&source.expert_matrix("expert", 0).unwrap()));
    assert!(!first.same_binding(&second));
    source.require_matrix_binding(&first).unwrap();
    source.require_matrix_binding(&second).unwrap();
    let first_range = first.byte_range().unwrap();
    let second_range = second.byte_range().unwrap();
    assert_eq!(first_range.start, first.tensor().byte_offset);
    assert_eq!(first_range.end, second_range.start);
    assert_eq!(first_range.end - first_range.start, 32);
    assert_eq!(second_range.end - second_range.start, 32);
    assert_eq!(
        second_range.end,
        second.tensor().byte_offset + second.tensor().byte_len
    );
    assert_eq!(
        source.read_rows(&second, 0..2).unwrap().source_offset(),
        second_range.start
    );

    assert_ne!(
        source.read_rows(&first, 0..1).unwrap().source_offset(),
        source.read_rows(&second, 0..1).unwrap().source_offset()
    );
    // Equal file/header/member geometry still does not replace the original issuer.
    let foreign = fixture.open(META - 4096);
    assert!(!source.same_issuer(&foreign));
    assert!(foreign.require_matrix_binding(&first).is_err());
    assert!(foreign.require_matrix_binding(&a).is_err());
    let foreign_a = foreign.matrix("matrix_a", MatrixAxes::OutputInput).unwrap();
    assert_eq!(a.tensor().name, foreign_a.tensor().name);
    assert_eq!(a.tensor().byte_offset, foreign_a.tensor().byte_offset);
    assert!(!a.same_binding(&foreign_a));
    assert!(source.matrix("matrix_a", MatrixAxes::InputOutput).is_err());
    drop((a, same, b, first, second, foreign_a, foreign, source));
    assert_eq!(
        fixture
            .host
            .arbiter
            .snapshot(&fixture.host.resource)
            .unwrap()
            .reserved,
        0
    );
}
