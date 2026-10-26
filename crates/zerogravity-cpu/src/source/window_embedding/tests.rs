//! Reuse authenticated synthetic source fixture; no trained model proof.
use super::*;
use crate::accelerator::EmbeddingRegion;
use crate::source::source_reference_embedding;
use std::{
    fs::{self, File},
    path::Path,
    sync::atomic::AtomicU64,
};
use zerogravity_artifacts::{
    artifact::Artifact,
    cache::TransferAdmission,
    source_weights::{ExpectedSourceShard, SourceOpenLimits, SourceWeights, WindowOpenLimits},
};
use zerogravity_core::{
    resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner},
    source::TensorSource,
};
const DATA: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
#[test]
fn real_full_descriptor_rows_match_existing_decoder_and_refuse_wrong_range_codec_or_source() {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let path = std::env::temp_dir().join(format!(
        "zg-window-embedding-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&path).unwrap();
    struct Remove(std::path::PathBuf);
    impl Drop for Remove {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let _remove = Remove(path.clone());
    fs::write(path.join("model.safetensors"), DATA).unwrap();
    let shard = || {
        vec![
            ExpectedSourceShard::new(
                "model.safetensors".into(),
                Artifact::from_file(
                    File::open(path.join("model.safetensors")).unwrap(),
                    Path::new("model.safetensors"),
                )
                .unwrap(),
                DATA.len() as u64,
                "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
            )
            .unwrap(),
        ]
    };
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "embedding-fixture").unwrap();
    arbiter.register(key.clone(), 64 * 1024 * 1024).unwrap();
    let host = TransferAdmission {
        arbiter: arbiter.clone(),
        resource: key.clone(),
        owner: ResourceOwner::new("fixture", "embedding").unwrap(),
    };
    let digest = "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149";
    let source = SourceWeights::prepare_verified_windows(
        INDEX,
        digest,
        shard(),
        WindowOpenLimits {
            max_source_bytes: 1024 * 1024,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 6,
            scan_chunk_bytes: 1024,
        },
        &host,
    )
    .unwrap()
    .open()
    .unwrap();
    let complete = SourceWeights::prepare_verified_shards(
        INDEX,
        digest,
        shard(),
        SourceOpenLimits {
            max_owned_bytes: 1024 * 1024,
            max_metadata_bytes: 16 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 6,
        },
        &host,
    )
    .unwrap()
    .materialize()
    .unwrap();
    for name in ["original.weight", "wide.weight"] {
        let matrix = source.matrix(name, MatrixAxes::OutputInput).unwrap();
        let descriptor = SourceTensor {
            source: source.source_identity(),
            name: name.into(),
            kind: matrix.tensor().kind,
            axes: matrix.axes(),
            columns: matrix.columns(),
            rows: matrix.rows(),
        };
        let raw = TensorSource::tensor(&complete, name).unwrap();
        for token in 0..matrix.rows() {
            let window = source.read_rows(&matrix, token..token + 1).unwrap();
            let actual =
                embedding_bf16_window(&descriptor, token as u32, &window, None, None).unwrap();
            let expected = source_reference_embedding(EmbeddingRegion {
                source_identity: complete.source_identity(),
                tensor_name: name,
                kind: raw.kind,
                packed_weights: raw.packed_bytes,
                token: token as u32,
                width: matrix.columns(),
                vocabulary: matrix.rows(),
            })
            .unwrap();
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert!(embedding_bf16_window(&descriptor, u32::MAX, &window, None, None).is_err());
            assert!(
                embedding_bf16_window(
                    &descriptor,
                    token as u32,
                    &window,
                    None,
                    Some(&AtomicBool::new(true))
                )
                .is_err()
            );
            for changed in [
                SourceTensor {
                    source: zerogravity_core::accelerator::SourceIdentity::for_new_allocation()
                        .unwrap(),
                    ..descriptor.clone()
                },
                SourceTensor {
                    axes: MatrixAxes::InputOutput,
                    ..descriptor.clone()
                },
                SourceTensor {
                    columns: matrix.columns() - 1,
                    ..descriptor.clone()
                },
            ] {
                assert!(
                    embedding_bf16_window(&changed, token as u32, &window, None, None).is_err()
                );
            }
        }
    }
    let bad = source
        .matrix("nonfinite.weight", MatrixAxes::OutputInput)
        .unwrap();
    let descriptor = SourceTensor {
        source: source.source_identity(),
        name: "nonfinite.weight".into(),
        kind: bad.tensor().kind,
        axes: bad.axes(),
        columns: bad.columns(),
        rows: bad.rows(),
    };
    let refused = (0..bad.rows())
        .filter(|&row| {
            let window = source.read_rows(&bad, row..row + 1).unwrap();
            embedding_bf16_window(&descriptor, row as u32, &window, None, None).is_err()
        })
        .count();
    assert!(refused > 0);
    drop(bad);
    drop(complete);
    drop(source);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    assert!(!arbiter.snapshot(&key).unwrap().fenced);
}
