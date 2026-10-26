//! First prove the existing original Q4_0_8x8 helper, using produced teachers.
#![cfg(all(
    feature = "cpu",
    feature = "source-binding",
    target_os = "linux",
    target_arch = "x86_64"
))]

use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
use zerogravity_artifacts::{
    artifact::Artifact, cache::TransferAdmission, source_weights::SourceWeights,
};
use zerogravity_core::{
    accelerator::MatVecRegion,
    quant::QuantKind,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_model_qwen35::{
    LoadedQwenModel,
    operator_slice::bulk_f32::{BulkF32HostAdmission, GgufPrefixSlice},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SOURCE_BYTES: u64 = 1_296_764_000;
const SOURCE_HASH: &str = "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6";
const CAPTURE: &str =
    ".agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport";
fn read(path: &Path, length: usize, hash: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    assert_eq!(file.metadata()?.len(), length as u64);
    let mut bytes = Vec::with_capacity(length + 1);
    (&mut file)
        .take(length as u64 + 1)
        .read_to_end(&mut bytes)?;
    assert_eq!(bytes.len(), length);
    assert_eq!(
        format!("{:x}", Sha256::digest(&bytes)),
        hash,
        "{}",
        path.display()
    );
    Ok(bytes)
}
fn words(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn same(stage: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{stage} word{index} actual={:08x} expected={:08x}",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}
fn reserved(admission: &BulkF32HostAdmission) -> u64 {
    admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .reserved
}

#[test]
#[ignore = "requires pinned original 2B source/captures; ROOT executes serialized"]
fn original_source_q4_helpers_replay_qkv_and_z() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("ZG_REFERENCE_ROOT").expect("explicit approved reference workspace"),
    );
    assert!(root.is_absolute());
    let capture = root.join(CAPTURE);
    let model_dir = root.join("target/models/qwen35-2b");
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "original-upstream-helper-replay")?;
    arbiter.register(resource.clone(), 4 * 1024 * 1024 * 1024)?;
    let admission = BulkF32HostAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original-2b", "upstream-source-helper")?,
    };
    let metadata = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 16 * 1024 * 1024,
        }],
    )?;
    let staging = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let opening = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::ImmutableWeights,
            bytes: SOURCE_BYTES,
        }],
    )?;
    let parsed = Artifact::open(&model_dir.join("model.gguf"))?;
    let weights = SourceWeights::from_verified_artifact(
        parsed,
        SOURCE_BYTES,
        SOURCE_HASH,
        &TransferAdmission {
            arbiter: admission.arbiter.clone(),
            resource: admission.resource.clone(),
            owner: admission.owner.clone(),
        },
    )?;
    drop(opening);
    let config = read(
        &model_dir.join("config.json"),
        2908,
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    )?;
    let model = LoadedQwenModel::from_source(weights, &config)?;
    drop(config);
    let weak = Arc::downgrade(&model);
    let raw_ids = read(
        &capture.join("input_ids.u32le"),
        32768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
    )?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&raw_ids[..256])),
        "ff7862336b7b2b5520723bf753d0f9e484ef9e66f285efd6189c1d945ec17625"
    );
    let ids = raw_ids[..256]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    drop(raw_ids);
    let slice = GgufPrefixSlice::new(model.clone(), 0, 8448, &ids, admission.clone(), None)?;
    let prefix = slice.execute(None)?;
    drop(slice);
    // Expected normalized values are checked independently before the leaf replay.
    same(
        "produced normalized",
        prefix.normalized()?,
        &words(&read(
            &capture.join("bulk64/field-4.f32le"),
            524288,
            "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
        )?),
    );
    let baseline = reserved(&admission);
    for (name, rows, field, hash) in [
        (
            "blk.0.attn_qkv.weight",
            6144,
            6,
            "8e4d4ae5fd79e14204de2064b5f0bfa7fa502964595ceb4e3a2f0abef7cf9b05",
        ),
        (
            "blk.0.attn_gate.weight",
            2048,
            38,
            "6b20f53dce195fb2b45442baa9ff3785d8a11112ed55cbe31d471a3600ddfa18",
        ),
    ] {
        let tensor = model.tensor_info(name)?;
        assert_eq!(tensor.kind, QuantKind::Q4_0);
        assert_eq!(tensor.shape, [2048, rows as u64]);
        let source = model.tensor_bytes(name)?;
        assert_eq!(source.len(), rows * 1152);
        let output = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Output,
                bytes: (rows * 64 * 4) as u64,
            }],
        )?;
        // One returned output row and Q8 dynamic activation blocks/headers.
        let scratch = admission.arbiter.admit(
            &admission.resource,
            admission.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: (rows * 4 + 4096) as u64,
            }],
        )?;
        let mut actual = Vec::with_capacity(rows * 64);
        for activation in prefix.normalized()?.chunks_exact(2048) {
            zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
            let values = zerogravity_cpu::source::source_reference_projection(MatVecRegion {
                source_identity: model.source_identity(),
                tensor_name: name,
                kind: tensor.kind,
                columns: 2048,
                rows,
                packed_weights: source,
                activation,
            })?;
            assert_eq!(values.capacity(), rows);
            actual.extend_from_slice(&values);
        }
        zerogravity_cpu::numerics::haswell_bulk_f32::verify_environment()?;
        // Native expected output is first read after the original helper replay.
        let expected = words(&read(
            &capture.join(format!("bulk64/field-{field}.f32le")),
            rows * 64 * 4,
            hash,
        )?);
        same(name, &actual, &expected);
        eprintln!(
            "existing Q4_0_8x8 helper exact {name}: {} words",
            actual.len()
        );
        drop(expected);
        drop(actual);
        drop(scratch);
        drop(output);
    }
    assert_eq!(reserved(&admission), baseline);
    drop(model);
    assert!(weak.upgrade().is_some());
    drop(prefix);
    assert!(weak.upgrade().is_none());
    drop(ids);
    drop(staging);
    drop(metadata);
    assert_eq!(reserved(&admission), 0);
    Ok(())
}

#[test]
#[ignore = "requires pinned original 2B source/captures; ROOT executes serialized"]
fn actual_original_upstream_feeds_gdn_without_observed_intermediates() -> Result<()> {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use zerogravity_core::accelerator::program::ProgramNumericalProfile;
    use zerogravity_core::accelerator::{MatVecKernel, RegionExecution};
    use zerogravity_model_qwen35::operator_slice::bulk_f32::{
        BulkGdnCoefficientSlice, BulkGdnGatedSlice, BulkGdnRecurrentSlice, BulkGdnSsmOutSlice,
        GgufGdnUpstreamSlice,
    };
    struct Denied(AtomicUsize);
    impl MatVecKernel for Denied {
        fn try_execute(&self, _: MatVecRegion<'_>) -> zerogravity_core::Result<RegionExecution> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }
    let root = PathBuf::from(
        std::env::var_os("ZG_REFERENCE_ROOT").expect("explicit approved reference workspace"),
    );
    assert!(root.is_absolute());
    let capture = root.join(CAPTURE);
    let model_dir = root.join("target/models/qwen35-2b");
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "actual-original-gdn-upstream")?;
    arbiter.register(resource.clone(), 4 * 1024 * 1024 * 1024)?;
    let admission = BulkF32HostAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original-2b", "actual-upstream")?,
    };
    let metadata = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: 16 * 1024 * 1024,
        }],
    )?;
    let staging = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: 32 * 1024 * 1024,
        }],
    )?;
    let opening = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::ImmutableWeights,
            bytes: SOURCE_BYTES,
        }],
    )?;
    let weights = SourceWeights::from_verified_artifact(
        Artifact::open(&model_dir.join("model.gguf"))?,
        SOURCE_BYTES,
        SOURCE_HASH,
        &TransferAdmission {
            arbiter: admission.arbiter.clone(),
            resource: admission.resource.clone(),
            owner: admission.owner.clone(),
        },
    )?;
    drop(opening);
    let config = read(
        &model_dir.join("config.json"),
        2908,
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    )?;
    let model = LoadedQwenModel::from_source(weights, &config)?;
    drop(config);
    let weak = Arc::downgrade(&model);
    let source = model.source_identity();
    let raw_ids = read(
        &capture.join("input_ids.u32le"),
        32768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
    )?;
    let ids = raw_ids[..256]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    drop(raw_ids);
    let prefix_slice = GgufPrefixSlice::new(model.clone(), 0, 8448, &ids, admission.clone(), None)?;
    let prefix = prefix_slice.execute(None)?;
    let before = reserved(&admission);
    let cancelled = AtomicBool::new(true);
    assert!(GgufGdnUpstreamSlice::new(prefix.clone(), Some(&cancelled)).is_err());
    assert_eq!(reserved(&admission), before);
    let upstream_slice = GgufGdnUpstreamSlice::new(prefix.clone(), None)?;
    let plan = upstream_slice.record()?;
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(plan.plan().source(), source);
    assert_eq!(
        plan.plan().numerical_profile(),
        ProgramNumericalProfile::GgufSourceV1
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert_eq!(plan.plan().frontier_delta(), 0);
    assert!(plan.plan().logits().is_none());
    drop(plan);
    let before = reserved(&admission);
    assert!(upstream_slice.execute(Some(&cancelled)).is_err());
    let denied = Denied(AtomicUsize::new(0));
    assert!(
        upstream_slice
            .execute_with_provider(Some(&denied), None)
            .is_err()
    );
    assert_eq!(denied.0.load(Ordering::Relaxed), 0);
    assert_eq!(reserved(&admission), before);
    let remaining = admission.arbiter.snapshot(&admission.resource)?.remaining;
    let pressure = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: remaining,
        }],
    )?;
    let pressed = reserved(&admission);
    assert!(upstream_slice.execute(None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    // QKV completes privately, then Z admission refuses and releases QKV.
    let remaining = admission.arbiter.snapshot(&admission.resource)?.remaining;
    let pressure = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: remaining - (6144 * 64 * 4 + 6144 * 4 + 4096),
        }],
    )?;
    let pressed = reserved(&admission);
    assert!(upstream_slice.execute(None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    let upstream = upstream_slice.execute(None)?;
    assert_eq!(upstream.descriptor().source(), source);
    assert_eq!(upstream.prefix().span(), prefix.span());
    // All expected tensors are first opened AFTER actual upstream arithmetic.
    let field = |index: usize, count: usize, hash: &str| -> Result<Vec<f32>> {
        Ok(words(&read(
            &capture.join(format!("bulk64/field-{index}.f32le")),
            count * 4,
            hash,
        )?))
    };
    for (name, actual, index, count, hash) in [
        (
            "actual QKV",
            upstream.qkv()?,
            6,
            393216,
            "8e4d4ae5fd79e14204de2064b5f0bfa7fa502964595ceb4e3a2f0abef7cf9b05",
        ),
        (
            "actual Z",
            upstream.z()?,
            38,
            131072,
            "6b20f53dce195fb2b45442baa9ff3785d8a11112ed55cbe31d471a3600ddfa18",
        ),
        (
            "actual conv raw",
            upstream.convolution_raw()?,
            10,
            393216,
            "a56f58a8e0d6929b8fffbb4211579309c375578aaf059312aafd1f8432bace14",
        ),
        (
            "actual conv SiLU",
            upstream.convolution_activated()?,
            11,
            393216,
            "226694ecc32852c1870b47114975f2b03dc3d38a92fc09cc521519df698f8db5",
        ),
        (
            "actual Q",
            upstream.query()?,
            26,
            131072,
            "3bb137807d2ec8e03b2afc2ea7b62648cd3baef2b526d0b3197da415b1582f20",
        ),
        (
            "actual K",
            upstream.key()?,
            27,
            131072,
            "a698d1d7838fb073f3a9ce694b16a2e93dbc2a2c944d4898ce9b3c0740e288c0",
        ),
        (
            "actual V",
            upstream.value()?,
            28,
            131072,
            "a7410d7bec7d53c3edb571f829f1e0e7d65cf154c27420c1167b7fafc83a8bfc",
        ),
    ] {
        same(name, actual, &field(index, count, hash)?);
    }
    let native_concat = field(
        9,
        411648,
        "9e05d485360096181482324227d4b93d4cc4f1c1e6cc9efd32230f0c66c2f4e6",
    )?;
    let history = upstream.final_convolution_history()?;
    assert_eq!(history.len(), 18432);
    for channel in 0..6144 {
        assert!(
            native_concat[channel * 67..channel * 67 + 3]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        same(
            "actual final conv history",
            &history[channel * 3..channel * 3 + 3],
            &native_concat[channel * 67 + 64..channel * 67 + 67],
        );
    }
    drop(native_concat);
    let transfer = admission
        .arbiter
        .snapshot(&admission.resource)?
        .per_class
        .get(&AllocationClass::TransferStaging)
        .copied()
        .unwrap_or(0);
    let coefficients = BulkGdnCoefficientSlice::from_prefix(prefix.clone(), None)?.execute(None)?;
    // Same textual IDs/issuer but independently executed normalized storage is foreign.
    let wrong_prefix = prefix_slice.execute(None)?;
    assert_eq!(wrong_prefix.span(), prefix.span());
    let wrong_coefficients =
        BulkGdnCoefficientSlice::from_prefix(wrong_prefix, None)?.execute(None)?;
    let before = reserved(&admission);
    assert!(
        BulkGdnRecurrentSlice::from_upstream(wrong_coefficients, upstream.clone(), None).is_err()
    );
    assert!(reserved(&admission) < before);
    let before = reserved(&admission);
    assert!(
        BulkGdnRecurrentSlice::from_upstream(
            coefficients.clone(),
            upstream.clone(),
            Some(&cancelled)
        )
        .is_err()
    );
    assert_eq!(reserved(&admission), before);
    let recurrent_slice =
        BulkGdnRecurrentSlice::from_upstream(coefficients, upstream.clone(), None)?;
    let recurrent = recurrent_slice.execute(None)?;
    // Native initial state is independently verified, never supplied to Rust.
    let native_prior = field(
        31,
        262144,
        "30e14955ebf1352266dc2ff8067e68104607e750abb9d3b36582b8af909fcb58",
    )?;
    assert!(native_prior.iter().all(|v| v.to_bits() == 0));
    drop(native_prior);
    same(
        "actual upstream GDN core",
        recurrent.core(),
        &field(
            32,
            131072,
            "1e10aa12ae051d7b6dc853f49e4302b99d833a61223bc3ce951ebe9de9afcf6d",
        )?,
    );
    same(
        "actual upstream final GDN state",
        recurrent.state(),
        &field(
            33,
            262144,
            "900193c0c231f4f4194628dce46f17b6d41982db06c9dd5a37e886c6b7cec2b1",
        )?,
    );
    let gated_slice = BulkGdnGatedSlice::from_upstream(recurrent, upstream.clone(), None)?;
    let gated = gated_slice.execute(None)?;
    same(
        "actual upstream gated output",
        gated.gated(),
        &field(
            41,
            131072,
            "4400491b963ae5fa113425e97ed6dafc3e8ca74b7af406fe77578bf5a6bee486",
        )?,
    );
    let projected_slice = BulkGdnSsmOutSlice::new(gated, None)?;
    let projected = projected_slice.execute(None)?;
    same(
        "actual original complete GDN projection",
        projected.values()?,
        &field(
            40,
            131072,
            "5477c01a3885382f18640bf835ec10b00e1d5b2611b5fb1c1889653b64027059",
        )?,
    );
    assert_eq!(
        admission
            .arbiter
            .snapshot(&admission.resource)?
            .per_class
            .get(&AllocationClass::TransferStaging)
            .copied()
            .unwrap_or(0),
        transfer,
        "actual intermediate producers retain backing without observation copies"
    );
    drop(projected_slice);
    drop(gated_slice);
    drop(recurrent_slice);
    drop(upstream_slice);
    drop(prefix_slice);
    drop(upstream);
    drop(prefix);
    drop(model);
    assert!(weak.upgrade().is_some());
    let clone = projected.clone();
    drop(projected);
    assert!(weak.upgrade().is_some());
    drop(clone);
    assert!(weak.upgrade().is_none());
    drop(ids);
    drop(staging);
    drop(metadata);
    assert_eq!(reserved(&admission), 0);
    eprintln!(
        "ACTUAL_GDN_UPSTREAM_PASS original_QKV_Z_conv_QK=true prior_zero_private=true observed_intermediates=false last_source_clone_retained=true final_reserved=0 full_model_unqualified=true"
    );
    Ok(())
}
