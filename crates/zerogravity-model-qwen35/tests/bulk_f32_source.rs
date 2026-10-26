//! SAME engine operator slice only; no model forward, StateRoot or GPU permit.
#![cfg(all(
    feature = "cpu",
    feature = "source-binding",
    target_os = "linux",
    target_arch = "x86_64"
))]

use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_artifacts::{
    artifact::Artifact, cache::TransferAdmission, source_weights::SourceWeights,
};
use zerogravity_core::accelerator::program::{BulkF32Contract, BulkF32Layout, PlanOperation};
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::operator_slice::bulk_f32::{
    BulkF32HostAdmission, BulkF32Slice, ExternalBulkF32,
};
use zerogravity_model_qwen35::{LoadedQwenModel, source::binding::QwenMatrixRole};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const LIMIT: u64 = 4 * 1024 * 1024 * 1024;
const CONTROL: u64 = 1024 * 1024;
const SOURCE_METADATA: u64 = 16 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const BULK_METADATA: u64 = 16_384;

fn read_pinned(path: &Path, bytes: usize, hash: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let before = file.metadata()?;
    assert!(before.is_file());
    assert_eq!(before.len(), bytes as u64);
    let mut raw = Vec::with_capacity(bytes + 1);
    (&mut file).take(bytes as u64 + 1).read_to_end(&mut raw)?;
    let after = file.metadata()?;
    assert_eq!(raw.len(), bytes, "complete source and EOF: {path:?}");
    assert_eq!(
        (
            before.dev(),
            before.ino(),
            before.len(),
            before.mtime(),
            before.mtime_nsec(),
            before.ctime(),
            before.ctime_nsec()
        ),
        (
            after.dev(),
            after.ino(),
            after.len(),
            after.mtime(),
            after.mtime_nsec(),
            after.ctime(),
            after.ctime_nsec()
        )
    );
    assert_eq!(format!("{:x}", Sha256::digest(&raw)), hash, "{path:?}");
    Ok(raw)
}

fn words(raw: &[u8]) -> Vec<f32> {
    assert!(raw.len().is_multiple_of(4));
    raw.chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().expect("four bytes")))
        .collect()
}

fn output_hash(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    for word in values {
        hash.update(word.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

fn class(arbiter: &ResourceArbiter, key: &PhysicalDeviceKey, kind: AllocationClass) -> u64 {
    arbiter
        .snapshot(key)
        .expect("original ledger")
        .per_class
        .get(&kind)
        .copied()
        .unwrap_or(0)
}

struct DeniedProvider(AtomicUsize);
impl MatVecKernel for DeniedProvider {
    fn try_execute(&self, _region: MatVecRegion<'_>) -> zerogravity_core::Result<RegionExecution> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(RegionExecution::Unsupported)
    }
}

fn replay(
    root: &Path,
    ids: &[u32],
    arbiter: &ResourceArbiter,
    key: &PhysicalDeviceKey,
    owner: &ResourceOwner,
    av: bool,
) -> Result<usize> {
    let (alias, source_bytes, source_hash, config_bytes, config_hash) = if av {
        (
            "qwen35-0.8b",
            563_036_064,
            "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
            2907,
            "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
        )
    } else {
        (
            "qwen35-2b",
            1_296_764_000,
            "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
            2908,
            "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
        )
    };
    let metadata = arbiter.admit(
        key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: SOURCE_METADATA,
        }],
    )?;
    // Conservative source-open guard; the verified owner's separate ticket
    // remains attached to its actual immutable allocation after this drops.
    let opening = arbiter.admit(
        key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::ImmutableWeights,
            bytes: source_bytes,
        }],
    )?;
    let source_folder = root.join("target/models").join(alias);
    let parsed = Artifact::open(&source_folder.join("model.gguf"))?;
    assert_eq!(parsed.info.file_len, source_bytes);
    let weights = SourceWeights::from_verified_artifact(
        parsed,
        source_bytes,
        source_hash,
        &TransferAdmission {
            arbiter: arbiter.clone(),
            resource: key.clone(),
            owner: owner.clone(),
        },
    )?;
    drop(opening);
    let config = read_pinned(
        &source_folder.join("config.json"),
        config_bytes,
        config_hash,
    )?;
    let model = LoadedQwenModel::from_source(weights, &config)?;
    drop(config);
    assert_eq!(model.source_bytes(), source_bytes as usize);
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    assert_eq!(
        class(arbiter, key, AllocationClass::ImmutableWeights),
        source_bytes
    );
    let weak = Arc::downgrade(&model);
    let source = model.source_identity();
    let staging = arbiter.admit(
        key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: STAGING,
        }],
    )?;
    let folder = root.join(if av {
        ".agents/runs/numerics-and-context-av-replay-20261020/root-fixtures"
    } else {
        ".agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-fixtures"
    });
    let (left, right, left_layout, right_layout, expected_shape, expected_hash) = if av {
        let dense = words(&read_pinned(
            &folder.join("bulk64.v.f32le"),
            524_288,
            "b97493ac1b6f2b6f935c65d9627f2581c912e3e48c1afea1919c2e7c2e5c5b29",
        )?);
        let mut left = vec![0.0; 4_317_184];
        for head in 0..2 {
            for row in 0..256 {
                let target = head * 2_162_688 + row * 8_448;
                let from = (head * 256 + row) * 256;
                left[target..target + 256].copy_from_slice(&dense[from..from + 256]);
            }
        }
        let right = words(&read_pinned(
            &folder.join("bulk64.p.f32le"),
            524_288,
            "2457d3616026d4307e61d832ebd22a612716efee3714345bfa6a740cf380d074",
        )?);
        (
            left,
            right,
            BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4_317_184)?,
            BulkF32Layout::dense([256, 64, 8, 1])?,
            [256, 64, 8, 1],
            "91adb9f1d4a55e57a068106d195f928205f1031a998a5b9d4cb3fdc633e507b9",
        )
    } else {
        let raw = read_pinned(
            &folder.join("weight.f32le"),
            131_072,
            "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876",
        )?;
        assert_eq!(model.tensor_bytes("blk.0.ssm_alpha.weight")?, raw);
        let left = words(&raw);
        let right = words(&read_pinned(
            &folder.join("bulk64.x.f32le"),
            524_288,
            "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
        )?);
        (
            left,
            right,
            BulkF32Layout::dense([2048, 16, 1, 1])?,
            BulkF32Layout::dense([2048, 64, 1, 1])?,
            [16, 64, 1, 1],
            "448f5626f2f06de0e5fe0f375a92589bd87b81833fcbb9daca530c20f8424b3e",
        )
    };
    let admission = BulkF32HostAdmission {
        arbiter: arbiter.clone(),
        resource: key.clone(),
        owner: owner.clone(),
    };
    let make = |cancelled: Option<&AtomicBool>| {
        let right = ExternalBulkF32 {
            name: if av { "kq_soft_max-3" } else { "attn_norm-0" },
            layout: right_layout,
            backing: &right,
        };
        if av {
            BulkF32Slice::observed_pair(
                model.clone(),
                0,
                8448,
                ids,
                ExternalBulkF32 {
                    name: "cache_v_l3 (view) (permuted)",
                    layout: left_layout,
                    backing: &left,
                },
                right,
                admission.clone(),
                cancelled,
            )
        } else {
            BulkF32Slice::source_projection(
                model.clone(),
                0,
                8448,
                ids,
                QwenMatrixRole::GdnAlpha(0),
                right,
                admission.clone(),
                cancelled,
            )
        }
    };
    let cancelled = AtomicBool::new(true);
    let before = arbiter.snapshot(key)?.reserved;
    assert!(make(Some(&cancelled)).is_err());
    assert_eq!(arbiter.snapshot(key)?.reserved, before);
    let slice = make(None)?;
    assert_eq!(slice.matrix().source(), source);
    assert_eq!(slice.matrix().left().layout(), left_layout);
    assert_eq!(slice.matrix().right().span().ids(), ids);
    let recorded = slice.record()?;
    let plan = recorded.plan();
    assert_eq!(plan.nodes().len(), 3);
    assert!(matches!(
        plan.nodes()[0].operation,
        PlanOperation::TypedBulkF32Input { .. }
    ));
    assert!(matches!(
        plan.nodes()[1].operation,
        PlanOperation::TypedBulkF32Input { .. }
    ));
    assert!(matches!(
        plan.nodes()[2].operation,
        PlanOperation::TypedBulkF32Matmul { .. }
    ));
    assert!(
        plan.is_operator_slice()
            && plan.required_domains().is_empty()
            && plan.proposed_domains().is_empty()
    );
    assert_eq!(plan.frontier_delta(), 0);
    assert!(plan.logits().is_none());
    drop(recorded);
    let before = arbiter.snapshot(key)?.reserved;
    assert!(slice.execute(Some(&cancelled)).is_err());
    let provider = DeniedProvider(AtomicUsize::new(0));
    assert!(slice.execute_with_provider(Some(&provider), None).is_err());
    assert_eq!(provider.0.load(Ordering::Relaxed), 0);
    assert_eq!(arbiter.snapshot(key)?.reserved, before);
    let output_words = expected_shape.iter().product::<usize>();
    let output_bytes = (output_words * 4) as u64;
    let pressure = arbiter.admit(
        key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: LIMIT - before - output_bytes - 256 + 1,
        }],
    )?;
    let reserved = arbiter.snapshot(key)?.reserved;
    assert!(matches!(
        slice.execute(None),
        Err(zerogravity_core::Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(key)?.reserved, reserved);
    drop(pressure);
    let completed = slice.execute(None)?;
    assert_eq!(completed.source_identity(), source);
    assert_eq!(completed.contract(), BulkF32Contract::HaswellTinyBlasF32V1);
    assert_eq!(completed.shape(), expected_shape);
    assert_eq!(completed.values()?.len(), output_words);
    assert_eq!(class(arbiter, key, AllocationClass::Output), output_bytes);
    assert_eq!(class(arbiter, key, AllocationClass::Scratch), 256);
    // First expected-output open follows actual SAME HostMath execution.
    let expected = read_pinned(
        &folder.join("bulk64.expected.f32le"),
        output_words * 4,
        expected_hash,
    )?;
    let unequal = completed
        .values()?
        .iter()
        .zip(expected.chunks_exact(4))
        .filter(|(word, raw)| {
            word.to_bits() != u32::from_le_bytes((*raw).try_into().expect("four bytes"))
        })
        .count();
    assert_eq!(unequal, 0, "{alias} full captured DEFAULT outputs");
    drop(expected);
    let clone = completed.clone();
    drop(completed);
    assert_eq!(clone.view(0..output_words)?.len(), output_words);
    assert!(clone.view(output_words..output_words + 1).is_err());
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    drop(slice);
    drop(model);
    drop(left);
    drop(right);
    drop(staging);
    assert!(
        weak.upgrade().is_some(),
        "output clone retains original source"
    );
    assert_eq!(class(arbiter, key, AllocationClass::TransferStaging), 0);
    assert_eq!(
        class(arbiter, key, AllocationClass::ImmutableWeights),
        source_bytes
    );
    assert_eq!(
        class(arbiter, key, AllocationClass::Other),
        CONTROL + SOURCE_METADATA + BULK_METADATA
    );
    assert_eq!(
        arbiter.snapshot(key)?.reserved,
        CONTROL + SOURCE_METADATA + BULK_METADATA + source_bytes + output_bytes + 256
    );
    assert_eq!(output_hash(clone.values()?), expected_hash);
    drop(clone);
    assert!(
        weak.upgrade().is_none(),
        "final output releases original source"
    );
    assert_eq!(arbiter.snapshot(key)?.reserved, CONTROL + SOURCE_METADATA);
    drop(metadata);
    assert_eq!(arbiter.snapshot(key)?.reserved, CONTROL);
    Ok(output_words)
}

#[test]
#[ignore = "requires original approved models/captures via ZG_REFERENCE_ROOT; cpu,source-binding"]
fn captured_bulk_slices_match_complete_default_outputs_and_retain_source() -> Result<()> {
    let root = PathBuf::from(std::env::var_os("ZG_REFERENCE_ROOT").expect("set ZG_REFERENCE_ROOT"));
    assert!(root.is_absolute() && root.is_dir());
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "same-engine-captured-default")?;
    arbiter.register(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("original-GGUF-bulk-slice", "source-test")?;
    let control = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: CONTROL,
        }],
    )?;
    let av_ids = read_pinned(&root.join(".agents/runs/numerics-and-context-bulk-causality-20261020/root-first64-ar1-budget/input_ids.u32le"), 32_768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d")?;
    let alpha_ids = read_pinned(&root.join(".agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport/input_ids.u32le"), 32_768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d")?;
    assert_eq!(av_ids, alpha_ids);
    assert_eq!(
        format!("{:x}", Sha256::digest(&av_ids[..256])),
        "ff7862336b7b2b5520723bf753d0f9e484ef9e66f285efd6189c1d945ec17625"
    );
    let ids = av_ids[..256]
        .chunks_exact(4)
        .map(|raw| u32::from_le_bytes(raw.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    drop(av_ids);
    drop(alpha_ids);
    assert_eq!(
        replay(&root, &ids, &arbiter, &key, &owner, true)?
            + replay(&root, &ids, &arbiter, &key, &owner, false)?,
        132_096
    );
    drop(ids);
    drop(control);
    let final_state = arbiter.snapshot(&key)?;
    assert_eq!(final_state.reserved, 0);
    assert_eq!(final_state.reservations, 0);
    assert!(!final_state.fenced);
    Ok(())
}
