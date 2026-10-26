//! Verified original-source embedding/RMS producer → existing connected GDN cut.
//! Q/K/V, prior state and projected Z remain observed; no model/root permit.
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
use zerogravity_core::accelerator::program::{PlanOperation, ProgramNumericalProfile, TeacherSpan};
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::LoadedQwenModel;
use zerogravity_model_qwen35::operator_slice::bulk_f32::source_bindings::PrefixContract;
use zerogravity_model_qwen35::operator_slice::bulk_f32::{
    BulkF32HostAdmission, BulkGdnCoefficientSlice, BulkGdnGatedSlice, BulkGdnRecurrentSlice,
    BulkGdnSsmOutSlice, GgufPrefixSlice, ObservedBulkGdnGate, ObservedBulkGdnInputs,
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const SOURCE_BYTES: u64 = 1_296_764_000;
const SOURCE_HASH: &str = "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6";
const SOURCE_METADATA: u64 = 16 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const CAPTURE: &str =
    ".agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport";

fn pin(bytes: &[u8], hash: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), hash);
}
fn read_pinned(path: &Path, length: usize, hash: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let before = file.metadata()?;
    assert!(before.is_file());
    assert_eq!(before.len(), length as u64, "{}", path.display());
    let mut bytes = Vec::with_capacity(length + 1);
    (&mut file)
        .take(length as u64 + 1)
        .read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    assert_eq!(
        bytes.len(),
        length,
        "complete input and EOF: {}",
        path.display()
    );
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
        ),
        "source changed during retained-FD read: {}",
        path.display(),
    );
    pin(&bytes, hash);
    Ok(bytes)
}
fn words(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn same(stage: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{stage} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "first differing {stage}: word={index} actual=0x{:08x} expected=0x{:08x}",
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
fn class(admission: &BulkF32HostAdmission, kind: AllocationClass) -> u64 {
    admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .per_class
        .get(&kind)
        .copied()
        .unwrap_or(0)
}
fn pin_selected_embedding_rows(model: &LoadedQwenModel, ids: &[u32]) -> Result<()> {
    let mut hash = Sha256::new();
    for &id in ids {
        let original = model.embedding_source(id)?;
        assert_eq!(original.source_identity, model.source_identity());
        assert_eq!(original.tensor_name, "token_embd.weight");
        assert_eq!(original.kind, QuantKind::Q6K);
        assert_eq!(original.width, 2048);
        assert_eq!(original.vocabulary, 248320);
        assert_eq!(original.packed_weights.len(), 417_177_600);
        let start = id as usize * 1680;
        hash.update(&original.packed_weights[start..start + 1680]);
    }
    assert_eq!(ids.len() * 1680, 107520);
    assert_eq!(
        format!("{:x}", hash.finalize()),
        "8f7c1354789167d66359f6b97f1fa8e32f320308250b61c8c8688070560bcd2c"
    );
    Ok(())
}
struct DeniedProvider(AtomicUsize);
impl MatVecKernel for DeniedProvider {
    fn try_execute(&self, _region: MatVecRegion<'_>) -> zerogravity_core::Result<RegionExecution> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(RegionExecution::Unsupported)
    }
}

#[test]
#[ignore = "requires pinned retained original 2B source/captures; ROOT runs serialized"]
fn verified_original_embedding_rms_feeds_existing_gdn_through_ssm_out() -> Result<()> {
    let reference = PathBuf::from(
        std::env::var_os("ZG_REFERENCE_ROOT").expect("set explicit approved reference workspace"),
    );
    assert!(reference.is_absolute() && reference.is_dir());
    let capture = reference.join(CAPTURE);
    let model_dir = reference.join("target/models/qwen35-2b");
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "original-gguf-prefix-capture")?;
    arbiter.register(resource.clone(), 4 * 1024 * 1024 * 1024)?;
    let admission = BulkF32HostAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original-2b", "private-gguf-prefix")?,
    };
    let source_metadata = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: SOURCE_METADATA,
        }],
    )?;
    let staging = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: STAGING,
        }],
    )?;
    // Same verified owner and source ticket as existing real-source tests.
    // The opening guard is conservative, not a whole-host capacity claim.
    let opening = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::ImmutableWeights,
            bytes: SOURCE_BYTES,
        }],
    )?;
    let parsed = Artifact::open(&model_dir.join("model.gguf"))?;
    assert_eq!(parsed.info.file_len, SOURCE_BYTES);
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
    let config = read_pinned(
        &model_dir.join("config.json"),
        2908,
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    )?;
    let parsed_config: serde_json::Value = serde_json::from_slice(&config)?;
    assert_eq!(
        (parsed_config["text_config"]["rms_norm_eps"]
            .as_f64()
            .unwrap() as f32)
            .to_bits(),
        0x358637bd
    );
    let model = LoadedQwenModel::from_source(weights, &config)?;
    drop(config);
    drop(parsed_config);
    assert_eq!(model.source_bytes(), SOURCE_BYTES as usize);
    assert_eq!(model.initialized_cpu_state_bytes(), 0);
    assert_eq!(
        class(&admission, AllocationClass::ImmutableWeights),
        SOURCE_BYTES
    );
    let weak_source = Arc::downgrade(&model);
    let source = model.source_identity();
    let raw_records = read_pinned(
        &capture.join("bulk64/records.jsonl"),
        55822,
        "eacee1e7cf45fdac3ea40e48152432347f1f8f3915c633ecc2ce4c8f59e54437",
    )?;
    let records: Vec<serde_json::Value> = std::str::from_utf8(&raw_records)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    drop(raw_records);
    for (field, layer, stage, operation, tensor, sources) in [
        (
            1,
            -1,
            "model.input_embed",
            "GET_ROWS",
            1,
            [2, 3, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            2,
            0,
            "norm-0",
            "RMS_NORM",
            4,
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            4,
            0,
            "attn_norm-0",
            "MUL",
            5,
            [4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
    ] {
        let rows = records
            .iter()
            .filter(|r| r["file"] == format!("field-{field}.f32le"))
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        let row = rows[0];
        assert_eq!(row["layer"], layer);
        assert_eq!(row["stage"], stage);
        assert_eq!(row["native_op"], operation);
        assert_eq!(row["native_tensor_id"], tensor);
        assert_eq!(row["native_source_ids"], serde_json::json!(sources));
        assert_eq!(row["occurrence"], 0);
        assert_eq!(row["mode"], "bulk64");
        assert_eq!(row["scope"], "actual_tensor");
        assert_eq!(row["call"], 0);
        assert_eq!(row["call_tokens"], 64);
        assert_eq!(row["window_start"], 0);
        assert_eq!(row["window_tokens"], 64);
        assert_eq!(row["elements"], 131072);
        assert_eq!(row["token_axis"], 1);
        assert_eq!(row["source_slice_start_elements"], 0);
        assert_eq!(row["dense_ne"], serde_json::json!([2048, 64, 1, 1]));
        assert_eq!(row["native_ne"], row["dense_ne"]);
        assert_eq!(
            row["native_nb"],
            serde_json::json!([4, 8192, 524288, 524288])
        );
        assert_eq!(row["masked_negative_infinity_allowed"], false);
    }
    drop(records);
    let raw_ids = read_pinned(
        &capture.join("input_ids.u32le"),
        32768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
    )?;
    pin(
        &raw_ids[..256],
        "ff7862336b7b2b5520723bf753d0f9e484ef9e66f285efd6189c1d945ec17625",
    );
    let ids = raw_ids[..256]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    drop(raw_ids);
    pin_selected_embedding_rows(&model, &ids)?;
    let source_pins = [
        (
            "blk.0.attn_norm.weight",
            "4f189e0e0cb573c744e0f3e4683304728ddd51e977bbc2ffbafdc96bb297cce2",
        ),
        (
            "blk.0.ssm_alpha.weight",
            "0203e536ec2d6fc0b475221577af74252cb17f2c57f73441a3b0b0efd5a2f876",
        ),
        (
            "blk.0.ssm_beta.weight",
            "f9f0d19e6c305e6627b29eeaba9b02abe9f32d6d8c9b019af1f045033748a3f0",
        ),
        (
            "blk.0.ssm_dt.bias",
            "97c8e8de150064484f59c7dbb4f23182f09eeddaa84d40ef48a82bd19c53c03d",
        ),
        (
            "blk.0.ssm_a",
            "75b8e5de5bd88fb15abbbfad58b18c00d262d2f8e72577ffd9680d7cfc6640b5",
        ),
        (
            "blk.0.ssm_norm.weight",
            "4f50f78402c71ebebcadcb6259c2883b4f0a441c17cacee8f626e5c9ad7118b7",
        ),
        (
            "blk.0.ssm_out.weight",
            "6d27d16495f7bc863fb414206fe3d8aa57c128aa1f464e7fc0e0f1339cec4c35",
        ),
    ];
    for (name, hash) in source_pins {
        pin(model.tensor_bytes(name)?, hash);
    }
    let raw_gamma = read_pinned(
        &capture.join("learned-operands/learned-1-0.bin"),
        8192,
        source_pins[0].1,
    )?;
    assert_eq!(model.tensor_bytes("blk.0.attn_norm.weight")?, raw_gamma);
    drop(raw_gamma);
    let gamma = model.tensor_info("blk.0.attn_norm.weight")?;
    assert_eq!(gamma.kind, QuantKind::F32);
    assert_eq!(gamma.shape, [2048]);
    let baseline = reserved(&admission);
    let cancelled = AtomicBool::new(true);
    assert!(
        GgufPrefixSlice::new(
            model.clone(),
            0,
            8448,
            &ids,
            admission.clone(),
            Some(&cancelled)
        )
        .is_err()
    );
    assert!(GgufPrefixSlice::new(model.clone(), 1, 8448, &ids, admission.clone(), None).is_err());
    assert!(
        GgufPrefixSlice::new(model.clone(), 0, 8448, &ids[..63], admission.clone(), None).is_err()
    );
    let mut bad_ids = ids.clone();
    bad_ids[0] = 248320;
    assert!(
        GgufPrefixSlice::new(model.clone(), 0, 8448, &bad_ids, admission.clone(), None).is_err()
    );
    drop(bad_ids);
    assert_eq!(reserved(&admission), baseline);
    let fill = || {
        admission
            .arbiter
            .admit(
                &admission.resource,
                admission.owner.clone(),
                &[AllocationRequest {
                    kind: AllocationClass::Other,
                    bytes: admission
                        .arbiter
                        .snapshot(&admission.resource)
                        .unwrap()
                        .remaining,
                }],
            )
            .unwrap()
    };
    let pressure = fill();
    let pressed = reserved(&admission);
    assert!(GgufPrefixSlice::new(model.clone(), 0, 8448, &ids, admission.clone(), None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    let prefix_slice = GgufPrefixSlice::new(model.clone(), 0, 8448, &ids, admission.clone(), None)?;
    assert_eq!(
        prefix_slice.descriptor().contract(),
        PrefixContract::First64GgmlF32V1
    );
    assert_eq!(prefix_slice.descriptor().epsilon().to_bits(), 0x358637bd);
    assert_eq!(prefix_slice.descriptor().embedding().source, source);
    assert_eq!(prefix_slice.descriptor().norm().source, source);
    let plan = prefix_slice.record()?;
    assert_eq!(plan.plan().source(), source);
    assert_eq!(
        plan.plan().numerical_profile(),
        ProgramNumericalProfile::GgufSourceV1
    );
    assert_eq!(plan.plan().nodes().len(), 2);
    assert!(matches!(
        &plan.plan().nodes()[0].operation,
        PlanOperation::SourceOperator { descriptor, .. } if matches!(descriptor.kind(), zerogravity_core::accelerator::program::SourceOperatorKind::EmbeddingRows)
    ));
    assert!(matches!(
        &plan.plan().nodes()[1].operation,
        PlanOperation::SourceOperator { descriptor, .. } if matches!(descriptor.kind(), zerogravity_core::accelerator::program::SourceOperatorKind::Rms(_))
    ));
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    drop(plan);
    let ready = reserved(&admission);
    assert!(prefix_slice.execute(Some(&cancelled)).is_err());
    let provider = DeniedProvider(AtomicUsize::new(0));
    assert!(
        prefix_slice
            .execute_with_provider(Some(&provider), None)
            .is_err()
    );
    assert_eq!(provider.0.load(Ordering::Relaxed), 0);
    assert_eq!(reserved(&admission), ready);
    let pressure = fill();
    let pressed = reserved(&admission);
    assert!(prefix_slice.execute(None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    // Embedding completes privately; later RMS admission must unwind its output.
    let remaining = admission.arbiter.snapshot(&admission.resource)?.remaining;
    let pressure = admission.arbiter.admit(
        &admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: remaining - (524288 + 9216),
        }],
    )?;
    let pressed = reserved(&admission);
    assert!(prefix_slice.execute(None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    let prefix = prefix_slice.execute(None)?;
    assert_eq!(prefix.source(), source);
    assert_eq!(prefix.span().source(), source);
    assert_eq!(prefix.span().ids(), ids);
    assert_eq!(prefix.span().base(), 0);
    assert_eq!(prefix.span().capacity(), 8448);
    let span = prefix.span().clone();
    let foreign = TeacherSpan::new(source, 0, 8448, 248320, &ids)?;
    assert_ne!(&foreign, &span);
    // Native expected values enter only after their producing arithmetic.
    let field = |index: usize, count: usize, hash: &str| -> Result<Vec<f32>> {
        Ok(words(&read_pinned(
            &capture.join(format!("bulk64/field-{index}.f32le")),
            count * 4,
            hash,
        )?))
    };
    same(
        "Q6_K embedding",
        prefix.embedding()?,
        &field(
            1,
            131072,
            "5f5b48c937e071478a4736abde71d9505b2308601f20c67660c85b754b5e86ef",
        )?,
    );
    same(
        "unweighted RMS occurrence0",
        prefix.unweighted()?,
        &field(
            2,
            131072,
            "1a4d27a87ff3580fdb5011510fd228c61bbf0fa22101408d484716fd1e4c53d5",
        )?,
    );
    // Original GGUF gamma already includes the exporter adjustment; no +1 here.
    same(
        "raw GGUF gamma weighted RMS",
        prefix.normalized()?,
        &field(
            4,
            131072,
            "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
        )?,
    );
    let ready = reserved(&admission);
    assert!(BulkGdnCoefficientSlice::from_prefix(prefix.clone(), Some(&cancelled)).is_err());
    assert_eq!(reserved(&admission), ready);
    let pressure = fill();
    let pressed = reserved(&admission);
    assert!(BulkGdnCoefficientSlice::from_prefix(prefix.clone(), None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    let transfer_bytes = class(&admission, AllocationClass::TransferStaging);
    let coefficient_slice = BulkGdnCoefficientSlice::from_prefix(prefix.clone(), None)?;
    assert_eq!(
        class(&admission, AllocationClass::TransferStaging),
        transfer_bytes,
        "produced normalized RHS is retained without a transport copy"
    );
    for recorded in coefficient_slice.record_projections()? {
        assert_eq!(recorded.plan().source(), source);
        assert!(recorded.plan().required_domains().is_empty());
        assert!(recorded.plan().proposed_domains().is_empty());
        assert_eq!(recorded.plan().frontier_delta(), 0);
    }
    let ready = reserved(&admission);
    assert!(coefficient_slice.execute(Some(&cancelled)).is_err());
    assert_eq!(reserved(&admission), ready);
    let pressure = fill();
    let pressed = reserved(&admission);
    assert!(coefficient_slice.execute(None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    let coefficients = coefficient_slice.execute(None)?;
    assert_eq!(
        class(&admission, AllocationClass::TransferStaging),
        transfer_bytes
    );
    assert_eq!(coefficients.source_identity(), source);
    assert_eq!(coefficients.span(), &span);
    same(
        "alpha projection",
        coefficients.projected_alpha()?,
        &field(
            16,
            1024,
            "448f5626f2f06de0e5fe0f375a92589bd87b81833fcbb9daca530c20f8424b3e",
        )?,
    );
    same(
        "beta projection",
        coefficients.projected_beta()?,
        &field(
            23,
            1024,
            "a68b4b2a99d136e2ac022601d989b8b46849082424ed6f797087ea2460a58088",
        )?,
    );
    let softplus = field(
        19,
        1024,
        "a887158e81499983145ed4c76d917c8b60f7897f4fa7352df7369b5ed3e22a92",
    )?;
    let gate = field(
        20,
        1024,
        "da35ebc5f9a4438f6dd2104b9491e2e355d5e5be3fdcc19ddf38c8dda347b1f0",
    )?;
    let beta = field(
        25,
        1024,
        "2176c9b3170b7074f765c422a25f46f39ff4ad5847499d2b13af9f33edea081d",
    )?;
    for teacher in 0..64 {
        let actual = coefficients.teacher(teacher)?;
        let row = teacher * 16..(teacher + 1) * 16;
        same(
            &format!("softplus teacher{teacher}"),
            &actual.softplus,
            &softplus[row.clone()],
        );
        same(
            &format!("raw gate teacher{teacher}"),
            &actual.gate_raw,
            &gate[row.clone()],
        );
        for (head, expected) in beta[row].iter().enumerate() {
            assert_eq!(
                actual.coefficients[head].beta.to_bits(),
                expected.to_bits(),
                "first differing sigmoid: teacher={teacher} head={head} actual=0x{:08x} expected=0x{:08x}",
                actual.coefficients[head].beta.to_bits(),
                expected.to_bits()
            );
        }
    }
    drop(softplus);
    drop(gate);
    drop(beta);
    let query = field(
        26,
        131072,
        "3bb137807d2ec8e03b2afc2ea7b62648cd3baef2b526d0b3197da415b1582f20",
    )?;
    let key = field(
        27,
        131072,
        "a698d1d7838fb073f3a9ce694b16a2e93dbc2a2c944d4898ce9b3c0740e288c0",
    )?;
    let value = field(
        28,
        131072,
        "a7410d7bec7d53c3edb571f829f1e0e7d65cf154c27420c1167b7fafc83a8bfc",
    )?;
    let prior = field(
        31,
        262144,
        "30e14955ebf1352266dc2ff8067e68104607e750abb9d3b36582b8af909fcb58",
    )?;
    let z = field(
        36,
        131072,
        "6b20f53dce195fb2b45442baa9ff3785d8a11112ed55cbe31d471a3600ddfa18",
    )?;
    let recurrence_slice = BulkGdnRecurrentSlice::new(
        coefficients,
        ObservedBulkGdnInputs {
            span: span.clone(),
            query: &query,
            key: &key,
            value: &value,
            previous_state: &prior,
        },
        None,
    )?;
    let recurrent = recurrence_slice.execute(None)?;
    same(
        "GDN core",
        recurrent.core(),
        &field(
            32,
            131072,
            "1e10aa12ae051d7b6dc853f49e4302b99d833a61223bc3ce951ebe9de9afcf6d",
        )?,
    );
    same(
        "GDN final state",
        recurrent.state(),
        &field(
            33,
            262144,
            "900193c0c231f4f4194628dce46f17b6d41982db06c9dd5a37e886c6b7cec2b1",
        )?,
    );
    let gate_slice = BulkGdnGatedSlice::new(
        recurrent,
        ObservedBulkGdnGate {
            span: span.clone(),
            z: &z,
        },
        None,
    )?;
    let gated = gate_slice.execute(None)?;
    same(
        "GDN unweighted RMS",
        gated.normalized(),
        &field(
            34,
            131072,
            "7f591d1fa8b605cfb00f7e81bef2472463c66f811610f0290feda2d350b8e126",
        )?,
    );
    same(
        "GDN raw gamma weighted RMS",
        gated.weighted(),
        &field(
            35,
            131072,
            "5e20b5ad7cee2e572a2642e0e245aaa5bd086af3aaf6a367206d79b06019171f",
        )?,
    );
    same(
        "GDN SiLU gated",
        gated.gated(),
        &field(
            39,
            131072,
            "4400491b963ae5fa113425e97ed6dafc3e8ca74b7af406fe77578bf5a6bee486",
        )?,
    );
    let slice = BulkGdnSsmOutSlice::new(gated, None)?;
    assert_eq!(slice.descriptor().span(), &span);
    assert_eq!(slice.descriptor().source(), source);
    let output = slice.execute(None)?;
    let expected = field(
        42,
        131072,
        "5477c01a3885382f18640bf835ec10b00e1d5b2611b5fb1c1889653b64027059",
    )?;
    same("original Q8_0 Zen ssm_out", output.values()?, &expected);
    assert_eq!(output.producer().recurrent().coefficients().span(), &span);
    assert_eq!(
        output
            .producer()
            .recurrent()
            .coefficients()
            .source_identity(),
        source
    );
    pin_selected_embedding_rows(&model, &ids)?;
    for (name, hash) in source_pins {
        pin(model.tensor_bytes(name)?, hash);
    }
    let retained = output.clone();
    let retained_bytes = reserved(&admission);
    drop(output);
    drop(slice);
    drop(gate_slice);
    drop(recurrence_slice);
    drop(coefficient_slice);
    drop(prefix);
    drop(prefix_slice);
    drop(model);
    assert!(weak_source.upgrade().is_some());
    assert_eq!(
        class(&admission, AllocationClass::ImmutableWeights),
        SOURCE_BYTES
    );
    assert_eq!(reserved(&admission), retained_bytes);
    same("retained ssm_out", retained.values()?, &expected);
    drop(retained);
    assert!(weak_source.upgrade().is_none());
    assert_eq!(class(&admission, AllocationClass::ImmutableWeights), 0);
    assert_eq!(reserved(&admission), SOURCE_METADATA + STAGING);
    drop(expected);
    drop(query);
    drop(key);
    drop(value);
    drop(prior);
    drop(z);
    drop(ids);
    drop(foreign);
    drop(span);
    drop(staging);
    drop(source_metadata);
    assert_eq!(reserved(&admission), 0);
    println!(
        "GGUF_PREFIX_CAPTURE_PASS prefix_words=393216 downstream_words=922624 original_source_ticket_retained=true QKV_prior_Z_external=true gain_not_individually_observed=true full_model_unqualified=true"
    );
    Ok(())
}
