//! Genuine retained 2B bulk producers and independently captured consumers.
//! No runtime oracle launch, reconstructed expected math or full-model claim.
#![cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_core::accelerator::program::{BulkF32Contract, BulkF32Layout, PlanOperation};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::LoadedQwenModel;
use zerogravity_model_qwen35::operator_slice::bulk_f32::{
    BulkF32HostAdmission, BulkGdnCoefficientContract, BulkGdnCoefficientSlice, ExternalBulkF32,
};

const CAPTURE: &str =
    ".agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport";
const INPUT_SHA: &str = "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6";

fn pin(bytes: &[u8], expected: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), expected);
}
fn pinned(path: &Path, bytes: usize, expected: &str) -> Vec<u8> {
    let actual = std::fs::read(path).unwrap();
    assert_eq!(actual.len(), bytes, "{}", path.display());
    pin(&actual, expected);
    actual
}
fn words(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}
fn same(stage: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{stage} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "first differing {stage}: word={index} teacher={} head={} actual=0x{:08x} expected=0x{:08x}",
            index / 16,
            index % 16,
            actual.to_bits(),
            expected.to_bits(),
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
fn normalized(values: &[f32], teachers: usize) -> ExternalBulkF32<'_> {
    ExternalBulkF32 {
        name: "attn_norm-0",
        layout: BulkF32Layout::dense([2048, teachers, 1, 1]).unwrap(),
        backing: values,
    }
}

#[test]
#[ignore = "requires pinned retained original 2B capture; ROOT runs serialized"]
fn authentic_bulk_alpha_beta_feed_the_existing_coefficient_consumer() {
    let reference = PathBuf::from(
        std::env::var_os("ZG_REFERENCE_ROOT")
            .expect("set explicit existing approved reference workspace"),
    );
    let capture = reference.join(CAPTURE);
    let model_dir = reference.join("target/models/qwen35-2b");
    pinned(
        &model_dir.join("config.json"),
        2908,
        "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
    );
    let records = pinned(
        &capture.join("bulk64/records.jsonl"),
        55822,
        "eacee1e7cf45fdac3ea40e48152432347f1f8f3915c633ecc2ce4c8f59e54437",
    );
    let records: Vec<serde_json::Value> = std::str::from_utf8(&records)
        .unwrap()
        .lines()
        .map(|row| serde_json::from_str(row).unwrap())
        .collect();
    for (field, stage, shape) in [
        (
            15,
            "projection_input::blk.0.ssm_alpha.weight",
            [2048, 64, 1, 1],
        ),
        (
            22,
            "projection_input::blk.0.ssm_beta.weight",
            [2048, 64, 1, 1],
        ),
        (
            16,
            "projection_output::blk.0.ssm_alpha.weight",
            [16, 64, 1, 1],
        ),
        (
            23,
            "projection_output::blk.0.ssm_beta.weight",
            [16, 64, 1, 1],
        ),
        (19, "a_softplus-0", [16, 64, 1, 1]),
        (20, "learned_op_output::blk.0.ssm_a", [16, 64, 1, 1]),
        (25, "beta_sigmoid-0", [1, 16, 64, 1]),
    ] {
        let selected = records
            .iter()
            .filter(|r| r["file"] == format!("field-{field}.f32le"))
            .collect::<Vec<_>>();
        assert_eq!(selected.len(), 1);
        let row = selected[0];
        assert_eq!(row["stage"], stage);
        assert_eq!(row["layer"], 0);
        assert_eq!(row["call_tokens"], 64);
        assert_eq!(row["window_start"], 0);
        assert_eq!(row["dense_ne"], serde_json::json!(shape));
    }
    let raw_ids = pinned(
        &capture.join("input_ids.u32le"),
        32768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
    );
    pin(
        &raw_ids[..64 * 4],
        "ff7862336b7b2b5520723bf753d0f9e484ef9e66f285efd6189c1d945ec17625",
    );
    let ids = raw_ids[..64 * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let input_bytes = pinned(&capture.join("bulk64/field-15.f32le"), 524288, INPUT_SHA);
    assert_eq!(
        input_bytes,
        pinned(&capture.join("bulk64/field-22.f32le"), 524288, INPUT_SHA)
    );
    let input = words(&input_bytes);
    let model = LoadedQwenModel::open(
        &model_dir.join("model.gguf"),
        &model_dir.join("config.json"),
    )
    .unwrap();
    let source = model.source_identity();
    let weak_source = Arc::downgrade(&model);
    let source_pins = [
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
    ];
    for (name, hash) in source_pins {
        pin(model.tensor_bytes(name).unwrap(), hash);
    }

    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "bulk-gdn-coefficient-capture").unwrap();
    arbiter.register(resource.clone(), 8 * 1024 * 1024).unwrap();
    let admission = BulkF32HostAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original-2b", "private-bulk-coefficients").unwrap(),
    };
    let cancelled = AtomicBool::new(true);
    assert!(
        BulkGdnCoefficientSlice::new(
            model.clone(),
            0,
            8448,
            &ids,
            0,
            normalized(&input, 64),
            admission.clone(),
            Some(&cancelled)
        )
        .is_err()
    );
    assert!(
        BulkGdnCoefficientSlice::new(
            model.clone(),
            0,
            8448,
            &ids,
            1,
            normalized(&input, 64),
            admission.clone(),
            None
        )
        .is_err()
    );
    assert!(
        BulkGdnCoefficientSlice::new(
            model.clone(),
            8192,
            8192,
            &ids,
            0,
            normalized(&input, 64),
            admission.clone(),
            None
        )
        .is_err()
    );
    assert_eq!(reserved(&admission), 0);
    let slice = BulkGdnCoefficientSlice::new(
        model.clone(),
        0,
        8448,
        &ids,
        0,
        normalized(&input, 64),
        admission.clone(),
        None,
    )
    .unwrap();
    for recorded in slice.record_projections().unwrap() {
        assert!(recorded.plan().is_operator_slice());
        assert_eq!(recorded.plan().source(), source);
        assert_eq!(recorded.plan().nodes().len(), 3);
        assert!(matches!(recorded.plan().nodes()[2].operation,
            PlanOperation::TypedBulkF32Matmul { ref matrix }
                if matrix.contract() == BulkF32Contract::HaswellTinyBlasF32V1));
    }
    let baseline = reserved(&admission);
    assert!(slice.execute(Some(&cancelled)).is_err());
    assert_eq!(reserved(&admission), baseline);
    let pressure = admission
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
        .unwrap();
    let pressed = reserved(&admission);
    assert!(slice.execute(None).is_err());
    assert_eq!(reserved(&admission), pressed);
    drop(pressure);
    assert_eq!(reserved(&admission), baseline);

    // Expected files are opened only after actual producer/consumer arithmetic.
    let output = slice.execute(None).unwrap();
    assert_eq!(
        output.contract(),
        BulkGdnCoefficientContract::HaswellTinyBlasF32GgufCoefficientsV1
    );
    assert_eq!(output.source_identity(), source);
    assert_eq!(output.span().ids(), ids);
    assert_eq!(output.span().base(), 0);
    for (name, hash) in source_pins {
        pin(model.tensor_bytes(name).unwrap(), hash);
    }
    let expected = |field: usize, hash: &str| {
        words(&pinned(
            &capture.join(format!("bulk64/field-{field}.f32le")),
            4096,
            hash,
        ))
    };
    same(
        "alpha",
        output.projected_alpha().unwrap(),
        &expected(
            16,
            "448f5626f2f06de0e5fe0f375a92589bd87b81833fcbb9daca530c20f8424b3e",
        ),
    );
    same(
        "beta projection",
        output.projected_beta().unwrap(),
        &expected(
            23,
            "a68b4b2a99d136e2ac022601d989b8b46849082424ed6f797087ea2460a58088",
        ),
    );
    let softplus = expected(
        19,
        "a887158e81499983145ed4c76d917c8b60f7897f4fa7352df7369b5ed3e22a92",
    );
    let gate = expected(
        20,
        "da35ebc5f9a4438f6dd2104b9491e2e355d5e5be3fdcc19ddf38c8dda347b1f0",
    );
    let beta = expected(
        25,
        "2176c9b3170b7074f765c422a25f46f39ff4ad5847499d2b13af9f33edea081d",
    );
    for teacher in 0..64 {
        let actual = output.teacher(teacher).unwrap();
        let row = teacher * 16..(teacher + 1) * 16;
        same(
            &format!("softplus teacher {teacher}"),
            &actual.softplus,
            &softplus[row.clone()],
        );
        same(
            &format!("gate_raw teacher {teacher}"),
            &actual.gate_raw,
            &gate[row.clone()],
        );
        for (head, coefficient) in actual.coefficients.iter().enumerate() {
            assert_eq!(
                coefficient.beta.to_bits(),
                beta[row.start + head].to_bits(),
                "first differing sigmoid beta: teacher={teacher} head={head} actual=0x{:08x} expected=0x{:08x}",
                coefficient.beta.to_bits(),
                beta[row.start + head].to_bits()
            );
        }
    }
    assert!(output.teacher(64).is_err());
    let retained = output.clone();
    let retained_bytes = reserved(&admission);
    drop(output);
    drop(slice);
    drop(model);
    assert!(weak_source.upgrade().is_some());
    assert_eq!(reserved(&admission), retained_bytes);
    assert_eq!(retained.projected_alpha().unwrap().len(), 1024);
    assert_eq!(retained.teacher(63).unwrap().coefficients.len(), 16);
    drop(retained);
    assert!(weak_source.upgrade().is_none());
    assert_eq!(reserved(&admission), 0);
    println!(
        "BULK_GDN_COEFFICIENTS_CAPTURE_PASS projected_words=2048 consumer_words=3072 native_gain_unobserved=true recurrence_model_and_state_unqualified=true"
    );
}
