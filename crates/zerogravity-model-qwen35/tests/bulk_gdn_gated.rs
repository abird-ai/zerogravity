//! Genuine connected bulk projections/coefficients/recurrence → RMS/SiLU gate.
//! Observed Q/K/V, prior state and Z are explicit; output BLAS remains separate.
#![cfg(all(feature = "cpu", target_os = "linux", target_arch = "x86_64"))]

use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_core::accelerator::program::{BulkF32Layout, TeacherSpan};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::LoadedQwenModel;
use zerogravity_model_qwen35::operator_slice::bulk_f32::{
    BulkF32HostAdmission, BulkGdnCoefficientSlice, BulkGdnGatedSlice, BulkGdnRecurrentSlice,
    ExternalBulkF32, ObservedBulkGdnGate, ObservedBulkGdnInputs,
};

const CAPTURE: &str =
    ".agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport";
fn pin(bytes: &[u8], hash: &str) {
    assert_eq!(format!("{:x}", Sha256::digest(bytes)), hash);
}
fn pinned(path: &Path, length: usize, hash: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len(), length, "{}", path.display());
    pin(&bytes, hash);
    bytes
}
fn words(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect()
}
fn reserved(admission: &BulkF32HostAdmission) -> u64 {
    admission
        .arbiter
        .snapshot(&admission.resource)
        .unwrap()
        .reserved
}
fn same(stage: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{stage} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "first differing {stage}: word={index} teacher={} head={} value={} actual=0x{:08x} expected=0x{:08x}",
            index / 2048,
            (index / 128) % 16,
            index % 128,
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

#[test]
#[ignore = "requires pinned retained original 2B captures; ROOT runs serialized"]
fn produced_recurrence_feeds_source_grouped_rms_and_silu_gate() {
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
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for (field, stage, occurrence, shape) in [
        (34, "norm-0", 1, [128, 16, 64, 1]),
        (
            35,
            "learned_op_output::blk.0.ssm_norm.weight",
            0,
            [128, 16, 64, 1],
        ),
        (36, "z-0", 0, [2048, 64, 1, 1]),
        (39, "final_output-0", 0, [2048, 64, 1, 1]),
    ] {
        let rows = records
            .iter()
            .filter(|r| r["file"] == format!("field-{field}.f32le"))
            .collect::<Vec<_>>();
        assert_eq!(rows.len(), 1);
        let row = rows[0];
        assert_eq!(row["layer"], 0);
        assert_eq!(row["stage"], stage);
        assert_eq!(row["occurrence"], occurrence);
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
        &raw_ids[..256],
        "ff7862336b7b2b5520723bf753d0f9e484ef9e66f285efd6189c1d945ec17625",
    );
    let ids = raw_ids[..256]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let field = |index: usize, count: usize, hash: &str| {
        words(&pinned(
            &capture.join(format!("bulk64/field-{index}.f32le")),
            count * 4,
            hash,
        ))
    };
    let normalized = field(
        15,
        131072,
        "731e6574de4578fa584c7d5f859141265fff531687b42b0c156ab6b95dc847d6",
    );
    let query = field(
        26,
        131072,
        "3bb137807d2ec8e03b2afc2ea7b62648cd3baef2b526d0b3197da415b1582f20",
    );
    let key = field(
        27,
        131072,
        "a698d1d7838fb073f3a9ce694b16a2e93dbc2a2c944d4898ce9b3c0740e288c0",
    );
    let value = field(
        28,
        131072,
        "a7410d7bec7d53c3edb571f829f1e0e7d65cf154c27420c1167b7fafc83a8bfc",
    );
    let prior = field(
        31,
        262144,
        "30e14955ebf1352266dc2ff8067e68104607e750abb9d3b36582b8af909fcb58",
    );
    let mut z = field(
        36,
        131072,
        "6b20f53dce195fb2b45442baa9ff3785d8a11112ed55cbe31d471a3600ddfa18",
    );
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
        (
            "blk.0.ssm_norm.weight",
            "4f50f78402c71ebebcadcb6259c2883b4f0a441c17cacee8f626e5c9ad7118b7",
        ),
    ];
    for (name, hash) in source_pins {
        pin(model.tensor_bytes(name).unwrap(), hash);
    }
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "bulk-gdn-gated-capture").unwrap();
    arbiter.register(resource.clone(), 8 * 1024 * 1024).unwrap();
    let admission = BulkF32HostAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original-2b", "private-bulk-gated").unwrap(),
    };
    let coefficient_slice = BulkGdnCoefficientSlice::new(
        model.clone(),
        0,
        8448,
        &ids,
        0,
        ExternalBulkF32 {
            name: "attn_norm-0",
            layout: BulkF32Layout::dense([2048, 64, 1, 1]).unwrap(),
            backing: &normalized,
        },
        admission.clone(),
        None,
    )
    .unwrap();
    let coefficients = coefficient_slice.execute(None).unwrap();
    let span = coefficients.span().clone();
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
    )
    .unwrap();
    let recurrent = recurrence_slice.execute(None).unwrap();
    let baseline = reserved(&admission);
    let cancelled = AtomicBool::new(true);
    assert!(
        BulkGdnGatedSlice::new(
            recurrent.clone(),
            ObservedBulkGdnGate {
                span: span.clone(),
                z: &z
            },
            Some(&cancelled)
        )
        .is_err()
    );
    let foreign = TeacherSpan::new(source, 0, 8448, model.geometry().vocab_size, &ids).unwrap();
    assert!(
        BulkGdnGatedSlice::new(
            recurrent.clone(),
            ObservedBulkGdnGate {
                span: foreign,
                z: &z
            },
            None
        )
        .is_err()
    );
    let first = z[0];
    z[0] = f32::NAN;
    assert!(
        BulkGdnGatedSlice::new(
            recurrent.clone(),
            ObservedBulkGdnGate {
                span: span.clone(),
                z: &z
            },
            None
        )
        .is_err()
    );
    z[0] = first;
    assert_eq!(reserved(&admission), baseline);
    let slice =
        BulkGdnGatedSlice::new(recurrent, ObservedBulkGdnGate { span, z: &z }, None).unwrap();
    let ready_baseline = reserved(&admission);
    assert!(slice.execute(Some(&cancelled)).is_err());
    assert_eq!(reserved(&admission), ready_baseline);
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
    assert_eq!(reserved(&admission), ready_baseline);
    let output = slice.execute(None).unwrap();
    // Independent native outputs are read only after connected execution.
    same(
        "unweighted RMS",
        output.normalized(),
        &field(
            34,
            131072,
            "7f591d1fa8b605cfb00f7e81bef2472463c66f811610f0290feda2d350b8e126",
        ),
    );
    same(
        "raw gamma weighted RMS",
        output.weighted(),
        &field(
            35,
            131072,
            "5e20b5ad7cee2e572a2642e0e245aaa5bd086af3aaf6a367206d79b06019171f",
        ),
    );
    let expected = field(
        39,
        131072,
        "4400491b963ae5fa113425e97ed6dafc3e8ca74b7af406fe77578bf5a6bee486",
    );
    same("SiLU gated output", output.gated(), &expected);
    for (name, hash) in source_pins {
        pin(model.tensor_bytes(name).unwrap(), hash);
    }
    let retained = output.clone();
    let retained_bytes = reserved(&admission);
    drop(output);
    drop(slice);
    drop(recurrence_slice);
    drop(coefficient_slice);
    drop(model);
    assert!(weak_source.upgrade().is_some());
    assert_eq!(reserved(&admission), retained_bytes);
    assert_eq!(
        retained.gated()[131071].to_bits(),
        expected[131071].to_bits()
    );
    assert_eq!(retained.recurrent().state().len(), 262144);
    drop(retained);
    assert!(weak_source.upgrade().is_none());
    assert_eq!(reserved(&admission), 0);
    println!(
        "BULK_GDN_GATED_CAPTURE_PASS stage_words=393216 QKV_prior_Z_external=true output_projection_and_full_model_unqualified=true"
    );
}
