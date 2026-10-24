//! Captured original2B Q8_0 BLAS leaf only; no GGUF issuer/model/state permit.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{File, Metadata};
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_cpu::math::haswell_bulk_f32::BulkF32Admission;
use zerogravity_cpu::math::zen_blas_q8_f32::{ZenQ8F32Request, execute};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const LIMIT: u64 = 128 * 1024 * 1024;
const METADATA: u64 = 4 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const WEIGHT_BYTES: usize = 4_456_448;
const OUTPUT_WORDS: usize = 131_072;
const OUTPUT_BYTES: u64 = 524_288;
const SCRATCH_BYTES: u64 = 16_777_472;
const CAPTURE: &str =
    ".agents/runs/numerics-and-context-2b-bulk-causality-20261020/root-first64-repack-transport";
const WEIGHT_HASH: &str = "6d27d16495f7bc863fb414206fe3d8aa57c128aa1f464e7fc0e0f1339cec4c35";
const OUTPUT_HASH: &str = "5477c01a3885382f18640bf835ec10b00e1d5b2611b5fb1c1889653b64027059";

fn unchanged(before: &Metadata, after: &Metadata) {
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
        "source changed during retained-FD read"
    );
}

fn pin(raw: &[u8], length: usize, hash: &str) {
    assert_eq!(raw.len(), length);
    assert_eq!(format!("{:x}", Sha256::digest(raw)), hash);
}

fn read_pinned(path: &Path, length: usize, hash: &str) -> Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let before = file.metadata()?;
    assert!(before.is_file());
    assert_eq!(before.len(), length as u64, "{path:?}");
    let mut raw = Vec::with_capacity(length + 1);
    (&mut file).take(length as u64 + 1).read_to_end(&mut raw)?;
    unchanged(&before, &file.metadata()?);
    pin(&raw, length, hash);
    Ok(raw)
}

fn original_weights(root: &Path) -> Result<Vec<u8>> {
    let mut file = File::open(root.join("target/models/qwen35-2b/model.gguf"))?;
    let before = file.metadata()?;
    assert!(before.is_file());
    assert_eq!(before.len(), 1_296_764_000);
    let mut header = vec![0; 10_962_784];
    file.read_exact(&mut header)?;
    pin(
        &header,
        10_962_784,
        "b1f61f89c0a3a68937595d00b1137c5ffbce9edac55a568ca27bdcdae16cb7e0",
    );
    drop(header);
    file.seek(SeekFrom::Start(459_983_328))?;
    let mut weights = vec![0; WEIGHT_BYTES];
    file.read_exact(&mut weights)?;
    pin(&weights, WEIGHT_BYTES, WEIGHT_HASH);
    assert_eq!(file.seek(SeekFrom::End(0))?, before.len());
    assert_eq!(file.read(&mut [0u8; 1])?, 0, "original file EOF");
    unchanged(&before, &file.metadata()?);
    Ok(weights)
}

fn records(raw: &[u8]) -> Result<Vec<Value>> {
    std::str::from_utf8(raw)?
        .lines()
        .map(|line| Ok(serde_json::from_str(line)?))
        .collect()
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

#[test]
#[ignore = "requires original approved2B GGUF/captured BLAS operands via absolute ZG_REFERENCE_ROOT"]
fn original_q8_blas_matches_complete_captured_output_and_retains_lease() -> Result<()> {
    let root = PathBuf::from(std::env::var_os("ZG_REFERENCE_ROOT").expect("set ZG_REFERENCE_ROOT"));
    assert!(root.is_absolute() && root.is_dir());
    let capture = root.join(CAPTURE);
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "original2B-zen-q8-blas-leaf")?;
    arbiter.register(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("captured-original2B-q8", "pure-leaf-test")?;
    let metadata = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: METADATA,
        }],
    )?;
    let rows = records(&read_pinned(
        &capture.join("bulk64/records.jsonl"),
        55_822,
        "eacee1e7cf45fdac3ea40e48152432347f1f8f3915c633ecc2ce4c8f59e54437",
    )?)?;
    for (field, stage, tensor, operation) in [
        (40, "linear_attn_out-0", 45, "MUL_MAT"),
        (41, "projection_input::blk.0.ssm_out.weight", 43, "RESHAPE"),
        (42, "projection_output::blk.0.ssm_out.weight", 45, "MUL_MAT"),
    ] {
        let found = rows
            .iter()
            .filter(|row| row["file"] == format!("field-{field}.f32le"))
            .collect::<Vec<_>>();
        assert_eq!(found.len(), 1);
        let row = found[0];
        assert_eq!(row["stage"], stage);
        assert_eq!(row["native_op"], operation);
        assert_eq!(row["native_tensor_id"], tensor);
        assert_eq!(row["mode"], "bulk64");
        assert_eq!(row["call"], 0);
        assert_eq!(row["call_tokens"], 64);
        assert_eq!(row["window_start"], 0);
        assert_eq!(row["window_tokens"], 64);
        assert_eq!(row["layer"], 0);
        assert_eq!(row["occurrence"], 0);
        assert_eq!(row["scope"], "actual_tensor");
        assert_eq!(row["dense_ne"], json!([2048, 64, 1, 1]));
        assert_eq!(row["native_ne"], json!([2048, 64, 1, 1]));
        assert_eq!(row["native_nb"], json!([4, 8192, 524288, 524288]));
        assert_eq!(row["elements"], OUTPUT_WORDS);
        if field != 41 {
            assert_eq!(row["native_source_ids"][1], 43);
        }
    }
    let ids = read_pinned(
        &capture.join("input_ids.u32le"),
        32_768,
        "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
    )?;
    pin(
        &ids[..256],
        256,
        "ff7862336b7b2b5520723bf753d0f9e484ef9e66f285efd6189c1d945ec17625",
    );
    drop(ids);
    let learned = records(&read_pinned(
        &capture.join("learned-operands/records.jsonl"),
        17_570,
        "26d8a9d96194a38d6125ab339abc7564229cd98e65dc6ef7bee1751fa7bb7590",
    )?)?;
    let selected = learned
        .iter()
        .filter(|row| row["name"] == "blk.0.ssm_out.weight")
        .collect::<Vec<_>>();
    assert_eq!(selected.len(), 1);
    let descriptor = selected[0];
    assert_eq!(descriptor["encoding"], "GGUF_RAW");
    assert_eq!(descriptor["type"], 8);
    assert_eq!(descriptor["type_name"], "q8_0");
    assert_eq!(descriptor["buffer_name"], "CPU_Mapped");
    assert_eq!(descriptor["ne"], json!([2048, 2048, 1, 1]));
    assert_eq!(descriptor["nb"], json!([34, 2176, 4456448, 4456448]));
    assert_eq!(descriptor["bytes"], WEIGHT_BYTES);
    let chunks = descriptor["chunks"]
        .as_array()
        .expect("original chunk records");
    assert_eq!(chunks.len(), 5);
    let staging = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::TransferStaging,
            bytes: STAGING,
        }],
    )?;
    let weights = original_weights(&root)?;
    let mut captured = Vec::with_capacity(WEIGHT_BYTES);
    for (index, chunk) in chunks.iter().enumerate() {
        let name = format!("learned-10-{index}.bin");
        let length = if index < 4 { 1_048_576 } else { 262_144 };
        assert_eq!(chunk["file"], name);
        assert_eq!(chunk["offset"], captured.len());
        assert_eq!(chunk["bytes"], length);
        let path = capture.join("learned-operands").join(name);
        let mut file = File::open(path)?;
        let before = file.metadata()?;
        assert!(before.is_file());
        assert_eq!(before.len(), length as u64);
        let start = captured.len();
        (&mut file)
            .take(length as u64 + 1)
            .read_to_end(&mut captured)?;
        assert_eq!(captured.len() - start, length, "complete chunk and EOF");
        unchanged(&before, &file.metadata()?);
    }
    pin(&captured, WEIGHT_BYTES, WEIGHT_HASH);
    assert_eq!(
        captured, weights,
        "complete original raw Q8_0 interval equals captured operand"
    );
    drop(captured);
    let raw = read_pinned(
        &capture.join("bulk64/field-41.f32le"),
        OUTPUT_BYTES as usize,
        "4400491b963ae5fa113425e97ed6dafc3e8ca74b7af406fe77578bf5a6bee486",
    )?;
    let input = raw
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes(raw.try_into().expect("four bytes")))
        .collect::<Vec<_>>();
    drop(raw);
    let source = SourceIdentity::for_new_allocation()?;
    let request = || ZenQ8F32Request {
        source,
        weights: &weights,
        input: &input,
    };
    let admission = || BulkF32Admission {
        arbiter: &arbiter,
        resource: &key,
        owner: &owner,
    };
    let baseline = arbiter.snapshot(&key)?.reserved;
    let cancelled = AtomicBool::new(true);
    assert!(execute(request(), admission(), Some(&cancelled)).is_err());
    assert_eq!(arbiter.snapshot(&key)?.reserved, baseline);
    let pressure = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: arbiter.snapshot(&key)?.remaining - OUTPUT_BYTES - SCRATCH_BYTES + 1,
        }],
    )?;
    let pressed = arbiter.snapshot(&key)?.reserved;
    assert!(matches!(
        execute(request(), admission(), None),
        Err(zerogravity_core::Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(&key)?.reserved, pressed);
    drop(pressure);
    let result = execute(request(), admission(), None)?;
    assert_eq!(result.source(), source);
    assert_eq!(result.shape(), [2048, 64, 1, 1]);
    assert_eq!(result.values().len(), OUTPUT_WORDS);
    assert_eq!(class(&arbiter, &key, AllocationClass::Output), OUTPUT_BYTES);
    assert_eq!(class(&arbiter, &key, AllocationClass::Scratch), 0);
    let live = arbiter.snapshot(&key)?.reserved;
    assert_eq!(live, METADATA + STAGING + OUTPUT_BYTES);
    // No expected arithmetic bytes are opened before the actual Rust leaf.
    let expected = read_pinned(
        &capture.join("bulk64/field-40.f32le"),
        OUTPUT_BYTES as usize,
        OUTPUT_HASH,
    )?;
    let paired = read_pinned(
        &capture.join("bulk64/field-42.f32le"),
        OUTPUT_BYTES as usize,
        OUTPUT_HASH,
    )?;
    assert_eq!(
        expected, paired,
        "same original output tensor at both captured boundaries"
    );
    let unequal = result
        .values()
        .iter()
        .zip(expected.chunks_exact(4))
        .filter(|(value, raw)| {
            value.to_bits() != u32::from_le_bytes((*raw).try_into().expect("four bytes"))
        })
        .count();
    assert_eq!(unequal, 0, "all131072 original DEFAULT BLAS output bits");
    drop(expected);
    drop(paired);
    drop(weights);
    drop(input);
    drop(staging);
    assert_eq!(arbiter.snapshot(&key)?.reserved, live - STAGING);
    assert_eq!(class(&arbiter, &key, AllocationClass::Output), OUTPUT_BYTES);
    let mut hash = Sha256::new();
    for word in result.values() {
        hash.update(word.to_bits().to_le_bytes());
    }
    assert_eq!(
        format!("{:x}", hash.finalize()),
        OUTPUT_HASH,
        "retained result after input release"
    );
    drop(result);
    assert_eq!(arbiter.snapshot(&key)?.reserved, METADATA);
    drop(rows);
    drop(learned);
    drop(metadata);
    let final_state = arbiter.snapshot(&key)?;
    assert_eq!(final_state.reserved, 0);
    assert_eq!(final_state.reservations, 0);
    assert!(!final_state.fenced);
    println!(
        "ZEN_Q8_BLAS_SOURCE_PASS output_words=131072 original_raw_interval_exact=true model_state_unqualified=true"
    );
    Ok(())
}
