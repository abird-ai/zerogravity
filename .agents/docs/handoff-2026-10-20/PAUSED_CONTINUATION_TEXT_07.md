# Paused continuation text — volume 7

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/factory.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 29779; SHA256 `2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69`.

````text
#![allow(dead_code)]
//! Root-only actual-source byte/window harness. No model or GPU execution.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::ops::Range;
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
use zerogravity_artifacts::Error as ZgError;
use zerogravity_artifacts::artifact::Artifact;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_artifacts::source_weights::{
    ExpectedSourceShard, SourceWeights, VerifiedMatrix, VerifiedTensorSource, VerifiedWindowPlan,
    WindowOpenLimits,
};
use zerogravity_core::accelerator::program::MatrixAxes;
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};

type Proof<T> = Result<T, Box<dyn Error>>;
const SNAPSHOT: &str = "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e";
const REVISION: &str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
const REPORT_SHA: &str = "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a";
const INDEX_SHA: &str = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df";
const BUNDLE_SHA: &str = "7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2";
const SELECTED_BYTES: u64 = 55_586_113_293;
const WEIGHT_BYTES: u64 = 55_563_006_776;
const PAYLOAD_BYTES: u64 = 55_562_855_904;
const META: u64 = 64 * 1024 * 1024;
const HOST: u64 = 128 * 1024 * 1024;
const SCAN: usize = 4 * 1024 * 1024;
const SOURCE_DIR: &str = "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files";

struct Shard {
    name: String,
    expected_bytes: u64,
    expected_sha: String,
    // Both Artifact and independent raw reference derive from this one open FD.
    file: File,
}

fn ensure(condition: bool, message: impl Into<String>) -> Proof<()> {
    if !condition {
        let message: String = message.into();
        return Err(message.into());
    }
    Ok(())
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn bounded_file(path: &Path, maximum: usize) -> Proof<Vec<u8>> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    ensure(file.metadata()?.is_file(), "control source is not regular")?;
    ensure(
        file.metadata()?.len() <= maximum as u64,
        "control source byte cap",
    )?;
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure(bytes.len() <= maximum, "control source grew beyond cap")?;
    Ok(bytes)
}

fn reserved(host: &TransferAdmission) -> Proof<u64> {
    Ok(host.arbiter.snapshot(&host.resource)?.reserved)
}

fn ledger(host: &TransferAdmission, metadata: u64, staging: u64, other: u64) -> Proof<()> {
    let current = host.arbiter.snapshot(&host.resource)?;
    ensure(!current.fenced, "host window pool unexpectedly fenced")?;
    ensure(current.limit == HOST, "configured pool limit changed")?;
    ensure(
        current.reserved == metadata + staging + other,
        "live total ledger mismatch",
    )?;
    for (class, expected) in [
        (AllocationClass::Scratch, metadata),
        (AllocationClass::TransferStaging, staging),
        (AllocationClass::Other, other),
    ] {
        ensure(
            current.per_class.get(&class).copied().unwrap_or(0) == expected,
            format!("live allocation-class mismatch: {class:?}"),
        )?;
    }
    ensure(
        current.remaining + current.reserved == HOST,
        "remaining ledger mismatch",
    )?;
    Ok(())
}

fn limits() -> WindowOpenLimits {
    WindowOpenLimits {
        max_source_bytes: 64 * 1024 * 1024 * 1024,
        max_metadata_bytes: META,
        max_shards: 18,
        max_tensors: 1199,
        scan_chunk_bytes: SCAN,
    }
}

fn expected_shards(shards: &[Shard]) -> Proof<Vec<ExpectedSourceShard>> {
    shards
        .iter()
        .map(|shard| {
            let clone = shard.file.try_clone()?;
            let original = shard.file.metadata()?;
            let same = clone.metadata()?;
            ensure(
                (same.dev(), same.ino(), same.len())
                    == (original.dev(), original.ino(), original.len()),
                "descriptor clone changed source",
            )?;
            Ok(ExpectedSourceShard::new(
                shard.name.clone(),
                Artifact::from_file(clone, Path::new(&shard.name))?,
                shard.expected_bytes,
                shard.expected_sha.clone(),
            )?)
        })
        .collect()
}

fn plan(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<VerifiedWindowPlan> {
    Ok(SourceWeights::prepare_verified_windows(
        index,
        INDEX_SHA,
        expected_shards(shards)?,
        limits(),
        host,
    )?)
}

fn expected_limit<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
    match result {
        Err(ZgError::Limit(message)) => Ok(message),
        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
        Ok(value) => {
            drop(value);
            Err(format!("{label}: operation unexpectedly succeeded").into())
        }
    }
}

fn expected_cancel<T>(result: zerogravity_artifacts::Result<T>, label: &str) -> Proof<String> {
    match result {
        Err(ZgError::State(message)) if message.contains("cancelled") => Ok(message),
        Err(error) => Err(format!("{label}: wrong refusal: {error}").into()),
        Ok(value) => {
            drop(value);
            Err(format!("{label}: operation unexpectedly succeeded").into())
        }
    }
}

fn guards(index: &[u8], shards: &[Shard], host: &TransferAdmission) -> Proof<Value> {
    let mut small_metadata = limits();
    small_metadata.max_metadata_bytes = 1;
    let metadata = expected_limit(
        SourceWeights::prepare_verified_windows(
            index,
            INDEX_SHA,
            expected_shards(shards)?,
            small_metadata,
            host,
        ),
        "metadata envelope",
    )?;
    ledger(host, 0, 0, 0)?;

    let mut source_limit = limits();
    source_limit.max_source_bytes = WEIGHT_BYTES - 1;
    let source = expected_limit(
        SourceWeights::prepare_verified_windows(
            index,
            INDEX_SHA,
            expected_shards(shards)?,
            source_limit,
            host,
        ),
        "whole-source file bound",
    )?;
    ledger(host, 0, 0, 0)?;

    let held_bytes = HOST - META + 1;
    let held = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: held_bytes,
        }],
    )?;
    let unavailable = expected_limit(
        SourceWeights::prepare_verified_windows(
            index,
            INDEX_SHA,
            expected_shards(shards)?,
            limits(),
            host,
        ),
        "metadata live-ledger admission",
    )?;
    ledger(host, 0, 0, held_bytes)?;
    drop(held);
    ledger(host, 0, 0, 0)?;

    let pending = plan(index, shards, host)?;
    ledger(host, META, 0, 0)?;
    let cancelled = expected_cancel(
        pending.open_cancellable(&AtomicBool::new(true)),
        "pre-cancelled factory open",
    )?;
    ledger(host, 0, 0, 0)?;
    Ok(json!({
        "metadata_envelope_refused":metadata,
        "source_file_bound_refused":source,
        "metadata_live_ledger_refused":unavailable,
        "pre_cancelled_factory_refused":cancelled,
        "scope":"real headers inspected; factory pre-cancel before whole payload hash scan; no source published",
        "post_guard_reserved_bytes":0,
    }))
}

fn expected_staging(shard_bytes: u64, offset: u64, bytes: usize) -> Proof<u64> {
    let end = offset
        .checked_add(bytes as u64)
        .ok_or("reference range overflow")?;
    ensure(
        bytes != 0 && end <= shard_bytes,
        "reference range outside source",
    )?;
    let first = offset / SCAN as u64;
    let last = (end - 1) / SCAN as u64;
    let staged = ((last + 1) * SCAN as u64).min(shard_bytes) - first * SCAN as u64;
    let offsets = (last - first + 1) * std::mem::size_of::<(usize, Range<usize>)>() as u64;
    Ok(staged + offsets)
}

fn compare_rows(
    source: &VerifiedTensorSource,
    matrix: &VerifiedMatrix,
    raw: &Shard,
    rows: Range<usize>,
    label: &str,
    host: &TransferAdmission,
) -> Proof<Value> {
    ledger(host, META, 0, 0)?;
    let offset = matrix.tensor().byte_offset + rows.start as u64 * matrix.columns() as u64 * 2;
    let bytes = (rows.end - rows.start) * matrix.columns() * 2;
    ensure(bytes <= 256 * 1024, "raw reference buffer cap")?;
    let window = source.read_rows(matrix, rows.clone())?;
    let charge = expected_staging(raw.expected_bytes, offset, bytes)?;
    ensure(
        window.charged_staging_bytes() as u64 == charge,
        "exact staged payload+offset charge",
    )?;
    ledger(host, META, charge, 0)?;
    ensure(
        window.row_range() == rows && window.source_offset() == offset,
        "window row/offset mismatch",
    )?;
    ensure(
        window.as_slice().len() == bytes,
        "window omitted full-width source row",
    )?;
    ensure(
        window.matrix().source_identity() == source.source_identity(),
        "window source identity changed",
    )?;
    ensure(
        window.verified_bundle_sha256() == BUNDLE_SHA,
        "window bundle changed",
    )?;
    ensure(
        window.shard_basename() == raw.name && window.shard_sha256() == raw.expected_sha,
        "window original shard binding changed",
    )?;
    ensure(
        window.extent_index_sha256().len() == 64,
        "extent receipt absent",
    )?;

    // The harness's independent comparison copy has a separate, explicit charge.
    let reference_lease = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: bytes as u64,
        }],
    )?;
    let mut reference = vec![0u8; bytes];
    raw.file.read_exact_at(&mut reference, offset)?;
    ledger(host, META, charge, bytes as u64)?;
    ensure(
        reference.as_slice() == window.as_slice(),
        "actual source row byte mismatch",
    )?;
    let record = json!({
        "matrix":matrix.tensor().name,
        "case":label,
        "rows":[rows.start,rows.end],
        "shape":[matrix.rows(),matrix.columns()],
        "dtype":"BF16",
        "axes":"OutputInput",
        "source_identity":format!("{:?}",source.source_identity()),
        "shard":raw.name,
        "shard_sha256":raw.expected_sha,
        "source_offset":offset,
        "requested_bytes":bytes,
        "charged_staging_bytes":charge,
        "crosses_scan_extent":offset / SCAN as u64 != (offset + bytes as u64 - 1) / SCAN as u64,
        "byte_exact":true,
        "raw_and_window_sha256":sha(&reference),
        "extent_index_sha256":window.extent_index_sha256(),
    });
    drop(reference);
    drop(reference_lease);
    ledger(host, META, charge, 0)?;
    drop(window);
    ledger(host, META, 0, 0)?;
    Ok(record)
}

fn exercise(
    source: VerifiedTensorSource,
    shards: &[Shard],
    host: &TransferAdmission,
) -> Proof<Value> {
    ensure(
        source.source_identity().is_bound(),
        "factory did not mint bound source identity",
    )?;
    let identity = source.source_identity();
    ensure(
        source.source_bytes() == WEIGHT_BYTES,
        "whole weight byte count",
    )?;
    ensure(
        source.metadata_reserved_bytes() == META,
        "source metadata charge",
    )?;
    ensure(
        source.verified_shard_index_sha256() == INDEX_SHA,
        "source index pin",
    )?;
    ensure(
        source.verified_bundle_sha256() == BUNDLE_SHA,
        "source bundle pin",
    )?;
    ledger(host, META, 0, 0)?;
    let clone = source.clone();
    ensure(
        clone.source_identity() == identity,
        "source clone minted another identity",
    )?;
    drop(clone);

    let specimens = [
        (
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            10_240,
            5_120,
            1_120_792,
            "model-00001-of-00018.safetensors",
        ),
        (
            "model.language_model.embed_tokens.weight",
            248_320,
            5_120,
            152,
            "model-00003-of-00018.safetensors",
        ),
        (
            "model.language_model.layers.63.mlp.down_proj.weight",
            5_120,
            17_408,
            1_364_259_120,
            "model-00017-of-00018.safetensors",
        ),
        (
            "lm_head.weight",
            248_320,
            5_120,
            1_760,
            "model-00018-of-00018.safetensors",
        ),
    ];
    let mut matrices = Vec::new();
    let mut samples = Vec::new();
    for (name, rows, columns, absolute_offset, basename) in specimens {
        let matrix = source.matrix(name, MatrixAxes::OutputInput)?;
        ensure(
            matrix.source_identity() == identity,
            "matrix changed source identity",
        )?;
        ensure(
            matrix.axes() == MatrixAxes::OutputInput,
            "matrix axes changed",
        )?;
        ensure(
            (matrix.rows(), matrix.columns()) == (rows, columns),
            "actual matrix shape differs from pinned header",
        )?;
        ensure(
            matrix.tensor().shape == [rows as u64, columns as u64],
            "raw source axes mismatch",
        )?;
        ensure(
            matrix.tensor().kind == QuantKind::Bf16,
            "raw source is not BF16",
        )?;
        ensure(
            matrix.tensor().byte_len == (rows * columns * 2) as u64,
            "raw source byte geometry",
        )?;
        ensure(
            matrix.tensor().byte_offset == absolute_offset,
            "actual source offset differs from independently inspected original header",
        )?;
        let raw = shards
            .iter()
            .find(|shard| shard.name == basename)
            .ok_or("missing raw source FD")?;
        samples.push(compare_rows(
            &source,
            &matrix,
            raw,
            0..1,
            "first_full_width_row",
            host,
        )?);
        samples.push(compare_rows(
            &source,
            &matrix,
            raw,
            rows - 1..rows,
            "last_full_width_row",
            host,
        )?);
        let start = matrix.tensor().byte_offset;
        let boundary = (start / SCAN as u64 + 1) * SCAN as u64;
        let split_row = ((boundary - start) / (columns * 2) as u64) as usize;
        let split = split_row.saturating_sub(1)..(split_row + 1).min(rows);
        let requested_start = start + split.start as u64 * (columns * 2) as u64;
        let requested_end = start + split.end as u64 * (columns * 2) as u64;
        ensure(
            requested_start < boundary && requested_end > boundary,
            "fixture does not straddle actual scan boundary",
        )?;
        samples.push(compare_rows(
            &source,
            &matrix,
            raw,
            split,
            "scan_extent_split_full_width_rows",
            host,
        )?);
        matrices.push(matrix);
    }

    let matrix = &matrices[0];
    let cancelled = expected_cancel(
        source.read_rows_cancellable(matrix, 0..1, &AtomicBool::new(true)),
        "pre-cancelled row",
    )?;
    ledger(host, META, 0, 0)?;
    ensure(
        source
            .matrix(&matrix.tensor().name, MatrixAxes::InputOutput)
            .is_err(),
        "wrong axes admitted",
    )?;
    ensure(
        source
            .matrix(
                "model.language_model.layers.0.input_layernorm.weight",
                MatrixAxes::OutputInput,
            )
            .is_err(),
        "actual rank-one parameter admitted as matrix",
    )?;
    ensure(
        source
            .matrix("__not_in_the_original_index__", MatrixAxes::OutputInput)
            .is_err(),
        "unknown matrix admitted",
    )?;
    for rows in [
        0..0,
        matrix.rows()..matrix.rows() + 1,
        usize::MAX..usize::MAX,
    ] {
        ensure(
            source.read_rows(matrix, rows).is_err(),
            "invalid row range admitted",
        )?;
        ledger(host, META, 0, 0)?;
    }
    let oversized = expected_limit(
        source.read_rows(&matrices[1], 0..matrices[1].rows()),
        "whole embedding tensor materialization",
    )?;
    ledger(host, META, 0, 0)?;

    let pin = matrices[0].clone();
    let last = &matrices[3];
    let retained = source.read_rows(last, last.rows() - 1..last.rows())?;
    let retained_sha = sha(retained.as_slice());
    let charge = retained.charged_staging_bytes() as u64;
    ledger(host, META, charge, 0)?;
    drop(matrices);
    drop(source);
    ensure(
        pin.source_identity() == identity && retained.matrix().source_identity() == identity,
        "descriptor/window lost original source identity",
    )?;
    ledger(host, META, charge, 0)?;
    drop(pin);
    // Only the owned immutable window retains source metadata and its live FD set.
    ledger(host, META, charge, 0)?;
    ensure(
        sha(retained.as_slice()) == retained_sha,
        "window bytes changed after all caller source handles dropped",
    )?;
    drop(retained);
    ledger(host, 0, 0, 0)?;
    Ok(json!({
        "source_identity":format!("{identity:?}"),
        "source_bytes":WEIGHT_BYTES,
        "source_bundle_sha256":BUNDLE_SHA,
        "index_sha256":INDEX_SHA,
        "matrix_count":4,
        "samples":samples,
        "pre_cancelled_row_refused":cancelled,
        "wrong_axes_missing_name_rank_one_bad_rows_refused":true,
        "whole_tensor_materialization_refused":oversized,
        "only_window_retains_source_metadata_and_stage_after_all_caller_sources_drop":true,
        "window_after_source_drop_sha256":retained_sha,
        "final_reserved_bytes":0,
    }))
}

fn run(host: &TransferAdmission) -> Proof<Value> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
    ensure(
        sha(&report_bytes) == REPORT_SHA,
        "acquired source report pin mismatch",
    )?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    ensure(
        report["snapshot"] == SNAPSHOT
            && report["repository"] == "Qwen/Qwen3.8-27B"
            && report["revision"] == REVISION
            && report["reference_directory"] == SOURCE_DIR,
        "acquired snapshot provenance differs",
    )?;
    ensure(
        report["selected_bytes"] == SELECTED_BYTES
            && report["weight_bytes"] == WEIGHT_BYTES
            && report["payload_bytes"] == PAYLOAD_BYTES
            && report["tensor_count"] == 1199,
        "acquired byte/count pin differs",
    )?;
    let files = report["files"]
        .as_array()
        .ok_or("source report files absent")?;
    ensure(files.len() == 31, "selected source file count")?;
    let mut pinned = BTreeMap::new();
    let mut total = 0u64;
    for entry in files {
        let name = entry["name"].as_str().ok_or("entry name absent")?;
        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
        total = total
            .checked_add(bytes)
            .ok_or("selected byte sum overflow")?;
        ensure(
            pinned
                .insert(name.to_string(), (bytes, hash.to_string()))
                .is_none(),
            "duplicate source report entry",
        )?;
    }
    ensure(
        total == SELECTED_BYTES,
        "selected report bytes do not close",
    )?;
    let index = bounded_file(
        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
        1024 * 1024,
    )?;
    ensure(
        sha(&index) == INDEX_SHA,
        "actual original index SHA256 mismatch",
    )?;
    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
    ensure(
        index_info.declared_total_size() == PAYLOAD_BYTES
            && index_info.tensors().len() == 1199
            && index_info.shards().len() == 18,
        "actual original index closure differs",
    )?;
    let mut shards = Vec::with_capacity(18);
    let mut descriptors = Vec::new();
    for number in 1..=18 {
        let name = format!("model-{number:05}-of-00018.safetensors");
        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
        ensure(
            index_info.shards().contains(name.as_str()),
            "pin has extra shard",
        )?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(Path::new(SOURCE_DIR).join(&name))?;
        let stat = file.metadata()?;
        ensure(
            stat.is_file() && stat.len() == *bytes,
            "opened original shard type/length differs",
        )?;
        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
        shards.push(Shard {
            name,
            expected_bytes: *bytes,
            expected_sha: hash.clone(),
            file,
        });
    }
    ensure(
        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
        "whole original weight-file byte sum differs",
    )?;
    eprintln!("phase=real_header_only_guards source_weight_bytes={WEIGHT_BYTES}");
    let refusals = guards(&index, &shards, host)?;
    let pending = plan(&index, &shards, host)?;
    ensure(
        pending.source_bytes() == WEIGHT_BYTES && pending.metadata_reserved_bytes() == META,
        "prepared whole-source bounds differ",
    )?;
    ledger(host, META, 0, 0)?;
    eprintln!(
        "phase=whole_source_sha256_scan_start shard_count=18 scan_buffer_bytes={SCAN} metadata_reserved_bytes={META}"
    );
    let started = Instant::now();
    let source = pending.open()?;
    let scan_milliseconds = started.elapsed().as_millis();
    eprintln!("phase=whole_source_sha256_scan_ready elapsed_ms={scan_milliseconds}");
    let windows = exercise(source, &shards, host)?;
    ledger(host, 0, 0, 0)?;
    Ok(json!({
        "schema":"zerogravity.original-owned-window-qualification.v1",
        "status":"PASS_SOURCE_WINDOWS_ONLY",
        "repository":"Qwen/Qwen3.8-27B",
        "revision":REVISION,
        "snapshot":SNAPSHOT,
        "selected_artifact_bytes":SELECTED_BYTES,
        "weight_file_bytes_scanned":WEIGHT_BYTES,
        "tensor_payload_bytes":PAYLOAD_BYTES,
        "whole_source_sha256_verified":true,
        "retained_raw_reference_fds":descriptors,
        "scan_chunk_bytes":SCAN,
        "whole_source_scan_milliseconds":scan_milliseconds,
        "configured_metadata_reserved_bytes":META,
        "configured_host_pool_limit_bytes":HOST,
        "refusals":refusals,
        "windows":windows,
        "physical_host_ram_admission_qualified":false,
        "trained_math_or_model_inference_qualified":false,
        "ssd_gpu_streaming_inference_qualified":false,
        "gpu_or_state_root_publication_performed":false,
        "scope":"one source issuer; whole weight SHA/index/header closure; owned bounded raw BF16 row bytes; configured host ledger only",
    }))
}

fn main() {
    let arbiter = ResourceArbiter::process_shared();
    let resource = PhysicalDeviceKey::new("host", "explicit-owned-window-qualification-pool")
        .expect("fixed resource label");
    arbiter
        .register_matching(resource.clone(), HOST)
        .expect("fresh explicit pool registration");
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new(
            "Qwen/Qwen3.8-27B",
            "actual-original-source-window-qualification",
        )
        .expect("fixed source owner"),
    };
    let result = run(&host);
    let final_state = host.arbiter.snapshot(&host.resource);
    let clean = matches!(&final_state, Ok(state) if state.reserved == 0 && state.reservations == 0 && !state.fenced);
    let success = result.is_ok() && clean;
    let mut report = match result {
        Ok(report) => report,
        Err(error) => json!({"status":"FAIL_UNQUALIFIED","error":error.to_string()}),
    };
    report["final_owned_ledger_clean"] = json!(clean);
    report["final_owned_ledger_reserved_bytes"] =
        json!(final_state.as_ref().ok().map(|state| state.reserved));
    if !clean {
        report["status"] = json!("FAIL_UNQUALIFIED");
        report["ledger_cleanup_error"] = json!(format!("{final_state:?}"));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("bounded report serialization")
    );
    if !success {
        std::process::exit(2);
    }
}

/// Only this additive wrapper is called by the new replay. Original factory
/// source below remains retained verbatim; its old main/exercise are not called.
pub struct SourceReady {
    pub source: VerifiedTensorSource,
    pub config: Vec<u8>,
    pub descriptors: Vec<Value>,
    pub scan_milliseconds: u128,
}
pub fn open_source(host: &TransferAdmission) -> Proof<SourceReady> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let report_bytes = bounded_file(&root.join("source-pin.json"), 64 * 1024)?;
    ensure(
        sha(&report_bytes) == REPORT_SHA,
        "acquired source report pin mismatch",
    )?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    ensure(
        report["snapshot"] == SNAPSHOT
            && report["repository"] == "Qwen/Qwen3.8-27B"
            && report["revision"] == REVISION
            && report["reference_directory"] == SOURCE_DIR,
        "acquired snapshot provenance differs",
    )?;
    ensure(
        report["selected_bytes"] == SELECTED_BYTES
            && report["weight_bytes"] == WEIGHT_BYTES
            && report["payload_bytes"] == PAYLOAD_BYTES
            && report["tensor_count"] == 1199,
        "acquired byte/count pin differs",
    )?;
    let files = report["files"]
        .as_array()
        .ok_or("source report files absent")?;
    ensure(files.len() == 31, "selected source file count")?;
    let mut pinned = BTreeMap::new();
    let mut total = 0u64;
    for entry in files {
        let name = entry["name"].as_str().ok_or("entry name absent")?;
        let bytes = entry["bytes"].as_u64().ok_or("entry bytes absent")?;
        let hash = entry["sha256"].as_str().ok_or("entry hash absent")?;
        total = total
            .checked_add(bytes)
            .ok_or("selected byte sum overflow")?;
        ensure(
            pinned
                .insert(name.to_string(), (bytes, hash.to_string()))
                .is_none(),
            "duplicate source report entry",
        )?;
    }
    ensure(
        total == SELECTED_BYTES,
        "selected report bytes do not close",
    )?;
    let index = bounded_file(
        &Path::new(SOURCE_DIR).join("model.safetensors.index.json"),
        1024 * 1024,
    )?;
    ensure(
        sha(&index) == INDEX_SHA,
        "actual original index SHA256 mismatch",
    )?;
    let index_info = zerogravity_artifacts::sharded::SafetensorsIndex::from_json(&index)?;
    ensure(
        index_info.declared_total_size() == PAYLOAD_BYTES
            && index_info.tensors().len() == 1199
            && index_info.shards().len() == 18,
        "actual original index closure differs",
    )?;
    let mut shards = Vec::with_capacity(18);
    let mut descriptors = Vec::new();
    for number in 1..=18 {
        let name = format!("model-{number:05}-of-00018.safetensors");
        let (bytes, hash) = pinned.get(&name).ok_or("pinned shard absent")?;
        ensure(
            index_info.shards().contains(name.as_str()),
            "pin has extra shard",
        )?;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(Path::new(SOURCE_DIR).join(&name))?;
        let stat = file.metadata()?;
        ensure(
            stat.is_file() && stat.len() == *bytes,
            "opened original shard type/length differs",
        )?;
        descriptors.push(json!({"shard":name,"dev":stat.dev(),"ino":stat.ino(),"bytes":stat.len(),"sha256":hash}));
        shards.push(Shard {
            name,
            expected_bytes: *bytes,
            expected_sha: hash.clone(),
            file,
        });
    }
    ensure(
        shards.iter().map(|shard| shard.expected_bytes).sum::<u64>() == WEIGHT_BYTES,
        "whole original weight-file byte sum differs",
    )?;
    let config = bounded_file(&Path::new(SOURCE_DIR).join("config.json"), 64 * 1024)?;
    ensure(
        sha(&config) == "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
        "original config SHA",
    )?;
    let pending = plan(&index, &shards, host)?;
    let started = Instant::now();
    let source = pending.open()?;
    ensure(
        source.source_bytes() == WEIGHT_BYTES
            && source.metadata_reserved_bytes() == META
            && source.verified_shard_index_sha256() == INDEX_SHA
            && source.verified_bundle_sha256() == BUNDLE_SHA,
        "source factory closure",
    )?;
    Ok(SourceReady {
        source,
        config,
        descriptors,
        scan_milliseconds: started.elapsed().as_millis(),
    })
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 25315; SHA256 `4ecfcb371d4b8cd9e0b06abc80f1da0af6078a9b87941d219117b7f08a0a2ea7`.

````text
//! ROOT ONLY: original H48 coefficient preparation, no model/session producer.
mod factory;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{FileExt, MetadataExt, OpenOptionsExt};
use std::path::{Component, Path};
use std::sync::atomic::AtomicBool;
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::accelerator::program::ProgramNumericalProfile;
use zerogravity_core::quant::QuantKind;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_model_qwen35::operator_slice::coefficient_preparation::{
    GdnCoefficientPreparationInput, GdnCoefficientPreparationOutput,
};
use zerogravity_model_qwen35::original_binding::{OriginalWindowOperators, QwenNumericalProfile};

type Proof<T> = Result<T, Box<dyn Error>>;
const MIB: u64 = 1024 * 1024;
const HOST: u64 = 128 * MIB;
const CONTROL: u64 = 8 * MIB;
const SOURCE_META: u64 = 64 * MIB;
const NEEDED: u64 = 2 * MIB; // Two original 64KiB Output +960KiB Scratch owners.
const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
const STAGES: [&str; 5] = [
    "alpha_bf16",
    "beta_input_bf16",
    "biased_alpha_f32",
    "softplus_f32",
    "beta_bf16",
];
fn ensure(ok: bool, message: impl Into<String>) -> Proof<()> {
    if ok {
        Ok(())
    } else {
        Err(message.into().into())
    }
}
fn text(v: &Value) -> Proof<&str> {
    v.as_str().ok_or_else(|| "required string".into())
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn stable(file: &File) -> Proof<[u64; 7]> {
    let m = file.metadata()?;
    ensure(m.is_file(), "regular retained input FD")?;
    Ok([
        m.dev(),
        m.ino(),
        m.len(),
        m.mtime() as u64,
        m.mtime_nsec() as u64,
        m.ctime() as u64,
        m.ctime_nsec() as u64,
    ])
}
fn open(path: &Path) -> Proof<File> {
    Ok(OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?)
}
fn read(path: &Path, cap: usize) -> Proof<Vec<u8>> {
    let f = open(path)?;
    let before = stable(&f)?;
    ensure(
        before[2] > 0 && before[2] <= cap as u64,
        "bounded regular metadata",
    )?;
    let mut raw = Vec::with_capacity(before[2] as usize);
    f.try_clone()?.take(cap as u64 + 1).read_to_end(&mut raw)?;
    ensure(
        raw.len() as u64 == before[2] && stable(&f)? == before,
        "metadata FD unchanged",
    )?;
    Ok(raw)
}
fn file_sha(file: &File) -> Proof<String> {
    let before = stable(file)?;
    let mut sum = Sha256::new();
    let mut buf = [0u8; 65536];
    let mut offset = 0u64;
    while offset < before[2] {
        let n = ((before[2] - offset) as usize).min(buf.len());
        file.read_exact_at(&mut buf[..n], offset)?;
        sum.update(&buf[..n]);
        offset += n as u64;
    }
    ensure(stable(file)? == before, "full retained archive FD stable")?;
    Ok(format!("{:x}", sum.finalize()))
}
fn relative(v: &Value) -> Proof<&Path> {
    let p = Path::new(text(v)?);
    ensure(
        !p.as_os_str().is_empty() && p.components().all(|p| matches!(p, Component::Normal(_))),
        "contained output name",
    )?;
    Ok(p)
}
fn write_json(path: &Path, value: &Value) -> Proof<()> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut f, value)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn snapshot(host: &TransferAdmission) -> Proof<DeviceBudgetSnapshot> {
    Ok(host.arbiter.snapshot(&host.resource)?)
}
fn class(row: &DeviceBudgetSnapshot, kind: AllocationClass) -> u64 {
    row.per_class.get(&kind).copied().unwrap_or(0)
}
fn ledger(row: &DeviceBudgetSnapshot) -> Value {
    json!({"reserved":row.reserved,"reservations":row.reservations,"fenced":row.fenced,"per_class":row.per_class.iter().map(|(k,v)|(format!("{k:?}"),*v)).collect::<BTreeMap<_,_>>()})
}
fn same_ledger(host: &TransferAdmission, before: &DeviceBudgetSnapshot) -> Proof<()> {
    let after = snapshot(host)?;
    ensure(
        after.reserved == before.reserved
            && after.reservations == before.reservations
            && after.per_class == before.per_class
            && !after.fenced,
        "failed call retires all private leases",
    )
}
struct Archive {
    file: File,
    pin: Value,
    identity: [u64; 7],
    header: Value,
    start: u64,
}
impl Archive {
    fn new(pin: &Value) -> Proof<Self> {
        let file = open(Path::new(text(&pin["path"])?))?;
        let identity = stable(&file)?;
        ensure(
            identity[2] == pin["bytes"].as_u64().ok_or("archive length")?
                && identity[2] <= 64 * MIB,
            "actual complete bounded archive",
        )?;
        ensure(
            file_sha(&file)? == text(&pin["sha256"])?,
            "whole actual archive SHA before extraction",
        )?;
        let mut size = [0u8; 8];
        file.read_exact_at(&mut size, 0)?;
        let n = u64::from_le_bytes(size);
        ensure(
            n > 1 && n <= MIB && 8 + n < identity[2],
            "archive header bound BEFORE read",
        )?;
        let mut raw = vec![0u8; n as usize];
        file.read_exact_at(&mut raw, 8)?;
        let header: Value = serde_json::from_slice(&raw)?;
        let fields = header.as_object().ok_or("archive fields")?;
        ensure(
            !fields.is_empty() && fields.len() <= 2048,
            "bounded archive field count",
        )?;
        let mut ranges = Vec::with_capacity(fields.len());
        for (name, item) in fields {
            if name == "__metadata__" {
                continue;
            }
            let dtype = text(&item["dtype"])?;
            ensure(matches!(dtype, "BF16" | "F32"), "source archive dtype")?;
            let shape = item["shape"].as_array().ok_or("archive rank")?;
            ensure(shape.len() <= 8, "archive rank bound")?;
            let mut words = 1u64;
            for v in shape {
                let axis = v
                    .as_u64()
                    .filter(|n| *n > 0 && *n <= 262144)
                    .ok_or("archive axis")?;
                words = words.checked_mul(axis).ok_or("archive count overflow")?;
            }
            let bounds = item["data_offsets"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or("archive offsets")?;
            let lo = bounds[0].as_u64().ok_or("archive begin")?;
            let hi = bounds[1].as_u64().ok_or("archive end")?;
            ensure(
                lo < hi
                    && hi <= identity[2] - 8 - n
                    && hi - lo
                        == words
                            .checked_mul(if dtype == "BF16" { 2 } else { 4 })
                            .ok_or("archive width overflow")?,
                "complete source archive extent",
            )?;
            ranges.push((lo, hi));
        }
        ranges.sort_unstable();
        let mut end = 0;
        for (lo, hi) in ranges {
            ensure(lo == end, "archive no gaps/overlap")?;
            end = hi;
        }
        ensure(
            8 + n + end == identity[2] && stable(&file)? == identity,
            "complete stable archive/no trailing",
        )?;
        Ok(Self {
            file,
            pin: pin.clone(),
            identity,
            header,
            start: 8 + n,
        })
    }
    fn raw(&self, row: &Value) -> Proof<Vec<u8>> {
        let item = &self.header[text(&row["name"])?];
        ensure(
            item["dtype"] == row["dtype"] && item["shape"] == row["shape"],
            "actual source name/dtype/shape",
        )?;
        let lo = item["data_offsets"][0].as_u64().ok_or("stage offset")?;
        let hi = item["data_offsets"][1].as_u64().ok_or("stage end")?;
        ensure(
            hi > lo && hi - lo == row["bytes"].as_u64().ok_or("stage bytes")? && hi - lo <= 192,
            "bounded complete coefficient stage",
        )?;
        ensure(
            stable(&self.file)? == self.identity,
            "archive FD before stage",
        )?;
        let mut raw = vec![0u8; (hi - lo) as usize];
        self.file.read_exact_at(&mut raw, self.start + lo)?;
        ensure(
            stable(&self.file)? == self.identity && sha(&raw) == text(&row["raw_sha256"])?,
            "exact raw observed stage before interpretation",
        )?;
        let width = if row["dtype"] == "BF16" { 2 } else { 4 };
        for b in raw.chunks_exact(width) {
            let bits = if width == 2 {
                u32::from(u16::from_le_bytes([b[0], b[1]])) << 16
            } else {
                u32::from_le_bytes(b.try_into()?)
            };
            ensure(
                f32::from_bits(bits).is_finite(),
                "selected original norm stage finite; preserve all bits",
            )?;
        }
        Ok(raw)
    }
    fn close_proof(&self) -> Proof<()> {
        ensure(
            stable(&self.file)? == self.identity
                && file_sha(&self.file)? == text(&self.pin["sha256"])?,
            "complete retained archive unchanged after all stage reads",
        )
    }
}
fn compare(
    actual: &[f32],
    row: &Value,
    archive: &Archive,
    out: &Path,
    label: &str,
    write: bool,
) -> Proof<Value> {
    let expected = archive.raw(row)?;
    let width = if row["dtype"] == "BF16" { 2 } else { 4 };
    ensure(
        actual.len() * width == expected.len(),
        "complete word coverage before zip",
    )?;
    let mut different = 0;
    let mut first = Value::Null;
    let mut max_abs = 0f64;
    for (i, (a, b)) in actual.iter().zip(expected.chunks_exact(width)).enumerate() {
        let bits = if width == 2 {
            u32::from(u16::from_le_bytes([b[0], b[1]])) << 16
        } else {
            u32::from_le_bytes(b.try_into()?)
        };
        ensure(
            a.is_finite() && (width != 2 || a.to_bits() & 0xffff == 0),
            "finite exact output dtype",
        )?;
        if a.to_bits() != bits {
            different += 1;
            max_abs = max_abs.max((f64::from(*a) - f64::from(f32::from_bits(bits))).abs());
            if first.is_null() {
                first = json!({"word":i,"actual":format!("{:08x}",a.to_bits()),"expected":format!("{bits:08x}")});
            }
        }
    }
    // Control envelope covers the one expected slice and this one emitted Vec.
    let bytes = actual
        .iter()
        .flat_map(|v| v.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let file = format!("{label}.f32le");
    if write {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(relative(&json!(file))?))?
            .write_all(&bytes)?;
    }
    Ok(
        json!({"label":label,"words":actual.len(),"different_words":different,"first_difference":first,"max_abs":max_abs,"file":file,"bytes":bytes.len(),"sha256":sha(&bytes),"reference":row}),
    )
}
fn typed(
    value: &GdnCoefficientPreparationOutput,
    rows: &Value,
    archive: &Archive,
    out: &Path,
    step: usize,
    write: bool,
) -> Proof<Value> {
    let values = [
        value.alpha_bf16()?,
        value.beta_input_bf16()?,
        value.biased_alpha_f32()?,
        value.softplus_f32()?,
        value.beta_bf16()?,
    ];
    let mut checks = Vec::new();
    for (name, actual) in STAGES.into_iter().zip(values) {
        ensure(actual.len() == 48, "full actual H48 private array")?;
        checks.push(compare(
            actual,
            &rows[name],
            archive,
            out,
            &format!("step{step}.typed.{name}"),
            write,
        )?);
    }
    let different = checks
        .iter()
        .map(|v| v["different_words"].as_u64().unwrap_or(u64::MAX))
        .sum::<u64>();
    Ok(
        json!({"step":step,"output_words":144,"retained_input_words":96,
        "different_words":different,"checks":checks}),
    )
}

fn source_parameters(
    ops: &OriginalWindowOperators,
    rows: &Value,
    archive: &Archive,
) -> Proof<Value> {
    let mut proofs = Vec::new();
    for (suffix, label, offset) in [
        ("A_log", "a_log_bf16", 55640u64),
        ("dt_bias", "dt_bias_bf16", 137656u64),
    ] {
        // These are provenance diagnostics only. A_log never enters preparation.
        let parameter = ops.source().parameter(&format!(
            "model.language_model.layers.0.linear_attn.{suffix}"
        ))?;
        ensure(
            parameter.tensor().kind == QuantKind::Bf16
                && parameter.tensor().shape == [48]
                && parameter.tensor().byte_len == 96,
            "genuine raw parameter codec/extent BEFORE read",
        )?;
        let window = ops.source().read_parameter(&parameter, 0..48)?;
        let original = archive.raw(&rows[label])?;
        ensure(
            window.is_complete()
                && window.source_offset() == offset
                && window.shard_basename() == "model-00001-of-00018.safetensors"
                && window.as_slice() == original,
            "actual source parameter exact original public bytes",
        )?;
        proofs.push(json!({"role":suffix,"bytes":96,"offset":offset,
            "shard":window.shard_basename(),"shard_sha256":window.shard_sha256(),
            "bundle_sha256":window.verified_bundle_sha256(),"raw_sha256":sha(window.as_slice()),
            "consumed_by_preparation":suffix=="dt_bias"}));
    }
    Ok(json!(proofs)) // All local parameter/window owners retire before return.
}

fn run() -> Proof<Value> {
    ensure(!cfg!(debug_assertions), "release-only operator replay")?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure(
        args.len() == 6 && args[5] == "27b",
        "original27B two-teacher H48 replay only",
    )?;
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "h48-coefficient-preparation-archive-replay")?;
    arbiter.register(key.clone(), HOST)?;
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("27b", "h48-coefficient-private-replay")?,
    };
    // Control ticket precedes metadata/archive extraction and the source factory.
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: CONTROL,
        }],
    )?;
    let input_bytes = read(Path::new(&args[0]), MIB as usize)?;
    let reference_bytes = read(Path::new(&args[2]), MIB as usize)?;
    ensure(
        sha(&input_bytes) == args[1] && sha(&reference_bytes) == args[3],
        "exact metadata arguments",
    )?;
    let inputs: Value = serde_json::from_slice(&input_bytes)?;
    let reference: Value = serde_json::from_slice(&reference_bytes)?;
    let bindings: Value = serde_json::from_str(include_str!("source-bindings.json"))?;
    for document in [&inputs, &reference] {
        ensure(
            document["model"] == "27b"
                && document["heads"] == 48
                && document["layer"] == 0
                && document["profile"] == "original27B-raw-source-bf16-v1"
                && document["teacher_tokens"] == json!([11, 353])
                && document["source_bindings_sha256"]
                    == sha(include_bytes!("source-bindings.json")),
            "exact original source/profile/two authentic calls",
        )?;
    }
    ensure(
        inputs["schema"] == "zerogravity.coefficient-preparation.archived-inputs.h48.v1"
            && reference["schema"]
                == "zerogravity.coefficient-preparation.archived-reference.h48.v1"
            && reference["inputs_sha256"] == args[1]
            && inputs["steps"].as_array().is_some_and(|v| v.len() == 2)
            && reference["steps"].as_array().is_some_and(|v| v.len() == 2),
        "complete metadata linkage",
    )?;
    let out = Path::new(&args[4]);
    std::fs::create_dir(out)?;
    let ready = factory::open_source(&host)?;
    let source = ready.source;
    ensure(
        source.metadata_reserved_bytes() == SOURCE_META,
        "unchanged original metadata charge",
    )?;
    let files = bindings["source"]["source_files"]
        .as_array()
        .ok_or("source pins")?;
    let source_hash = |name: &str| -> Proof<&str> {
        text(
            &files
                .iter()
                .find(|v| v["path"] == name)
                .ok_or("source pin")?["sha256"],
        )
    };
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &ready.config,
        source_hash("config.json")?,
        source_hash("model.safetensors.index.json")?,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )?;
    ensure(
        ops.binding().text_tensor_count() == 851
            && ops.binding().config().geometry().recurrent_value_heads == Some(48)
            && ops.binding().config().geometry().layers.get(3)
                == Some(&zerogravity_model_qwen35::model::LayerKind::FullAttention),
        "genuine851-role H48 source and actual full-attention layer3",
    )?;
    let source_binding = json!({"identity":format!("{:?}",source.source_identity()),
        "verified_bundle_sha256":source.verified_bundle_sha256(),
        "verified_index_sha256":source.verified_shard_index_sha256(),"complete_text_roles":851});
    let before = snapshot(&host)?;
    ensure(
        before.reserved == SOURCE_META + 8 * MIB + CONTROL && before.reservations == 4,
        "steady original source/binding/control reservations",
    )?;
    let archives = [
        Archive::new(&bindings["step0_archive"])?,
        Archive::new(&bindings["step1_archive"])?,
    ];
    let mut results = Vec::with_capacity(2);
    let mut comparisons = Vec::new();
    let mut negatives = Vec::new();
    let mut original_parameters = Vec::new();
    for (step, archive) in archives.iter().enumerate() {
        let rows = &bindings["steps"][step]["tensors"];
        for name in ["alpha_bf16", "beta_input_bf16"] {
            ensure(
                inputs["steps"][step]["operands"][name] == rows[name],
                "bound actual input metadata",
            )?;
        }
        for name in ["biased_alpha_f32", "softplus_f32", "beta_bf16"] {
            ensure(
                reference["steps"][step]["targets"][name] == rows[name],
                "bound actual target metadata",
            )?;
        }
        original_parameters.push(source_parameters(&ops, rows, archive)?);
        let a = archive.raw(&rows["alpha_bf16"])?;
        let b = archive.raw(&rows["beta_input_bf16"])?;
        let input = GdnCoefficientPreparationInput::from_bf16_bytes(
            source.source_identity(),
            PROFILE,
            &a,
            &b,
        )?;
        let prior = snapshot(&host)?;
        ensure(
            ops.execute_gdn_coefficient_preparation_slice(0, input, Some(&AtomicBool::new(true)))
                .is_err(),
            "pre-cancel refuses before import",
        )?;
        same_ledger(&host, &prior)?;
        ensure(
            ops.execute_gdn_coefficient_preparation_slice(3, input, None)
                .is_err(),
            "wrong layer refuses",
        )?;
        same_ledger(&host, &prior)?;
        let foreign = GdnCoefficientPreparationInput::from_bf16_bytes(
            zerogravity_core::accelerator::SourceIdentity::for_new_allocation()?,
            PROFILE,
            &a,
            &b,
        )?;
        ensure(
            ops.execute_gdn_coefficient_preparation_slice(0, foreign, None)
                .is_err(),
            "foreign input issuer refuses",
        )?;
        same_ledger(&host, &prior)?;
        let held = snapshot(&host)?
            .remaining
            .checked_sub(NEEDED - 1)
            .ok_or("pressure fit")?;
        let mut remaining = held;
        let mut pressure = Vec::new();
        while remaining > 0 {
            let n = remaining.min(32 * MIB);
            pressure.push(source.reserve_workspace(n as usize)?);
            remaining -= n;
        }
        let pressured = snapshot(&host)?;
        ensure(
            pressured.remaining == NEEDED - 1
                && ops
                    .execute_gdn_coefficient_preparation_slice(0, input, None)
                    .is_err(),
            "actual pool one byte short of both original owners",
        )?;
        same_ledger(&host, &pressured)?;
        drop(pressure);
        same_ledger(&host, &prior)?;
        let result = ops.execute_gdn_coefficient_preparation_slice(0, input, None)?;
        ensure(
            result.source_identity() == source.source_identity()
                && result.numerical_profile() == PROFILE
                && result.raw_dt_bias_parameter().kind == QuantKind::Bf16
                && result.raw_dt_bias_parameter().shape == [48]
                && result.raw_dt_bias_parameter().bytes == 96
                && result.unevaluated_raw_a_log_parameter().kind == QuantKind::Bf16
                && result.unevaluated_raw_a_log_parameter().shape == [48]
                && result.unevaluated_raw_a_log_parameter().bytes == 96,
            "actual typed original source/profile/raw parameter descriptors",
        )?;
        comparisons.push(typed(&result, rows, archive, out, step, true)?);
        results.push(result);
        ensure(
            snapshot(&host)?.reserved == before.reserved + (step as u64 + 1) * NEEDED,
            "all retained actual input/output/scratch owners charged",
        )?;
        negatives.push(json!({"step":step,"pre_cancel":true,"wrong_layer":true,
            "foreign_input_source":true,"real_one_byte_short_aggregate_refusal":true}));
    }
    let differences = comparisons
        .iter()
        .map(|v| v["different_words"].as_u64().unwrap_or(u64::MAX))
        .sum::<u64>();
    write_json(
        &out.join("comparisons.json"),
        &json!({"typed":comparisons,"different_words":differences}),
    )?;
    ensure(
        differences == 0,
        "complete coefficient stages differ; preserve all mismatch records",
    )?;
    let live = snapshot(&host)?;
    ensure(
        class(&live, AllocationClass::Output) == 4 * 64 * 1024
            && class(&live, AllocationClass::TransferStaging) == 0,
        "original payload classes and source staging retired",
    )?;
    drop(ops);
    drop(source);
    drop(ready.config);
    let retained = snapshot(&host)?;
    ensure(
        retained.reserved == SOURCE_META + CONTROL + 2 * NEEDED,
        "source retained by2 actual private results",
    )?;
    let mut retained_checks = Vec::new();
    for (step, (result, archive)) in results.iter().zip(&archives).enumerate() {
        retained_checks.push(typed(
            result,
            &bindings["steps"][step]["tensors"],
            archive,
            out,
            step,
            false,
        )?);
    }
    ensure(
        retained_checks == comparisons,
        "all480 private words persist after caller owners drop",
    )?;
    for archive in &archives {
        archive.close_proof()?;
    }
    drop(results);
    ensure(
        snapshot(&host)?.reserved == CONTROL,
        "last result drop releases original source/workspaces",
    )?;
    drop(archives);
    drop(inputs);
    drop(reference);
    drop(bindings);
    drop(input_bytes);
    drop(reference_bytes);
    drop(control);
    let final_row = snapshot(&host)?;
    ensure(
        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
        "original host arbiter final0",
    )?;
    let affinity = std::fs::read_to_string("/proc/self/status")?
        .lines()
        .find(|v| v.starts_with("Cpus_allowed_list:"))
        .ok_or("actual affinity")?
        .to_owned();
    Ok(
        json!({"status":"PASS_PRIVATE_ORIGINAL_H48_COEFFICIENT_PREPARATION_ONLY",
        "inputs_sha256":args[1],"reference_sha256":args[3],"actual_affinity":affinity,
        "output_stage_checks":6,"output_stage_words":288,"retained_input_words":192,
        "comparisons":comparisons,"retained_comparisons":retained_checks,"negative":negatives,
        "original_parameters":original_parameters,"steady_ledger":ledger(&before),"live_ledger":ledger(&live),
        "retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),"source_binding":source_binding,
        "metadata_and_payload_admitted_before_native_extraction":true,"A_log_consumed_by_preparation":false,
        "full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified":false}),
    )
}

fn main() {
    let result = run();
    let report = match &result {
        Ok(v) => v.clone(),
        Err(e) => json!({"status":"FAIL_UNQUALIFIED","error":e.to_string()}),
    };
    let written = std::env::args()
        .nth(5)
        .ok_or("output missing")
        .and_then(|p| {
            write_json(&Path::new(&p).join("report.json"), &report)
                .map_err(|_| "report write failed")
        });
    println!("{report}");
    if result.is_err() || written.is_err() {
        std::process::exit(1);
    }
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/owned_lifetime.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3702; SHA256 `c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8`.

````text
"""Exact accepted owned-native leader cleanup functions; no child on import."""
import os
import resource
import signal
import time
from contracts import require
CAP = 4 * 1024 * 1024


def unreaped(child):
    require(child.returncode is None, "leader already reaped")
    value = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if value is None:
        return None
    require(value.si_pid == child.pid, "owned child identity")
    return value.si_status if value.si_code == os.CLD_EXITED else -value.si_status


def cleanup_owned(child, record, errors, *, observe=unreaped, pause=time.sleep):
    """Independent bounded stages; never signal after the first reap attempt."""
    def bound():
        observe(child)
        require(os.getpgid(child.pid) == child.pid and os.getsid(child.pid) == child.pid,
                "owned session/group binding")
        return child.pid

    try:
        os.killpg(bound(), signal.SIGTERM)
        record["group_term_before_reap"] = True
    except ProcessLookupError:
        record["group_term_absent"] = True
    except BaseException as error:
        errors.append("owned TERM: " + repr(error))
    try:
        grace = time.monotonic() + 0.5
        while time.monotonic() < grace:
            observe(child)
            pause(0.05)
    except BaseException as error:
        errors.append("owned grace: " + repr(error))
    # TERM/grace/one KILL error cannot skip the separately bound KILL stage.
    # Both attempts occur before ANY wait, while the owned leader is unreaped.
    for attempt in range(2):
        try:
            os.killpg(bound(), signal.SIGKILL)
            record["group_kill_before_reap"] = True
            break
        except ProcessLookupError:
            record["group_kill_absent_before_reap"] = True
            break
        except BaseException as error:
            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
    # ALWAYS attempt reap even after signal/ownership failures. Popen.wait only
    # targets this child. An interrupted first wait cannot skip the final wait.
    # No signal operation occurs below this point, including after timeout.
    try:
        record["reaped_exit_code"] = child.wait(timeout=5)
        record["leader_reaped"] = True
    except BaseException as error:
        errors.append("owned initial reap: " + repr(error))
    finally:
        if child.returncode is None:
            try:
                record["reaped_exit_code"] = child.wait(timeout=5)
                record["leader_reaped"] = True
            except BaseException as error:
                errors.append("owned final reap: " + repr(error))
        else:
            record["reaped_exit_code"] = child.returncode
            record["leader_reaped"] = True
    if record.get("leader_reaped"):
        try:
            deadline = time.monotonic() + 5
            while True:
                try:
                    os.killpg(child.pid, 0)  # Read-only absence check, NEVER kill after reap.
                except ProcessLookupError:
                    record["group_absent"] = True
                    break
                require(time.monotonic() < deadline, "owned process group remains")
                pause(0.05)
        except BaseException as error:
            errors.append("owned group absence: " + repr(error))


def launch_limits():
    # Python is single-threaded at Popen; no foreign runtime is imported.
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (3601, 3601))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/owned_native.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5707; SHA256 `2bf4b2ea543e7bf3f888f2b9cd86c3db18d0c5884934cbb67cc9d1d2378942f1`.

````text
"""ROOT ONLY native CPU helper orchestration. No Torch/model/container imports."""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import time

from contracts import HERE, frozen, pin, read_json, require, sha, write
from owned_lifetime import cleanup_owned, launch_limits, unreaped

CAP = 4 * 1024 * 1024


def source_files(binding):
    """ROOT performs complete source scans; never provider/opaque-memory math."""
    root = Path(binding["source_directory"])
    rows = []
    for item in binding["source_files"]:
        path = root / item["path"]
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        with os.fdopen(os.open(path, flags), "rb") as opened:
            before = os.fstat(opened.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == item["bytes"], "original source regular FD/full length")
            import hashlib
            digest = hashlib.file_digest(opened, "sha256").hexdigest()
            after = os.fstat(opened.fileno())
        identity = lambda value: [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
        require(identity(before) == identity(after) and digest == item["sha256"], "complete original source FD digest/identity")
        rows.append({"path": str(path), "bytes": before.st_size, "sha256": digest,
                     "fd_identity": identity(before)})
    return rows


def execute(binary, binary_fd, inputs, inputs_sha, reference, reference_sha, output, model, cpus, freeze, source_before, cancel=None):
    if cancel is not None:
        cancel.check()
    argv = [str(binary),str(inputs),inputs_sha,str(reference),reference_sha,str(output),model]
    environment = {key: value for key, value in os.environ.items()
                   if key in ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "XDG_CACHE_HOME")}
    environment.update(RAYON_NUM_THREADS="4", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       CUDA_VISIBLE_DEVICES="-1", NVIDIA_VISIBLE_DEVICES="void",
                       ATEN_CPU_CAPABILITY="avx2")
    record = {"status": "UNQUALIFIED", "argv": argv, "freeze_manifest_sha256": freeze,
              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "private archived original H48 coefficient preparation only",
              "address_space_cap_bytes": 512 * 1024**2, "wall_cap_seconds": 900,
              "log_cap_bytes": CAP, "source_before": source_before,
              "no_model_or_session_or_GPU_execution": True}
    child = selector = pipe = log = None
    errors, primary, eof, retained = [], None, False, 0
    started = time.monotonic()
    try:
        log = (output.parent / (model + "-lifecycle") / "execution.log").open("xb")
        child = subprocess.Popen(argv, executable=f"/proc/self/fd/{binary_fd}", pass_fds=(binary_fd,),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True, preexec_fn=launch_limits)
        # Every post-spawn operation is covered by cleanup below.
        record["owned_cli_pid"] = child.pid
        pipe = child.stdout
        os.set_blocking(pipe.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(pipe, selectors.EVENT_READ)

        def drain(timeout):
            nonlocal retained, eof
            for key, _ in selector.select(timeout):
                try:
                    data = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                if not data:
                    selector.unregister(key.fileobj)
                    eof = True
                else:
                    require(retained + len(data) <= CAP, "bounded child combined log")
                    log.write(data)
                    retained += len(data)
        while True:
            if cancel is not None:
                cancel.check()
            drain(0.05)
            status = unreaped(child)
            if status is not None:
                record["exit_code"] = status
                break
            require(time.monotonic() - started < 900, "finite helper deadline")
    except BaseException as error:
        primary = repr(error)
    finally:
        if child is not None:
            cleanup_owned(child, record, errors)
        if selector is not None:
            try:
                deadline = time.monotonic() + 2
                while not eof and time.monotonic() < deadline:
                    drain(0.05)
                require(eof, "owned pipe EOF deadline")
            except BaseException as error:
                errors.append("final drain: " + repr(error))
        for name, owned in (("selector", selector), ("pipe", pipe), ("log", log)):
            if owned is not None:
                try:
                    owned.close()
                    record[name + "_closed"] = True
                except BaseException as error:
                    errors.append(name + " close: " + repr(error))
        record.update(primary_error=primary, cleanup_errors=errors, pipe_eof=eof, log_bytes=retained,
                      elapsed_seconds_not_benchmark=time.monotonic() - started)
    success = primary is None and not errors and record.get("exit_code") == record.get("reaped_exit_code") == 0 \
        and record.get("leader_reaped") and record.get("group_absent") and eof
    record["status"] = "CHILD_EXECUTED_REQUIRES_VALIDATION" if success else "FAIL_UNQUALIFIED"
    write(output.parent / (model + "-lifecycle") / "invocation.json", record)
    require(success, "owned native helper failed; preserve fresh attempt")
    return record


````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/prepare_reference.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 9701; SHA256 `5bb22f05a84d1d91e3c33da6165fa7b6f8abb13482bf2d1673f256b52f78479f`.

````text
"""ROOT-only metadata transport. Does not read/hash any archive/source payload."""
import argparse
import json
from pathlib import Path
from contracts import HERE, INPUTS, OUTPUTS, PROFILE, frozen, pin, read_json, require, sha, write


def validate_step(step, bound, readsets):
    require(step['step'] == bound['step'] and step['teacher_token'] == bound['teacher_token'] and
            step['plain_observed_all_outputs_and_state_exact'] is True,
            'genuine phase/teacher and unchanged plain/observed math')
    dispatch = step['dispatch'][:8]
    require(dispatch == bound['actual_coefficient_dispatch'] and
            [d['operation'] for d in dispatch] == ['aten.sigmoid.default','aten.to.dtype',
             'aten.exp.default','aten.neg.default','aten.to.dtype','aten.add.Tensor',
             'aten.softplus.default','aten.mul.Tensor'] and
            [d['source_line'] for d in dispatch] == [617,619,619,619,619,619,619,619] and
            all(d['scope'] == 'gdn' and d['index'] == i for i,d in enumerate(dispatch)),
            'actual original public source617/619 call; exp remains outside replay')
    for label,row in bound['tensors'].items():
        actual = dict(name=row['name'], **step['tensors'][row['name']])
        actual['original_torch_dtype'] = actual['dtype']
        actual['dtype'] = {'torch.bfloat16':'BF16','torch.float32':'F32'}[actual['dtype']]
        require(actual == row and row['device'] == 'cpu' and row['contiguous'] is True and
                type(row['numel']) is int and row['numel'] == 48 and
                row['shape'] == ([48] if label in ('a_log_bf16','dt_bias_bf16') else [1,1,48]) and
                row['stride'] == ([1] if label in ('a_log_bf16','dt_bias_bf16') else [48,48,1]) and
                row['dtype'] == ('F32' if label in ('biased_alpha_f32','softplus_f32') else 'BF16') and
                row['bytes'] == (192 if row['dtype'] == 'F32' else 96),
                'all genuine original coefficient descriptor fields: '+label)
    tensors,rows = step['tensors'],bound['tensors']
    for source,target in [('gdn.projections.a','dispatch.004.args.0'),
                          ('gdn.projections.b','dispatch.000.args.0'),
                          ('dispatch.000.result','gdn.coefficients.beta'),
                          ('dispatch.004.result','dispatch.005.args.0'),
                          ('dispatch.005.result','dispatch.006.args.0'),
                          ('dispatch.006.result','dispatch.007.args.1')]:
        require(tensors[source]['raw_sha256'] == tensors[target]['raw_sha256'],
                'actual frame/dispatch producer boundary raw identity')
    for label,name in [('a_log_bf16','A_log'),('dt_bias_bf16','dt_bias')]:
        require(rows[label]['raw_sha256'] == readsets['model.language_model.layers.0.linear_attn.'+name],
                'actual original source parameter raw readset, not a converted source')


def authenticate_metadata():
    binding = read_json(HERE/'source-bindings.json')
    for key in ('capture_manifest','capture_validation','capture_invocation','capture_preparation',
                'source_plan','factory_report','step0_metadata','step1_metadata'):
        pin(binding[key]['path'],binding[key])
    capture = read_json(binding['capture_manifest']['path'])
    validation = read_json(binding['capture_validation']['path'])
    invocation = read_json(binding['capture_invocation']['path'])
    require(capture['status'] == 'CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION' and
            capture['source_repo'] == binding['source']['repo'] == 'Qwen/Qwen3.8-27B' and
            capture['source_revision'] == binding['source']['revision'] == '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0' and
            capture['snapshot'] == binding['source']['snapshot'] == '98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e' and
            capture['parameter_profile'] == PROFILE and capture['tokens'] == [11,353] and
            capture['all_source_parameters_BF16'] is True and
            capture['full_model_StateRoot_native_exp_GPU_eligible'] is False,
            'complete original27B public raw-source profile')
    require(validation['status'] == 'PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY' and
            validation['manifest_sha256'] == binding['capture_manifest']['sha256'] and
            validation['source_plan_sha256'] == capture['source_plan_sha256'] == binding['source_plan']['sha256'] and
            validation['heads'] == 48 and validation['teachers'] == [11,353] and
            validation['full_model_112_StateRoots_native_exp_allocation_GPU_or_performance_qualified'] is False,
            'independent original reference accepted first')
    require(invocation['status'] == 'EXECUTED_GENUINE_ORIGINAL27B_LAYER0_REQUIRES_INDEPENDENT_VALIDATION' and
            invocation['exit_code'] == 0 and invocation['cleanup_errors'] == [] and
            invocation['source_FD_cleanup_errors'] == [] and invocation['source_FDs_closed'] is True and
            invocation['owned_container_absent_after_reap'] is True and
            invocation['source_before'] == invocation['source_after'] and
            sha(binding['capture_invocation']['path']) == validation['invocation_sha256'],
            'prior real source/runtime owner lifecycle closed')
    require({k:capture['runtime'][k] for k in binding['runtime']} == binding['runtime'] and
            binding['runtime'] == dict(torch='2.13.0+cu129',
             torch_git='cf30153c4c131c8164ee7798e5022d810682e2cb',transformers='5.17.0',
             cpu_capability='AVX2',threads=4,interop_threads=1,cuda_available=False,
             modeling_sha256='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'),
            'pinned original public CPU operator profile')
    plan = read_json(binding['source_plan']['path'])
    for name,row in binding['original_parameters'].items():
        require(row == plan['tensors']['model.language_model.layers.0.linear_attn.'+name] and
                row['dtype'] == 'BF16' and row['shape'] == [48] and row['bytes'] == 96,
                'exact raw original source parameter extent/codec')
    steps = capture['steps']
    require(len(steps) == len(binding['steps']) == 2 and [s['step'] for s in steps] == [0,1] and
            [s['teacher_token'] for s in steps] == [11,353], 'exact two authentic calls')
    for i,(step,bound) in enumerate(zip(steps,binding['steps'])):
        require(read_json(binding[f'step{i}_metadata']['path']) ==
                dict(step,plain_observed_all_outputs_and_state_exact=False) and
                step['archive_bytes'] == binding[f'step{i}_archive']['bytes'] and
                step['archive_sha256'] == binding[f'step{i}_archive']['sha256'],
                'original prevalidation metadata and full archive identity unchanged')
        validate_step(step,bound,capture['source_readsets'])
    require(steps[0]['tensors']['chunk.return.actual_return_1']['raw_sha256'] ==
            steps[1]['tensors']['recurrent.input.initial_state']['raw_sha256'] and
            steps[1]['actual_calls'] == {'Qwen3_5GatedDeltaNet.forward':1,
             'Qwen3_5RMSNormGated.forward':1,'causal_conv1d_update':1,'l2norm':2,
             'torch_recurrent_gated_delta_rule':1},
            'teacher353 is genuine cached continuation, not a fresh substitute')
    return binding


def documents():
    binding = authenticate_metadata()
    common = dict(model='27b',heads=48,layer=0,profile=PROFILE,teacher_tokens=[11,353],
                  source_bindings_sha256=sha(HERE/'source-bindings.json'),
                  no_payload_extracted_and_no_new_operator_run=True)
    inputs = dict(common,schema='zerogravity.coefficient-preparation.archived-inputs.h48.v1',steps=[
        dict(step=s['step'],teacher_token=s['teacher_token'],operands={n:s['tensors'][n] for n in INPUTS})
        for s in binding['steps']])
    reference = dict(common,schema='zerogravity.coefficient-preparation.archived-reference.h48.v1',steps=[
        dict(step=s['step'],teacher_token=s['teacher_token'],targets={n:s['tensors'][n] for n in OUTPUTS})
        for s in binding['steps']],expected_output_checks=6,expected_output_words=288,
        expected_retained_input_words=192,unevaluated_A_log=True)
    return inputs,reference


def validate(directory):
    inputs,reference = documents()
    require(read_json(directory/'inputs.json') == inputs,'complete metadata-only operands')
    reference['inputs_sha256'] = sha(directory/'inputs.json')
    require(read_json(directory/'reference.json') == reference and
            {p.name for p in directory.iterdir()} == {'inputs.json','reference.json'},
            'complete metadata-only targets/link/tree')
    return dict(status='PASS_METADATA_BINDING_ONLY_RAW_BYTES_NOT_VALIDATED',
                inputs_sha256=sha(directory/'inputs.json'),reference_sha256=sha(directory/'reference.json'),
                source_bindings_sha256=sha(HERE/'source-bindings.json'),
                full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-output',type=Path,required=True)
    args = parser.parse_args()
    frozen()
    inputs,reference = documents()
    require(args.new_output.resolve().parent == HERE,'fresh own bounded metadata directory')
    args.new_output.mkdir(mode=0o700)
    write(args.new_output/'inputs.json',inputs)
    reference['inputs_sha256'] = sha(args.new_output/'inputs.json')
    write(args.new_output/'reference.json',reference)
    print(json.dumps(validate(args.new_output),sort_keys=True))


if __name__ == '__main__':
    main()

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/run_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3915; SHA256 `8e68bc2ab3cfe05b65464004b8cc6c0edb798b301ac34ffa8fe7c2127f19e706`.

````text
"""ROOT ONLY fresh owned native child; no provider/model/operator on import."""
import argparse
import hashlib
import os
from pathlib import Path
from contracts import HERE, frozen, implementation, read_json, require, sha, write
from prepare_reference import validate as validate_archived
from owned_native import execute, source_files
from sigterm_cancellation import SigtermCancellation

def main(cancel):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--binary-sha256', required=True)
    parser.add_argument('--reference-directory', type=Path, required=True)
    parser.add_argument('--implementation', type=Path, required=True)
    parser.add_argument('--new-output', type=Path, required=True)
    args = parser.parse_args()
    cancel.check()
    freeze = frozen()
    source_attestation = implementation(args.implementation)
    archived = validate_archived(args.reference_directory)
    require(args.reference_directory.resolve().parent == HERE and
            args.new_output.resolve().parent == HERE, 'own bounded reference/attempt only')
    require(not args.binary.is_symlink() and args.binary.is_file() and
            sha(args.binary) == args.binary_sha256, 'ROOT reviewed actual native binary')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four available CPUs')
    os.sched_setaffinity(0, cpus)
    args.new_output.mkdir(mode=0o700)
    (args.new_output/'27b-lifecycle').mkdir(mode=0o700)
    inputs, reference = args.reference_directory/'inputs.json', args.reference_directory/'reference.json'
    binding = read_json(HERE/'source-bindings.json')['source']
    fd = os.open(args.binary.resolve(), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        with os.fdopen(os.dup(fd), 'rb') as opened:
            require(hashlib.file_digest(opened, 'sha256').hexdigest() == args.binary_sha256, 'actual exec-FD SHA')
        before = source_files(binding)
        cancel.check()
        output = args.new_output/'27b'
        execute(args.binary.resolve(), fd, inputs.resolve(), sha(inputs), reference.resolve(), sha(reference),
                output.resolve(), '27b', cpus, freeze, before, cancel=cancel)
        cancel.check()
        after = source_files(binding)
        require(before == after, 'all full original source files unchanged')
        require(implementation(args.implementation) == source_attestation and frozen() == freeze and
                sha(args.binary) == args.binary_sha256 and validate_archived(args.reference_directory) == archived,
                'all actual build/source/public-reference bindings unchanged')
        cancel.check()
        write(output/'source-after.json', {'source_before':before,'source_after':after,'all_equal':True})
        report = read_json(output/'report.json')
        require(report['status'] == 'PASS_PRIVATE_ORIGINAL_H48_COEFFICIENT_PREPARATION_ONLY' and
                report['output_stage_words'] == 288 and report['retained_input_words'] == 192,
                'actual coefficient stages and retained inputs require independent raw validation')
        write(args.new_output/'completed.json', {
            'status':'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION',
            'freeze_manifest_sha256':freeze,'binary_sha256':args.binary_sha256,
            'ROOT_build_source_attestation_sha256':source_attestation,
            'archived_reference_validation':archived,
            'report_sha256':sha(output/'report.json'),
            'source_after_sha256':sha(output/'source-after.json'),
            'invocation_sha256':sha(args.new_output/'27b-lifecycle/invocation.json'),
            'full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified':False})
    finally:
        os.close(fd)

if __name__ == '__main__':
    with SigtermCancellation() as cancellation:
        main(cancellation)

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/sigterm_cancellation.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1428; SHA256 `00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128`.

````text
"""SIGTERM/SIGINT checkpoints around the original bounded log drain, not new ownership."""
import signal


class SigtermCancellation:
    def __init__(self):
        self.received = False
        self.previous = {}

    def _record(self, signum, frame):
        # Never asynchronously raise across Popen assignment or retirement.
        self.received = True

    def __enter__(self):
        for signum in (signal.SIGTERM, signal.SIGINT):
            self.previous[signum] = signal.signal(signum, self._record)
        return self

    def __exit__(self, kind, value, traceback):
        for signum, previous in self.previous.items():
            signal.signal(signum, previous)
        return False

    def exception(self):
        return InterruptedError("interrupted: retire original owned serving regression before failure")

    def check(self):
        if self.received:
            raise self.exception()

    def observe(self, log, process, seconds):
        original_drain = log.drain

        def checked_drain(timeout):
            self.check()
            original_drain(timeout)
            self.check()

        self.check()
        log.drain = checked_drain
        try:
            result = log.observe(process, seconds)
            self.check()
            return result
        finally:
            # Original cleanup/finish drain proceeds even after repeated SIGTERM.
            log.drain = original_drain

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/source-bindings.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 25831; SHA256 `670f1b5812c58de4355a04879b6decac8aee3964d4440f0b8699e4c6d21f3be9`.

````text
{
  "archive_payloads_read_or_hashed_by_preparer": false,
  "capture_invocation": {
    "bytes": 22448,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/invocation.json",
    "sha256": "61b1787ca857a1464fd61c8132475dbe7bac2b4282b3b7ff4db7bc15d427512f"
  },
  "capture_manifest": {
    "bytes": 233492,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/manifest.json",
    "sha256": "62e82637c7d24a58b733f2358b3061efd767725901314df4ff7378707d946ab8"
  },
  "capture_preparation": {
    "bytes": 2322,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/prepared-manifest-v3.json",
    "sha256": "d3614b48bb41033a4052d765d3f7633d456812670783049db75ceadf33d8817d"
  },
  "capture_validation": {
    "bytes": 596,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/validation.json",
    "sha256": "951f74e33de53c4b6b5254bd980df1f6c05f6b9869a46bf617c7fa93184d344d"
  },
  "expected_totals": {
    "all_typed_input_and_target_words": 480,
    "input_a_b_words": 192,
    "output_stage_checks": 6,
    "output_stage_words": 288,
    "unevaluated_a_log_and_consumed_dt_unique_source_words": 96
  },
  "factory_report": {
    "bytes": 7058,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source-pin.json",
    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
  },
  "full_model_StateRoots_exp_GDN_GPU_or_performance_qualified": false,
  "original_parameters": {
    "A_log": {
      "absolute_offset": 55640,
      "bytes": 96,
      "dtype": "BF16",
      "mapping_bytes": 2488,
      "name": "model.language_model.layers.0.linear_attn.A_log",
      "page_aligned_offset": 53248,
      "shape": [
        48
      ],
      "shard": "model-00001-of-00018.safetensors",
      "view_offset": 2392
    },
    "dt_bias": {
      "absolute_offset": 137656,
      "bytes": 96,
      "dtype": "BF16",
      "mapping_bytes": 2584,
      "name": "model.language_model.layers.0.linear_attn.dt_bias",
      "page_aligned_offset": 135168,
      "shape": [
        48
      ],
      "shard": "model-00001-of-00018.safetensors",
      "view_offset": 2488
    }
  },
  "plain_archive": {
    "bytes": 6518376,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/plain.safetensors",
    "sha256": "9b20753b733f5f870abf4691eaecf9ce42f5edec0ecec8ae1260684273d5bab8"
  },
  "producer_scope": "External actual original two-teacher H48 a/b; SAME private beta/F32 biased-alpha/softplus only. A_log descriptor and raw source provenance are UNEVALUATED, not exp/g/decay.",
  "profile": "original27B-raw-source-bf16-v1",
  "runtime": {
    "cpu_capability": "AVX2",
    "cuda_available": false,
    "interop_threads": 1,
    "modeling_sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
    "threads": 4,
    "torch": "2.13.0+cu129",
    "torch_git": "cf30153c4c131c8164ee7798e5022d810682e2cb",
    "transformers": "5.17.0"
  },
  "schema": "zerogravity.original27b.h48-coefficient-preparation.archived-bindings.v1",
  "source": {
    "heads": 48,
    "repo": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
    "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
    "source_files": [
      {
        "bytes": 11544,
        "path": "LICENSE",
        "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"
      },
      {
        "bytes": 65012,
        "path": "README.md",
        "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641"
      },
      {
        "bytes": 8952,
        "path": "chat_template.jinja",
        "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
      },
      {
        "bytes": 4312,
        "path": "config.json",
        "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
      },
      {
        "bytes": 238,
        "path": "crc32.txt",
        "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83"
      },
      {
        "bytes": 202,
        "path": "generation_config.json",
        "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
      },
      {
        "bytes": 3353259,
        "path": "merges.txt",
        "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
      },
      {
        "bytes": 3966730552,
        "path": "model-00001-of-00018.safetensors",
        "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c"
      },
      {
        "bytes": 3043080328,
        "path": "model-00002-of-00018.safetensors",
        "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb"
      },
      {
        "bytes": 2542796952,
        "path": "model-00003-of-00018.safetensors",
        "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968"
      },
      {
        "bytes": 3988973152,
        "path": "model-00004-of-00018.safetensors",
        "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a"
      },
      {
        "bytes": 2099339864,
        "path": "model-00005-of-00018.safetensors",
        "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701"
      },
      {
        "bytes": 3979553696,
        "path": "model-00006-of-00018.safetensors",
        "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0"
      },
      {
        "bytes": 2108759344,
        "path": "model-00007-of-00018.safetensors",
        "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
      },
      {
        "bytes": 3979553696,
        "path": "model-00008-of-00018.safetensors",
        "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79"
      },
      {
        "bytes": 2108759344,
        "path": "model-00009-of-00018.safetensors",
        "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8"
      },
      {
        "bytes": 3979553696,
        "path": "model-00010-of-00018.safetensors",
        "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9"
      },
      {
        "bytes": 2108759344,
        "path": "model-00011-of-00018.safetensors",
        "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8"
      },
      {
        "bytes": 3979553696,
        "path": "model-00012-of-00018.safetensors",
        "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55"
      },
      {
        "bytes": 2108759344,
        "path": "model-00013-of-00018.safetensors",
        "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a"
      },
      {
        "bytes": 3979553696,
        "path": "model-00014-of-00018.safetensors",
        "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b"
      },
      {
        "bytes": 2108759344,
        "path": "model-00015-of-00018.safetensors",
        "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22"
      },
      {
        "bytes": 3979564040,
        "path": "model-00016-of-00018.safetensors",
        "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe"
      },
      {
        "bytes": 2108759344,
        "path": "model-00017-of-00018.safetensors",
        "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8"
      },
      {
        "bytes": 3392197344,
        "path": "model-00018-of-00018.safetensors",
        "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"
      },
      {
        "bytes": 112216,
        "path": "model.safetensors.index.json",
        "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
      },
      {
        "bytes": 390,
        "path": "preprocessor_config.json",
        "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
      },
      {
        "bytes": 12809320,
        "path": "tokenizer.json",
        "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
      },
      {
        "bytes": 17928,
        "path": "tokenizer_config.json",
        "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
      },
      {
        "bytes": 385,
        "path": "video_preprocessor_config.json",
        "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
      },
      {
        "bytes": 6722759,
        "path": "vocab.json",
        "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
      }
    ]
  },
  "source_plan": {
    "bytes": 15423,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json",
    "sha256": "658280444d4e3d1b6a3a8f2ce4b97e4be7cbff2c8a6ea7c66b3c733c78f31638"
  },
  "step0_archive": {
    "bytes": 46720712,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/actual-gdn.safetensors",
    "sha256": "360ec87dbaaa0647ec72f4959d869e7d04fa1e5eed1fe7ab5b91b22bdc33b29a"
  },
  "step0_metadata": {
    "bytes": 84104,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/prevalidation.json",
    "sha256": "17bd8f5430931cb33a22af32ac056f7357b02df32c9224d1c67a6361be20f46b"
  },
  "step1_archive": {
    "bytes": 27153160,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/actual-gdn.safetensors",
    "sha256": "907044c0889af975d573cb0bbacc0e8f3ac1212aa60822c36bc770d9868de038"
  },
  "step1_metadata": {
    "bytes": 74924,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/prevalidation.json",
    "sha256": "7ceceff0e776b7c4f1c97074de7b065264b69e374397f97bc9f1535f4517dc60"
  },
  "steps": [
    {
      "actual_coefficient_dispatch": [
        {
          "arguments": [
            {
              "tensor": "dispatch.000.args.0"
            }
          ],
          "index": 0,
          "keyword_arguments": {},
          "operation": "aten.sigmoid.default",
          "result": {
            "tensor": "dispatch.000.result"
          },
          "schema": "aten::sigmoid(Tensor self) -> Tensor",
          "scope": "gdn",
          "source_line": 617
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.001.args.0"
            },
            {
              "torch_value": "torch.float32"
            }
          ],
          "index": 1,
          "keyword_arguments": {},
          "operation": "aten.to.dtype",
          "result": {
            "tensor": "dispatch.001.result"
          },
          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.002.args.0"
            }
          ],
          "index": 2,
          "keyword_arguments": {},
          "operation": "aten.exp.default",
          "result": {
            "tensor": "dispatch.002.result"
          },
          "schema": "aten::exp(Tensor self) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.003.args.0"
            }
          ],
          "index": 3,
          "keyword_arguments": {},
          "operation": "aten.neg.default",
          "result": {
            "tensor": "dispatch.003.result"
          },
          "schema": "aten::neg(Tensor self) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.004.args.0"
            },
            {
              "torch_value": "torch.float32"
            }
          ],
          "index": 4,
          "keyword_arguments": {},
          "operation": "aten.to.dtype",
          "result": {
            "tensor": "dispatch.004.result"
          },
          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.005.args.0"
            },
            {
              "tensor": "dispatch.005.args.1"
            }
          ],
          "index": 5,
          "keyword_arguments": {},
          "operation": "aten.add.Tensor",
          "result": {
            "tensor": "dispatch.005.result"
          },
          "schema": "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.006.args.0"
            }
          ],
          "index": 6,
          "keyword_arguments": {},
          "operation": "aten.softplus.default",
          "result": {
            "tensor": "dispatch.006.result"
          },
          "schema": "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.007.args.0"
            },
            {
              "tensor": "dispatch.007.args.1"
            }
          ],
          "index": 7,
          "keyword_arguments": {},
          "operation": "aten.mul.Tensor",
          "result": {
            "tensor": "dispatch.007.result"
          },
          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        }
      ],
      "step": 0,
      "teacher_token": 11,
      "tensors": {
        "a_log_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "dispatch.001.args.0",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "429d53cd15ec0ea1a2d638100c3bbdf8d53505e69110fb17ee09d055c1fdbd34",
          "shape": [
            48
          ],
          "stride": [
            1
          ]
        },
        "alpha_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "gdn.projections.a",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "212f5c22e6d3868a54c96c4c7c2da3a0a481b241e14e53f42b0273d0b6215a62",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "beta_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "dispatch.000.result",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "89be83e7c814bcbb23e593a40f527cb962e84322761ff4ba7803aed20df0ca03",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "beta_input_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "gdn.projections.b",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "28e6feeabc48df35af7e13168d1805129036299567c3b22286647241be154d51",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "biased_alpha_f32": {
          "bytes": 192,
          "contiguous": true,
          "device": "cpu",
          "dtype": "F32",
          "name": "dispatch.005.result",
          "numel": 48,
          "original_torch_dtype": "torch.float32",
          "raw_sha256": "df3e389f0e938c6f70e12d6e472e6cb90151641f255b7abfc0502c7c820999eb",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "dt_bias_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "dispatch.005.args.1",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "969ce6b2149d20f0c365373fd89613167e373905c15f0e1cf5af62adf52228de",
          "shape": [
            48
          ],
          "stride": [
            1
          ]
        },
        "softplus_f32": {
          "bytes": 192,
          "contiguous": true,
          "device": "cpu",
          "dtype": "F32",
          "name": "dispatch.006.result",
          "numel": 48,
          "original_torch_dtype": "torch.float32",
          "raw_sha256": "233fe3c2ccb43df307ccf93ef4bdb96262190ed3e05b92c4e410e29b8fac58ce",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        }
      }
    },
    {
      "actual_coefficient_dispatch": [
        {
          "arguments": [
            {
              "tensor": "dispatch.000.args.0"
            }
          ],
          "index": 0,
          "keyword_arguments": {},
          "operation": "aten.sigmoid.default",
          "result": {
            "tensor": "dispatch.000.result"
          },
          "schema": "aten::sigmoid(Tensor self) -> Tensor",
          "scope": "gdn",
          "source_line": 617
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.001.args.0"
            },
            {
              "torch_value": "torch.float32"
            }
          ],
          "index": 1,
          "keyword_arguments": {},
          "operation": "aten.to.dtype",
          "result": {
            "tensor": "dispatch.001.result"
          },
          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.002.args.0"
            }
          ],
          "index": 2,
          "keyword_arguments": {},
          "operation": "aten.exp.default",
          "result": {
            "tensor": "dispatch.002.result"
          },
          "schema": "aten::exp(Tensor self) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.003.args.0"
            }
          ],
          "index": 3,
          "keyword_arguments": {},
          "operation": "aten.neg.default",
          "result": {
            "tensor": "dispatch.003.result"
          },
          "schema": "aten::neg(Tensor self) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.004.args.0"
            },
            {
              "torch_value": "torch.float32"
            }
          ],
          "index": 4,
          "keyword_arguments": {},
          "operation": "aten.to.dtype",
          "result": {
            "tensor": "dispatch.004.result"
          },
          "schema": "aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.005.args.0"
            },
            {
              "tensor": "dispatch.005.args.1"
            }
          ],
          "index": 5,
          "keyword_arguments": {},
          "operation": "aten.add.Tensor",
          "result": {
            "tensor": "dispatch.005.result"
          },
          "schema": "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.006.args.0"
            }
          ],
          "index": 6,
          "keyword_arguments": {},
          "operation": "aten.softplus.default",
          "result": {
            "tensor": "dispatch.006.result"
          },
          "schema": "aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        },
        {
          "arguments": [
            {
              "tensor": "dispatch.007.args.0"
            },
            {
              "tensor": "dispatch.007.args.1"
            }
          ],
          "index": 7,
          "keyword_arguments": {},
          "operation": "aten.mul.Tensor",
          "result": {
            "tensor": "dispatch.007.result"
          },
          "schema": "aten::mul.Tensor(Tensor self, Tensor other) -> Tensor",
          "scope": "gdn",
          "source_line": 619
        }
      ],
      "step": 1,
      "teacher_token": 353,
      "tensors": {
        "a_log_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "dispatch.001.args.0",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "429d53cd15ec0ea1a2d638100c3bbdf8d53505e69110fb17ee09d055c1fdbd34",
          "shape": [
            48
          ],
          "stride": [
            1
          ]
        },
        "alpha_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "gdn.projections.a",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "fcdc41eac47bf8754b27638ad060d7378af48c9e211d13e2a4c172817b7ce0ea",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "beta_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "dispatch.000.result",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "30f2b3c9d186d815300eb49d18b6fa15e7777fe3a4fd2ee647da46dd9f94690f",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "beta_input_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "gdn.projections.b",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "3a1b5ad825f263b8e1cfb7d6dc32912c7c216bf3881c8bce9abd0d983e3c80e2",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "biased_alpha_f32": {
          "bytes": 192,
          "contiguous": true,
          "device": "cpu",
          "dtype": "F32",
          "name": "dispatch.005.result",
          "numel": 48,
          "original_torch_dtype": "torch.float32",
          "raw_sha256": "c25c96be62fdbc821a24f5951665a26d31572402c523cfb2c20ef91752ae9421",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        },
        "dt_bias_bf16": {
          "bytes": 96,
          "contiguous": true,
          "device": "cpu",
          "dtype": "BF16",
          "name": "dispatch.005.args.1",
          "numel": 48,
          "original_torch_dtype": "torch.bfloat16",
          "raw_sha256": "969ce6b2149d20f0c365373fd89613167e373905c15f0e1cf5af62adf52228de",
          "shape": [
            48
          ],
          "stride": [
            1
          ]
        },
        "softplus_f32": {
          "bytes": 192,
          "contiguous": true,
          "device": "cpu",
          "dtype": "F32",
          "name": "dispatch.006.result",
          "numel": 48,
          "original_torch_dtype": "torch.float32",
          "raw_sha256": "ad6f601579e977768f20755934844bb627776d74362d7804c42e14eac6a4d2a7",
          "shape": [
            1,
            1,
            48
          ],
          "stride": [
            48,
            48,
            1
          ]
        }
      }
    }
  ]
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/source-pin.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7058; SHA256 `4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a`.

````text
{
  "classification": "authenticated_cache_source_only_not_model_execution",
  "snapshot": "98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e",
  "cache": "/home/pvl/.cache/abird/zerogravity/models",
  "reference_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e/files",
  "repository": "Qwen/Qwen3.8-27B",
  "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  "selected_bytes": 55586113293,
  "weight_bytes": 55563006776,
  "payload_bytes": 55562855904,
  "files": [
    {
      "name": "LICENSE",
      "bytes": 11544,
      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
      "git_blob_verified": true
    },
    {
      "name": "README.md",
      "bytes": 65012,
      "sha256": "57e4bdb258ee1a7d2635c5174ebd4e56abe392505cdb5f8bbb356b0dc4293641",
      "git_blob_verified": true
    },
    {
      "name": "chat_template.jinja",
      "bytes": 8952,
      "sha256": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
      "git_blob_verified": true
    },
    {
      "name": "config.json",
      "bytes": 4312,
      "sha256": "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
      "git_blob_verified": true
    },
    {
      "name": "crc32.txt",
      "bytes": 238,
      "sha256": "b42dd291f5f98b05807e458f5b47849969a9b3326acbf0c97d28b05826740b83",
      "git_blob_verified": true
    },
    {
      "name": "generation_config.json",
      "bytes": 202,
      "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
      "git_blob_verified": true
    },
    {
      "name": "merges.txt",
      "bytes": 3353259,
      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
      "git_blob_verified": true
    },
    {
      "name": "model-00001-of-00018.safetensors",
      "bytes": 3966730552,
      "sha256": "ba0ce20aae489ad196733da5064bcdf159a1fe84f53336648196e1ebb7751b1c",
      "git_blob_verified": false
    },
    {
      "name": "model-00002-of-00018.safetensors",
      "bytes": 3043080328,
      "sha256": "06a148c01bfbe3faa14a5f184a7ff29a706f7ae1c8b2705d2058e26d17a001fb",
      "git_blob_verified": false
    },
    {
      "name": "model-00003-of-00018.safetensors",
      "bytes": 2542796952,
      "sha256": "2e1bf62cbcd406eaa64b60d10353e1f0ef4039d0976e56f05cabe953454f9968",
      "git_blob_verified": false
    },
    {
      "name": "model-00004-of-00018.safetensors",
      "bytes": 3988973152,
      "sha256": "511e34063187882659753c4d93f3859f93c019fd438d8813071921c81d9a3f1a",
      "git_blob_verified": false
    },
    {
      "name": "model-00005-of-00018.safetensors",
      "bytes": 2099339864,
      "sha256": "635cb53446dc74f219740fc59e18b774f877b803b9722e289ca62575a6efa701",
      "git_blob_verified": false
    },
    {
      "name": "model-00006-of-00018.safetensors",
      "bytes": 3979553696,
      "sha256": "0bc5214fac607f0e6cc92eec3789d4b8559410ef9fce66621ba8158e8410dae0",
      "git_blob_verified": false
    },
    {
      "name": "model-00007-of-00018.safetensors",
      "bytes": 2108759344,
      "sha256": "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892",
      "git_blob_verified": false
    },
    {
      "name": "model-00008-of-00018.safetensors",
      "bytes": 3979553696,
      "sha256": "7192c5b66185d3592927daabee1cc19e6f6e0ce75988ee20e824b624765fda79",
      "git_blob_verified": false
    },
    {
      "name": "model-00009-of-00018.safetensors",
      "bytes": 2108759344,
      "sha256": "af3c48cc37af44f3db6ae0579baf019180d48d9c527caa0a1f03ff85813a56d8",
      "git_blob_verified": false
    },
    {
      "name": "model-00010-of-00018.safetensors",
      "bytes": 3979553696,
      "sha256": "163490a76f3bea3a40855b7efc04ce6d27afaf1a34f0bbde495b9491f76457c9",
      "git_blob_verified": false
    },
    {
      "name": "model-00011-of-00018.safetensors",
      "bytes": 2108759344,
      "sha256": "5f3ae1b948aeee39da77aec558e8236cd65fe4d7cb7686a76bb007acc563c6d8",
      "git_blob_verified": false
    },
    {
      "name": "model-00012-of-00018.safetensors",
      "bytes": 3979553696,
      "sha256": "a3de1c7114677a8f5ac5c4892c90e8238ea5c1e2038c80e757dfc87c3902ca55",
      "git_blob_verified": false
    },
    {
      "name": "model-00013-of-00018.safetensors",
      "bytes": 2108759344,
      "sha256": "06ab79a41f74c9c5cb734816feb0c7fc364104b227165ee7391231e1155aa02a",
      "git_blob_verified": false
    },
    {
      "name": "model-00014-of-00018.safetensors",
      "bytes": 3979553696,
      "sha256": "4138ed94603065ba884bbcadedb04d7718bb40117e85e6f5c6fc5b9c05b7a85b",
      "git_blob_verified": false
    },
    {
      "name": "model-00015-of-00018.safetensors",
      "bytes": 2108759344,
      "sha256": "69224e27b9de4e7dbf6fc936c6eaae08447bda3b80a6c31a871ab451173afd22",
      "git_blob_verified": false
    },
    {
      "name": "model-00016-of-00018.safetensors",
      "bytes": 3979564040,
      "sha256": "73cb9a1089fb6155cb648609478d6633be8a5c7d9ca5a05bc8925ce8a553cefe",
      "git_blob_verified": false
    },
    {
      "name": "model-00017-of-00018.safetensors",
      "bytes": 2108759344,
      "sha256": "beb51f01056142ac4984bd800507b0dd0fd18de57f8e9ef6ea41d1a3598983a8",
      "git_blob_verified": false
    },
    {
      "name": "model-00018-of-00018.safetensors",
      "bytes": 3392197344,
      "sha256": "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06",
      "git_blob_verified": false
    },
    {
      "name": "model.safetensors.index.json",
      "bytes": 112216,
      "sha256": "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
      "git_blob_verified": true
    },
    {
      "name": "preprocessor_config.json",
      "bytes": 390,
      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
      "git_blob_verified": true
    },
    {
      "name": "tokenizer.json",
      "bytes": 12809320,
      "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
      "git_blob_verified": false
    },
    {
      "name": "tokenizer_config.json",
      "bytes": 17928,
      "sha256": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
      "git_blob_verified": true
    },
    {
      "name": "video_preprocessor_config.json",
      "bytes": 385,
      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
      "git_blob_verified": true
    },
    {
      "name": "vocab.json",
      "bytes": 6722759,
      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
      "git_blob_verified": true
    }
  ],
  "tensor_count": 1199,
  "ordinary_text_tensors": 851,
  "other_tensors": 348,
  "ordinary_text_bytes": 53791996928,
  "full_file_sha256s_independent": true,
  "ordinary_source_shapes_match_config": true,
  "headers_match_preflight": true,
  "buffer_bytes": 1048576,
  "statvfs_free_after": 544125792256,
  "finished_utc": "2026-10-19T15:50:57Z"
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/source_contract.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5092; SHA256 `7508c42512c77c3a1ffabd311eaa6ff3a5b674e169480bc45504f24d82af6dd9`.

````text
"""Explicit isolated ROOT source attestation; never compiler/numerical proof."""
from pathlib import Path
import hashlib
import tomllib

ORIGINAL_RUN = Path('/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020')


def frozen(here, read_json, pin, require, sha):
    run = here.parent
    original = read_json(ORIGINAL_RUN/'freeze-manifest.json')
    require(sha(ORIGINAL_RUN/'freeze-manifest.json') ==
            '05095281de89dfc2ed136e869b5ca11458ca3664293b94f79a2fae900fc07e2e',
            'all original28 source files remain immutable')
    for row in original['files']:
        pin(ORIGINAL_RUN/row['file'],row)
        pin(ORIGINAL_RUN/'frozen-prepared'/row['file'],row)
    for row in original['parents']:
        pin(row['path'],row)
    manifest = read_json(run/'freeze-manifest.json')
    require(manifest['schema'] == 'zerogravity.h48-coefficient.prepared-source.v1' and
            manifest['runtime_or_payload_read_by_preparer'] is False,
            'explicit source-only preparation')
    names = set()
    for row in manifest['files']:
        name = Path(row['file'])
        require(not name.is_absolute() and all(p not in ('.','..') for p in name.parts) and
                name.as_posix() not in names, 'complete unique bounded source paths')
        names.add(name.as_posix())
        pin(run/name, row)
        pin(run/'frozen-prepared'/name, row)
    for row in manifest['parents']:
        pin(row['path'], row)
    return sha(run/'freeze-manifest.json')


def expected(here, read_json, pin, require):
    config = read_json(ORIGINAL_RUN/'BASELINE.json')
    pin(config['root_source_freeze']['path'], config['root_source_freeze'])
    prior = read_json(config['root_source_freeze']['path'])
    require(len(prior['files']) == 521 and prior['full_model_or_GPU_qualified'] is False,
            'exact previous isolated ROOT521 source closure')
    patch = read_json(ORIGINAL_RUN/'owned-delta.json')
    require(len(patch['files']) == 3, 'only two existing metadata/test paths and new fixture')
    rows = {r['file']:r for r in prior['files']}
    require(len(rows) == 521, 'unique original source paths')
    for row in patch['files']:
        require(row['file'] in config['owned_paths'], 'only explicitly owned source paths')
        rows[row['file']] = row
    require(len(rows) == 522, 'exact ROOT521 plus one H48 fixture source')
    root = Path(config['baseline_root'])
    require(root.is_absolute() and root.resolve() == root and
            'zerogravity-worktrees' in root.parts, 'explicit isolated ROOT tree; never MAIN')
    actual = {str(root/'Cargo.toml'), str(root/'Cargo.lock')}
    actual.update(str(p) for p in (root/'crates').rglob('*') if p.is_file() and
                  (p.suffix == '.rs' or p.name == 'Cargo.toml'))
    result = {}
    for rel, row in rows.items():
        path = root/rel
        require(path.resolve() == path, 'no build-source symlink/remap')
        pin(path, row)
        result[str(path)] = dict(path=str(path), bytes=row['bytes'], sha256=row['sha256'])
    require(actual == set(result), 'no missing or extra Rust/manifests in actual ROOT tree')
    for name in ('Cargo.toml','main.rs','factory.rs','source-pin.json','source-bindings.json'):
        path = here/name
        result[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    lock = here/'Cargo.lock'
    require(lock.is_file() and not lock.is_symlink(), 'actual standalone dependency lock required')
    previous = tomllib.loads((root/'Cargo.lock').read_text())['package']
    known = {(p['name'],p['version'],p.get('source'),p.get('checksum')) for p in previous}
    packages = tomllib.loads(lock.read_text())['package']
    for p in packages:
        require(p['name'] == 'original27b-h48-coefficient-preparation-replay' and p['version'] == '0.0.0'
                and 'source' not in p and 'checksum' not in p or
                (p['name'],p['version'],p.get('source'),p.get('checksum')) in known,
                'standalone external packages are exact parent-lock subset')
    result[str(lock)] = dict(path=str(lock), bytes=lock.stat().st_size,
                            sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
    return root, result


def implementation(here, path, read_json, pin, require, sha):
    root, rows = expected(here, read_json, pin, require)
    value = read_json(path)
    require(value['status'] == 'ROOT_ISOLATED_SOURCE_ATTESTATION_AFTER_H48_COEFFICIENT_PATCH' and
            value['MAIN_merged'] is False and value['compiler_or_numerical_proof'] is False and
            value['workspace_root'] == str(root) and
            value['preparation_sha256'] == sha(here.parent/'freeze-manifest.json') and
            value['prior_ROOT521_source_sha256'] == '036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615' and
            value['files'] == [rows[k] for k in sorted(rows)],
            'complete explicit isolated build-source attestation; not MAIN or binary proof')
    return sha(path)

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/test_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4012; SHA256 `297fb3084f2a523f4b35dcbbcbfecff9f2a14c54a48eb869722eed49bbeed0bd`.

````text
"""ROOT-only source-free metadata negatives; no archive/model/provider payload."""
import copy
import unittest
from contracts import HERE, read_json
from prepare_reference import validate_step
from sigterm_cancellation import SigtermCancellation


class Metadata(unittest.TestCase):
    def setUp(self):
        self.binding = read_json(HERE/'source-bindings.json')
        self.capture = read_json(self.binding['capture_manifest']['path'])
        self.step = copy.deepcopy(self.capture['steps'][1])
        self.bound = copy.deepcopy(self.binding['steps'][1])
        self.readsets = copy.deepcopy(self.capture['source_readsets'])

    def reject(self):
        with self.assertRaises(ValueError):
            validate_step(self.step,self.bound,self.readsets)

    def test_both_real_metadata_calls_contain_all_seven_arrays(self):
        for step,bound in zip(self.capture['steps'],self.binding['steps']):
            validate_step(step,bound,self.capture['source_readsets'])
            self.assertEqual(sum(bound['tensors'][k]['numel'] for k in
                ('biased_alpha_f32','softplus_f32','beta_bf16')),144)

    def test_wrong_original_parameter_readset_refuses(self):
        self.readsets['model.language_model.layers.0.linear_attn.A_log'] = '0'*64
        self.reject()

    def test_raw_f32_A_log_cannot_replace_original_bf16_extent(self):
        row = self.bound['tensors']['a_log_bf16']
        row['dtype'],row['original_torch_dtype'],row['bytes'] = 'F32','torch.float32',192
        self.reject()

    def test_unknown_dispatch_scope_refuses(self):
        self.step['dispatch'][6]['scope'] = 'guessed_math'
        self.bound['actual_coefficient_dispatch'][6]['scope'] = 'guessed_math'
        self.reject()

    def test_wrong_original_source_line_refuses(self):
        self.step['dispatch'][6]['source_line'] = 620
        self.bound['actual_coefficient_dispatch'][6]['source_line'] = 620
        self.reject()

    def test_partial_head_capture_refuses_even_self_consistent_descriptor(self):
        name = self.bound['tensors']['softplus_f32']['name']
        self.step['tensors'][name].update(shape=[1,1,47],numel=47,bytes=188)
        self.bound['tensors']['softplus_f32'].update(shape=[1,1,47],numel=47,bytes=188)
        self.reject()

    def test_wrong_actual_frame_beta_dependency_refuses(self):
        self.step['tensors']['gdn.coefficients.beta']['raw_sha256'] = '0'*64
        self.reject()

    def test_noncontiguous_singleton_input_refuses(self):
        name = self.bound['tensors']['alpha_bf16']['name']
        self.step['tensors'][name]['stride'] = [48,48,2]
        self.bound['tensors']['alpha_bf16']['stride'] = [48,48,2]
        self.reject()

    def test_swapped_teacher_scope_refuses(self):
        self.bound['teacher_token'] = 11
        self.reject()

    def test_missing_output_refuses(self):
        del self.step['tensors']['dispatch.006.result']
        with self.assertRaises((KeyError,ValueError)):
            validate_step(self.step,self.bound,self.readsets)

    def test_real_source_negative_layer3_is_full_attention_while_layer1_is_gdn(self):
        binding = self.binding['source']
        row = next(r for r in binding['source_files'] if r['path'] == 'config.json')
        from contracts import read
        import json
        from pathlib import Path
        config = json.loads(read(Path(binding['source_directory'])/'config.json',row=row))['text_config']
        self.assertEqual(config['full_attention_interval'],4)
        self.assertEqual(config['num_hidden_layers'],64)
        self.assertEqual(config['layer_types'][:4],
                         ['linear_attention','linear_attention','linear_attention','full_attention'])

    def test_signal_is_record_only_until_explicit_safe_checkpoint(self):
        guard = SigtermCancellation()
        guard._record(15,None)
        self.assertTrue(guard.received)
        with self.assertRaises(InterruptedError):
            guard.check()


if __name__ == '__main__':
    unittest.main()

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/validate_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 12438; SHA256 `0e4596f0d5e80ab9db2b75538184c48a4937af8147a6f23eede1c84c42b75fff`.

````text
"""ROOT independent full signed-bit/raw-archive/source/ledger/lifetime validation."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import stat
from contracts import (HERE, INPUTS, OUTPUTS, RawArchive, frozen, identity, implementation, read,
                       read_json, require, sha, widened, write)
from prepare_reference import authenticate_metadata, validate as validate_metadata

def exact_int(value, expected):
    require(type(value) is int and value == expected, 'exact nonbool integer count')

def affinity(text):
    require(type(text) is str and text.startswith('Cpus_allowed_list:'), 'actual proc affinity')
    values = []
    for item in text.split(':',1)[1].strip().split(','):
        span = item.split('-')
        require(1 <= len(span) <= 2 and all(v.isdecimal() for v in span), 'strict CPU span')
        lo, hi = int(span[0]), int(span[-1])
        require(0 <= lo <= hi < 4096, 'bounded CPU span')
        values.extend(range(lo,hi+1))
    require(len(values) == 4 and values == sorted(set(values)), 'four actual distinct CPUs')
    return values

def tree(directory, reserve=0):
    paths = list(directory.rglob('*'))
    require(len(paths) <= 64, 'bounded native diagnostic tree entries')
    total = 0
    for path in paths:
        info = path.lstat()
        require(not path.is_symlink() and (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)),
                'owned ordinary diagnostic tree')
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    require(total+reserve <= 8*1024**2, 'bounded complete diagnostic artifact tree')
    return total

def comparison(records, names, prefix, rows, archive, directory):
    require(type(records) is list and len(records) == len(names), 'complete comparison records')
    files, words = set(), 0
    for record, name in zip(records, names):
        label = prefix+name
        expected = widened(archive.raw(rows[name]), rows[name]['dtype'])
        require(record['label'] == label and record['reference'] == rows[name] and
                record['different_words'] == 0 and record['first_difference'] is None and
                record['max_abs'] == 0 and record['file'] == label+'.f32le',
                'bound actual stage/order/source/counters: '+label)
        exact_int(record['words'], len(expected)//4)
        exact_int(record['bytes'], len(expected))
        require(read(directory/record['file'],192,record) == expected,
                'independent every raw signed output bit: '+label)
        files.add(record['file'])
        words += record['words']
    return files, words

def original_parameters(binding, source):
    """Original same-FD payload proof; no exp or parameter conversion."""
    name = 'model-00001-of-00018.safetensors'
    pinned = next(r for r in binding['source']['source_files'] if r['path'] == name)
    path = Path(binding['source']['source_directory'])/name
    prior = next(r for r in source['source_after'] if r['path'] == str(path))
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW),'rb') as opened:
        before = os.fstat(opened.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == pinned['bytes'] and
                identity(before) == tuple(prior['fd_identity']) and
                hashlib.file_digest(opened,'sha256').hexdigest() == pinned['sha256'],
                'independent original whole-shard digest/FD identity')
        raw = {}
        for role,row in binding['original_parameters'].items():
            require(row['shard'] == name and row['dtype'] == 'BF16' and row['shape'] == [48] and
                    row['bytes'] == 96 and type(row['absolute_offset']) is int and
                    0 <= row['absolute_offset'] <= before.st_size-96,
                    'closed original parameter extent BEFORE read')
            raw[role] = os.pread(opened.fileno(),96,row['absolute_offset'])
            require(len(raw[role]) == 96 and identity(os.fstat(opened.fileno())) == identity(before),
                    'full same original FD parameter bytes')
        opened.seek(0)
        require(hashlib.file_digest(opened,'sha256').hexdigest() == pinned['sha256'] and
                identity(os.fstat(opened.fileno())) == identity(before),
                'original whole-shard unchanged after selected reads')
    return raw,pinned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-directory',type=Path,required=True)
    parser.add_argument('--implementation',type=Path,required=True)
    args = parser.parse_args()
    require(args.output.resolve().parent == args.reference_directory.resolve().parent == HERE,
            'owned attempt/reference only')
    freeze,build = frozen(),implementation(args.implementation)
    metadata,binding = validate_metadata(args.reference_directory),authenticate_metadata()
    done = read_json(args.output/'completed.json')
    require(done['status'] == 'EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION' and
            done['freeze_manifest_sha256'] == freeze and done['ROOT_build_source_attestation_sha256'] == build and
            done['archived_reference_validation'] == metadata and
            done['full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified'] is False,
            'explicit isolated source/build/reference scope')
    directory,lifecycle = args.output/'27b',args.output/'27b-lifecycle'
    for file,key in [('report.json','report_sha256'),('source-after.json','source_after_sha256')]:
        require(sha(directory/file) == done[key],'actual report/source receipt digest')
    require(sha(lifecycle/'invocation.json') == done['invocation_sha256'],'actual owned native invocation digest')
    report,invocation = read_json(directory/'report.json'),read_json(lifecycle/'invocation.json')
    require(report['status'] == 'PASS_PRIVATE_ORIGINAL_H48_COEFFICIENT_PREPARATION_ONLY' and
            report['inputs_sha256'] == metadata['inputs_sha256'] and report['reference_sha256'] == metadata['reference_sha256'] and
            report['A_log_consumed_by_preparation'] is False and
            report['full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified'] is False and
            report['metadata_and_payload_admitted_before_native_extraction'] is True,
            'same private coefficient preparation; no A_log/exp eligibility')
    for key,count in [('output_stage_checks',6),('output_stage_words',288),('retained_input_words',192)]:
        exact_int(report[key],count)
    require(invocation['status'] == 'CHILD_EXECUTED_REQUIRES_VALIDATION' and
            invocation['binary_sha256'] == done['binary_sha256'] and invocation['freeze_manifest_sha256'] == freeze and
            invocation['primary_error'] is None and invocation['cleanup_errors'] == [] and
            invocation['exit_code'] == invocation['reaped_exit_code'] == 0 and
            all(invocation[k] is True for k in ('leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed','log_closed')) and
            affinity(report['actual_affinity']) == invocation['cpu_affinity'] and
            invocation['address_space_cap_bytes'] == 512*1024**2 and invocation['wall_cap_seconds'] == 900,
            'actual bounded owned child/lifetime/CPU proof')
    source = read_json(directory/'source-after.json')
    require(source['all_equal'] is True and source['source_before'] == source['source_after'] == invocation['source_before'] and
            len(source['source_after']) == 31 and
            sum(r['path'].endswith('.safetensors') for r in source['source_after']) == 18,
            'all31 original source files including18 shards unchanged')
    sb = report['source_binding']
    require(type(sb['identity']) is str and sb['identity'].startswith('SourceIdentity(') and sb['identity'] != 'SourceIdentity(0)' and
            sb['verified_index_sha256'] == '77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df' and
            sb['verified_bundle_sha256'] == '7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2' and
            sb['complete_text_roles'] == 851,'actual original source issuer/index/bundle/full text role closure')
    mib = 1024**2
    steady,live,retained,final = [report[k+'_ledger'] for k in ('steady','live','retained','final')]
    require(steady['reserved'] == 80*mib and steady['reservations'] == 4 and
            steady['per_class'].get('Other',0) == 8*mib and steady['per_class'].get('Scratch',0) == 72*mib,
            'original admitted source/binding/control before extraction')
    require(live['reserved'] == 84*mib and retained['reserved'] == 76*mib,
            'two original1MiB input plus1MiB output owners retained')
    for row in (live,retained):
        require(row['per_class'].get('Output',0) == 4*64*1024 and
                row['per_class'].get('TransferStaging',0) == 0 and row['fenced'] is False,
                'all original private backing classes and staging retirement')
    require(final['reserved'] == final['reservations'] == 0 and final['fenced'] is False,'lastdrop original host ledger0')
    require(report['negative'] == [dict(step=i,pre_cancel=True,wrong_layer=True,foreign_input_source=True,
                real_one_byte_short_aggregate_refusal=True) for i in range(2)],
            'actual both-teacher cancel/foreign/layer/real-pool refusal proof')
    require(len(report['comparisons']) == len(report['original_parameters']) == 2 and
            report['retained_comparisons'] == report['comparisons'],'both phases retain all inputs/outputs')
    require(read_json(directory/'comparisons.json') == dict(typed=report['comparisons'],different_words=0),
            'actual complete comparison receipt')
    files = {'report.json','source-after.json','comparisons.json'}
    total = 0
    raw,pinned = original_parameters(binding,source)
    for step in range(2):
        with RawArchive(binding[f'step{step}_archive']) as archive:
            rows = binding['steps'][step]['tensors']
            for role,label in [('A_log','a_log_bf16'),('dt_bias','dt_bias_bf16')]:
                require(archive.raw(rows[label]) == raw[role],
                        'actual original raw parameter equals both genuine public calls')
            require(report['original_parameters'][step] == [dict(role=role,bytes=96,
                offset=binding['original_parameters'][role]['absolute_offset'],shard=pinned['path'],
                shard_sha256=pinned['sha256'],bundle_sha256=sb['verified_bundle_sha256'],
                raw_sha256=hashlib.sha256(raw[role]).hexdigest(),consumed_by_preparation=role=='dt_bias')
                for role in ('A_log','dt_bias')],'exact original parameter proofs; A_log unevaluated')
            typed = report['comparisons'][step]
            require(typed['step'] == step and typed['output_words'] == 144 and typed['retained_input_words'] == 96 and
                    typed['different_words'] == 0,'complete phase counts')
            current,words = comparison(typed['checks'],INPUTS+OUTPUTS,f'step{step}.typed.',rows,archive,directory)
            files |= current
            total += words
    require(total == 480 and {p.name for p in directory.iterdir()} == files and
            {p.name for p in lifecycle.iterdir()} == {'execution.log','invocation.json'},
            'all288 outputs and192 retained inputs, exact complete artifact tree')
    tree(args.output,reserve=65536)
    value = dict(status='PASS_PRIVATE_ORIGINAL27B_H48_COEFFICIENT_PREPARATION_ONLY',
        independent_output_stage_checks=6,independent_output_stage_words=288,retained_input_words=192,
        unevaluated_original_A_log_source_words=48,actual_original_dt_source_words=48,
        completed_sha256=sha(args.output/'completed.json'),freeze_manifest_sha256=freeze,
        ROOT_build_source_attestation_sha256=build,
        full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified=False)
    require(len((json.dumps(value,indent=2,sort_keys=True)+'\n').encode()) <= 65536,
            'reserved final metadata bytes before write')
    require(frozen() == freeze and implementation(args.implementation) == build,'all prepared/build sources unchanged')
    write(args.output/'validation.json',value)
    tree(args.output)
    print(json.dumps(value,sort_keys=True))


if __name__ == '__main__':
    main()


````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/followup-wrong-layer-v2/replay/write_implementation.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1016; SHA256 `71dc18ed5196403ced8894fd82bd8cb7f7b317273e69dfe5af17318d7041b540`.

````text
"""ROOT ONLY after isolated merge/build: source attestation, not compiler proof."""
import argparse
from pathlib import Path
from contracts import HERE, frozen, pin, read_json, require, sha, write
from source_contract import expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-output',type=Path,required=True)
    args = parser.parse_args()
    freeze = frozen()
    root,rows = expected(HERE,read_json,pin,require)
    value = dict(status='ROOT_ISOLATED_SOURCE_ATTESTATION_AFTER_H48_COEFFICIENT_PATCH',
        MAIN_merged=False,compiler_or_numerical_proof=False,workspace_root=str(root),
        preparation_sha256=freeze,
        prior_ROOT521_source_sha256='036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615',
        files=[rows[k] for k in sorted(rows)])
    write(args.new_output,value)
    require(frozen() == freeze,'all immutable prepared sources unchanged')
    print(sha(args.new_output))


if __name__ == '__main__':
    main()

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 9455; SHA256 `05095281de89dfc2ed136e869b5ca11458ca3664293b94f79a2fae900fc07e2e`.

````text
{
  "MAIN_modified": false,
  "expected_retained_input_words": 192,
  "expected_stage_words": 288,
  "files": [
    {
      "bytes": 1308,
      "file": "BASELINE.json",
      "sha256": "9cdbc3d3cc94943e8c6f91ec896b32beefd92fb702dd6c773facddfb63b3d6fb"
    },
    {
      "bytes": 759,
      "file": "MATH_UNCHANGED.json",
      "sha256": "bfb70d3e503ba0f39e91f8e59c365abd422b0662220ce8a7450c3b8e1bd3c045"
    },
    {
      "bytes": 14635,
      "file": "NEW_DELTA.patch",
      "sha256": "3cf6da6802d83089c45485eb606e79d0d3a0e22be815b2890005306f0b61cdca"
    },
    {
      "bytes": 2661,
      "file": "README.md",
      "sha256": "80e87387d670e165d1ed1a14c2badc02f9bb59969c0ab2b377d1a87fe7519d41"
    },
    {
      "bytes": 45513,
      "file": "REPLAY_REVIEW.diff",
      "sha256": "9453eb00ed8744fa5a1ca763040e657560b6f59d2c609e38b82e4d75833da186"
    },
    {
      "bytes": 1112,
      "file": "TODO.md",
      "sha256": "290d409a06aa5d9538db641ec5a5066ad96dbddf119b49de1cc9ba172bde4104"
    },
    {
      "bytes": 8890,
      "file": "baseline/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
      "sha256": "74b8a311402e42c04cd3060dd34a8c4dfa05dd2d061f4d2ae89f896eecb7fec8"
    },
    {
      "bytes": 10372,
      "file": "baseline/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs",
      "sha256": "27d7f314a00c599e6d7a90a4e0c6519a820c7c46ec31f3ae44788b89735363ce"
    },
    {
      "bytes": 667,
      "file": "owned-delta.json",
      "sha256": "741b46c43ea9f3b056e2e06a442c14464437069fe77de1ac77b6f8d72e955e9a"
    },
    {
      "bytes": 12695,
      "file": "proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs",
      "sha256": "72c12e1036956930eec8de9456b1900b21c4421bc652473b7ffa8f7e8aaedcfd"
    },
    {
      "bytes": 8949,
      "file": "proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
      "sha256": "0410ae2727f6ba7248b830412a0ba243edbc4dfd979d75141a6560834609f80e"
    },
    {
      "bytes": 10591,
      "file": "proposed/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs",
      "sha256": "8ad73434fc0b29a92e55cfbee8ef193413004683e13ced27ef72fea0b0072528"
    },
    {
      "bytes": 993,
      "file": "replay/Cargo.toml",
      "sha256": "3ed29dfdc498807ae1c88a5d8d2a41e3aac011fbb2927d86738a0ce68ba00d58"
    },
    {
      "bytes": 3661,
      "file": "replay/ROOT.md",
      "sha256": "94f1b34a1eb40d0a933713619c7f4de311ebfeaa65d1f45d303dcc746ddfa12c"
    },
    {
      "bytes": 6305,
      "file": "replay/contracts.py",
      "sha256": "70a4c8d960da4b158e847fe62aa8b648c1d14be8eb75f2bbb00a1442113b370d"
    },
    {
      "bytes": 29779,
      "file": "replay/factory.rs",
      "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
    },
    {
      "bytes": 25135,
      "file": "replay/main.rs",
      "sha256": "901ed3fbba449b0e2b58f5a2e5c61d9ea9ce8b5156a7378dceb0cc1db3da57c8"
    },
    {
      "bytes": 3702,
      "file": "replay/owned_lifetime.py",
      "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
    },
    {
      "bytes": 5707,
      "file": "replay/owned_native.py",
      "sha256": "2bf4b2ea543e7bf3f888f2b9cd86c3db18d0c5884934cbb67cc9d1d2378942f1"
    },
    {
      "bytes": 9701,
      "file": "replay/prepare_reference.py",
      "sha256": "5bb22f05a84d1d91e3c33da6165fa7b6f8abb13482bf2d1673f256b52f78479f"
    },
    {
      "bytes": 3915,
      "file": "replay/run_replay.py",
      "sha256": "8e68bc2ab3cfe05b65464004b8cc6c0edb798b301ac34ffa8fe7c2127f19e706"
    },
    {
      "bytes": 1428,
      "file": "replay/sigterm_cancellation.py",
      "sha256": "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128"
    },
    {
      "bytes": 25831,
      "file": "replay/source-bindings.json",
      "sha256": "670f1b5812c58de4355a04879b6decac8aee3964d4440f0b8699e4c6d21f3be9"
    },
    {
      "bytes": 7058,
      "file": "replay/source-pin.json",
      "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
    },
    {
      "bytes": 4499,
      "file": "replay/source_contract.py",
      "sha256": "f899457cff49b3295b380158b0b2973129110228481dd621bd0a54fdc81f61a1"
    },
    {
      "bytes": 3332,
      "file": "replay/test_contracts.py",
      "sha256": "11af9dac715ec78cfb824869e12f06292f2cf05a0d8e23d71019811208a8ac01"
    },
    {
      "bytes": 12438,
      "file": "replay/validate_replay.py",
      "sha256": "0e4596f0d5e80ab9db2b75538184c48a4937af8147a6f23eede1c84c42b75fff"
    },
    {
      "bytes": 1016,
      "file": "replay/write_implementation.py",
      "sha256": "71dc18ed5196403ced8894fd82bd8cb7f7b317273e69dfe5af17318d7041b540"
    }
  ],
  "full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified": false,
  "parents": [
    {
      "bytes": 40673,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/freeze-manifest.json",
      "sha256": "d744cb9ddab32ab94e144758ae954548c90fec0ee14ec4eb0f7b1c5c44915660"
    },
    {
      "bytes": 772,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/root-validator-v3.json",
      "sha256": "cc6955beeb87f151436902c1bec0cf29c3dd85aa20dc4ab91c5ae307fe66196f"
    },
    {
      "bytes": 8760,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/root_validate_reference_v3.py",
      "sha256": "f6fb730320246c515ceec09c0d0063342ee264bed52decf35de481514711c4c6"
    },
    {
      "bytes": 52055,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-gdn-coefficient-preparation-20261020/followup-v2/root-public-1/reference/reference.json",
      "sha256": "47efae4666f833df28c808fef7c91831931a5ce81a0b388e1c60c3c5e4af1d7a"
    },
    {
      "bytes": 99248,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
      "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
    },
    {
      "bytes": 233492,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/manifest.json",
      "sha256": "62e82637c7d24a58b733f2358b3061efd767725901314df4ff7378707d946ab8"
    },
    {
      "bytes": 596,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/validation.json",
      "sha256": "951f74e33de53c4b6b5254bd980df1f6c05f6b9869a46bf617c7fa93184d344d"
    },
    {
      "bytes": 22448,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/invocation.json",
      "sha256": "61b1787ca857a1464fd61c8132475dbe7bac2b4282b3b7ff4db7bc15d427512f"
    },
    {
      "bytes": 2322,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/prepared-manifest-v3.json",
      "sha256": "d3614b48bb41033a4052d765d3f7633d456812670783049db75ceadf33d8817d"
    },
    {
      "bytes": 15423,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/source-plan.json",
      "sha256": "658280444d4e3d1b6a3a8f2ce4b97e4be7cbff2c8a6ea7c66b3c733c78f31638"
    },
    {
      "bytes": 84104,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-0/prevalidation.json",
      "sha256": "17bd8f5430931cb33a22af32ac056f7357b02df32c9224d1c67a6361be20f46b"
    },
    {
      "bytes": 74924,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-qwen38-gdn-reference-20261020/root-reference-v3-1/captured/step-1/prevalidation.json",
      "sha256": "7ceceff0e776b7c4f1c97074de7b065264b69e374397f97bc9f1535f4517dc60"
    },
    {
      "bytes": 29779,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/factory.rs",
      "sha256": "2302f6a05ffbfcac0965cc4e20781ddaae5604f2320a984214507eaab7d09f69"
    },
    {
      "bytes": 7058,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/source-pin.json",
      "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
    },
    {
      "bytes": 3702,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-bf16-gamma-gated-rms-20261020/replay/isolated-root-v2/owned_lifetime.py",
      "sha256": "c10d034bd8ae0ca1dc065b68bd27f2bbe6f092a0cda0cbf8af738a363f0229b8"
    },
    {
      "bytes": 1428,
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-native-before-after-20261020/sigterm_cancellation.py",
      "sha256": "00d1efc11b302527ae8baa24ee76af0333c02946872db84f5f1d5e9d5e1ba128"
    }
  ],
  "runtime_or_payload_read_by_preparer": false,
  "schema": "zerogravity.h48-coefficient.prepared-source.v1",
  "status": "SOURCE_ONLY_UNEXECUTED_REQUIRES_ROOT_REVIEW_AND_QUALIFICATION"
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/BASELINE.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1308; SHA256 `9cdbc3d3cc94943e8c6f91ec896b32beefd92fb702dd6c773facddfb63b3d6fb`.

````text
{
  "baseline_files": [
    {
      "bytes": 10372,
      "path": "crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs",
      "sha256": "27d7f314a00c599e6d7a90a4e0c6519a820c7c46ec31f3ae44788b89735363ce"
    },
    {
      "bytes": 8890,
      "path": "crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
      "sha256": "74b8a311402e42c04cd3060dd34a8c4dfa05dd2d061f4d2ae89f896eecb7fec8"
    }
  ],
  "baseline_root": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020",
  "owned_paths": [
    "crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs",
    "crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
    "crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs"
  ],
  "parent_plan": ".agents/plans/model-execution/TODO.md",
  "root_source_freeze": {
    "bytes": 99248,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-root-bf16-composition-20261020/root-composed-source-freeze-7.json",
    "sha256": "036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615"
  },
  "runtime_tests_or_payload_reads_by_preparer": false,
  "scope": "Private H48 coefficient PREPARATION only; unevaluated raw A_log descriptor accommodation"
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/MATH_UNCHANGED.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 759; SHA256 `bfb70d3e503ba0f39e91f8e59c365abd422b0662220ce8a7450c3b8e1bd3c045`.

````text
{
  "CPU_body_bytes": 3837,
  "CPU_body_sha256": "805c060c4300106f2faffb689bff8a2477de7e886a8d9a582b2f1c6590113f66",
  "CPU_coefficient_body_equal_to_d744": true,
  "CPU_current_full_file": {
    "bytes": 7234,
    "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu/src/original_hf/coefficient_preparation.rs",
    "sha256": "a3032c27074f9a49545052f3d96062a02539616f45c1d0d8e3c9d578ed75e572"
  },
  "GPU_matches_changed_by_proposal": false,
  "actual_helper_and_core_body_changed_by_proposal": false,
  "consumed_operands": "same BF16 a/b and genuine rank1 BF16/F32 dt_bias; A_log never passed to coefficient helper",
  "scope": "source byte check only; no model/math/provider/runtime execution"
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/baseline/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8890; SHA256 `74b8a311402e42c04cd3060dd34a8c4dfa05dd2d061f4d2ae89f896eecb7fec8`.

````text
//! Synthetic source/transport/lease guards only; never trained coefficients.
use super::*;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, PlanBuilder, PlanOperation, ProgramNumericalProfile,
};
use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
fn coefficient_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_num_value_heads"] = json!(16);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, fallback| {
            if name.ends_with(".dt_bias") {
                0
            } else {
                fallback
            }
        },
        |_| {},
    )
}
fn raw(value: u16) -> Vec<u8> {
    [value.to_le_bytes(); 16].concat()
}
fn input<'a>(
    source: crate::accelerator::SourceIdentity,
    a: &'a [u8],
    b: &'a [u8],
) -> GdnCoefficientPreparationInput<'a> {
    GdnCoefficientPreparationInput::from_bf16_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        a,
        b,
    )
    .unwrap()
}
#[test]
fn synthetic_coefficient_stages_retain_two_aggregate_backings_and_source() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let a = raw(0x41a8);
    let b = raw(0); // a21, b0: public linear softplus and sigmoid0.
    let before = reserved(&admission);
    let output = operators
        .execute_gdn_coefficient_preparation_slice(0, input(source.source_identity(), &a, &b), None)
        .unwrap();
    assert!(
        output
            .biased_alpha_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .softplus_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .beta_bf16()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 0.5f32.to_bits())
    );
    assert_eq!(
        output.raw_dt_bias_parameter().kind,
        zerogravity_core::quant::QuantKind::Bf16
    );
    assert_eq!(
        output.unevaluated_raw_a_log_parameter().kind,
        zerogravity_core::quant::QuantKind::F32
    );
    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 34 * 1024 * 1024);
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 16]);
    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 16]);
    drop(output);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn wrong_layer_foreign_input_cancel_and_pressure_refuse_without_leak() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let before = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                3,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(
                    crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
                    &zero,
                    &zero
                ),
                None
            )
            .is_err()
    );
    let cancelled = AtomicBool::new(true);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                Some(&cancelled)
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    let mut pressure = Vec::new();
    loop {
        let remaining = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let pressured = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), pressured);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn raw_a_log_alias_and_unaware_provider_refuse_before_callback_or_output() {
    struct Provider(AtomicBool);
    impl crate::accelerator::MatVecKernel for Provider {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            panic!("unaware coefficient provider invoked");
        }
    }
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let (a, b) = {
        let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        math.import_coefficient_preparation_input(input(source.source_identity(), &zero, &zero), 16)
            .unwrap()
    };
    let before = reserved(&admission);
    let provider = Provider(AtomicBool::new(false));
    {
        let mut math = crate::host_math::HostMath::for_window_operator_slice(
            &operators,
            None,
            Some(&provider),
        );
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnDt(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    {
        let mut math =
            crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnA(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(a);
    drop(b);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn plan_appends_preparation_without_exp_state_token_or_producer_invention() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (model, _, _) = bind(loaded);
    let mut builder =
        PlanBuilder::new_operator_slice(model.source_identity(), model.program_numerical_profile())
            .unwrap();
    // Synthetic same-builder views are transport ONLY, never actual trained a/b.
    let embed = builder
        .embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let a = builder.view(&embed, TensorView::vector(0, 16)).unwrap();
    let b = builder.view(&embed, TensorView::vector(16, 16)).unwrap();
    let values = model
        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
        .unwrap();
    let plan = builder.finish_operator_slice(values.softplus).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(
        plan.plan()
            .nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TypedGdnAlphaPreparation { .. }
                    | PlanOperation::TypedGdnBetaPreparation { .. }
            ))
            .count(),
        2
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, true).is_err());
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/baseline/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10372; SHA256 `27d7f314a00c599e6d7a90a4e0c6519a820c7c46ec31f3ae44788b89735363ce`.

````text
//! Source-bound coefficient PREPARATION on caller-owned a/b; no g/decay/exp.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues, MatrixAxes, PlanBuilder,
    PlanValue, ProgramNumericalProfile, SourceParameter,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};
use zerogravity_core::quant::QuantKind;

pub(crate) struct CoefficientPreparationRoles {
    pub(crate) heads: usize,
    pub(crate) dt: SourceParameter,
    pub(crate) a_log: SourceParameter,
}
/// Immutable complete role descriptors; A_log is not evaluated by this slice.
pub(crate) fn coefficient_roles(
    source: &(impl OperatorSliceSource + ?Sized),
    layer: usize,
) -> Result<CoefficientPreparationRoles> {
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || !source.source_identity().is_bound()
        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "coefficient preparation original LinearAttention source/profile".into(),
        ));
    }
    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
        .map_err(|_| Error::Limit("coefficient heads".into()))?;
    let hidden = usize::try_from(source.geometry().hidden_size)
        .map_err(|_| Error::Limit("coefficient hidden".into()))?;
    if !matches!(heads, 16 | 48) || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
        return Err(Error::Unsupported(
            "coefficient preparation closed source geometry".into(),
        ));
    }
    let namespace = format!("model.language_model.layers.{layer}.linear_attn");
    for (role, suffix) in [
        (QwenMatrixRole::GdnAlpha(layer), "in_proj_a"),
        (QwenMatrixRole::GdnBeta(layer), "in_proj_b"),
    ] {
        let tensor = source.matrix_descriptor(role)?;
        if tensor.source != source.source_identity()
            || tensor.name != format!("{namespace}.{suffix}.weight")
            || tensor.kind != QuantKind::Bf16
            || tensor.axes != MatrixAxes::OutputInput
            || tensor.rows != heads
            || tensor.columns != hidden
        {
            return Err(Error::State(
                "coefficient preparation original a/b full matrix role".into(),
            ));
        }
    }
    let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
    let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
    if dt.source != source.source_identity()
        || dt.name != format!("{namespace}.dt_bias")
        || a_log.source != source.source_identity()
        || a_log.name != format!("{namespace}.A_log")
        || a_log.kind != QuantKind::F32
        || a_log.shape != [heads as u64]
        || a_log.bytes != heads * 4
    {
        return Err(Error::State(
            "coefficient preparation original raw A_log/dt_bias roles".into(),
        ));
    }
    GdnCoefficientPreparationContract::TorchCpuBf16Avx2.validate(&dt, heads, heads)?;
    Ok(CoefficientPreparationRoles { heads, dt, a_log })
}

impl LoadedQwenModel {
    /// Append only beta/biased-alpha/softplus to caller's original plan values.
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
}

/// External actual source/operator a/b bytes; caller authenticates provenance.
/// No hidden state, matrix producer, token or frontier is asserted by import.
#[derive(Clone, Copy)]
pub struct GdnCoefficientPreparationInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    alpha: &'a [u8],
    beta: &'a [u8],
}
impl<'a> GdnCoefficientPreparationInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        alpha: &'a [u8],
        beta: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            alpha,
            beta,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn alpha_bytes(self) -> &'a [u8] {
        self.alpha
    }
    pub fn beta_bytes(self) -> &'a [u8] {
        self.beta
    }
    pub(crate) fn validate(self) -> Result<usize> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::State(
                "coefficient preparation private input source/profile".into(),
            ));
        }
        if !matches!(self.alpha.len(), 32 | 96)
            || self.beta.len() != self.alpha.len()
            || self
                .alpha
                .chunks_exact(2)
                .chain(self.beta.chunks_exact(2))
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "coefficient preparation one finite BF16 a/b call".into(),
            ));
        }
        Ok(self.alpha.len() / 2)
    }
}

#[cfg(feature = "cpu")]
pub struct GdnCoefficientPreparationOutput {
    source: SourceIdentity,
    layer: usize,
    dt: SourceParameter,
    a_log: SourceParameter,
    alpha: crate::host_math::HostValue,
    beta_input: crate::host_math::HostValue,
    values: GdnCoefficientPreparationValues<crate::host_math::HostValue>,
}
#[cfg(feature = "cpu")]
impl GdnCoefficientPreparationOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn raw_dt_bias_parameter(&self) -> &SourceParameter {
        &self.dt
    }
    pub fn unevaluated_raw_a_log_parameter(&self) -> &SourceParameter {
        &self.a_log
    }
    pub fn alpha_bf16(&self) -> Result<&[f32]> {
        self.alpha.as_slice()
    }
    pub fn beta_input_bf16(&self) -> Result<&[f32]> {
        self.beta_input.as_slice()
    }
    pub fn biased_alpha_f32(&self) -> Result<&[f32]> {
        self.values.biased_alpha.as_slice()
    }
    pub fn softplus_f32(&self) -> Result<&[f32]> {
        self.values.softplus.as_slice()
    }
    pub fn beta_bf16(&self) -> Result<&[f32]> {
        self.values.beta.as_slice()
    }
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_gdn_coefficient_preparation_slice(
        &self,
        layer: usize,
        input: GdnCoefficientPreparationInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<GdnCoefficientPreparationOutput> {
        let roles = coefficient_roles(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (alpha, beta_input) = math.import_coefficient_preparation_input(input, roles.heads)?;
        let values = crate::forward::gdn_coefficient_preparation_slice(
            self,
            &mut math,
            layer,
            &alpha,
            &beta_input,
        )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient preparation cancelled".into(),
            ));
        }
        Ok(GdnCoefficientPreparationOutput {
            source: self.source_identity(),
            layer,
            dt: roles.dt,
            a_log: roles.a_log,
            alpha,
            beta_input,
            values,
        })
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn prepare_coefficient_stages(
        &self,
        dt: &SourceParameter,
        alpha: &[f32],
        beta: &[f32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<zerogravity_cpu::original_hf::coefficient_preparation::GdnCoefficientPreparation>
    {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient raw parameter cancelled".into(),
            ));
        }
        let raw = self.source().parameter(&dt.name)?;
        raw.validate_source_parameter(dt)?;
        let window = match cancelled {
            Some(flag) => {
                self.source()
                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
            }
            None => self.source().read_parameter(&raw, 0..raw.elements())?,
        };
        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
            return Err(Error::State(
                "window coefficient complete raw dt_bias".into(),
            ));
        }
        let result =
            zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
                self.source_identity(),
                zerogravity_core::source::SourceTensorRef {
                    kind: raw.tensor().kind,
                    shape: &raw.tensor().shape,
                    packed_bytes: window.as_slice(),
                },
                alpha,
                beta,
            )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient cancelled after synchronous math".into(),
            ));
        }
        Ok(result)
    }
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/owned-delta.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 667; SHA256 `741b46c43ea9f3b056e2e06a442c14464437069fe77de1ac77b6f8d72e955e9a`.

````text
{
  "files": [
    {
      "bytes": 10591,
      "file": "crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs",
      "sha256": "8ad73434fc0b29a92e55cfbee8ef193413004683e13ced27ef72fea0b0072528"
    },
    {
      "bytes": 8949,
      "file": "crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs",
      "sha256": "0410ae2727f6ba7248b830412a0ba243edbc4dfd979d75141a6560834609f80e"
    },
    {
      "bytes": 12695,
      "file": "crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs",
      "sha256": "72c12e1036956930eec8de9456b1900b21c4421bc652473b7ffa8f7e8aaedcfd"
    }
  ],
  "source_only": true
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_h48_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 12695; SHA256 `72c12e1036956930eec8de9456b1900b21c4421bc652473b7ffa8f7e8aaedcfd`.

````text
//! Sealed synthetic H48 source/metadata/lease checks; no trained math claim.
use super::*;
use crate::accelerator::program::{ProjectionContract, SourceParameter, SourceTensor};
use crate::operator_slice::OperatorSliceSource;
use zerogravity_core::quant::QuantKind;

fn h48_source(fixture: &Fixture) -> (VerifiedTensorSource, TransferAdmission, Vec<u8>, String) {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    let text = &mut cfg["text_config"];
    text["num_hidden_layers"] = json!(2);
    text["full_attention_interval"] = json!(2);
    text["layer_types"] = json!(["linear_attention", "full_attention"]);
    text["hidden_size"] = json!(256);
    text["linear_num_key_heads"] = json!(16);
    text["linear_num_value_heads"] = json!(48);
    text["linear_key_head_dim"] = json!(128);
    text["linear_value_head_dim"] = json!(128);
    let cfg = serde_json::to_vec(&cfg).unwrap();
    let mut header = BTreeMap::new();
    let mut map = BTreeMap::new();
    let mut payload = Vec::new();
    for spec in QwenTextConfig::from_json(&cfg)
        .unwrap()
        .tensor_specs()
        .unwrap()
    {
        let begin = payload.len();
        let value = if spec.name.ends_with("dt_bias") {
            0
        } else {
            0x3e80u16
        };
        for _ in 0..spec.shape.iter().product::<u64>() {
            payload.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            spec.name.clone(),
            json!({"dtype":"BF16", "shape":spec.shape,
            "data_offsets":[begin,payload.len()]}),
        );
        map.insert(spec.name, "model.safetensors");
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(&payload);
    assert!(bytes.len() < 16 * 1024 * 1024);
    let path = fixture.0.join("model.safetensors");
    fs::write(&path, &bytes).unwrap();
    let index = serde_json::to_vec(&json!({"metadata":{"total_size":payload.len()},
        "weight_map":map}))
    .unwrap();
    let index_sha = digest(&index);
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "coefficient-h48-source-test").unwrap();
    arbiter.register(key.clone(), 128 * 1024 * 1024).unwrap();
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("synthetic", "coefficient-h48").unwrap(),
    };
    let artifact =
        Artifact::from_file(File::open(&path).unwrap(), Path::new("model.safetensors")).unwrap();
    let source = SourceWeights::prepare_verified_windows(
        &index,
        &index_sha,
        vec![
            ExpectedSourceShard::new(
                "model.safetensors".into(),
                artifact,
                bytes.len() as u64,
                digest(&bytes),
            )
            .unwrap(),
        ],
        WindowOpenLimits {
            max_source_bytes: 16 * 1024 * 1024,
            max_metadata_bytes: 32 * 1024 * 1024,
            max_shards: 1,
            max_tensors: 200,
            scan_chunk_bytes: 64 * 1024,
        },
        &host,
    )
    .unwrap()
    .open()
    .unwrap();
    (source, host, cfg, index_sha)
}

struct MetadataFault<'a> {
    source: &'a OriginalWindowOperators,
    a_log: SourceParameter,
    profile: ProgramNumericalProfile,
}
impl OperatorSliceSource for MetadataFault<'_> {
    fn geometry(&self) -> &crate::model::ModelGeometry {
        self.source.geometry()
    }
    fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source.source_identity()
    }
    fn program_numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    fn rms_epsilon(&self) -> f32 {
        self.source.rms_epsilon()
    }
    fn matrix_descriptor(&self, role: QwenMatrixRole) -> Result<SourceTensor> {
        self.source.matrix_descriptor(role)
    }
    fn parameter_descriptor(&self, role: QwenParameterRole) -> Result<SourceParameter> {
        if role == QwenParameterRole::GdnA(0) {
            Ok(self.a_log.clone())
        } else {
            self.source.parameter_descriptor(role)
        }
    }
    fn embedding_contract(&self) -> EmbeddingContract {
        self.source.embedding_contract()
    }
    fn projection_contract(&self) -> ProjectionContract {
        self.source.projection_contract()
    }
}

#[test]
fn only_genuine_h48_bf16_unevaluated_a_log_extent_is_admitted() {
    use crate::operator_slice::coefficient_preparation::coefficient_roles;
    let old = Fixture::new();
    let loaded = coefficient_source(&old);
    let (source, host) = windows(&old, &loaded);
    let ops = bind_windows(source.clone(), &loaded);
    let roles = coefficient_roles(&ops, 0).unwrap();
    assert_eq!(
        (roles.heads, roles.a_log.kind, roles.a_log.bytes),
        (16, QuantKind::F32, 64)
    );
    let mut unsupported = roles.a_log;
    unsupported.kind = QuantKind::Bf16;
    unsupported.bytes = 32;
    assert!(
        coefficient_roles(
            &MetadataFault {
                source: &ops,
                a_log: unsupported,
                profile: ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
            },
            0
        )
        .is_err()
    );
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);

    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let roles = coefficient_roles(&ops, 0).unwrap();
    assert_eq!(
        (
            roles.heads,
            roles.a_log.kind,
            roles.a_log.shape.clone(),
            roles.a_log.bytes
        ),
        (48, QuantKind::Bf16, vec![48], 96)
    );
    assert_eq!(
        (roles.dt.kind, roles.dt.shape, roles.dt.bytes),
        (QuantKind::Bf16, vec![48], 96)
    );
    let before = reserved(&host);
    for (kind, bytes, shape, foreign, name, profile) in [
        (
            QuantKind::F16,
            96,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            192,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::F32,
            96,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![47],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![48],
            true,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![48],
            false,
            "foreign.A_log".into(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        ),
        (
            QuantKind::Bf16,
            96,
            vec![48],
            false,
            roles.a_log.name.clone(),
            ProgramNumericalProfile::GgufSourceV1,
        ),
    ] {
        let mut parameter = roles.a_log.clone();
        parameter.kind = kind;
        parameter.bytes = bytes;
        parameter.shape = shape;
        parameter.name = name;
        if foreign {
            parameter.source = crate::accelerator::SourceIdentity::for_new_allocation().unwrap();
        }
        assert!(
            coefficient_roles(
                &MetadataFault {
                    source: &ops,
                    a_log: parameter,
                    profile
                },
                0
            )
            .is_err()
        );
        assert_eq!(reserved(&host), before);
    }
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn h48_preparation_preserves_source_lease_cancel_pressure_and_all_private_words() {
    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let a = 0x41a8u16.to_le_bytes().repeat(48);
    let b = 0u16.to_le_bytes().repeat(48);
    let make = |identity| input(identity, &a, &b);
    let before = reserved(&host);
    for (layer, identity, cancel) in [
        (1, source.source_identity(), None),
        (
            0,
            crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
            None,
        ),
        (0, source.source_identity(), Some(AtomicBool::new(true))),
    ] {
        assert!(
            ops.execute_gdn_coefficient_preparation_slice(layer, make(identity), cancel.as_ref())
                .is_err()
        );
        assert_eq!(reserved(&host), before);
    }
    let mut pressure = Vec::new();
    loop {
        let remaining = host.arbiter.snapshot(&host.resource).unwrap().remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let pressured = reserved(&host);
    assert!(
        ops.execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
            .is_err()
    );
    assert_eq!(reserved(&host), pressured);
    drop(pressure);
    assert_eq!(reserved(&host), before);
    let output = ops
        .execute_gdn_coefficient_preparation_slice(0, make(source.source_identity()), None)
        .unwrap();
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 48]);
    assert_eq!(output.biased_alpha_f32().unwrap(), &[21.0; 48]);
    assert_eq!(output.softplus_f32().unwrap(), &[21.0; 48]);
    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
    assert_eq!(
        output.unevaluated_raw_a_log_parameter().kind,
        QuantKind::Bf16
    );
    assert_eq!(reserved(&host), before + 2 * 1024 * 1024);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 34 * 1024 * 1024);
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 48]);
    assert_eq!(output.beta_bf16().unwrap(), &[0.5; 48]);
    drop(output);
    assert_eq!(reserved(&host), 0);
}

#[test]
fn h48_recording_adds_only_existing_two_preparation_nodes() {
    let fixture = Fixture::new();
    let (source, host, cfg, index) = h48_source(&fixture);
    let ops = OriginalWindowOperators::bind(
        source.clone(),
        &cfg,
        &digest(&cfg),
        &index,
        QwenNumericalProfile::OriginalHfTorchCpuSlowBf16Avx2,
    )
    .unwrap();
    let mut builder = PlanBuilder::new_operator_slice(
        source.source_identity(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
    )
    .unwrap();
    // Same-builder views exercise transport; no authentic projection is claimed.
    let embed = builder
        .embed_profiled(
            ops.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let a = builder.view(&embed, TensorView::vector(0, 48)).unwrap();
    let b = builder.view(&embed, TensorView::vector(48, 48)).unwrap();
    let values = ops
        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
        .unwrap();
    let plan = builder.finish_operator_slice(values.softplus).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(
        plan.plan()
            .nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TypedGdnAlphaPreparation { .. }
                    | PlanOperation::TypedGdnBetaPreparation { .. }
            ))
            .count(),
        2
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    drop(ops);
    drop(source);
    assert_eq!(reserved(&host), 0);
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/proposed/crates/zerogravity-model-qwen35/src/coefficient_preparation_tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 8949; SHA256 `0410ae2727f6ba7248b830412a0ba243edbc4dfd979d75141a6560834609f80e`.

````text
//! Synthetic source/transport/lease guards only; never trained coefficients.
use super::*;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, PlanBuilder, PlanOperation, ProgramNumericalProfile,
};
use crate::operator_slice::coefficient_preparation::GdnCoefficientPreparationInput;
fn coefficient_source(fixture: &Fixture) -> Loaded {
    let mut cfg: Value = serde_json::from_slice(&config(true)).unwrap();
    cfg["text_config"]["hidden_size"] = json!(256);
    cfg["text_config"]["linear_num_value_heads"] = json!(16);
    source_with_config(
        fixture,
        serde_json::to_vec(&cfg).unwrap(),
        true,
        |name, _, fallback| {
            if name.ends_with(".dt_bias") {
                0
            } else {
                fallback
            }
        },
        |_| {},
    )
}
fn raw(value: u16) -> Vec<u8> {
    [value.to_le_bytes(); 16].concat()
}
fn input<'a>(
    source: crate::accelerator::SourceIdentity,
    a: &'a [u8],
    b: &'a [u8],
) -> GdnCoefficientPreparationInput<'a> {
    GdnCoefficientPreparationInput::from_bf16_bytes(
        source,
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        a,
        b,
    )
    .unwrap()
}
#[test]
fn synthetic_coefficient_stages_retain_two_aggregate_backings_and_source() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let a = raw(0x41a8);
    let b = raw(0); // a21, b0: public linear softplus and sigmoid0.
    let before = reserved(&admission);
    let output = operators
        .execute_gdn_coefficient_preparation_slice(0, input(source.source_identity(), &a, &b), None)
        .unwrap();
    assert!(
        output
            .biased_alpha_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .softplus_f32()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 21.0f32.to_bits())
    );
    assert!(
        output
            .beta_bf16()
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 0.5f32.to_bits())
    );
    assert_eq!(
        output.raw_dt_bias_parameter().kind,
        zerogravity_core::quant::QuantKind::Bf16
    );
    assert_eq!(
        output.unevaluated_raw_a_log_parameter().kind,
        zerogravity_core::quant::QuantKind::F32
    );
    assert_eq!(reserved(&admission), before + 2 * 1024 * 1024);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 34 * 1024 * 1024);
    assert_eq!(output.alpha_bf16().unwrap(), &[21.0; 16]);
    assert_eq!(output.beta_input_bf16().unwrap(), &[0.0; 16]);
    drop(output);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn wrong_layer_foreign_input_cancel_and_pressure_refuse_without_leak() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let before = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                3,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(
                    crate::accelerator::SourceIdentity::for_new_allocation().unwrap(),
                    &zero,
                    &zero
                ),
                None
            )
            .is_err()
    );
    let cancelled = AtomicBool::new(true);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                Some(&cancelled)
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), before);
    let mut pressure = Vec::new();
    loop {
        let remaining = admission
            .arbiter
            .snapshot(&admission.resource)
            .unwrap()
            .remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(
            source
                .reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)
                .unwrap(),
        );
    }
    let pressured = reserved(&admission);
    assert!(
        operators
            .execute_gdn_coefficient_preparation_slice(
                0,
                input(source.source_identity(), &zero, &zero),
                None
            )
            .is_err()
    );
    assert_eq!(reserved(&admission), pressured);
    drop(pressure);
    assert_eq!(reserved(&admission), before);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn raw_a_log_alias_and_unaware_provider_refuse_before_callback_or_output() {
    struct Provider(AtomicBool);
    impl crate::accelerator::MatVecKernel for Provider {
        fn try_execute(
            &self,
            _: crate::accelerator::MatVecRegion<'_>,
        ) -> Result<crate::accelerator::RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            panic!("unaware coefficient provider invoked");
        }
    }
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (source, admission) = windows(&fixture, &loaded);
    let operators = bind_windows(source.clone(), &loaded);
    let zero = raw(0);
    let (a, b) = {
        let math = crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        math.import_coefficient_preparation_input(input(source.source_identity(), &zero, &zero), 16)
            .unwrap()
    };
    let before = reserved(&admission);
    let provider = Provider(AtomicBool::new(false));
    {
        let mut math = crate::host_math::HostMath::for_window_operator_slice(
            &operators,
            None,
            Some(&provider),
        );
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnDt(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    {
        let mut math =
            crate::host_math::HostMath::for_window_operator_slice(&operators, None, None);
        assert!(
            math.coefficient_preparation_profiled(
                operators
                    .parameter_descriptor(QwenParameterRole::GdnA(0))
                    .unwrap(),
                &a,
                &b,
                GdnCoefficientPreparationContract::TorchCpuBf16Avx2
            )
            .is_err()
        );
    }
    assert!(!provider.0.load(Ordering::Relaxed));
    assert_eq!(reserved(&admission), before);
    drop(a);
    drop(b);
    drop(operators);
    drop(source);
    assert_eq!(reserved(&admission), 0);
}
#[test]
fn plan_appends_preparation_without_exp_state_token_or_producer_invention() {
    let fixture = Fixture::new();
    let loaded = coefficient_source(&fixture);
    let (model, _, _) = bind(loaded);
    let mut builder =
        PlanBuilder::new_operator_slice(model.source_identity(), model.program_numerical_profile())
            .unwrap();
    // Synthetic same-builder views are transport ONLY, never actual trained a/b.
    let embed = builder
        .embed_profiled(
            model.matrix_descriptor(QwenMatrixRole::Embedding).unwrap(),
            PlanBuilder::inputs().token,
            EmbeddingContract::Bf16OutputInput,
        )
        .unwrap();
    let a = builder.view(&embed, TensorView::vector(0, 16)).unwrap();
    let b = builder.view(&embed, TensorView::vector(16, 16)).unwrap();
    let values = model
        .append_gdn_coefficient_preparation_slice(&mut builder, 0, &a, &b)
        .unwrap();
    let plan = builder.finish_operator_slice(values.softplus).unwrap();
    assert_eq!(plan.plan().nodes().len(), 5);
    assert_eq!(
        plan.plan()
            .nodes()
            .iter()
            .filter(|n| matches!(
                n.operation,
                PlanOperation::TypedGdnAlphaPreparation { .. }
                    | PlanOperation::TypedGdnBetaPreparation { .. }
            ))
            .count(),
        2
    );
    assert!(plan.plan().required_domains().is_empty());
    assert!(plan.plan().proposed_domains().is_empty());
    assert!(plan.plan().logits().is_none());
    assert_eq!(plan.plan().frontier_delta(), 0);
    assert!(model.new_session().is_err());
    assert!(model.forward_plan(8, true, true).is_err());
}

#[path = "coefficient_preparation_h48_tests.rs"]
mod h48;

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/proposed/crates/zerogravity-model-qwen35/src/operator_slice/coefficient_preparation.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10591; SHA256 `8ad73434fc0b29a92e55cfbee8ef193413004683e13ced27ef72fea0b0072528`.

````text
//! Source-bound coefficient PREPARATION on caller-owned a/b; no g/decay/exp.
use super::OperatorSliceSource;
use crate::accelerator::SourceIdentity;
use crate::accelerator::program::{
    GdnCoefficientPreparationContract, GdnCoefficientPreparationValues, MatrixAxes, PlanBuilder,
    PlanValue, ProgramNumericalProfile, SourceParameter,
};
use crate::original_binding::{QwenMatrixRole, QwenParameterRole};
use crate::{Error, LoadedQwenModel, Result};
use zerogravity_core::quant::QuantKind;

pub(crate) struct CoefficientPreparationRoles {
    pub(crate) heads: usize,
    pub(crate) dt: SourceParameter,
    pub(crate) a_log: SourceParameter,
}
/// Immutable complete role descriptors; A_log is not evaluated by this slice.
pub(crate) fn coefficient_roles(
    source: &(impl OperatorSliceSource + ?Sized),
    layer: usize,
) -> Result<CoefficientPreparationRoles> {
    if source.program_numerical_profile() != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        || !source.source_identity().is_bound()
        || source.geometry().layers.get(layer) != Some(&crate::model::LayerKind::LinearAttention)
    {
        return Err(Error::Unsupported(
            "coefficient preparation original LinearAttention source/profile".into(),
        ));
    }
    let heads = usize::try_from(source.geometry().recurrent_value_heads.unwrap_or(0))
        .map_err(|_| Error::Limit("coefficient heads".into()))?;
    let hidden = usize::try_from(source.geometry().hidden_size)
        .map_err(|_| Error::Limit("coefficient hidden".into()))?;
    if !matches!(heads, 16 | 48) || !matches!(hidden, 256 | 1024 | 2048 | 5120) {
        return Err(Error::Unsupported(
            "coefficient preparation closed source geometry".into(),
        ));
    }
    let namespace = format!("model.language_model.layers.{layer}.linear_attn");
    for (role, suffix) in [
        (QwenMatrixRole::GdnAlpha(layer), "in_proj_a"),
        (QwenMatrixRole::GdnBeta(layer), "in_proj_b"),
    ] {
        let tensor = source.matrix_descriptor(role)?;
        if tensor.source != source.source_identity()
            || tensor.name != format!("{namespace}.{suffix}.weight")
            || tensor.kind != QuantKind::Bf16
            || tensor.axes != MatrixAxes::OutputInput
            || tensor.rows != heads
            || tensor.columns != hidden
        {
            return Err(Error::State(
                "coefficient preparation original a/b full matrix role".into(),
            ));
        }
    }
    let dt = source.parameter_descriptor(QwenParameterRole::GdnDt(layer))?;
    let a_log = source.parameter_descriptor(QwenParameterRole::GdnA(layer))?;
    // A_log is metadata only; no exp/g or learned-parameter cast is granted.
    let a_log_codec_valid = match a_log.kind {
        QuantKind::F32 => a_log.bytes == heads * 4,
        QuantKind::Bf16 => heads == 48 && a_log.bytes == 96,
        _ => false,
    };
    if dt.source != source.source_identity()
        || dt.name != format!("{namespace}.dt_bias")
        || a_log.source != source.source_identity()
        || a_log.name != format!("{namespace}.A_log")
        || a_log.shape != [heads as u64]
        || !a_log_codec_valid
    {
        return Err(Error::State(
            "coefficient preparation original raw A_log/dt_bias roles".into(),
        ));
    }
    GdnCoefficientPreparationContract::TorchCpuBf16Avx2.validate(&dt, heads, heads)?;
    Ok(CoefficientPreparationRoles { heads, dt, a_log })
}

impl LoadedQwenModel {
    /// Append only beta/biased-alpha/softplus to caller's original plan values.
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        self.require_operator_slice()?;
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
}

/// External actual source/operator a/b bytes; caller authenticates provenance.
/// No hidden state, matrix producer, token or frontier is asserted by import.
#[derive(Clone, Copy)]
pub struct GdnCoefficientPreparationInput<'a> {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    alpha: &'a [u8],
    beta: &'a [u8],
}
impl<'a> GdnCoefficientPreparationInput<'a> {
    pub fn from_bf16_bytes(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        alpha: &'a [u8],
        beta: &'a [u8],
    ) -> Result<Self> {
        let input = Self {
            source,
            profile,
            alpha,
            beta,
        };
        input.validate()?;
        Ok(input)
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn alpha_bytes(self) -> &'a [u8] {
        self.alpha
    }
    pub fn beta_bytes(self) -> &'a [u8] {
        self.beta
    }
    pub(crate) fn validate(self) -> Result<usize> {
        if !self.source.is_bound()
            || self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::State(
                "coefficient preparation private input source/profile".into(),
            ));
        }
        if !matches!(self.alpha.len(), 32 | 96)
            || self.beta.len() != self.alpha.len()
            || self
                .alpha
                .chunks_exact(2)
                .chain(self.beta.chunks_exact(2))
                .any(|b| u16::from_le_bytes([b[0], b[1]]) & 0x7f80 == 0x7f80)
        {
            return Err(Error::Malformed(
                "coefficient preparation one finite BF16 a/b call".into(),
            ));
        }
        Ok(self.alpha.len() / 2)
    }
}

#[cfg(feature = "cpu")]
pub struct GdnCoefficientPreparationOutput {
    source: SourceIdentity,
    layer: usize,
    dt: SourceParameter,
    a_log: SourceParameter,
    alpha: crate::host_math::HostValue,
    beta_input: crate::host_math::HostValue,
    values: GdnCoefficientPreparationValues<crate::host_math::HostValue>,
}
#[cfg(feature = "cpu")]
impl GdnCoefficientPreparationOutput {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn layer(&self) -> usize {
        self.layer
    }
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
    }
    pub fn raw_dt_bias_parameter(&self) -> &SourceParameter {
        &self.dt
    }
    pub fn unevaluated_raw_a_log_parameter(&self) -> &SourceParameter {
        &self.a_log
    }
    pub fn alpha_bf16(&self) -> Result<&[f32]> {
        self.alpha.as_slice()
    }
    pub fn beta_input_bf16(&self) -> Result<&[f32]> {
        self.beta_input.as_slice()
    }
    pub fn biased_alpha_f32(&self) -> Result<&[f32]> {
        self.values.biased_alpha.as_slice()
    }
    pub fn softplus_f32(&self) -> Result<&[f32]> {
        self.values.softplus.as_slice()
    }
    pub fn beta_bf16(&self) -> Result<&[f32]> {
        self.values.beta.as_slice()
    }
}

#[cfg(all(feature = "original-hf-binding", target_os = "linux"))]
impl crate::original_binding::OriginalWindowOperators {
    pub fn append_gdn_coefficient_preparation_slice(
        &self,
        builder: &mut PlanBuilder,
        layer: usize,
        a: &PlanValue,
        b: &PlanValue,
    ) -> Result<GdnCoefficientPreparationValues<PlanValue>> {
        crate::forward::gdn_coefficient_preparation_slice(self, builder, layer, a, b)
    }
    #[cfg(feature = "cpu")]
    pub fn execute_gdn_coefficient_preparation_slice(
        &self,
        layer: usize,
        input: GdnCoefficientPreparationInput<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<GdnCoefficientPreparationOutput> {
        let roles = coefficient_roles(self, layer)?;
        let mut math = crate::host_math::HostMath::for_window_operator_slice(self, cancelled, None);
        let (alpha, beta_input) = math.import_coefficient_preparation_input(input, roles.heads)?;
        let values = crate::forward::gdn_coefficient_preparation_slice(
            self,
            &mut math,
            layer,
            &alpha,
            &beta_input,
        )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient preparation cancelled".into(),
            ));
        }
        Ok(GdnCoefficientPreparationOutput {
            source: self.source_identity(),
            layer,
            dt: roles.dt,
            a_log: roles.a_log,
            alpha,
            beta_input,
            values,
        })
    }
    #[cfg(feature = "cpu")]
    pub(crate) fn prepare_coefficient_stages(
        &self,
        dt: &SourceParameter,
        alpha: &[f32],
        beta: &[f32],
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<zerogravity_cpu::original_hf::coefficient_preparation::GdnCoefficientPreparation>
    {
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient raw parameter cancelled".into(),
            ));
        }
        let raw = self.source().parameter(&dt.name)?;
        raw.validate_source_parameter(dt)?;
        let window = match cancelled {
            Some(flag) => {
                self.source()
                    .read_parameter_cancellable(&raw, 0..raw.elements(), flag)?
            }
            None => self.source().read_parameter(&raw, 0..raw.elements())?,
        };
        if !window.is_complete() || window.element_range() != (0..raw.elements()) {
            return Err(Error::State(
                "window coefficient complete raw dt_bias".into(),
            ));
        }
        let result =
            zerogravity_cpu::original_hf::coefficient_preparation::prepare_gdn_coefficients(
                self.source_identity(),
                zerogravity_core::source::SourceTensorRef {
                    kind: raw.tensor().kind,
                    shape: &raw.tensor().shape,
                    packed_bytes: window.as_slice(),
                },
                alpha,
                beta,
            )?;
        if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(Error::State(
                "window coefficient cancelled after synchronous math".into(),
            ));
        }
        Ok(result)
    }
}

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/replay/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 993; SHA256 `3ed29dfdc498807ae1c88a5d8d2a41e3aac011fbb2927d86738a0ce68ba00d58`.

````text
[package]
name = "original27b-h48-coefficient-preparation-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu", features = ["cache"] }
zerogravity-model-qwen35 = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-model-qwen35", features = ["cpu", "original-hf-binding"] }
serde_json = "=1.0.151"
sha2 = "=0.10.9"
libc = "=0.2.190"

[profile.release]
opt-level = 3
codegen-units = 1

[[bin]]
name = "original27b-h48-coefficient-preparation-replay"
path = "main.rs"

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/frozen-prepared/replay/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6305; SHA256 `70a4c8d960da4b158e847fe62aa8b648c1d14be8eb75f2bbb00a1442113b370d`.

````text
"""ROOT-only bounded metadata/raw transport; no model/provider imports/calls."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HERE = Path(__file__).resolve().parent
INPUTS = ('alpha_bf16', 'beta_input_bf16')
OUTPUTS = ('biased_alpha_f32', 'softplus_f32', 'beta_bf16')
PROFILE = 'original27B-raw-source-bf16-v1'

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def sha(path):
    with Path(path).open('rb') as opened:
        return hashlib.file_digest(opened, 'sha256').hexdigest()

def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON key')
        out[key] = value
    return out

def identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

def read(path, cap=2*1024**2, row=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb') as opened:
        before = os.fstat(opened.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded regular metadata')
        raw = opened.read(cap+1)
        require(len(raw) == before.st_size and identity(before) == identity(os.fstat(opened.fileno())),
                'bounded unchanged metadata FD')
    if row is not None:
        require(len(raw) == row['bytes'] and digest(raw) == row['sha256'], 'exact metadata bytes/SHA')
    return raw

def pin(path, row, cap=2*1024**2):
    read(path, cap, row)

def read_json(path):
    return json.loads(read(path), object_pairs_hook=unique)

def write(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    require(len(raw) <= 2*1024**2, 'bounded metadata receipt')
    with Path(path).open('xb') as opened:
        opened.write(raw)

def frozen():
    from source_contract import frozen as checked_frozen
    return checked_frozen(HERE, read_json, pin, require, sha)

def implementation(path):
    from source_contract import implementation as checked_implementation
    return checked_implementation(HERE, path, read_json, pin, require, sha)


def finite(raw, dtype):
    require(dtype in ('BF16', 'F32'), 'closed observed dtype')
    width, fmt, mask = (2, '<H', 0x7f80) if dtype == 'BF16' else (4, '<I', 0x7f800000)
    require(raw and len(raw) % width == 0 and
            all(word & mask != mask for word, in struct.iter_unpack(fmt, raw)),
            'finite complete observed words; no signed-zero canonicalization')

def widened(raw, dtype):
    finite(raw, dtype)
    return b''.join((word << 16).to_bytes(4, 'little') for word, in struct.iter_unpack('<H', raw)) \
        if dtype == 'BF16' else raw

class RawArchive:
    """Whole-file streaming authentication; bounded selected reads on SAME FD."""
    def __init__(self, row):
        self.row, self.opened = row, None
    def __enter__(self):
        self.opened = os.fdopen(os.open(self.row['path'], os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW), 'rb')
        try:
            self.before = os.fstat(self.opened.fileno())
            require(stat.S_ISREG(self.before.st_mode) and self.before.st_size == self.row['bytes'] and
                    8 < self.before.st_size <= 64*1024**2, 'bounded full raw archive FD')
            self._authenticate()
            size, = struct.unpack('<Q', os.pread(self.opened.fileno(), 8, 0))
            require(1 < size <= 1024**2 and 8+size < self.before.st_size,
                    'header bound BEFORE reading declared extent')
            raw = os.pread(self.opened.fileno(), size, 8)
            require(len(raw) == size, 'complete original header read')
            self.header = json.loads(raw, object_pairs_hook=unique)
            require(type(self.header) is dict and 1 <= len(self.header) <= 2048, 'bounded archive fields')
            self.start, spans = 8+size, []
            for name, item in self.header.items():
                if name == '__metadata__':
                    continue
                require(item['dtype'] in ('BF16', 'F32') and type(item['shape']) is list and
                        len(item['shape']) <= 8, 'original rank/dtype')
                words = 1
                for axis in item['shape']:
                    require(type(axis) is int and 0 < axis <= 262144, 'bounded original axis')
                    words *= axis
                require(type(item['data_offsets']) is list and len(item['data_offsets']) == 2,
                        'two original offsets')
                lo, hi = item['data_offsets']
                require(type(lo) is type(hi) is int and 0 <= lo < hi <= self.before.st_size-self.start and
                        hi-lo == words*(2 if item['dtype'] == 'BF16' else 4), 'complete actual archive span')
                spans.append((lo, hi))
            end = 0
            for lo, hi in sorted(spans):
                require(lo == end, 'no overlapping/gapped actual archive data')
                end = hi
            require(self.start+end == self.before.st_size, 'no unaccounted trailing archive bytes')
            return self
        except BaseException:
            self.opened.close()
            raise
    def _authenticate(self):
        self.opened.seek(0)
        require(hashlib.file_digest(self.opened, 'sha256').hexdigest() == self.row['sha256'] and
                identity(os.fstat(self.opened.fileno())) == identity(self.before),
                'complete actual archive digest/FD stable')
    def raw(self, row):
        item = self.header[row['name']]
        require(item['dtype'] == row['dtype'] and item['shape'] == row['shape'], 'actual observed name/dtype/shape')
        lo, hi = item['data_offsets']
        require(0 < hi-lo == row['bytes'] <= 24576, 'bounded actual selected stage')
        raw = os.pread(self.opened.fileno(), hi-lo, self.start+lo)
        require(len(raw) == hi-lo and digest(raw) == row['raw_sha256'] and
                identity(os.fstat(self.opened.fileno())) == identity(self.before), 'all raw stage bits and same retained FD')
        finite(raw, row['dtype'])
        return raw
    def __exit__(self, typ, value, trace):
        try:
            if typ is None:
                self._authenticate()
        finally:
            self.opened.close()

````

