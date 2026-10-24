# Paused continuation text — volume 9

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 25135; SHA256 `901ed3fbba449b0e2b58f5a2e5c61d9ea9ce8b5156a7378dceb0cc1db3da57c8`.

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
            && ops.binding().config().geometry().recurrent_value_heads == Some(48),
        "genuine851-role H48 source",
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
            ops.execute_gdn_coefficient_preparation_slice(1, input, None)
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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/owned_lifetime.py

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/owned_native.py

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/prepare_reference.py

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/run_replay.py

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/sigterm_cancellation.py

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source-bindings.json

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source-pin.json

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source_contract.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4499; SHA256 `f899457cff49b3295b380158b0b2973129110228481dd621bd0a54fdc81f61a1`.

````text
"""Explicit isolated ROOT source attestation; never compiler/numerical proof."""
from pathlib import Path
import hashlib
import tomllib


def frozen(here, read_json, pin, require, sha):
    run = here.parent
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
    config = read_json(here.parent/'BASELINE.json')
    pin(config['root_source_freeze']['path'], config['root_source_freeze'])
    prior = read_json(config['root_source_freeze']['path'])
    require(len(prior['files']) == 521 and prior['full_model_or_GPU_qualified'] is False,
            'exact previous isolated ROOT521 source closure')
    patch = read_json(here.parent/'owned-delta.json')
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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/test_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3332; SHA256 `11af9dac715ec78cfb824869e12f06292f2cf05a0d8e23d71019811208a8ac01`.

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

    def test_signal_is_record_only_until_explicit_safe_checkpoint(self):
        guard = SigtermCancellation()
        guard._record(15,None)
        self.assertTrue(guard.received)
        with self.assertRaises(InterruptedError):
            guard.check()


if __name__ == '__main__':
    unittest.main()

````

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/validate_replay.py

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

## Source: .agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/write_implementation.py

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

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 458; SHA256 `1a86c009f3a6e2d288b5bfa1678431e578e757fecdbce6b12777975232d4c210`.

````text
[package]
name = "zg-haswell-bulk-f32-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[[bin]]
name = "bulk-f32-replay"
path = "main.rs"

[dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
serde_json = "=1.0.151"
zerogravity-core = { path = "../../../../crates/zerogravity-core" }
zerogravity-cpu = { path = "../../../../crates/zerogravity-cpu" }
libc = "=0.2.190"

[profile.release]
lto = false
codegen-units = 1

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/HISTORICAL-RECOVERY.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5690; SHA256 `3c456c0b57c3da890083889d32c0830085b929b9b24902a0838b30222f09527a`.

````text
{
  "scope": "author-permitted bounded raw-token/artifact hashing ONLY; no tokenizer/model/operator execution",
  "historical_digest": "24054920794c549afc0c678f58f6f44a1ecc3228bcfa793aeacc9c8516fb86a2",
  "historical_digest_source": "7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md:22312-22320",
  "historical_serialization": "unspecified in recovered text; surviving cpu_context_step consumes {raw_tokens,count} JSON",
  "cases": [
    {
      "kind": "existing bounded raw-token artifact",
      "path": ".agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le",
      "bytes": 32768,
      "sha256": "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "existing bounded raw-token artifact",
      "path": ".agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.json",
      "bytes": 42944,
      "sha256": "1f079b0cbfe73c634be8f324cd386627c894601e85ce091a1e4516ee4567906d",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "compact",
      "newline": false,
      "bytes": 34780,
      "sha256": "fe8199d1d8e067ac3dd83e54656d64883ddc99fa2272c3f6c4ca1ce8dd77b164",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "compact",
      "newline": true,
      "bytes": 34781,
      "sha256": "f20e748a1eadc29d2ea3ed09e51b232eae920edf3853695878419771a2e46181",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "default",
      "newline": false,
      "bytes": 42974,
      "sha256": "1c384dfa5caa28f37fb8ff4ed367033b14162379c9e942ed4ffa026bb616fde1",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "default",
      "newline": true,
      "bytes": 42975,
      "sha256": "45abe21f457571cb3d0bdb79aad45ece898fbd0841618a398c91ea54edf65f8c",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "pretty2",
      "newline": false,
      "bytes": 75752,
      "sha256": "3c6631c347a4b3c9e72929fcac8cbaf3f818a2ff4fd2622533d8cb6d9e6db3de",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "pretty2",
      "newline": true,
      "bytes": 75753,
      "sha256": "8e7aee0e1b9c5ecd83556cf8dcef0fc981bb4d210fb45c4f0c02677ee8f13fd6",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "compact",
      "newline": false,
      "bytes": 34780,
      "sha256": "c02a13cf08c967cea35d2c759f36822bd71cd44a80cc60ba88000f14555ac88a",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "compact",
      "newline": true,
      "bytes": 34781,
      "sha256": "ad31479d103b6f1d5d06021e93ed93ce968ccf6ba3e2841320ae19be27fa175d",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "default",
      "newline": false,
      "bytes": 42974,
      "sha256": "8ec6d4bdb8e5a0a2a4514c5242174d09a1255bf9ecdf70000ec4041d5c03b250",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "default",
      "newline": true,
      "bytes": 42975,
      "sha256": "d0008f5cc8e6b6b10d700d3496edff72d25767aaff8d1396034f4ce4990c7e35",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "pretty2",
      "newline": false,
      "bytes": 75752,
      "sha256": "fad45678e40f0f7ae454cd57d8f1075b3800ab18ef545f2a3a39f2855850eebd",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "pretty2",
      "newline": true,
      "bytes": 75753,
      "sha256": "aa50484d97246e719d2c536f0346eb1107cba54b7aea84290b097d8fb3c7a742",
      "matches_historical_artifact_digest": false
    }
  ],
  "matches": 0,
  "historical_raw_ids_recovered": false,
  "different_historical_token_sequence_proven": false,
  "remaining": "Recover original task/ids file/REFERENCE_TOKENS stderr or original generator+format. These bounded negative artifact checks do not prove different token sequences."
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/baseline/math.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13197; SHA256 `12c6c6a44db2a216a87a23542b7fc675586298045d2539e45ef5fa4d6e3c048f`.

````text
//! Deterministic CPU reference operations used to validate compiled regions.
//!
//! These are mathematical building blocks, NOT full Qwen GDN or Gemma MoE layer
//! implementations. Model recipes will register exact higher-level semantics.

use crate::{Error, Result};

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Malformed(message.into()))
    }
}

/// Reference row-major matrix vector product, using diagnostic f64 accumulation.
pub fn matvec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    check(
        rows.checked_mul(cols) == Some(matrix.len()),
        "matvec matrix shape",
    )?;
    check(input.len() == cols, "matvec input size")?;
    check(rows <= 16_000_000, "matvec output too large")?;
    let mut out = Vec::new();
    out.try_reserve_exact(rows)
        .map_err(|_| Error::Limit("matvec output allocation failed".into()))?;
    for row in matrix.chunks_exact(cols.max(1)).take(rows) {
        let acc = row
            .iter()
            .zip(input)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        out.push(acc as f32);
    }
    if cols == 0 {
        out.resize(rows, 0.0);
    }
    Ok(out)
}

/// RMS norm reference, without residual addition or model-specific postscaling.
pub fn rms_norm(values: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    check(
        values.len() == weight.len() && !values.is_empty(),
        "rms norm shape",
    )?;
    check(epsilon.is_finite() && epsilon > 0.0, "rms norm epsilon")?;
    let sum: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    let inv = (sum / values.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    Ok(values
        .iter()
        .zip(weight)
        .map(|(x, w)| (*x as f64 * inv * f64::from(*w)) as f32)
        .collect())
}

/// Stable probability normalization. Reject NaN and infinities instead of masking
/// an unknown model's attention or router semantics.
pub fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    check(!values.is_empty(), "softmax empty")?;
    check(
        values.iter().all(|x| x.is_finite()),
        "softmax requires finite logits",
    )?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = values.iter().map(|v| f64::from(*v - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    check(sum > 0.0 && sum.is_finite(), "softmax unstable")?;
    Ok(weights.into_iter().map(|v| (v / sum) as f32).collect())
}

/// Router top-k selection is separated from model-specific scoring,
/// renormalization, and shared-expert accumulation rules.
pub fn top_k(logits: &[f32], k: usize) -> Result<Vec<usize>> {
    check(k > 0 && k <= logits.len(), "top-k size")?;
    check(
        logits.iter().all(|x| x.is_finite()),
        "top-k requires finite logits",
    )?;
    let mut rank: Vec<usize> = (0..logits.len()).collect();
    rank.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    rank.truncate(k);
    Ok(rank)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn gelu_tanh(x: f32) -> f32 {
    let a = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (a * (x + 0.044715 * x * x * x)).tanh())
}

/// One causal attention query over keys/values at positions 0..T.
/// `window=None` is full attention; `window=Some(n)` keeps the last n
/// positions inclusive of query position. This is diagnostic CPU math.
pub fn attention_one(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_position: usize,
    window: Option<usize>,
) -> Result<Vec<f32>> {
    let d = query.len();
    check(
        d > 0 && keys.len() == values.len() && keys.len().is_multiple_of(d),
        "attention head geometry",
    )?;
    let tokens = keys.len() / d;
    check(
        tokens > 0 && query_position < tokens,
        "attention query position",
    )?;
    if let Some(n) = window {
        check(n > 0, "sliding attention window")?;
    }
    let start = window.map_or(0, |n| (query_position + 1).saturating_sub(n));
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = Vec::new();
    for t in start..=query_position {
        let v = keys[t * d..(t + 1) * d]
            .iter()
            .zip(query)
            .map(|(&k, &q)| f64::from(k) * f64::from(q))
            .sum::<f64>();
        scores.push((v * scale) as f32);
    }
    let probs = softmax(&scores)?;
    let mut out = vec![0.0; d];
    for (i, p) in probs.iter().enumerate() {
        let v = &values[(start + i) * d..(start + i + 1) * d];
        for (dst, &x) in out.iter_mut().zip(v) {
            *dst += x * p;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_norm() {
        assert_eq!(
            matvec(&[1., 2., 3., 4.], 2, 2, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        assert!(matvec(&[1., 2.], 2, 2, &[1., 2.]).is_err());
        assert_eq!(matvec(&[], 3, 0, &[]).unwrap(), [0., 0., 0.]);
        let norm = rms_norm(&[3., 4.], &[1., 1.], 1e-6).unwrap();
        assert!((norm[0] - 0.848528).abs() < 1e-5);
        assert!((norm[1] - 1.131_37).abs() < 1e-5);
    }

    #[test]
    fn stable_softmax_and_tie_routing() {
        let probs = softmax(&[1000., 1000., 999.]).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert_eq!(top_k(&[3., 4., 4., 3.], 2).unwrap(), [1, 2]);
        assert!(top_k(&[f32::NAN, 1.], 1).is_err());
        assert!(softmax(&[f32::NEG_INFINITY]).is_err());
        assert!((silu(0.0)).abs() < 1e-9);
        assert!((gelu_tanh(0.0)).abs() < 1e-9);
    }

    #[test]
    fn sliding_vs_global_attention_and_future_mask() {
        let q = [1.];
        let k = [0., 0., 0., 0.];
        let v = [1., 2., 3., 100.];
        let global = attention_one(&q, &k, &v, 2, None).unwrap();
        let sliding = attention_one(&q, &k, &v, 2, Some(2)).unwrap();
        assert!((global[0] - 2.0).abs() < 1e-6);
        assert!((sliding[0] - 2.5).abs() < 1e-6);
        assert!(attention_one(&q, &k, &v, 1, Some(0)).is_err());
    }
}

/// Deterministic unfiltered logits/log-probability audit. These are produced
/// *before* temperature, penalties, truncation or grammar; a sampler's
/// post-filter probabilities are not an equivalent numerical oracle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogitRank {
    pub id: u32,
    pub delta: f32,
    pub logprob: f64,
}

pub fn top_logit_ranks(logits: &[f32], n: usize) -> Result<Vec<LogitRank>> {
    if logits.is_empty() || n == 0 || n > 128 || logits.len() > u32::MAX as usize {
        return Err(Error::Malformed("invalid logits audit width/rank".into()));
    }
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed("nonfinite logits in audit".into()));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|x| f64::from(*x - max).exp())
        .sum::<f64>();
    let ln_sum = sum.ln();
    if !ln_sum.is_finite() {
        return Err(Error::Malformed("nonfinite logits normalization".into()));
    }
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&i, &j| logits[j].total_cmp(&logits[i]).then_with(|| i.cmp(&j)));
    Ok(indices
        .into_iter()
        .take(n)
        .map(|id| {
            let delta = logits[id] - max;
            LogitRank {
                id: id as u32,
                delta,
                logprob: f64::from(delta) - ln_sum,
            }
        })
        .collect())
}

#[cfg(test)]
mod logit_audit_tests {
    use super::*;
    #[test]
    fn test_raw_log_probs_pre_sampler() {
        let ranks = top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 3).unwrap();
        assert_eq!(ranks.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 0]);
        assert_eq!(ranks[0].delta, 0.0);
        assert_eq!(ranks[2].delta, -2.0);
        assert!(
            (ranks.iter().map(|r| r.logprob.exp()).sum::<f64>()
                + top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 4).unwrap()[3]
                    .logprob
                    .exp()
                - 1.0)
                .abs()
                < 1e-10
        );
        assert!(top_logit_ranks(&[0.0, f32::NAN], 2).is_err());
    }
}

#[cfg(test)]
mod long_context_reference_tests {
    use super::*;

    #[test]
    fn thirty_two_thousand_causal_attention_positions_vs_independent_double_oracle() {
        const CONTEXT: usize = 32_768;
        const WIDTH: usize = 4;
        let query = [0.4, -0.7, 0.2, 0.1];
        let keys = (0..=CONTEXT)
            .flat_map(|t| (0..WIDTH).map(move |i| (((t * 7 + i * 37) % 113) as f32 - 56.0) / 97.0))
            .collect::<Vec<_>>();
        let mut values = (0..=CONTEXT)
            .flat_map(|t| {
                (0..WIDTH).map(move |i| (((t * 13 + i * 17) % 199) as f32 - 99.0) / 201.0)
            })
            .collect::<Vec<_>>();
        // Future token must never contribute; sentinel catches mask bugs.
        values[CONTEXT * WIDTH..].fill(1_000_000.0);
        let actual = attention_one(&query, &keys, &values, CONTEXT - 1, None).unwrap();
        let window = attention_one(&query, &keys, &values, CONTEXT - 1, Some(1024)).unwrap();
        let independently_evaluate = |start: usize| {
            let mut scores = Vec::new();
            for t in start..CONTEXT {
                let score = query
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| f64::from(q) * f64::from(keys[t * WIDTH + i]))
                    .sum::<f64>()
                    / (WIDTH as f64).sqrt();
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            std::array::from_fn::<_, WIDTH, _>(|i| {
                (weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(values[(start + j) * WIDTH + i]))
                    .sum::<f64>()
                    / sum) as f32
            })
        };
        let global_oracle = independently_evaluate(0);
        let sliding_oracle = independently_evaluate(CONTEXT - 1024);
        for i in 0..WIDTH {
            assert!(
                (actual[i] - global_oracle[i]).abs() < 2e-6,
                "full attention reference mismatch in channel {i}"
            );
            assert!(
                (window[i] - sliding_oracle[i]).abs() < 2e-6,
                "sliding attention reference mismatch in channel {i}"
            );
        }
    }
}

/// GGML Haswell/AVX2 scalar-equivalent approximate exponential used in the
/// source execution mode's SIMD SiLU implementation. This is *not* the
/// mathematical exp reference: its FMA polynomial and float-bit exponent
/// construction are part of the requested source-numerical specialization.
///
/// The common finite model gate range is handled by the exact polynomial
/// branch. For other finite ranges, follow the original SIMD scaling branches;
/// NaN/Inf values are rejected by the model state boundary before this stage.
pub fn ggml_source_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return x.exp();
    }
    let r = f32::from_bits(0x4b40_0000);
    let z = x.mul_add(f32::from_bits(0x3fb8_aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(
        f32::from_bits(0x35bf_be8e),
        (-n).mul_add(f32::from_bits(0x3f31_7200), x),
    );
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = f32::from_bits(0x3c07_2010)
        .mul_add(b, f32::from_bits(0x3d2b_9f17))
        .mul_add(
            u,
            f32::from_bits(0x3e2a_af33).mul_add(b, f32::from_bits(0x3eff_fedb)),
        )
        .mul_add(u, f32::from_bits(0x3f7f_fff6) * b);
    if n.abs() <= 126.0 {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// Opt-in, model-recipe-specific source-compatible SiLU on f32. This matches
/// GGML's vector approximation, rather than silently redefining generic SiLU.
pub fn ggml_source_silu(x: f32) -> f32 {
    x / (1.0 + ggml_source_exp(0.0 - x))
}

#[cfg(test)]
mod ggml_source_silu_tests {
    use super::*;
    #[test]
    fn source_approx_is_bounded_and_retains_general_silu_behavior() {
        for integer in -100..=100 {
            let x = integer as f32 / 10.0;
            let y = ggml_source_silu(x);
            assert!(y.is_finite());
            assert!(
                (y - silu(x)).abs() <= 1e-5,
                "GGML source approximation unexpectedly divergent at {x}"
            );
        }
        assert_eq!(ggml_source_silu(0.0), 0.0);
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3132; SHA256 `d648156caaff3c354215927808b085e353aec6e4f753b3564c927ea5fa83f175`.

````text
"""ROOT-only source/transport guards; importing performs no arithmetic/process work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AV = HERE.parent / 'numerics-and-context-av-replay-20261020'
ALPHA = HERE.parent / 'numerics-and-context-2b-alpha-replay-20261020'
CASES = ('av', 'alpha')
CAP = 8 * 1024 * 1024

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path, cap=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and (cap is None or before.st_size <= cap), 'bounded regular pinned file')
        total, value = 0, hashlib.sha256()
        while raw := stream.read(1024 * 1024):
            total += len(raw)
            value.update(raw)
        after = os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
                and total == before.st_size, 'immutable whole FD')
        return {'bytes': total, 'sha256': value.hexdigest()}

def read(path, cap=1024 * 1024):
    require(Path(path).stat().st_size <= cap, 'bounded JSON')
    with Path(path).open('rb') as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, 'bounded JSON actual bytes')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result, 'duplicate metadata key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)

def write(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    require(len(raw) <= 1024 * 1024, 'bounded report')
    with Path(path).open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(raw)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def frozen():
    freeze = read(HERE / 'freeze-manifest.json')
    for row in freeze['files']:
        require(digest(HERE / row['path'], 2 * 1024 * 1024) == row['digest'], 'exact new proposed source')
    for row in freeze['source_pins']:
        require(digest(row['path'], 2 * 1024 * 1024) == row['digest'], 'exact original source/metadata pin')
    return digest(HERE / 'freeze-manifest.json')['sha256']

def parents():
    """ROOT ONLY full authentic closure: can hash original model/DSO inputs."""
    av = load(AV / 'prepare.py', 'bulk_f32_original_av_prepare')
    alpha = load(ALPHA / 'prepare.py', 'bulk_f32_original_alpha_prepare')
    av_pins = av.jread(AV / 'pins.json')
    av.verify(av_pins)
    av.verify_fixtures(AV / 'root-fixtures', av_pins)
    alpha_pins = alpha.authenticate()
    alpha.capture_guards()
    alpha.verify_fixtures(ALPHA / 'root-fixtures', alpha_pins)
    return av,alpha

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/followup-attribution/UPSTREAM-NOTICE.txt

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1119; SHA256 `733a637fc0bd23db9abe25e821ff4e90b277626e4095aabfe473d62ab4d242b5`.

````text
// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/followup-attribution/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1740; SHA256 `a7e8cc3b9c2addb2c64ec994efb2368a79cbb359bcac9f7989a9d550ecfe6eb4`.

````text
{
  "schema": "zerogravity.closed-haswell-bulk-f32.attribution-followup.v1",
  "status": "SOURCE_ONLY_ATTRIBUTION_CORRECTION_NO_RUNTIME",
  "original_freeze_sha256": "689221fe60b982e93e4684012b5c4bf8e3dc061546e86fc0ed4be7f63c78f61f",
  "original_delta": {
    "bytes": 22333,
    "sha256": "e1460ed2f9a5db11f82a57cb456091d9072ef0cd87d782e2729171c2a8a03b29"
  },
  "original_module": {
    "bytes": 13182,
    "sha256": "1f17327c62f36e8d813aab902a907a01233d84967a92ed84033e6ebc4a03ac13"
  },
  "upstream": {
    "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.cpp",
    "digest": {
      "bytes": 166263,
      "sha256": "8413c9a0828ddea7289aee72a32c03bd56a23ee40179ec3600f9b0589a21e338"
    }
  },
  "files": [
    {
      "path": "ATTRIBUTION_ONLY.patch",
      "digest": {
        "bytes": 1534,
        "sha256": "ce21ffea1afe0851004dba001c3d239d1522d79be84155ac14f424a8c56706b1"
      }
    },
    {
      "path": "README.md",
      "digest": {
        "bytes": 683,
        "sha256": "244319f470fb0fee1e1d47cf4fba43d8532f454f1ec956e4a022d942eac51c30"
      }
    },
    {
      "path": "UPSTREAM-NOTICE.txt",
      "digest": {
        "bytes": 1119,
        "sha256": "733a637fc0bd23db9abe25e821ff4e90b277626e4095aabfe473d62ab4d242b5"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs",
      "digest": {
        "bytes": 14301,
        "sha256": "7557b599baf43d2f6e8b598484aee4fe95200b87972288df5c396a60c38331a5"
      }
    }
  ],
  "source_change": "Exact upstream Mozilla MIT comment insertion only; all remaining original module bytes unchanged",
  "original_owned_files_checked": 16
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/followup-attribution/proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 14301; SHA256 `7557b599baf43d2f6e8b598484aee4fe95200b87972288df5c396a60c38331a5`.

````text
//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
//!
//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.

// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

const MAX_SPAN_WORDS: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_WORDS: usize = 1024 * 1024;
const MAX_FMAS: usize = 128 * 1024 * 1024;
const SCRATCH_BYTES: u64 = 256;

#[cfg(test)]
std::thread_local! {
    static AFTER_FIRST_OUTPUT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Distinct closed arithmetic contract; no executable model/profile capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BulkF32Profile {
    GgmlD812350HaswellTinyBlas,
}

/// Original F32 axes `[inner,rows,head,sequence]`; strides/offset are in words.
/// Borrowed raw activations/weights retain their caller's owner and admission.
/// This descriptor validates a view, not its model-file SHA or source issuer.
#[derive(Clone, Copy)]
pub struct F32View<'a> {
    pub source: SourceIdentity,
    pub name: &'a str,
    pub values: &'a [f32],
    pub offset: usize,
    pub shape: [usize; 4],
    pub strides: [usize; 4],
}

/// Same source for both operands. B is the original contiguous RHS; A may
/// retain a strided V-cache view. All broadcast heads are independently read.
#[derive(Clone, Copy)]
pub struct BulkF32Request<'a> {
    pub profile: BulkF32Profile,
    pub source: SourceIdentity,
    pub left: F32View<'a>,
    pub right: F32View<'a>,
}

/// Existing host ledger, not a new pool or a claim about all host RAM.
pub struct BulkF32Admission<'a> {
    pub arbiter: &'a ResourceArbiter,
    pub resource: &'a PhysicalDeviceKey,
    pub owner: &'a ResourceOwner,
}

/// Payload drops before its original output/scratch ticket. No unleased Vec
/// escape or state publication; synchronous borrowed inputs outlive execution.
pub struct BulkF32Completed {
    values: Vec<f32>,
    source: SourceIdentity,
    shape: [usize; 4],
    _lease: GlobalResourceLease,
}
impl BulkF32Completed {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub fn profile(&self) -> BulkF32Profile {
        BulkF32Profile::GgmlD812350HaswellTinyBlas
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Haswell bulk F32 cancelled".into()));
    }
    Ok(())
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &next| {
        acc.checked_mul(next)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 geometry overflow".into()))
    })
}

impl F32View<'_> {
    fn validate(self, source: SourceIdentity, cancelled: Option<&AtomicBool>) -> Result<()> {
        not_cancelled(cancelled)?;
        if !source.is_bound()
            || self.source != source
            || self.name.is_empty()
            || self.name.len() > 256
            || self.name.contains('\0')
            || self.shape.contains(&0)
            || self.shape[0] > 8192
            || self.shape[1] > 4096
            || self.shape[2] > 64
            || self.shape[3] > 64
            || self.strides[0] != 1
        {
            return Err(Error::Malformed(
                "Haswell bulk F32 source/view geometry".into(),
            ));
        }
        // Original matrix rows do not overlap. Axis3's declared stride is
        // checked even when that dimension has size1, preserving raw geometry.
        let mut previous = self.shape[0];
        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
            if stride < previous {
                return Err(Error::Malformed(
                    "Haswell bulk F32 overlapping strides".into(),
                ));
            }
            previous = stride
                .checked_mul(size)
                .ok_or_else(|| Error::Limit("Haswell bulk F32 stride overflow".into()))?;
        }
        let end = self
            .shape
            .iter()
            .zip(self.strides)
            .try_fold(self.offset, |at, (&size, stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|n| at.checked_add(n))
                    .ok_or_else(|| Error::Limit("Haswell bulk F32 view span overflow".into()))
            })?
            .checked_add(1)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 view end overflow".into()))?;
        if end > self.values.len() || self.values.len() > MAX_SPAN_WORDS {
            return Err(Error::Limit(
                "Haswell bulk F32 view exceeds backing/bound".into(),
            ));
        }
        let words = checked_product(&self.shape)?;
        if words > MAX_OUTPUT_WORDS {
            return Err(Error::Limit(
                "Haswell bulk F32 logical operand bound".into(),
            ));
        }
        // Only actual logical words are read; storage gaps/canaries remain
        // outside the arithmetic and may contain arbitrary IEEE754 words.
        for sequence in 0..self.shape[3] {
            for head in 0..self.shape[2] {
                for row in 0..self.shape[1] {
                    not_cancelled(cancelled)?;
                    let start = self.at(0, row, head, sequence);
                    if self.values[start..start + self.shape[0]]
                        .iter()
                        .any(|x| !x.is_finite())
                    {
                        return Err(Error::Malformed(
                            "nonfinite Haswell bulk F32 operand".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    fn at(self, inner: usize, row: usize, head: usize, sequence: usize) -> usize {
        // Whole view's final index and products were checked before access.
        self.offset
            + inner
            + row * self.strides[1]
            + head * self.strides[2]
            + sequence * self.strides[3]
    }
}

fn validate(
    request: BulkF32Request<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<([usize; 4], usize)> {
    not_cancelled(cancelled)?;
    let BulkF32Profile::GgmlD812350HaswellTinyBlas = request.profile;
    let a = request.left;
    let b = request.right;
    a.validate(request.source, cancelled)?;
    b.validate(request.source, cancelled)?;
    if a.shape[0] != b.shape[0]
        || !(2..=64).contains(&b.shape[1])
        || !a.shape[1].is_multiple_of(4)
        || !b.shape[2].is_multiple_of(a.shape[2])
        || !b.shape[3].is_multiple_of(a.shape[3])
    {
        // n<2 selects the original vec-dot path, never this bulk schedule.
        return Err(Error::Unsupported(
            "unsupported Haswell tinyBLAS F32 shape".into(),
        ));
    }
    let mut stride = 1usize;
    for (&actual, &size) in b.strides.iter().zip(&b.shape) {
        if actual != stride {
            return Err(Error::Malformed(
                "Haswell tinyBLAS requires original contiguous RHS".into(),
            ));
        }
        stride = stride
            .checked_mul(size)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 contiguous span overflow".into()))?;
    }
    let shape = [a.shape[1], b.shape[1], b.shape[2], b.shape[3]];
    let count = checked_product(&shape)?;
    if count > MAX_OUTPUT_WORDS || count.checked_mul(a.shape[0]).is_none_or(|n| n > MAX_FMAS) {
        return Err(Error::Limit("Haswell bulk F32 output/work bound".into()));
    }
    not_cancelled(cancelled)?;
    Ok((shape, count))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let full = a.len() & !7usize;
    for base in (0..full).step_by(8) {
        for (lane, sum) in lanes.iter_mut().enumerate() {
            *sum = a[base + lane].mul_add(b[base + lane], *sum);
        }
    }
    if full != a.len() {
        // tinyBLAS load_partial zero-fills BOTH remaining Vec8 operands.
        // Execute those +0*+0 FMAs too; skipping them changes zero signs.
        for (lane, sum) in lanes.iter_mut().enumerate() {
            let at = full + lane;
            let (x, y) = if at < a.len() {
                (a[at], b[at])
            } else {
                (0.0, 0.0)
            };
            *sum = x.mul_add(y, *sum);
        }
    }
    let low = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    // Actual hsum(__m128): add(movehl), then add_ss(movehdup).
    // This is (0+2)+(1+3), not vec_dot's (0+1)+(2+3).
    (low[0] + low[2]) + (low[1] + low[3])
}

/// Pure source-derived primitive. Provider/model/session/checkpoint eligibility
/// is unchanged. Caller authenticates operands and retains input reservations;
/// this function admits only its own private result and bounded stack scratch.
pub fn execute(
    request: BulkF32Request<'_>,
    admission: BulkF32Admission<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<BulkF32Completed> {
    let (shape, count) = validate(request, cancelled)?;
    if admission.resource.backend != "host" {
        return Err(Error::Unsupported(
            "Haswell bulk F32 requires existing host admission".into(),
        ));
    }
    let output_bytes = count
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("Haswell bulk F32 output bytes overflow".into()))?;
    let lease = admission.arbiter.admit(
        admission.resource,
        admission.owner.clone(),
        &[
            AllocationRequest {
                kind: AllocationClass::Output,
                bytes: output_bytes as u64,
            },
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: SCRATCH_BYTES,
            },
        ],
    )?;
    not_cancelled(cancelled)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("Haswell bulk F32 output allocation".into()))?;
    if values.capacity() != count {
        return Err(Error::Limit(
            "Haswell bulk F32 unexpected output capacity".into(),
        ));
    }
    let a = request.left;
    let b = request.right;
    let ratio2 = b.shape[2] / a.shape[2];
    let ratio3 = b.shape[3] / a.shape[3];
    for sequence in 0..shape[3] {
        for head in 0..shape[2] {
            for column in 0..shape[1] {
                not_cancelled(cancelled)?;
                let br = b.at(0, column, head, sequence);
                for row in 0..shape[0] {
                    not_cancelled(cancelled)?;
                    let ar = a.at(0, row, head / ratio2, sequence / ratio3);
                    let value = dot(
                        &a.values[ar..ar + a.shape[0]],
                        &b.values[br..br + b.shape[0]],
                    );
                    if !value.is_finite() {
                        return Err(Error::Limit("Haswell bulk F32 nonfinite result".into()));
                    }
                    values.push(value);
                    #[cfg(test)]
                    if values.len() == 1 {
                        AFTER_FIRST_OUTPUT.with(|hook| {
                            if let Some(hook) = hook.borrow_mut().take() {
                                hook();
                            }
                        });
                    }
                }
            }
        }
    }
    not_cancelled(cancelled)?;
    Ok(BulkF32Completed {
        values,
        source: request.source,
        shape,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11376; SHA256 `689221fe60b982e93e4684012b5c4bf8e3dc061546e86fc0ed4be7f63c78f61f`.

````text
{
  "schema": "zerogravity.closed-haswell-bulk-f32.source-freeze.v1",
  "status": "SOURCE_ONLY_UNCOMPILED_UNEXECUTED_PRIMITIVE_PROPOSAL",
  "files": [
    {
      "path": "Cargo.toml",
      "digest": {
        "bytes": 458,
        "sha256": "1a86c009f3a6e2d288b5bfa1678431e578e757fecdbce6b12777975232d4c210"
      }
    },
    {
      "path": "HISTORICAL-RECOVERY.json",
      "digest": {
        "bytes": 5690,
        "sha256": "3c456c0b57c3da890083889d32c0830085b929b9b24902a0838b30222f09527a"
      }
    },
    {
      "path": "NEW_DELTA.patch",
      "digest": {
        "bytes": 22333,
        "sha256": "e1460ed2f9a5db11f82a57cb456091d9072ef0cd87d782e2729171c2a8a03b29"
      }
    },
    {
      "path": "README.md",
      "digest": {
        "bytes": 7558,
        "sha256": "afa11f70d6b06d6838b23f2a05bca2e221f3a1d8b38fa080a7a3d5851a2dce90"
      }
    },
    {
      "path": "TODO.md",
      "digest": {
        "bytes": 1474,
        "sha256": "03c99e42754eb17860ec6cf7203736c7d654ff26b3c0d66233fae1961b91a63c"
      }
    },
    {
      "path": "baseline/math.rs",
      "digest": {
        "bytes": 13197,
        "sha256": "12c6c6a44db2a216a87a23542b7fc675586298045d2539e45ef5fa4d6e3c048f"
      }
    },
    {
      "path": "contracts.py",
      "digest": {
        "bytes": 3132,
        "sha256": "d648156caaff3c354215927808b085e353aec6e4f753b3564c927ea5fa83f175"
      }
    },
    {
      "path": "main.rs",
      "digest": {
        "bytes": 15344,
        "sha256": "c4e96a19bb2e44e8ca5991af31f0fcd66747730cfab62b60ccdf28c6255c1758"
      }
    },
    {
      "path": "owned_lifetime.py",
      "digest": {
        "bytes": 11840,
        "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
      }
    },
    {
      "path": "prepare.py",
      "digest": {
        "bytes": 6030,
        "sha256": "2797547b5cc5f8df21f445d75de114b84587a81fe39dfe5650e8e5bfce779a47"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs",
      "digest": {
        "bytes": 7696,
        "sha256": "dac2aed8e6f9e9f4a8c50a509811a4dffe44a35e961c50b87efa14884e50064a"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs",
      "digest": {
        "bytes": 13182,
        "sha256": "1f17327c62f36e8d813aab902a907a01233d84967a92ed84033e6ebc4a03ac13"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math.rs",
      "digest": {
        "bytes": 13304,
        "sha256": "16b2a904fe208bbdeecc8de2c072c885ff12839041cb42b53675a50bb4d71526"
      }
    },
    {
      "path": "run_replay.py",
      "digest": {
        "bytes": 5557,
        "sha256": "e82614f3ecc80324e947fbeed431efde0f90dd6ba15616f3d5aed0c450b7f2ec"
      }
    },
    {
      "path": "test_contracts.py",
      "digest": {
        "bytes": 4834,
        "sha256": "a84e7590f87da81e40c339b87f6e0d71fd27a43c632083d327d884bf58430feb"
      }
    },
    {
      "path": "validate.py",
      "digest": {
        "bytes": 6074,
        "sha256": "abab362a92a7347f09d7f9ab8cf50da2ca4e6e9f120f0f56b7f88f7af2dcf5c3"
      }
    }
  ],
  "source_pins": [
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/Cargo.toml",
      "digest": {
        "bytes": 994,
        "sha256": "e9deb4ce0673d1a783b3b748ad5daec45a2e617f85b49fca0830e01c0a06daff"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/Cargo.lock",
      "digest": {
        "bytes": 139446,
        "sha256": "d46da60a19caa05aeeca07315a837709117b9a84e5bec980c4fb2591dd87d4ec"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/Cargo.toml",
      "digest": {
        "bytes": 463,
        "sha256": "6135312f01bc9eb12c395e4c0220b01ee5074ec20d681a6d5b1f53f7424170e5"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/lib.rs",
      "digest": {
        "bytes": 158,
        "sha256": "17e653a7fde1b47e17780d19a3af2a01f6c09d4ad9c2eba58b0bf80bd4a8fc7b"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/math.rs",
      "digest": {
        "bytes": 13197,
        "sha256": "12c6c6a44db2a216a87a23542b7fc675586298045d2539e45ef5fa4d6e3c048f"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/weights.rs",
      "digest": {
        "bytes": 41744,
        "sha256": "8282a25b231c5d126b5cba1100764919ca6fd2bcff503c08d5e6119e8dafc74a"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/accelerator.rs",
      "digest": {
        "bytes": 34440,
        "sha256": "50fcb42a211847bef44286c8e93e8e23cd3c48ee6dc6312cef446a25979a0998"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/resource_arbiter.rs",
      "digest": {
        "bytes": 29216,
        "sha256": "8e59d4d4a82b1e36f594573bac2a879df7008a2191b434f06f1ad151b75d9339"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/lib.rs",
      "digest": {
        "bytes": 89161,
        "sha256": "9a26849b304a8e9dfdd57478414c7845c6069f41a8d49c48742ec6debae22c4b"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/checkpoint.rs",
      "digest": {
        "bytes": 24080,
        "sha256": "dba8d20d034c16980812ec7616089a72ec6fbe4bbf0d055c05aea6c3350d03c4"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/LICENSE",
      "digest": {
        "bytes": 1078,
        "sha256": "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.cpp",
      "digest": {
        "bytes": 166263,
        "sha256": "8413c9a0828ddea7289aee72a32c03bd56a23ee40179ec3600f9b0589a21e338"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/vec.cpp",
      "digest": {
        "bytes": 25373,
        "sha256": "31e43ec66a1e2920a549dd2f71e6ce474e68f2fc298c4d2942404ca327e763d1"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/simd-mappings.h",
      "digest": {
        "bytes": 55790,
        "sha256": "a936298928e4dcfc1b61fb9e99cf8a152572b413e30f38dc8e7c3211ee9cb6a0"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.c",
      "digest": {
        "bytes": 132440,
        "sha256": "45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/prepare.py",
      "digest": {
        "bytes": 7867,
        "sha256": "4de1b3a06523805628bc4fddf90fd9c0dac45fe25b24f5b0f5157731a5cd8607"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/root-fixtures/manifest.json",
      "digest": {
        "bytes": 230559,
        "sha256": "a4fbb5428164deedd9e3405557edb3b9fe1a2c4eff2bead97081ceaf3ec5aa94"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/root-validation.json",
      "digest": {
        "bytes": 19917,
        "sha256": "ecb713f8c989fcc8d6745664a285c4449cde489c1ac98f7528664e3b69e65586"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/pins.json",
      "digest": {
        "bytes": 57155,
        "sha256": "345765351ca36d927d4c57b56e0fc7f46cf9b6f38a50f169d15860e82390977f"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/root-replay.invocation.json",
      "digest": {
        "bytes": 1524,
        "sha256": "5b383158ca0e6432966cabf6dd2c334e42f8b85b8e0db25349ba7b01b91b4233"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/prepare.py",
      "digest": {
        "bytes": 10742,
        "sha256": "5629603355d2e4593fce22f6273c5eab9eeffd63e83805adb658c7b66766b1af"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-fixtures/manifest.json",
      "digest": {
        "bytes": 193103,
        "sha256": "025a00e6803528a1d497fd069ba92155ea879f0e5dbe6748d501212b5a9d1354"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-validation.json",
      "digest": {
        "bytes": 20517,
        "sha256": "dcc47697478fe51906ab9c4d77cfed2459cfcd5b6d346bcb30c56a3a7f14817e"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/source-pins.json",
      "digest": {
        "bytes": 46227,
        "sha256": "bbbb612fd1c9dc5bf3caa893d5caedff6ff5ac20c4f63b68c627c22848910854"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-plain.invocation.json",
      "digest": {
        "bytes": 1892,
        "sha256": "694b5b14127880b66bf861eab3d671109f25b4541e164c33e3c3b37d73e00680"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/owned_container_v4.py",
      "digest": {
        "bytes": 11840,
        "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity/examples/cpu_context_step.rs",
      "digest": {
        "bytes": 10212,
        "sha256": "1a68eaa93d3906272c484aa0db9707fc22390164021b247de0e4e0a7b00ddde8"
      }
    }
  ],
  "production_paths": [
    "crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs",
    "crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs",
    "crates/zerogravity-cpu/src/math.rs"
  ],
  "scope": "explicit private CPU tinyBLAS F32 primitive only; defaultfourVec8/model/state/session/GPU/profile unchanged",
  "author_validation": {
    "rustfmt": true,
    "python_AST_only": true,
    "bounded_source_metadata_hashes": true,
    "bounded_raw_token_artifact_hashes_only": 14,
    "tests_builds_operators_providers_models_GPU": false
  },
  "root_pending": [
    "independent full source review",
    "sourcefree tests/strict compilation/full isolated gate",
    "genuine captured defaultAV+alpha132096word exact replay/ledger/lifetime/cleanup",
    "separate explicit production schedule integration and BOTHmodel short/long fullV/all42 gates"
  ],
  "historical_recovery": "240549 artifact target recovered; encoding unknown;14bounded candidates0matches do not prove different token histories"
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 458; SHA256 `1a86c009f3a6e2d288b5bfa1678431e578e757fecdbce6b12777975232d4c210`.

````text
[package]
name = "zg-haswell-bulk-f32-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[[bin]]
name = "bulk-f32-replay"
path = "main.rs"

[dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
serde_json = "=1.0.151"
zerogravity-core = { path = "../../../../crates/zerogravity-core" }
zerogravity-cpu = { path = "../../../../crates/zerogravity-cpu" }
libc = "=0.2.190"

[profile.release]
lto = false
codegen-units = 1

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/HISTORICAL-RECOVERY.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5690; SHA256 `3c456c0b57c3da890083889d32c0830085b929b9b24902a0838b30222f09527a`.

````text
{
  "scope": "author-permitted bounded raw-token/artifact hashing ONLY; no tokenizer/model/operator execution",
  "historical_digest": "24054920794c549afc0c678f58f6f44a1ecc3228bcfa793aeacc9c8516fb86a2",
  "historical_digest_source": "7e61cb6:.agents/research/COMPREHENSIVE_MULTIMODAL_DESIGN_HANDOFF_2026-10-15.md:22312-22320",
  "historical_serialization": "unspecified in recovered text; surviving cpu_context_step consumes {raw_tokens,count} JSON",
  "cases": [
    {
      "kind": "existing bounded raw-token artifact",
      "path": ".agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.u32le",
      "bytes": 32768,
      "sha256": "44c93512cadf90f5ace94e0baa6b55c70e9c5ff10dbae871c8ee0fb57dffdd7d",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "existing bounded raw-token artifact",
      "path": ".agents/runs/numerics-and-context-root-20261020/fresh-mixed-v1-8192.json",
      "bytes": 42944,
      "sha256": "1f079b0cbfe73c634be8f324cd386627c894601e85ce091a1e4516ee4567906d",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "compact",
      "newline": false,
      "bytes": 34780,
      "sha256": "fe8199d1d8e067ac3dd83e54656d64883ddc99fa2272c3f6c4ca1ce8dd77b164",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "compact",
      "newline": true,
      "bytes": 34781,
      "sha256": "f20e748a1eadc29d2ea3ed09e51b232eae920edf3853695878419771a2e46181",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "default",
      "newline": false,
      "bytes": 42974,
      "sha256": "1c384dfa5caa28f37fb8ff4ed367033b14162379c9e942ed4ffa026bb616fde1",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "default",
      "newline": true,
      "bytes": 42975,
      "sha256": "45abe21f457571cb3d0bdb79aad45ece898fbd0841618a398c91ea54edf65f8c",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "pretty2",
      "newline": false,
      "bytes": 75752,
      "sha256": "3c6631c347a4b3c9e72929fcac8cbaf3f818a2ff4fd2622533d8cb6d9e6db3de",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "raw_tokens",
        "count"
      ],
      "json_mode": "pretty2",
      "newline": true,
      "bytes": 75753,
      "sha256": "8e7aee0e1b9c5ecd83556cf8dcef0fc981bb4d210fb45c4f0c02677ee8f13fd6",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "compact",
      "newline": false,
      "bytes": 34780,
      "sha256": "c02a13cf08c967cea35d2c759f36822bd71cd44a80cc60ba88000f14555ac88a",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "compact",
      "newline": true,
      "bytes": 34781,
      "sha256": "ad31479d103b6f1d5d06021e93ed93ce968ccf6ba3e2841320ae19be27fa175d",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "default",
      "newline": false,
      "bytes": 42974,
      "sha256": "8ec6d4bdb8e5a0a2a4514c5242174d09a1255bf9ecdf70000ec4041d5c03b250",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "default",
      "newline": true,
      "bytes": 42975,
      "sha256": "d0008f5cc8e6b6b10d700d3496edff72d25767aaff8d1396034f4ce4990c7e35",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "pretty2",
      "newline": false,
      "bytes": 75752,
      "sha256": "fad45678e40f0f7ae454cd57d8f1075b3800ab18ef545f2a3a39f2855850eebd",
      "matches_historical_artifact_digest": false
    },
    {
      "kind": "bounded candidate serialization of KNOWN FRESH IDs",
      "key_order": [
        "count",
        "raw_tokens"
      ],
      "json_mode": "pretty2",
      "newline": true,
      "bytes": 75753,
      "sha256": "aa50484d97246e719d2c536f0346eb1107cba54b7aea84290b097d8fb3c7a742",
      "matches_historical_artifact_digest": false
    }
  ],
  "matches": 0,
  "historical_raw_ids_recovered": false,
  "different_historical_token_sequence_proven": false,
  "remaining": "Recover original task/ids file/REFERENCE_TOKENS stderr or original generator+format. These bounded negative artifact checks do not prove different token sequences."
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/baseline/math.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13197; SHA256 `12c6c6a44db2a216a87a23542b7fc675586298045d2539e45ef5fa4d6e3c048f`.

````text
//! Deterministic CPU reference operations used to validate compiled regions.
//!
//! These are mathematical building blocks, NOT full Qwen GDN or Gemma MoE layer
//! implementations. Model recipes will register exact higher-level semantics.

use crate::{Error, Result};

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Malformed(message.into()))
    }
}

/// Reference row-major matrix vector product, using diagnostic f64 accumulation.
pub fn matvec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    check(
        rows.checked_mul(cols) == Some(matrix.len()),
        "matvec matrix shape",
    )?;
    check(input.len() == cols, "matvec input size")?;
    check(rows <= 16_000_000, "matvec output too large")?;
    let mut out = Vec::new();
    out.try_reserve_exact(rows)
        .map_err(|_| Error::Limit("matvec output allocation failed".into()))?;
    for row in matrix.chunks_exact(cols.max(1)).take(rows) {
        let acc = row
            .iter()
            .zip(input)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        out.push(acc as f32);
    }
    if cols == 0 {
        out.resize(rows, 0.0);
    }
    Ok(out)
}

/// RMS norm reference, without residual addition or model-specific postscaling.
pub fn rms_norm(values: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    check(
        values.len() == weight.len() && !values.is_empty(),
        "rms norm shape",
    )?;
    check(epsilon.is_finite() && epsilon > 0.0, "rms norm epsilon")?;
    let sum: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    let inv = (sum / values.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    Ok(values
        .iter()
        .zip(weight)
        .map(|(x, w)| (*x as f64 * inv * f64::from(*w)) as f32)
        .collect())
}

/// Stable probability normalization. Reject NaN and infinities instead of masking
/// an unknown model's attention or router semantics.
pub fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    check(!values.is_empty(), "softmax empty")?;
    check(
        values.iter().all(|x| x.is_finite()),
        "softmax requires finite logits",
    )?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = values.iter().map(|v| f64::from(*v - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    check(sum > 0.0 && sum.is_finite(), "softmax unstable")?;
    Ok(weights.into_iter().map(|v| (v / sum) as f32).collect())
}

/// Router top-k selection is separated from model-specific scoring,
/// renormalization, and shared-expert accumulation rules.
pub fn top_k(logits: &[f32], k: usize) -> Result<Vec<usize>> {
    check(k > 0 && k <= logits.len(), "top-k size")?;
    check(
        logits.iter().all(|x| x.is_finite()),
        "top-k requires finite logits",
    )?;
    let mut rank: Vec<usize> = (0..logits.len()).collect();
    rank.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    rank.truncate(k);
    Ok(rank)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn gelu_tanh(x: f32) -> f32 {
    let a = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (a * (x + 0.044715 * x * x * x)).tanh())
}

/// One causal attention query over keys/values at positions 0..T.
/// `window=None` is full attention; `window=Some(n)` keeps the last n
/// positions inclusive of query position. This is diagnostic CPU math.
pub fn attention_one(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_position: usize,
    window: Option<usize>,
) -> Result<Vec<f32>> {
    let d = query.len();
    check(
        d > 0 && keys.len() == values.len() && keys.len().is_multiple_of(d),
        "attention head geometry",
    )?;
    let tokens = keys.len() / d;
    check(
        tokens > 0 && query_position < tokens,
        "attention query position",
    )?;
    if let Some(n) = window {
        check(n > 0, "sliding attention window")?;
    }
    let start = window.map_or(0, |n| (query_position + 1).saturating_sub(n));
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = Vec::new();
    for t in start..=query_position {
        let v = keys[t * d..(t + 1) * d]
            .iter()
            .zip(query)
            .map(|(&k, &q)| f64::from(k) * f64::from(q))
            .sum::<f64>();
        scores.push((v * scale) as f32);
    }
    let probs = softmax(&scores)?;
    let mut out = vec![0.0; d];
    for (i, p) in probs.iter().enumerate() {
        let v = &values[(start + i) * d..(start + i + 1) * d];
        for (dst, &x) in out.iter_mut().zip(v) {
            *dst += x * p;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_norm() {
        assert_eq!(
            matvec(&[1., 2., 3., 4.], 2, 2, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        assert!(matvec(&[1., 2.], 2, 2, &[1., 2.]).is_err());
        assert_eq!(matvec(&[], 3, 0, &[]).unwrap(), [0., 0., 0.]);
        let norm = rms_norm(&[3., 4.], &[1., 1.], 1e-6).unwrap();
        assert!((norm[0] - 0.848528).abs() < 1e-5);
        assert!((norm[1] - 1.131_37).abs() < 1e-5);
    }

    #[test]
    fn stable_softmax_and_tie_routing() {
        let probs = softmax(&[1000., 1000., 999.]).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert_eq!(top_k(&[3., 4., 4., 3.], 2).unwrap(), [1, 2]);
        assert!(top_k(&[f32::NAN, 1.], 1).is_err());
        assert!(softmax(&[f32::NEG_INFINITY]).is_err());
        assert!((silu(0.0)).abs() < 1e-9);
        assert!((gelu_tanh(0.0)).abs() < 1e-9);
    }

    #[test]
    fn sliding_vs_global_attention_and_future_mask() {
        let q = [1.];
        let k = [0., 0., 0., 0.];
        let v = [1., 2., 3., 100.];
        let global = attention_one(&q, &k, &v, 2, None).unwrap();
        let sliding = attention_one(&q, &k, &v, 2, Some(2)).unwrap();
        assert!((global[0] - 2.0).abs() < 1e-6);
        assert!((sliding[0] - 2.5).abs() < 1e-6);
        assert!(attention_one(&q, &k, &v, 1, Some(0)).is_err());
    }
}

/// Deterministic unfiltered logits/log-probability audit. These are produced
/// *before* temperature, penalties, truncation or grammar; a sampler's
/// post-filter probabilities are not an equivalent numerical oracle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogitRank {
    pub id: u32,
    pub delta: f32,
    pub logprob: f64,
}

pub fn top_logit_ranks(logits: &[f32], n: usize) -> Result<Vec<LogitRank>> {
    if logits.is_empty() || n == 0 || n > 128 || logits.len() > u32::MAX as usize {
        return Err(Error::Malformed("invalid logits audit width/rank".into()));
    }
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed("nonfinite logits in audit".into()));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|x| f64::from(*x - max).exp())
        .sum::<f64>();
    let ln_sum = sum.ln();
    if !ln_sum.is_finite() {
        return Err(Error::Malformed("nonfinite logits normalization".into()));
    }
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&i, &j| logits[j].total_cmp(&logits[i]).then_with(|| i.cmp(&j)));
    Ok(indices
        .into_iter()
        .take(n)
        .map(|id| {
            let delta = logits[id] - max;
            LogitRank {
                id: id as u32,
                delta,
                logprob: f64::from(delta) - ln_sum,
            }
        })
        .collect())
}

#[cfg(test)]
mod logit_audit_tests {
    use super::*;
    #[test]
    fn test_raw_log_probs_pre_sampler() {
        let ranks = top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 3).unwrap();
        assert_eq!(ranks.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 0]);
        assert_eq!(ranks[0].delta, 0.0);
        assert_eq!(ranks[2].delta, -2.0);
        assert!(
            (ranks.iter().map(|r| r.logprob.exp()).sum::<f64>()
                + top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 4).unwrap()[3]
                    .logprob
                    .exp()
                - 1.0)
                .abs()
                < 1e-10
        );
        assert!(top_logit_ranks(&[0.0, f32::NAN], 2).is_err());
    }
}

#[cfg(test)]
mod long_context_reference_tests {
    use super::*;

    #[test]
    fn thirty_two_thousand_causal_attention_positions_vs_independent_double_oracle() {
        const CONTEXT: usize = 32_768;
        const WIDTH: usize = 4;
        let query = [0.4, -0.7, 0.2, 0.1];
        let keys = (0..=CONTEXT)
            .flat_map(|t| (0..WIDTH).map(move |i| (((t * 7 + i * 37) % 113) as f32 - 56.0) / 97.0))
            .collect::<Vec<_>>();
        let mut values = (0..=CONTEXT)
            .flat_map(|t| {
                (0..WIDTH).map(move |i| (((t * 13 + i * 17) % 199) as f32 - 99.0) / 201.0)
            })
            .collect::<Vec<_>>();
        // Future token must never contribute; sentinel catches mask bugs.
        values[CONTEXT * WIDTH..].fill(1_000_000.0);
        let actual = attention_one(&query, &keys, &values, CONTEXT - 1, None).unwrap();
        let window = attention_one(&query, &keys, &values, CONTEXT - 1, Some(1024)).unwrap();
        let independently_evaluate = |start: usize| {
            let mut scores = Vec::new();
            for t in start..CONTEXT {
                let score = query
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| f64::from(q) * f64::from(keys[t * WIDTH + i]))
                    .sum::<f64>()
                    / (WIDTH as f64).sqrt();
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            std::array::from_fn::<_, WIDTH, _>(|i| {
                (weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(values[(start + j) * WIDTH + i]))
                    .sum::<f64>()
                    / sum) as f32
            })
        };
        let global_oracle = independently_evaluate(0);
        let sliding_oracle = independently_evaluate(CONTEXT - 1024);
        for i in 0..WIDTH {
            assert!(
                (actual[i] - global_oracle[i]).abs() < 2e-6,
                "full attention reference mismatch in channel {i}"
            );
            assert!(
                (window[i] - sliding_oracle[i]).abs() < 2e-6,
                "sliding attention reference mismatch in channel {i}"
            );
        }
    }
}

/// GGML Haswell/AVX2 scalar-equivalent approximate exponential used in the
/// source execution mode's SIMD SiLU implementation. This is *not* the
/// mathematical exp reference: its FMA polynomial and float-bit exponent
/// construction are part of the requested source-numerical specialization.
///
/// The common finite model gate range is handled by the exact polynomial
/// branch. For other finite ranges, follow the original SIMD scaling branches;
/// NaN/Inf values are rejected by the model state boundary before this stage.
pub fn ggml_source_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return x.exp();
    }
    let r = f32::from_bits(0x4b40_0000);
    let z = x.mul_add(f32::from_bits(0x3fb8_aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(
        f32::from_bits(0x35bf_be8e),
        (-n).mul_add(f32::from_bits(0x3f31_7200), x),
    );
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = f32::from_bits(0x3c07_2010)
        .mul_add(b, f32::from_bits(0x3d2b_9f17))
        .mul_add(
            u,
            f32::from_bits(0x3e2a_af33).mul_add(b, f32::from_bits(0x3eff_fedb)),
        )
        .mul_add(u, f32::from_bits(0x3f7f_fff6) * b);
    if n.abs() <= 126.0 {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// Opt-in, model-recipe-specific source-compatible SiLU on f32. This matches
/// GGML's vector approximation, rather than silently redefining generic SiLU.
pub fn ggml_source_silu(x: f32) -> f32 {
    x / (1.0 + ggml_source_exp(0.0 - x))
}

#[cfg(test)]
mod ggml_source_silu_tests {
    use super::*;
    #[test]
    fn source_approx_is_bounded_and_retains_general_silu_behavior() {
        for integer in -100..=100 {
            let x = integer as f32 / 10.0;
            let y = ggml_source_silu(x);
            assert!(y.is_finite());
            assert!(
                (y - silu(x)).abs() <= 1e-5,
                "GGML source approximation unexpectedly divergent at {x}"
            );
        }
        assert_eq!(ggml_source_silu(0.0), 0.0);
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3132; SHA256 `d648156caaff3c354215927808b085e353aec6e4f753b3564c927ea5fa83f175`.

````text
"""ROOT-only source/transport guards; importing performs no arithmetic/process work."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
AV = HERE.parent / 'numerics-and-context-av-replay-20261020'
ALPHA = HERE.parent / 'numerics-and-context-2b-alpha-replay-20261020'
CASES = ('av', 'alpha')
CAP = 8 * 1024 * 1024

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path, cap=None):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and (cap is None or before.st_size <= cap), 'bounded regular pinned file')
        total, value = 0, hashlib.sha256()
        while raw := stream.read(1024 * 1024):
            total += len(raw)
            value.update(raw)
        after = os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
                and total == before.st_size, 'immutable whole FD')
        return {'bytes': total, 'sha256': value.hexdigest()}

def read(path, cap=1024 * 1024):
    require(Path(path).stat().st_size <= cap, 'bounded JSON')
    with Path(path).open('rb') as stream:
        raw = stream.read(cap + 1)
    require(len(raw) <= cap, 'bounded JSON actual bytes')
    def unique(pairs):
        result = {}
        for key,value in pairs:
            require(key not in result, 'duplicate metadata key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)

def write(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    require(len(raw) <= 1024 * 1024, 'bounded report')
    with Path(path).open('xb') as stream:
        os.chmod(path, 0o600)
        stream.write(raw)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def frozen():
    freeze = read(HERE / 'freeze-manifest.json')
    for row in freeze['files']:
        require(digest(HERE / row['path'], 2 * 1024 * 1024) == row['digest'], 'exact new proposed source')
    for row in freeze['source_pins']:
        require(digest(row['path'], 2 * 1024 * 1024) == row['digest'], 'exact original source/metadata pin')
    return digest(HERE / 'freeze-manifest.json')['sha256']

def parents():
    """ROOT ONLY full authentic closure: can hash original model/DSO inputs."""
    av = load(AV / 'prepare.py', 'bulk_f32_original_av_prepare')
    alpha = load(ALPHA / 'prepare.py', 'bulk_f32_original_alpha_prepare')
    av_pins = av.jread(AV / 'pins.json')
    av.verify(av_pins)
    av.verify_fixtures(AV / 'root-fixtures', av_pins)
    alpha_pins = alpha.authenticate()
    alpha.capture_guards()
    alpha.verify_fixtures(ALPHA / 'root-fixtures', alpha_pins)
    return av,alpha

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/freeze-manifest.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11376; SHA256 `689221fe60b982e93e4684012b5c4bf8e3dc061546e86fc0ed4be7f63c78f61f`.

````text
{
  "schema": "zerogravity.closed-haswell-bulk-f32.source-freeze.v1",
  "status": "SOURCE_ONLY_UNCOMPILED_UNEXECUTED_PRIMITIVE_PROPOSAL",
  "files": [
    {
      "path": "Cargo.toml",
      "digest": {
        "bytes": 458,
        "sha256": "1a86c009f3a6e2d288b5bfa1678431e578e757fecdbce6b12777975232d4c210"
      }
    },
    {
      "path": "HISTORICAL-RECOVERY.json",
      "digest": {
        "bytes": 5690,
        "sha256": "3c456c0b57c3da890083889d32c0830085b929b9b24902a0838b30222f09527a"
      }
    },
    {
      "path": "NEW_DELTA.patch",
      "digest": {
        "bytes": 22333,
        "sha256": "e1460ed2f9a5db11f82a57cb456091d9072ef0cd87d782e2729171c2a8a03b29"
      }
    },
    {
      "path": "README.md",
      "digest": {
        "bytes": 7558,
        "sha256": "afa11f70d6b06d6838b23f2a05bca2e221f3a1d8b38fa080a7a3d5851a2dce90"
      }
    },
    {
      "path": "TODO.md",
      "digest": {
        "bytes": 1474,
        "sha256": "03c99e42754eb17860ec6cf7203736c7d654ff26b3c0d66233fae1961b91a63c"
      }
    },
    {
      "path": "baseline/math.rs",
      "digest": {
        "bytes": 13197,
        "sha256": "12c6c6a44db2a216a87a23542b7fc675586298045d2539e45ef5fa4d6e3c048f"
      }
    },
    {
      "path": "contracts.py",
      "digest": {
        "bytes": 3132,
        "sha256": "d648156caaff3c354215927808b085e353aec6e4f753b3564c927ea5fa83f175"
      }
    },
    {
      "path": "main.rs",
      "digest": {
        "bytes": 15344,
        "sha256": "c4e96a19bb2e44e8ca5991af31f0fcd66747730cfab62b60ccdf28c6255c1758"
      }
    },
    {
      "path": "owned_lifetime.py",
      "digest": {
        "bytes": 11840,
        "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
      }
    },
    {
      "path": "prepare.py",
      "digest": {
        "bytes": 6030,
        "sha256": "2797547b5cc5f8df21f445d75de114b84587a81fe39dfe5650e8e5bfce779a47"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs",
      "digest": {
        "bytes": 7696,
        "sha256": "dac2aed8e6f9e9f4a8c50a509811a4dffe44a35e961c50b87efa14884e50064a"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs",
      "digest": {
        "bytes": 13182,
        "sha256": "1f17327c62f36e8d813aab902a907a01233d84967a92ed84033e6ebc4a03ac13"
      }
    },
    {
      "path": "proposed/crates/zerogravity-cpu/src/math.rs",
      "digest": {
        "bytes": 13304,
        "sha256": "16b2a904fe208bbdeecc8de2c072c885ff12839041cb42b53675a50bb4d71526"
      }
    },
    {
      "path": "run_replay.py",
      "digest": {
        "bytes": 5557,
        "sha256": "e82614f3ecc80324e947fbeed431efde0f90dd6ba15616f3d5aed0c450b7f2ec"
      }
    },
    {
      "path": "test_contracts.py",
      "digest": {
        "bytes": 4834,
        "sha256": "a84e7590f87da81e40c339b87f6e0d71fd27a43c632083d327d884bf58430feb"
      }
    },
    {
      "path": "validate.py",
      "digest": {
        "bytes": 6074,
        "sha256": "abab362a92a7347f09d7f9ab8cf50da2ca4e6e9f120f0f56b7f88f7af2dcf5c3"
      }
    }
  ],
  "source_pins": [
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/Cargo.toml",
      "digest": {
        "bytes": 994,
        "sha256": "e9deb4ce0673d1a783b3b748ad5daec45a2e617f85b49fca0830e01c0a06daff"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/Cargo.lock",
      "digest": {
        "bytes": 139446,
        "sha256": "d46da60a19caa05aeeca07315a837709117b9a84e5bec980c4fb2591dd87d4ec"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/Cargo.toml",
      "digest": {
        "bytes": 463,
        "sha256": "6135312f01bc9eb12c395e4c0220b01ee5074ec20d681a6d5b1f53f7424170e5"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/lib.rs",
      "digest": {
        "bytes": 158,
        "sha256": "17e653a7fde1b47e17780d19a3af2a01f6c09d4ad9c2eba58b0bf80bd4a8fc7b"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/math.rs",
      "digest": {
        "bytes": 13197,
        "sha256": "12c6c6a44db2a216a87a23542b7fc675586298045d2539e45ef5fa4d6e3c048f"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-cpu/src/weights.rs",
      "digest": {
        "bytes": 41744,
        "sha256": "8282a25b231c5d126b5cba1100764919ca6fd2bcff503c08d5e6119e8dafc74a"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/accelerator.rs",
      "digest": {
        "bytes": 34440,
        "sha256": "50fcb42a211847bef44286c8e93e8e23cd3c48ee6dc6312cef446a25979a0998"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-core/src/resource_arbiter.rs",
      "digest": {
        "bytes": 29216,
        "sha256": "8e59d4d4a82b1e36f594573bac2a879df7008a2191b434f06f1ad151b75d9339"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/lib.rs",
      "digest": {
        "bytes": 89161,
        "sha256": "9a26849b304a8e9dfdd57478414c7845c6069f41a8d49c48742ec6debae22c4b"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity-model-qwen35/src/checkpoint.rs",
      "digest": {
        "bytes": 24080,
        "sha256": "dba8d20d034c16980812ec7616089a72ec6fbe4bbf0d055c05aea6c3350d03c4"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/LICENSE",
      "digest": {
        "bytes": 1078,
        "sha256": "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/llamafile/sgemm.cpp",
      "digest": {
        "bytes": 166263,
        "sha256": "8413c9a0828ddea7289aee72a32c03bd56a23ee40179ec3600f9b0589a21e338"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/vec.cpp",
      "digest": {
        "bytes": 25373,
        "sha256": "31e43ec66a1e2920a549dd2f71e6ce474e68f2fc298c4d2942404ca327e763d1"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/simd-mappings.h",
      "digest": {
        "bytes": 55790,
        "sha256": "a936298928e4dcfc1b61fb9e99cf8a152572b413e30f38dc8e7c3211ee9cb6a0"
      }
    },
    {
      "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/llama-cpp-v0.6.0-d812350/unpack/llama.cpp-0.6.0/ggml/src/ggml-cpu/ggml-cpu.c",
      "digest": {
        "bytes": 132440,
        "sha256": "45b22f8c83b2cbccabfc2f81a38ef8a5810223e1ce366b7c6f77d42faa5cc6f3"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/prepare.py",
      "digest": {
        "bytes": 7867,
        "sha256": "4de1b3a06523805628bc4fddf90fd9c0dac45fe25b24f5b0f5157731a5cd8607"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/root-fixtures/manifest.json",
      "digest": {
        "bytes": 230559,
        "sha256": "a4fbb5428164deedd9e3405557edb3b9fe1a2c4eff2bead97081ceaf3ec5aa94"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/root-validation.json",
      "digest": {
        "bytes": 19917,
        "sha256": "ecb713f8c989fcc8d6745664a285c4449cde489c1ac98f7528664e3b69e65586"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/pins.json",
      "digest": {
        "bytes": 57155,
        "sha256": "345765351ca36d927d4c57b56e0fc7f46cf9b6f38a50f169d15860e82390977f"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-av-replay-20261020/root-replay.invocation.json",
      "digest": {
        "bytes": 1524,
        "sha256": "5b383158ca0e6432966cabf6dd2c334e42f8b85b8e0db25349ba7b01b91b4233"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/prepare.py",
      "digest": {
        "bytes": 10742,
        "sha256": "5629603355d2e4593fce22f6273c5eab9eeffd63e83805adb658c7b66766b1af"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-fixtures/manifest.json",
      "digest": {
        "bytes": 193103,
        "sha256": "025a00e6803528a1d497fd069ba92155ea879f0e5dbe6748d501212b5a9d1354"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-validation.json",
      "digest": {
        "bytes": 20517,
        "sha256": "dcc47697478fe51906ab9c4d77cfed2459cfcd5b6d346bcb30c56a3a7f14817e"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/source-pins.json",
      "digest": {
        "bytes": 46227,
        "sha256": "bbbb612fd1c9dc5bf3caa893d5caedff6ff5ac20c4f63b68c627c22848910854"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/numerics-and-context-2b-alpha-replay-20261020/root-plain.invocation.json",
      "digest": {
        "bytes": 1892,
        "sha256": "694b5b14127880b66bf861eab3d671109f25b4541e164c33e3c3b37d73e00680"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-native-cpu-provider-envelope-20261020/startup-v1/owned_container_v4.py",
      "digest": {
        "bytes": 11840,
        "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
      }
    },
    {
      "path": "/home/pvl/spaces/abird/src/zerogravity/crates/zerogravity/examples/cpu_context_step.rs",
      "digest": {
        "bytes": 10212,
        "sha256": "1a68eaa93d3906272c484aa0db9707fc22390164021b247de0e4e0a7b00ddde8"
      }
    }
  ],
  "production_paths": [
    "crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs",
    "crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs",
    "crates/zerogravity-cpu/src/math.rs"
  ],
  "scope": "explicit private CPU tinyBLAS F32 primitive only; defaultfourVec8/model/state/session/GPU/profile unchanged",
  "author_validation": {
    "rustfmt": true,
    "python_AST_only": true,
    "bounded_source_metadata_hashes": true,
    "bounded_raw_token_artifact_hashes_only": 14,
    "tests_builds_operators_providers_models_GPU": false
  },
  "root_pending": [
    "independent full source review",
    "sourcefree tests/strict compilation/full isolated gate",
    "genuine captured defaultAV+alpha132096word exact replay/ledger/lifetime/cleanup",
    "separate explicit production schedule integration and BOTHmodel short/long fullV/all42 gates"
  ],
  "historical_recovery": "240549 artifact target recovered; encoding unknown;14bounded candidates0matches do not prove different token histories"
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 15344; SHA256 `c4e96a19bb2e44e8ca5991af31f0fcd66747730cfab62b60ccdf28c6255c1758`.

````text
//! ROOT-only captured primitive replay; no model/provider/session/token engine.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_cpu::math::haswell_bulk_f32::{
    BulkF32Admission, BulkF32Profile, BulkF32Request, F32View, execute,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const META: u64 = 4 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const LIMIT: u64 = 128 * 1024 * 1024;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Digest {
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    file: String,
    name: String,
    shape: [usize; 4],
    strides: [usize; 4],
    digest: Digest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case: String,
    model_sha256: String,
    left: Part,
    right: Part,
    expected: Part,
    source_provenance: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    status: String,
    profile: String,
    cases: Vec<Case>,
    arithmetic_executed: bool,
    expected_from_candidate: bool,
    gap_storage: String,
}
fn snapshot(arbiter: &ResourceArbiter, key: &PhysicalDeviceKey) -> Result<Value> {
    let s = arbiter.snapshot(key)?;
    Ok(
        json!({"reserved":s.reserved,"reservations":s.reservations,"fenced":s.fenced,
        "output":s.per_class.get(&AllocationClass::Output).copied().unwrap_or(0),
        "scratch":s.per_class.get(&AllocationClass::Scratch).copied().unwrap_or(0),
        "staging":s.per_class.get(&AllocationClass::TransferStaging).copied().unwrap_or(0),
        "metadata":s.per_class.get(&AllocationClass::Other).copied().unwrap_or(0)}),
    )
}
fn mxcsr() -> u32 {
    let mut value = 0u32;
    // SAFETY: read-only CPU-control instruction writes one valid local word.
    unsafe {
        std::arch::asm!("stmxcsr [{pointer}]", pointer = in(reg) &mut value,
        options(nostack, preserves_flags));
    }
    value
}
fn affinity() -> Result<Vec<usize>> {
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    need(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask) } == 0,
        "actual CPU mask",
    )?;
    let actual = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect::<Vec<_>>();
    need(actual.len() == 4, "ROOT exact four available CPUs")?;
    Ok(actual)
}
fn read_words(folder: &Path, part: &Part) -> Result<Vec<f32>> {
    need(
        Path::new(&part.file).components().count() == 1 && !part.file.starts_with('.'),
        "closed file component",
    )?;
    need(
        part.digest.bytes > 0
            && part.digest.bytes <= 1024 * 1024
            && part.digest.bytes.is_multiple_of(4)
            && part.digest.sha256.len() == 64,
        "bounded complete raw F32 file",
    )?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(folder.join(&part.file))?;
    let before = file.metadata()?;
    need(
        before.is_file() && before.len() == part.digest.bytes as u64,
        "actual exact raw length",
    )?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(part.digest.bytes)?;
    (&mut file)
        .take(part.digest.bytes as u64 + 1)
        .read_to_end(&mut raw)?;
    use std::os::unix::fs::MetadataExt;
    let after = file.metadata()?;
    need(
        raw.len() == part.digest.bytes
            && before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.len() == after.len()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.mtime() == after.mtime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.ctime() == after.ctime(),
        "same immutable input FD",
    )?;
    let mut words = Vec::new();
    words.try_reserve_exact(raw.len() / 4)?;
    for value in raw.chunks_exact(4) {
        let v = f32::from_le_bytes(value.try_into()?);
        need(
            v.is_finite(),
            "actual finite F32 words; no canonicalization",
        )?;
        words.push(v);
    }
    Ok(words)
}
fn expand(words: &[f32], part: &Part) -> Result<Vec<f32>> {
    let logical = part
        .shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v).ok_or("logical overflow"))?;
    need(words.len() == logical, "complete captured logical source")?;
    let span = part
        .shape
        .iter()
        .zip(part.strides)
        .try_fold(1usize, |n, (&v, stride)| {
            v.checked_sub(1)
                .and_then(|v| v.checked_mul(stride))
                .and_then(|v| n.checked_add(v))
                .ok_or("source span overflow")
        })?;
    need(
        span <= 8 * 1024 * 1024 && part.strides[0] == 1,
        "bounded original physical span",
    )?;
    let mut result = Vec::new();
    result.try_reserve_exact(span)?;
    result.resize(span, 0.0);
    let mut at = 0;
    for sequence in 0..part.shape[3] {
        for head in 0..part.shape[2] {
            for row in 0..part.shape[1] {
                let offset =
                    row * part.strides[1] + head * part.strides[2] + sequence * part.strides[3];
                result[offset..offset + part.shape[0]]
                    .copy_from_slice(&words[at..at + part.shape[0]]);
                at += part.shape[0];
            }
        }
    }
    Ok(result)
}
fn run(fixtures: &Path, output: &Path) -> Result<()> {
    let cpus = affinity()?;
    let fp_before = mxcsr();
    need(
        fp_before & 0xffc0 == 0x1f80,
        "RNE/masked/noFTZ/noDAZ source controls",
    )?;
    need(!output.exists(), "fresh ROOT output")?;
    std::fs::create_dir(output)?;
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", "bulk-f32-primitive-qualification")?;
    arbiter.register_matching(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("captured-default-GGUF-F32", "ROOT-only")?;
    let before = snapshot(&arbiter, &key)?;
    let metadata = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: META,
        }],
    )?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(fixtures.join("manifest.json"))?;
    need(
        file.metadata()?.len() <= 512 * 1024,
        "bounded two-case metadata",
    )?;
    let manifest: Manifest = serde_json::from_reader(file.take(512 * 1024 + 1))?;
    need(
        manifest.status == "AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY"
            && manifest.profile == "GgmlD812350HaswellTinyBlas"
            && !manifest.arithmetic_executed
            && !manifest.expected_from_candidate
            && manifest.cases.len() == 2
            && manifest.gap_storage
                == "actual dense logical V restored at original strides; uncaptured gaps zero-filled",
        "authentic byte-only input scope",
    )?;
    let mut reports = Vec::new();
    for (index, case) in manifest.cases.iter().enumerate() {
        let (id, model, a_shape, a_stride, b_shape, b_stride, out_shape, a_name, b_name) =
            if index == 0 {
                (
                    "av",
                    "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
                    [256, 256, 2, 1],
                    [1, 8448, 2162688, 4325376],
                    [256, 64, 8, 1],
                    [1, 256, 16384, 131072],
                    [256, 64, 8, 1],
                    "cache_v_l3 (view) (permuted)",
                    "kq_soft_max-3",
                )
            } else {
                (
                    "alpha",
                    "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
                    [2048, 16, 1, 1],
                    [1, 2048, 32768, 32768],
                    [2048, 64, 1, 1],
                    [1, 2048, 131072, 131072],
                    [16, 64, 1, 1],
                    "blk.0.ssm_alpha.weight",
                    "attn_norm-0",
                )
            };
        need(
            case.case == id
                && case.model_sha256 == model
                && case.left.shape == a_shape
                && case.left.strides == a_stride
                && case.right.shape == b_shape
                && case.right.strides == b_stride
                && case.expected.shape == out_shape
                && case.left.name == a_name
                && case.right.name == b_name
                && case.source_provenance.is_object(),
            "exact accepted original operand roles/geometry",
        )?;
        let staging = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: STAGING,
            }],
        )?;
        let dense_left = read_words(fixtures, &case.left)?;
        let left = expand(&dense_left, &case.left)?;
        drop(dense_left);
        let right = read_words(fixtures, &case.right)?;
        let source = SourceIdentity::for_new_allocation()?;
        let request = BulkF32Request {
            source,
            profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
            left: F32View {
                source,
                name: &case.left.name,
                values: &left,
                offset: 0,
                shape: a_shape,
                strides: a_stride,
            },
            right: F32View {
                source,
                name: &case.right.name,
                values: &right,
                offset: 0,
                shape: b_shape,
                strides: b_stride,
            },
        };
        let admission = || BulkF32Admission {
            arbiter: &arbiter,
            resource: &key,
            owner: &owner,
        };
        let baseline = arbiter.snapshot(&key)?.reserved;
        let flag = AtomicBool::new(true);
        need(
            execute(request, admission(), Some(&flag)).is_err()
                && arbiter.snapshot(&key)?.reserved == baseline,
            "actual pre-cancel unchanged ledger",
        )?;
        let output_bytes = out_shape.iter().product::<usize>() * 4;
        let pressure = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: LIMIT - baseline - (output_bytes as u64 + 256) + 1,
            }],
        )?;
        let pressure_live = snapshot(&arbiter, &key)?;
        let denied = execute(request, admission(), None);
        need(
            matches!(denied, Err(zerogravity_core::Error::Limit(ref reason))
            if reason == "physical device memory admission refused"),
            "actual original capacity refusal",
        )?;
        need(
            snapshot(&arbiter, &key)? == pressure_live,
            "no output charge after refusal",
        )?;
        drop(pressure);
        let result = execute(request, admission(), None)?;
        need(
            result.shape() == out_shape && result.source() == source,
            "same bound complete output",
        )?;
        let live = snapshot(&arbiter, &key)?;
        need(
            live["reserved"] == META + STAGING + output_bytes as u64 + 256
                && live["output"] == output_bytes as u64
                && live["scratch"] == 256,
            "actual aggregate payload+scratch reservations",
        )?;
        // Expected bytes are opened ONLY AFTER the public Rust math returns.
        let expected = read_words(fixtures, &case.expected)?;
        need(
            expected.len() == result.values().len(),
            "complete expected output cardinality",
        )?;
        let unequal = result
            .values()
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mut actual = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join(format!("{id}.actual.f32le")))?;
        for word in result.values() {
            actual.write_all(&word.to_le_bytes())?;
        }
        actual.sync_all()?;
        drop(actual);
        drop(expected);
        drop(left);
        drop(right);
        drop(staging);
        let retained = snapshot(&arbiter, &key)?;
        need(
            retained["reserved"] == META + output_bytes as u64 + 256,
            "output retains original ticket after input release",
        )?;
        drop(result);
        need(
            arbiter.snapshot(&key)?.reserved == META,
            "case private payloads reclaimed",
        )?;
        reports.push(json!({"case":id,"model_sha256":model,"words":output_bytes/4,"unequal_words":unequal,
            "pre_cancel_refused":true,"capacity_refused":true,"live":live,"retained_output":retained,
            "after_payload_drop":snapshot(&arbiter,&key)?,"physical_left_span_words":
                a_shape.iter().zip(a_stride).map(|(&n,s)| (n-1)*s).sum::<usize>()+1,
            "expected_opened_after_math":true}));
    }
    drop(metadata);
    let final_state = snapshot(&arbiter, &key)?;
    need(
        final_state["reserved"] == 0 && final_state["fenced"] == false,
        "actual all leases final0",
    )?;
    need(mxcsr() & 0xffc0 == 0x1f80, "FP controls unchanged")?;
    let bit_exact = reports.iter().all(|row| row["unequal_words"] == 0);
    let report = json!({"status":if bit_exact {"PASS_CAPTURED_DEFAULT_BULK_F32_ONLY"} else {"FAIL_SOURCE_BITS"},
        "profile":"GgmlD812350HaswellTinyBlas","before":before,"final":final_state,"cases":reports,
        "actual_affinity":cpus,"mxcsr_before":fp_before,"mxcsr_after":mxcsr(),"output_words":132096,
        "default_production_changed":false,"historical_prefix_recovered":false,"model_or_state_qualified":false,
        "source_memory_admission_scope":"configured diagnostic host, actual logical payload reservations; no whole physical RAM claim"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("report.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    need(
        bit_exact,
        "raw DEFAULT source output differences retained; no tolerance/fallback",
    )
}
fn main() {
    let args = std::env::args().map(PathBuf::from).collect::<Vec<_>>();
    let result = (|| -> Result<()> {
        need(args.len() == 3, "FIXTURES NEW_OUTPUT")?;
        run(&args[1], &args[2])
    })();
    if let Err(error) = result {
        eprintln!("BULK_F32_UNQUALIFIED:{error}");
        std::process::exit(2);
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/owned_lifetime.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11840; SHA256 `53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d`.

````text
"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.

Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
before leader reap: its reserved PID prevents accidental process-group reuse.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
CAP = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def command(argv, timeout=10):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
    return result


def leader_status_unreaped(process):
    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if observed is None:
        return None
    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
            "owned leader exit observation")
    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status


class Log:
    """Main-thread bounded drain; no thread/future can survive cleanup."""
    def __init__(self, pipe, path):
        self.pipe, self.path = pipe, Path(path)
        self.selector = self.file = None
        self.retained = self.total = 0
        self.eof = self.closed = False
        self.errors = []
        try:
            self.file = self.path.open("xb")
            os.chmod(self.path, 0o600)
            os.set_blocking(pipe.fileno(), False)
            self.selector = selectors.DefaultSelector()
            self.selector.register(pipe, selectors.EVENT_READ)
        except BaseException:
            if self.selector is not None:
                self.selector.close()
            if self.file is not None:
                self.file.close()
            raise

    def drain(self, timeout):
        for key, _ in self.selector.select(timeout):
            # One bounded block per wake; the exit/deadline remains observable
            # even when an owned writer continuously produces output.
            try:
                raw = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not raw:
                self.selector.unregister(key.fileobj)
                self.eof = True
                continue
            self.total += len(raw)
            keep = raw[:max(0, CAP - self.retained)]
            if keep:
                try:
                    self.file.write(keep)
                    self.retained += len(keep)
                except OSError as error:
                    self.errors.append(repr(error))

    def observe(self, process, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.drain(0.05)
            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
            status = leader_status_unreaped(process)
            if status is not None:
                return status
        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")

    def finish(self):
        failures = []
        try:
            deadline = time.monotonic() + 3
            while not self.eof and time.monotonic() < deadline:
                self.drain(0.05)
            require(self.eof, "owned stdout pipe EOF deadline")
            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException as error:
            failures.append("drain/evidence: " + repr(error))
        finally:
            # Close all three even when EOF, evidence I/O or one close fails.
            for name, close in (("selector", self.selector.close), ("log", self.file.close),
                                ("pipe", self.pipe.close)):
                try:
                    close()
                except BaseException as error:
                    failures.append(name + ": " + repr(error))
            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
        if failures:
            raise RuntimeError("owned log cleanup: " + "; ".join(failures))


def owned_container(name, nonce, cidfile):
    target, recorded_cid = name, None
    if cidfile.exists():
        cid = cidfile.read_text().strip()
        if cid:
            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
            target = recorded_cid = cid
    exists = command(PODMAN + ["container", "exists", target])
    if exists.returncode == 1:
        return None
    require(exists.returncode == 0, "owned container existence check failed")
    inspected = command(PODMAN + ["container", "inspect", target])
    require(inspected.returncode == 0, "owned container inspect failed")
    value, = json.loads(inspected.stdout)
    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
            "container name/nonce mismatch; no signal")
    cid = value["Id"]
    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
    if recorded_cid is not None:
        require(recorded_cid == cid, "owned CID changed; no signal")
    return cid


def cleanup_container(name, nonce, cidfile):
    errors = []
    try:
        cid = owned_container(name, nonce, cidfile)
    except BaseException as error:
        return ["container owner binding: " + repr(error)]
    if cid is None:
        return errors
    try:
        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
        if stopped.returncode != 0:
            errors.append("owned stop returned nonzero")
    except BaseException as error:
        errors.append("container stop: " + repr(error))
    try:
        # Fresh exact owner binding is required even after a failed/timedout
        # stop. A stop failure must not skip the bounded force-remove attempt.
        if owned_container(name, nonce, cidfile) is not None:
            removed = command(PODMAN + ["rm", "--force", cid], 15)
            require(removed.returncode == 0, "owned container force cleanup failed")
    except BaseException as error:
        errors.append("container force cleanup: " + repr(error))
    try:
        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
    except BaseException as error:
        errors.append("container final absence: " + repr(error))
    return errors


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group_unreaped(process, sig):
    # waitid authenticates that this is still our unreaped child. start_new_session
    # made it the session/group leader; no foreign group is inferred from a PID.
    leader_status_unreaped(process)
    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
            "owned leader session/PGID binding")
    os.killpg(process.pid, sig)


def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
    """Independent bounded stages; no nonzero signal after any reap attempt."""
    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
    errors = []
    try:
        signal_group_unreaped(process, signal.SIGTERM)
        state["term_sent"] = True
    except ProcessLookupError:
        state["term_group_absent_before_reap"] = True
    except BaseException as error:
        errors.append("owned TERM: " + repr(error))
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            observe(process)
            pause(0.05)
    except BaseException as error:
        errors.append("owned grace: " + repr(error))
    # Both independently authenticated KILL attempts precede ANY wait/reap.
    for attempt in range(2):
        state["kill_attempts_before_reap"] += 1
        try:
            signal_group_unreaped(process, signal.SIGKILL)
            state["kill_sent_before_reap"] = True
        except ProcessLookupError:
            state["kill_group_absent_before_reap"] = True
        except BaseException as error:
            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
    state["all_nonzero_signals_before_reap"] = True
    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
    try:
        state["leader_reap_attempts"] += 1
        state["exit_code"] = process.wait(timeout=5)
        state["leader_reaped"] = True
    except BaseException as error:
        errors.append("owned first leader reap: " + repr(error))
    finally:
        if process.returncode is None:
            try:
                state["leader_reap_attempts"] += 1
                state["exit_code"] = process.wait(timeout=5)
                state["leader_reaped"] = True
            except BaseException as error:
                errors.append("owned final leader reap: " + repr(error))
        else:
            state["exit_code"] = process.returncode
            state["leader_reaped"] = True
    try:
        require(state["leader_reaped"], "owned leader was not reaped")
        deadline = time.monotonic() + 5
        while group_exists(process.pid) and time.monotonic() < deadline:
            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
        state["group_absent"] = not group_exists(process.pid)
        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
    except BaseException as error:
        errors.append("owned group absence: " + repr(error))
    state["stage_errors"] = list(errors)
    if errors:
        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))


def cleanup(process, name, nonce, cidfile, receipt):
    errors = []
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("initial owned container cleanup: " + repr(error))
    try:
        cleanup_group(process, receipt)
    except BaseException as error:
        errors.append("owned foreground cleanup: " + repr(error))
    # Rediscover startup races using only the original nonce/name/CID binding.
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("final owned container cleanup: " + repr(error))
    try:
        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
        receipt["owned_container_absent_after_reap"] = True
    except BaseException as error:
        errors.append("final owned absence: " + repr(error))
    return errors

````

