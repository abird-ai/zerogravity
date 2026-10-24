# Paused continuation text — volume 5

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 20064; SHA256 `df242ea2dc34ea024b6b8bca80940dae2a0385c578001a80e4bd948c000c9b73`.

````text
//! ROOT-only genuine initial-query/prefix leaf replay; no model/session/exp.
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
use zerogravity_core::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_cpu::original_hf::initial_prelude::{
    GdnPreludePhase, InitialPreludeContract, InitialPreludeInput, InitialPreludeOperation,
    InitialPreludeOutput, execute_initial_prelude,
};
type Proof<T> = Result<T, Box<dyn Error>>;
const MIB: u64 = 1024 * 1024;
const CONTROL: u64 = 8 * MIB;
const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
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
            hi > lo && hi - lo == row["bytes"].as_u64().ok_or("stage bytes")? && hi - lo <= 24576,
            "bounded complete initial leaf stage",
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
                "selected original initial stage finite; preserve all bits",
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

fn run() -> Proof<Value> {
    ensure(!cfg!(debug_assertions), "release-only initial leaf replay")?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure(
        args.len() == 6 && args[5] == "27b",
        "bound initial-only replay arguments",
    )?;
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "initial-prefix-archive-replay")?;
    arbiter.register(key.clone(), 128 * MIB)?;
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("27b", "initial-prefix-leaves")?,
    };
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
        "actual metadata arguments",
    )?;
    let inputs: Value = serde_json::from_slice(&input_bytes)?;
    let reference: Value = serde_json::from_slice(&reference_bytes)?;
    let bindings: Value = serde_json::from_str(include_str!("source-bindings.json"))?;
    ensure(
        inputs["schema"] == "zg.initial-prefix.inputs.v1"
            && reference["schema"] == "zg.initial-prefix.reference.v1"
            && inputs["source_bindings_sha256"] == sha(include_bytes!("source-bindings.json"))
            && reference["source_bindings_sha256"] == inputs["source_bindings_sha256"]
            && reference["inputs_sha256"] == args[1]
            && inputs["heads"] == 48
            && inputs["teacher_token"] == 11
            && reference["heads"] == 48
            && reference["teacher_token"] == 11,
        "exact original initial teacher/phase/source binding",
    )?;
    let out = Path::new(&args[4]);
    std::fs::create_dir(out)?;
    let ready = factory::open_source(&host)?;
    let source = ready.source;
    let before = snapshot(&host)?;
    ensure(
        before.reserved == 72 * MIB && before.reservations == 2,
        "source64MiB +control8MiB before extraction",
    )?;
    let archive = Archive::new(&bindings["step0_archive"])?;
    let mut results = Vec::with_capacity(2);
    let mut checks = Vec::new();
    let mut negatives = Vec::new();
    let mut charged = 0u64;
    for (name, operation) in [
        ("query", InitialPreludeOperation::QueryScale128),
        ("prefix", InitialPreludeOperation::DecayPrefix64),
    ] {
        let input_row = &bindings["tensors"][format!("{name}_input")];
        let target_row = &bindings["tensors"][format!("{name}_output")];
        ensure(
            inputs["operands"][name] == *input_row && reference["targets"][name] == *target_row,
            "all exact input/target metadata",
        )?;
        let raw = archive.raw(input_row)?;
        let contract = InitialPreludeContract::new(
            source.source_identity(),
            PROFILE,
            GdnPreludePhase::InitialChunk64,
            operation,
            48,
        )?;
        let imported = InitialPreludeInput::from_f32le(contract, &raw)?;
        let prior = snapshot(&host)?;
        ensure(
            execute_initial_prelude(&source, imported, Some(&AtomicBool::new(true))).is_err(),
            "pre-cancel refuses",
        )?;
        same_ledger(&host, &prior)?;
        let foreign = InitialPreludeContract::new(
            SourceIdentity::for_new_allocation()?,
            PROFILE,
            GdnPreludePhase::InitialChunk64,
            operation,
            48,
        )?;
        ensure(
            execute_initial_prelude(
                &source,
                InitialPreludeInput::from_f32le(foreign, &raw)?,
                None,
            )
            .is_err(),
            "foreign issuer refuses",
        )?;
        ensure(
            InitialPreludeContract::new(
                source.source_identity(),
                PROFILE,
                GdnPreludePhase::Cached,
                operation,
                48,
            )
            .is_err(),
            "cached contract remains closed",
        )?;
        same_ledger(&host, &prior)?;
        let need = InitialPreludeOutput::required_bytes(contract) as u64;
        let mut hold = snapshot(&host)?
            .remaining
            .checked_sub(need - 1)
            .ok_or("pressure fit")?;
        let mut pressure = Vec::new();
        while hold > 0 {
            let n = hold.min(32 * MIB);
            pressure.push(source.reserve_workspace(n as usize)?);
            hold -= n;
        }
        let pressured = snapshot(&host)?;
        ensure(
            pressured.remaining == need - 1
                && execute_initial_prelude(&source, imported, None).is_err(),
            "real aggregate one-byte-short refusal",
        )?;
        same_ledger(&host, &pressured)?;
        drop(pressure);
        same_ledger(&host, &prior)?;
        let result = execute_initial_prelude(&source, imported, None)?;
        ensure(
            result.same_issuer(&source)
                && result.contract() == contract
                && result.charged_bytes() as u64 == need,
            "actual original source/phase/admission",
        )?;
        checks.push(compare(
            result.imported_input(),
            input_row,
            &archive,
            out,
            &format!("{name}.input"),
            true,
        )?);
        checks.push(compare(
            result.output(),
            target_row,
            &archive,
            out,
            &format!("{name}.output"),
            true,
        )?);
        charged += need;
        ensure(
            snapshot(&host)?.reserved == before.reserved + charged,
            "all actual private buffers retained",
        )?;
        results.push(result);
        negatives.push(json!({"operation":name,"pre_cancel":true,"foreign_issuer":true,"cached_refusal":true,"real_one_byte_short":true}));
    }
    write_json(&out.join("comparisons.json"), &json!({"checks":checks}))?;
    ensure(
        checks.len() == 4 && checks.iter().all(|v| v["different_words"] == 0),
        "all genuine query/prefix bits exact",
    )?;
    let live = snapshot(&host)?;
    ensure(
        class(&live, AllocationClass::Output) == 36864
            && class(&live, AllocationClass::TransferStaging) == 0
            && charged == 74240,
        "actual complete output/scratch payload charges",
    )?;
    let source_binding = json!({"identity":format!("{:?}",source.source_identity()),"bundle":source.verified_bundle_sha256(),"index":source.verified_shard_index_sha256()});
    let clones = results.clone();
    drop(results);
    drop(source);
    drop(ready.config);
    let retained = snapshot(&host)?;
    ensure(
        retained.reserved == before.reserved + charged,
        "last clones retain original source and all buffers",
    )?;
    let mut retained_checks = Vec::new();
    for (name, value) in ["query", "prefix"].into_iter().zip(&clones) {
        retained_checks.push(compare(
            value.imported_input(),
            &bindings["tensors"][format!("{name}_input")],
            &archive,
            out,
            &format!("{name}.input"),
            false,
        )?);
        retained_checks.push(compare(
            value.output(),
            &bindings["tensors"][format!("{name}_output")],
            &archive,
            out,
            &format!("{name}.output"),
            false,
        )?);
    }
    ensure(
        retained_checks == checks,
        "all18432 words exact after caller drop",
    )?;
    archive.close_proof()?;
    drop(clones);
    ensure(
        snapshot(&host)?.reserved == CONTROL,
        "source and backing retire on lastdrop",
    )?;
    drop(archive);
    drop(bindings);
    drop(inputs);
    drop(reference);
    drop(input_bytes);
    drop(reference_bytes);
    drop(control);
    let final_row = snapshot(&host)?;
    ensure(
        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
        "original host final0",
    )?;
    Ok(
        json!({"status":"PASS_PRIVATE_INITIAL_QUERY_PREFIX_ONLY","inputs_sha256":args[1],"reference_sha256":args[3],"source_binding":source_binding,
        "output_words":9216,"retained_input_words":9216,"checks":checks,"retained_checks":retained_checks,"negative":negatives,
        "steady_ledger":ledger(&before),"live_ledger":ledger(&live),"retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),
        "cached_DIV_exp_initial_core_state_model_GPU_or_performance_qualified":false}),
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

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/owned_lifetime.py

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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/owned_native.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5707; SHA256 `05577904b1915f08aa92dd319e5412f297965e60ce67bc855317e6181652aec2`.

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
              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "private archived original initial-query/prefix leaves only",
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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/prepare_reference.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5616; SHA256 `e03197d639ff993426c7616325a9b9f3fef9fba2d365ff1a511f63cb45cc135f`.

````text
"""ROOT-only metadata transport; no archive payload or expected math generated."""
import argparse
import json
from pathlib import Path
from contracts import HERE, frozen, pin, read_json, require, sha, write

def authenticated():
    b=read_json(HERE/'source-bindings.json')
    for k in ('capture_manifest','capture_validation','capture_invocation','capture_preparation',
              'source_plan','factory_report','step0_metadata'):
        pin(b[k]['path'],b[k])
    capture=read_json(b['capture_manifest']['path'])
    validation=read_json(b['capture_validation']['path'])
    invocation=read_json(b['capture_invocation']['path'])
    require(capture['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION'
        and capture['source_repo']==b['source']['repo']=='Qwen/Qwen3.8-27B'
        and capture['source_revision']==b['source']['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
        and capture['snapshot']==b['source']['snapshot']=='98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e'
        and capture['all_source_parameters_BF16'] is True and capture['tokens']==[11,353]
        and capture['full_model_StateRoot_native_exp_GPU_eligible'] is False,
        'genuine original source profile and two real teacher calls')
    require(validation['status']=='PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY'
        and validation['manifest_sha256']==b['capture_manifest']['sha256']
        and validation['source_plan_sha256']==b['source_plan']['sha256']
        and validation['invocation_sha256']==b['capture_invocation']['sha256']
        and validation['heads']==48 and validation['teachers']==[11,353], 'independent original archive acceptance')
    require(invocation['exit_code']==0 and invocation['cleanup_errors']==[]
        and invocation['source_FD_cleanup_errors']==[] and invocation['source_FDs_closed'] is True
        and invocation['owned_container_absent_after_reap'] is True
        and invocation['source_before']==invocation['source_after'], 'prior actual source/owner closure')
    require({k:capture['runtime'][k] for k in b['runtime']}==b['runtime'], 'pinned original runtime profile')
    s=capture['steps'][0]
    require(s['step']==b['step']==0 and s['teacher_token']==b['teacher_token']==11
        and s['plain_observed_all_outputs_and_state_exact'] is True
        and s['actual_calls']['torch_chunk_gated_delta_rule']==1
        and s['archive_bytes']==b['step0_archive']['bytes']
        and s['archive_sha256']==b['step0_archive']['sha256']
        and read_json(b['step0_metadata']['path'])==dict(s,plain_observed_all_outputs_and_state_exact=False),
        'actual initial chunk observation and full archive identity')
    expected={
        'query_input':('chunk.qk_normalized.query',[1,48,1,128],[6144,128,128,1],6144),
        'query_output':('chunk.query_scaled.query',[1,48,1,128],[6144,128,128,1],6144),
        'prefix_input':('chunk.cumulative_decay.decay',[1,48,1,64],[3072,64,64,1],3072),
        'prefix_output':('chunk.cumulative_decay.cum_decay',[1,48,1,64],[3072,64,64,1],3072),
    }
    require(set(b['tensors'])==set(expected), 'exact four original stage boundaries')
    for label,(name,shape,stride,words) in expected.items():
        row=b['tensors'][label];actual=dict(name=name,**s['tensors'][name])
        actual['original_torch_dtype']=actual['dtype'];actual['dtype']={'torch.float32':'F32'}[actual['dtype']]
        require(row==actual and row['name']==name and row['shape']==shape and row['stride']==stride
            and row['numel']==words and row['bytes']==words*4 and row['device']=='cpu'
            and row['contiguous'] is True, 'all actual source layout/dtype/raw fields: '+label)
    # Prefix input is an original padded view, not reconstructed from log-decay.
    require(s['tensors']['chunk.padded64.decay']['raw_sha256']==b['tensors']['prefix_input']['raw_sha256'],
        'actual padded decay byte identity')
    return b

def documents():
    b=authenticated();common=dict(teacher_token=11,heads=48,phase='InitialChunk64',
        source_bindings_sha256=sha(HERE/'source-bindings.json'),metadata_only=True)
    inputs=dict(common,schema='zg.initial-prefix.inputs.v1',operands={n:b['tensors'][n+'_input'] for n in ('query','prefix')})
    reference=dict(common,schema='zg.initial-prefix.reference.v1',targets={n:b['tensors'][n+'_output'] for n in ('query','prefix')},
        output_words=9216,retained_input_words=9216,no_expected_math_generated=True)
    return inputs,reference

def validate(directory):
    inputs,reference=documents()
    require(read_json(directory/'inputs.json')==inputs,'exact original operand metadata')
    reference['inputs_sha256']=sha(directory/'inputs.json')
    require(read_json(directory/'reference.json')==reference and {p.name for p in directory.iterdir()}=={'inputs.json','reference.json'},
        'exact original target metadata and tree')
    return dict(status='PASS_METADATA_BINDING_ONLY_RAW_BYTES_NOT_VALIDATED',inputs_sha256=sha(directory/'inputs.json'),reference_sha256=sha(directory/'reference.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-output',type=Path,required=True);a=p.parse_args()
    frozen();inputs,reference=documents();require(a.new_output.resolve().parent==HERE,'fresh own metadata directory')
    a.new_output.mkdir(mode=0o700);write(a.new_output/'inputs.json',inputs);reference['inputs_sha256']=sha(a.new_output/'inputs.json')
    write(a.new_output/'reference.json',reference);print(json.dumps(validate(a.new_output),sort_keys=True))
if __name__=='__main__': main()

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/run_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3891; SHA256 `197e5f51513125b0a62259a39ce71b04307e32160e1b841ed5329aab2e955f91`.

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
        require(report['status'] == 'PASS_PRIVATE_INITIAL_QUERY_PREFIX_ONLY' and
                report['output_words'] == 9216 and report['retained_input_words'] == 9216,
                'actual initial leaves and retained inputs require independent raw validation')
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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/sigterm_cancellation.py

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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/source-bindings.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10387; SHA256 `df9c0a7cf55f85feaeb85e6f5f4e10593556e51a06585e826bee28d6af490f6f`.

````text
{
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
  "factory_report": {
    "bytes": 7058,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source-pin.json",
    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
  },
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
  "scope": "External observed initial-only operands, no model producer or exp",
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
  "step": 0,
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
  "teacher_token": 11,
  "tensors": {
    "prefix_input": {
      "bytes": 12288,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.cumulative_decay.decay",
      "numel": 3072,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "7e0ab7f449cc2474452c07bd04b8a68d24091fc31d0eefc227f76a45ae6c8ffa",
      "shape": [
        1,
        48,
        1,
        64
      ],
      "stride": [
        3072,
        64,
        64,
        1
      ]
    },
    "prefix_output": {
      "bytes": 12288,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.cumulative_decay.cum_decay",
      "numel": 3072,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "763f02c1af3cca8dbfb638b0b414699005e55ad4bae89fc33ba2171759ee9814",
      "shape": [
        1,
        48,
        1,
        64
      ],
      "stride": [
        3072,
        64,
        64,
        1
      ]
    },
    "query_input": {
      "bytes": 24576,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.qk_normalized.query",
      "numel": 6144,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "06cc994dc69d4f2f605e0c04f63b9300bddb8bbe413b4efa1b2563af78f28d00",
      "shape": [
        1,
        48,
        1,
        128
      ],
      "stride": [
        6144,
        128,
        128,
        1
      ]
    },
    "query_output": {
      "bytes": 24576,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.query_scaled.query",
      "numel": 6144,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "588f23adece25456b3c1fa4d43de433aefde45cb1712948a3f4555e8a43fa102",
      "shape": [
        1,
        48,
        1,
        128
      ],
      "stride": [
        6144,
        128,
        128,
        1
      ]
    }
  }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/source-pin.json

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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/source_contract.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4947; SHA256 `a6780e9737b48c64f33611118a3f253906cf275de6039579db9e8161a4bfc2b8`.

````text
"""Explicit isolated ROOT source attestation; never compiler/numerical proof."""
from pathlib import Path
import hashlib
import tomllib


def frozen(here, read_json, pin, require, sha):
    run = here.parent
    manifest = read_json(run/'freeze-manifest.json')
    require(manifest['schema'] == 'zerogravity.initial-prefix.prepared-source.v1' and
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
    require(len(patch['files']) == 4, 'only CPU module wiring and three new leaf sources')
    rows = {r['file']:r for r in prior['files']}
    require(len(rows) == 521, 'unique original source paths')
    for row in patch['files']:
        require(row['file'] in config['owned_paths'], 'only explicitly owned source paths')
        rows[row['file']] = row
    require(len(rows) == 524, 'exact ROOT521 plus three new CPU sources')
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
    license_rel = 'crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE'
    auxiliary = read_json(here.parent/'owned-delta.json')['auxiliary_files']
    require(len(auxiliary) == 1 and auxiliary[0]['file'] == license_rel,
            'exact retained upstream attribution')
    pin(root/license_rel, auxiliary[0])
    result[str(root/license_rel)] = dict(path=str(root/license_rel),
        bytes=auxiliary[0]['bytes'],sha256=auxiliary[0]['sha256'])
    for name in ('Cargo.toml','main.rs','factory.rs','source-pin.json','source-bindings.json'):
        path = here/name
        result[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    lock = here/'Cargo.lock'
    require(lock.is_file() and not lock.is_symlink(), 'actual standalone dependency lock required')
    previous = tomllib.loads((root/'Cargo.lock').read_text())['package']
    known = {(p['name'],p['version'],p.get('source'),p.get('checksum')) for p in previous}
    packages = tomllib.loads(lock.read_text())['package']
    for p in packages:
        require(p['name'] == 'original27b-initial-prefix-replay' and p['version'] == '0.0.0'
                and 'source' not in p and 'checksum' not in p or
                (p['name'],p['version'],p.get('source'),p.get('checksum')) in known,
                'standalone external packages are exact parent-lock subset')
    result[str(lock)] = dict(path=str(lock), bytes=lock.stat().st_size,
                            sha256=hashlib.sha256(lock.read_bytes()).hexdigest())
    return root, result


def implementation(here, path, read_json, pin, require, sha):
    root, rows = expected(here, read_json, pin, require)
    value = read_json(path)
    require(value['status'] == 'ROOT_ISOLATED_SOURCE_ATTESTATION_AFTER_INITIAL_PREFIX_PATCH' and
            value['MAIN_merged'] is False and value['compiler_or_numerical_proof'] is False and
            value['workspace_root'] == str(root) and
            value['preparation_sha256'] == sha(here.parent/'freeze-manifest.json') and
            value['prior_ROOT521_source_sha256'] == '036ce0dc73bf0d1671d521059fac0b75c6aa62bc3e8b30475a576ae8d7d7e615' and
            value['files'] == [rows[k] for k in sorted(rows)],
            'complete explicit isolated build-source attestation; not MAIN or binary proof')
    return sha(path)

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/test_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1184; SHA256 `28faad90311dbdd1df445cbc0dae6d0988199555aaa0c9fd505a6d7ae7920d76`.

````text
"""ROOT-only metadata negatives; no provider/model/archive-payload import."""
import copy
import unittest
from contracts import HERE,read_json
from prepare_reference import documents
from sigterm_cancellation import SigtermCancellation

class PreparedMetadata(unittest.TestCase):
    def test_actual_target_documents_are_metadata_only(self):
        inputs,reference=documents()
        self.assertTrue(inputs['metadata_only']);self.assertTrue(reference['no_expected_math_generated'])
        self.assertEqual(sum(x['numel'] for x in inputs['operands'].values()),9216)
        self.assertEqual(sum(x['numel'] for x in reference['targets'].values()),9216)
    def test_raw_boundaries_are_distinct_original_arrays(self):
        b=read_json(HERE/'source-bindings.json')
        for role in ('query','prefix'):
            self.assertNotEqual(b['tensors'][role+'_input']['raw_sha256'],b['tensors'][role+'_output']['raw_sha256'])
    def test_signal_is_record_only_until_safe_checkpoint(self):
        guard=SigtermCancellation();guard._record(15,None);self.assertTrue(guard.received)
        with self.assertRaises(InterruptedError):guard.check()
if __name__=='__main__':unittest.main()

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/validate_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6640; SHA256 `865d81e6069c6787c24e3feb7d5beadf5a6eca2983c282b42514545fe6928a07`.

````text
"""ROOT independent raw-output/source/lease/owner validator; no expected math."""
import argparse
import json
import stat
from pathlib import Path
from contracts import HERE, RawArchive, frozen, implementation, read, read_json, require, sha, write
from prepare_reference import authenticated, validate as validate_metadata

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','reference-directory','implementation'): p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();require(a.output.resolve().parent==a.reference_directory.resolve().parent==HERE,'owned attempt/reference')
    freeze,build=frozen(),implementation(a.implementation);b=authenticated();metadata=validate_metadata(a.reference_directory)
    done=read_json(a.output/'completed.json')
    require(done['status']=='EXECUTED_REQUIRES_INDEPENDENT_RAW_VALIDATION' and done['freeze_manifest_sha256']==freeze
        and done['ROOT_build_source_attestation_sha256']==build and done['archived_reference_validation']==metadata
        and done['full_model_StateRoots_exp_initial_GDN_GPU_or_performance_qualified'] is False,'actual source/build/reference binding')
    d=a.output/'27b';life=a.output/'27b-lifecycle'
    for file,key in [('report.json','report_sha256'),('source-after.json','source_after_sha256')]: require(sha(d/file)==done[key],'bound actual receipt')
    require(sha(life/'invocation.json')==done['invocation_sha256'],'bound owned invocation')
    report=read_json(d/'report.json');inv=read_json(life/'invocation.json')
    require(report['status']=='PASS_PRIVATE_INITIAL_QUERY_PREFIX_ONLY' and report['inputs_sha256']==metadata['inputs_sha256']
        and report['reference_sha256']==metadata['reference_sha256']
        and type(report['output_words']) is int and report['output_words']==9216
        and type(report['retained_input_words']) is int and report['retained_input_words']==9216
        and report['cached_DIV_exp_initial_core_state_model_GPU_or_performance_qualified'] is False,'closed actual leaf scope/counts')
    require(inv['status']=='CHILD_EXECUTED_REQUIRES_VALIDATION' and inv['binary_sha256']==done['binary_sha256']
        and inv['freeze_manifest_sha256']==freeze and inv['primary_error'] is None and inv['cleanup_errors']==[]
        and inv['exit_code']==inv['reaped_exit_code']==0 and len(inv['cpu_affinity'])==len(set(inv['cpu_affinity']))==4
        and all(inv[k] is True for k in ('leader_reaped','group_absent','pipe_eof','selector_closed','pipe_closed','log_closed'))
        and inv['address_space_cap_bytes']==512*1024**2 and inv['wall_cap_seconds']==900,'actual finite owned child retired')
    source=read_json(d/'source-after.json')
    require(source['all_equal'] is True and source['source_before']==source['source_after']==inv['source_before']
        and len(source['source_after'])==31,'all original source file readsets unchanged')
    sb=report['source_binding'];require(sb['index']=='77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df'
        and sb['bundle']=='7e6a80bc4fc06b6237914007bde59a48a6e4fb0ea570bd5078b3101eb169faf2'
        and sb['identity'].startswith('SourceIdentity(') and sb['identity']!='SourceIdentity(0)','original source identity/index/bundle')
    checks=report['checks'];require(type(checks) is list and len(checks)==4 and report['retained_checks']==checks
        and read_json(d/'comparisons.json')=={'checks':checks},'complete retained raw comparisons')
    files={'report.json','source-after.json','comparisons.json'};words=0
    with RawArchive(b['step0_archive']) as archive:
        for record,label in zip(checks,('query.input','query.output','prefix.input','prefix.output')):
            row=b['tensors'][label.replace('.','_')];raw=archive.raw(row)
            require(record['reference']==row and record['label']==label and record['file']==label+'.f32le'
                and type(record['words']) is int and record['words']==len(raw)//4
                and type(record['bytes']) is int and record['bytes']==len(raw)
                and record['different_words']==0 and record['first_difference'] is None and record['max_abs']==0
                and read(d/record['file'],24576,record)==raw,'all signed original FP32 bits: '+label)
            files.add(record['file']);words+=record['words']
    require(words==18432 and {p.name for p in d.iterdir()}==files
        and {p.name for p in life.iterdir()}=={'execution.log','invocation.json'},'complete bounded artifact tree')
    steady,live,retained,final=[report[k+'_ledger'] for k in ('steady','live','retained','final')]
    mib=1024**2
    require(steady['reserved']==72*mib and steady['reservations']==2
        and steady['per_class'].get('Scratch')==64*mib and steady['per_class'].get('Other')==8*mib,
        'original source64+control8 before payload extraction')
    require(live==retained and live['reserved']==72*mib+74240 and live['per_class'].get('Output')==36864
        and live['per_class'].get('Scratch')==64*mib+37376 and live['per_class'].get('TransferStaging',0)==0
        and all(x['fenced'] is False for x in (steady,live,retained,final)),'actual input/output/scratch/source retention')
    require(final['reserved']==final['reservations']==0,'original final0')
    require(report['negative']==[dict(operation=n,pre_cancel=True,foreign_issuer=True,cached_refusal=True,real_one_byte_short=True) for n in ('query','prefix')],
        'actual both-operation refusal scopes')
    # Small complete tree: raw outputs73,728B plus bounded metadata/log.
    paths=list(a.output.rglob('*'));require(len(paths)<=32 and all(not x.is_symlink() and
        (stat.S_ISREG(x.lstat().st_mode) or stat.S_ISDIR(x.lstat().st_mode)) for x in paths)
        and sum(x.stat().st_size for x in paths if x.is_file())+65536<=8*mib,'final artifact envelope before validation write')
    require(frozen()==freeze and implementation(a.implementation)==build,'sources unchanged through independent validation')
    value=dict(status='PASS_PRIVATE_ORIGINAL27B_INITIAL_QUERY_PREFIX_ONLY',independent_output_words=9216,retained_input_words=9216,
        completed_sha256=sha(a.output/'completed.json'),freeze_manifest_sha256=freeze,
        cached_DIV_exp_initial_core_state_model_GPU_or_performance_qualified=False)
    require(len((json.dumps(value,indent=2,sort_keys=True)+'\n').encode())<=65536,'reserved final receipt extent')
    write(a.output/'validation.json',value)
    require(sum(x.stat().st_size for x in a.output.rglob('*') if x.is_file())<=8*mib,'post-finalization artifact bound')
    print(json.dumps(value,sort_keys=True))
if __name__=='__main__': main()

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/replay/write_implementation.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1015; SHA256 `462c3f4a6fcfb71b4cb8207eefe69cee32da7064da7e0be8451d22299fae9150`.

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
    value = dict(status='ROOT_ISOLATED_SOURCE_ATTESTATION_AFTER_INITIAL_PREFIX_PATCH',
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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/owned-delta.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1043; SHA256 `7af432ac3d718de0f41b0d82b65b15c009e31047d3f87370aba7a360ab49530e`.

````text
{
  "auxiliary_files": [
    {
      "bytes": 3464,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE",
      "sha256": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43"
    }
  ],
  "files": [
    {
      "bytes": 3578,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs",
      "sha256": "16145cf1ce0ee6d3e2336fcd06169097a5c70c60142811359e0a5819d11b9199"
    },
    {
      "bytes": 9988,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs",
      "sha256": "cfe6f7d1688a33251450bcfe075f166642a34e3c997d3e5b3a48fb8f0807c2f6"
    },
    {
      "bytes": 5561,
      "file": "crates/zerogravity-cpu/src/original_hf/initial_prelude.rs",
      "sha256": "2e8cf09b4c8cdfc1b6130abb3a7da9dea0ac3fb96a3d6ff27bac02db64c7b425"
    },
    {
      "bytes": 11227,
      "file": "crates/zerogravity-cpu/src/original_hf.rs",
      "sha256": "da11a4a82664c93797ed8dba0847302f364273c9ad546446f6e8fb1d5a6b6f5a"
    }
  ],
  "source_only": true
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/AccumulateType.h

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6111; SHA256 `f1defbdc257362abe5c4530baf558bb58880d8d5522b411da12b466cf6de7c08`.

````text
#pragma once
#include <ATen/Config.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>

// Defines the accumulation type for a scalar type.
// Example:
//   using accscalar_t = acc_type<scalar_t, /*is_cuda*/true>;
//
// Accumulation types are an important concept in numeric computing
// because you frequently want to perform intermediate computations
// at a higher precision than the input and output precision, to avoid
// compounding internal rounding errors.  Accumulation is the most
// well-known intermediate computation (it is of great importance for
// sum reduction and matrix multiply, for example), but in PyTorch
// acc_type ends up getting used for all sorts of other intermediate
// computations, so it perhaps would be more accurately (ahem) called an
// "accurate" type.  acc_type is especially important for reduced
// precision operations like float16 and bfloat16, where relatively
// benign looking inputs can easily end up overflowing/underflowing.
//
// acc_type is parametrized by whether or not you are running on CUDA
// or not, because on CUDA double precision operations are expensive
// and so by default, we don't actually want to use double as an
// acc_type on CUDA.  A lot of things are typed out below, but
// basically, the table is generated by a few rules:
//
//  If bool:
//      Use 'bool' as acc_type.
//  If floating point:
//      If CUDA, use 'float' as acc_type (unless scalar_t is double),
//      otherwise (CPU) use 'double'
//  If integral:
//      Use 'int64_t' as acc_type
//
// You're not forced to use this template; if you happen to know
// something specific about your use case, you can specify your own
// desired behavior.  This template, however, will give you a reasonable
// default that will work for all dtypes supported in PyTorch.

#if defined(__CUDACC__)
#include <cuda.h>
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

namespace at {

template <typename T, c10::DeviceType D>
struct AccumulateTypeDevice {};

template <typename T, bool>
struct AccumulateType {};

template <typename T>
struct AccumulateType<T, false> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CPU>::type;
};

template <typename T>
struct AccumulateType<T, true> {
  using type = typename AccumulateTypeDevice<T, c10::DeviceType::CUDA>::type;
};

template <typename T, c10::DeviceType device>
using acc_type_device = typename AccumulateTypeDevice<T, device>::type;

template <typename T, bool is_cuda>
using acc_type = typename AccumulateType<T, is_cuda>::type;

#define ACC_TYPE(t, acc_t, device_type)         \
  template <>                                   \
  struct AccumulateTypeDevice<t, device_type> { \
    using type = acc_t;                         \
  };
#define MPS_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::MPS)
#define XPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::XPU)
#define CUDA_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CUDA)
#define CPU_ACC_TYPE(t, acc_t) ACC_TYPE(t, acc_t, c10::DeviceType::CPU)

MPS_ACC_TYPE(BFloat16, float)
MPS_ACC_TYPE(Half, float)
MPS_ACC_TYPE(Float8_e5m2, float)
MPS_ACC_TYPE(Float8_e4m3fn, float)
MPS_ACC_TYPE(Float8_e5m2fnuz, float)
MPS_ACC_TYPE(Float8_e4m3fnuz, float)
MPS_ACC_TYPE(float, float)
MPS_ACC_TYPE(double, float)
MPS_ACC_TYPE(int8_t, int64_t)
MPS_ACC_TYPE(uint8_t, int64_t)
MPS_ACC_TYPE(char, int64_t)
MPS_ACC_TYPE(int16_t, int64_t)
MPS_ACC_TYPE(int32_t, int64_t)
MPS_ACC_TYPE(int64_t, int64_t)
MPS_ACC_TYPE(bool, bool)
MPS_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
MPS_ACC_TYPE(c10::complex<float>, c10::complex<float>)
MPS_ACC_TYPE(c10::complex<double>, c10::complex<float>)

XPU_ACC_TYPE(BFloat16, float)
XPU_ACC_TYPE(Half, float)
XPU_ACC_TYPE(Float8_e5m2, float)
XPU_ACC_TYPE(Float8_e4m3fn, float)
XPU_ACC_TYPE(Float8_e5m2fnuz, float)
XPU_ACC_TYPE(Float8_e4m3fnuz, float)
XPU_ACC_TYPE(float, float)
XPU_ACC_TYPE(double, double)
XPU_ACC_TYPE(int8_t, int64_t)
XPU_ACC_TYPE(uint8_t, int64_t)
XPU_ACC_TYPE(char, int64_t)
XPU_ACC_TYPE(int16_t, int64_t)
XPU_ACC_TYPE(int32_t, int64_t)
XPU_ACC_TYPE(int64_t, int64_t)
XPU_ACC_TYPE(bool, bool)
XPU_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
XPU_ACC_TYPE(c10::complex<float>, c10::complex<float>)
XPU_ACC_TYPE(c10::complex<double>, c10::complex<double>)

#if defined(__CUDACC__) || defined(__HIPCC__)
CUDA_ACC_TYPE(half, float)
#endif
CUDA_ACC_TYPE(BFloat16, float)
CUDA_ACC_TYPE(Half, float)
CUDA_ACC_TYPE(Float8_e5m2, float)
CUDA_ACC_TYPE(Float8_e4m3fn, float)
CUDA_ACC_TYPE(Float8_e5m2fnuz, float)
CUDA_ACC_TYPE(Float8_e4m3fnuz, float)
CUDA_ACC_TYPE(float, float)
CUDA_ACC_TYPE(double, double)
CUDA_ACC_TYPE(int8_t, int64_t)
CUDA_ACC_TYPE(uint8_t, int64_t)
CUDA_ACC_TYPE(char, int64_t)
CUDA_ACC_TYPE(int16_t, int64_t)
CUDA_ACC_TYPE(int32_t, int64_t)
CUDA_ACC_TYPE(int64_t, int64_t)
CUDA_ACC_TYPE(bool, bool)
CUDA_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
CUDA_ACC_TYPE(c10::complex<float>, c10::complex<float>)
CUDA_ACC_TYPE(c10::complex<double>, c10::complex<double>)

CPU_ACC_TYPE(BFloat16, float)
CPU_ACC_TYPE(Half, float)
CPU_ACC_TYPE(Float8_e5m2, float)
CPU_ACC_TYPE(Float8_e4m3fn, float)
CPU_ACC_TYPE(Float8_e5m2fnuz, float)
CPU_ACC_TYPE(Float8_e4m3fnuz, float)
CPU_ACC_TYPE(float, double)
CPU_ACC_TYPE(double, double)
CPU_ACC_TYPE(int8_t, int64_t)
CPU_ACC_TYPE(uint8_t, int64_t)
CPU_ACC_TYPE(char, int64_t)
CPU_ACC_TYPE(int16_t, int64_t)
CPU_ACC_TYPE(int32_t, int64_t)
CPU_ACC_TYPE(int64_t, int64_t)
CPU_ACC_TYPE(bool, bool)
CPU_ACC_TYPE(c10::complex<Half>, c10::complex<float>)
CPU_ACC_TYPE(c10::complex<float>, c10::complex<double>)
CPU_ACC_TYPE(c10::complex<double>, c10::complex<double>)

TORCH_API c10::ScalarType toAccumulateType(
    c10::ScalarType type,
    c10::DeviceType device);
TORCH_API c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda);

} // namespace at

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/ReduceOpsKernel.cpp

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 20804; SHA256 `4b85f8c711278cea2066f2a3418fd804cb4072fecfb2c3f3ce549a2c37e2b271`.

````text
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <algorithm>

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/cpu/Reduce.h>
#include <ATen/native/cpu/LogAddExp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/imag.h>
#endif

#include <c10/util/irange.h>
#include <ATen/AccumulateType.h>

namespace at::native { namespace {

using namespace vec;

template <typename scalar_t, typename func_t>
inline void cpu_cum_base_kernel(const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const func_t& f,
    scalar_t init_val) {
  if (result.sizes() != self.sizes()) {
    at::native::resize_output(result, self.sizes());
  }
  if (self.numel() == 0) {
    return;
  }
  const auto input_ndim = self.dim();
  if (input_ndim == 0) {
    result.fill_(self);
    return;
  }

  // TODO This probably should be using at::native::make_reduction
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .resize_outputs(false)
    .declare_static_shape(self.sizes(), /*squash_dims=*/dim)
    .add_output(result)
    .add_const_input(self)
    .build();

  auto result_dim_stride = ensure_nonempty_stride(result, dim);
  auto self_dim_stride = ensure_nonempty_stride(self, dim);

  auto loop = [&](char** data, const int64_t* strides, int64_t n) {
    auto* result_data_bytes = data[0];
    const auto* self_data_bytes = data[1];

    for ([[maybe_unused]] const auto i : c10::irange(n)) {
      f((scalar_t*)result_data_bytes,
        result_dim_stride,
        (scalar_t*)self_data_bytes,
        self_dim_stride,
        init_val);
      result_data_bytes += strides[0];
      self_data_bytes += strides[1];
    }
  };

  int64_t grain_size = internal::GRAIN_SIZE / std::max(int64_t{1}, self.size(dim));
  iter.for_each(loop, grain_size);
}

void cumsum_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumsum_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          cum_number += self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 0
    );
  });
}

void cumprod_cpu_kernel(const Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, self.scalar_type(), "cumprod_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        // NOLINTNEXTLINE(bugprone-signed-char-misuse)
        auto cum_number = (at::acc_type<scalar_t, false>)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          cum_number *= self_data[i * self_dim_stride];
          result_data[i * result_dim_stride] = (scalar_t)cum_number;
        }
      }, /*init_val=*/ 1
    );
  });
}

void logcumsumexp_cpu_kernel(Tensor& result, const Tensor& self, int64_t dim) {
  auto wrap_dim = maybe_wrap_dim(dim, self.dim());
  int64_t self_dim_size = ensure_nonempty_size(self, wrap_dim);

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(), "logcumsumexp_out_cpu", [&] {
    cpu_cum_base_kernel<scalar_t>(result, self, wrap_dim, [&] (
      scalar_t* result_data, auto result_dim_stride,
      const scalar_t* self_data, auto self_dim_stride, scalar_t init_val) {
        using accscalar_t = at::acc_type<scalar_t, false>;
        auto cum_number = (accscalar_t)init_val;
        for (const auto i : c10::irange(self_dim_size)) {
          accscalar_t x = self_data[i * self_dim_stride];

          cum_number = _log_add_exp_helper(x, cum_number);
          result_data[i * result_dim_stride] = static_cast<scalar_t>(cum_number);
        }
      }, /*init_val=*/ -std::numeric_limits<scalar_t>::infinity()
    );
  });
}

void std_var_kernel_impl(TensorIterator& iter, double correction, bool take_sqrt) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(), "std_cpu", [&] {
    binary_kernel_reduce(
        iter,
        WelfordOps<
            scalar_t,
            double,
            int64_t,
            std::tuple<scalar_t, scalar_t>>{correction, take_sqrt},
        WelfordData<double, int64_t>());
  });
}

void prod_kernel_impl(TensorIterator& iter) {
  // Workaround for the error: '*' in boolean context, suggest '&&' instead
  if (iter.dtype() == ScalarType::Bool) {
    using scalar_t = bool;
    binary_kernel_reduce_vec(
        iter,
        [=](scalar_t a, scalar_t b)
            -> scalar_t { return a && b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
            { return a && b; },
        /*ident=*/1);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, iter.dtype(), "prod_out_cpu", [&] {
      binary_kernel_reduce_vec(
          iter,
          [=](scalar_t a, scalar_t b)
              -> scalar_t { return a * b; },
          [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b)
              { return a * b; },
          /*ident=*/1);
    });
  }
}

template <typename scalar_t, typename acc_t>
inline void norm_two_reduce_step(Vectorized<acc_t>& acc_vec, Vectorized<scalar_t>& data_vec) {
  acc_vec += data_vec * data_vec;
}

template <>
inline void norm_two_reduce_step(Vectorized<float>& acc_fvec, Vectorized<BFloat16>& data_bvec) {
  auto [data_fvec0, data_fvec1] = convert_bfloat16_float(data_bvec);
  acc_fvec += data_fvec0 * data_fvec0;
  acc_fvec += data_fvec1 * data_fvec1;
}

template <typename scalar_t, typename out_t=typename scalar_value_type<scalar_t>::type>
void norm_kernel_cpu_impl(TensorIterator& iter, const double& val) {
  // This reduction accumulates results as the type `acc_t`.
  using acc_t = at::opmath_type<typename scalar_value_type<scalar_t>::type>;
  if (val == 0.0) {
    binary_kernel_reduce(iter, NormZeroOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == 1.0) {
    binary_kernel_reduce(iter, NormOneOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == 2.0) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == INFINITY) {
    binary_kernel_reduce(iter, AbsMaxOps<scalar_t, acc_t, out_t>(), acc_t(0));
  } else if (val == -INFINITY) {
    binary_kernel_reduce(iter, AbsMinOps<scalar_t, acc_t, out_t>(), std::numeric_limits<acc_t>::infinity());
  } else {
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t, out_t>{acc_t(val)}, acc_t(0));
  }
}

void norm_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    const Scalar& p) {
  double val = 0;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<double>();
  } else {
    TORCH_CHECK(false, "norm_kernel_cpu expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((val < 0) ? INFINITY : 0);
    return;
  }

  if (val == 2.0 && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    // If we can vectorize over the last dimension and the dtype
    // of the output is the same as that of the input,
    // then we go through the vectorised path.
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "norm_cpu", [&] {
        // use float as accumulate type for BFloat16
        using acc_t = at::opmath_type<scalar_t>;
        binary_kernel_reduce_lastdim(iter, [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
          scalar_t* result_data = (scalar_t*)result_data_bytes;
          scalar_t* self_data = (scalar_t*)self_data_bytes;

          using Vec = Vectorized<scalar_t>;
          using fVec = Vectorized<acc_t>;
          fVec acc_vec{acc_t(0)};
          acc_t buffer[fVec::size()];
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(self_data + d);
            norm_two_reduce_step(acc_vec, data_vec);
          }
          acc_vec.store(buffer);
          for (int j = 1; j < fVec::size(); j++) {
            buffer[0] = buffer[0] + buffer[j];
          }
          for (; d < size; d++) {
            acc_t data_val = acc_t(self_data[d]);
            buffer[0] += data_val * data_val;
          }
          result_data[0] = scalar_t(std::sqrt(buffer[0]));
        });
      });
  } else {
    if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
      // type promotion that does cast and reduction in a single kernel
      norm_kernel_cpu_impl<at::Half, float>(iter, val); return;
    } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
      // type promotion that does cast and reduction in a single kernel
      norm_kernel_cpu_impl<at::BFloat16, float>(iter, val); return;
    }

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kComplexHalf, iter.input_dtype(), "norm_cpu", [&] {
      norm_kernel_cpu_impl<scalar_t>(iter, val);
    });

    // For complex outputs, the above kernels do not touch the imaginary values,
    // so we must zero them out
    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

void and_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a && b) ? 1 : 0; },
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          // NB: != returns 0xFF rather than 0x01, so we must negate to get
          // the desired result
          return (a != Vectorized<uint8_t>(0)).neg() & (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/true);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a && b; },
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          // Adding the implementation here instead of in vec256_base to avoid
          // return value inconsistency. Other comparison operators in
          // vec256_base return -1/0 (all bit 1 / all bit 0) as true/false to
          // follow the AVX2 convention. This would be convenient when combined
          // with other vectorized operations. For example, one can use the
          // logical operation results as a mask for a bit operation to
          // retrieve/reset multiple elements in a vector.
          //
          // In this method, users would expect, e.g., all(), to return 1/0 as
          // true/false.
          Vectorized<bool> c = Vectorized<bool>();

          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] && b[i];
          }
          return c;
        },
        /*ident=*/true);
  }
}

void or_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == ScalarType::Byte) {
    // Refer [all, any : uint8 compatibility]
    binary_kernel_reduce_vec(
        iter,
        [=](uint8_t a, uint8_t b) -> uint8_t { return (a || b) ? 1 : 0; },
        [=](Vectorized<uint8_t> a, Vectorized<uint8_t> b) {
          return (a != Vectorized<uint8_t>(0)).neg() | (b != Vectorized<uint8_t>(0)).neg();
        },
        /*ident=*/false);
  } else {
    binary_kernel_reduce_vec(
        iter,
        [=](bool a, bool b) -> bool { return a || b; },
        [=](Vectorized<bool> a, Vectorized<bool> b) {
          Vectorized<bool> c = Vectorized<bool>();

          for (decltype(c.size()) i = 0; i != Vectorized<bool>::size(); i++) {
            c[i] = a[i] || b[i];
          }
          return c;
        },
        /*ident=*/false);
  }
}

template<typename scalar_t>
struct MinValuesOps: public at::native::MinOps<scalar_t> {
  using arg_t = typename MinOps<scalar_t>::arg_t;
  static scalar_t project(arg_t arg) {
    return arg.first;
  }
};

void min_values_kernel_impl(TensorIterator& iter) {
  if (iter.dtype() == kLong) {
    // This case is special because of Vectorized<int64_t> does not
    // handle upper_bound<int64_t>().
    // See: https://github.com/pytorch/pytorch/issues/43254
    using scalar_t = int64_t;
    binary_kernel_reduce(
      iter,
      MinValuesOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), -1));
    return;
  }
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "min_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return min_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return minimum(a, b); },
      static_cast<double>(upper_bound<scalar_t>()));
  });
}

void max_values_kernel_impl(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.dtype(), "max_values_cpu", [&iter] {
    binary_kernel_reduce_vec(
      iter,
      [](scalar_t a, scalar_t b) -> scalar_t { return max_impl(a, b); },
      [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return maximum(a, b); },
      lower_bound<scalar_t>());
  });
}

void argmax_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmax_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMaxOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(lower_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMaxOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(lower_bound<scalar_t>(), 0));
  });
}

void argmin_kernel_impl(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1), "argmin_cpu", [&] {
    if (is_reduce_lastdim(iter)) {
      using arg_t = std::pair<scalar_t, int64_t>;
      auto op = ArgMinOps<scalar_t>{};
      binary_kernel_reduce_lastdim(iter, [&](char* result_data_bytes, char* self_data_bytes, int64_t size) {
        int64_t* result_data = (int64_t*)result_data_bytes;
        scalar_t* self_data = (scalar_t*)self_data_bytes;

        arg_t acc = arg_t(upper_bound<scalar_t>(), 0);
        for (int64_t i = 0; i < size; i++) {
          acc = op.reduce(acc, self_data[i], i);
        }
        result_data[0] = acc.second;
      });
      return;
    }
    binary_kernel_reduce(
      iter,
      ArgMinOps<scalar_t>{},
      std::pair<scalar_t, int64_t>(upper_bound<scalar_t>(), 0));
  });
}

template <typename scalar_t, typename acc_t = uint64_t, typename out_t = acc_t>
struct XorSumOps {
  inline C10_DEVICE acc_t reduce(acc_t acc, scalar_t data, int64_t /*idx*/) const {
    if (std::is_same<scalar_t, bool>::value) {
      return acc ^ (data ? 1 : 0);
    } else if (
        std::is_same<scalar_t, float>::value ||
        std::is_same<scalar_t, double>::value ||
        std::is_same<scalar_t, at::BFloat16>::value ||
        std::is_same<scalar_t, at::Half>::value) {
      union {
        double d;
        uint64_t u;
      } converter;
      converter.d = static_cast<double>(data);
      return acc ^ converter.u;
    } else {
      return acc ^ static_cast<uint64_t>(data);
    }
  }

  inline C10_DEVICE acc_t combine(acc_t a, acc_t b) const {
    return a ^ b;
  }

  inline C10_DEVICE out_t project(acc_t a) const {
    return a;
  }

  static C10_DEVICE acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) {
    return acc;
  }
};

void xor_sum_kernel_impl(TensorIterator& iter) {
  // Use iter.dtype(1) to dispatch based on the type of the input tensor
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(1), "xor_sum_cpu", [&] {
        binary_kernel_reduce(
            iter, XorSumOps<scalar_t>(), static_cast<uint64_t>(0));
      });
}

// powsum: computes sum(|x|^p) without the final root
template <typename scalar_t, typename out_t=typename scalar_value_type<scalar_t>::type>
void powsum_kernel_cpu_impl(TensorIterator& iter, const double& val) {
  using acc_t = at::opmath_type<typename scalar_value_type<scalar_t>::type>;
  if (val == 2.0) {
    binary_kernel_reduce(iter, NormTwoOps<scalar_t, acc_t, out_t, false>(), acc_t(0));
  } else {
    binary_kernel_reduce(iter, NormOps<scalar_t, acc_t, out_t, false>{acc_t(val)}, acc_t(0));
  }
}

void powsum_kernel_tensor_iterator_impl(
    TensorIterator& iter,
    const Scalar& p) {
  double val = 0;
  if (p.isIntegral(false)) {
    val = p.to<int64_t>();
  } else if (p.isFloatingPoint()) {
    val = p.to<double>();
  } else {
    TORCH_CHECK(false, "powsum_kernel_cpu expects ord to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_(0);
    return;
  }

  if (val == 2.0 && is_reduce_lastdim(iter) &&
      iter.dtype(0) == iter.input_dtype() &&
      (iter.input_dtype() == kFloat || iter.input_dtype() == kDouble ||
       iter.input_dtype() == kBFloat16)) {
    // Vectorized path for L2 powsum (no sqrt at end)
    AT_DISPATCH_FLOATING_TYPES_AND(kBFloat16, iter.input_dtype(), "powsum_cpu", [&] {
        using acc_t = at::opmath_type<scalar_t>;
        binary_kernel_reduce_lastdim(iter, [](char* result_data_bytes, char* self_data_bytes, int64_t size) {
          scalar_t* result_data = (scalar_t*)result_data_bytes;
          scalar_t* self_data = (scalar_t*)self_data_bytes;

          using Vec = Vectorized<scalar_t>;
          using fVec = Vectorized<acc_t>;
          fVec acc_vec{acc_t(0)};
          acc_t buffer[fVec::size()];
          int64_t d = 0;
          for (; d < size - (size % Vec::size()); d += Vec::size()) {
            Vec data_vec = Vec::loadu(self_data + d);
            norm_two_reduce_step(acc_vec, data_vec);
          }
          acc_vec.store(buffer);
          for (int j = 1; j < fVec::size(); j++) {
            buffer[0] = buffer[0] + buffer[j];
          }
          for (; d < size; d++) {
            acc_t data_val = acc_t(self_data[d]);
            buffer[0] += data_val * data_val;
          }
          result_data[0] = scalar_t(buffer[0]);  // No sqrt!
        });
      });
  } else {
    if (iter.input_dtype() == kHalf && iter.dtype(0) == kFloat) {
      powsum_kernel_cpu_impl<at::Half, float>(iter, val); return;
    } else if (iter.input_dtype() == kBFloat16 && iter.dtype(0) == kFloat) {
      powsum_kernel_cpu_impl<at::BFloat16, float>(iter, val); return;
    }

    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kComplexHalf, iter.input_dtype(), "powsum_cpu", [&] {
      powsum_kernel_cpu_impl<scalar_t>(iter, val);
    });

    if (isComplexType(iter.output().scalar_type())) {
      at::imag(iter.output()).zero_();
    }
  }
}

}  // anonymous namespace

REGISTER_DISPATCH(std_var_stub, &std_var_kernel_impl)
REGISTER_DISPATCH(prod_stub, &prod_kernel_impl)
// mean implementation for CPU is in aten/src/ATen/native/ReduceOps.cpp
// but mean_stub must be defined for CPU as well
REGISTER_DISPATCH(mean_stub, nullptr)
REGISTER_DISPATCH(norm_stub, &norm_kernel_tensor_iterator_impl)
REGISTER_DISPATCH(powsum_stub, &powsum_kernel_tensor_iterator_impl)
REGISTER_DISPATCH(and_stub, &and_kernel_impl)
REGISTER_DISPATCH(or_stub, &or_kernel_impl)
REGISTER_DISPATCH(min_values_stub, &min_values_kernel_impl)
REGISTER_DISPATCH(max_values_stub, &max_values_kernel_impl)
REGISTER_DISPATCH(argmax_stub, &argmax_kernel_impl)
REGISTER_DISPATCH(argmin_stub, &argmin_kernel_impl)
REGISTER_DISPATCH(xor_sum_stub, &xor_sum_kernel_impl)

REGISTER_DISPATCH(cumprod_stub, &cumprod_cpu_kernel)
REGISTER_DISPATCH(cumsum_stub, &cumsum_cpu_kernel)
REGISTER_DISPATCH(logcumsumexp_stub, &logcumsumexp_cpu_kernel)

}  // namespace at::native

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/SOURCE.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1483; SHA256 `5df21e2106e935c9e931aa9fc1836e1f00dc8ebebcd41d375219be921b837a46`.

````text
{
  "files": [
    {
      "bytes": 20804,
      "file": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/ReduceOpsKernel.cpp",
      "sha256": "4b85f8c711278cea2066f2a3418fd804cb4072fecfb2c3f3ce549a2c37e2b271",
      "url": "https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/cpu/ReduceOpsKernel.cpp"
    },
    {
      "bytes": 6111,
      "file": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/AccumulateType.h",
      "sha256": "f1defbdc257362abe5c4530baf558bb58880d8d5522b411da12b466cf6de7c08",
      "url": "https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/AccumulateType.h"
    },
    {
      "bytes": 3464,
      "file": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-initial-prefix-20261020/pinned-source/LICENSE-PYTORCH",
      "sha256": "bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43",
      "url": "https://raw.githubusercontent.com/pytorch/pytorch/cf30153c4c131c8164ee7798e5022d810682e2cb/LICENSE"
    }
  ],
  "reviewed_schedule": {
    "accumulator": "CPU_ACC_TYPE(float,double)",
    "cumsum": "cpu_cum_base_kernel +cumsum_cpu_kernel, F64 carry/+0/F32 each store"
  },
  "revision": "cf30153c4c131c8164ee7798e5022d810682e2cb",
  "scope": "primary code only; no operator executed"
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3578; SHA256 `16145cf1ce0ee6d3e2336fcd06169097a5c70c60142811359e0a5819d11b9199`.

````text
//! Actual same-source host backing, not a producer/session/provider authority.
use super::*;
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};

/// External authenticated operator bytes. Caller proves archive/producer origin;
/// importing them never claims a current state, model projection or decay math.
#[derive(Clone, Copy)]
pub struct InitialPreludeInput<'a> {
    contract: InitialPreludeContract,
    raw: &'a [u8],
}
impl<'a> InitialPreludeInput<'a> {
    pub fn from_f32le(contract: InitialPreludeContract, raw: &'a [u8]) -> Result<Self> {
        if raw.len() != contract.elements() * 4
            || raw
                .chunks_exact(4)
                .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
        {
            return Err(Error::Malformed(
                "initial GDN external finite raw FP32".into(),
            ));
        }
        Ok(Self { contract, raw })
    }
    pub fn contract(self) -> InitialPreludeContract {
        self.contract
    }
}
struct Backing {
    contract: InitialPreludeContract,
    input: Vec<f32>,
    output: Vec<f32>,
    // Actual buffers retire before their admission/source owner.
    workspace: WindowWorkspace,
}
/// Clones retain the SAME source/backing/lease; no uncharged Vec escape.
#[derive(Clone)]
pub struct InitialPreludeOutput(Arc<Backing>);
impl InitialPreludeOutput {
    pub fn contract(&self) -> InitialPreludeContract {
        self.0.contract
    }
    pub fn imported_input(&self) -> &[f32] {
        &self.0.input
    }
    pub fn output(&self) -> &[f32] {
        &self.0.output
    }
    pub fn charged_bytes(&self) -> usize {
        self.0.workspace.bytes()
    }
    pub fn same_issuer(&self, source: &VerifiedTensorSource) -> bool {
        self.0.workspace.source().same_issuer(source)
    }
    pub fn required_bytes(contract: InitialPreludeContract) -> usize {
        contract.elements() * 8 + 256
    }
}

/// One pure synchronous leaf. All payload allocation follows aggregate admission.
/// No GPU/BLAS/provider path and no fallback; cancelled output is discarded.
pub fn execute_initial_prelude(
    source: &VerifiedTensorSource,
    input: InitialPreludeInput<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<InitialPreludeOutput> {
    not_cancelled(cancelled)?;
    let contract = input.contract;
    if contract.source_identity() != source.source_identity() {
        return Err(Error::State("initial GDN foreign raw source issuer".into()));
    }
    let bytes = contract.elements() * 4;
    let workspace = source.reserve_operator_buffers(bytes, bytes + 256)?;
    not_cancelled(cancelled)?;
    let mut imported = Vec::with_capacity(contract.elements());
    imported.extend(
        input
            .raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
    );
    let mut output = vec![0.0f32; contract.elements()];
    if imported.capacity() != contract.elements() || output.capacity() != contract.elements() {
        return Err(Error::Limit("initial GDN exact payload capacities".into()));
    }
    initial_prelude_into(contract, &imported, &mut output, cancelled)?;
    let result = InitialPreludeOutput(Arc::new(Backing {
        contract,
        input: imported,
        output,
        workspace,
    }));
    completed(result, cancelled)
}
pub(super) fn completed(
    result: InitialPreludeOutput,
    cancelled: Option<&AtomicBool>,
) -> Result<InitialPreludeOutput> {
    not_cancelled(cancelled)?;
    Ok(result)
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 9988; SHA256 `cfe6f7d1688a33251450bcfe075f166642a34e3c997d3e5b3a48fb8f0807c2f6`.

````text
use super::*;
fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
    InitialPreludeContract::new(
        SourceIdentity::for_new_allocation().unwrap(),
        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
        GdnPreludePhase::InitialChunk64,
        operation,
        heads,
    )
    .unwrap()
}
#[test]
fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
    let mut input = vec![0.0; c.elements()];
    input[0] = 1.0;
    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
    input[2] = -1.0;
    input[64] = -0.0;
    input[128] = -1.0;
    let mut output = vec![9.0; input.len()];
    initial_prelude_into(c, &input, &mut output, None).unwrap();
    assert_eq!(output[0].to_bits(), 0x3f80_0000);
    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
    assert_eq!(output[2].to_bits(), 0x3380_0000);
    assert_eq!(output[63].to_bits(), 0x3380_0000);
    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
    assert_eq!(output[127].to_bits(), 0);
    assert_eq!(output[128].to_bits(), 0xbf80_0000);
    assert_eq!(output[191].to_bits(), 0xbf80_0000);
}
#[test]
fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
    let c = contract(InitialPreludeOperation::QueryScale128, 48);
    let mut input = vec![1.0; c.elements()];
    input[..6].copy_from_slice(&[
        1.0,
        -1.0,
        0.0,
        -0.0,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
    ]);
    let mut out = vec![0.0; input.len()];
    initial_prelude_into(c, &input, &mut out, None).unwrap();
    assert_eq!(
        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
    );
    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
}
#[test]
fn cached_profile_source_geometry_and_complete_span_refuse() {
    for (source, profile, phase, heads) in [
        (
            SourceIdentity::unbound(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::GgufSourceV1,
            GdnPreludePhase::InitialChunk64,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::Cached,
            16,
        ),
        (
            SourceIdentity::for_new_allocation().unwrap(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            32,
        ),
    ] {
        assert!(
            InitialPreludeContract::new(
                source,
                profile,
                phase,
                InitialPreludeOperation::DecayPrefix64,
                heads
            )
            .is_err()
        );
    }
    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
    let mut input = vec![0.0; c.elements()];
    let mut out = vec![7.0; c.elements()];
    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
    input[0] = f32::NAN;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
    input[0] = f32::INFINITY;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
    assert!(out.iter().all(|v| *v == 7.0));
    input[0] = 0.0;
    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
    assert!(out.iter().all(|v| *v == 7.0));
    input[0] = f32::MAX;
    input[1] = f32::MAX;
    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod physical {
    use super::*;
    use std::fs::{self, File};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::AtomicU64;
    use zerogravity_artifacts::artifact::Artifact;
    use zerogravity_artifacts::cache::TransferAdmission;
    use zerogravity_artifacts::source_weights::{
        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
    };
    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
    const META: u64 = 16 * 1024 * 1024;
    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(1);
            let p = std::env::temp_dir().join(format!(
                "zg-initial-prefix-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&p).unwrap();
            fs::write(p.join("model.safetensors"), BYTES).unwrap();
            Self(p)
        }
        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
            SourceWeights::prepare_verified_windows(
                INDEX,
                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
                vec![
                    ExpectedSourceShard::new(
                        "model.safetensors".into(),
                        Artifact::from_file(
                            File::open(self.0.join("model.safetensors")).unwrap(),
                            Path::new("model.safetensors"),
                        )
                        .unwrap(),
                        BYTES.len() as u64,
                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
                    )
                    .unwrap(),
                ],
                WindowOpenLimits {
                    max_source_bytes: 1024 * 1024,
                    max_metadata_bytes: META,
                    max_shards: 1,
                    max_tensors: 6,
                    scan_chunk_bytes: 1024,
                },
                h,
            )
            .unwrap()
            .open()
            .unwrap()
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn host() -> TransferAdmission {
        let arbiter = ResourceArbiter::new();
        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
        arbiter
            .register(resource.clone(), 64 * 1024 * 1024)
            .unwrap();
        TransferAdmission {
            arbiter,
            resource,
            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
        }
    }
    fn reserved(h: &TransferAdmission) -> u64 {
        h.arbiter.snapshot(&h.resource).unwrap().reserved
    }
    fn raw(c: InitialPreludeContract) -> Vec<u8> {
        1.0f32.to_le_bytes().repeat(c.elements())
    }
    #[test]
    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
        let f = Fixture::new();
        let h = host();
        let source = f.source(&h);
        let foreign = f.source(&h);
        let c = InitialPreludeContract::new(
            source.source_identity(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            InitialPreludeOperation::QueryScale128,
            48,
        )
        .unwrap();
        let raw = raw(c);
        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
        let before = reserved(&h);
        assert!(execute_initial_prelude(&foreign, input, None).is_err());
        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
        assert_eq!(reserved(&h), before);
        let need = InitialPreludeOutput::required_bytes(c) as u64;
        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
        let pressure = source.reserve_workspace(hold as usize).unwrap();
        let pressured = reserved(&h);
        assert!(execute_initial_prelude(&source, input, None).is_err());
        assert_eq!(reserved(&h), pressured);
        drop(pressure);
        assert_eq!(reserved(&h), before);
        let output = execute_initial_prelude(&source, input, None).unwrap();
        let clone = output.clone();
        assert!(output.same_issuer(&source));
        assert!(!output.same_issuer(&foreign));
        assert_eq!(reserved(&h), before + need);
        assert_eq!(output.charged_bytes() as u64, need);
        drop(foreign);
        drop(source);
        assert_eq!(reserved(&h), META + need);
        assert_eq!(clone.imported_input().len(), 48 * 128);
        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
        drop(output);
        assert_eq!(reserved(&h), META + need);
        drop(clone);
        assert_eq!(reserved(&h), 0);
    }
    #[test]
    fn completed_result_cancel_drops_real_admitted_backing() {
        let f = Fixture::new();
        let h = host();
        let source = f.source(&h);
        let c = InitialPreludeContract::new(
            source.source_identity(),
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            GdnPreludePhase::InitialChunk64,
            InitialPreludeOperation::DecayPrefix64,
            16,
        )
        .unwrap();
        let raw = raw(c);
        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
        let value = execute_initial_prelude(&source, input, None).unwrap();
        assert!(reserved(&h) > META);
        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
        assert_eq!(reserved(&h), META);
        drop(source);
        assert_eq!(reserved(&h), 0);
    }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/proposed/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5561; SHA256 `2e8cf09b4c8cdfc1b6130abb3a7da9dea0ac3fb96a3d6ff27bac02db64c7b425`.

````text
//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};

/// Phase is explicit; cached query division is deliberately unsupported here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GdnPreludePhase {
    InitialChunk64,
    Cached,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InitialPreludeOperation {
    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
    QueryScale128,
    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
    DecayPrefix64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitialPreludeContract {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    operation: InitialPreludeOperation,
    heads: usize,
}
impl InitialPreludeContract {
    pub fn new(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
        phase: GdnPreludePhase,
        operation: InitialPreludeOperation,
        heads: usize,
    ) -> Result<Self> {
        if phase != GdnPreludePhase::InitialChunk64
            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
        {
            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
        }
        if !source.is_bound() || !matches!(heads, 16 | 48) {
            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
        }
        Ok(Self {
            source,
            profile,
            operation,
            heads,
        })
    }
    pub fn source_identity(self) -> SourceIdentity {
        self.source
    }
    pub fn numerical_profile(self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn phase(self) -> GdnPreludePhase {
        GdnPreludePhase::InitialChunk64
    }
    pub fn operation(self) -> InitialPreludeOperation {
        self.operation
    }
    pub fn heads(self) -> usize {
        self.heads
    }
    pub fn elements(self) -> usize {
        self.heads
            * match self.operation {
                InitialPreludeOperation::QueryScale128 => 128,
                InitialPreludeOperation::DecayPrefix64 => 64,
            }
    }
}
fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("initial GDN leaf cancelled".into()));
    }
    Ok(())
}
fn environment() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut mxcsr = 0u32;
        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
        // Passive read only; exception status bits may change during arithmetic.
        unsafe {
            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
                options(nostack, preserves_flags));
        }
        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
            return Err(Error::Unsupported(
                "initial GDN pinned CPU FP controls".into(),
            ));
        }
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(Error::Unsupported(
        "initial GDN pinned CPU architecture".into(),
    ))
}

/// No allocation/provider callback. Caller owns both complete, disjoint slices.
/// Output remains private on error; successful completion grants no state effect.
pub fn initial_prelude_into(
    contract: InitialPreludeContract,
    input: &[f32],
    output: &mut [f32],
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    not_cancelled(cancelled)?;
    environment()?;
    if input.len() != contract.elements()
        || output.len() != input.len()
        || input.iter().any(|v| !v.is_finite())
    {
        return Err(Error::Malformed(
            "initial GDN complete finite FP32 span".into(),
        ));
    }
    match contract.operation {
        InitialPreludeOperation::QueryScale128 => {
            // Source query * (128 ** -0.5), scalar converted once to FP32.
            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
                not_cancelled(cancelled)?;
                for (&value, result) in row.iter().zip(target) {
                    *result = value * SCALE;
                }
            }
        }
        InitialPreludeOperation::DecayPrefix64 => {
            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
                not_cancelled(cancelled)?;
                let mut carry = 0.0f64;
                for (&value, result) in row.iter().zip(target) {
                    carry += f64::from(value);
                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
                    if !result.is_finite() {
                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
                    }
                }
            }
        }
    }
    not_cancelled(cancelled)
}

#[cfg(all(feature = "cache", target_os = "linux"))]
mod admitted;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
#[cfg(test)]
mod tests;

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/proposed/crates/zerogravity-cpu/src/original_hf.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11227; SHA256 `da11a4a82664c93797ed8dba0847302f364273c9ad546446f6e8fb1d5a6b6f5a`.

````text
//! Original Hugging Face output-first source rows using the existing CPU math.
//! Torch linear matrices are [output,input] in row-major memory. No whole
//! learned matrix transpose or duplicate mathematical runtime is required.
//! This does not enable complete trained Qwen3.8 inference.

pub mod coefficient_preparation;
pub mod initial_prelude;
pub mod conv;
pub mod norm;
pub mod recurrent;
pub mod reduction;
pub mod transcendentals;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_embedding;
#[cfg(all(feature = "cache", target_os = "linux"))]
pub mod window_projection;

use crate::accelerator::{MatVecKernel, MatVecRegion, RegionExecution, validate_completed};
use crate::quant::QuantKind;
use crate::weights::{
    MAX_MATRIX_ROWS, SourceProjectionProfile, source_reference_projection_with_profile,
};
use crate::{Error, Result};
use zerogravity_core::source::SourceTensorRef;

/// Project one already source-qualified original [output,input] tensor
/// through the same real CPU kernel and optional physical provider.
pub fn output_first_projection(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
) -> Result<Vec<f32>> {
    output_first_projection_with_profile(
        source_id,
        tensor_name,
        tensor,
        input,
        provider,
        SourceProjectionProfile::Source,
    )
}

/// Closed CPU numerical profile; an unaware physical provider must refuse.
pub fn output_first_projection_with_profile(
    source_id: crate::accelerator::SourceIdentity,
    tensor_name: &str,
    tensor: SourceTensorRef<'_>,
    input: &[f32],
    provider: Option<&dyn MatVecKernel>,
    profile: SourceProjectionProfile,
) -> Result<Vec<f32>> {
    tensor.validate()?;
    if !source_id.is_bound()
        || tensor_name.is_empty()
        || tensor.shape.len() != 2
        || tensor.shape[1] != input.len() as u64
        || tensor.shape[0] > MAX_MATRIX_ROWS
        || input.is_empty()
        || !input.iter().all(|x| x.is_finite())
        || !matches!(
            tensor.kind,
            QuantKind::F32 | QuantKind::F16 | QuantKind::Bf16
        )
    {
        return Err(Error::Malformed(
            "original output-first source GEMV geometry/encoding invalid".into(),
        ));
    }
    let rows = usize::try_from(tensor.shape[0])
        .map_err(|_| Error::Limit("output-first matrix rows exceed address space".into()))?;
    let columns = usize::try_from(tensor.shape[1])
        .map_err(|_| Error::Limit("output-first matrix columns exceed address space".into()))?;
    let region = MatVecRegion {
        source_identity: source_id,
        tensor_name,
        kind: tensor.kind,
        columns,
        rows,
        packed_weights: tensor.packed_bytes,
        activation: input,
    };
    if let Some(provider) = provider {
        if profile != SourceProjectionProfile::Source {
            return Err(Error::Unsupported(
                "physical provider does not declare the CPU projection profile".into(),
            ));
        }
        match provider.try_execute(region)? {
            RegionExecution::Unsupported => {}
            RegionExecution::Completed(values) => {
                validate_completed(region, &values)?;
                return Ok(values);
            }
        }
    }
    source_reference_projection_with_profile(region, profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, RegionExecution, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct UnawareProvider(AtomicBool);
    impl MatVecKernel for UnawareProvider {
        fn try_execute(&self, region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Completed(vec![77.0; region.rows]))
        }
    }
    #[test]
    fn explicit_cpu_profile_refuses_unaware_provider_without_invoking_it() {
        let provider = UnawareProvider(AtomicBool::new(false));
        let packed = 0x3f80u16.to_le_bytes();
        let tensor = SourceTensorRef {
            kind: QuantKind::Bf16,
            shape: &[1, 1],
            packed_bytes: &packed,
        };
        let source = source_id();
        assert!(matches!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                Some(&provider),
                SourceProjectionProfile::TorchCpuBf16Avx2,
            ),
            Err(Error::Unsupported(_))
        ));
        assert!(!provider.0.load(Ordering::Relaxed));
        assert_eq!(
            output_first_projection_with_profile(
                source,
                "synthetic.weight",
                tensor,
                &[1.0],
                None,
                SourceProjectionProfile::TorchCpuBf16Avx2,
            )
            .unwrap(),
            [1.0]
        );
        assert_eq!(
            output_first_projection(source, "synthetic.weight", tensor, &[1.0], Some(&provider))
                .unwrap(),
            [77.0]
        );
        assert!(provider.0.load(Ordering::Relaxed));
    }

    fn source_id() -> SourceIdentity {
        SourceIdentity::for_new_allocation().unwrap()
    }
    fn projected(kind: QuantKind, shape: &[u64], bytes: &[u8], input: &[f32]) -> Result<Vec<f32>> {
        output_first_projection(
            source_id(),
            "original.linear.weight",
            SourceTensorRef {
                kind,
                shape,
                packed_bytes: bytes,
            },
            input,
            None,
        )
    }
    #[test]
    fn original_f32_hf_output_first_uses_same_reference_projection_without_transposition() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4., 5., 6.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        assert_eq!(
            projected(QuantKind::F32, &[2, 3], &bytes, &[7., 8., 9.]).unwrap(),
            [50., 122.]
        );
        assert!(projected(QuantKind::F32, &[3, 2], &bytes, &[7., 8., 9.]).is_err());
    }
    #[test]
    fn original_bf16_and_f16_rows_reuse_existing_raw_source_math() {
        let mut bf16 = Vec::new();
        for bits in [0x3f80u16, 0x4000, 0x4040, 0x4080] {
            bf16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::Bf16, &[2, 2], &bf16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        let mut f16 = Vec::new();
        for bits in [0x3c00u16, 0x4000, 0x4200, 0x4400] {
            f16.extend(bits.to_le_bytes());
        }
        assert_eq!(
            projected(QuantKind::F16, &[2, 2], &f16, &[5., 6.]).unwrap(),
            [17., 39.]
        );
    }
    struct UnsupportedProvider;
    impl MatVecKernel for UnsupportedProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Unsupported)
        }
    }
    struct BadProvider;
    impl MatVecKernel for BadProvider {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            Ok(RegionExecution::Completed(vec![f32::NAN]))
        }
    }
    #[test]
    fn optional_physical_unsupported_allows_cpu_but_invalid_completion_refuses() {
        let mut bytes = Vec::new();
        for x in [1., 2., 3., 4.] {
            bytes.extend_from_slice(&f32::to_le_bytes(x));
        }
        let tensor = SourceTensorRef {
            kind: QuantKind::F32,
            shape: &[2, 2],
            packed_bytes: &bytes,
        };
        let normal =
            output_first_projection(source_id(), "q.weight", tensor, &[5., 6.], None).unwrap();
        let fallback = output_first_projection(
            source_id(),
            "q.weight",
            tensor,
            &[5., 6.],
            Some(&UnsupportedProvider),
        )
        .unwrap();
        assert_eq!(normal, fallback);
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                tensor,
                &[5., 6.],
                Some(&BadProvider)
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                SourceIdentity::unbound(),
                "q.weight",
                tensor,
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(source_id(), "q.weight", tensor, &[f32::NAN, 6.], None)
                .is_err()
        );
        assert!(output_first_projection(source_id(), "q.weight", tensor, &[1.], None).is_err());
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    kind: QuantKind::Q4_0,
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        assert!(
            output_first_projection(
                source_id(),
                "q.weight",
                SourceTensorRef {
                    shape: &[2, 3],
                    ..tensor
                },
                &[5., 6.],
                None
            )
            .is_err()
        );
        // Original source bound MUST apply before a physical provider
        // is ever permitted to override the CPU's matrix safety cap.
        let large = vec![0u8; 2 * 1_000_001];
        let huge = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &large,
        };
        assert!(
            output_first_projection(
                source_id(),
                "oversized.hf.weight",
                huge,
                &[1.0],
                Some(&UnsupportedProvider)
            )
            .is_err()
        );
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::accelerator::{MatVecKernel, SourceIdentity};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ObservedProvider<'a>(&'a AtomicBool);
    impl MatVecKernel for ObservedProvider<'_> {
        fn try_execute(&self, _region: MatVecRegion<'_>) -> Result<RegionExecution> {
            self.0.store(true, Ordering::Relaxed);
            Ok(RegionExecution::Unsupported)
        }
    }

    #[test]
    fn oversize_hf_output_rows_refuse_before_accelerator_invocation() {
        let called = AtomicBool::new(false);
        let source = SourceIdentity::for_new_allocation().unwrap();
        let bytes = vec![0u8; 2 * 1_000_001];
        let tensor = SourceTensorRef {
            kind: QuantKind::F16,
            shape: &[1_000_001, 1],
            packed_bytes: &bytes,
        };
        let result = output_first_projection(
            source,
            "original.large.weight",
            tensor,
            &[1.0],
            Some(&ObservedProvider(&called)),
        );
        assert!(matches!(result, Err(Error::Malformed(_))));
        assert!(!called.load(Ordering::Relaxed));
    }
}

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 772; SHA256 `8a214e5c6e5b7f333ad741cb49b843bdf1966a2a1796a0b5500d19e5e20f23ab`.

````text
[package]
name = "original27b-initial-prefix-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[dependencies]
zerogravity-artifacts = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-artifacts", features = ["cache"] }
zerogravity-core = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-core" }
zerogravity-cpu = { path = "/home/pvl/spaces/abird/src/zerogravity-worktrees/root-bf16-composition-20261020/crates/zerogravity-cpu", features = ["cache"] }
serde_json = "=1.0.151"
sha2 = "=0.10.9"
libc = "=0.2.190"

[profile.release]
opt-level = 3
codegen-units = 1

[[bin]]
name = "original27b-initial-prefix-replay"
path = "main.rs"

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6287; SHA256 `4d544e248e71d61cb46c459679929184df9d3a3146c86effd3ad94feb2dc77f8`.

````text
"""ROOT-only bounded metadata/raw transport; no model/provider imports/calls."""
import hashlib
import json
import os
from pathlib import Path
import stat
import struct

HERE = Path(__file__).resolve().parent
INPUTS = ('query_input', 'prefix_input')
OUTPUTS = ('query_output', 'prefix_output')
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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/factory.rs

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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 20064; SHA256 `df242ea2dc34ea024b6b8bca80940dae2a0385c578001a80e4bd948c000c9b73`.

````text
//! ROOT-only genuine initial-query/prefix leaf replay; no model/session/exp.
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
use zerogravity_core::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_cpu::original_hf::initial_prelude::{
    GdnPreludePhase, InitialPreludeContract, InitialPreludeInput, InitialPreludeOperation,
    InitialPreludeOutput, execute_initial_prelude,
};
type Proof<T> = Result<T, Box<dyn Error>>;
const MIB: u64 = 1024 * 1024;
const CONTROL: u64 = 8 * MIB;
const PROFILE: ProgramNumericalProfile = ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1;
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
            hi > lo && hi - lo == row["bytes"].as_u64().ok_or("stage bytes")? && hi - lo <= 24576,
            "bounded complete initial leaf stage",
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
                "selected original initial stage finite; preserve all bits",
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

fn run() -> Proof<Value> {
    ensure(!cfg!(debug_assertions), "release-only initial leaf replay")?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    ensure(
        args.len() == 6 && args[5] == "27b",
        "bound initial-only replay arguments",
    )?;
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "initial-prefix-archive-replay")?;
    arbiter.register(key.clone(), 128 * MIB)?;
    let host = TransferAdmission {
        arbiter,
        resource: key,
        owner: ResourceOwner::new("27b", "initial-prefix-leaves")?,
    };
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
        "actual metadata arguments",
    )?;
    let inputs: Value = serde_json::from_slice(&input_bytes)?;
    let reference: Value = serde_json::from_slice(&reference_bytes)?;
    let bindings: Value = serde_json::from_str(include_str!("source-bindings.json"))?;
    ensure(
        inputs["schema"] == "zg.initial-prefix.inputs.v1"
            && reference["schema"] == "zg.initial-prefix.reference.v1"
            && inputs["source_bindings_sha256"] == sha(include_bytes!("source-bindings.json"))
            && reference["source_bindings_sha256"] == inputs["source_bindings_sha256"]
            && reference["inputs_sha256"] == args[1]
            && inputs["heads"] == 48
            && inputs["teacher_token"] == 11
            && reference["heads"] == 48
            && reference["teacher_token"] == 11,
        "exact original initial teacher/phase/source binding",
    )?;
    let out = Path::new(&args[4]);
    std::fs::create_dir(out)?;
    let ready = factory::open_source(&host)?;
    let source = ready.source;
    let before = snapshot(&host)?;
    ensure(
        before.reserved == 72 * MIB && before.reservations == 2,
        "source64MiB +control8MiB before extraction",
    )?;
    let archive = Archive::new(&bindings["step0_archive"])?;
    let mut results = Vec::with_capacity(2);
    let mut checks = Vec::new();
    let mut negatives = Vec::new();
    let mut charged = 0u64;
    for (name, operation) in [
        ("query", InitialPreludeOperation::QueryScale128),
        ("prefix", InitialPreludeOperation::DecayPrefix64),
    ] {
        let input_row = &bindings["tensors"][format!("{name}_input")];
        let target_row = &bindings["tensors"][format!("{name}_output")];
        ensure(
            inputs["operands"][name] == *input_row && reference["targets"][name] == *target_row,
            "all exact input/target metadata",
        )?;
        let raw = archive.raw(input_row)?;
        let contract = InitialPreludeContract::new(
            source.source_identity(),
            PROFILE,
            GdnPreludePhase::InitialChunk64,
            operation,
            48,
        )?;
        let imported = InitialPreludeInput::from_f32le(contract, &raw)?;
        let prior = snapshot(&host)?;
        ensure(
            execute_initial_prelude(&source, imported, Some(&AtomicBool::new(true))).is_err(),
            "pre-cancel refuses",
        )?;
        same_ledger(&host, &prior)?;
        let foreign = InitialPreludeContract::new(
            SourceIdentity::for_new_allocation()?,
            PROFILE,
            GdnPreludePhase::InitialChunk64,
            operation,
            48,
        )?;
        ensure(
            execute_initial_prelude(
                &source,
                InitialPreludeInput::from_f32le(foreign, &raw)?,
                None,
            )
            .is_err(),
            "foreign issuer refuses",
        )?;
        ensure(
            InitialPreludeContract::new(
                source.source_identity(),
                PROFILE,
                GdnPreludePhase::Cached,
                operation,
                48,
            )
            .is_err(),
            "cached contract remains closed",
        )?;
        same_ledger(&host, &prior)?;
        let need = InitialPreludeOutput::required_bytes(contract) as u64;
        let mut hold = snapshot(&host)?
            .remaining
            .checked_sub(need - 1)
            .ok_or("pressure fit")?;
        let mut pressure = Vec::new();
        while hold > 0 {
            let n = hold.min(32 * MIB);
            pressure.push(source.reserve_workspace(n as usize)?);
            hold -= n;
        }
        let pressured = snapshot(&host)?;
        ensure(
            pressured.remaining == need - 1
                && execute_initial_prelude(&source, imported, None).is_err(),
            "real aggregate one-byte-short refusal",
        )?;
        same_ledger(&host, &pressured)?;
        drop(pressure);
        same_ledger(&host, &prior)?;
        let result = execute_initial_prelude(&source, imported, None)?;
        ensure(
            result.same_issuer(&source)
                && result.contract() == contract
                && result.charged_bytes() as u64 == need,
            "actual original source/phase/admission",
        )?;
        checks.push(compare(
            result.imported_input(),
            input_row,
            &archive,
            out,
            &format!("{name}.input"),
            true,
        )?);
        checks.push(compare(
            result.output(),
            target_row,
            &archive,
            out,
            &format!("{name}.output"),
            true,
        )?);
        charged += need;
        ensure(
            snapshot(&host)?.reserved == before.reserved + charged,
            "all actual private buffers retained",
        )?;
        results.push(result);
        negatives.push(json!({"operation":name,"pre_cancel":true,"foreign_issuer":true,"cached_refusal":true,"real_one_byte_short":true}));
    }
    write_json(&out.join("comparisons.json"), &json!({"checks":checks}))?;
    ensure(
        checks.len() == 4 && checks.iter().all(|v| v["different_words"] == 0),
        "all genuine query/prefix bits exact",
    )?;
    let live = snapshot(&host)?;
    ensure(
        class(&live, AllocationClass::Output) == 36864
            && class(&live, AllocationClass::TransferStaging) == 0
            && charged == 74240,
        "actual complete output/scratch payload charges",
    )?;
    let source_binding = json!({"identity":format!("{:?}",source.source_identity()),"bundle":source.verified_bundle_sha256(),"index":source.verified_shard_index_sha256()});
    let clones = results.clone();
    drop(results);
    drop(source);
    drop(ready.config);
    let retained = snapshot(&host)?;
    ensure(
        retained.reserved == before.reserved + charged,
        "last clones retain original source and all buffers",
    )?;
    let mut retained_checks = Vec::new();
    for (name, value) in ["query", "prefix"].into_iter().zip(&clones) {
        retained_checks.push(compare(
            value.imported_input(),
            &bindings["tensors"][format!("{name}_input")],
            &archive,
            out,
            &format!("{name}.input"),
            false,
        )?);
        retained_checks.push(compare(
            value.output(),
            &bindings["tensors"][format!("{name}_output")],
            &archive,
            out,
            &format!("{name}.output"),
            false,
        )?);
    }
    ensure(
        retained_checks == checks,
        "all18432 words exact after caller drop",
    )?;
    archive.close_proof()?;
    drop(clones);
    ensure(
        snapshot(&host)?.reserved == CONTROL,
        "source and backing retire on lastdrop",
    )?;
    drop(archive);
    drop(bindings);
    drop(inputs);
    drop(reference);
    drop(input_bytes);
    drop(reference_bytes);
    drop(control);
    let final_row = snapshot(&host)?;
    ensure(
        final_row.reserved == 0 && final_row.reservations == 0 && !final_row.fenced,
        "original host final0",
    )?;
    Ok(
        json!({"status":"PASS_PRIVATE_INITIAL_QUERY_PREFIX_ONLY","inputs_sha256":args[1],"reference_sha256":args[3],"source_binding":source_binding,
        "output_words":9216,"retained_input_words":9216,"checks":checks,"retained_checks":retained_checks,"negative":negatives,
        "steady_ledger":ledger(&before),"live_ledger":ledger(&live),"retained_ledger":ledger(&retained),"final_ledger":ledger(&final_row),
        "cached_DIV_exp_initial_core_state_model_GPU_or_performance_qualified":false}),
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

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/owned_lifetime.py

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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/owned_native.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5707; SHA256 `05577904b1915f08aa92dd319e5412f297965e60ce67bc855317e6181652aec2`.

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
              "binary_sha256": sha(binary), "cpu_affinity": cpus, "scope": "private archived original initial-query/prefix leaves only",
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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/prepare_reference.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5616; SHA256 `e03197d639ff993426c7616325a9b9f3fef9fba2d365ff1a511f63cb45cc135f`.

````text
"""ROOT-only metadata transport; no archive payload or expected math generated."""
import argparse
import json
from pathlib import Path
from contracts import HERE, frozen, pin, read_json, require, sha, write

def authenticated():
    b=read_json(HERE/'source-bindings.json')
    for k in ('capture_manifest','capture_validation','capture_invocation','capture_preparation',
              'source_plan','factory_report','step0_metadata'):
        pin(b[k]['path'],b[k])
    capture=read_json(b['capture_manifest']['path'])
    validation=read_json(b['capture_validation']['path'])
    invocation=read_json(b['capture_invocation']['path'])
    require(capture['status']=='CAPTURED_GENUINE_ORIGINAL27B_LAYER0_GDN_ONLY_REQUIRES_VALIDATION'
        and capture['source_repo']==b['source']['repo']=='Qwen/Qwen3.8-27B'
        and capture['source_revision']==b['source']['revision']=='1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
        and capture['snapshot']==b['source']['snapshot']=='98cea13c169310196c4d8a113765f8fe58ab7a7961f0d1be69212c70dba8015e'
        and capture['all_source_parameters_BF16'] is True and capture['tokens']==[11,353]
        and capture['full_model_StateRoot_native_exp_GPU_eligible'] is False,
        'genuine original source profile and two real teacher calls')
    require(validation['status']=='PASS_GENUINE_ORIGINAL27B_LAYER0_GDN_REFERENCE_ONLY'
        and validation['manifest_sha256']==b['capture_manifest']['sha256']
        and validation['source_plan_sha256']==b['source_plan']['sha256']
        and validation['invocation_sha256']==b['capture_invocation']['sha256']
        and validation['heads']==48 and validation['teachers']==[11,353], 'independent original archive acceptance')
    require(invocation['exit_code']==0 and invocation['cleanup_errors']==[]
        and invocation['source_FD_cleanup_errors']==[] and invocation['source_FDs_closed'] is True
        and invocation['owned_container_absent_after_reap'] is True
        and invocation['source_before']==invocation['source_after'], 'prior actual source/owner closure')
    require({k:capture['runtime'][k] for k in b['runtime']}==b['runtime'], 'pinned original runtime profile')
    s=capture['steps'][0]
    require(s['step']==b['step']==0 and s['teacher_token']==b['teacher_token']==11
        and s['plain_observed_all_outputs_and_state_exact'] is True
        and s['actual_calls']['torch_chunk_gated_delta_rule']==1
        and s['archive_bytes']==b['step0_archive']['bytes']
        and s['archive_sha256']==b['step0_archive']['sha256']
        and read_json(b['step0_metadata']['path'])==dict(s,plain_observed_all_outputs_and_state_exact=False),
        'actual initial chunk observation and full archive identity')
    expected={
        'query_input':('chunk.qk_normalized.query',[1,48,1,128],[6144,128,128,1],6144),
        'query_output':('chunk.query_scaled.query',[1,48,1,128],[6144,128,128,1],6144),
        'prefix_input':('chunk.cumulative_decay.decay',[1,48,1,64],[3072,64,64,1],3072),
        'prefix_output':('chunk.cumulative_decay.cum_decay',[1,48,1,64],[3072,64,64,1],3072),
    }
    require(set(b['tensors'])==set(expected), 'exact four original stage boundaries')
    for label,(name,shape,stride,words) in expected.items():
        row=b['tensors'][label];actual=dict(name=name,**s['tensors'][name])
        actual['original_torch_dtype']=actual['dtype'];actual['dtype']={'torch.float32':'F32'}[actual['dtype']]
        require(row==actual and row['name']==name and row['shape']==shape and row['stride']==stride
            and row['numel']==words and row['bytes']==words*4 and row['device']=='cpu'
            and row['contiguous'] is True, 'all actual source layout/dtype/raw fields: '+label)
    # Prefix input is an original padded view, not reconstructed from log-decay.
    require(s['tensors']['chunk.padded64.decay']['raw_sha256']==b['tensors']['prefix_input']['raw_sha256'],
        'actual padded decay byte identity')
    return b

def documents():
    b=authenticated();common=dict(teacher_token=11,heads=48,phase='InitialChunk64',
        source_bindings_sha256=sha(HERE/'source-bindings.json'),metadata_only=True)
    inputs=dict(common,schema='zg.initial-prefix.inputs.v1',operands={n:b['tensors'][n+'_input'] for n in ('query','prefix')})
    reference=dict(common,schema='zg.initial-prefix.reference.v1',targets={n:b['tensors'][n+'_output'] for n in ('query','prefix')},
        output_words=9216,retained_input_words=9216,no_expected_math_generated=True)
    return inputs,reference

def validate(directory):
    inputs,reference=documents()
    require(read_json(directory/'inputs.json')==inputs,'exact original operand metadata')
    reference['inputs_sha256']=sha(directory/'inputs.json')
    require(read_json(directory/'reference.json')==reference and {p.name for p in directory.iterdir()}=={'inputs.json','reference.json'},
        'exact original target metadata and tree')
    return dict(status='PASS_METADATA_BINDING_ONLY_RAW_BYTES_NOT_VALIDATED',inputs_sha256=sha(directory/'inputs.json'),reference_sha256=sha(directory/'reference.json'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-output',type=Path,required=True);a=p.parse_args()
    frozen();inputs,reference=documents();require(a.new_output.resolve().parent==HERE,'fresh own metadata directory')
    a.new_output.mkdir(mode=0o700);write(a.new_output/'inputs.json',inputs);reference['inputs_sha256']=sha(a.new_output/'inputs.json')
    write(a.new_output/'reference.json',reference);print(json.dumps(validate(a.new_output),sort_keys=True))
if __name__=='__main__': main()

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/run_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 3891; SHA256 `197e5f51513125b0a62259a39ce71b04307e32160e1b841ed5329aab2e955f91`.

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
        require(report['status'] == 'PASS_PRIVATE_INITIAL_QUERY_PREFIX_ONLY' and
                report['output_words'] == 9216 and report['retained_input_words'] == 9216,
                'actual initial leaves and retained inputs require independent raw validation')
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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/sigterm_cancellation.py

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

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/replay/source-bindings.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 10387; SHA256 `df9c0a7cf55f85feaeb85e6f5f4e10593556e51a06585e826bee28d6af490f6f`.

````text
{
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
  "factory_report": {
    "bytes": 7058,
    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf-h48-coefficient-preparation-20261020/replay/source-pin.json",
    "sha256": "4c74fa28ed248eff33975de1b2b36742a7b089dd6e7a53ea60ee3c09d970866a"
  },
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
  "scope": "External observed initial-only operands, no model producer or exp",
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
  "step": 0,
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
  "teacher_token": 11,
  "tensors": {
    "prefix_input": {
      "bytes": 12288,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.cumulative_decay.decay",
      "numel": 3072,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "7e0ab7f449cc2474452c07bd04b8a68d24091fc31d0eefc227f76a45ae6c8ffa",
      "shape": [
        1,
        48,
        1,
        64
      ],
      "stride": [
        3072,
        64,
        64,
        1
      ]
    },
    "prefix_output": {
      "bytes": 12288,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.cumulative_decay.cum_decay",
      "numel": 3072,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "763f02c1af3cca8dbfb638b0b414699005e55ad4bae89fc33ba2171759ee9814",
      "shape": [
        1,
        48,
        1,
        64
      ],
      "stride": [
        3072,
        64,
        64,
        1
      ]
    },
    "query_input": {
      "bytes": 24576,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.qk_normalized.query",
      "numel": 6144,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "06cc994dc69d4f2f605e0c04f63b9300bddb8bbe413b4efa1b2563af78f28d00",
      "shape": [
        1,
        48,
        1,
        128
      ],
      "stride": [
        6144,
        128,
        128,
        1
      ]
    },
    "query_output": {
      "bytes": 24576,
      "contiguous": true,
      "device": "cpu",
      "dtype": "F32",
      "name": "chunk.query_scaled.query",
      "numel": 6144,
      "original_torch_dtype": "torch.float32",
      "raw_sha256": "588f23adece25456b3c1fa4d43de433aefde45cb1712948a3f4555e8a43fa102",
      "shape": [
        1,
        48,
        1,
        128
      ],
      "stride": [
        6144,
        128,
        128,
        1
      ]
    }
  }
}

````

