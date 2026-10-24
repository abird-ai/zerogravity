# Paused continuation text — volume 17

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 19992; SHA256 `b424884ef3f5e442bf9706ec9c60d0816b7fdc180f2bd38f56c66a74ea33bc55`.

````text
//! ROOT-only SAME engine pure matrix replay; no forward/session/token loop.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use zerogravity_core::accelerator::program::{BulkF32Layout, PlanOperation};
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_model_qwen35::operator_slice::bulk_f32::{
    BulkF32HostAdmission, BulkF32Slice, ExternalBulkF32,
};
use zerogravity_model_qwen35::{LoadedQwenModel, original_binding::QwenMatrixRole};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const META: u64 = 4 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const CACHE_METADATA: u64 = 16 * 1024 * 1024;

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
        "metadata":s.per_class.get(&AllocationClass::Other).copied().unwrap_or(0),
        "weights":s.per_class.get(&AllocationClass::ImmutableWeights).copied().unwrap_or(0)}),
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

struct DeniedProvider(AtomicUsize);
impl MatVecKernel for DeniedProvider {
    fn try_execute(&self, _region: MatVecRegion<'_>) -> zerogravity_core::Result<RegionExecution> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(RegionExecution::Unsupported)
    }
}
fn run(
    fixtures: &Path,
    output: &Path,
    models: &[(PathBuf, PathBuf); 2],
    teacher: &Path,
) -> Result<()> {
    let cpus = affinity()?;
    let fp_before = mxcsr();
    need(fp_before & 0xffc0 == 0x1f80, "actual RNE/noFTZ/noDAZ")?;
    need(!output.exists(), "fresh output")?;
    std::fs::create_dir(output)?;
    let raw = std::fs::read(teacher)?;
    need(raw.len() == 256, "actual64 original teacher IDs")?;
    let ids = raw
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect::<Vec<_>>();
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
        "authentic unchanged default reference scope",
    )?;
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", "same-engine-default-bulk-F32")?;
    arbiter.register_matching(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("original-GGUF-pure-matrix", "ROOT-qualification")?;
    let before = snapshot(&arbiter, &key)?;
    let meta = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: META,
        }],
    )?;
    let mut reports = Vec::new();
    for (index, case) in manifest.cases.iter().enumerate() {
        let id = if index == 0 { "av" } else { "alpha" };
        need(case.case == id, "two distinct ordered cases")?;
        need(
            case.model_sha256
                == if index == 0 {
                    "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"
                } else {
                    "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6"
                },
            "original source SHA metadata",
        )?;
        need(
            case.source_provenance.is_object(),
            "original producer/reference provenance",
        )?;
        let source_bytes = std::fs::metadata(&models[index].0)?.len();
        // Diagnostic envelope before materialization; actual cached-vector/metadata
        // capacity and outer process physical RAM remain outside this claim.
        let source_ticket = arbiter.admit(
            &key,
            owner.clone(),
            &[
                AllocationRequest {
                    kind: AllocationClass::ImmutableWeights,
                    bytes: source_bytes,
                },
                AllocationRequest {
                    kind: AllocationClass::Other,
                    bytes: CACHE_METADATA,
                },
            ],
        )?;
        let model = LoadedQwenModel::open(&models[index].0, &models[index].1)?;
        let weak = Arc::downgrade(&model);
        let source = model.source_identity();
        let staging = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: STAGING,
            }],
        )?;
        let dense = read_words(fixtures, &case.left)?;
        let left = expand(&dense, &case.left)?;
        drop(dense);
        let right = read_words(fixtures, &case.right)?;
        let ll = BulkF32Layout::new(case.left.shape, case.left.strides, 0, left.len())?;
        let rl = BulkF32Layout::dense(case.right.shape)?;
        need(
            rl.strides() == case.right.strides && rl.backing_words() == right.len(),
            "actual full right contiguous geometry",
        )?;
        if index == 1 {
            let original = model.tensor_bytes("blk.0.ssm_alpha.weight")?;
            need(
                original.len() == left.len() * 4
                    && original
                        .chunks_exact(4)
                        .zip(&left)
                        .all(|(raw, value)| raw == value.to_le_bytes()),
                "all32768 original source alpha words",
            )?;
        }
        let admission = BulkF32HostAdmission {
            arbiter: arbiter.clone(),
            resource: key.clone(),
            owner: owner.clone(),
        };
        let make = |flag: Option<&AtomicBool>| -> zerogravity_core::Result<BulkF32Slice> {
            let rhs = ExternalBulkF32 {
                name: &case.right.name,
                layout: rl,
                backing: &right,
            };
            if index == 0 {
                BulkF32Slice::observed_pair(
                    model.clone(),
                    0,
                    8448,
                    &ids,
                    ExternalBulkF32 {
                        name: &case.left.name,
                        layout: ll,
                        backing: &left,
                    },
                    rhs,
                    admission.clone(),
                    flag,
                )
            } else {
                BulkF32Slice::source_projection(
                    model.clone(),
                    0,
                    8448,
                    &ids,
                    QwenMatrixRole::GdnAlpha(0),
                    rhs,
                    admission.clone(),
                    flag,
                )
            }
        };
        let cancelled = AtomicBool::new(true);
        let prior = snapshot(&arbiter, &key)?;
        need(
            make(Some(&cancelled)).is_err() && snapshot(&arbiter, &key)? == prior,
            "pre-cancel before any private context lease",
        )?;
        let slice = make(None)?;
        need(
            slice.matrix().source() == source && slice.matrix().left().layout() == ll,
            "same immutable source/full-stride descriptor",
        )?;
        let recorded = slice.record()?;
        let plan = recorded.plan();
        need(
            plan.nodes().len() == 3
                && matches!(
                    plan.nodes()[0].operation,
                    PlanOperation::TypedBulkF32Input { .. }
                )
                && matches!(
                    plan.nodes()[1].operation,
                    PlanOperation::TypedBulkF32Input { .. }
                )
                && matches!(
                    plan.nodes()[2].operation,
                    PlanOperation::TypedBulkF32Matmul { .. }
                )
                && plan.is_operator_slice()
                && plan.required_domains().is_empty()
                && plan.proposed_domains().is_empty()
                && plan.frontier_delta() == 0
                && plan.logits().is_none(),
            "SAME exact three-node pure ForwardPlan",
        )?;
        drop(recorded);
        let baseline = snapshot(&arbiter, &key)?;
        need(
            slice.execute(Some(&cancelled)).is_err() && snapshot(&arbiter, &key)? == baseline,
            "pre-cancel execution retains exact input ledger",
        )?;
        let provider = DeniedProvider(AtomicUsize::new(0));
        need(
            slice.execute_with_provider(Some(&provider), None).is_err()
                && provider.0.load(Ordering::Relaxed) == 0
                && snapshot(&arbiter, &key)? == baseline,
            "unaware provider no callback or output allocation",
        )?;
        let out_words = case.expected.shape.iter().product::<usize>();
        let out_bytes = (out_words * 4) as u64;
        let reserved = arbiter.snapshot(&key)?.reserved;
        let pressure = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: LIMIT - reserved - out_bytes - 256 + 1,
            }],
        )?;
        let pressure_state = snapshot(&arbiter, &key)?;
        need(
            matches!(slice.execute(None),Err(zerogravity_core::Error::Limit(ref text))if text=="physical device memory admission refused")
                && snapshot(&arbiter, &key)? == pressure_state,
            "real one-byte output/scratch admission refusal",
        )?;
        drop(pressure);
        let completed = slice.execute(None)?;
        need(
            completed.source_identity() == source
                && completed.shape() == case.expected.shape
                && completed.values()?.len() == out_words,
            "source-bound complete actual matrix result",
        )?;
        let live = snapshot(&arbiter, &key)?;
        // No expected arithmetic bytes are read until SAME HostMath has returned.
        let expected = read_words(fixtures, &case.expected)?;
        need(
            expected.len() == completed.values()?.len(),
            "complete expected cardinality",
        )?;
        let unequal = completed
            .values()?
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mut actual = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join(format!("{id}.actual.f32le")))?;
        for word in completed.values()? {
            actual.write_all(&word.to_le_bytes())?;
        }
        actual.sync_all()?;
        drop(actual);
        drop(expected);
        let clone = completed.clone();
        drop(completed);
        need(
            clone.view(0..out_words)?.len() == out_words
                && clone.view(out_words..out_words + 1).is_err(),
            "borrowed retained output range",
        )?;
        drop(slice);
        drop(model);
        drop(left);
        drop(right);
        drop(staging);
        need(
            weak.upgrade().is_some(),
            "actual output retains original immutable model source",
        )?;
        let retained = snapshot(&arbiter, &key)?;
        need(
            retained["staging"] == 0
                && retained["output"] == out_bytes
                && retained["scratch"] == 256
                && retained["metadata"] == META + CACHE_METADATA + 16384,
            "input payloads gone; clone retains output ticket + issuer",
        )?;
        drop(clone);
        need(
            weak.upgrade().is_none(),
            "final output releases original source owner",
        )?;
        need(
            arbiter.snapshot(&key)?.reserved == META + source_bytes + CACHE_METADATA,
            "only external source envelope remains",
        )?;
        drop(source_ticket);
        let after = snapshot(&arbiter, &key)?;
        need(after["reserved"] == META, "case resources returned")?;
        reports.push(json!({"case":id,"model_sha256":case.model_sha256,"words":out_words,"unequal_words":unequal,
            "source_bound":source.is_bound(),"source_bytes":source_bytes,"left_shape":case.left.shape,"left_strides":case.left.strides,
            "right_shape":case.right.shape,"plan_nodes":3,"frontier_delta":0,"state_domains":0,"expected_opened_after_math":true,
            "pre_cancel_constructor":true,"pre_cancel_execution":true,"provider_callbacks":0,"real_one_byte_pressure_refused":true,
            "live":live,"retained_output":retained,"after_case":after,"output_clone_retains_source":true}));
    }
    drop(meta);
    let final_state = snapshot(&arbiter, &key)?;
    need(
        final_state["reserved"] == 0 && final_state["fenced"] == false,
        "all original leases final0",
    )?;
    need(mxcsr() & 0xffc0 == 0x1f80, "unchanged source FP controls")?;
    let exact = reports.iter().all(|row| row["unequal_words"] == 0);
    let report = json!({"status":if exact{"PASS_SAME_ENGINE_PURE_BULK_F32_ONLY"}else{"FAIL_SOURCE_BITS"},"cases":reports,
        "before":before,"final":final_state,"actual_affinity":cpus,"mxcsr_before":fp_before,"mxcsr_after":mxcsr(),
        "words":132096,"default_forward_changed":false,"fullmodel_or_StateRoot_qualified":false,"historical_prefix_recovered":false,
        "actual_mid_operation_cancel_qualified":false,"physical_RAM_admission_qualified":false});
    let mut f = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("report.json"))?;
    serde_json::to_writer_pretty(&mut f, &report)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    need(
        exact,
        "unchanged default public outputs must all match; no tolerance",
    )
}
fn main() {
    let args = std::env::args().map(PathBuf::from).collect::<Vec<_>>();
    let result = (|| -> Result<()> {
        need(
            args.len() == 8,
            "FIXTURES NEW_OUTPUT GGUF08 CONFIG08 GGUF2 CONFIG2 TEACHER64",
        )?;
        run(
            &args[1],
            &args[2],
            &[
                (args[3].clone(), args[4].clone()),
                (args[5].clone(), args[6].clone()),
            ],
            &args[7],
        )
    })();
    if let Err(error) = result {
        eprintln!("BULK_TYPED_UNQUALIFIED:{error}");
        std::process::exit(2)
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/owned_lifetime.py

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

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/prepare.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 2978; SHA256 `2771c29d4d02019957dda1b5be2e42598b19ea9620a6cec20ac29031f493c554`.

````text
"""ROOT ONLY copy original operands/defaults unchanged; no model/operator runtime."""
import argparse
from pathlib import Path
from contracts import HERE, REPO, CASES, digest, frozen, parents, require, verify_transport, write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-fixtures',type=Path,required=True)
    a=p.parse_args();pin=frozen();old,prep,av,alpha=parents()
    require(a.new_fixtures.is_absolute()and a.new_fixtures.parent.resolve()==HERE and not a.new_fixtures.exists(),'fresh own fixture directory')
    a.new_fixtures.mkdir(mode=0o700);rows=[]
    for case in CASES:
        row,paths=prep.transport_descriptor(case);rows.append(row)
        for key,path in zip(('left','right','expected'),paths,strict=True):
            require(digest(path,1024*1024)==row[key]['digest'],'original full raw source bytes')
            target=a.new_fixtures/row[key]['file']
            with path.open('rb')as src,target.open('xb')as dst:
                remaining=row[key]['digest']['bytes']
                while remaining:
                    raw=src.read(min(remaining,1024*1024));require(raw,'complete raw source');dst.write(raw);remaining-=len(raw)
                require(not src.read(1),'source did not grow')
            target.chmod(0o600)
    write(a.new_fixtures/'manifest.json',{'status':'AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY',
          'profile':'GgmlD812350HaswellTinyBlas','cases':rows,'arithmetic_executed':False,'expected_from_candidate':False,
          'gap_storage':'actual dense logical V restored at original strides; uncaptured gaps zero-filled'})
    av_ids=old.AV.parent/'numerics-and-context-bulk-causality-20261020/root-first64-ar1-budget/input_ids.u32le'
    alpha_ids=alpha.CAPTURE/'input_ids.u32le';require(av_ids.read_bytes()==alpha_ids.read_bytes(),'actual same teachers')
    ids=av_ids.read_bytes();require(len(ids)==32768,'full8192 source IDs')
    (a.new_fixtures/'teacher64.u32le').write_bytes(ids[:256]);(a.new_fixtures/'teacher64.u32le').chmod(0o600)
    models=[]
    for case,alias in zip(CASES,('qwen35-0.8b','qwen35-2b'),strict=True):
        model=REPO/'target/models'/alias/'model.gguf';config=model.with_name('config.json')
        models.append({'case':case,'gguf':str(model),'gguf_digest':digest(model),
                       'config':str(config),'config_digest':digest(config,1024*1024)})
    write(a.new_fixtures/'typed-inputs.json',{'status':'AUTHENTIC_SAME_ENGINE_BULK_F32_INPUTS_ONLY',
          'arithmetic_executed':False,'sourceprofile':'GGUF_ORIGINAL','base':0,'capacity':8448,'teachers':64,
          'teacher_digest':digest(a.new_fixtures/'teacher64.u32le',256),
          'teacher_sources':[{'path':str(path),'digest':digest(path,32768)}for path in(av_ids,alpha_ids)],'models':models})
    verify_transport(a.new_fixtures);require(frozen()==pin,'same reviewed source after transport')
    print('AUTHENTIC_SAME_ENGINE_BULK_F32_INPUTS_ONLY')
if __name__=='__main__':main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6711; SHA256 `fbbd8b45b26c201e558d555dbc41335010bf391e5578ceb0e50c9cc8a028c1fc`.

````text
use super::*;
fn span() -> TeacherSpan {
    TeacherSpan::new(
        SourceIdentity::for_new_allocation().unwrap(),
        0,
        8448,
        248320,
        &(0..64).collect::<Vec<_>>(),
    )
    .unwrap()
}
fn matrix(span: &TeacherSpan) -> BulkF32Matrix {
    let left = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Left,
        "actual external V",
        BulkF32Layout::new([256, 256, 2, 1], [1, 8448, 2162688, 4325376], 0, 4317184).unwrap(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "actual external P",
        BulkF32Layout::dense([256, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    BulkF32Matrix::new(left, right, BulkF32Contract::HaswellTinyBlasF32V1).unwrap()
}
#[test]
fn actual_temporal_n_is_not_cohort_width_and_full_k_is_preserved() {
    let span = span();
    let operation = matrix(&span);
    assert_eq!(operation.left().layout().shape(), [256, 256, 2, 1]);
    assert_eq!(operation.right().layout().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().shape(), [256, 64, 8, 1]);
    assert_eq!(operation.output().backing_words(), 131072);
    // Real N2 is a different declared teacher span, never a B2 request label.
    let wrong = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([256, 2, 8, 1]).unwrap(),
    );
    assert!(wrong.is_err());
    assert!(TeacherSpan::new(span.source(), 0, 8448, 248320, &[11]).is_err());
    assert!(TeacherSpan::new(span.source(), 8447, 8448, 248320, &[11, 353]).is_err());
}
#[test]
fn same_ids_source_names_do_not_merge_issuers_and_wrong_full_k_refuses() {
    let span = span();
    let operation = matrix(&span);
    let foreign = TeacherSpan::new(
        span.source(),
        span.base(),
        span.capacity(),
        248320,
        span.ids(),
    )
    .unwrap();
    let right = BulkF32Operand::external(
        foreign,
        BulkF32Side::Right,
        "actual external P",
        operation.right().layout(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
    let right = BulkF32Operand::external(
        span.clone(),
        BulkF32Side::Right,
        "P",
        BulkF32Layout::dense([128, 64, 8, 1]).unwrap(),
    )
    .unwrap();
    assert!(
        BulkF32Matrix::new(
            operation.left().clone(),
            right,
            BulkF32Contract::HaswellTinyBlasF32V1
        )
        .is_err()
    );
}
#[test]
fn real_nonzero_offsets_and_gaps_need_full_checked_storage() {
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 47).is_ok());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], 3, 46).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 7, 28, 28], 0, 32).is_err());
    assert!(BulkF32Layout::new([8, 4, 1, 1], [1, 12, 48, 48], usize::MAX, 64).is_err());
    assert!(BulkF32Layout::dense([8192, 4096, 8, 1]).is_err());
    let span = span();
    assert!(
        BulkF32Operand::external(
            span,
            BulkF32Side::Right,
            "strided P",
            BulkF32Layout::new([8, 64, 1, 1], [1, 12, 768, 768], 0, 764).unwrap()
        )
        .is_err()
    );
}
#[test]
fn same_builder_three_nodes_stay_pure_and_legacy_use_or_foreign_values_refuse() {
    let span = span();
    let operation = matrix(&span);
    let mut b =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let left = b.bulk_f32_input(operation.left()).unwrap();
    let right = b.bulk_f32_input(operation.right()).unwrap();
    let mut other =
        PlanBuilder::new_operator_slice(span.source(), ProgramNumericalProfile::GgufSourceV1)
            .unwrap();
    let foreign = other.bulk_f32_input(operation.right()).unwrap();
    assert!(b.bulk_f32_matmul(&operation, &left, &foreign).is_err());
    assert!(b.silu("legacy", &right, None).is_err());
    let output = b.bulk_f32_matmul(&operation, &left, &right).unwrap();
    let slice = b.finish_operator_slice(output).unwrap();
    assert_eq!(slice.plan().nodes().len(), 3);
    assert!(slice.plan().required_domains().is_empty());
    assert!(slice.plan().proposed_domains().is_empty());
    assert_eq!(slice.plan().frontier_delta(), 0);
    assert!(slice.plan().logits().is_none());
    assert!(slice.plan().is_operator_slice());
    let slot = StateSlot::Conv(ConvDomain(0));
    let mut full = PlanBuilder::new(span.source(), 8448, vec![slot], false).unwrap();
    assert!(full.bulk_f32_input(operation.left()).is_err());
}

struct Original {
    identity: SourceIdentity,
    kind: QuantKind,
    shape: [u64; 2],
    bytes: Vec<u8>,
}
impl TensorSource for Original {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "original.weight" {
            return Err(Error::Malformed("unknown original name".into()));
        }
        Ok(SourceTensorRef {
            kind: self.kind,
            shape: &self.shape,
            packed_bytes: &self.bytes,
        })
    }
}
#[test]
fn whole_raw_source_matrix_keeps_axes_dtype_extent_and_exact_issuer() {
    let original = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 128],
    };
    let span = TeacherSpan::new(original.identity, 0, 8, 248320, &[11, 353]).unwrap();
    let left = BulkF32Operand::source_matrix(span.clone(), &original, "original.weight").unwrap();
    assert_eq!(left.layout().shape(), [8, 4, 1, 1]);
    let BulkF32Origin::SourceMatrix(raw) = left.origin() else {
        panic!("original raw source expected")
    };
    assert_eq!(raw.axes, MatrixAxes::InputOutput);
    raw.validate_source(&original).unwrap();
    let foreign = Original {
        identity: SourceIdentity::for_new_allocation().unwrap(),
        ..original
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &foreign, "original.weight").is_err());
    let bf16 = Original {
        identity: span.source(),
        kind: QuantKind::Bf16,
        shape: [8, 4],
        bytes: vec![0; 64],
    };
    assert!(BulkF32Operand::source_matrix(span.clone(), &bf16, "original.weight").is_err());
    let short = Original {
        identity: span.source(),
        kind: QuantKind::F32,
        shape: [8, 4],
        bytes: vec![0; 124],
    };
    assert!(BulkF32Operand::source_matrix(span, &short, "original.weight").is_err());
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-core/src/accelerator/program/bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 12667; SHA256 `57805fe39fdee6bee7ece57e0b8fc2163ea73a1c85d78ca6c11b716d82ae110f`.

````text
//! Closed temporal matrix metadata; no session, model route or producer permit.
use super::*;

/// ONE consecutive teacher sequence. Clones share one immutable issuer; equal
/// text/IDs from another constructor do not silently become the same scope.
#[derive(Clone, Debug)]
pub struct TeacherSpan(Arc<TeacherSpanData>);
#[derive(Debug)]
struct TeacherSpanData {
    source: SourceIdentity,
    base: usize,
    capacity: usize,
    ids: Arc<[u32]>,
}
impl PartialEq for TeacherSpan {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for TeacherSpan {}
impl TeacherSpan {
    /// Metadata only; these IDs are not a current SessionAuthority/StateRoot.
    pub fn new(
        source: SourceIdentity,
        base: usize,
        capacity: usize,
        vocabulary: u64,
        ids: &[u32],
    ) -> Result<Self> {
        if !source.is_bound()
            || !(2..=64).contains(&ids.len())
            || capacity == 0
            || capacity > 8448
            || base.checked_add(ids.len()).is_none_or(|end| end > capacity)
            || vocabulary == 0
            || vocabulary > u64::from(u32::MAX) + 1
            || ids.iter().any(|&id| u64::from(id) >= vocabulary)
        {
            return Err(Error::Malformed(
                "closed teacher span source/geometry/IDs".into(),
            ));
        }
        Ok(Self(Arc::new(TeacherSpanData {
            source,
            base,
            capacity,
            ids: Arc::from(ids),
        })))
    }
    pub fn source(&self) -> SourceIdentity {
        self.0.source
    }
    pub fn base(&self) -> usize {
        self.0.base
    }
    pub fn capacity(&self) -> usize {
        self.0.capacity
    }
    pub fn ids(&self) -> &[u32] {
        &self.0.ids
    }
    pub fn len(&self) -> usize {
        self.0.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Actual F32 storage axes `[full_K,M_or_N,head,sequence]`, word strides.
/// Offset/backing describe THIS allocation; original capture provenance stays
/// separate if transport reconstructed gaps. Full K is never split/reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BulkF32Layout {
    shape: [usize; 4],
    strides: [usize; 4],
    offset: usize,
    backing: usize,
}
impl BulkF32Layout {
    pub fn new(
        shape: [usize; 4],
        strides: [usize; 4],
        offset: usize,
        backing: usize,
    ) -> Result<Self> {
        let value = Self {
            shape,
            strides,
            offset,
            backing,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn dense(shape: [usize; 4]) -> Result<Self> {
        let mut strides = [1usize; 4];
        for i in 1..4 {
            strides[i] = strides[i - 1]
                .checked_mul(shape[i - 1])
                .ok_or_else(|| Error::Limit("bulk dense stride".into()))?;
        }
        let backing = strides[3]
            .checked_mul(shape[3])
            .ok_or_else(|| Error::Limit("bulk dense words".into()))?;
        Self::new(shape, strides, 0, backing)
    }
    pub fn shape(self) -> [usize; 4] {
        self.shape
    }
    pub fn strides(self) -> [usize; 4] {
        self.strides
    }
    pub fn offset(self) -> usize {
        self.offset
    }
    pub fn backing_words(self) -> usize {
        self.backing
    }
    pub fn logical_words(self) -> Result<usize> {
        self.shape.iter().try_fold(1usize, |n, &x| {
            n.checked_mul(x)
                .ok_or_else(|| Error::Limit("bulk logical words".into()))
        })
    }
    pub fn validate(self) -> Result<()> {
        if self.shape.contains(&0)
            || self.shape[0] > 8192
            || self.shape[1] > 4096
            || self.shape[2] > 64
            || self.shape[3] != 1
            || self.strides[0] != 1
            || self.backing == 0
            || self.backing > 8 * 1024 * 1024
        {
            return Err(Error::Unsupported("closed bulk F32 axes/backing".into()));
        }
        let mut previous = self.shape[0];
        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
            if stride < previous {
                return Err(Error::Malformed("bulk overlapping rows/axes".into()));
            }
            previous = stride
                .checked_mul(size)
                .ok_or_else(|| Error::Limit("bulk stride product".into()))?;
        }
        let end = self
            .shape
            .iter()
            .zip(self.strides)
            .try_fold(self.offset, |n, (&size, stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|x| n.checked_add(x))
                    .ok_or_else(|| Error::Limit("bulk full span".into()))
            })?
            .checked_add(1)
            .ok_or_else(|| Error::Limit("bulk span end".into()))?;
        if end > self.backing || self.logical_words()? > 1024 * 1024 {
            return Err(Error::Limit("bulk operand span/logical bound".into()));
        }
        Ok(())
    }
    pub fn is_dense(self) -> bool {
        Self::dense(self.shape).is_ok_and(|dense| dense == self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkF32Side {
    Left,
    Right,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BulkF32Origin {
    /// Caller-supplied pure operator input, explicitly NOT a model/state producer.
    ExternalObserved { name: String },
    /// Exact whole original raw learned matrix, no dequantization/retag/cache.
    SourceMatrix(SourceTensor),
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BulkF32Operand {
    span: TeacherSpan,
    side: BulkF32Side,
    layout: BulkF32Layout,
    origin: BulkF32Origin,
}
impl BulkF32Operand {
    pub fn external(
        span: TeacherSpan,
        side: BulkF32Side,
        name: &str,
        layout: BulkF32Layout,
    ) -> Result<Self> {
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(Error::Malformed("bulk external input label bound".into()));
        }
        let value = Self {
            span,
            side,
            layout,
            origin: BulkF32Origin::ExternalObserved { name: name.into() },
        };
        value.validate()?;
        Ok(value)
    }
    pub fn source_matrix(span: TeacherSpan, source: &dyn TensorSource, name: &str) -> Result<Self> {
        if name.is_empty() || name.len() > 256 || name.contains('\0') {
            return Err(Error::Malformed("bulk learned input label bound".into()));
        }
        let tensor = SourceTensor::from_source(source, name, MatrixAxes::InputOutput)?;
        if tensor.kind != QuantKind::F32 || tensor.source != span.source() {
            return Err(Error::Unsupported(
                "bulk original F32 source matrix only".into(),
            ));
        }
        let layout = BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?;
        let value = Self {
            span,
            side: BulkF32Side::Left,
            layout,
            origin: BulkF32Origin::SourceMatrix(tensor),
        };
        value.validate()?;
        Ok(value)
    }
    pub fn span(&self) -> &TeacherSpan {
        &self.span
    }
    pub fn side(&self) -> BulkF32Side {
        self.side
    }
    pub fn layout(&self) -> BulkF32Layout {
        self.layout
    }
    pub fn origin(&self) -> &BulkF32Origin {
        &self.origin
    }
    pub fn name(&self) -> &str {
        match &self.origin {
            BulkF32Origin::ExternalObserved { name } => name,
            BulkF32Origin::SourceMatrix(tensor) => &tensor.name,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.layout.validate()?;
        if self.side == BulkF32Side::Right
            && (self.layout.shape[1] != self.span.len() || !self.layout.is_dense())
        {
            return Err(Error::Unsupported(
                "actual teacher N and original contiguous RHS required".into(),
            ));
        }
        if let BulkF32Origin::SourceMatrix(tensor) = &self.origin {
            if self.side != BulkF32Side::Left
                || tensor.source != self.span.source()
                || tensor.kind != QuantKind::F32
                || tensor.axes != MatrixAxes::InputOutput
                || self.layout != BulkF32Layout::dense([tensor.columns, tensor.rows, 1, 1])?
            {
                return Err(Error::State("bulk whole learned descriptor/axes".into()));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkF32Contract {
    HaswellTinyBlasF32V1,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BulkF32Matrix {
    left: BulkF32Operand,
    right: BulkF32Operand,
    contract: BulkF32Contract,
    output: BulkF32Layout,
}
impl BulkF32Matrix {
    pub fn new(
        left: BulkF32Operand,
        right: BulkF32Operand,
        contract: BulkF32Contract,
    ) -> Result<Self> {
        left.validate()?;
        right.validate()?;
        let a = left.layout.shape;
        let b = right.layout.shape;
        if left.side != BulkF32Side::Left
            || right.side != BulkF32Side::Right
            || left.span != right.span
            || a[0] != b[0]
            || !a[1].is_multiple_of(4)
            || !b[2].is_multiple_of(a[2])
        {
            return Err(Error::Unsupported(
                "closed temporal matrix issuer/fullK/head broadcast".into(),
            ));
        }
        let output = BulkF32Layout::dense([a[1], b[1], b[2], 1])?;
        if output
            .logical_words()?
            .checked_mul(a[0])
            .is_none_or(|n| n > 128 * 1024 * 1024)
        {
            return Err(Error::Limit("bulk complete K work bound".into()));
        }
        Ok(Self {
            left,
            right,
            contract,
            output,
        })
    }
    pub fn left(&self) -> &BulkF32Operand {
        &self.left
    }
    pub fn right(&self) -> &BulkF32Operand {
        &self.right
    }
    pub fn contract(&self) -> BulkF32Contract {
        self.contract
    }
    pub fn output(&self) -> BulkF32Layout {
        self.output
    }
    pub fn source(&self) -> SourceIdentity {
        self.left.span.source()
    }
}

fn check_builder(builder: &PlanBuilder, source: SourceIdentity) -> Result<()> {
    if !builder.operator_slice
        || builder.profile != ProgramNumericalProfile::GgufSourceV1
        || builder.source != source
    {
        return Err(Error::Unsupported(
            "bulk F32 requires explicit state-free GGUF operator slice".into(),
        ));
    }
    Ok(())
}
pub(super) fn input_precision(
    builder: &PlanBuilder,
    operand: &BulkF32Operand,
) -> Result<ActivationPrecision> {
    check_builder(builder, operand.span.source())?;
    operand.validate()?;
    Ok(ActivationPrecision::F32)
}
pub(super) fn matrix_precision(
    builder: &PlanBuilder,
    matrix: &BulkF32Matrix,
    inputs: &[PlanValue],
) -> Result<ActivationPrecision> {
    check_builder(builder, matrix.source())?;
    if inputs.len() != 2 {
        return Err(Error::Malformed("bulk input arity".into()));
    }
    for (input, expected) in inputs.iter().zip([matrix.left(), matrix.right()]) {
        let producer = builder
            .nodes
            .iter()
            .find(|node| node.outputs.contains(&input.id));
        if input.precision != ActivationPrecision::F32
            || input.elements != expected.layout.backing_words()
            || !matches!(producer.map(|node|&node.operation),Some(PlanOperation::TypedBulkF32Input{operand})if operand==expected)
        {
            return Err(Error::State(
                "bulk exact input producer/layout/span membership".into(),
            ));
        }
    }
    Ok(ActivationPrecision::F32)
}
pub(super) fn record_input(
    builder: &mut PlanBuilder,
    operand: &BulkF32Operand,
) -> Result<PlanValue> {
    input_precision(builder, operand)?;
    builder.one(
        operand.name(),
        PlanOperation::TypedBulkF32Input {
            operand: operand.clone(),
        },
        &[],
        ValueShape::Vector(operand.layout.backing_words()),
        None,
    )
}
pub(super) fn record_matrix(
    builder: &mut PlanBuilder,
    matrix: &BulkF32Matrix,
    left: &PlanValue,
    right: &PlanValue,
) -> Result<PlanValue> {
    matrix_precision(builder, matrix, &[*left, *right])?;
    builder.one(
        "closed-bulk-f32",
        PlanOperation::TypedBulkF32Matmul {
            matrix: matrix.clone(),
        },
        &[*left, *right],
        ValueShape::Vector(matrix.output.backing_words()),
        None,
    )
}

#[cfg(test)]
mod tests;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-core/src/accelerator/program.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 78390; SHA256 `a40a3923d0ae6f8734bc62853edc588d15d4dc5a496949e2aa5b6ab432ec65fb`.

````text
//! Typed forward-program seam. No device runtime or publication owner.
//!
//! The model's ONE mathematical schedule can evaluate host values or record
//! this bounded straight-line plan. Runtime parameters/state generations are
//! bound after compilation; the plan never contains a session or GPU handle.

use super::SourceIdentity;
use crate::quant::QuantKind;
use crate::source::{SourceTensorRef, TensorSource};
use crate::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod gdn_conv;
pub use gdn_conv::{CausalConvSiluContract, CausalConvValues};
mod typed;
pub use typed::{
    ActivationPrecision, EmbeddingContract, OrdinaryRmsContract, ProgramNumericalProfile,
    ProjectionContract, SourceParameter,
};

mod bulk_f32;
pub use bulk_f32::{
    BulkF32Contract, BulkF32Layout, BulkF32Matrix, BulkF32Operand, BulkF32Origin, BulkF32Side,
    TeacherSpan,
};

static NEXT_PLAN_BUILDER: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub offset: usize,
    pub rows: usize,
    pub width: usize,
    pub row_stride: usize,
}
impl TensorView {
    pub fn vector(offset: usize, elements: usize) -> Self {
        Self {
            offset,
            rows: 1,
            width: elements,
            row_stride: elements,
        }
    }
    pub fn elements(self) -> Result<usize> {
        if self.rows == 0 || self.width == 0 {
            return Err(Error::Malformed("empty mathematical tensor view".into()));
        }
        self.rows
            .checked_mul(self.width)
            .ok_or_else(|| Error::Limit("tensor view count overflow".into()))
    }
    pub fn validate(self, source_elements: usize) -> Result<()> {
        self.elements()?;
        if self.row_stride < self.width {
            return Err(Error::Malformed(
                "overlapping mathematical tensor rows".into(),
            ));
        }
        let end = (self.rows - 1)
            .checked_mul(self.row_stride)
            .and_then(|n| n.checked_add(self.offset))
            .and_then(|n| n.checked_add(self.width))
            .ok_or_else(|| Error::Limit("tensor view range overflow".into()))?;
        if end > source_elements {
            return Err(Error::Malformed(
                "mathematical tensor view exceeds input".into(),
            ));
        }
        Ok(())
    }
    pub fn is_contiguous(self) -> bool {
        self.rows == 1 || self.row_stride == self.width
    }
}

/// Raw matrix header order; packed output rows are never transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixAxes {
    /// GGUF: [input width, output rows].
    InputOutput,
    /// Hugging Face: [output rows, input width].
    OutputInput,
}
impl MatrixAxes {
    fn shape(self, columns: usize, rows: usize) -> [u64; 2] {
        match self {
            Self::InputOutput => [columns as u64, rows as u64],
            Self::OutputInput => [rows as u64, columns as u64],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTensor {
    pub source: SourceIdentity,
    pub name: String,
    pub kind: QuantKind,
    pub axes: MatrixAxes,
    pub columns: usize,
    pub rows: usize,
}
impl SourceTensor {
    /// Bind actual source metadata/bytes without copying or changing raw axes.
    /// This descriptor grants no operator or trained-model eligibility.
    pub fn from_source(source: &dyn TensorSource, name: &str, axes: MatrixAxes) -> Result<Self> {
        let identity = source.source_identity();
        if !identity.is_bound() || name.is_empty() {
            return Err(Error::State("unbound forward source tensor".into()));
        }
        let actual = source.tensor(name)?;
        actual.validate()?;
        let [first, second] = actual.shape else {
            return Err(Error::Malformed("forward source is not a matrix".into()));
        };
        let (columns, rows) = match axes {
            MatrixAxes::InputOutput => (*first, *second),
            MatrixAxes::OutputInput => (*second, *first),
        };
        let tensor = Self {
            source: identity,
            name: name.into(),
            kind: actual.kind,
            axes,
            columns: usize::try_from(columns)
                .map_err(|_| Error::Limit("forward matrix columns overflow".into()))?,
            rows: usize::try_from(rows)
                .map_err(|_| Error::Limit("forward matrix rows overflow".into()))?,
        };
        tensor.validate_ref(identity, actual)?;
        Ok(tensor)
    }

    fn packed_bytes(&self) -> Result<u64> {
        if !self.source.is_bound() || self.name.is_empty() || self.columns == 0 || self.rows == 0 {
            return Err(Error::State("invalid forward source descriptor".into()));
        }
        self.kind
            .byte_len(self.columns as u64)?
            .checked_mul(self.rows as u64)
            .ok_or_else(|| Error::Limit("forward matrix byte size overflow".into()))
    }

    fn validate_ref(&self, identity: SourceIdentity, actual: SourceTensorRef<'_>) -> Result<()> {
        let bytes = self.packed_bytes()?;
        actual.validate()?;
        if self.source != identity
            || actual.kind != self.kind
            || actual.shape != self.axes.shape(self.columns, self.rows)
            || actual.packed_bytes.len() as u64 != bytes
        {
            return Err(Error::State(
                "forward tensor differs from immutable source".into(),
            ));
        }
        Ok(())
    }

    /// Revalidate a plan descriptor against the same immutable source owner.
    pub fn validate_source(&self, source: &dyn TensorSource) -> Result<()> {
        self.packed_bytes()?;
        let identity = source.source_identity();
        if self.source != identity {
            return Err(Error::State("forward tensor source mismatch".into()));
        }
        self.validate_ref(identity, source.tensor(&self.name)?)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SourceVector<'a> {
    pub source: SourceIdentity,
    pub name: &'a str,
    pub values: &'a [f32],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ConvDomain(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct RecurrentDomain(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KvDomain(pub usize);
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StateSlot {
    Conv(ConvDomain),
    Recurrent(RecurrentDomain),
    Kv(KvDomain),
}

#[derive(Clone, Copy, Debug)]
pub struct ForwardInputs<I> {
    pub token: I,
    pub position: I,
    pub valid_tokens: I,
}
#[derive(Clone, Copy, Debug)]
pub struct AttentionGeometry {
    pub width: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
}
impl AttentionGeometry {
    fn validate(self) -> Result<(usize, usize)> {
        if self.width == 0
            || self.query_heads == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
        {
            return Err(Error::Malformed(
                "invalid forward attention geometry".into(),
            ));
        }
        let q = self
            .width
            .checked_mul(self.query_heads)
            .ok_or_else(|| Error::Limit("forward query count overflow".into()))?;
        let kv = self
            .width
            .checked_mul(self.kv_heads)
            .ok_or_else(|| Error::Limit("forward KV count overflow".into()))?;
        Ok((q, kv))
    }
}
#[derive(Clone, Copy, Debug)]
pub struct RopeGeometry {
    pub attention: AttentionGeometry,
    pub rotate: usize,
    pub theta_scale: f32,
}
pub struct CoefficientValues<V> {
    pub packed: V,
    pub softplus: V,
    pub gate_raw: V,
}
pub struct RopeValues<V> {
    pub query: V,
    pub key: V,
    pub value: V,
}
pub struct QkValues<V> {
    pub dots: V,
    pub scores: V,
}

/// Value operations plus typed PRIVATE state effects. Layer scheduling lives
/// only in qwen35::forward, never in a backend implementation of this trait.
#[allow(clippy::too_many_arguments)] // Source closed-op inputs stay explicit.
pub trait MathematicalBackend {
    type Value: Clone;
    type Index: Copy;
    fn elements(&self, value: &Self::Value) -> usize;
    fn precision(&self, _value: &Self::Value) -> ActivationPrecision {
        ActivationPrecision::F32
    }
    fn embed_profiled(
        &mut self,
        tensor: SourceTensor,
        token: Self::Index,
        contract: EmbeddingContract,
    ) -> Result<Self::Value> {
        match contract {
            EmbeddingContract::Source => self.embed(tensor, token),
            _ => Err(Error::Unsupported(
                "backend does not implement explicit embedding contract".into(),
            )),
        }
    }
    fn project_profiled(
        &mut self,
        tensor: SourceTensor,
        input: &Self::Value,
        contract: ProjectionContract,
    ) -> Result<Self::Value> {
        match contract {
            ProjectionContract::Source => self.project(tensor, input),
            _ => Err(Error::Unsupported(
                "backend does not implement explicit projection contract".into(),
            )),
        }
    }
    fn ordinary_rms_profiled(
        &mut self,
        _parameter: SourceParameter,
        _input: &Self::Value,
        _contract: OrdinaryRmsContract,
    ) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "backend does not implement explicit ordinary RMS contract".into(),
        ))
    }
    /// Pure original convolution window; no state owner or implicit fallback.
    fn causal_conv_silu_profiled(
        &mut self,
        _parameter: SourceParameter,
        _input: &Self::Value,
        _history: &Self::Value,
        _contract: CausalConvSiluContract,
    ) -> Result<CausalConvValues<Self::Value>> {
        Err(Error::Unsupported(
            "backend does not implement explicit causal convolution SiLU contract".into(),
        ))
    }
    fn bulk_f32_input(&mut self, _operand: &BulkF32Operand) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "backend has no closed bulk F32 input contract".into(),
        ))
    }
    fn bulk_f32_matmul(
        &mut self,
        _matrix: &BulkF32Matrix,
        _left: &Self::Value,
        _right: &Self::Value,
    ) -> Result<Self::Value> {
        Err(Error::Unsupported(
            "backend has no closed Haswell bulk F32 matrix contract".into(),
        ))
    }
    fn embed(&mut self, tensor: SourceTensor, token: Self::Index) -> Result<Self::Value>;
    fn project(&mut self, tensor: SourceTensor, input: &Self::Value) -> Result<Self::Value>;
    fn view(&mut self, input: &Self::Value, view: TensorView) -> Result<Self::Value>;
    fn contiguous(&mut self, input: &Self::Value) -> Result<Self::Value>;
    fn rms(
        &mut self,
        name: &str,
        input: &Self::Value,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<Self::Value>;
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &Self::Value,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<Self::Value>;
    fn silu(
        &mut self,
        name: &str,
        input: &Self::Value,
        multiplier: Option<&Self::Value>,
    ) -> Result<Self::Value>;
    fn add(&mut self, name: &str, left: &Self::Value, right: &Self::Value) -> Result<Self::Value>;
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &Self::Value,
        gate: &Self::Value,
    ) -> Result<Self::Value>;
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &Self::Value,
        weights: SourceVector<'_>,
    ) -> Result<Self::Value>;
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &Self::Value,
        beta: &Self::Value,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<Self::Value>>;
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &Self::Value,
        k: &Self::Value,
        v: &Self::Value,
        coefficients: &Self::Value,
        width: usize,
    ) -> Result<Self::Value>;
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: Self::Index) -> Result<()>;
    fn rope(
        &mut self,
        name: &str,
        q: &Self::Value,
        k: &Self::Value,
        v: &Self::Value,
        position: Self::Index,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<Self::Value>>;
    fn append_kv(&mut self, domain: KvDomain, k: &Self::Value, v: &Self::Value) -> Result<()>;
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &Self::Value,
        valid_tokens: Self::Index,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<Self::Value>>;
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &Self::Value,
        heads: usize,
        valid_tokens: Self::Index,
    ) -> Result<Self::Value>;
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &Self::Value,
        valid_tokens: Self::Index,
        geometry: AttentionGeometry,
    ) -> Result<Self::Value>;
    fn assert_finite(&mut self, layer: usize, value: &Self::Value) -> Result<()>;
    fn observe(&mut self, layer: usize, stage: &'static str, value: &Self::Value) -> Result<()>;
    fn observe_state(&mut self, layer: usize, stage: &'static str, domain: StateSlot)
    -> Result<()>;
    /// Advance PRIVATE proposed metadata. This never publishes a session root.
    fn advance_frontier(&mut self) -> Result<()>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterSlot {
    Token,
    Position,
    ValidTokens,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterRef(pub ParameterSlot);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueId(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueShape {
    Vector(usize),
    CausalRows { heads: usize, capacity: usize },
}
impl ValueShape {
    pub fn capacity_elements(self) -> Result<usize> {
        match self {
            Self::Vector(n) if n > 0 => Ok(n),
            Self::CausalRows { heads, capacity } if heads > 0 && capacity > 0 => heads
                .checked_mul(capacity)
                .ok_or_else(|| Error::Limit("causal value capacity overflow".into())),
            _ => Err(Error::Malformed("empty forward-plan value".into())),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct PlanValue {
    pub id: ValueId,
    pub shape: ValueShape,
    owner: u64,
    elements: usize,
    contiguous: bool,
    precision: ActivationPrecision,
}
impl PlanValue {
    pub fn precision(self) -> ActivationPrecision {
        self.precision
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticVectorId(pub u32);
#[derive(Clone, Debug)]
pub struct StaticVector {
    pub source: SourceIdentity,
    pub name: String,
    pub values: Arc<[f32]>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateVersion {
    pub slot: StateSlot,
    pub ordinal: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct StateEffect {
    pub read: StateVersion,
    pub write: StateVersion,
}

/// Per-op source semantics, NOT backend launches or a second Qwen schedule.
/// Lowering may add explicit Pack/Copy instructions for qualified kernel ABIs.
#[derive(Clone, Debug)]
pub enum PlanOperation {
    TypedBulkF32Input {
        operand: BulkF32Operand,
    },
    TypedBulkF32Matmul {
        matrix: BulkF32Matrix,
    },
    TypedEmbed {
        tensor: SourceTensor,
        token: ParameterRef,
        contract: EmbeddingContract,
    },
    TypedProject {
        tensor: SourceTensor,
        contract: ProjectionContract,
    },
    TypedCausalConvSilu {
        parameter: SourceParameter,
        contract: CausalConvSiluContract,
    },
    TypedOrdinaryRms {
        parameter: SourceParameter,
        contract: OrdinaryRmsContract,
    },
    Embed {
        tensor: SourceTensor,
        token: ParameterRef,
    },
    Project {
        tensor: SourceTensor,
    },
    View(TensorView),
    Contiguous,
    Rms {
        gamma: StaticVectorId,
        eps: f32,
        groups: usize,
    },
    Silu {
        multiplied: bool,
    },
    Add,
    SigmoidGate,
    Conv {
        weights: StaticVectorId,
    },
    Coefficients {
        dt: StaticVectorId,
        a: StaticVectorId,
        scale: f32,
    },
    Recurrent {
        width: usize,
    },
    KvPrefixGuard {
        state: StateVersion,
        position: ParameterRef,
    },
    Rope {
        geometry: RopeGeometry,
        position: ParameterRef,
    },
    AppendKv,
    AttentionQk {
        state: StateVersion,
        geometry: AttentionGeometry,
        valid_tokens: ParameterRef,
        scale: f32,
    },
    AttentionSoftmax {
        heads: usize,
        valid_tokens: ParameterRef,
    },
    AttentionValue {
        state: StateVersion,
        geometry: AttentionGeometry,
        valid_tokens: ParameterRef,
        padding: usize,
    },
    CheckFinite {
        layer: usize,
    },
    Observe {
        layer: usize,
        stage: &'static str,
    },
    ObserveState {
        layer: usize,
        stage: &'static str,
        state: StateVersion,
    },
}
#[derive(Clone, Debug)]
pub struct PlanNode {
    pub name: String,
    pub operation: PlanOperation,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub state: Option<StateEffect>,
}
#[derive(Clone, Debug)]
pub struct ForwardPlan {
    source: SourceIdentity,
    profile: ProgramNumericalProfile,
    operator_slice: bool,
    precisions: Vec<ActivationPrecision>,
    prefix_capacity: usize,
    nodes: Vec<PlanNode>,
    values: Vec<ValueShape>,
    constants: Vec<StaticVector>,
    required_domains: Vec<StateSlot>,
    proposed_domains: Vec<StateVersion>,
    logits: Option<ValueId>,
    frontier_delta: usize,
    state_node_count: usize,
}
/// State-free fragment recorded by the same builder; no token/root advance.
#[derive(Clone, Debug)]
pub struct RecordedOperatorSlice {
    plan: ForwardPlan,
    output: ValueId,
}
impl RecordedOperatorSlice {
    pub fn plan(&self) -> &ForwardPlan {
        &self.plan
    }
    pub fn output(&self) -> ValueId {
        self.output
    }
}
impl ForwardPlan {
    pub fn numerical_profile(&self) -> ProgramNumericalProfile {
        self.profile
    }
    pub fn is_operator_slice(&self) -> bool {
        self.operator_slice
    }
    pub fn precisions(&self) -> &[ActivationPrecision] {
        &self.precisions
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn prefix_capacity(&self) -> usize {
        self.prefix_capacity
    }
    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }
    /// Complete state transition, including its numerical checks and observations.
    /// The schedule's private frontier advance closes this prefix. Any remaining
    /// nodes compute optional outputs and cannot access state.
    pub fn state_nodes(&self) -> &[PlanNode] {
        &self.nodes[..self.state_node_count]
    }
    pub fn values(&self) -> &[ValueShape] {
        &self.values
    }
    pub fn constants(&self) -> &[StaticVector] {
        &self.constants
    }
    pub fn required_domains(&self) -> &[StateSlot] {
        &self.required_domains
    }
    pub fn proposed_domains(&self) -> &[StateVersion] {
        &self.proposed_domains
    }
    pub fn logits(&self) -> Option<ValueId> {
        self.logits
    }
    pub fn frontier_delta(&self) -> usize {
        self.frontier_delta
    }
}

pub struct PlanBuilder {
    owner: u64,
    profile: ProgramNumericalProfile,
    operator_slice: bool,
    precisions: Vec<ActivationPrecision>,
    source: SourceIdentity,
    prefix_capacity: usize,
    nodes: Vec<PlanNode>,
    values: Vec<ValueShape>,
    constants: Vec<StaticVector>,
    constant_names: BTreeMap<String, StaticVectorId>,
    domains: BTreeMap<StateSlot, StateVersion>,
    required: BTreeSet<StateSlot>,
    capture: bool,
    advanced: Option<usize>,
}
impl PlanBuilder {
    pub fn new(
        source: SourceIdentity,
        prefix_capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
    ) -> Result<Self> {
        Self::new_inner(
            source,
            prefix_capacity,
            required,
            capture,
            ProgramNumericalProfile::GgufSourceV1,
            false,
        )
    }
    pub fn new_with_profile(
        source: SourceIdentity,
        prefix_capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        Self::new_inner(source, prefix_capacity, required, capture, profile, false)
    }
    pub fn new_operator_slice(
        source: SourceIdentity,
        profile: ProgramNumericalProfile,
    ) -> Result<Self> {
        Self::new_inner(source, 1, Vec::new(), false, profile, true)
    }
    fn new_inner(
        source: SourceIdentity,
        prefix_capacity: usize,
        required: Vec<StateSlot>,
        capture: bool,
        profile: ProgramNumericalProfile,
        operator_slice: bool,
    ) -> Result<Self> {
        let expected_count = required.len();
        let required: BTreeSet<_> = required.into_iter().collect();
        if !source.is_bound()
            || prefix_capacity == 0
            || required.is_empty() != operator_slice
            || required.len() != expected_count
        {
            return Err(Error::Malformed("invalid source-bound forward plan".into()));
        }
        let owner = NEXT_PLAN_BUILDER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| Error::Limit("forward plan builder identity exhausted".into()))?;
        Ok(Self {
            owner,
            profile,
            operator_slice,
            precisions: Vec::new(),
            source,
            prefix_capacity,
            nodes: Vec::new(),
            values: Vec::new(),
            constants: Vec::new(),
            constant_names: BTreeMap::new(),
            domains: BTreeMap::new(),
            required,
            capture,
            advanced: None,
        })
    }
    pub fn inputs() -> ForwardInputs<ParameterRef> {
        ForwardInputs {
            token: ParameterRef(ParameterSlot::Token),
            position: ParameterRef(ParameterSlot::Position),
            valid_tokens: ParameterRef(ParameterSlot::ValidTokens),
        }
    }
    fn source_tensor(&self, tensor: &SourceTensor) -> Result<()> {
        tensor.packed_bytes()?;
        if tensor.source != self.source {
            return Err(Error::State(
                "forward plan changed source tensor identity".into(),
            ));
        }
        Ok(())
    }
    fn constant(&mut self, vector: SourceVector<'_>) -> Result<StaticVectorId> {
        if self.profile != ProgramNumericalProfile::GgufSourceV1 {
            return Err(Error::Unsupported(
                "profiled operators require raw source parameter descriptors".into(),
            ));
        }
        if vector.source != self.source
            || vector.name.is_empty()
            || vector.values.is_empty()
            || !vector.values.iter().all(|v| v.is_finite())
        {
            return Err(Error::Malformed(
                "invalid forward-plan static vector".into(),
            ));
        }
        if let Some(id) = self.constant_names.get(vector.name).copied() {
            let old = &self.constants[id.0 as usize];
            if old.values.len() != vector.values.len()
                || old
                    .values
                    .iter()
                    .zip(vector.values)
                    .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                return Err(Error::State(
                    "source vector role changed within one plan".into(),
                ));
            }
            return Ok(id);
        }
        let id = StaticVectorId(
            u32::try_from(self.constants.len())
                .map_err(|_| Error::Limit("static vector slot overflow".into()))?,
        );
        self.constants.push(StaticVector {
            source: vector.source,
            name: vector.name.into(),
            values: vector.values.into(),
        });
        self.constant_names.insert(vector.name.into(), id);
        Ok(id)
    }
    fn result_precision(
        &self,
        operation: &PlanOperation,
        inputs: &[PlanValue],
    ) -> Result<ActivationPrecision> {
        match operation {
            PlanOperation::TypedBulkF32Input { operand } => {
                bulk_f32::input_precision(self, operand)
            }
            PlanOperation::TypedBulkF32Matmul { matrix } => {
                bulk_f32::matrix_precision(self, matrix, inputs)
            }
            PlanOperation::TypedEmbed {
                tensor,
                contract: EmbeddingContract::Bf16OutputInput,
                ..
            } => {
                typed::validate_bf16_matrix(tensor)?;
                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1 {
                    return Err(Error::Unsupported(
                        "typed embedding numerical profile".into(),
                    ));
                }
                Ok(ActivationPrecision::Bf16WidenedF32)
            }
            PlanOperation::TypedProject {
                tensor,
                contract: ProjectionContract::TorchCpuBf16Avx2,
            } => {
                typed::validate_bf16_matrix(tensor)?;
                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
                    || inputs.len() != 1
                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
                {
                    return Err(Error::Unsupported(
                        "typed projection numerical profile/precision".into(),
                    ));
                }
                Ok(ActivationPrecision::Bf16WidenedF32)
            }
            PlanOperation::TypedOrdinaryRms {
                parameter,
                contract,
            } => {
                if self.profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
                    || parameter.source != self.source
                    || inputs.len() != 1
                    || inputs[0].precision != ActivationPrecision::Bf16WidenedF32
                {
                    return Err(Error::Unsupported(
                        "ordinary RMS numerical profile/source/precision".into(),
                    ));
                }
                contract.validate(parameter, inputs[0].elements)?;
                Ok(ActivationPrecision::Bf16WidenedF32)
            }
            PlanOperation::TypedCausalConvSilu {
                parameter,
                contract,
            } => gdn_conv::result_precision(self, parameter, *contract, inputs),
            PlanOperation::View(_) | PlanOperation::Contiguous => inputs
                .first()
                .map(|value| value.precision)
                .ok_or_else(|| Error::State("missing typed view input".into())),
            PlanOperation::Observe { .. } | PlanOperation::CheckFinite { .. } => {
                Ok(ActivationPrecision::F32)
            }
            _ if self.profile == ProgramNumericalProfile::GgufSourceV1
                && inputs
                    .iter()
                    .all(|input| input.precision == ActivationPrecision::F32)
                && !matches!(
                    operation,
                    PlanOperation::TypedBulkF32Input { .. }
                        | PlanOperation::TypedBulkF32Matmul { .. }
                        | PlanOperation::TypedEmbed { .. }
                        | PlanOperation::TypedProject { .. }
                        | PlanOperation::TypedOrdinaryRms { .. }
                        | PlanOperation::TypedCausalConvSilu { .. }
                ) =>
            {
                Ok(ActivationPrecision::F32)
            }
            _ => Err(Error::Unsupported(
                "operator lacks explicit numerical/activation contract".into(),
            )),
        }
    }
    fn record(
        &mut self,
        name: &str,
        operation: PlanOperation,
        inputs: &[PlanValue],
        shapes: &[ValueShape],
        state: Option<StateEffect>,
    ) -> Result<Vec<PlanValue>> {
        if !matches!(operation, PlanOperation::TypedBulkF32Matmul { .. })
            && inputs.iter().any(|input| {
                self.nodes.iter().any(|node| {
                    node.outputs.contains(&input.id)
                        && matches!(
                            node.operation,
                            PlanOperation::TypedBulkF32Input { .. }
                                | PlanOperation::TypedBulkF32Matmul { .. }
                        )
                })
            })
        {
            return Err(Error::Unsupported(
                "bulk typed value cannot enter a legacy operation".into(),
            ));
        }
        for input in inputs {
            if input.owner != self.owner
                || self.values.get(input.id.0 as usize) != Some(&input.shape)
                || input.elements != input.shape.capacity_elements()?
                || self.precisions.get(input.id.0 as usize) != Some(&input.precision)
            {
                return Err(Error::State(
                    "foreign or malformed forward-plan value".into(),
                ));
            }
            if !input.contiguous && !matches!(&operation, PlanOperation::Contiguous) {
                return Err(Error::State(
                    "plan operator requires explicit contiguous input".into(),
                ));
            }
        }
        let precision = self.result_precision(&operation, inputs)?;
        let elements: Vec<_> = shapes
            .iter()
            .map(|shape| shape.capacity_elements())
            .collect::<Result<_>>()?;
        let next_count = self
            .values
            .len()
            .checked_add(shapes.len())
            .ok_or_else(|| Error::Limit("forward value count overflow".into()))?;
        if next_count > u32::MAX as usize {
            return Err(Error::Limit("forward value slot overflow".into()));
        }
        if let Some(effect) = state
            && (self.current(effect.read.slot)? != effect.read
                || effect.write.slot != effect.read.slot
                || effect.read.ordinal.checked_add(1) != Some(effect.write.ordinal))
        {
            return Err(Error::State("stale forward-plan state effect".into()));
        }
        let mut outputs = Vec::with_capacity(shapes.len());
        for (&shape, elements) in shapes.iter().zip(elements) {
            let id = ValueId(self.values.len() as u32);
            self.values.push(shape);
            self.precisions.push(precision);
            outputs.push(PlanValue {
                id,
                shape,
                owner: self.owner,
                elements,
                contiguous: true,
                precision,
            });
        }
        self.nodes.push(PlanNode {
            name: name.into(),
            operation,
            inputs: inputs.iter().map(|v| v.id).collect(),
            outputs: outputs.iter().map(|v| v.id).collect(),
            state,
        });
        if let Some(effect) = state {
            self.domains.insert(effect.write.slot, effect.write);
        }
        Ok(outputs)
    }
    fn one(
        &mut self,
        name: &str,
        operation: PlanOperation,
        inputs: &[PlanValue],
        shape: ValueShape,
        state: Option<StateEffect>,
    ) -> Result<PlanValue> {
        Ok(self
            .record(name, operation, inputs, &[shape], state)?
            .remove(0))
    }
    fn current(&self, slot: StateSlot) -> Result<StateVersion> {
        if self.advanced.is_some() {
            return Err(Error::State(
                "state access after forward-plan frontier advance".into(),
            ));
        }
        if !self.required.contains(&slot) {
            return Err(Error::State("unregistered forward state domain".into()));
        }
        Ok(self
            .domains
            .get(&slot)
            .copied()
            .unwrap_or(StateVersion { slot, ordinal: 0 }))
    }
    fn effect(&self, slot: StateSlot) -> Result<StateEffect> {
        let read = self.current(slot)?;
        if read.ordinal != 0 {
            return Err(Error::State(
                "state domain written twice in one forward step".into(),
            ));
        }
        let write = StateVersion {
            slot,
            ordinal: read
                .ordinal
                .checked_add(1)
                .ok_or_else(|| Error::Limit("state proposal ordinal overflow".into()))?,
        };
        Ok(StateEffect { read, write })
    }
    pub fn finish_operator_slice(self, output: PlanValue) -> Result<RecordedOperatorSlice> {
        if !self.operator_slice
            || !self.required.is_empty()
            || !self.domains.is_empty()
            || self.advanced.is_some()
            || output.owner != self.owner
            || self.values.get(output.id.0 as usize) != Some(&output.shape)
            || self.precisions.get(output.id.0 as usize) != Some(&output.precision)
        {
            return Err(Error::State("invalid state-free operator slice".into()));
        }
        Ok(RecordedOperatorSlice {
            output: output.id,
            plan: ForwardPlan {
                source: self.source,
                profile: self.profile,
                operator_slice: true,
                precisions: self.precisions,
                prefix_capacity: self.prefix_capacity,
                nodes: self.nodes,
                values: self.values,
                constants: self.constants,
                required_domains: Vec::new(),
                proposed_domains: Vec::new(),
                logits: None,
                frontier_delta: 0,
                state_node_count: 0,
            },
        })
    }
    pub fn finish(self, logits: Option<PlanValue>) -> Result<ForwardPlan> {
        if self.operator_slice
            || self.advanced.is_none()
            || self.domains.len() != self.required.len()
            || self
                .required
                .iter()
                .any(|slot| self.domains.get(slot).is_none_or(|v| v.ordinal != 1))
        {
            return Err(Error::State(
                "incomplete forward-plan state proposal".into(),
            ));
        }
        if let Some(value) = logits
            && (value.owner != self.owner
                || self.values.get(value.id.0 as usize) != Some(&value.shape)
                || self.precisions.get(value.id.0 as usize) != Some(&value.precision))
        {
            return Err(Error::State("foreign forward-plan logits".into()));
        }
        Ok(ForwardPlan {
            source: self.source,
            profile: self.profile,
            operator_slice: false,
            precisions: self.precisions,
            prefix_capacity: self.prefix_capacity,
            nodes: self.nodes,
            values: self.values,
            constants: self.constants,
            required_domains: self.required.into_iter().collect(),
            proposed_domains: self.domains.into_values().collect(),
            logits: logits.map(|v| v.id),
            frontier_delta: 1,
            state_node_count: self.advanced.expect("checked frontier advance"),
        })
    }
}

#[allow(clippy::too_many_arguments)] // Source closed-op inputs stay explicit.
impl MathematicalBackend for PlanBuilder {
    type Value = PlanValue;
    type Index = ParameterRef;
    fn bulk_f32_input(&mut self, operand: &BulkF32Operand) -> Result<PlanValue> {
        bulk_f32::record_input(self, operand)
    }
    fn bulk_f32_matmul(
        &mut self,
        matrix: &BulkF32Matrix,
        left: &PlanValue,
        right: &PlanValue,
    ) -> Result<PlanValue> {
        bulk_f32::record_matrix(self, matrix, left, right)
    }
    fn elements(&self, value: &PlanValue) -> usize {
        value.elements
    }
    fn precision(&self, value: &PlanValue) -> ActivationPrecision {
        value.precision
    }
    fn embed_profiled(
        &mut self,
        tensor: SourceTensor,
        token: ParameterRef,
        contract: EmbeddingContract,
    ) -> Result<PlanValue> {
        if contract == EmbeddingContract::Source {
            return self.embed(tensor, token);
        }
        self.source_tensor(&tensor)?;
        if token.0 != ParameterSlot::Token {
            return Err(Error::Malformed("typed embedding token slot".into()));
        }
        let columns = tensor.columns;
        self.one(
            "embedding",
            PlanOperation::TypedEmbed {
                tensor,
                token,
                contract,
            },
            &[],
            ValueShape::Vector(columns),
            None,
        )
    }
    fn project_profiled(
        &mut self,
        tensor: SourceTensor,
        input: &PlanValue,
        contract: ProjectionContract,
    ) -> Result<PlanValue> {
        if contract == ProjectionContract::Source {
            return self.project(tensor, input);
        }
        self.source_tensor(&tensor)?;
        if input.elements != tensor.columns {
            return Err(Error::Malformed("typed projection width".into()));
        }
        let rows = tensor.rows;
        let name = tensor.name.clone();
        self.one(
            &name,
            PlanOperation::TypedProject { tensor, contract },
            &[*input],
            ValueShape::Vector(rows),
            None,
        )
    }
    fn ordinary_rms_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &PlanValue,
        contract: OrdinaryRmsContract,
    ) -> Result<PlanValue> {
        let name = parameter.name.clone();
        self.one(
            &name,
            PlanOperation::TypedOrdinaryRms {
                parameter,
                contract,
            },
            &[*input],
            input.shape,
            None,
        )
    }
    fn causal_conv_silu_profiled(
        &mut self,
        parameter: SourceParameter,
        input: &PlanValue,
        history: &PlanValue,
        contract: CausalConvSiluContract,
    ) -> Result<CausalConvValues<PlanValue>> {
        gdn_conv::record(self, parameter, input, history, contract)
    }
    fn embed(&mut self, tensor: SourceTensor, token: ParameterRef) -> Result<PlanValue> {
        self.source_tensor(&tensor)?;
        let columns = tensor.columns;
        self.one(
            "embedding",
            PlanOperation::Embed { tensor, token },
            &[],
            ValueShape::Vector(columns),
            None,
        )
    }
    fn project(&mut self, tensor: SourceTensor, input: &PlanValue) -> Result<PlanValue> {
        self.source_tensor(&tensor)?;
        if input.elements != tensor.columns {
            return Err(Error::Malformed("plan projection width mismatch".into()));
        }
        let rows = tensor.rows;
        let name = tensor.name.clone();
        self.one(
            &name,
            PlanOperation::Project { tensor },
            &[*input],
            ValueShape::Vector(rows),
            None,
        )
    }
    fn view(&mut self, input: &PlanValue, view: TensorView) -> Result<PlanValue> {
        if !matches!(input.shape, ValueShape::Vector(_)) {
            return Err(Error::Unsupported(
                "causal rows require a bounded runtime view".into(),
            ));
        }
        if !input.contiguous {
            return Err(Error::State(
                "nested strided plan view requires packing".into(),
            ));
        }
        view.validate(input.elements)?;
        let mut output = self.one(
            "view",
            PlanOperation::View(view),
            &[*input],
            ValueShape::Vector(view.elements()?),
            None,
        )?;
        output.contiguous = view.is_contiguous();
        Ok(output)
    }
    fn contiguous(&mut self, input: &PlanValue) -> Result<PlanValue> {
        self.one(
            "contiguous",
            PlanOperation::Contiguous,
            &[*input],
            input.shape,
            None,
        )
    }
    fn rms(
        &mut self,
        name: &str,
        input: &PlanValue,
        gamma: SourceVector<'_>,
        eps: f32,
    ) -> Result<PlanValue> {
        self.grouped_rms(name, input, gamma, 1, eps)
    }
    fn grouped_rms(
        &mut self,
        name: &str,
        input: &PlanValue,
        gamma: SourceVector<'_>,
        groups: usize,
        eps: f32,
    ) -> Result<PlanValue> {
        if groups == 0
            || gamma.values.len().checked_mul(groups) != Some(input.elements)
            || !eps.is_finite()
            || eps <= 0.0
        {
            return Err(Error::Malformed(
                "plan normalization geometry invalid".into(),
            ));
        }
        let gamma = self.constant(gamma)?;
        self.one(
            name,
            PlanOperation::Rms { gamma, eps, groups },
            &[*input],
            input.shape,
            None,
        )
    }
    fn silu(
        &mut self,
        name: &str,
        input: &PlanValue,
        multiplier: Option<&PlanValue>,
    ) -> Result<PlanValue> {
        let mut inputs = vec![*input];
        if let Some(m) = multiplier {
            if m.shape != input.shape {
                return Err(Error::Malformed("plan SiLU multiplier shape".into()));
            }
            inputs.push(*m);
        }
        self.one(
            name,
            PlanOperation::Silu {
                multiplied: multiplier.is_some(),
            },
            &inputs,
            input.shape,
            None,
        )
    }
    fn add(&mut self, name: &str, left: &PlanValue, right: &PlanValue) -> Result<PlanValue> {
        if left.shape != right.shape {
            return Err(Error::Malformed("plan binary shape".into()));
        }
        self.one(name, PlanOperation::Add, &[*left, *right], left.shape, None)
    }
    fn sigmoid_mul(
        &mut self,
        name: &str,
        value: &PlanValue,
        gate: &PlanValue,
    ) -> Result<PlanValue> {
        if value.shape != gate.shape {
            return Err(Error::Malformed("plan gate shape".into()));
        }
        self.one(
            name,
            PlanOperation::SigmoidGate,
            &[*value, *gate],
            value.shape,
            None,
        )
    }
    fn conv_step(
        &mut self,
        name: &str,
        domain: ConvDomain,
        input: &PlanValue,
        weights: SourceVector<'_>,
    ) -> Result<PlanValue> {
        if weights.values.len()
            != input
                .elements
                .checked_mul(4)
                .ok_or_else(|| Error::Limit("plan convolution shape overflow".into()))?
        {
            return Err(Error::Malformed("plan convolution weights".into()));
        }
        let weights = self.constant(weights)?;
        let effect = self.effect(StateSlot::Conv(domain))?;
        self.one(
            name,
            PlanOperation::Conv { weights },
            &[*input],
            input.shape,
            Some(effect),
        )
    }
    fn coefficients(
        &mut self,
        name: &str,
        alpha: &PlanValue,
        beta: &PlanValue,
        dt: SourceVector<'_>,
        a: SourceVector<'_>,
        scale: f32,
    ) -> Result<CoefficientValues<PlanValue>> {
        if alpha.shape != beta.shape
            || alpha.elements != dt.values.len()
            || alpha.elements != a.values.len()
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(Error::Malformed("plan GDN coefficient geometry".into()));
        }
        let heads = alpha.elements;
        let dt = self.constant(dt)?;
        let a = self.constant(a)?;
        let packed = heads
            .checked_mul(3)
            .ok_or_else(|| Error::Limit("plan coefficient count overflow".into()))?;
        let output = self.record(
            name,
            PlanOperation::Coefficients { dt, a, scale },
            &[*alpha, *beta],
            &[
                ValueShape::Vector(packed),
                ValueShape::Vector(heads),
                ValueShape::Vector(heads),
            ],
            None,
        )?;
        Ok(CoefficientValues {
            packed: output[0],
            softplus: output[1],
            gate_raw: output[2],
        })
    }
    fn recurrent_step(
        &mut self,
        name: &str,
        domain: RecurrentDomain,
        q: &PlanValue,
        k: &PlanValue,
        v: &PlanValue,
        coefficients: &PlanValue,
        width: usize,
    ) -> Result<PlanValue> {
        if width == 0
            || q.shape != k.shape
            || q.shape != v.shape
            || !q.elements.is_multiple_of(width)
            || (q.elements / width).checked_mul(3) != Some(coefficients.elements)
        {
            return Err(Error::Malformed("plan recurrent geometry".into()));
        }
        let effect = self.effect(StateSlot::Recurrent(domain))?;
        self.one(
            name,
            PlanOperation::Recurrent { width },
            &[*q, *k, *v, *coefficients],
            q.shape,
            Some(effect),
        )
    }
    fn validate_kv_prefix(&mut self, domain: KvDomain, position: ParameterRef) -> Result<()> {
        let state = self.current(StateSlot::Kv(domain))?;
        if state.ordinal != 0 {
            return Err(Error::State("KV prefix guard must precede append".into()));
        }
        self.record(
            "kv-prefix",
            PlanOperation::KvPrefixGuard { state, position },
            &[],
            &[],
            None,
        )?;
        Ok(())
    }
    fn rope(
        &mut self,
        name: &str,
        q: &PlanValue,
        k: &PlanValue,
        v: &PlanValue,
        position: ParameterRef,
        geometry: RopeGeometry,
    ) -> Result<RopeValues<PlanValue>> {
        let (queries, keys) = geometry.attention.validate()?;
        if q.elements != queries
            || k.elements != keys
            || v.elements != keys
            || geometry.rotate == 0
            || geometry.rotate > geometry.attention.width
            || !geometry.rotate.is_multiple_of(2)
            || !geometry.theta_scale.is_finite()
            || geometry.theta_scale <= 0.0
            || geometry.theta_scale > 1.0
        {
            return Err(Error::Malformed("plan source RoPE geometry".into()));
        }
        let output = self.record(
            name,
            PlanOperation::Rope { geometry, position },
            &[*q, *k, *v],
            &[q.shape, k.shape, v.shape],
            None,
        )?;
        Ok(RopeValues {
            query: output[0],
            key: output[1],
            value: output[2],
        })
    }
    fn append_kv(&mut self, domain: KvDomain, k: &PlanValue, v: &PlanValue) -> Result<()> {
        if k.shape != v.shape {
            return Err(Error::Malformed("plan KV slab shape".into()));
        }
        let effect = self.effect(StateSlot::Kv(domain))?;
        self.record(
            "kv-append",
            PlanOperation::AppendKv,
            &[*k, *v],
            &[],
            Some(effect),
        )?;
        Ok(())
    }
    fn attention_qk(
        &mut self,
        name: &str,
        domain: KvDomain,
        query: &PlanValue,
        valid_tokens: ParameterRef,
        geometry: AttentionGeometry,
        scale: f32,
    ) -> Result<QkValues<PlanValue>> {
        let (queries, _) = geometry.validate()?;
        if query.elements != queries || !scale.is_finite() || scale <= 0.0 {
            return Err(Error::Malformed("plan source QK geometry".into()));
        }
        let state = self.current(StateSlot::Kv(domain))?;
        if state.ordinal != 1 {
            return Err(Error::State(
                "QK requires the private appended KV version".into(),
            ));
        }
        let shape = ValueShape::CausalRows {
            heads: geometry.query_heads,
            capacity: self.prefix_capacity,
        };
        let output = self.record(
            name,
            PlanOperation::AttentionQk {
                state,
                geometry,
                valid_tokens,
                scale,
            },
            &[*query],
            &[shape, shape],
            None,
        )?;
        Ok(QkValues {
            dots: output[0],
            scores: output[1],
        })
    }
    fn attention_softmax(
        &mut self,
        name: &str,
        scores: &PlanValue,
        heads: usize,
        valid_tokens: ParameterRef,
    ) -> Result<PlanValue> {
        if scores.shape
            != (ValueShape::CausalRows {
                heads,
                capacity: self.prefix_capacity,
            })
        {
            return Err(Error::Malformed("plan softmax shape".into()));
        }
        self.one(
            name,
            PlanOperation::AttentionSoftmax {
                heads,
                valid_tokens,
            },
            &[*scores],
            scores.shape,
            None,
        )
    }
    fn attention_value(
        &mut self,
        name: &str,
        domain: KvDomain,
        probabilities: &PlanValue,
        valid_tokens: ParameterRef,
        geometry: AttentionGeometry,
    ) -> Result<PlanValue> {
        geometry.validate()?;
        let state = self.current(StateSlot::Kv(domain))?;
        if state.ordinal != 1 {
            return Err(Error::State(
                "value matmul requires private appended KV".into(),
            ));
        }
        if probabilities.shape
            != (ValueShape::CausalRows {
                heads: geometry.query_heads,
                capacity: self.prefix_capacity,
            })
        {
            return Err(Error::Malformed("plan attention value shape".into()));
        }
        let count = geometry
            .query_heads
            .checked_mul(geometry.width)
            .ok_or_else(|| Error::Limit("plan attention output overflow".into()))?;
        self.one(
            name,
            PlanOperation::AttentionValue {
                state,
                geometry,
                valid_tokens,
                padding: 256,
            },
            &[*probabilities],
            ValueShape::Vector(count),
            None,
        )
    }
    fn assert_finite(&mut self, layer: usize, value: &PlanValue) -> Result<()> {
        self.record(
            "finite",
            PlanOperation::CheckFinite { layer },
            &[*value],
            &[],
            None,
        )?;
        Ok(())
    }
    fn observe(&mut self, layer: usize, stage: &'static str, value: &PlanValue) -> Result<()> {
        if self.capture {
            self.record(
                stage,
                PlanOperation::Observe { layer, stage },
                &[*value],
                &[],
                None,
            )?;
        }
        Ok(())
    }
    fn observe_state(
        &mut self,
        layer: usize,
        stage: &'static str,
        domain: StateSlot,
    ) -> Result<()> {
        if self.capture {
            let state = self.current(domain)?;
            self.record(
                stage,
                PlanOperation::ObserveState {
                    layer,
                    stage,
                    state,
                },
                &[],
                &[],
                None,
            )?;
        }
        Ok(())
    }
    fn advance_frontier(&mut self) -> Result<()> {
        if self.operator_slice {
            return Err(Error::State(
                "operator slice cannot advance a session frontier".into(),
            ));
        }
        if self.advanced.is_some() {
            return Err(Error::State(
                "duplicate forward-plan frontier advance".into(),
            ));
        }
        if self.domains.len() != self.required.len()
            || self
                .required
                .iter()
                .any(|slot| self.domains.get(slot).is_none_or(|v| v.ordinal != 1))
        {
            return Err(Error::State(
                "frontier advance before complete state transition".into(),
            ));
        }
        self.advanced = Some(self.nodes.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MatrixSource {
        identity: SourceIdentity,
        kind: QuantKind,
        shape: Vec<u64>,
        packed: Vec<u8>,
    }
    impl TensorSource for MatrixSource {
        fn source_identity(&self) -> SourceIdentity {
            self.identity
        }
        fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
            if name != "matrix" {
                return Err(Error::Malformed("unknown fixture tensor".into()));
            }
            Ok(SourceTensorRef {
                kind: self.kind,
                shape: &self.shape,
                packed_bytes: &self.packed,
            })
        }
    }
    fn matrix_source(shape: &[u64]) -> MatrixSource {
        MatrixSource {
            identity: source(),
            kind: QuantKind::F32,
            shape: shape.into(),
            packed: [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        }
    }

    #[test]
    fn explicit_matrix_axes_preserve_original_packed_rows_and_plan_shapes() {
        for (axes, shape) in [
            (MatrixAxes::InputOutput, [3, 2]),
            (MatrixAxes::OutputInput, [2, 3]),
        ] {
            let original = matrix_source(&shape);
            let before = original.packed.clone();
            let tensor = SourceTensor::from_source(&original, "matrix", axes).unwrap();
            assert_eq!((tensor.columns, tensor.rows, tensor.axes), (3, 2, axes));
            tensor.validate_source(&original).unwrap();
            assert_eq!(original.packed, before);
            assert_eq!(original.shape, shape);
            let mut plan = builder(vec![StateSlot::Kv(KvDomain(0))]);
            let x = plan
                .embed(tensor.clone(), ParameterRef(ParameterSlot::Token))
                .unwrap();
            assert_eq!(plan.elements(&x), 3);
            let y = plan.project(tensor.clone(), &x).unwrap();
            assert_eq!(plan.elements(&y), 2);
            assert!(matches!(
                &plan.nodes[1].operation,
                PlanOperation::Project { tensor: recorded } if recorded == &tensor
            ));
        }
    }

    #[test]
    fn source_matrix_binding_rejects_foreign_identity_axes_kind_and_bytes() {
        let original = matrix_source(&[2, 3]);
        let tensor =
            SourceTensor::from_source(&original, "matrix", MatrixAxes::OutputInput).unwrap();
        for changed in [
            SourceTensor {
                source: SourceIdentity::unbound(),
                ..tensor.clone()
            },
            SourceTensor {
                source: SourceIdentity::from_issued_id(114),
                ..tensor.clone()
            },
            SourceTensor {
                axes: MatrixAxes::InputOutput,
                ..tensor.clone()
            },
            SourceTensor {
                kind: QuantKind::F16,
                ..tensor.clone()
            },
            SourceTensor {
                columns: 2,
                rows: 3,
                ..tensor.clone()
            },
            SourceTensor {
                name: "missing".into(),
                ..tensor.clone()
            },
            SourceTensor {
                rows: 0,
                ..tensor.clone()
            },
            SourceTensor {
                columns: usize::MAX,
                rows: 2,
                ..tensor.clone()
            },
        ] {
            assert!(changed.validate_source(&original).is_err());
        }
        let mut foreign = matrix_source(&[2, 3]);
        foreign.identity = SourceIdentity::from_issued_id(114);
        assert!(tensor.validate_source(&foreign).is_err());
        for count in [23, 25] {
            let mut damaged = matrix_source(&[2, 3]);
            damaged.packed.resize(count, 0);
            assert!(tensor.validate_source(&damaged).is_err());
            assert!(
                SourceTensor::from_source(&damaged, "matrix", MatrixAxes::OutputInput).is_err()
            );
        }
    }

    #[test]
    fn source_matrix_constructor_refuses_bad_rank_overflow_and_cross_row_quant_blocks() {
        for shape in [
            vec![],
            vec![6],
            vec![1, 2, 3],
            vec![0, 6],
            vec![u64::MAX, 2],
        ] {
            let original = matrix_source(&shape);
            assert!(
                SourceTensor::from_source(&original, "matrix", MatrixAxes::OutputInput).is_err()
            );
        }
        let mut original = matrix_source(&[16, 2]);
        original.kind = QuantKind::Q8_0;
        original.packed = vec![0; 34];
        // Total source elements form one block, but output rows do not.
        assert!(original.tensor("matrix").unwrap().validate().is_ok());
        assert!(SourceTensor::from_source(&original, "matrix", MatrixAxes::InputOutput).is_err());
        let mut unbound = matrix_source(&[3, 2]);
        unbound.identity = SourceIdentity::unbound();
        assert!(SourceTensor::from_source(&unbound, "matrix", MatrixAxes::InputOutput).is_err());
        assert!(
            SourceTensor::from_source(&matrix_source(&[3, 2]), "", MatrixAxes::InputOutput)
                .is_err()
        );
    }

    fn source() -> SourceIdentity {
        SourceIdentity::from_issued_id(113)
    }
    fn source_tensor(columns: usize) -> SourceTensor {
        SourceTensor {
            source: source(),
            name: "token_embd.weight".into(),
            kind: QuantKind::F32,
            axes: MatrixAxes::InputOutput,
            columns,
            rows: 4,
        }
    }
    fn builder(domains: Vec<StateSlot>) -> PlanBuilder {
        PlanBuilder::new(source(), 257, domains, true).unwrap()
    }
    fn input(builder: &mut PlanBuilder, columns: usize) -> PlanValue {
        builder
            .embed(source_tensor(columns), ParameterRef(ParameterSlot::Token))
            .unwrap()
    }

    #[test]
    fn tensor_view_bounds_overflow_and_strided_packing_are_explicit() {
        assert!(
            TensorView {
                offset: 1,
                rows: 2,
                width: 1,
                row_stride: usize::MAX
            }
            .validate(usize::MAX)
            .is_err()
        );
        assert!(
            TensorView {
                offset: 0,
                rows: 0,
                width: 1,
                row_stride: 1
            }
            .validate(1)
            .is_err()
        );
        assert!(
            TensorView {
                offset: 0,
                rows: 2,
                width: 2,
                row_stride: 1
            }
            .validate(4)
            .is_err()
        );
        assert!(TensorView::vector(1, 4).validate(4).is_err());
        let mut b = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let x = input(&mut b, 32);
        let layout = TensorView {
            offset: 1,
            rows: 4,
            width: 4,
            row_stride: 8,
        };
        let strided = b.view(&x, layout).unwrap();
        assert_eq!(b.elements(&strided), 16);
        assert!(b.view(&strided, TensorView::vector(0, 16)).is_err());
        assert!(b.add("illegal-strided-add", &strided, &strided).is_err());
        let packed = b.contiguous(&strided).unwrap();
        assert!(b.view(&packed, TensorView::vector(0, 16)).is_ok());
        assert!(
            b.nodes
                .iter()
                .any(|n| matches!(n.operation, PlanOperation::View(v) if v == layout))
        );
        assert!(
            b.nodes
                .iter()
                .any(|n| matches!(n.operation, PlanOperation::Contiguous))
        );
    }

    #[test]
    fn builder_source_value_and_constant_identities_fail_closed() {
        assert!(
            PlanBuilder::new(
                SourceIdentity::unbound(),
                1,
                vec![StateSlot::Kv(KvDomain(0))],
                false
            )
            .is_err()
        );
        assert!(PlanBuilder::new(source(), 0, vec![StateSlot::Kv(KvDomain(0))], false).is_err());
        assert!(
            PlanBuilder::new(
                source(),
                1,
                vec![StateSlot::Kv(KvDomain(0)), StateSlot::Kv(KvDomain(0))],
                false
            )
            .is_err()
        );
        let mut a = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let mut b = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let x = input(&mut a, 32);
        let foreign = input(&mut b, 32);
        assert_eq!(x.id, foreign.id);
        assert!(a.add("foreign", &x, &foreign).is_err());
        let mut wrong = source_tensor(32);
        wrong.source = SourceIdentity::from_issued_id(114);
        assert!(a.embed(wrong, ParameterRef(ParameterSlot::Token)).is_err());
        let gamma = [1.0f32; 32];
        a.rms(
            "rms",
            &x,
            SourceVector {
                source: source(),
                name: "gamma",
                values: &gamma,
            },
            1e-6,
        )
        .unwrap();
        let mut changed = gamma;
        changed[0] = -0.0;
        assert!(
            a.rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "gamma",
                    values: &changed
                },
                1e-6
            )
            .is_err()
        );
        let mut invalid = gamma;
        invalid[0] = f32::NAN;
        assert!(
            a.rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "nan-gamma",
                    values: &invalid
                },
                1e-6
            )
            .is_err()
        );
        assert!(
            a.rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "gamma",
                    values: &gamma
                },
                f32::NAN
            )
            .is_err()
        );
        assert!(
            a.grouped_rms(
                "rms",
                &x,
                SourceVector {
                    source: source(),
                    name: "gamma",
                    values: &gamma
                },
                2,
                1e-6
            )
            .is_err()
        );
    }

    #[test]
    fn state_coverage_duplicate_effects_and_frontier_are_checked() {
        let conv = StateSlot::Conv(ConvDomain(0));
        let recurrent = StateSlot::Recurrent(RecurrentDomain(0));
        let weights = [1.0f32; 128];
        let mut incomplete = builder(vec![conv, recurrent]);
        let x = input(&mut incomplete, 32);
        incomplete
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "conv.weight",
                    values: &weights,
                },
            )
            .unwrap();
        assert!(incomplete.advance_frontier().is_err());
        assert!(incomplete.finish(None).is_err());

        let mut missing_frontier = builder(vec![conv]);
        let x = input(&mut missing_frontier, 32);
        missing_frontier
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "conv.weight",
                    values: &weights,
                },
            )
            .unwrap();
        assert!(missing_frontier.finish(None).is_err());

        let mut complete = builder(vec![conv]);
        let x = input(&mut complete, 32);
        let y = complete
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "conv.weight",
                    values: &weights,
                },
            )
            .unwrap();
        assert!(
            complete
                .conv_step(
                    "conv-again",
                    ConvDomain(0),
                    &x,
                    SourceVector {
                        source: source(),
                        name: "conv.weight",
                        values: &weights
                    }
                )
                .is_err()
        );
        assert!(
            complete
                .conv_step(
                    "foreign-domain",
                    ConvDomain(2),
                    &x,
                    SourceVector {
                        source: source(),
                        name: "conv.weight",
                        values: &weights
                    }
                )
                .is_err()
        );
        complete.advance_frontier().unwrap();
        assert!(complete.advance_frontier().is_err());
        let plan = complete.finish(Some(y)).unwrap();
        assert_eq!(plan.frontier_delta(), 1);
        assert_eq!(
            plan.proposed_domains(),
            &[StateVersion {
                slot: conv,
                ordinal: 1
            }]
        );
        let effects: Vec<_> = plan.nodes().iter().filter_map(|node| node.state).collect();
        assert_eq!(effects.len(), 1);
        assert_eq!(effects[0].read.ordinal, 0);
        assert_eq!(effects[0].write.ordinal, 1);
    }

    #[test]
    fn causal_plan_uses_symbolic_prefix_and_appended_private_version() {
        let slot = StateSlot::Kv(KvDomain(0));
        let mut b = builder(vec![slot]);
        let q = input(&mut b, 64);
        let k = b.view(&q, TensorView::vector(0, 32)).unwrap();
        let parameters = PlanBuilder::inputs();
        let geometry = AttentionGeometry {
            width: 32,
            query_heads: 2,
            kv_heads: 1,
        };
        assert!(
            b.attention_qk(
                "premature",
                KvDomain(0),
                &q,
                parameters.valid_tokens,
                geometry,
                0.125
            )
            .is_err()
        );
        b.validate_kv_prefix(KvDomain(0), parameters.position)
            .unwrap();
        let rope = b
            .rope(
                "rope",
                &q,
                &k,
                &k,
                parameters.position,
                RopeGeometry {
                    attention: geometry,
                    rotate: 2,
                    theta_scale: 0.5,
                },
            )
            .unwrap();
        b.append_kv(KvDomain(0), &rope.key, &rope.value).unwrap();
        assert!(
            b.validate_kv_prefix(KvDomain(0), parameters.position)
                .is_err()
        );
        assert!(b.append_kv(KvDomain(0), &rope.key, &rope.value).is_err());
        let qk = b
            .attention_qk(
                "qk",
                KvDomain(0),
                &rope.query,
                parameters.valid_tokens,
                geometry,
                0.125,
            )
            .unwrap();
        assert_eq!(
            qk.scores.shape,
            ValueShape::CausalRows {
                heads: 2,
                capacity: 257
            }
        );
        assert!(b.view(&qk.scores, TensorView::vector(0, 2)).is_err());
        let probabilities = b
            .attention_softmax("softmax", &qk.scores, 2, parameters.valid_tokens)
            .unwrap();
        let result = b
            .attention_value(
                "value",
                KvDomain(0),
                &probabilities,
                parameters.valid_tokens,
                geometry,
            )
            .unwrap();
        b.advance_frontier().unwrap();
        let node_count = b.nodes.len();
        assert!(
            b.validate_kv_prefix(KvDomain(0), parameters.position)
                .is_err()
        );
        assert!(
            b.attention_qk(
                "late-qk",
                KvDomain(0),
                &q,
                parameters.valid_tokens,
                geometry,
                0.125
            )
            .is_err()
        );
        assert!(
            b.attention_value(
                "late-value",
                KvDomain(0),
                &probabilities,
                parameters.valid_tokens,
                geometry
            )
            .is_err()
        );
        assert!(b.observe_state(0, "late-state", slot).is_err());
        assert_eq!(b.nodes.len(), node_count);
        let plan = b.finish(Some(result)).unwrap();
        for node in plan.nodes() {
            match node.operation {
                PlanOperation::KvPrefixGuard { state, position } => {
                    assert_eq!(state.ordinal, 0);
                    assert_eq!(position, parameters.position);
                }
                PlanOperation::AttentionQk {
                    state,
                    valid_tokens,
                    ..
                }
                | PlanOperation::AttentionValue {
                    state,
                    valid_tokens,
                    ..
                } => {
                    assert_eq!(state.ordinal, 1);
                    assert_eq!(valid_tokens, parameters.valid_tokens);
                }
                _ => {}
            }
        }
        assert_eq!(
            plan.values()[plan.logits().unwrap().0 as usize],
            ValueShape::Vector(64)
        );
    }

    #[test]
    fn state_prefix_keeps_checks_and_excludes_optional_output_math() {
        let mut b = builder(vec![StateSlot::Conv(ConvDomain(0))]);
        let x = input(&mut b, 32);
        b.conv_step(
            "conv",
            ConvDomain(0),
            &x,
            SourceVector {
                source: source(),
                name: "conv.weight",
                values: &[1.0; 128],
            },
        )
        .unwrap();
        b.assert_finite(0, &x).unwrap();
        b.observe(0, "state-stage-output", &x).unwrap();
        b.advance_frontier().unwrap();
        let count = b.nodes.len();
        let output = b.project(source_tensor(32), &x).unwrap();
        let plan = b.finish(Some(output)).unwrap();
        assert_eq!(plan.state_nodes().len(), count);
        assert!(matches!(
            plan.state_nodes()[count - 2].operation,
            PlanOperation::CheckFinite { .. }
        ));
        assert!(matches!(
            plan.state_nodes()[count - 1].operation,
            PlanOperation::Observe { .. }
        ));
        assert_eq!(plan.nodes().len(), count + 1);
        assert_eq!(plan.proposed_domains().len(), 1);
    }

    #[test]
    fn attention_geometry_and_coefficient_layout_refuse_overflow() {
        assert!(
            AttentionGeometry {
                width: usize::MAX,
                query_heads: 2,
                kv_heads: 1
            }
            .validate()
            .is_err()
        );
        assert!(
            AttentionGeometry {
                width: 32,
                query_heads: 3,
                kv_heads: 2
            }
            .validate()
            .is_err()
        );
        let mut b = builder(vec![StateSlot::Kv(KvDomain(0))]);
        let q = input(&mut b, 64);
        let k = b.view(&q, TensorView::vector(0, 32)).unwrap();
        let geometry = AttentionGeometry {
            width: 32,
            query_heads: 2,
            kv_heads: 1,
        };
        let position = ParameterRef(ParameterSlot::Position);
        assert!(
            b.rope(
                "odd-rope",
                &q,
                &k,
                &k,
                position,
                RopeGeometry {
                    attention: geometry,
                    rotate: 3,
                    theta_scale: 0.5
                }
            )
            .is_err()
        );
        assert!(
            b.rope(
                "nan-rope",
                &q,
                &k,
                &k,
                position,
                RopeGeometry {
                    attention: geometry,
                    rotate: 2,
                    theta_scale: f32::NAN
                }
            )
            .is_err()
        );
        let constants = [1.0f32; 64];
        assert!(
            b.coefficients(
                "wrong-heads",
                &q,
                &k,
                SourceVector {
                    source: source(),
                    name: "dt",
                    values: &constants
                },
                SourceVector {
                    source: source(),
                    name: "a",
                    values: &constants
                },
                0.125
            )
            .is_err()
        );
    }

    #[test]
    fn public_value_retargeting_cannot_reuse_cached_capacity() {
        let mut plan = builder(vec![StateSlot::Conv(ConvDomain(0))]);
        let mut narrow_tensor = source_tensor(32);
        narrow_tensor.name = "narrow_embedding".into();
        let narrow = plan
            .embed(narrow_tensor, ParameterRef(ParameterSlot::Token))
            .unwrap();
        let mut wide_tensor = source_tensor(256);
        wide_tensor.name = "wide_embedding".into();
        let wide = plan
            .embed(wide_tensor, ParameterRef(ParameterSlot::Token))
            .unwrap();
        let mut projection = source_tensor(256);
        projection.name = "projection".into();
        plan.project(projection.clone(), &wide).unwrap();

        // Both public fields now name the honest narrow value, but copying
        // the wide value retained its private 256-element capacity cache.
        let mut retargeted = wide;
        retargeted.id = narrow.id;
        retargeted.shape = narrow.shape;
        assert_eq!(plan.elements(&retargeted), 256);
        let values_before = plan.values.clone();
        let nodes_before = plan.nodes.len();
        let domains_before = plan.domains.clone();
        assert!(matches!(
            plan.project(projection, &retargeted),
            Err(Error::State(_))
        ));
        assert_eq!(plan.values, values_before);
        assert_eq!(plan.nodes.len(), nodes_before);
        assert_eq!(plan.domains, domains_before);

        // Honest descriptors remain usable after the refusal.
        let logits = plan.project(source_tensor(32), &narrow).unwrap();
        assert_eq!(plan.elements(&logits), 4);
        assert_eq!(plan.nodes.len(), nodes_before + 1);
        assert_eq!(plan.domains, domains_before);
    }

    #[test]
    fn failed_record_never_advances_state_or_leaves_partial_outputs() {
        let slot = StateSlot::Conv(ConvDomain(0));
        let mut a = builder(vec![slot]);
        let mut b = builder(vec![slot]);
        let x = input(&mut a, 32);
        let foreign = input(&mut b, 32);
        let weights = [1.0f32; 128];
        assert!(
            a.conv_step(
                "foreign-conv",
                ConvDomain(0),
                &foreign,
                SourceVector {
                    source: source(),
                    name: "weights",
                    values: &weights
                }
            )
            .is_err()
        );
        assert_eq!(a.current(slot).unwrap().ordinal, 0);
        let previous_values = a.values.len();
        assert!(
            a.record(
                "partial-output",
                PlanOperation::Contiguous,
                &[x],
                &[ValueShape::Vector(32), ValueShape::Vector(0)],
                None
            )
            .is_err()
        );
        assert_eq!(a.values.len(), previous_values);
        let output = a
            .conv_step(
                "conv",
                ConvDomain(0),
                &x,
                SourceVector {
                    source: source(),
                    name: "weights",
                    values: &weights,
                },
            )
            .unwrap();
        assert_eq!(a.current(slot).unwrap().ordinal, 1);
        let stale = StateEffect {
            read: StateVersion { slot, ordinal: 0 },
            write: StateVersion { slot, ordinal: 1 },
        };
        assert!(
            a.record(
                "stale-effect",
                PlanOperation::Contiguous,
                &[x],
                &[x.shape],
                Some(stale)
            )
            .is_err()
        );
        a.advance_frontier().unwrap();
        let plan = a.finish(Some(output)).unwrap();
        assert_eq!(
            plan.nodes()
                .iter()
                .filter(|node| node.state.is_some())
                .count(),
            1
        );
    }
}

/// Source-numeric diagnostic record shared by family and prepared execution.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PlanObservation {
    pub position: usize,
    pub layer: usize,
    pub stage: &'static str,
    pub values: Vec<f32>,
}
impl std::fmt::Debug for PlanObservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerObservation")
            .field("position", &self.position)
            .field("layer", &self.layer)
            .field("stage", &self.stage)
            .field("values", &self.values)
            .finish()
    }
}

#[cfg(test)]
#[path = "program/typed_tests.rs"]
mod typed_tests;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/proposed/crates/zerogravity-gpu/src/resident_program/batch.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 73872; SHA256 `8c85e2ec2b465e79cdaf4ae97cd7fa27a0ec49ad3919b33722a9e46175dccd34`.

````text
//! Batch-shaped physical lowering of the original, source-bound ForwardPlan.
//!
//! No model schedule, session authority or Ready constructor lives here. The
//! physical owner admits every allocation, retains original leases before
//! launch, and publishes independent roots only after real completion.

use super::*;
use crate::activation_quant::{self, PARAMETER_WORDS};
use crate::activation_quant_fused::{BlockBindings, PreparedBlocks};
use crate::resident_state_transport::{
    Direction, PreparedCertifiedScatter, PreparedTransport, Row, Shape,
};

enum PreparedStateTransport {
    Opaque(PreparedTransport),
    Certified(PreparedCertifiedScatter),
}
impl PreparedStateTransport {
    fn launch(&self, client: &Client) -> Result<usize> {
        match self {
            Self::Opaque(transport) => transport.launch(client),
            Self::Certified(transport) => transport.launch(client),
        }
    }
}
use crate::source_float::{round_source, widen_source};

fn transport_shape(layout: StateLayout) -> Shape {
    match layout {
        StateLayout::Vector { elements } => Shape::Vector { words: elements },
        StateLayout::Kv { slab, capacity } => Shape::Kv { slab, capacity },
    }
}

/// One original plan, shared immutable tensors, and a private B-shaped workspace.
pub(crate) struct BatchProgram {
    inner: ResidentProgram,
    rows: usize,
    nodes: Vec<PlanNode>,
    packed: Vec<StateBinding>,
    positions: Handle,
    greedy: [Handle; 3],
    projection_blocks: Vec<Option<PreparedBlocks>>,
    owned: Vec<Handle>,
    #[cfg(all(feature = "qualification", feature = "cuda"))]
    graph_generation: u64,
    #[cfg(all(feature = "qualification", feature = "cuda"))]
    graph_pin: Arc<()>,
}

/// Physical kernel launch sites in the existing node body, not graph enqueues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BatchBodyReceipt {
    pub dispatches: usize,
    pub fused_dispatches: usize,
}

#[cfg(all(feature = "qualification", feature = "cuda"))]
mod graph;
#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) use graph::{BatchGraphProfile, BatchGraphReplay};
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
mod state_transport_qualification;

impl BatchProgram {
    /// New allocations only: the original source handles remain shared.
    /// External per-session prior/candidate allocations are reported separately.
    pub fn preflight(original: &ResidentProgram, rows: usize) -> Result<ResidentPreflight> {
        ResidentProgram::require_legacy_contract(&original.plan)?;
        if !matches!(rows, 2 | 4 | 8) {
            return Err(Error::Unsupported(
                "resident native batch rows must be 2/4/8".into(),
            ));
        }
        let mut budget = original.preflight.clone();
        let scale = |n: usize| {
            n.checked_mul(rows)
                .ok_or_else(|| Error::Limit("resident batch allocation overflow".into()))
        };
        // Device indexing uses u32 element offsets throughout the original ABI.
        for shape in original.plan.values() {
            bytes(scale(shape.capacity_elements()?)?, 4)?;
        }
        for state in &budget.states {
            bytes(scale(state.bytes / 4)?, 4)?;
            crate::resident_state_transport::geometry(transport_shape(state.layout), rows)?;
        }
        budget.source_bytes = 0;
        budget.value_bytes = scale(budget.value_bytes)?;
        budget.status_bytes = checked_sum([scale(budget.status_bytes - 4)?, 4])?;
        budget.parameter_bytes = bytes(rows, 8)?;
        budget.prior_state_bytes = scale(budget.prior_state_bytes)?;
        budget.candidate_state_bytes = scale(budget.candidate_state_bytes)?;
        for size in &mut budget.scratch {
            *size = scale(*size)?;
            bytes(*size / 4, 4)?;
        }
        // Packed copies are scratch, never a shared mutable StateRoot. Both
        // complete cohorts are charged in addition to the independent roots.
        budget.workspace_bytes = checked_sum([
            checked_sum(budget.scratch)?,
            budget.prior_state_bytes,
            budget.candidate_state_bytes,
            checked_sum(Self::greedy_sizes(original, rows)?)?,
        ])?;
        Ok(budget)
    }

    fn greedy_sizes(original: &ResidentProgram, rows: usize) -> Result<[usize; 3]> {
        let logits = original
            .plan
            .logits()
            .ok_or_else(|| Error::Unsupported("native batch requires planned logits".into()))?;
        let vocabulary = original.plan.values()[logits.0 as usize].capacity_elements()?;
        crate::resident_batch_greedy::workspace_bytes(vocabulary, rows)
    }

    pub fn prepare(
        client: &Client,
        original: &ResidentProgram,
        rows: usize,
        mut allocate: impl FnMut(ResidentBufferRole, usize, AllocationInit<'_>) -> Result<Handle>,
    ) -> Result<Self> {
        let preflight = Self::preflight(original, rows)?;
        ResidentProgram::ensure_capability(client)?;
        if !crate::activation_quant_fused::is_cuda_service(client) {
            return Err(Error::Unsupported(
                "native source batch requires CUDA lowering".into(),
            ));
        }
        for state in &preflight.states {
            crate::resident_state_transport::ensure_certified_capability(
                client,
                transport_shape(state.layout),
                rows,
            )?;
        }
        let mut owned = Vec::new();
        let mut recorded =
            |role: ResidentBufferRole, n: usize, init: AllocationInit<'_>| -> Result<Handle> {
                let handle = allocate(role, n, init)?;
                owned.push(handle.clone());
                Ok(handle)
            };
        let mut allocator = ProgramAllocator {
            client,
            allocate: &mut recorded,
        };
        let aliases = contiguous_aliases(&original.plan)?;
        let mut values: Vec<DeviceBuffer> = Vec::with_capacity(original.values.len());
        for (shape, alias) in original.plan.values().iter().zip(aliases) {
            let elements = rows * shape.capacity_elements()?;
            values.push(if let Some(original) = alias {
                values[original.0 as usize].clone()
            } else {
                DeviceBuffer::new(
                    allocator.empty(ResidentBufferRole::Value, bytes(elements, 4)?)?,
                    elements,
                )
            });
        }
        let scratch: [Handle; SCRATCH_SLOTS] = (0..SCRATCH_SLOTS)
            .map(|i| allocator.empty(ResidentBufferRole::Workspace, preflight.scratch[i]))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("fixed scratch slot count");
        let greedy: [Handle; 3] = Self::greedy_sizes(original, rows)?
            .into_iter()
            .map(|size| allocator.empty(ResidentBufferRole::Workspace, size))
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .expect("fixed greedy workspace count");
        let node_status = original
            .plan
            .nodes()
            .iter()
            .map(|node| {
                let (blocks, outputs) = match &node.operation {
                    PlanOperation::Embed { tensor, .. } => (rows * tensor.columns, 0),
                    PlanOperation::Project { tensor } => original.tensors[&tensor.name]
                        .layout
                        .map_or((0, 0), |l| (rows * l.blocks, rows * l.rows)),
                    _ => (0, 0),
                };
                Ok(NodeStatus {
                    block: if blocks == 0 {
                        None
                    } else {
                        Some(allocator.empty(ResidentBufferRole::Status, bytes(blocks, 4)?)?)
                    },
                    row: if outputs == 0 {
                        None
                    } else {
                        Some(allocator.empty(ResidentBufferRole::Status, bytes(outputs, 4)?)?)
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut packed = Vec::with_capacity(preflight.states.len());
        for requirement in &preflight.states {
            let mut state = || -> Result<StateBuffers> {
                let mut buffer = |elements| -> Result<DeviceBuffer> {
                    Ok(DeviceBuffer::new(
                        allocator.empty(ResidentBufferRole::Workspace, bytes(elements, 4)?)?,
                        elements,
                    ))
                };
                Ok(match requirement.layout {
                    StateLayout::Vector { elements } => {
                        StateBuffers::Vector(buffer(rows * elements)?)
                    }
                    StateLayout::Kv { slab, capacity } => StateBuffers::Kv {
                        keys: buffer(rows * slab * capacity)?,
                        values: buffer(rows * slab * capacity)?,
                    },
                })
            };
            packed.push(StateBinding {
                slot: requirement.slot,
                prior: state()?,
                candidate: state()?,
                prior_tokens: 0,
            });
        }
        let tensors = original
            .tensors
            .iter()
            .map(|(name, tensor)| {
                (
                    name.clone(),
                    ResidentTensor {
                        packed: tensor.packed.clone(),
                        elements: tensor.elements,
                        scales: tensor.scales.clone(),
                        biases: tensor.biases.clone(),
                        layout: tensor.layout,
                    },
                )
            })
            .collect();
        let mut nodes = original.plan.nodes().to_vec();
        for node in &mut nodes {
            match &mut node.operation {
                PlanOperation::Rms { groups, .. }
                | PlanOperation::AttentionSoftmax { heads: groups, .. } => *groups *= rows,
                PlanOperation::Rope { geometry, .. } => {
                    geometry.attention.query_heads *= rows;
                    geometry.attention.kv_heads *= rows;
                }
                PlanOperation::AttentionQk { geometry, .. }
                | PlanOperation::AttentionValue { geometry, .. } => {
                    geometry.query_heads *= rows;
                    geometry.kv_heads *= rows;
                }
                _ => {}
            }
        }
        let positions = allocator.empty(ResidentBufferRole::Parameter, bytes(rows, 4)?)?;
        let mut inner = ResidentProgram {
            _source_owner: Arc::clone(&original._source_owner),
            plan: original.plan.clone(),
            preflight,
            tensors,
            constants: original.constants.clone(),
            values,
            scratch,
            status: allocator.empty(ResidentBufferRole::Status, 4)?,
            node_status,
            // Quantized batch input preparation below binds existing originals.
            projections: Vec::new(),
            token: allocator.empty(ResidentBufferRole::Parameter, bytes(rows, 4)?)?,
            exp_tables: original.exp_tables.clone(),
            trig_tables: original.trig_tables.clone(),
            owned_ranges: OwnedRanges::new(client.service_id()),
        };
        for handle in inner.owned_handles() {
            inner.owned_ranges.insert(BufferRange::new(&handle)?)?;
        }
        for handle in &greedy {
            inner.owned_ranges.insert(BufferRange::new(handle)?)?;
        }
        inner.owned_ranges.insert(BufferRange::new(&positions)?)?;
        for binding in &packed {
            for handle in binding
                .prior
                .handles()
                .into_iter()
                .chain(binding.candidate.handles())
            {
                inner.owned_ranges.insert(BufferRange::new(&handle)?)?;
            }
        }
        let projection_blocks = inner
            .plan
            .nodes()
            .iter()
            .enumerate()
            .map(|(index, node)| {
                let PlanOperation::Project { tensor } = &node.operation else {
                    return Ok(None);
                };
                if tensor.kind == QuantKind::F32 {
                    return Ok(None);
                }
                let layout = inner.tensors[&tensor.name]
                    .layout
                    .ok_or_else(|| Error::State("batch projection layout absent".into()))?;
                let flags = &inner.node_status[index];
                PreparedBlocks::prepare_cuda(
                    client,
                    &layout,
                    rows,
                    &BlockBindings {
                        input: &inner.values[node.inputs[0].0 as usize].handle,
                        parameters: &inner.scratch[0],
                        scaled: &inner.scratch[1],
                        codes: &inner.scratch[2],
                        block_status: flags.block.as_ref().ok_or_else(|| {
                            Error::State("batch projection block status absent".into())
                        })?,
                    },
                )
                .map(Some)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            inner,
            rows,
            nodes,
            packed,
            positions,
            greedy,
            projection_blocks,
            owned,
            #[cfg(all(feature = "qualification", feature = "cuda"))]
            graph_generation: graph::next_generation()?,
            #[cfg(all(feature = "qualification", feature = "cuda"))]
            graph_pin: Arc::new(()),
        })
    }

    pub fn budget(&self) -> &ResidentPreflight {
        &self.inner.preflight
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn greedy_workspace(&self) -> &[Handle; 3] {
        &self.greedy
    }

    pub fn logits(&self) -> Option<DeviceBuffer> {
        self.inner
            .plan
            .logits()
            .map(|id| self.inner.values[id.0 as usize].clone())
    }

    pub fn status(&self) -> Handle {
        self.inner.status.clone()
    }

    pub fn owned_handles(&self) -> Vec<Handle> {
        self.owned.clone()
    }

    pub fn execution_handles(&self, output: ResidentOutput) -> Vec<Handle> {
        let mut handles = self.inner.execution_handles(output);
        handles.push(self.positions.clone());
        for binding in &self.packed {
            handles.extend(binding.prior.handles());
            handles.extend(binding.candidate.handles());
        }
        if output == ResidentOutput::Logits {
            handles.extend(self.greedy.iter().cloned());
        }
        handles
    }

    pub fn validate_runtime(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
    ) -> Result<()> {
        if parameters.len() != self.rows || bindings.len() != self.rows {
            return Err(Error::Malformed("resident batch row count changed".into()));
        }
        let mut priors = Vec::new();
        let mut candidates = Vec::new();
        for (parameter, binding) in parameters.iter().zip(bindings) {
            self.inner.validate_runtime(client, *parameter, binding)?;
            for node in self.inner.plan.nodes() {
                if let PlanOperation::Embed { tensor, .. } = &node.operation
                    && parameter.token as usize >= tensor.rows
                {
                    return Err(Error::Malformed(
                        "native batch token exceeds source vocabulary".into(),
                    ));
                }
            }
            for state in binding {
                for handle in state.prior.handles() {
                    priors.push(BufferRange::new(&handle)?);
                }
                for handle in state.candidate.handles() {
                    candidates.push(BufferRange::new(&handle)?);
                }
            }
        }
        for (i, candidate) in candidates.iter().enumerate() {
            if priors
                .iter()
                .chain(candidates[..i].iter())
                .any(|other| candidate.overlaps(other))
            {
                return Err(Error::State(
                    "native batch candidate aliases another row".into(),
                ));
            }
        }
        // Both directions and every domain are checked before the enclosing
        // original physical submission begins. No allocation/upload occurs.
        self.prepare_state_transports(client, bindings, false)?;
        self.prepare_state_transports(client, bindings, true)?;
        Ok(())
    }

    /// ONE traversal of the original plan. Arithmetic stages cover every row;
    /// private physical state handles are only gathered/scattered, never shared.
    pub fn launch(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(client, parameters, bindings, None, output)
    }

    /// Trusted owner checks each token's producer RootBinding and retains the
    /// independent original packet leases before this physical preflight.
    pub fn validate_device_tokens(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        tokens: &[DeviceBuffer],
    ) -> Result<()> {
        self.validate_runtime(client, parameters, bindings)?;
        if tokens.len() != self.rows {
            return Err(Error::Malformed("native device-token row count".into()));
        }
        let mut ranges = Vec::with_capacity(tokens.len());
        for token in tokens {
            let range = token.check(client, 1)?;
            if self.inner.owned_ranges.overlaps(&range)
                || ranges
                    .iter()
                    .any(|prior: &BufferRange| range.overlaps(prior))
            {
                return Err(Error::State(
                    "native device-token aliases program/another row".into(),
                ));
            }
            for row in bindings {
                for binding in row {
                    for handle in binding.candidate.handles() {
                        if range.overlaps(&BufferRange::new(&handle)?) {
                            return Err(Error::State(
                                "native device-token aliases proposed state".into(),
                            ));
                        }
                    }
                }
            }
            ranges.push(range);
        }
        Ok(())
    }

    /// One native gather reads the row-bound original device packets. It never
    /// serializes their token IDs through host scalars or uploads a token.
    pub fn launch_device_tokens(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        tokens: &[DeviceBuffer],
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        self.launch_impl(client, parameters, bindings, Some(tokens), output)
    }

    fn launch_impl(
        &mut self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        device_tokens: Option<&[DeviceBuffer]>,
        output: ResidentOutput,
    ) -> Result<ResidentSubmission> {
        if let Some(tokens) = device_tokens {
            self.validate_device_tokens(client, parameters, bindings, tokens)?;
        } else {
            self.validate_runtime(client, parameters, bindings)?;
        }
        let prefix_dispatches = self.launch_inputs(client, parameters, bindings, device_tokens)?;
        let body = self.launch_body(client, parameters, output)?;
        self.launch_outputs(client, bindings, output, prefix_dispatches, body)
    }

    fn launch_inputs(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        bindings: &[Vec<StateBinding>],
        device_tokens: Option<&[DeviceBuffer]>,
    ) -> Result<usize> {
        unsafe {
            crate::program_kernels::clear_step_status::launch_unchecked(
                client,
                grid(1),
                threads(),
                arg(&self.inner.status, 1),
            );
            if let Some(tokens) = device_tokens {
                // Inactive read-only arguments reuse row zero; compile-time
                // row bounds prevent them from participating in the gather.
                let packet = |row: usize| &tokens.get(row).unwrap_or(&tokens[0]).handle;
                batch_device_tokens::launch_unchecked(
                    client,
                    grid(self.rows),
                    threads(),
                    arg(packet(0), 1),
                    arg(packet(1), 1),
                    arg(packet(2), 1),
                    arg(packet(3), 1),
                    arg(packet(4), 1),
                    arg(packet(5), 1),
                    arg(packet(6), 1),
                    arg(packet(7), 1),
                    arg(&self.inner.token, self.rows),
                    self.rows,
                );
            } else {
                let mut tokens = [0u32; 8];
                for (token, parameter) in tokens.iter_mut().zip(parameters) {
                    *token = parameter.token;
                }
                batch_tokens::launch_unchecked(
                    client,
                    grid(self.rows),
                    threads(),
                    arg(&self.inner.token, self.rows),
                    tokens[0],
                    tokens[1],
                    tokens[2],
                    tokens[3],
                    tokens[4],
                    tokens[5],
                    tokens[6],
                    tokens[7],
                    self.rows,
                );
            }
            let mut positions = [0u32; 8];
            for (position, parameter) in positions.iter_mut().zip(parameters) {
                *position = parameter.position as u32;
            }
            batch_tokens::launch_unchecked(
                client,
                grid(self.rows),
                threads(),
                arg(&self.positions, self.rows),
                positions[0],
                positions[1],
                positions[2],
                positions[3],
                positions[4],
                positions[5],
                positions[6],
                positions[7],
                self.rows,
            );
        }
        Ok(3 + self.transport_state(client, bindings, false)?)
    }

    /// Original nodes and their certifications, in unchanged source order.
    pub(crate) fn launch_body(
        &self,
        client: &Client,
        parameters: &[RuntimeParameters],
        output: ResidentOutput,
    ) -> Result<BatchBodyReceipt> {
        let mut dispatches = 0;
        // One actual native projection-dot launch per Project. Activation
        // preparation/status stages remain in the total dispatch count only.
        let mut fused_dispatches = 0;
        let count = output.nodes(&self.inner.plan).len();
        for (index, node) in self.nodes[..count].iter().enumerate() {
            dispatches += self.launch_node(client, index, node, parameters)?;
            if matches!(node.operation, PlanOperation::Project { .. }) {
                fused_dispatches += 1;
            }
            if matches!(node.operation, PlanOperation::View(_)) {
                continue;
            }
            for (out_index, id) in node.outputs.iter().enumerate() {
                let value = &self.inner.values[id.0 as usize];
                if let PlanOperation::AttentionQk { geometry, .. } = &node.operation
                    && out_index == 1
                {
                    unsafe {
                        crate::resident_bindings::certify_causal_rows::launch_unchecked(
                            client,
                            grid(value.elements),
                            threads(),
                            arg(&value.handle, value.elements),
                            arg(&self.inner.status, 1),
                            arg(&self.positions, self.rows),
                            geometry.query_heads / self.rows,
                            self.rows,
                            self.inner.plan.prefix_capacity(),
                        );
                    }
                } else {
                    dispatches += self.certify(client, &value.handle, value.elements);
                    continue;
                }
                dispatches += 1;
            }
        }
        Ok(BatchBodyReceipt {
            dispatches,
            fused_dispatches,
        })
    }

    fn launch_outputs(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        output: ResidentOutput,
        prefix_dispatches: usize,
        body: BatchBodyReceipt,
    ) -> Result<ResidentSubmission> {
        let mut dispatches = prefix_dispatches + body.dispatches;
        let fused_dispatches = body.fused_dispatches;
        let count = output.nodes(&self.inner.plan).len();
        dispatches += self.transport_state(client, bindings, true)?;
        let mut proposals = Vec::new();
        for row in bindings {
            for version in self.inner.plan.proposed_domains() {
                let binding = row
                    .iter()
                    .find(|b| b.slot == version.slot)
                    .expect("validated domain");
                // Typed scatter copied and certified EVERY original candidate
                // word in the same launches; keep the exact all-domain proposal.
                proposals.push((*version, binding.candidate.clone()));
            }
        }
        let mut quant_status = Vec::new();
        let mut embedding_status = Vec::new();
        for (index, node) in self.nodes[..count].iter().enumerate() {
            match &node.operation {
                PlanOperation::Project { tensor } => {
                    if let Some(layout) = self.inner.tensors[&tensor.name].layout {
                        quant_status.push((
                            self.inner.node_status[index]
                                .block
                                .as_ref()
                                .expect("quant status")
                                .clone(),
                            self.rows * layout.blocks,
                            self.inner.node_status[index]
                                .row
                                .as_ref()
                                .expect("quant row status")
                                .clone(),
                            self.rows * layout.rows,
                        ));
                    }
                }
                PlanOperation::Embed { tensor, .. } => embedding_status.push((
                    self.inner.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status")
                        .clone(),
                    self.rows * tensor.columns,
                )),
                _ => {}
            }
        }
        Ok(ResidentSubmission {
            logits: self
                .inner
                .plan
                .logits()
                .filter(|_| output == ResidentOutput::Logits)
                .map(|id| self.inner.values[id.0 as usize].clone()),
            proposals,
            status: self.inner.status.clone(),
            quant_status,
            embedding_status,
            dispatches,
            fused_dispatches,
        })
    }

    fn prepare_state_transports(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        scatter: bool,
    ) -> Result<Vec<PreparedStateTransport>> {
        if bindings.len() != self.rows || self.packed.is_empty() || self.packed.len() > 256 {
            return Err(Error::State("state transport domain/row coverage".into()));
        }
        let mut prepared = Vec::with_capacity(self.packed.len());
        for packed in &self.packed {
            let dense = if scatter {
                &packed.candidate
            } else {
                &packed.prior
            };
            let private = bindings
                .iter()
                .map(|row| {
                    let binding = row
                        .iter()
                        .find(|binding| binding.slot == packed.slot)
                        .ok_or_else(|| {
                            Error::State("state transport original domain absent".into())
                        })?;
                    Ok(if scatter {
                        &binding.candidate
                    } else {
                        &binding.prior
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            match dense {
                StateBuffers::Vector(dense) => {
                    let words = dense.elements / self.rows;
                    if words.checked_mul(self.rows) != Some(dense.elements) {
                        return Err(Error::State("state transport dense vector geometry".into()));
                    }
                    let rows = private
                        .iter()
                        .map(|value| match value {
                            StateBuffers::Vector(value) if value.elements == words => Ok(Row {
                                first: &value.handle,
                                second: None,
                            }),
                            _ => Err(Error::State("state transport vector row changed".into())),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    prepared.push(if scatter {
                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
                            client,
                            &rows,
                            &dense.handle,
                            None,
                            Shape::Vector { words },
                            &self.inner.status,
                        )?)
                    } else {
                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
                            client,
                            &rows,
                            &dense.handle,
                            None,
                            Shape::Vector { words },
                            Direction::Gather,
                        )?)
                    });
                }
                StateBuffers::Kv { keys, values } => {
                    let capacity = self.inner.plan.prefix_capacity();
                    let stride = self
                        .rows
                        .checked_mul(capacity)
                        .filter(|&n| n != 0)
                        .ok_or_else(|| Error::Limit("state transport KV stride".into()))?;
                    let slab = keys.elements / stride;
                    let words = slab
                        .checked_mul(capacity)
                        .ok_or_else(|| Error::Limit("state transport KV row words".into()))?;
                    if slab.checked_mul(stride) != Some(keys.elements)
                        || keys.elements != values.elements
                    {
                        return Err(Error::State("state transport dense KV geometry".into()));
                    }
                    let rows = private
                        .iter()
                        .map(|value| match value {
                            StateBuffers::Kv { keys, values }
                                if keys.elements == words && values.elements == words =>
                            {
                                Ok(Row {
                                    first: &keys.handle,
                                    second: Some(&values.handle),
                                })
                            }
                            _ => Err(Error::State("state transport KV row changed".into())),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    prepared.push(if scatter {
                        PreparedStateTransport::Certified(PreparedCertifiedScatter::prepare(
                            client,
                            &rows,
                            &keys.handle,
                            Some(&values.handle),
                            Shape::Kv { slab, capacity },
                            &self.inner.status,
                        )?)
                    } else {
                        PreparedStateTransport::Opaque(PreparedTransport::prepare(
                            client,
                            &rows,
                            &keys.handle,
                            Some(&values.handle),
                            Shape::Kv { slab, capacity },
                            Direction::Gather,
                        )?)
                    });
                }
            }
        }
        Ok(prepared)
    }

    fn transport_state(
        &self,
        client: &Client,
        bindings: &[Vec<StateBinding>],
        scatter: bool,
    ) -> Result<usize> {
        self.prepare_state_transports(client, bindings, scatter)?
            .iter()
            .try_fold(0, |count, transport| Ok(count + transport.launch(client)?))
    }

    fn certify(&self, client: &Client, source: &Handle, elements: usize) -> usize {
        let mut dispatches = 0;
        for offset in (0..elements).step_by(crate::program_kernels::MAX_WORD_OPS) {
            let count = (elements - offset).min(crate::program_kernels::MAX_WORD_OPS);
            unsafe {
                crate::program_kernels::certify_finite_strided::launch_unchecked(
                    client,
                    grid(count),
                    threads(),
                    arg(source, elements),
                    arg(&self.inner.status, 1),
                    offset as u32,
                    count as u32,
                    1,
                    count,
                );
            }
            dispatches += 1;
        }
        dispatches
    }

    #[allow(clippy::too_many_arguments)] // Explicit physical ranges, bounded per dispatch.
    fn copy_dense(
        &self,
        client: &Client,
        source: &Handle,
        source_elements: usize,
        source_offset: usize,
        destination: &Handle,
        destination_elements: usize,
        destination_offset: usize,
        elements: usize,
    ) -> usize {
        let mut dispatches = 0;
        for offset in (0..elements).step_by(crate::program_kernels::MAX_WORD_OPS) {
            let count = (elements - offset).min(crate::program_kernels::MAX_WORD_OPS);
            self.inner.copy::<false>(
                client,
                source,
                source_elements,
                destination,
                destination_elements,
                TensorView::vector(source_offset + offset, count),
                destination_offset + offset,
            );
            dispatches += 1;
        }
        dispatches
    }

    fn kv(&self, version: StateVersion) -> (&DeviceBuffer, &DeviceBuffer) {
        let binding = self
            .packed
            .iter()
            .find(|b| b.slot == version.slot)
            .expect("validated state domain");
        let state = if version.ordinal == 0 {
            &binding.prior
        } else {
            &binding.candidate
        };
        match state {
            StateBuffers::Kv { keys, values } => (keys, values),
            _ => unreachable!("typed KV state"),
        }
    }

    fn launch_node(
        &self,
        client: &Client,
        index: usize,
        node: &PlanNode,
        parameters: &[RuntimeParameters],
    ) -> Result<usize> {
        let p = parameters[0];
        let input = |i: usize| &self.inner.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.inner.values[node.outputs[i].0 as usize];
        let scratch = |i: usize| &self.inner.scratch[i];
        unsafe {
            match &node.operation {
                PlanOperation::TypedBulkF32Input { .. }
                | PlanOperation::TypedBulkF32Matmul { .. }
                | PlanOperation::TypedEmbed { .. }
                | PlanOperation::TypedProject { .. }
                | PlanOperation::TypedOrdinaryRms { .. }
                | PlanOperation::TypedCausalConvSilu { .. } => Err(Error::Unsupported(
                    "resident batch has no qualified typed lowering".into(),
                )),
                PlanOperation::Embed { tensor, .. } => {
                    let t = &self.inner.tensors[&tensor.name];
                    let status = self.inner.node_status[index]
                        .block
                        .as_ref()
                        .expect("embedding status");
                    let elements = self.rows * tensor.columns;
                    batch_embedding::launch_unchecked(
                        client,
                        grid(elements),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(&self.inner.token, self.rows),
                        arg(&output(0).handle, elements),
                        arg(status, elements),
                        tensor.rows,
                        tensor.columns,
                        self.rows,
                        tensor.kind == QuantKind::Q6K,
                    );
                    crate::resident_bindings::merge_status::launch_unchecked(
                        client,
                        grid(elements),
                        threads(),
                        arg(status, elements),
                        arg(&self.inner.status, 1),
                        elements,
                    );
                    Ok(2)
                }
                PlanOperation::Project { tensor } => {
                    self.project(client, index, tensor, input(0), output(0))
                }
                PlanOperation::View(view) => {
                    batch_view::launch_unchecked(
                        client,
                        grid(output(0).elements),
                        threads(),
                        arg(&input(0).handle, input(0).elements),
                        arg(&output(0).handle, output(0).elements),
                        arg(&self.inner.status, 1),
                        input(0).elements / self.rows,
                        view.offset,
                        view.rows,
                        view.width,
                        view.row_stride,
                        self.rows,
                    );
                    Ok(1)
                }
                PlanOperation::Conv { weights } => {
                    let elements = input(0).elements;
                    let effect = node.state.expect("conv effect");
                    let state = self
                        .packed
                        .iter()
                        .find(|b| b.slot == effect.read.slot)
                        .expect("conv state");
                    let (StateBuffers::Vector(prior), StateBuffers::Vector(candidate)) =
                        (&state.prior, &state.candidate)
                    else {
                        unreachable!("typed conv state")
                    };
                    batch_conv_products::launch_unchecked(
                        client,
                        grid(elements),
                        threads(),
                        arg(&input(0).handle, elements),
                        arg(
                            &self.inner.constants[weights.0 as usize],
                            elements / self.rows * 4,
                        ),
                        arg(&prior.handle, elements * 3),
                        arg(scratch(0), elements * 4),
                        arg(&candidate.handle, elements * 3),
                        elements / self.rows,
                        self.rows,
                    );
                    for level in 0..3 {
                        crate::gdn_conv::gdn_conv_reduce::launch_unchecked(
                            client,
                            grid(elements),
                            threads(),
                            arg(scratch(0), elements * 4),
                            arg(&output(0).handle, elements),
                            elements,
                            level,
                        );
                    }
                    Ok(4)
                }
                PlanOperation::Coefficients { dt, a, scale } => {
                    let heads = input(0).elements;
                    for i in 0..2 {
                        crate::resident_bindings::certify_interval::launch_unchecked(
                            client,
                            grid(heads),
                            threads(),
                            arg(&input(i).handle, heads),
                            arg(&self.inner.status, 1),
                            -128.0,
                            128.0,
                            heads,
                        );
                    }
                    batch_coefficient_input::launch_unchecked(
                        client,
                        grid(heads),
                        threads(),
                        arg(&input(0).handle, heads),
                        arg(&input(1).handle, heads),
                        arg(&self.inner.constants[dt.0 as usize], heads / self.rows),
                        arg(&self.inner.constants[a.0 as usize], heads / self.rows),
                        arg(scratch(0), heads * 4),
                        heads / self.rows,
                        self.rows,
                    );
                    for stage in 0..8 {
                        crate::gdn_coefficients::coefficient_step::launch_unchecked(
                            client,
                            grid(heads),
                            threads(),
                            arg(scratch(0), heads * 4),
                            arg(
                                &self.inner.exp_tables,
                                crate::source_transcendentals::TABLES.len(),
                            ),
                            arg(scratch(1), heads * 6),
                            arg(scratch(2), heads * 3),
                            arg(scratch(3), heads * 4),
                            heads,
                            stage,
                        );
                    }
                    crate::resident_bindings::coefficient_output::launch_unchecked(
                        client,
                        grid(heads),
                        threads(),
                        arg(scratch(3), heads * 4),
                        arg(&output(0).handle, heads * 3),
                        arg(&output(1).handle, heads),
                        arg(&output(2).handle, heads),
                        *scale,
                        heads,
                    );
                    Ok(12)
                }
                PlanOperation::Rope { geometry, .. } => self.rope(client, node, *geometry),
                PlanOperation::AppendKv => {
                    let effect = node.state.expect("KV append effect");
                    let (previous_k, previous_v) = self.kv(effect.read);
                    let (candidate_k, candidate_v) = self.kv(effect.write);
                    let cap = self.inner.plan.prefix_capacity();
                    let slab = input(0).elements / self.rows;
                    append_kv_rows::launch_unchecked(
                        client,
                        grid(self.rows * slab * cap),
                        threads(),
                        arg(&previous_k.handle, self.rows * slab * cap),
                        arg(&previous_v.handle, self.rows * slab * cap),
                        arg(&input(0).handle, self.rows * slab),
                        arg(&input(1).handle, self.rows * slab),
                        arg(&candidate_k.handle, self.rows * slab * cap),
                        arg(&candidate_v.handle, self.rows * slab * cap),
                        arg(&self.positions, self.rows),
                        slab,
                        cap,
                        self.rows,
                    );
                    Ok(1)
                }
                PlanOperation::AttentionQk {
                    state,
                    geometry,
                    scale,
                    ..
                } => {
                    let (keys, _) = self.kv(*state);
                    let cap = self.inner.plan.prefix_capacity();
                    let query = geometry.query_heads * geometry.width;
                    let key = cap * geometry.kv_heads * geometry.width;
                    let columns = geometry.query_heads * cap;
                    let mut count = self.copy_dense(
                        client,
                        &input(0).handle,
                        query,
                        0,
                        scratch(0),
                        query + key,
                        0,
                        query,
                    );
                    count += self.copy_dense(
                        client,
                        &keys.handle,
                        key,
                        0,
                        scratch(0),
                        query + key,
                        query,
                        key,
                    );
                    for block in 0..geometry.width / 32 {
                        let (previous, lanes) = if block % 2 == 0 {
                            (scratch(2), scratch(1))
                        } else {
                            (scratch(1), scratch(2))
                        };
                        for stage in 0..3 {
                            crate::attention_qk::accumulate::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(scratch(0), query + key),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(scratch(3), columns * 32),
                                arg(scratch(4), columns * 32),
                                geometry.width,
                                cap,
                                geometry.query_heads,
                                geometry.kv_heads,
                                block,
                                stage,
                            );
                            count += 1;
                        }
                    }
                    let (last, other) = if (geometry.width / 32 - 1) % 2 == 0 {
                        (scratch(1), scratch(2))
                    } else {
                        (scratch(2), scratch(1))
                    };
                    count += self
                        .inner
                        .attention_reduce(client, last, other, scratch(5), columns);
                    crate::attention_qk::scale::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(scratch(5), columns * 2),
                        *scale,
                        columns,
                    );
                    count += 1;
                    count += self.copy_dense(
                        client,
                        scratch(5),
                        columns * 2,
                        0,
                        &output(0).handle,
                        columns,
                        0,
                        columns,
                    );
                    crate::resident_bindings::masked_scores_rows::launch_unchecked(
                        client,
                        grid(columns),
                        threads(),
                        arg(scratch(5), columns * 2),
                        arg(&output(1).handle, columns),
                        arg(&self.positions, self.rows),
                        geometry.query_heads / self.rows,
                        self.rows,
                        cap,
                    );
                    Ok(count + 1)
                }
                PlanOperation::AttentionSoftmax { heads, .. } => self.inner.softmax_masked(
                    client,
                    input(0),
                    output(0),
                    *heads,
                    self.inner.plan.prefix_capacity(),
                    CausalMask::Rows {
                        positions: &self.positions,
                        heads_per_row: *heads / self.rows,
                    },
                ),
                PlanOperation::AttentionValue {
                    state,
                    geometry,
                    padding,
                    ..
                } => {
                    let (_, values) = self.kv(*state);
                    let cap = self.inner.plan.prefix_capacity();
                    let probabilities = geometry.query_heads * cap;
                    let value = cap * geometry.kv_heads * geometry.width;
                    let columns = geometry.query_heads * geometry.width;
                    let mut count = self.copy_dense(
                        client,
                        &input(0).handle,
                        probabilities,
                        0,
                        scratch(0),
                        probabilities + value,
                        0,
                        probabilities,
                    );
                    count += self.copy_dense(
                        client,
                        &values.handle,
                        value,
                        0,
                        scratch(0),
                        probabilities + value,
                        probabilities,
                        value,
                    );
                    // Preserve the original source's padded FMA count, including
                    // its signed-zero behavior beyond the active prefix.
                    let blocks = parameters
                        .iter()
                        .map(|parameter| parameter.valid_tokens.div_ceil(*padding) * padding / 32)
                        .max()
                        .expect("validated native rows");
                    for block in 0..blocks {
                        let (previous, lanes) = if block % 2 == 0 {
                            (scratch(2), scratch(1))
                        } else {
                            (scratch(1), scratch(2))
                        };
                        for stage in 0..3 {
                            crate::attention_value::accumulate_rows::launch_unchecked(
                                client,
                                grid(columns * 32),
                                threads(),
                                arg(scratch(0), probabilities + value),
                                arg(previous, columns * 32),
                                arg(lanes, columns * 32),
                                arg(scratch(3), columns * 32),
                                arg(scratch(4), columns * 32),
                                arg(&self.positions, self.rows),
                                block as u32,
                                geometry.width,
                                cap,
                                geometry.query_heads,
                                geometry.kv_heads,
                                self.rows,
                                *padding,
                                stage,
                            );
                            count += 1;
                        }
                    }
                    let (last, other) = if (blocks - 1) % 2 == 0 {
                        (scratch(1), scratch(2))
                    } else {
                        (scratch(2), scratch(1))
                    };
                    Ok(count
                        + self.inner.attention_reduce(
                            client,
                            last,
                            other,
                            &output(0).handle,
                            columns,
                        ))
                }
                PlanOperation::CheckFinite { .. } => {
                    Ok(self.certify(client, &input(0).handle, input(0).elements))
                }
                // Original source kernels, accumulation order, physical rounding
                // boundaries and attention padding remain the authority.
                _ => self.inner.launch_node(client, index, node, p, &self.packed),
            }
        }
    }

    fn rope(&self, client: &Client, node: &PlanNode, geometry: RopeGeometry) -> Result<usize> {
        let input = |i: usize| &self.inner.values[node.inputs[i].0 as usize];
        let output = |i: usize| &self.inner.values[node.outputs[i].0 as usize];
        let scratch = |i: usize| &self.inner.scratch[i];
        let g = geometry.attention;
        let heads = g.query_heads + g.kv_heads;
        let half = geometry.rotate / 2;
        let q = input(0).elements;
        let k = input(1).elements;
        let count = q + k + input(2).elements;
        let ordered_angles = crate::rope_angles::PreparedAngles::rows_for_cuda(
            client,
            scratch(1),
            &self.positions,
            geometry.theta_scale,
            half,
            self.rows,
        )?;
        let mut dispatches = 0;
        for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
            dispatches += self.copy_dense(
                client,
                &input(i).handle,
                input(i).elements,
                0,
                scratch(0),
                count,
                offset,
                input(i).elements,
            );
        }
        unsafe {
            dispatches += if let Some(angles) = ordered_angles {
                angles.launch(client)?
            } else {
                for stage in 0..half {
                    crate::rope_kv::angle_step_rows::launch_unchecked(
                        client,
                        grid(self.rows),
                        threads(),
                        arg(scratch(1), self.rows * half),
                        arg(&self.positions, self.rows),
                        geometry.theta_scale,
                        half,
                        self.rows,
                        stage,
                    );
                }
                half
            };
            crate::source_trig::sincos_angles::launch_unchecked(
                client,
                grid(self.rows * half),
                threads(),
                arg(scratch(1), self.rows * half),
                arg(
                    &self.inner.trig_tables,
                    crate::source_trig::TRIG_TABLES.len(),
                ),
                arg(scratch(2), self.rows * half * 2),
                self.rows * half,
            );
            crate::rope_kv::rope_products::launch_unchecked(
                client,
                grid(heads * half),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                g.width,
                half,
                heads,
                g.query_heads / self.rows,
                self.rows,
            );
            crate::rope_kv::rope_sums::launch_unchecked(
                client,
                grid(count),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                arg(scratch(4), heads * half * 2),
                arg(scratch(6), count),
                g.width,
                half,
                heads,
                count,
                g.query_heads / self.rows,
                self.rows,
            );
            crate::rope_kv::rope_subtract::launch_unchecked(
                client,
                grid(heads * half * 2),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                arg(scratch(4), heads * half * 2),
                arg(scratch(5), heads * half * 2),
                g.width,
                half,
                heads,
                g.query_heads / self.rows,
                self.rows,
            );
            crate::rope_kv::rope_apply::launch_unchecked(
                client,
                grid(heads * half * 2),
                threads(),
                arg(scratch(0), count),
                arg(scratch(2), self.rows * half * 2),
                arg(scratch(3), heads * half * 2),
                arg(scratch(4), heads * half * 2),
                arg(scratch(5), heads * half * 2),
                arg(scratch(6), count),
                g.width,
                half,
                heads,
                g.query_heads / self.rows,
                self.rows,
            );
            dispatches += 5;
        }
        for (i, offset) in [(0, 0), (1, q), (2, q + k)] {
            dispatches += self.copy_dense(
                client,
                scratch(6),
                count,
                offset,
                &output(i).handle,
                output(i).elements,
                0,
                output(i).elements,
            );
        }
        Ok(dispatches)
    }

    fn project(
        &self,
        client: &Client,
        index: usize,
        tensor: &SourceTensor,
        input: &DeviceBuffer,
        output: &DeviceBuffer,
    ) -> Result<usize> {
        let t = &self.inner.tensors[&tensor.name];
        unsafe {
            if tensor.kind == QuantKind::F32 {
                crate::batch_projection::f32_source_batched_projection::launch_unchecked(
                    client,
                    grid(output.elements),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(&input.handle, input.elements),
                    arg(&output.handle, output.elements),
                    tensor.columns,
                    tensor.rows,
                    self.rows,
                    true,
                );
                return Ok(1);
            }
            let layout = t.layout.expect("source quant layout");
            let blocks = self.rows * layout.blocks;
            let columns = self.rows * layout.columns;
            let rows = self.rows * layout.rows;
            let composite = self.rows * layout.composite_count;
            let params = blocks * PARAMETER_WORDS;
            let offsets = if layout.with_bias { composite } else { 1 };
            let s = |i: usize| &self.inner.scratch[i];
            let flags = &self.inner.node_status[index];
            let block_status = flags.block.as_ref().expect("quant block status");
            let row_status = flags.row.as_ref().expect("quant row status");
            let mut dispatches = self
                .projection_blocks
                .get(index)
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::State("native projection cold binding absent".into()))?
                .launch(client)?;
            batch_composites::launch_unchecked(
                client,
                grid(rows),
                threads(),
                arg(&t.scales, layout.composite_count),
                arg(
                    &t.biases,
                    if layout.with_bias {
                        layout.composite_count
                    } else {
                        1
                    },
                ),
                arg(s(0), params),
                arg(s(3), composite),
                arg(s(4), offsets),
                arg(row_status, rows),
                layout.blocks,
                layout.rows,
                self.rows,
                layout.with_bias,
            );
            dispatches += 1;
            for (handle, count) in [(block_status, blocks), (row_status, rows)] {
                crate::resident_bindings::merge_status::launch_unchecked(
                    client,
                    grid(count),
                    threads(),
                    arg(handle, count),
                    arg(&self.inner.status, 1),
                    count,
                );
                dispatches += 1;
            }
            match tensor.kind {
                QuantKind::Q4_0 => crate::batch_quant_projection::q4_0_batched::launch_unchecked(
                    client,
                    grid(rows),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(s(3), composite),
                    arg(s(2), columns),
                    arg(&output.handle, rows),
                    layout.blocks,
                    layout.columns,
                    layout.rows,
                    self.rows,
                    true,
                ),
                QuantKind::Q4_1 => crate::batch_quant_projection::q4_1_batched::launch_unchecked(
                    client,
                    grid(rows),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(s(3), composite),
                    arg(s(4), offsets),
                    arg(s(2), columns),
                    arg(&output.handle, rows),
                    layout.blocks,
                    layout.columns,
                    layout.rows,
                    self.rows,
                    true,
                ),
                QuantKind::Q8_0 => {
                    crate::batch_quant_projection::q8_source_batched_projection::launch_unchecked(
                        client,
                        grid(rows),
                        threads(),
                        arg(&t.packed, t.elements),
                        arg(s(3), composite),
                        arg(s(2), columns),
                        arg(&output.handle, rows),
                        layout.blocks,
                        layout.columns,
                        layout.rows,
                        self.rows,
                        true,
                    )
                }
                QuantKind::Q6K => crate::batch_quant_projection::q6_k_batched::launch_unchecked(
                    client,
                    grid(rows),
                    threads(),
                    arg(&t.packed, t.elements),
                    arg(s(3), composite),
                    arg(s(2), columns),
                    arg(&output.handle, rows),
                    layout.blocks,
                    layout.columns,
                    layout.rows,
                    self.rows,
                    true,
                ),
                _ => return Err(Error::Unsupported("native resident source codec".into())),
            }
            dispatches += 1;
            Ok(dispatches)
        }
    }
}

#[cube(launch_unchecked)]
fn batch_device_tokens(
    t0: &[u32],
    t1: &[u32],
    t2: &[u32],
    t3: &[u32],
    t4: &[u32],
    t5: &[u32],
    t6: &[u32],
    t7: &[u32],
    output: &mut [u32],
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows {
        let mut token = 0u32;
        if i == 0usize {
            token = t0[0];
        }
        if i == 1usize {
            token = t1[0];
        }
        if i == 2usize {
            token = t2[0];
        }
        if i == 3usize {
            token = t3[0];
        }
        if i == 4usize {
            token = t4[0];
        }
        if i == 5usize {
            token = t5[0];
        }
        if i == 6usize {
            token = t6[0];
        }
        if i == 7usize {
            token = t7[0];
        }
        output[i] = token;
    }
}

#[cube(launch_unchecked)]
fn batch_tokens(
    output: &mut [u32],
    t0: u32,
    t1: u32,
    t2: u32,
    t3: u32,
    t4: u32,
    t5: u32,
    t6: u32,
    t7: u32,
    #[comptime] rows: usize,
) {
    let i = ABSOLUTE_POS;
    if i < rows {
        let mut value = t0;
        if i == 1usize {
            value = t1;
        }
        if i == 2usize {
            value = t2;
        }
        if i == 3usize {
            value = t3;
        }
        if i == 4usize {
            value = t4;
        }
        if i == 5usize {
            value = t5;
        }
        if i == 6usize {
            value = t6;
        }
        if i == 7usize {
            value = t7;
        }
        output[i] = value;
    }
}

#[cube(launch_unchecked)]
fn batch_view(
    input: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    #[comptime] source_row: usize,
    #[comptime] offset: usize,
    #[comptime] view_rows: usize,
    #[comptime] width: usize,
    #[comptime] stride: usize,
    #[comptime] batch: usize,
) {
    let i = ABSOLUTE_POS;
    let row_elements = view_rows * width;
    if i < batch * row_elements {
        let request = i / row_elements;
        let local = i % row_elements;
        let bits = input[request * source_row + offset + (local / width) * stride + local % width];
        output[i] = bits;
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            status[0].fetch_or(2u32);
        }
    }
}

/// Every row appends at its own position and writes canonical future KV zeros.
#[cube(launch_unchecked)]
fn append_kv_rows(
    previous_k: &[u32],
    previous_v: &[u32],
    k: &[u32],
    v: &[u32],
    candidate_k: &mut [u32],
    candidate_v: &mut [u32],
    positions: &[u32],
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] batch: usize,
) {
    let i = ABSOLUTE_POS;
    if i < capacity * batch * slab {
        let token = i / (batch * slab);
        let row = (i / slab) % batch;
        let position = positions[row] as usize;
        let mut key = 0u32;
        let mut value = 0u32;
        if token < position {
            key = previous_k[i];
            value = previous_v[i];
        } else if token == position {
            key = k[row * slab + i % slab];
            value = v[row * slab + i % slab];
        }
        candidate_k[i] = key;
        candidate_v[i] = value;
    }
}

/// Private [T,H,D] handles ↔ physically packed [T,B,H,D] workspace.
#[cfg(all(test, feature = "qualification", feature = "cuda"))]
#[cube(launch_unchecked)]
fn transport_kv(
    private_k: &mut [u32],
    private_v: &mut [u32],
    dense_k: &mut [u32],
    dense_v: &mut [u32],
    #[comptime] row: usize,
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] batch: usize,
    #[comptime] scatter: bool,
) {
    let i = ABSOLUTE_POS;
    if i < capacity * slab {
        let dense = ((i / slab) * batch + row) * slab + i % slab;
        if scatter {
            private_k[i] = dense_k[dense];
            private_v[i] = dense_v[dense];
        } else {
            dense_k[dense] = private_k[i];
            dense_v[dense] = private_v[i];
        }
    }
}

#[cube(launch_unchecked)]
fn batch_conv_products(
    activation: &[f32],
    weights: &[f32],
    history: &[f32],
    products: &mut [f32],
    candidate: &mut [f32],
    #[comptime] channels: usize,
    #[comptime] batch: usize,
) {
    let channel = ABSOLUTE_POS;
    if channel < batch * channels {
        let h = channel * 3;
        let w = (channel % channels) * 4;
        let dst = channel * 4;
        let c0 = history[h];
        let c1 = history[h + 1];
        let c2 = history[h + 2];
        let x = activation[channel];
        products[dst] = f32::cast_from(f64::cast_from(c0) * f64::cast_from(weights[w]));
        products[dst + 1] = f32::cast_from(f64::cast_from(c1) * f64::cast_from(weights[w + 1]));
        products[dst + 2] = f32::cast_from(f64::cast_from(c2) * f64::cast_from(weights[w + 2]));
        products[dst + 3] = f32::cast_from(f64::cast_from(x) * f64::cast_from(weights[w + 3]));
        candidate[h] = c1;
        candidate[h + 1] = c2;
        candidate[h + 2] = x;
    }
}

#[cube(launch_unchecked)]
fn batch_coefficient_input(
    alpha: &[u32],
    beta: &[u32],
    dt: &[u32],
    a: &[u32],
    output: &mut [u32],
    #[comptime] heads: usize,
    #[comptime] batch: usize,
) {
    let h = ABSOLUTE_POS;
    if h < batch * heads {
        output[h * 4] = alpha[h];
        output[h * 4 + 1] = beta[h];
        output[h * 4 + 2] = dt[h % heads];
        output[h * 4 + 3] = a[h % heads];
    }
}

#[cube(launch_unchecked)]
fn batch_composites(
    weight_scales: &[f32],
    biases: &[f32],
    parameters: &[f32],
    scales: &mut [f32],
    offsets: &mut [f32],
    status: &mut [u32],
    #[comptime] blocks: usize,
    #[comptime] rows: usize,
    #[comptime] batch: usize,
    #[comptime] with_bias: bool,
) {
    let row = ABSOLUTE_POS;
    if row < rows * batch {
        let request = row / rows;
        let source_row = row % rows;
        let mut flags = 0u32;
        for block in 0..blocks {
            let index = row * blocks + block;
            let source = source_row * blocks + block;
            let parameter = (request * blocks + block) * 4;
            let a = weight_scales[source];
            let b = parameters[parameter + 2];
            let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
            let scale = round_source(widen_source(a) * widen_source(b), sign);
            scales[index] = scale;
            if (u32::reinterpret(scale) & 0x7fffffffu32) >= 0x7f800000u32 {
                flags |= activation_quant::COMPOSITE_NONFINITE;
            }
            if with_bias {
                let a = biases[source];
                let b = parameters[parameter + 3];
                let sign = (u32::reinterpret(a) ^ u32::reinterpret(b)) & 0x80000000u32;
                let offset = round_source(widen_source(a) * widen_source(b), sign);
                offsets[index] = offset;
                if (u32::reinterpret(offset) & 0x7fffffffu32) >= 0x7f800000u32 {
                    flags |= activation_quant::COMPOSITE_NONFINITE;
                }
            }
        }
        status[row] = flags;
    }
}

/// Original embedding equation, with only request/token offsets added.
#[cube(launch_unchecked)]
fn batch_embedding(
    packed: &[u8],
    tokens: &[u32],
    output: &mut [f32],
    status: &mut [u32],
    #[comptime] vocabulary: usize,
    #[comptime] width: usize,
    #[comptime] batch: usize,
    #[comptime] q6_k: bool,
) {
    let at = ABSOLUTE_POS;
    if at < batch * width {
        let request = at / width;
        let i = at % width;
        let mut flags = 0u32;
        let mut decoded = 0.0f32;
        if tokens[request] >= u32::cast_from(vocabulary) {
            flags = crate::embedding::TOKEN_OUT_OF_RANGE;
        } else {
            let row = usize::cast_from(tokens[request]);
            let (scale_bits, code, local_scale) = if q6_k {
                let base = (row * (width / 256) + i / 256) * 210;
                let scale_bits = u32::cast_from(packed[base + 208])
                    | (u32::cast_from(packed[base + 209]) << 8u32);
                let half = (i % 256) / 128;
                let group = (i % 128) / 32;
                let lane = i % 32;
                let low = u32::cast_from(packed[base + half * 64 + (group % 2) * 32 + lane]);
                let high = u32::cast_from(packed[base + 128 + half * 32 + lane]);
                let mut nibble = low & 15u32;
                if group >= 2usize {
                    nibble = low >> 4u32;
                }
                let code =
                    i32::cast_from(nibble | (((high >> u32::cast_from(2 * group)) & 3u32) << 4u32))
                        - 32i32;
                let local_scale = crate::embedding::signed_byte(
                    packed[base + 192 + half * 8 + group * 2 + lane / 16],
                );
                (scale_bits, code, local_scale)
            } else {
                let base = (row * (width / 32) + i / 32) * 34;
                let scale_bits =
                    u32::cast_from(packed[base]) | (u32::cast_from(packed[base + 1]) << 8u32);
                (
                    scale_bits,
                    crate::embedding::signed_byte(packed[base + 2 + i % 32]),
                    1i32,
                )
            };
            if (scale_bits & 0x7c00u32) == 0x7c00u32 {
                flags = crate::embedding::NONFINITE_SCALE;
            } else {
                let d = crate::embedding::half_to_float(scale_bits);
                let scale = f32::cast_from(local_scale);
                let value = f32::cast_from(code);
                let sign =
                    (u32::reinterpret(d) ^ u32::reinterpret(scale) ^ u32::reinterpret(value))
                        & 0x80000000u32;
                decoded = round_source(
                    widen_source(d) * widen_source(scale) * widen_source(value),
                    sign,
                );
                if (u32::reinterpret(decoded) & 0x7f800000u32) == 0x7f800000u32 {
                    flags = crate::embedding::NONFINITE_OUTPUT;
                    decoded = 0.0f32;
                }
            }
        }
        output[at] = decoded;
        status[at] = flags;
    }
}

````

