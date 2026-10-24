#[path = "io.rs"]
mod io;
#[path = "reference.rs"]
mod reference;
#[path = "reference_pin.rs"]
mod reference_pin;
#[path = "source.rs"]
mod source;
use io::*;
use serde_json::json;
use std::{path::PathBuf, sync::atomic::AtomicBool};
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::{
    Error,
    accelerator::program::ProgramNumericalProfile,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};
use zerogravity_model_qwen35::operator_slice::pure::PureSliceInput;
const HOST: u64 = 128 * 1024 * 1024;
fn args() -> Proof<(PathBuf, PathBuf)> {
    let mut a = std::env::args().skip(1);
    let mut source = None;
    let mut report = None;
    while let Some(k) = a.next() {
        let v = a.next().ok_or("missing argument value")?;
        match k.as_str() {
            "--source" if source.is_none() => source = Some(PathBuf::from(v)),
            "--report" if report.is_none() => report = Some(PathBuf::from(v)),
            _ => return Err("expected --source ABS --report NEW_ABS".into()),
        }
    }
    let source = source.ok_or("--source required")?;
    let report = report.ok_or("--report required")?;
    need(
        source.is_absolute() && report.is_absolute() && !report.exists(),
        "absolute original source/fresh report",
    )?;
    let dest = report
        .parent()
        .ok_or("report parent")?
        .canonicalize()?
        .join(report.file_name().ok_or("report name")?);
    need(
        !dest.starts_with(source.canonicalize()?),
        "replay report outside original source",
    )?;
    Ok((source, dest))
}
pub(super) fn run() -> Proof<()> {
    let (directory, report) = args()?;
    let start = std::time::Instant::now();
    // A real ROOT expected-only acceptance is required before learned source reads.
    let accepted = reference::accepted(reference_pin::ACTUAL_REFERENCE.as_ref())?;
    let capture = std::path::Path::new(string(&accepted["raw"], "path")?)
        .parent()
        .and_then(std::path::Path::parent)
        .ok_or("original capture parent")?
        .canonicalize()?;
    need(
        !report.starts_with(capture),
        "fresh replay report outside preserved original capture",
    )?;
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "original27b-final-tail")?;
    arbiter.register(resource.clone(), HOST)?;
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original27b", "final63-tail")?,
    };
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 2 * 1024 * 1024,
        }],
    )?;
    let reference = reference::Reference::open(&accepted)?;
    let (operators, held) = source::open_source(&directory, &host)?;
    let source = operators.source().clone();
    let identity = source.source_identity();
    let before = host.arbiter.snapshot(&host.resource)?.reserved;
    let raw = reference.input(0, 0)?;
    let input = || {
        PureSliceInput::from_bf16_bytes(
            identity,
            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
            &raw,
        )
    };
    need(
        matches!(operators.execute_output_tail(input()?,Some(&AtomicBool::new(true))),Err(Error::State(ref s))if s.contains("cancelled")),
        "pre-cancelled source pipeline refuses",
    )?;
    let mut pressure = Vec::new();
    loop {
        let remaining = host.arbiter.snapshot(&host.resource)?.remaining;
        if remaining == 0 {
            break;
        }
        pressure.push(source.reserve_workspace(remaining.min(32 * 1024 * 1024) as usize)?);
    }
    let pressure_before = host.arbiter.snapshot(&host.resource)?.reserved;
    need(
        matches!(
            operators.execute_output_tail(input()?, None),
            Err(Error::Limit(_))
        ),
        "real exhausted host source admission refuses",
    )?;
    need(
        host.arbiter.snapshot(&host.resource)?.reserved == pressure_before,
        "failed private computation released all reservations",
    )?;
    drop(pressure);
    need(
        host.arbiter.snapshot(&host.resource)?.reserved == before,
        "cancel/pressure no leaked private buffers",
    )?;
    drop(raw);
    let mut compared = 0;
    let mut last = None;
    for case in 0..4 {
        for repeat in 0..2 {
            let raw = reference.input(case, repeat)?;
            let input = PureSliceInput::from_bf16_bytes(
                identity,
                ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
                &raw,
            )?;
            let actual = operators.execute_output_tail(input, None)?;
            // Synchronous same-engine completion precedes every expected-output read.
            let _expected = source.reserve_workspace(512 * 1024)?;
            compared += reference.compare(case, repeat, &actual)?;
            last = Some(actual);
        }
    }
    need(
        compared == 2_609_152,
        "all actual input/stage/full-vocabulary words",
    )?;
    drop(operators);
    drop(source);
    let last = last.ok_or("completed source output")?;
    need(
        last.source_identity() == identity
            && last.full_vocabulary()?.len() == 248320
            && host.arbiter.snapshot(&host.resource)?.reserved > 2 * 1024 * 1024,
        "complete output retains original source after caller drops",
    )?;
    reference.verify_after()?;
    source::verify_after(&held)?;
    drop(last);
    drop(held);
    drop(reference);
    drop(control);
    let end = host.arbiter.snapshot(&host.resource)?;
    need(
        end.reserved == 0 && !end.fenced,
        "all source/private host reservations released",
    )?;
    write_new(
        &report,
        &json!({"status":"PASS_ORIGINAL27B_FINAL63_TAIL_ONLY","snapshot":source::SNAPSHOT,"accepted_reference":accepted,"cases":4,"independent_cohorts":[1,3],"repetitions":2,"compared_bf16_words":compared,"all_stages_full_vocabulary_exact":true,"source_after_full_sha_exact":true,"pre_cancel_refused":true,"real_pressure_refused":true,"output_retained_after_caller_drops":true,"final_host_reserved":end.reserved,"host_fenced":end.fenced,"full64layer_model_producer_StateRoot_GPU_serving_performance_qualified":false,"elapsed_seconds_not_benchmark":start.elapsed().as_secs_f64()}),
    )
}
