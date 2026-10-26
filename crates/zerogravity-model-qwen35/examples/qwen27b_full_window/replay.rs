#[path = "budget.rs"]
mod budget;
#[path = "core_prefix.rs"]
mod core_prefix;
#[path = "../qwen27b_attention_tail/io.rs"]
mod io;
#[path = "reference.rs"]
mod reference;
#[path = "reference_pin.rs"]
mod reference_pin;
#[path = "../qwen27b_attention_tail/source.rs"]
mod source;
use io::*;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::cache::TransferAdmission;
use zerogravity_core::{
    Error,
    resource_arbiter::{
        AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
    },
};

fn args() -> Proof<(PathBuf, PathBuf, bool, bool)> {
    let mut args = std::env::args().skip(1);
    let (mut source, mut report, mut layer9, mut initial_core) = (None, None, false, false);
    while let Some(key) = args.next() {
        if key == "--initial-core" && !initial_core {
            initial_core = true;
            continue;
        }
        if key == "--layer9" && !layer9 {
            layer9 = true;
            continue;
        }
        let value = PathBuf::from(args.next().ok_or("argument requires value")?);
        match key.as_str() {
            "--source" if source.is_none() => source = Some(value),
            "--report" if report.is_none() => report = Some(value),
            _ => return Err("expected --source ABS --report NEW_ABS".into()),
        }
    }
    let source = source.ok_or("--source required")?;
    let report = report.ok_or("--report required")?;
    need(
        source.is_absolute() && report.is_absolute() && !report.exists(),
        "source/fresh absolute report",
    )?;
    let report = report
        .parent()
        .ok_or("report parent")?
        .canonicalize()?
        .join(report.file_name().ok_or("report filename")?);
    need(
        !report.starts_with(source.canonicalize()?),
        "report outside preserved source",
    )?;
    need(!(layer9 && initial_core), "one closed diagnostic scope")?;
    Ok((source, report, layer9, initial_core))
}
fn verify_source_paths(directory: &Path, held: &[source::Held]) -> Proof<()> {
    let pins = source::pins()?;
    let rows = pins["files"].as_array().ok_or("source registry")?;
    need(rows.len() == held.len(), "same complete source FD cohort")?;
    for (row, h) in rows.iter().zip(held) {
        let path = directory.join(string(row, "name")?);
        need(
            stamp(&h.file)? == h.before && stamp(&open(&path)?)? == h.before,
            "source FD/current path before SHA",
        )?;
    }
    source::verify_after(held)?;
    for (row, h) in rows.iter().zip(held) {
        need(
            stamp(&h.file)? == h.before
                && stamp(&open(&directory.join(string(row, "name")?))?)? == h.before,
            "source full SHA/EOF/FD/current path after computation",
        )?;
    }
    Ok(())
}
pub(super) fn run() -> Proof<()> {
    let (directory, report, layer9, initial_core) = args()?;
    if initial_core {
        return core_prefix::run(&directory, &report);
    }
    let start = std::time::Instant::now();
    let accepted = reference::accepted(reference_pin::ACTUAL_REFERENCE.as_ref())?;
    let capture = Path::new(string(&accepted["raw"], "path")?)
        .parent()
        .and_then(Path::parent)
        .ok_or("actual original capture parent")?
        .canonicalize()?;
    need(
        !report.starts_with(capture),
        "report outside preserved original capture",
    )?;
    let accepted_prefix = if layer9 {
        Some(reference::layer9::accepted()?)
    } else {
        None
    };
    let derived = budget::derived_upper_with_layer9(layer9)?;
    let arbiter = ResourceArbiter::new();
    let resource = PhysicalDeviceKey::new("host", "original27b-full-window")?;
    arbiter.register(resource.clone(), budget::HOST)?;
    let host = TransferAdmission {
        arbiter,
        resource,
        owner: ResourceOwner::new("original27b", "full64-diagnostic")?,
    };
    // Expected manifest/record tickets precede callback parsing; no whole archive buffer.
    let control = host.arbiter.admit(
        &host.resource,
        host.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: 32 * budget::MIB,
        }],
    )?;
    let (operators, held) = source::open_source(&directory, &host)?;
    let operators = Arc::new(operators);
    let source = operators.source().clone();
    let source_identity = source.source_identity();
    let mut expected = None;
    let mut prefix = None;
    let mut prefix_comparisons = Vec::with_capacity(2);
    let mut compared = 0usize;
    let mut comparisons = Vec::with_capacity(4);
    let mut committed = 0usize;
    let mut callbacks = 0usize;
    let mut max_observed = host.arbiter.snapshot(&host.resource)?.reserved;
    let mut last = None;
    let mut failure = None;
    let mut failed_root_unchanged = false;
    let mut precancel = false;
    let mut pressure_refused = false;
    for repeat in 0..2 {
        let mut session = operators.diagnostic_full_convolution_session(2, None)?;
        need(
            session.domain_count() == 112
                && session.original_convolution_cache_frames() == 4
                && !session.trained_numerical_qualified(),
            "original64/112 diagnostic schema with authentic four-frame Conv owners",
        )?;
        for layer in (0..64).filter(|layer| layer % 4 != 3) {
            let state = session.state(zerogravity_core::accelerator::program::StateSlot::Conv(
                zerogravity_core::accelerator::program::ConvDomain(layer),
            ))?;
            need(
                state.len() == 1 && state[0].len() == 10240 * 4,
                "actual retained full-four-frame owner before expensive traversal",
            )?;
        }
        if repeat == 0 {
            let base = session.binding();
            let baseline = host.arbiter.snapshot(&host.resource)?.reserved;
            precancel = matches!(session.step_traced_with(11,Some(&AtomicBool::new(true)),|_| {
                Err(Error::State("cancelled request reached callback".into()))
            }),Err(Error::State(ref why)) if why.contains("cancelled"));
            need(
                precancel
                    && session.binding() == base
                    && host.arbiter.snapshot(&host.resource)?.reserved == baseline,
                "actual pre-cancel preserves source/session/admissions",
            )?;
            let mut pressure = Vec::new();
            loop {
                let remaining = host.arbiter.snapshot(&host.resource)?.remaining;
                if remaining == 0 {
                    break;
                }
                pressure.push(source.reserve_workspace(remaining.min(32 * budget::MIB) as usize)?);
            }
            let full = host.arbiter.snapshot(&host.resource)?.reserved;
            pressure_refused = matches!(
                session.step_traced_with(11, None, |_| {
                    Err(Error::State("exhausted admission reached callback".into()))
                }),
                Err(Error::Limit(_))
            );
            need(
                pressure_refused
                    && session.binding() == base
                    && host.arbiter.snapshot(&host.resource)?.reserved == full,
                "real exhausted admission refuses before compute/root; no leaked proposal",
            )?;
            drop(pressure);
            need(
                host.arbiter.snapshot(&host.resource)?.reserved == baseline,
                "pressure reservations released",
            )?;
        }
        for (call, &token) in (if layer9 { &[11][..] } else { &[11, 353][..] })
            .iter()
            .enumerate()
        {
            let base = session.binding();
            let validate =
                |actual: &zerogravity_model_qwen35::source::binding::WindowDiagnosticReadback<
                    '_,
                >| {
                    callbacks += 1;
                    max_observed =
                        max_observed.max(host.arbiter.snapshot(&host.resource)?.reserved);
                    let checked = (|| -> Proof<reference::Comparison> {
                        if expected.is_none() {
                            expected = Some(reference::Reference::open(&accepted)?);
                        }
                        expected
                            .as_ref()
                            .ok_or("completed expected reader")?
                            .compare(repeat, call, actual)
                    })();
                    match checked {
                        Ok(observed) => {
                            compared += observed.words();
                            let mut refusal = observed.refusal().map(str::to_owned);
                            comparisons.push(observed.into_report());
                            if let Some(accepted) = &accepted_prefix {
                                let check = (|| -> Proof<_> {
                                    if prefix.is_none() {
                                        prefix = Some(reference::layer9::Prefix::open(accepted)?);
                                    }
                                    prefix
                                        .as_ref()
                                        .ok_or("completed prefix reader")?
                                        .compare(repeat, actual)
                                })();
                                match check {
                                    Ok(p) => {
                                        if let Some(why) = p.refusal() {
                                            refusal = Some(why.to_owned());
                                        }
                                        prefix_comparisons.push(p.into_report());
                                    }
                                    Err(why) => {
                                        refusal = Some(format!("prefix validation: {why}"));
                                    }
                                }
                            }
                            if let Some(why) = refusal {
                                Err(Error::State(format!(
                                    "completed original full64 comparison: {why}"
                                )))
                            } else {
                                Ok(())
                            }
                        }
                        Err(why) => Err(Error::State(format!(
                            "completed original full64 comparison: {why}"
                        ))),
                    }
                };
            let result = if layer9 {
                session.step_layer9_traced_with(token, None, validate)
            } else {
                session.step_traced_with(token, None, validate)
            };
            match result {
                Ok(output) => {
                    need(
                        output.binding() == session.binding()
                            && output.binding().frontier() == (call + 1) as u64
                            && output.logits()?.len() == 248320
                            && !output.trained_numerical_qualified(),
                        "same original publication after trusted complete comparison",
                    )?;
                    committed += 1;
                    last = Some(output);
                }
                Err(why) => {
                    failed_root_unchanged = session.binding() == base;
                    need(
                        failed_root_unchanged,
                        "failed completed validation cannot publish root",
                    )?;
                    failure = Some(why.to_string());
                    break;
                }
            }
        }
        drop(session);
        if failure.is_some() {
            break;
        }
    }
    if failure.is_none() {
        need(
            committed == if layer9 { 2 } else { 4 }
                && callbacks == committed
                && (layer9 || compared == reference::comparison_words()),
            "all two-call independent repeats compared",
        )?;
    }
    drop(operators);
    drop(source);
    let output_retained = if let Some(output) = &last {
        need(
            output.logits()?.len() == 248320
                && host.arbiter.snapshot(&host.resource)?.reserved > 32 * budget::MIB,
            "actual completed output retains its admitted original source after caller drops",
        )?;
        true
    } else {
        false
    };
    if let Some(reader) = &expected {
        reader.verify_after()?;
    }
    if let Some(reader) = &prefix {
        reader.verify_after()?;
    }
    // Full source checks also run on a numerical refusal; expected data never repairs it.
    verify_source_paths(&directory, &held)?;
    let expected_opened = expected.is_some();
    drop(last);
    drop(expected);
    drop(prefix);
    drop(held);
    drop(control);
    let end = host.arbiter.snapshot(&host.resource)?;
    need(
        end.reserved == 0 && !end.fenced,
        "all output/source/scratch reservations released",
    )?;
    write_new(
        &report,
        &json!({
            "schema":"zg.original27b-rust-full-window-diagnostic.v1",
            "status":if failure.is_none(){"PASS_DIAGNOSTIC_WORD_COMPARISON_ONLY"}else{"FAIL_DIAGNOSTIC_NUMERICAL_OR_EXECUTION"},
            "failure":failure,"snapshot":source::SNAPSHOT,"source_identity":format!("{source_identity:?}"),
            "actual_source_roles":851,"actual_source_codec":"BF16","source_headers_and_all_bytes_verified":true,
            "accepted_original_reference":accepted,"completed_callbacks":callbacks,"committed_calls":committed,
            "compared_float_words":compared,"expected_float_words":reference::comparison_words(),
            "proposal_comparisons":comparisons,"layer9_mode":layer9,
            "accepted_original_prefix":accepted_prefix,"layer9_comparisons":prefix_comparisons,"comparison_order":"all available original records before callback refusal",
            "full_hf_conv4_comparison_requested":true,"full_hf_conv4_compared":compared>0,"conv_mapping":"all four channel-major frames; exact original order",
            "kv_mapping":"original head-major to diagnostic token-major; complete defined prefix",
            "expected_opened_only_after_computation":expected_opened,
            "failed_root_unchanged":failed_root_unchanged,"pre_cancel_refused":precancel,"actual_pressure_refused":pressure_refused,
            "output_retained_after_caller_drops":output_retained,"host_limit":budget::HOST,"derived_reservation_upper":derived,
            "max_observed_boundary_reserved_not_peak":max_observed,"final_host_reserved":end.reserved,"host_fenced":end.fenced,
            "mkl_exp_producer_qualified":false,"selected_normal_exp_profile":"MklExHaSse2HaFtzOffNormalV1","whole_model_numerical_qualified":false,
            "native_serving_performance_qualified":false,"elapsed_seconds_not_benchmark":start.elapsed().as_secs_f64(),
        }),
    )?;
    if let Some(why) = failure {
        return Err(why.into());
    }
    Ok(())
}
