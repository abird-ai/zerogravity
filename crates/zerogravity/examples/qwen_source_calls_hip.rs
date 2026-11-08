//! Explicit original-call qualification over the ordinary resident executor.
#[path = "../tests/support/drive.rs"]
mod drive;
#[path = "../tests/support/qwen_source_call_lifetimes.rs"]
mod lifecycle;
#[path = "../tests/support/qwen_source_calls_reference.rs"]
mod reference;
#[allow(dead_code)]
#[path = "../tests/support/qwen_source.rs"]
mod source;
#[path = "../tests/support/qwen_source_budget.rs"]
mod source_budget;
#[allow(dead_code)]
#[path = "../tests/support/qwen_source_import.rs"]
mod source_import;
#[allow(dead_code)]
#[path = "../tests/support/qwen_source_pins.rs"]
mod source_pins;
use reference::{Expected, ObservationMap, Observer};
use source::{Loaded, need};
use std::{path::PathBuf, sync::Arc};
use zerogravity::{
    Result,
    accelerator::program::SourceCallMode,
    gpu::{
        Device, ResidentModel, ResidentOutput, SourceCallModel, SourceCallPreparation,
        SourceCallPublication,
    },
    qwen35::{plan::OrdinaryQwenPlan, source::binding::OriginalWindowModelSource},
};

fn selected(logits: &[f32]) -> Result<(u32, u32)> {
    need(
        logits.len() == 248320 && logits.iter().all(|v| v.is_finite()),
        "actual complete finite vocabulary",
    )?;
    let (id, maximum) = zerogravity::numerics::greedy_finite(logits)?;
    Ok((id, maximum.to_bits()))
}
async fn trajectories(
    model: &Arc<SourceCallModel>,
    plan: &OrdinaryQwenPlan,
    expected: &Expected,
    map: &ObservationMap,
    observed: bool,
) -> Result<()> {
    let mut words = 0usize;
    let mut calls = 0usize;
    let mut zeros = 0usize;
    let mut tails = 0usize;
    for case in 0..6 {
        let mut sessions = [model.session().await?, model.session().await?];
        need(
            sessions[0].binding().session() != sessions[1].binding().session(),
            "independent original session authorities",
        )?;
        let mut feedback = [None, None];
        let count = expected
            .calls()
            .iter()
            .filter(|c| c.case == case && c.repeat == 0)
            .count();
        for index in 0..count {
            for repeat in 0..2 {
                let call = expected
                    .calls()
                    .iter()
                    .find(|c| c.key() == (repeat, case, index))
                    .ok_or_else(|| zerogravity::Error::State("original call absent".into()))?;
                let peer = sessions[1 - repeat].binding();
                let session = &mut sessions[repeat];
                let before = session.binding();
                need(
                    before.frontier() == call.position as u64,
                    "actual root frontier",
                )?;
                let ids = if call.phase == "decode" {
                    vec![feedback[repeat].ok_or_else(|| {
                        zerogravity::Error::State("actual natural feedback absent".into())
                    })?]
                } else {
                    call.input_ids.clone()
                };
                need(
                    ids == call.input_ids,
                    "actual natural feedback matches original input",
                )?;
                let invocation = plan.bind_invocation(
                    if call.phase == "decode" {
                        SourceCallMode::Decode
                    } else {
                        SourceCallMode::Prefill
                    },
                    call.position,
                    ids.into(),
                    true,
                )?;
                let mut observer = observed
                    .then(|| Observer::new(expected, map, call))
                    .transpose()?;
                let pending = if let Some(observer) = &mut observer {
                    session
                        .prepare_observed(
                            invocation.bound_invocation(),
                            invocation.mode(),
                            ResidentOutput::Logits,
                            None,
                            observer,
                        )
                        .await?
                } else {
                    session
                        .prepare(
                            invocation.bound_invocation(),
                            invocation.mode(),
                            ResidentOutput::Logits,
                            None,
                        )
                        .await?
                };
                let SourceCallPreparation::Ready(pending) = pending else {
                    return Err(zerogravity::Error::State(
                        "uncancelled reference call refused".into(),
                    ));
                };
                need(
                    session.binding() == before,
                    "completed private proposal has not published",
                )?;
                if let Some(observer) = &observer {
                    observer.finish()?;
                }
                let actual = selected(&pending.readback().logits)?;
                need(
                    actual == (call.selected, call.max_bits),
                    "actual original full-vocabulary selection",
                )?;
                let SourceCallPublication::Committed(output) = session.commit(pending)? else {
                    return Err(zerogravity::Error::State(
                        "uncancelled reference call discarded".into(),
                    ));
                };
                need(
                    selected(&output.logits)? == actual
                        && session.binding().frontier() == (call.position + call.rows) as u64,
                    "same original Ready result and committed root",
                )?;
                need(
                    sessions[1 - repeat].binding() == peer,
                    "peer root unchanged by original commit",
                )?;
                feedback[repeat] = Some(actual.0);
                calls += 1;
                if let Some(observer) = observer {
                    words += observer.words;
                    zeros += observer.initial_zero_words;
                    tails += observer.kv_tail_words;
                    println!(
                        "SOURCE_CALL_PARITY_CALL repeat={repeat} case={case} call={index} base={} rows={} words={} initial_zero_words={} unused_kv_zero_words={} position_metadata_words={} mismatches=0 selected={} dispatches={}",
                        call.position,
                        call.rows,
                        observer.words,
                        observer.initial_zero_words,
                        observer.kv_tail_words,
                        observer.position_words,
                        actual.0,
                        output.dispatches
                    );
                } else {
                    let logits = expected.compare_logits(call, &output.logits)?;
                    let snapshot = sessions[repeat].snapshot()?;
                    let domains = snapshot.read_domains().await?;
                    let (state, tail) = expected.compare_domains(call, &domains, 514)?;
                    words += logits + state;
                    tails += tail;
                    println!(
                        "SOURCE_CALL_STATE_PARITY_CALL repeat={repeat} case={case} call={index} base={} rows={} logit_words={logits} state_words={state} runtime_checked_kv_tail_words={tail} mismatches=0 selected={} observer=false",
                        call.position, call.rows, actual.0
                    );
                }
                drop(output);
            }
        }
    }
    need(calls == 66, "all original trained-model calls observed")?;
    expected.unchanged()?;
    let scope = if observed {
        "SOURCE_CALL_PARITY_NUMERICS"
    } else {
        "SOURCE_CALL_STATE_PARITY_NUMERICS"
    };
    println!(
        "{scope} calls={calls} words={words} initial_zero_words={zeros} unused_kv_zero_words={tails} mismatches=0 sdk=false fault_lifecycle=false"
    );
    Ok(())
}
fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        args.len() == 4,
        "usage: qwen_source_calls_hip preflight|cache-preflight|cold|parity|state-parity 08|2b|27b SOURCE REFERENCE",
    )?;
    let mode = args[0].as_str();
    need(
        matches!(
            mode,
            "preflight" | "cache-preflight" | "cold" | "parity" | "state-parity"
        ) || lifecycle::supported(mode),
        "explicit original-call diagnostic mode",
    )?;
    let expected = Expected::open(&PathBuf::from(&args[3]), &args[1])?;
    let loaded = Loaded::open_with_chunk(
        &args[1],
        &PathBuf::from(&args[2]),
        mode != "preflight",
        1024 * 1024,
    )?;
    expected.require_source(&loaded)?;
    let original = OriginalWindowModelSource::from_owner(loaded.model.owner().clone())?;
    need(
        Arc::ptr_eq(original.owner(), loaded.model.owner()),
        "same original authenticated source owner",
    )?;
    println!(
        "SOURCE_CALL_AUTH config={} index={} bytes={} chunk={} text={} other={}",
        original.owner().config_sha256(),
        original.owner().verified_index_sha256(),
        original.source().source_bytes(),
        original.source().staging_chunk_bytes(),
        original.owner().text_tensor_count(),
        original.owner().other_tensor_count()
    );
    let plan = original.bind_ordinary_cpu_source()?.build_plan(514)?;
    let map = ObservationMap::new(&plan)?;
    for call in expected.calls() {
        let invocation = plan.bind_invocation(
            if call.phase == "decode" {
                SourceCallMode::Decode
            } else {
                SourceCallMode::Prefill
            },
            call.position,
            call.input_ids.clone().into(),
            true,
        )?;
        need(
            Arc::ptr_eq(invocation.plan(), plan.plan()),
            "same original call-free plan",
        )?;
        invocation.materialize_source_plan()?;
        Observer::new(&expected, &map, call)?;
    }
    let budget = source_budget::explicit()?.unwrap_or_default();
    let inventory = ResidentModel::ordinary_cpu_source_preflight_with_budget(
        plan.source().clone(),
        plan.plan().clone(),
        2,
        budget,
    )?;
    need(
        inventory.total_device_budget_bytes == budget.total_device_bytes()
            && inventory.model_device_budget_bytes + inventory.context_device_bytes
                == budget.total_device_bytes()
            && inventory.physical_admission_bytes == budget.physical_admission_bytes(),
        "preflight prices the complete original device envelope",
    )?;
    println!(
        "SOURCE_CALL_PREFLIGHT model={} calls={} inventory={inventory:?} gpu_execution=false",
        args[1],
        expected.calls().len()
    );
    if matches!(mode, "preflight" | "cache-preflight") {
        loaded.verify_after()?;
        expected.unchanged()?;
        return Ok(());
    }
    need(
        std::env::var("HIP_VISIBLE_DEVICES").as_deref() == Ok("0")
            && std::env::var("ROCR_VISIBLE_DEVICES").as_deref() == Ok("0")
            && std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/vendor")?.trim()
                == "0x1002"
            && std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/device")?.trim()
                == "0x1586",
        "explicit approved x2 gfx1151 source-call diagnostic",
    )?;
    drop(original);
    let label = args[1].as_str();
    drive::block_on(async move {
        let resident = ResidentModel::load_ordinary_cpu_source_with_budget(
            Device::Hip(Default::default()),
            plan.source().clone(),
            loaded.clone(),
            plan.plan().clone(),
            2,
            budget,
            None,
        )
        .await?;
        let probe = resident.ordinary_cpu_source_native_probe().await?;
        let active = probe.observe()?;
        need(
            active.retained
                && active.modules > 0
                && active.pending_events == 0
                && active.quarantined_handles == 0
                && active.completion_owners == 0,
            "original cold native set and clean service",
        )?;
        println!(
            "SOURCE_CALL_COLD modules={} resident_bytes={} model_inference=false",
            active.modules, active.resident_bytes
        );
        let physical = zerogravity::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&active.physical)?;
        need(
            physical.reserved == budget.total_device_bytes() as u64
                && physical.limit == budget.physical_admission_bytes()
                && !physical.fenced,
            "live original envelope includes, not adds, context",
        )?;
        need(
            active.resident_bytes >= inventory.resident_weight_bytes + inventory.workspace_bytes,
            "original resident ledger covers declared weights and workspace",
        )?;
        println!(
            "SOURCE_CALL_BUDGET total_bytes={} context_bytes={} model_capacity_bytes={} source_weight_bytes={} resident_weight_bytes={} resident_ledger_bytes={} physical_reserved={} physical_limit={}",
            budget.total_device_bytes(),
            budget.context_device_bytes(),
            budget.model_device_bytes(),
            inventory.source_weight_bytes,
            inventory.resident_weight_bytes,
            active.resident_bytes,
            physical.reserved,
            physical.limit
        );
        if lifecycle::supported(mode) {
            let owner = Arc::downgrade(&loaded);
            let cache = loaded.cache_witness()?;
            loaded.verify_after()?;
            need(
                loaded.close().is_err(),
                "original native owner prevents source close",
            )?;
            lifecycle::run(mode, resident, plan, &expected).await?;
            let fault = mode == "unknown" || mode.ends_with("-fault");
            need(
                (owner.strong_count() != 0) == fault && (cache.strong_count() != 0) == fault,
                "original source/cache retention follows physical result",
            )?;
            let host = zerogravity::resource_arbiter::ResourceArbiter::process_shared().snapshot(
                &zerogravity::resource_arbiter::PhysicalDeviceKey::new("host", "system-ram")?,
            )?;
            if !fault {
                need(
                    host.reserved == 2 * 1024 * 1024 * 1024 + 64 * 1024 * 1024 && !host.fenced,
                    "known final native/source/readers leave only persistent host context plus reference support",
                )?;
            }
            expected.unchanged()?;
            println!(
                "SOURCE_CALL_LIFECYCLE_SOURCE model={} mode={mode} source_retained={fault} cache_retained={fault} host_reserved={} host_fenced={}",
                label, host.reserved, host.fenced
            );
            return Ok(());
        }
        if matches!(mode, "parity" | "state-parity") {
            trajectories(&resident, &plan, &expected, &map, mode == "parity").await?;
        }
        resident.shutdown_ordinary_cpu_source().await?;
        let terminal = probe.observe()?;
        need(
            !terminal.retained
                && terminal.modules == 0
                && terminal.pending_events == 0
                && terminal.quarantined_handles == 0
                && terminal.completion_owners == 0
                && terminal.resident_bytes == 0,
            "actual original cold/state/native owners retired",
        )?;
        let device = zerogravity::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&active.physical)?;
        need(
            device.reserved == inventory.context_device_bytes as u64 && !device.fenced,
            "original persistent context only, unfenced",
        )?;
        println!(
            "SOURCE_CALL_TERMINAL mode={mode} native_retained=false resident_bytes=0 device_baseline={} fenced=false fault_lifecycle=false sdk=false",
            device.reserved
        );
        loaded.verify_after()?;
        expected.unchanged()?;
        Ok::<(), zerogravity::Error>(())
    })
}
fn main() {
    if let Err(error) = run() {
        eprintln!("SOURCE_CALL_DIAGNOSTIC_REFUSED: {error}");
        std::process::exit(1);
    }
}
