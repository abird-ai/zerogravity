//! Explicit source-model diagnostic; no production run/serve route.
#[path = "../tests/support/qwen_source_reference.rs"]
mod reference;
#[path = "../tests/support/qwen_source.rs"]
mod source;
use reference::Expected;
use sha2::{Digest, Sha256};
use source::{Loaded, need};
#[path = "../tests/support/drive.rs"]
mod drive;
#[path = "../tests/support/qwen_source_lifetime.rs"]
mod lifetime;
#[path = "../tests/support/qwen_source_preflight.rs"]
mod preflight_controls;
#[path = "../tests/support/qwen_source_import.rs"]
mod source_import;
#[path = "../tests/support/qwen_source_pins.rs"]
mod source_pins;
use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity::Result;
use zerogravity::gpu::{Device, SourceTextCandidate, SourceTextSession, SourceTextState};

fn root_hash(state: SourceTextState) -> String {
    let mut h = Sha256::new();
    for (slot, values) in state.domains() {
        h.update(format!("{slot:?}"));
        for value in values {
            h.update(value.to_le_bytes());
        }
    }
    format!("{:x}", h.finalize())
}
fn selected(values: &[f32]) -> Result<(u32, u32)> {
    need(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "complete finite actual logits",
    )?;
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok((best as u32, values[best].to_bits()))
}
async fn trajectory(
    session: &mut SourceTextSession,
    expected: &mut Expected,
    repeat: usize,
) -> Result<usize> {
    let mut token = 11;
    let mut total = 0;
    for position in 0..64 {
        need(
            session.binding().frontier() == position as u64,
            "actual source root frontier",
        )?;
        let mut observer = expected.observe(repeat, position, token, None)?;
        let pending = session.prepare(token, Some(&mut observer), None).await?;
        observer.finish()?;
        let words = observer.words;
        let zero_words = observer.initial_zero_words;
        drop(observer);
        let actual = selected(pending.logits())?;
        let call = expected.call(repeat, position)?;
        need(
            actual == (call.selected, call.max_bits),
            "actual fullV selection equals original autoregressive feedback",
        )?;
        let dispatches = pending.dispatches();
        let output = session.commit(pending)?;
        need(
            selected(output.logits())? == actual
                && output.dispatches() == dispatches
                && session.binding().frontier() == (position + 1) as u64,
            "same completed result and published root",
        )?;
        drop(output);
        token = actual.0;
        total += words;
        println!(
            "SOURCE_MODEL_CALL repeat={repeat} position={position} words={words} zero_control_words={zero_words} mismatches=0 selected={} dispatches={dispatches}",
            actual.0
        );
    }
    need(
        session.prepare(token, None, None).await.is_err(),
        "source actual context boundary refuses",
    )?;
    Ok(total)
}
async fn cases(candidate: &Arc<SourceTextCandidate>, expected: &mut Expected) -> Result<()> {
    let mut first = candidate.session().await?;
    let mut second = candidate.session().await?;
    need(
        first.binding().session() != second.binding().session(),
        "independent original session authorities",
    )?;
    need(
        candidate.session().await.is_err(),
        "actual session pressure refusal",
    )?;
    let snapshot = first.snapshot()?;
    let spare = second.snapshot()?;
    need(
        first.snapshot().is_err(),
        "actual retained snapshot pressure refusal",
    )?;
    drop(spare);
    need(
        candidate.clone().shutdown().is_err(),
        "shutdown refuses live original owners",
    )?;
    let base = first.binding();
    let before = root_hash(first.diagnostic_state().await?);
    let cancelled = AtomicBool::new(true);
    need(
        first.prepare(11, None, Some(&cancelled)).await.is_err() && first.binding() == base,
        "prelaunch cancellation cannot publish",
    )?;
    let flag = AtomicBool::new(false);
    let mut observer = expected.observe(0, 0, 11, Some(&flag))?;
    need(
        first
            .prepare(11, Some(&mut observer), Some(&flag))
            .await
            .is_err(),
        "completed-logit cancellation must refuse publication",
    )?;
    observer.finish()?;
    drop(observer);
    need(
        first.binding() == base
            && root_hash(first.diagnostic_state().await?) == before
            && candidate.pending_events()? == 0,
        "late cancellation preserves original root with known completion",
    )?;
    let mut observer = expected.observe(0, 0, 11, None)?;
    let pending = first.prepare(11, Some(&mut observer), None).await?;
    observer.finish()?;
    drop(observer);
    drop(pending);
    need(
        first.binding() == base && root_hash(first.diagnostic_state().await?) == before,
        "discarded completed proposal preserves original root",
    )?;
    println!(
        "SOURCE_MODEL_CONTROLS pre_cancel=true late_cancel=true discard=true session_pressure=true snapshot_pressure=true live_owner_shutdown_refused=true"
    );
    let words0 = trajectory(&mut first, expected, 0).await?;
    let first_binding = first.binding();
    let first_state = root_hash(first.diagnostic_state().await?);
    let words1 = trajectory(&mut second, expected, 1).await?;
    need(
        first.binding() == first_binding
            && root_hash(first.diagnostic_state().await?) == first_state,
        "peer execution leaves complete original root unchanged",
    )?;
    need(
        root_hash(second.diagnostic_state().await?) == first_state,
        "independent roots reproduce complete original state",
    )?;
    first.restore(&snapshot)?;
    need(
        first.binding().frontier() == 0 && root_hash(first.diagnostic_state().await?) == before,
        "restore original snapshot after all populated contexts",
    )?;
    let mut observer = expected.observe(0, 0, 11, None)?;
    let pending = first.prepare(11, Some(&mut observer), None).await?;
    observer.finish()?;
    drop(observer);
    drop(first.commit(pending)?);
    first.reset().await?;
    need(
        first.binding().frontier() == 0 && root_hash(first.diagnostic_state().await?) == before,
        "reset actual source roots",
    )?;
    need(
        candidate.pending_events()? == 0,
        "all actual source events completed",
    )?;
    println!(
        "SOURCE_MODEL_FULL_PARITY calls=128 active_context=64 repeat_words={} mismatches=0 fullV=true all_state=true independent_roots=true snapshot_restore=true reset=true",
        words0 + words1
    );
    Ok(())
}
fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    need(
        args.len() == 4,
        "usage: qwen_source_hip MODE 08|2b SOURCE REFERENCE (preflight, cache-preflight, import, parity, parity-import, session/snapshot/pending-known/fault, unknown)",
    )?;
    let mode = &args[0];
    if mode == "import" {
        source_import::import(&args[1], &PathBuf::from(&args[2]))?;
        return Ok(());
    }

    let lifetime_mode = lifetime::Mode::parse(mode);
    need(
        matches!(
            mode.as_str(),
            "preflight" | "cache-preflight" | "parity" | "parity-import"
        ) || lifetime_mode.is_some(),
        "explicit diagnostic mode",
    )?;
    // Independent authenticated reference precedes source import/open or device work.
    let mut expected = Expected::open(&PathBuf::from(&args[3]), &args[1])?;
    let source_directory = if mode == "parity-import" {
        source_import::import(&args[1], &PathBuf::from(&args[2]))?
    } else {
        PathBuf::from(&args[2])
    };
    let pins = if lifetime_mode.is_some() || mode == "cache-preflight" {
        let pins = source_pins::Pins::new(&args[1], &source_directory)?;
        pins.require(false)?;
        Some(pins)
    } else {
        None
    };
    let loaded = if mode == "preflight" {
        Loaded::open(&args[1], &source_directory)?
    } else {
        Loaded::open_native(&args[1], &source_directory)?
    };
    expected.require_source(&loaded)?;
    let records = (0..64)
        .map(|p| loaded.model.record_source_token(p, 64).map(Arc::new))
        .collect::<Result<Vec<_>>>()?;
    let inventory =
        SourceTextCandidate::preflight(loaded.model.source().clone(), records.clone(), 2)?;
    println!(
        "SOURCE_MODEL_PREFLIGHT model={} weights={} workspace={} root={} roots={} model_device_upper={} context_device={} physical_device_upper={} host_upper={} production_enabled=false",
        args[1],
        inventory.resident_weight_bytes,
        inventory.workspace_bytes,
        inventory.root_bytes,
        inventory.root_count,
        inventory.model_device_upper_bytes,
        inventory.context_device_bytes,
        inventory.physical_device_upper_bytes,
        inventory.host_upper_bytes
    );
    preflight_controls::controls(&loaded, &records, &inventory)?;
    if mode == "preflight" || mode == "cache-preflight" {
        loaded.verify_after()?;
        if let Some(pins) = pins {
            pins.require(true)?;
            let cache = loaded.cache_witness()?;
            drop(records);
            loaded.close()?;
            need(cache.upgrade().is_none(), "cold canonical snapshot retired")?;
            pins.require(false)?;
            println!(
                "SOURCE_MODEL_CACHE_CONTROLS files={} pinned_then_released=true device_context_created=false",
                pins.len()
            );
        }
        return Ok(());
    }
    need(
        std::env::var("ZG_SOURCE_MODEL_GFX1151").as_deref() == Ok("1")
            && std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/vendor")?.trim()
                == "0x1002"
            && std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/device")?.trim()
                == "0x1586",
        "explicit approved x2 gfx1151 source-model diagnostic",
    )?;
    let candidate = drive::block_on(SourceTextCandidate::prepare(
        Device::Hip(Default::default()),
        loaded.model.source().clone(),
        loaded.clone(),
        records,
        2,
        None,
    ))?;
    if let Some(lifetime_mode) = lifetime_mode {
        return lifetime::run(
            lifetime_mode,
            mode,
            &args[1],
            candidate,
            loaded,
            &mut expected,
            pins.ok_or_else(|| zerogravity::Error::State("lifetime pin probe absent".into()))?,
        );
    }
    let probe = candidate.native_probe();
    let native = probe.observe()?;
    need(
        native.retained && native.modules > 0,
        "actual prepared native modules live",
    )?;
    println!(
        "SOURCE_MODEL_NATIVE modules={} resident_bytes={}",
        native.modules,
        candidate.resident_bytes()?
    );
    let result = drive::block_on(cases(&candidate, &mut expected));
    let closed = candidate.shutdown();
    let after = probe.observe()?;
    println!(
        "SOURCE_MODEL_RETIRE namespace_closed={} retained={} modules={} code_bytes={} host_bytes={} device_bytes={}",
        closed.is_ok(),
        after.retained,
        after.modules,
        after.code_bytes,
        after.host_bytes,
        after.device_bytes
    );
    closed?;
    need(
        !after.retained
            && after.modules == 0
            && after.code_bytes == 0
            && after.host_bytes == 0
            && after.device_bytes == 0,
        "actual native model modules/admission retired",
    )?;
    loaded.verify_after()?;
    drop(expected);
    need(
        Arc::strong_count(&loaded) == 1,
        "original full source final owner",
    )?;
    let arbiter = loaded.host.arbiter.clone();
    let key = loaded.host.resource.clone();
    let cache = loaded.cache_witness()?;
    loaded.close()?;
    need(
        cache.upgrade().is_none(),
        "original managed snapshot retired",
    )?;
    let host = arbiter.snapshot(&key)?;
    println!(
        "SOURCE_MODEL_SOURCE_CLOSED host_reserved={} host_fenced={}",
        host.reserved, host.fenced
    );
    result?;
    need(!host.fenced, "healthy actual host scope")?;
    println!(
        "SOURCE_MODEL_DIAGNOSTIC_PASS model={} production_enabled=false serving_qualified=false",
        args[1]
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("SOURCE_MODEL_DIAGNOSTIC_REFUSED: {error}");
        std::process::exit(1);
    }
}
