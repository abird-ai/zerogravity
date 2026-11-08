//! Explicit populated-context qualification; source/physical owners stay original.
#[path = "../tests/support/qwen_source_capacity.rs"]
mod capacity;
#[path = "../tests/support/qwen_source_context_run.rs"]
mod context_run;
#[path = "../tests/support/qwen_source_context_timing.rs"]
mod context_timing;
#[path = "../tests/support/drive.rs"]
mod drive;
#[path = "../tests/support/qwen_source_calls_reference.rs"]
mod reference;
#[allow(dead_code)]
#[path = "../tests/support/qwen_source.rs"]
mod source;
#[allow(dead_code)]
#[path = "../tests/support/qwen_source_import.rs"]
mod source_import;
#[allow(dead_code)]
#[path = "../tests/support/qwen_source_pins.rs"]
mod source_pins;
#[path = "../tests/support/qwen_source_trig.rs"]
mod trig;
pub use reference::Expected;
use source::{Loaded, need};
use std::{path::PathBuf, sync::Arc};
use zerogravity::{
    Result,
    gpu::{Device, ResidentModel},
    qwen35::source::binding::OriginalWindowModelSource,
    resource_arbiter::ResourceArbiter,
};

fn run() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    need(
        matches!(args.len(), 3 | 5 | 6),
        "usage: qwen_source_contexts_hip preflight|cache-preflight 08|2b|27b SOURCE; reference-preflight|parity|diagnose-latency|diagnose-kernels 08|2b|27b SOURCE REFERENCE CONTEXT MANIFEST_SHA256; trig 08|2b SOURCE REFERENCE MANIFEST_SHA256",
    )?;
    let mode = args[0].as_str();
    let matrix = args.len() == 3;
    need(
        if matrix {
            matches!(mode, "preflight" | "cache-preflight")
        } else if args.len() == 5 {
            mode == "trig"
        } else {
            matches!(
                mode,
                "reference-preflight" | "parity" | "diagnose-latency" | "diagnose-kernels"
            )
        },
        "explicit context diagnostic mode",
    )?;
    let loaded = Loaded::open_with_chunk(
        &args[1],
        &PathBuf::from(&args[2]),
        matches!(
            mode,
            "cache-preflight" | "parity" | "trig" | "diagnose-latency" | "diagnose-kernels"
        ),
        1024 * 1024,
    )?;
    let original = OriginalWindowModelSource::from_owner(loaded.model.owner().clone())?;
    need(
        Arc::ptr_eq(original.owner(), loaded.model.owner()),
        "same original source owner",
    )?;
    let seal = original.bind_ordinary_cpu_source()?;
    if mode == "trig" {
        require_device()?;
        let plan = seal.build_plan(512)?;
        drop((seal, original));
        return trig::run(loaded, plan, &PathBuf::from(&args[3]), &args[4]);
    }
    if matrix {
        capacity::inspect(&seal.build_plan(512)?, &args[1])?;
        loaded.verify_after()?;
        println!(
            "SOURCE_CONTEXT_PREFLIGHT_DONE model={} gpu=false source_unchanged=true",
            args[1]
        );
        return Ok(());
    }
    let repository = match args[1].as_str() {
        "08" => "Qwen/Qwen3.5-0.8B",
        "2b" => "Qwen/Qwen3.5-2B",
        "27b" => "Qwen/Qwen3.8-27B",
        _ => {
            return Err(zerogravity::Error::Unsupported(
                "context source model".into(),
            ));
        }
    };
    let capacity: usize = args[4]
        .parse()
        .map_err(|_| zerogravity::Error::Malformed("context integer".into()))?;
    let scope = reference::ReferenceScope::for_repository(repository)?;
    let case = scope
        .contexts
        .iter()
        .position(|n| *n == capacity)
        .ok_or_else(|| zerogravity::Error::Unsupported("context reference case".into()))?;
    let expected =
        Expected::open_pinned(&PathBuf::from(&args[3]), repository, &args[5], None, true)?;
    expected.require_source(&loaded)?;
    let plan = seal.build_plan(capacity)?;
    let map = reference::ObservationMap::new(&plan)?;
    for call in expected.calls().iter().filter(|c| c.case == case) {
        if call.checkpoint {
            reference::Observer::new(&expected, &map, call)?;
        }
    }
    let inventory = ResidentModel::ordinary_cpu_source_preflight(
        plan.source().clone(),
        plan.plan().clone(),
        2,
    )?;
    println!(
        "SOURCE_CONTEXT_REFERENCE model={} context={capacity} manifest={} inventory={inventory:?}",
        args[1], args[5]
    );
    if mode == "reference-preflight" {
        loaded.verify_after()?;
        expected.unchanged()?;
        println!(
            "SOURCE_CONTEXT_REFERENCE_READY model={} context={capacity} gpu=false",
            args[1]
        );
        return Ok(());
    }
    require_device()?;
    drop((seal, original));
    let diagnose_latency = matches!(mode, "diagnose-latency" | "diagnose-kernels");
    let measure_kernels = mode == "diagnose-kernels";
    drive::block_on(async move {
        let resident = ResidentModel::load_ordinary_cpu_source(
            Device::Hip(Default::default()),
            plan.source().clone(),
            loaded.clone(),
            plan.plan().clone(),
            2,
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
            "original clean cold context",
        )?;
        if diagnose_latency {
            context_timing::run(&resident, &plan, &expected, case, capacity, measure_kernels)
                .await?;
        } else {
            context_run::trajectories(&resident, &plan, &expected, case, capacity).await?;
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
            "original context native retirement",
        )?;
        let device = ResourceArbiter::process_shared().snapshot(&active.physical)?;
        need(
            device.reserved == inventory.context_device_bytes as u64 && !device.fenced,
            "persistent context only, unfenced",
        )?;
        loaded.verify_after()?;
        expected.unchanged()?;
        println!(
            "SOURCE_CONTEXT_TERMINAL model={} context={capacity} native_retained=false resident_bytes=0 device_context_bytes={} fenced=false sdk=false performance=false",
            args[1], device.reserved
        );
        Ok(())
    })
}
fn require_device() -> Result<()> {
    need(
        std::env::var("HIP_VISIBLE_DEVICES").as_deref() == Ok("0")
            && std::env::var("ROCR_VISIBLE_DEVICES").as_deref() == Ok("0")
            && std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/vendor")?.trim()
                == "0x1002"
            && std::fs::read_to_string("/sys/bus/pci/devices/0000:c6:00.0/device")?.trim()
                == "0x1586",
        "approved x2 context diagnostic",
    )
}
fn main() {
    if let Err(error) = run() {
        eprintln!("SOURCE_CONTEXT_DIAGNOSTIC_REFUSED: {error}");
        std::process::exit(1);
    }
}
