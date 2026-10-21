//! Source-real, opt-in hardware acceptance for multiple independent Qwen
//! sessions sharing ONE immutable GGUF and ONE physical GPU weight upload.
//!
//! The CPU/GPU engines are the SAME QwenDenseCpu; only the kernel adapter
//! changes. This proves one loaded model source identity across sessions,
//! equal full-vocabulary output and distinct per-session StateRoots.
//! No new HTTP/router executor or ad-hoc weight copy is created.
use crate::resident_provider::CubeResidentProvider;
use cubecl::Device;
use std::{path::Path, sync::Arc, thread};
use zerogravity_core::resource_arbiter::{ResourceArbiter, ResourceOwner};
use zerogravity_model_qwen35::LoadedQwenModel;

pub fn run(device: Device, model: &Path) -> Result<(), String> {
    let config = model
        .parent()
        .ok_or("GGUF has no containing directory")?
        .join("config.json");
    let loaded = LoadedQwenModel::open(model, &config).map_err(|e| e.to_string())?;
    let input_a = [11u32];
    let input_b = [353u32];
    let mut cpu_a = loaded.new_session().map_err(|e| e.to_string())?;
    let mut cpu_b = loaded.new_session().map_err(|e| e.to_string())?;
    let expected_a = cpu_a
        .prefill_transaction(&input_a)
        .map_err(|e| e.to_string())?;
    let expected_b = cpu_b
        .prefill_transaction(&input_b)
        .map_err(|e| e.to_string())?;
    let fingerprint_a = cpu_a.state_fingerprint();
    let fingerprint_b = cpu_b.state_fingerprint();
    if fingerprint_a == fingerprint_b {
        return Err("distinct token sessions unexpectedly shared mutable state".into());
    }
    let provider = Arc::new(
        CubeResidentProvider::new(
            device,
            ["blk.0.attn_gate.weight".to_owned()],
            8192,
            256 * 1024 * 1024,
            512 * 1024 * 1024,
            ResourceOwner::new("qwen35-sessions", "physical-card").map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
    );
    let key = provider.physical_key();
    let (result_a, result_b) = thread::scope(|scope| {
        let model_a = loaded.clone();
        let provider_a = provider.clone();
        let a = scope.spawn(move || -> Result<_, String> {
            let mut session = model_a
                .new_session()
                .map_err(|e| e.to_string())?
                .with_kernel(provider_a);
            let logits = session
                .prefill_transaction(&input_a)
                .map_err(|e| e.to_string())?;
            Ok((logits, session.root_id(), session.state_fingerprint()))
        });
        let model_b = loaded.clone();
        let provider_b = provider.clone();
        let b = scope.spawn(move || -> Result<_, String> {
            let mut session = model_b
                .new_session()
                .map_err(|e| e.to_string())?
                .with_kernel(provider_b);
            let logits = session
                .prefill_transaction(&input_b)
                .map_err(|e| e.to_string())?;
            Ok((logits, session.root_id(), session.state_fingerprint()))
        });
        (a.join(), b.join())
    });
    let (logits_a, root_a, gpu_fingerprint_a) =
        result_a.map_err(|_| "GPU Qwen session A panicked".to_owned())??;
    let (logits_b, root_b, gpu_fingerprint_b) =
        result_b.map_err(|_| "GPU Qwen session B panicked".to_owned())??;
    if logits_a != expected_a
        || logits_b != expected_b
        || gpu_fingerprint_a != fingerprint_a
        || gpu_fingerprint_b != fingerprint_b
        || root_a != 1
        || root_b != 1
    {
        return Err("GPU model sessions lost exact CPU source/StateRoot isolation".into());
    }
    let stats = provider.stats().map_err(|e| e.to_string())?;
    let queue = provider.worker_stats().map_err(|e| e.to_string())?;
    if stats.source_uploads != 1
        || stats.source_cache_hits != 1
        || stats.completed_regions != 2
        || queue.completed != 2
        || queue.active
        || queue.queued != 0
        || provider
            .in_flight_device_bytes()
            .map_err(|e| e.to_string())?
            != 0
        || provider
            .quarantined_physical_handles()
            .map_err(|e| e.to_string())?
            != 0
    {
        return Err(format!(
            "physical GPU shared source/session ledger incorrect: {stats:?}, {queue:?}"
        ));
    }
    drop(provider);
    let final_state = ResourceArbiter::process_shared()
        .snapshot(&key)
        .map_err(|e| e.to_string())?;
    if final_state.reserved != 0 || final_state.fenced {
        return Err(format!(
            "physical GPU session memory was not retired: {final_state:?}"
        ));
    }
    println!(
        "P07_PHYSICAL_SHARED_MODEL_SESSIONS_PASS device={key:?} unique_gguf_sources=1 separate_sessions=2 physical_regions=2 source_uploads=1 source_cache_hits=1 strict_qwen_full_vocab=1 independent_roots=1 charged_bytes_after_teardown=0"
    );
    Ok(())
}
