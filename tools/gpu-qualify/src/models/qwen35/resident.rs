//! P07-C real-model partial-GPU integration smoke for source Q4_1, Q8_0, Q6_K and F32.
//!
//! Runs the real Qwen3.5 Rust decoder twice over identical raw text tokens:
//! one pure source-qualified CPU, one using a physical CubeCL mixed-quant matvec
//! for the first-layer projection and CPU fallback for unsupported regions.
//! The comparison never returns success unless a device readback completed,
//! the GPU provider reports at least one real dispatched region, and output
//! and StateRoot numerical parity pass. This is NOT a whole-model GPU backend.

use std::{path::Path, sync::Arc};
use zerogravity_core::Result as ZgResult;
use zerogravity_core::accelerator::{MatVecKernel, MatVecRegion, RegionExecution};
use zerogravity_core::resource_arbiter::ResourceOwner;
use zerogravity_cpu::ops::top_logit_ranks;
use zerogravity_cpu::source::source_reference_projection;
use zerogravity_gpu::CubeResidentProvider;
use zerogravity_gpu::Device;
use zerogravity_model_qwen35::QwenDenseCpu;
use zerogravity_model_qwen35::tokenize::QwenTokenizer;

/// Opt-in physical acceptance probe: compare EXACTLY the region sent to the
/// device with the native source GGUF quantized GEMV before the rest of Qwen
/// amplifies small FP32 differences. Never a production hot-path wrapper.
struct CheckedPhysicalRegion(Arc<CubeResidentProvider>);
impl MatVecKernel for CheckedPhysicalRegion {
    fn try_execute(&self, region: MatVecRegion<'_>) -> ZgResult<RegionExecution> {
        let execution = self.0.try_execute(region)?;
        if let RegionExecution::Completed(ref actual) = execution {
            let expected = source_reference_projection(region)?;
            if actual.len() != expected.len() {
                return Err(zerogravity_core::Error::State(
                    "physical GPU source projection returned malformed row count".into(),
                ));
            }
            let mut mismatched_bits = 0usize;
            let mut worst = (0usize, 0.0f32);
            for (row, (&reference, &device)) in expected.iter().zip(actual).enumerate() {
                if reference.to_bits() != device.to_bits() {
                    mismatched_bits += 1;
                }
                let difference = (reference - device).abs();
                if difference > worst.1 {
                    worst = (row, difference);
                }
            }
            println!(
                "P07_G2_PHYSICAL_REGION_NUMERICS kind={:?} rows={} cols={} differing_f32_bits={} max_abs={} worst_row={} source={} gpu={}",
                region.kind,
                region.rows,
                region.columns,
                mismatched_bits,
                worst.1,
                worst.0,
                expected[worst.0],
                actual[worst.0]
            );
            if mismatched_bits > 0 {
                return Err(zerogravity_core::Error::State(format!(
                    "physical GPU source projection not bit-exact; differing_rows={mismatched_bits}, max_abs={} — refusing StateRoot publication",
                    worst.1,
                )));
            }
        }
        Ok(execution)
    }
}

pub fn run(device: Device, model: &Path, tensor_name: &str, prompt: &str) -> Result<(), String> {
    let directory = model.parent().ok_or("GGUF model directory missing")?;
    let config = directory.join("config.json");
    let tokenizer = QwenTokenizer::from_file(&directory.join("tokenizer.json"))
        .map_err(|e| format!("official source tokenizer: {e}"))?;
    let ids = tokenizer
        .prompt_tokens(prompt, false)
        .map_err(|e| format!("encode source prompt: {e}"))?;
    if ids.is_empty() || ids.len() > 32 {
        return Err("P07-E physical hybrid smoke requires 1..32 raw model input tokens".into());
    }

    let mut reference =
        QwenDenseCpu::open(model, &config).map_err(|e| format!("source model CPU: {e}"))?;
    let oracle = reference
        .prefill_transaction(&ids)
        .map_err(|e| format!("pure CPU source math: {e}"))?;

    let provider = Arc::new(
        CubeResidentProvider::new(
            device,
            [tensor_name.to_owned()],
            8192,
            512 * 1024 * 1024, // bounded provider envelope
            512 * 1024 * 1024, // explicit operator physical-device cap
            ResourceOwner::new("qwen35-hardware-smoke", "source-model-check")
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("create bounded residency: {e}"))?,
    );
    let mut accelerated = QwenDenseCpu::open_with_kernel(
        model,
        &config,
        Arc::new(CheckedPhysicalRegion(provider.clone())),
    )
    .map_err(|e| format!("inject real device provider: {e}"))?;
    let prior_root = accelerated.root_id();
    let prior_pos = accelerated.position();
    let prior_fingerprint = accelerated.state_fingerprint();
    let gpu = match accelerated.prefill_transaction(&ids) {
        Ok(gpu) => gpu,
        Err(error) => {
            if accelerated.root_id() != prior_root
                || accelerated.position() != prior_pos
                || accelerated.state_fingerprint() != prior_fingerprint
            {
                return Err(format!(
                    "physical GPU source mismatch published a partial StateRoot: {error}"
                ));
            }
            return Err(format!(
                "physical GPU source/StateRoot refused safely (prior root preserved): {error}"
            ));
        }
    };
    // Query the process-shared budget: a model cannot claim physical safety
    // solely from its own source-cache accounting.
    let worker = provider.worker_stats().map_err(|e| e.to_string())?;
    if worker.completed != ids.len() as u64 || worker.active || worker.queued != 0 {
        return Err(format!(
            "physical bounded worker did not complete exact source regions: {worker:?}"
        ));
    }
    println!(
        "P07_G2B_PHYSICAL_WORKER_COMPLETED={} capacity={}",
        worker.completed, worker.capacity
    );
    // Hardware smoke only: no outstanding GPU work may cross the next epoch.
    let next_epoch = worker
        .epoch
        .checked_add(1)
        .ok_or("GPU worker epoch exhausted")?;
    if provider
        .advance_physical_card_epoch(next_epoch)
        .map_err(|e| e.to_string())?
        != 0
    {
        return Err("hardware smoke left obsolete queued GPU work".into());
    }
    let global_unreserved = provider
        .unreserved_physical_bytes()
        .map_err(|e| format!("shared physical device admission: {e}"))?;
    println!("P07_G2_SHARED_PHYSICAL_UNRESERVED_BYTES={global_unreserved}");
    if provider.is_fenced().map_err(|e| e.to_string())? {
        return Err(
            "P07-E source residency is fenced; cannot report successful model execution".into(),
        );
    }
    if provider
        .in_flight_device_bytes()
        .map_err(|e| e.to_string())?
        != 0
    {
        return Err("P07-E physical GPU completion left unresolved in-flight buffer leases".into());
    }
    if provider
        .quarantined_physical_handles()
        .map_err(|e| e.to_string())?
        != 0
    {
        return Err(
            "P07-F physical GPU quarantine is nonempty after successful model execution".into(),
        );
    }
    let stats = provider.stats().map_err(|e| e.to_string())?;
    if stats.source_uploads != 1 || stats.source_cache_hits != ids.len() - 1 {
        return Err(format!(
            "P07-C must upload exact model source once, got uploads={} cache_hits={} for {} tokens",
            stats.source_uploads,
            stats.source_cache_hits,
            ids.len()
        ));
    }
    if stats.completed_regions != ids.len() {
        return Err(format!(
            "P07-C must complete exactly one Q4 GPU region per source token, got {} for {} tokens",
            provider
                .stats()
                .map_err(|e| e.to_string())?
                .completed_regions,
            ids.len(),
        ));
    }
    if oracle.len() != gpu.len() || oracle.iter().chain(gpu.iter()).any(|v| !v.is_finite()) {
        return Err("P07-C GPU produced wrong or nonfinite logits".into());
    }
    if reference.position() != accelerated.position()
        || reference.root_id() != accelerated.root_id()
    {
        return Err("P07-C CPU/GPU state position or root differs".into());
    }

    // This compares the complete native model output, not a standalone
    // pre-dequantized F32 test fixture. Device cross-platform numerical
    // tolerances are provisional P07-C diagnostics, not a GPU release policy.
    let cpu_top = top_logit_ranks(&oracle, 20).map_err(|e| e.to_string())?;
    let gpu_top = top_logit_ranks(&gpu, 20).map_err(|e| e.to_string())?;
    let shared = cpu_top
        .iter()
        .filter_map(|v| gpu_top.iter().find(|x| x.id == v.id).map(|x| (v, x)))
        .collect::<Vec<_>>();
    if shared.len() < 14 || cpu_top[0].id != gpu_top[0].id {
        return Err(format!(
            "P07-C full-model numerical mismatch: greedy CPU {} GPU {}, shared {}/20",
            cpu_top[0].id,
            gpu_top[0].id,
            shared.len(),
        ));
    }
    let logprob_error = shared
        .iter()
        .map(|(a, b)| (a.logprob - b.logprob).abs())
        .fold(0f64, f64::max);
    let top_probability_error = (cpu_top[0].logprob.exp() - gpu_top[0].logprob.exp()).abs();
    let max_logit_difference = oracle
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "P07_C_REAL_GPU_MIXED_MATVEC_COMPLETE model={} tensor={tensor_name} tokens={} completed_regions={} shared_top20={} \
         greedy={} max_full_vocab_logit_diff={max_logit_difference:.7} \
         max_top20_logprob_diff={logprob_error:.7} top_probability_diff={top_probability_error:.7}",
        model.display(),
        ids.len(),
        provider
            .stats()
            .map_err(|e| e.to_string())?
            .completed_regions,
        shared.len(),
        gpu_top[0].id,
    );
    if logprob_error > 0.05 || top_probability_error > 0.005 {
        return Err(format!(
            "P07-C strict hybrid numerical criterion failed: logprob={logprob_error}, p={top_probability_error}"
        ));
    }
    println!(
        "P07_C_PARTIAL_REAL_GPU_MODEL_PARITY_PASS (one closed mixed-quant source region per input token)"
    );
    Ok(())
}
