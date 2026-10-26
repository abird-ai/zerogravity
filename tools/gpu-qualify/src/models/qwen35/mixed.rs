//! P07-B real-model partial-GPU integration smoke for source Q4_1, Q8_0, Q6_K and F32.
//!
//! Runs the real Qwen3.5 Rust decoder twice over identical raw text tokens:
//! one pure source-qualified CPU, one using a physical CubeCL mixed-quant matvec
//! for the first-layer projection and CPU fallback for unsupported regions.
//! The comparison never returns success unless a device readback completed,
//! the GPU provider reports at least one real dispatched region, and output
//! and StateRoot numerical parity pass. This is NOT a whole-model GPU backend.

use std::{path::Path, sync::Arc};
use zerogravity_cpu::ops::top_logit_ranks;
use zerogravity_gpu::Device;
use zerogravity_gpu::qualification::CubeMixedProvider;
use zerogravity_model_qwen35::QwenDenseCpu;
use zerogravity_model_qwen35::tokenize::QwenTokenizer;

pub fn run(device: Device, model: &Path, tensor_name: &str, prompt: &str) -> Result<(), String> {
    let directory = model.parent().ok_or("GGUF model directory missing")?;
    let config = directory.join("config.json");
    let tokenizer = QwenTokenizer::from_file(&directory.join("tokenizer.json"))
        .map_err(|e| format!("official source tokenizer: {e}"))?;
    let ids = tokenizer
        .prompt_tokens(prompt, false)
        .map_err(|e| format!("encode source prompt: {e}"))?;
    if ids.is_empty() || ids.len() > 32 {
        return Err("P07-B physical hybrid smoke requires 1..32 raw model input tokens".into());
    }

    let mut reference =
        QwenDenseCpu::open(model, &config).map_err(|e| format!("source model CPU: {e}"))?;
    let oracle = reference
        .prefill_transaction(&ids)
        .map_err(|e| format!("pure CPU source math: {e}"))?;

    let provider = Arc::new(CubeMixedProvider::new(device, tensor_name, 8192));
    let mut accelerated = QwenDenseCpu::open_with_kernel(model, &config, provider.clone())
        .map_err(|e| format!("inject real device provider: {e}"))?;
    let gpu = accelerated
        .prefill_transaction(&ids)
        .map_err(|e| format!("physical device execution/StateRoot failed: {e}"))?;
    if provider.completed_regions() != ids.len() {
        return Err(format!(
            "P07-B must complete exactly one Q4 GPU region per source token, got {} for {} tokens",
            provider.completed_regions(),
            ids.len(),
        ));
    }
    if oracle.len() != gpu.len() || oracle.iter().chain(gpu.iter()).any(|v| !v.is_finite()) {
        return Err("P07-B GPU produced wrong or nonfinite logits".into());
    }
    if reference.position() != accelerated.position()
        || reference.root_id() != accelerated.root_id()
    {
        return Err("P07-B CPU/GPU state position or root differs".into());
    }

    // This compares the complete native model output, not a standalone
    // pre-dequantized F32 test fixture. Device cross-platform numerical
    // tolerances are provisional P07-B diagnostics, not a GPU release policy.
    let cpu_top = top_logit_ranks(&oracle, 20).map_err(|e| e.to_string())?;
    let gpu_top = top_logit_ranks(&gpu, 20).map_err(|e| e.to_string())?;
    let shared = cpu_top
        .iter()
        .filter_map(|v| gpu_top.iter().find(|x| x.id == v.id).map(|x| (v, x)))
        .collect::<Vec<_>>();
    if shared.len() < 14 || cpu_top[0].id != gpu_top[0].id {
        return Err(format!(
            "P07-B full-model numerical mismatch: greedy CPU {} GPU {}, shared {}/20",
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
        "P07_B_REAL_GPU_MIXED_MATVEC_COMPLETE model={} tensor={tensor_name} tokens={} completed_regions={} shared_top20={} \
         greedy={} max_full_vocab_logit_diff={max_logit_difference:.7} \
         max_top20_logprob_diff={logprob_error:.7} top_probability_diff={top_probability_error:.7}",
        model.display(),
        ids.len(),
        provider.completed_regions(),
        shared.len(),
        gpu_top[0].id,
    );
    if logprob_error > 0.05 || top_probability_error > 0.005 {
        return Err(format!(
            "P07-B strict hybrid numerical criterion failed: logprob={logprob_error}, p={top_probability_error}"
        ));
    }
    println!(
        "P07_B_PARTIAL_REAL_GPU_MODEL_PARITY_PASS (one closed mixed-quant source region per input token)"
    );
    Ok(())
}
