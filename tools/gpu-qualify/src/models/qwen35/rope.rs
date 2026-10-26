//! Qwen full-model RoPE comparison orchestration; physical operators remain GPU-owned.
use std::path::Path;
use zerogravity_core::{Error, Result};
use zerogravity_gpu::{Device, qualification::qualify_rope_angles_with_device_cap};

/// Compose existing accepted operators/whole-model qualifiers, not a second
/// model loop. Each route retains its original CPU/B1/state/physical checks.
pub async fn qualify_model(
    device: Device,
    path: &Path,
    provider_cap: usize,
    device_cap: u64,
    rows: usize,
) -> Result<()> {
    if ![2, 4, 8].contains(&rows) {
        return Err(Error::Malformed("angle full-model rows2/4/8".into()));
    }
    qualify_rope_angles_with_device_cap(device.clone(), path, device_cap).await?;
    super::layer::rope_stress_with_device_cap(device.clone(), path, device_cap)
        .map_err(|e| Error::State(format!("unchanged trained source RoPE oracle: {e}")))?;
    zerogravity_gpu::qualification::qualify_fused(
        device.clone(),
        path,
        provider_cap,
        device_cap,
        rows,
    )
    .await?;
    zerogravity_gpu::qualification::qualify_fused_greedy(
        device,
        path,
        provider_cap,
        device_cap,
        rows,
    )
    .await?;
    println!(
        "CUDA_ORDERED_ROPE_MODEL_PASS B={rows} source={} CPU_B1_native_full_vocab_and_all42=existing_qualifiers old_physical_angle_oracles=exact fullmodel_old_binary_comparison=separate_gate state_only_worker_HTTP_pressure_abort=separate_gates speed_and_vllm=not_qualified",
        path.display()
    );
    Ok(())
}
