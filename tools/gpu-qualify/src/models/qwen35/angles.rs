//! Qwen metadata setup for the shared physical angle comparison.
use std::path::Path;
use zerogravity_core::{Error, Result};
use zerogravity_gpu::Device;
use zerogravity_model_qwen35::LoadedQwenModel;

pub async fn qualify_angles(device: Device, path: &Path) -> Result<()> {
    qualify_angles_with_device_cap(device, path, 5120 * 1024 * 1024).await
}
pub async fn qualify_angles_with_device_cap(
    device: Device,
    path: &Path,
    device_cap: u64,
) -> Result<()> {
    if cfg!(debug_assertions) {
        return Err(Error::Unsupported(
            "physical angle qualification requires --release".into(),
        ));
    }
    if !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "ordered angle qualifier requires CUDA".into(),
        ));
    }
    let parent = path
        .parent()
        .ok_or_else(|| Error::Malformed("angle model directory".into()))?;
    let model = LoadedQwenModel::open(path, &parent.join("config.json"))?;
    let plan = model.forward_plan(512, true, false)?;
    zerogravity_gpu::qualification::qualify_angles_with_plan(device, model, plan, path, device_cap)
        .await
}
