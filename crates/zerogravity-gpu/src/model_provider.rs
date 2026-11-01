//! Ordinary resident whole-model preparation, with original physical ownership.
use crate::{ResidentModel, backend::Provider};
use cubecl::Device;
use std::sync::Arc;
use zerogravity_core::{
    Result,
    model::ModelProgram,
    provider::{ProviderKind, RegionExecutionProvider},
};

/// Selection is cold; construction grants no source, kernel, placement or
/// allocation permission. The original loader performs all those checks.
pub struct ResidentModelProvider {
    device: Device,
    provider: Provider,
    provider_cap: usize,
    device_cap: u64,
}
impl ResidentModelProvider {
    pub fn new(device: Device, provider_cap: usize, device_cap: u64) -> Result<Self> {
        Ok(Self {
            provider: Provider::selected(&device)?,
            device,
            provider_cap,
            device_cap,
        })
    }
}
impl RegionExecutionProvider for ResidentModelProvider {
    type Executable = Arc<ResidentModel>;
    fn kind(&self) -> ProviderKind {
        match self.provider {
            Provider::Cuda => ProviderKind::Cuda,
            Provider::Hip => ProviderKind::Hip,
            Provider::Vulkan => ProviderKind::Vulkan,
        }
    }
    async fn prepare(&self, program: ModelProgram) -> Result<Self::Executable> {
        let (source, plan) = program.into_parts();
        ResidentModel::load(
            self.device.clone(),
            source,
            plan,
            self.provider_cap,
            self.device_cap,
        )
        .await
    }
}
