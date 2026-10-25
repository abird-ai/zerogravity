//! Family supplies only genuine source-issued Vulkan concrete specializations.
use super::*;
use crate::Device;
impl GemmaTextNativeOwners {
    pub(in crate::resident_program::gemma_text) fn vulkan_profile(
        storage: GemmaTextStorage,
    ) -> crate::backend::vulkan_native_policy::VulkanColdProfile {
        use crate::backend::vulkan_native_policy::VulkanColdProfile;
        match storage {
            GemmaTextStorage::Streamed => VulkanColdProfile::Streamed16,
            GemmaTextStorage::ResidentProjections | GemmaTextStorage::ResidentProjectionsVulkan => {
                VulkanColdProfile::Resident20
            }
        }
    }

    pub(crate) async fn prepare_vulkan_qualification(
        device: Device,
        recorded: &Gemma4Bf16TextPlan,
        source_owner: crate::physical_quarantine::CompletionOwner,
        storage: GemmaTextStorage,
        _inventory: &GemmaTextResourceInventory,
    ) -> Result<(Arc<ResidentService>, Arc<Self>)> {
        let profile = Self::vulkan_profile(storage);
        let cold = crate::backend::vulkan_native_policy::begin_gemma_qualification(
            device,
            recorded.source().source(),
            source_owner,
            profile,
        )?;
        let service = cold.service();
        let specs = math::prepared::enumerate(cold.client(), recorded, storage)?;
        let prepared = cold.prepare(specs.kernels, specs._metadata).await?;
        let owners = Arc::new(Self {
            source: recorded.plan().source(),
            service: service.client().service_id(),
            host_bytes: prepared.host_bytes(),
            device_bytes: prepared.device_bytes(),
            context_host_bytes: profile.cost().context_host,
            context_device_bytes: profile.cost().context_device,
            _owners: vec![prepared.clone()],
            #[cfg(all(feature = "qualification", feature = "hip"))]
            hip: None,
            vulkan: Some(prepared),
        });
        Ok((service, owners))
    }
}
