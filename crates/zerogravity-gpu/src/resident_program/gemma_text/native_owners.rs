//! Sealed actual client/compiled owners; declarations cannot grant GPU fit.
use super::*;
use cubecl_common::device::ServiceId;
#[cfg(all(feature = "qualification", feature = "hip"))]
mod hip;
#[cfg(all(
    feature = "gemma-bf16-qualification",
    feature = "vulkan",
    target_os = "linux"
))]
mod vulkan;

/// No public/raw constructor. The backend must enumerate, price and retain
/// COMPLETE prepared kernels, tables and client owners before constructing it.
pub struct GemmaTextNativeOwners {
    source: SourceIdentity,
    service: ServiceId,
    host_bytes: usize,
    device_bytes: usize,
    context_host_bytes: usize,
    context_device_bytes: usize,
    _owners: Vec<crate::physical_quarantine::CompletionOwner>,
    #[cfg(all(feature = "qualification", feature = "hip"))]
    hip: Option<Arc<crate::backend::hip_native_policy::PreparedSourceNative>>,
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        target_os = "linux"
    ))]
    vulkan: Option<Arc<crate::backend::vulkan_native_policy::PreparedVulkanSourceNative>>,
}
impl GemmaTextNativeOwners {
    #[cfg(all(
        feature = "hardware-abort-probe",
        any(feature = "hip", feature = "vulkan")
    ))]
    pub(crate) fn final_owner_probe(
        &self,
    ) -> Result<crate::backend::final_native::FinalNativeProbe> {
        #[cfg(all(feature = "qualification", feature = "hip"))]
        if let Some(native) = &self.hip {
            return Ok(crate::backend::final_native::FinalNativeProbe::Hip(
                native.final_owner_probe(),
            ));
        }
        #[cfg(all(
            feature = "gemma-bf16-qualification",
            feature = "vulkan",
            target_os = "linux"
        ))]
        if let Some(native) = &self.vulkan {
            return Ok(crate::backend::final_native::FinalNativeProbe::Vulkan(
                native.final_owner_probe(),
            ));
        }
        Err(Error::State(
            "Gemma original prepared native owner absent".into(),
        ))
    }
    #[cfg(all(
        feature = "gemma-bf16-qualification",
        feature = "vulkan",
        target_os = "linux"
    ))]
    pub(crate) fn vulkan_identity(
        &self,
        service: &ResidentService,
    ) -> Result<crate::backend::vulkan_native_policy::VulkanSourceSetIdentity> {
        self.vulkan
            .as_ref()
            .ok_or_else(|| Error::State("Gemma original Vulkan native identity absent".into()))?
            .identity(service)
    }
    #[cfg(all(feature = "hip", feature = "qwen-source-qualification"))]
    pub(crate) fn identity(
        &self,
        service: &ResidentService,
    ) -> Result<crate::backend::hip_native_policy::HipSourceSetIdentity> {
        self.hip
            .as_ref()
            .map(|native| native.identity(service))
            .ok_or_else(|| Error::State("Gemma original HIP native identity absent".into()))
    }
    /// Missing backend boundary, explicitly refused before model allocation.
    pub fn prepare(
        _client: &Client,
        _source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    ) -> Result<Arc<Self>> {
        Err(Error::Unsupported("Generic Gemma preparation has no finite backend policy; use the complete source-issued qualification consumer".into()))
    }
    pub(crate) fn require(&self, client: &Client, source: SourceIdentity) -> Result<()> {
        if self.source != source || self.service != client.service_id() || self._owners.is_empty() {
            return Err(Error::State(
                "Gemma native owners source/service/completeness differs".into(),
            ));
        }
        #[cfg(all(feature = "qualification", feature = "hip"))]
        if let Some(prepared) = &self.hip {
            prepared
                .require_live(client, source)
                .map_err(|e| Error::State(format!("Gemma native set: {e}")))?;
        }
        #[cfg(all(
            feature = "gemma-bf16-qualification",
            feature = "vulkan",
            target_os = "linux"
        ))]
        if let Some(prepared) = &self.vulkan {
            prepared
                .require_live(client, source)
                .map_err(|e| Error::State(format!("Gemma Vulkan native set: {e}")))?;
        }
        Ok(())
    }
    pub(super) fn context_host_bytes(&self) -> usize {
        self.context_host_bytes
    }
    pub(super) fn context_device_bytes(&self) -> usize {
        self.context_device_bytes
    }
    pub(super) fn host_bytes(&self) -> usize {
        self.host_bytes
    }
    /// Shared source-page transport on the original actor when this owner holds
    /// the live scoped set; `None` keeps the serial generic path.
    #[cfg(all(feature = "qualification", feature = "hip"))]
    pub(super) fn source_page_actor(
        &self,
        service: &crate::resident_leases::ResidentService,
    ) -> Option<crate::resident_leases::source_page::PageActor> {
        self.hip
            .as_ref()
            .and_then(|native| native.source_page_actor(service))
    }
    #[cfg(not(all(feature = "qualification", feature = "hip")))]
    pub(super) fn source_page_actor(
        &self,
        _service: &crate::resident_leases::ResidentService,
    ) -> Option<crate::resident_leases::source_page::PageActor> {
        None
    }
    pub(super) fn device_bytes(&self) -> usize {
        self.device_bytes
    }
}
