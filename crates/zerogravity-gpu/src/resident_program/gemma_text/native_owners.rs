//! Sealed actual client/compiled owners; declarations cannot grant GPU fit.
use super::*;
use cubecl_common::device::ServiceId;
#[cfg(all(feature = "qualification", feature = "hip"))]
mod hip;

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
}
impl GemmaTextNativeOwners {
    #[cfg(all(
        feature = "qualification",
        feature = "hip",
        feature = "hardware-abort-probe"
    ))]
    pub(crate) fn final_owner_probe(
        &self,
    ) -> Result<crate::backend::hip_native_policy::FinalNativeProbe> {
        self.hip
            .as_ref()
            .map(|native| native.final_owner_probe())
            .ok_or_else(|| Error::State("Gemma original prepared native owner absent".into()))
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
    pub(super) fn device_bytes(&self) -> usize {
        self.device_bytes
    }
}
