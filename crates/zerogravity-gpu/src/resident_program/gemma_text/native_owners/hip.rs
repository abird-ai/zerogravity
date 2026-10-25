//! Family supplies only genuine source-issued concrete kernel specializations.
use super::*;
impl GemmaTextNativeOwners {
    pub(in crate::resident_program::gemma_text) fn profile(
        storage: GemmaTextStorage,
    ) -> Result<crate::backend::hip_native_policy::GemmaColdProfile> {
        use crate::backend::hip_native_policy::GemmaColdProfile;
        match storage {
            GemmaTextStorage::Streamed => Ok(GemmaColdProfile::Streamed16),
            GemmaTextStorage::ResidentProjections => Ok(GemmaColdProfile::Resident20),
            GemmaTextStorage::ResidentProjectionsVulkan => Err(Error::Unsupported(
                "HIP cold policy cannot adopt a Vulkan launch-row selection".into(),
            )),
        }
    }

    pub(crate) async fn prepare_hip_qualification(
        owner: crate::backend::hip_native_policy::HipColdOwner,
        recorded: &Gemma4Bf16TextPlan,
        source_owner: crate::physical_quarantine::CompletionOwner,
        storage: GemmaTextStorage,
        _inventory: &GemmaTextResourceInventory,
    ) -> Result<(Arc<ResidentService>, Arc<Self>)> {
        let cold = match owner {
            crate::backend::hip_native_policy::HipColdOwner::Single(device) => {
                crate::backend::hip_native_policy::begin_gemma_qualification(
                    device,
                    recorded.source().source(),
                    source_owner,
                    Self::profile(storage)?,
                )?
            }
            #[cfg(feature = "qwen-source-qualification")]
            crate::backend::hip_native_policy::HipColdOwner::Shared(shared) => {
                if storage != GemmaTextStorage::ResidentProjections {
                    return Err(Error::Unsupported(
                        "shared Gemma cold candidate requires resident projections".into(),
                    ));
                }
                shared.begin_gemma(recorded.source().source(), source_owner, _inventory)?
            }
        };
        let service = cold.service();
        let specs = math::prepared::enumerate(cold.client(), recorded, storage)?;
        let prepared = cold.prepare(specs.kernels, specs._metadata).await?;
        let owners = Arc::new(Self {
            source: recorded.plan().source(),
            service: service.client().service_id(),
            host_bytes: prepared.host_bytes(),
            device_bytes: prepared.device_bytes(),
            context_host_bytes: Self::profile(storage)?.cost().context_host,
            context_device_bytes: Self::profile(storage)?.cost().context_device,
            _owners: vec![prepared.clone()],
            hip: Some(prepared),
            #[cfg(all(
                feature = "gemma-bf16-qualification",
                feature = "vulkan",
                target_os = "linux"
            ))]
            vulkan: None,
        });
        Ok((service, owners))
    }
}
