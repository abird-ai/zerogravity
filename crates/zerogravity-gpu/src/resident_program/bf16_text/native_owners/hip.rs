//! Family supplies only genuine source-issued concrete kernel specializations.
use super::*;
impl TextNativeOwners {
    pub(in crate::resident_program::bf16_text) fn profile(
        storage: TextStorage,
    ) -> Result<crate::hip::native::GemmaColdProfile> {
        use crate::hip::native::GemmaColdProfile;
        match storage {
            TextStorage::Streamed => Ok(GemmaColdProfile::Streamed16),
            TextStorage::ResidentProjections => Ok(GemmaColdProfile::Resident20),
            TextStorage::ResidentProjectionsVulkan => Err(Error::Unsupported(
                "HIP cold policy cannot adopt a Vulkan launch-row selection".into(),
            )),
        }
    }

    pub(crate) async fn prepare_hip_qualification(
        owner: crate::hip::native::HipColdOwner,
        recorded: &TextSourcePlan,
        source_owner: crate::physical_quarantine::CompletionOwner,
        storage: TextStorage,
        _inventory: &TextResourceInventory,
    ) -> Result<(Arc<ResidentService>, Arc<Self>)> {
        let cold = match owner {
            crate::hip::native::HipColdOwner::Single(device) => {
                crate::hip::native::begin_gemma_qualification(
                    device,
                    recorded.source(),
                    source_owner,
                    Self::profile(storage)?,
                )?
            }
            #[cfg(feature = "qwen-source-qualification")]
            crate::hip::native::HipColdOwner::Shared(shared) => {
                if storage != TextStorage::ResidentProjections {
                    return Err(Error::Unsupported(
                        "shared BF16 text cold candidate requires resident projections".into(),
                    ));
                }
                shared.begin_gemma(recorded.source(), source_owner, _inventory)?
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
