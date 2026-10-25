//! Backend-neutral final native-owner observation for Gemma reader probes.
//!
//! HIP and Vulkan each retain their own private admission token. This envelope
//! only lets the existing Gemma final-reader test compare the original owner
//! before and after wrapper drop; it grants no numerical or serving permit.
use zerogravity_core::Result;

#[cfg(all(
    feature = "qualification",
    feature = "hip",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
use crate::backend::hip_native_policy;
#[cfg(all(
    feature = "qualification",
    feature = "vulkan",
    feature = "hardware-abort-probe",
    target_os = "linux"
))]
use crate::backend::vulkan_native_policy;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FinalNativeObservation {
    pub(crate) retained: bool,
    pub(crate) modules: usize,
    pub(crate) code_bytes: usize,
    pub(crate) host_bytes: usize,
    pub(crate) device_bytes: usize,
}

pub(crate) enum FinalNativeProbe {
    #[cfg(all(
        feature = "qualification",
        feature = "hip",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    Hip(hip_native_policy::FinalNativeProbe),
    #[cfg(all(
        feature = "qualification",
        feature = "vulkan",
        feature = "hardware-abort-probe",
        target_os = "linux"
    ))]
    Vulkan(vulkan_native_policy::FinalNativeProbe),
}

impl FinalNativeProbe {
    pub(crate) fn observe(&self) -> Result<FinalNativeObservation> {
        match self {
            #[cfg(all(
                feature = "qualification",
                feature = "hip",
                feature = "hardware-abort-probe",
                target_os = "linux"
            ))]
            Self::Hip(probe) => {
                let observation = probe.observe()?;
                Ok(FinalNativeObservation {
                    retained: observation.retained,
                    modules: observation.modules,
                    code_bytes: observation.code_bytes,
                    host_bytes: observation.host_bytes,
                    device_bytes: observation.device_bytes,
                })
            }
            #[cfg(all(
                feature = "qualification",
                feature = "vulkan",
                feature = "hardware-abort-probe",
                target_os = "linux"
            ))]
            Self::Vulkan(probe) => {
                let observation = probe.observe()?;
                Ok(FinalNativeObservation {
                    retained: observation.retained,
                    modules: observation.modules,
                    code_bytes: observation.code_bytes,
                    host_bytes: observation.host_bytes,
                    device_bytes: observation.device_bytes,
                })
            }
            #[allow(unreachable_patterns)]
            _ => Ok(FinalNativeObservation::default()),
        }
    }
}
