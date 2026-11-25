//! Window-backed resident text preparation through the same semantic contract.
use super::*;
use crate::ResidentTextModel;
use std::{any::Any, sync::atomic::AtomicBool};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;

pub enum ResidentTextPolicy {
    #[cfg(all(feature = "hip-source", feature = "hip"))]
    Hip(crate::hip::native::HipSourceBudget),
    #[cfg(feature = "cuda-cublas")]
    Cuda {
        provider_bytes: usize,
        device_bytes: u64,
        state: crate::CudaSourceStatePolicy,
    },
}

/// Original family/source lease and selected cold placement policy.
/// The existing resident loader retains the source lease through completion.
pub struct ResidentTextProvider<'a> {
    pub device: Device,
    pub policy: ResidentTextPolicy,
    pub source_owner: Arc<dyn Any + Send + Sync>,
    pub max_sessions: usize,
    pub cohort_width: usize,
    pub cancelled: Option<&'a AtomicBool>,
}
impl RegionExecutionProvider for ResidentTextProvider<'_> {
    type Executable = Arc<ResidentTextModel>;

    fn selected(&self) -> SelectedMode {
        use zerogravity_core::provider::ProviderKind;
        let provider = match self.policy {
            #[cfg(all(feature = "hip-source", feature = "hip"))]
            ResidentTextPolicy::Hip(_) => ProviderKind::Hip,
            #[cfg(feature = "cuda-cublas")]
            ResidentTextPolicy::Cuda { .. } => ProviderKind::Cuda,
        };
        SelectedMode {
            provider,
            mode: ExecutionMode::Native,
        }
    }

    async fn prepare(
        &self,
        program: ModelProgram<VerifiedTensorSource>,
    ) -> Result<Self::Executable> {
        if Provider::selected(&self.device)?.kind() != self.kind() {
            return Err(zerogravity_core::Error::Unsupported(
                "resident text policy contradicts explicitly selected device".into(),
            ));
        }
        let (source, plan) = program.into_parts();
        match self.policy {
            #[cfg(all(feature = "hip-source", feature = "hip"))]
            ResidentTextPolicy::Hip(budget) => {
                ResidentModel::load_hip_forward_source_with_budget_and_cohort(
                    self.device.clone(),
                    source,
                    self.source_owner.clone(),
                    plan,
                    self.max_sessions,
                    self.cohort_width,
                    budget,
                    self.cancelled,
                )
                .await
            }
            #[cfg(feature = "cuda-cublas")]
            ResidentTextPolicy::Cuda {
                provider_bytes,
                device_bytes,
                state,
            } => {
                ResidentModel::load_cuda_forward_source_with_state_policy(
                    self.device.clone(),
                    source,
                    self.source_owner.clone(),
                    plan,
                    self.max_sessions,
                    self.cohort_width,
                    provider_bytes,
                    device_bytes,
                    state,
                    self.cancelled,
                )
                .await
            }
        }
    }
}
