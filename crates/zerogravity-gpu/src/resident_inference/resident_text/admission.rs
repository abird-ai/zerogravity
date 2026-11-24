//! Immutable selected fit, derived from the actual resident loader inventory.
use super::*;
use zerogravity_core::resource_arbiter::PhysicalDeviceKey;

/// Cold resource bounds and original physical admission, not current VRAM
/// usage. Context budgets are separate from model capacity/required bytes.
/// Paged state reserve can differ from logical root bytes * root count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentTextAdmission {
    pub device: PhysicalDeviceKey,
    pub configured_model_capacity_bytes: usize,
    pub admitted_model_capacity_bytes: usize,
    pub model_required_bytes: usize,
    pub context_device_budget_bytes: usize,
    pub physical_limit_bytes: u64,
    pub host_upper_bytes: Option<usize>,
    pub source_weight_bytes: usize,
    pub resident_weight_bytes: usize,
    pub root_bytes: usize,
    pub root_count: usize,
    pub state_reserve_bytes: usize,
    pub max_sessions: usize,
    pub cohort_width: usize,
}
impl ResidentTextAdmission {
    #[cfg(all(feature = "hip-source", feature = "hip"))]
    pub(super) fn hip(
        service: &ResidentService,
        inventory: &ResidentTextInventory,
        max_sessions: usize,
        cohort_width: usize,
    ) -> Result<Self> {
        let report = Self {
            device: service.physical_key(),
            configured_model_capacity_bytes: inventory.model_device_budget_bytes,
            admitted_model_capacity_bytes: inventory.model_device_budget_bytes,
            model_required_bytes: inventory.model_device_upper_bytes,
            context_device_budget_bytes: inventory.context_device_bytes,
            physical_limit_bytes: inventory.physical_admission_bytes,
            host_upper_bytes: Some(inventory.host_upper_bytes),
            source_weight_bytes: inventory.source_weight_bytes,
            resident_weight_bytes: inventory.resident_weight_bytes,
            root_bytes: inventory.root_bytes,
            root_count: inventory.root_count,
            state_reserve_bytes: inventory.state_reserve_bytes,
            max_sessions,
            cohort_width,
        };
        if report
            .admitted_model_capacity_bytes
            .checked_add(report.context_device_budget_bytes)
            != Some(inventory.total_device_budget_bytes)
            || report
                .model_required_bytes
                .checked_add(report.context_device_budget_bytes)
                != Some(inventory.physical_device_upper_bytes)
        {
            return Err(Error::State(
                "HIP source original context partition differs".into(),
            ));
        }
        report.require_service(service)?;
        Ok(report)
    }

    #[cfg(feature = "cuda-cublas")]
    pub(super) fn cuda(
        service: &ResidentService,
        inventory: &CudaForwardSourceInventory,
        configured_model_capacity_bytes: usize,
        max_sessions: usize,
    ) -> Result<Self> {
        let report = Self {
            device: service.physical_key(),
            configured_model_capacity_bytes,
            admitted_model_capacity_bytes: inventory.provider_capacity_bytes,
            model_required_bytes: inventory.total_admission_bytes,
            context_device_budget_bytes: crate::physical_service::CUDA_COMPILER_DEVICE_BYTES
                as usize,
            physical_limit_bytes: service.budget().device_capacity(),
            host_upper_bytes: None, // No complete cold CUDA host upper reported.
            source_weight_bytes: inventory.source_weight_bytes,
            resident_weight_bytes: inventory.resident_weight_bytes,
            root_bytes: inventory.root_bytes,
            root_count: inventory.root_count,
            state_reserve_bytes: inventory.state_reserve_bytes,
            max_sessions,
            cohort_width: inventory.cohort_width,
        };
        report.require_service(service)?;
        Ok(report)
    }

    fn require_service(&self, service: &ResidentService) -> Result<()> {
        if self.device != service.physical_key()
            || self.admitted_model_capacity_bytes as u64 != service.budget().provider_capacity()
            || self.physical_limit_bytes != service.budget().device_capacity()
            || self.admitted_model_capacity_bytes > self.configured_model_capacity_bytes
            || self.model_required_bytes > self.admitted_model_capacity_bytes
            || self
                .admitted_model_capacity_bytes
                .checked_add(self.context_device_budget_bytes)
                .is_none_or(|total| total as u64 > self.physical_limit_bytes)
            || self.state_reserve_bytes > self.model_required_bytes
            || self.resident_weight_bytes > self.source_weight_bytes
            || self.root_bytes == 0
            || self
                .max_sessions
                .checked_mul(2)
                .and_then(|n| n.checked_add(2))
                != Some(self.root_count)
            || self.cohort_width > self.max_sessions
        {
            return Err(Error::State(
                "resident text fit differs from original admission".into(),
            ));
        }
        Ok(())
    }
}
