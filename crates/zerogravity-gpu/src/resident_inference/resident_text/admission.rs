//! Immutable selected fit, derived from the actual resident loader inventory.
use super::*;

#[cfg(all(feature = "hip-source", feature = "hip"))]
pub(super) fn hip(
    service: &ResidentService,
    inventory: &ResidentTextInventory,
    max_sessions: usize,
    cohort_width: usize,
) -> Result<ResidentTextAdmission> {
    let report = ResidentTextAdmission {
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
    require_service(&report, service)?;
    Ok(report)
}

#[cfg(feature = "cuda-cublas")]
pub(super) fn cuda(
    service: &ResidentService,
    inventory: &CudaForwardSourceInventory,
    configured_model_capacity_bytes: usize,
    max_sessions: usize,
) -> Result<ResidentTextAdmission> {
    let report = ResidentTextAdmission {
        device: service.physical_key(),
        configured_model_capacity_bytes,
        admitted_model_capacity_bytes: inventory.provider_capacity_bytes,
        model_required_bytes: inventory.total_admission_bytes,
        context_device_budget_bytes: crate::physical_service::CUDA_COMPILER_DEVICE_BYTES as usize,
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
    require_service(&report, service)?;
    Ok(report)
}

fn require_service(report: &ResidentTextAdmission, service: &ResidentService) -> Result<()> {
    if report.device != service.physical_key()
        || report.admitted_model_capacity_bytes as u64 != service.budget().provider_capacity()
        || report.physical_limit_bytes != service.budget().device_capacity()
        || report.admitted_model_capacity_bytes > report.configured_model_capacity_bytes
        || report.model_required_bytes > report.admitted_model_capacity_bytes
        || report
            .admitted_model_capacity_bytes
            .checked_add(report.context_device_budget_bytes)
            .is_none_or(|total| total as u64 > report.physical_limit_bytes)
        || report.state_reserve_bytes > report.model_required_bytes
        || report.resident_weight_bytes > report.source_weight_bytes
        || report.root_bytes == 0
        || report
            .max_sessions
            .checked_mul(2)
            .and_then(|n| n.checked_add(2))
            != Some(report.root_count)
        || report.cohort_width > report.max_sessions
    {
        return Err(Error::State(
            "resident text fit differs from original admission".into(),
        ));
    }
    Ok(())
}
