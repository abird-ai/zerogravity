//! Call-free CPU-source lowering under the original resident physical owner.
//! Model topology and source issuers remain in the original ForwardPlan.
use super::{
    ForwardPlan, PlanOperation, ResidentBufferLayout, ResidentPhysicalType, ResidentPreflight,
    StateLayout, StateRequirement, StateSlot,
};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_core::{Error, Result, accelerator::program::NumericalContract};
mod body;
mod cohort;
mod cold;
#[cfg(all(
    feature = "hardware-abort-probe",
    feature = "qwen-source-qualification"
))]
mod diagnostic;
mod execute;
mod kernels;
mod layout;
mod math;
mod observation;
mod prepared;
pub(crate) use cohort::{CohortBuffers, CohortSubmission};
pub(super) use cold::Cold;

fn tables() -> Vec<Vec<u8>> {
    use zerogravity_core::accelerator::program::standalone_exp::{
        MKL_EXHA_EXP2_64_F64, MKL_EXHA_SSE2_EXP2_128,
    };
    vec![
        MKL_EXHA_SSE2_EXP2_128
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        crate::source_transcendentals::TABLES
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        crate::source_trig::cpu::TABLES
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        (0..32)
            .flat_map(|i| (1.0f32 / 10_000_000f32.powf((2 * i) as f32 / 64.0)).to_le_bytes())
            .collect(),
        MKL_EXHA_EXP2_64_F64
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    ]
}

pub(crate) use cold::Admission;
pub use cold::ResidentTextInventory;
pub(crate) struct Prepared {
    pub program: super::ResidentProgram,
    pub accesses: Vec<(
        crate::resident_leases::ResidentLease,
        zerogravity_core::device_events::DeviceAccess,
    )>,
    pub service: Arc<crate::resident_leases::ResidentService>,
    pub inventory: ResidentTextInventory,
}

pub(crate) async fn prepare(
    device: cubecl::Device,
    admission: Admission,
    source_owner: Arc<dyn std::any::Any + Send + Sync>,
    cancelled: Option<&std::sync::atomic::AtomicBool>,
) -> Result<Prepared> {
    use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest};
    let check = || {
        if cancelled.is_some_and(|f| f.load(std::sync::atomic::Ordering::Acquire)) {
            Err(Error::State("text-execution preparation cancelled".into()))
        } else {
            Ok(())
        }
    };
    check()?;
    let inventory = admission.inventory().clone();
    let native = crate::hip::native::begin_resident_source_with_budget(
        device,
        &admission.source,
        source_owner,
        inventory.model_device_upper_bytes,
        admission.budget,
        admission.native_policy,
    )?;
    let catalogue = admission.catalogue(native.client())?;
    let service = native.service();
    let native = native.prepare(catalogue.0, catalogue.1).await?;
    check()?;
    let initialization_host = admission.source.reserve_workspace(32 * 1024 * 1024)?;
    let mut slot = service
        .queue()
        .submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: (inventory.resident_weight_bytes + inventory.workspace_bytes) as u64,
            }],
            None,
        )?
        .wait()
        .await?;
    check()?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(0)?;
    init.retain_completion_owner(Arc::new(initialization_host))?;
    // Once transfer submission starts, retain and complete original owners even
    // if the caller cancels. No cancelled future is raced or dropped here.
    let pending = admission
        .prepare(service.clone(), &mut init, native, None)
        .await?;
    let ready = init.complete_sync().await?;
    let cold = pending.bind()?;
    if ready.outputs().len() != cold.accesses.len() {
        return Err(Error::State(
            "text-execution initialization owner coverage".into(),
        ));
    }
    let program = cold.bind()?;
    let accesses = cold.accesses.clone();
    slot.finish()?;
    check()?;
    Ok(Prepared {
        program,
        accesses,
        service,
        inventory,
    })
}
