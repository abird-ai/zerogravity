//! Host-only admission sweep. Estimates do not reserve memory or qualify inference.
use super::source::need;
use zerogravity::{Error, Result, gpu::ResidentModel, qwen35::plan::QwenForwardPlan};

pub fn inspect(original: &QwenForwardPlan, label: &str) -> Result<()> {
    let source = original.source();
    let baseline = source.workspace_reserved_bytes()?;
    for context in [512, 1024, 2048, 4096, 8192, 16384, 32768] {
        for active in [1, 2, 4, 8] {
            {
                let plan = original.source_seal().build_plan(context)?;
                need(
                    plan.model_profile() == original.model_profile(),
                    "same source capacity geometry",
                )?;
                match ResidentModel::cpu_forward_source_preflight(
                    source.clone(),
                    plan.plan().clone(),
                    active,
                ) {
                    Ok(inventory) => {
                        need(
                            inventory.root_count == 2 * active + 2
                                && inventory.context_device_bytes == 4 * 1024 * 1024 * 1024
                                && inventory.physical_device_upper_bytes <= 24 * 1024 * 1024 * 1024
                                && inventory.host_upper_bytes <= 16 * 1024 * 1024 * 1024
                                && inventory.source_weight_bytes > 0
                                && (inventory.resident_weight_bytes == 0
                                    || inventory.resident_weight_bytes
                                        == inventory.source_weight_bytes),
                            "actual capacity/root/storage caps",
                        )?;
                        println!(
                            "SOURCE_CONTEXT_PREFLIGHT model={label} context={context} active={active} admitted=true inventory={inventory:?} reservation=false gpu=false"
                        );
                    }
                    Err(Error::Resource(reason)) => {
                        println!(
                            "SOURCE_CONTEXT_PREFLIGHT model={label} context={context} active={active} admitted=false refusal={reason:?} reservation=false gpu=false"
                        );
                    }
                    Err(error) => return Err(error),
                }
            }
            need(
                source.workspace_reserved_bytes()? == baseline,
                "capacity/refusal releases original metadata",
            )?;
        }
    }
    Ok(())
}
