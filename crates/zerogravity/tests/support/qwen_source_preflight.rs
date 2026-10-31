//! Real-source cold refusals/accounting; no native or numerical qualification.
use super::source::{Loaded, need};
use std::sync::Arc;
use zerogravity::Result;
use zerogravity::accelerator::program::TextStepBindings;
use zerogravity_gpu::{SourceSetInventory, SourceTextCandidate};

pub fn controls(
    loaded: &Loaded,
    records: &[Arc<TextStepBindings>],
    inventory: &SourceSetInventory,
) -> Result<()> {
    let source = loaded.model.source();
    let before = source.workspace_reserved_bytes()?;
    for sessions in [0, 65] {
        need(
            SourceTextCandidate::preflight(source.clone(), records.to_vec(), sessions).is_err(),
            "cold session population refusal",
        )?;
        need(
            source.workspace_reserved_bytes()? == before,
            "cold refusal releases metadata",
        )?;
    }
    let mut duplicate = records.to_vec();
    duplicate[63] = records[62].clone();
    let mut reordered = records.to_vec();
    reordered.swap(0, 1);
    for bad in [Vec::new(), records[..63].to_vec(), duplicate, reordered] {
        need(
            SourceTextCandidate::preflight(source.clone(), bad, 2).is_err(),
            "cold complete ordered source frontier refusal",
        )?;
        need(
            source.workspace_reserved_bytes()? == before,
            "frontier refusal releases metadata",
        )?;
    }
    {
        let mut mixed = records.to_vec();
        mixed[1] = Arc::new(loaded.model.record_source_token(1, 63)?);
        need(
            SourceTextCandidate::preflight(source.clone(), mixed, 2).is_err(),
            "mixed source capacity refusal",
        )?;
    }
    need(
        source.workspace_reserved_bytes()? == before,
        "mixed capacity releases original metadata",
    )?;
    for (position, capacity) in [(0, 0), (64, 64), (0, 65)] {
        need(
            loaded
                .model
                .record_source_token(position, capacity)
                .is_err(),
            "source arithmetic bound remains explicit",
        )?;
    }
    need(
        loaded.model.record_full_source_token(0).is_err(),
        "finite27 adapter is not widened to this source",
    )?;
    need(
        source.workspace_reserved_bytes()? == before,
        "record refusals release metadata",
    )?;
    let serial = SourceTextCandidate::preflight(source.clone(), records.to_vec(), 1)?;
    let many = SourceTextCandidate::preflight(source.clone(), records.to_vec(), 64)?;
    need(
        serial.root_count == 4
            && inventory.root_count == 6
            && many.root_count == 130
            && serial.resident_weight_bytes == many.resident_weight_bytes
            && serial.workspace_bytes == many.workspace_bytes
            && many.model_device_upper_bytes - serial.model_device_upper_bytes
                == 126 * inventory.root_bytes
            && many.physical_device_upper_bytes
                == many.model_device_upper_bytes + many.context_device_bytes,
        "one shared weight/workspace owner, independently priced root population",
    )?;
    need(
        source.workspace_reserved_bytes()? == before,
        "cold pricing has no retained preparation",
    )?;
    println!(
        "SOURCE_MODEL_COLD_CONTROLS refusals=11 root_counts=4,6,130 device_context_created=false numerical_qualified=false serving_qualified=false"
    );
    Ok(())
}
