//! Two actual fixed47 leaf completions; original fair queue and resident owners.
use super::*;
use crate::cuda::rms::RmsLeafSource;
use crate::resident_leases::{ResidentLease, ResidentService};
use cubecl::{Device, bytes::Bytes};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
const PROVIDER: usize = 128 * 1024 * 1024;
struct Inputs {
    source: RmsLeafSource,
    teachers: [u32; COUNT],
    leaf: FixedPromptDualRmsLeaf,
    _payload: Option<WindowWorkspace>,
    _metadata: WindowWorkspace,
}
impl Inputs {
    fn new(
        source: RmsLeafSource,
        teachers: [u32; COUNT],
        leaf: FixedPromptDualRmsLeaf,
        validate_ids: impl Fn(&[u32]) -> Result<()>,
    ) -> Result<Arc<Self>> {
        validate_ids(&teachers)?;
        let metadata = source.source().reserve_workspace(64 * 1024)?;
        let mut owner = Self {
            source,
            teachers,
            leaf,
            _payload: None,
            _metadata: metadata,
        };
        let bytes = owner
            .leaf
            .branch
            .capacity()
            .checked_add(owner.leaf.residual.capacity())
            .and_then(|n| n.checked_add(64 * 1024))
            .ok_or_else(|| Error::Limit("fixed_prompt leaf payload overflow".into()))?;
        // Install the charge in a payload-first composite before fallible checks.
        owner._payload = Some(owner.source.source().reserve_workspace(bytes)?);
        validate_fixture(&owner.teachers, &owner.leaf)?;
        Ok(Arc::new(owner))
    }
}

async fn ordinary(
    service: &Arc<ResidentService>,
    owner: &Arc<Inputs>,
    validate: &mut impl FnMut(FixedPromptRmsReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    let requests = [AllocationRequest {
        kind: AllocationClass::Other,
        bytes: PROVIDER as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let prepared = RmsPrepared::prepare_fixed_prompt(
        service.clone(),
        owner.source.clone(),
        &owner.teachers,
        1,
        cancelled,
    )
    .await?;
    slot.finish()?;
    let reads = prepared.readbacks()?;
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut flight = service.submit(1, prepared.accesses())?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(prepared.completion_owner())?;
    prepared.enqueue()?;
    let (ready, raw) = flight
        .complete_with_readback(prepared.outputs(), &reads, |raw: &[Bytes]| {
            let normalized = prepared.validate_readback(raw)?;
            validate(FixedPromptRmsReadback::SourceInput {
                teacher_ids: &owner.teachers,
                embedding: &raw[1][..],
                normalized,
            })
        })
        .await?;
    slot.finish()?;
    drop((ready, raw, reads, prepared));
    Ok(())
}

async fn dual_leaf(
    service: &Arc<ResidentService>,
    owner: &Arc<Inputs>,
    validate: &mut impl FnMut(FixedPromptRmsReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
) -> Result<()> {
    let requests = [AllocationRequest {
        kind: AllocationClass::Other,
        bytes: PROVIDER as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut init = service.begin_initialization(2)?;
    init.retain_completion_owner(owner.clone())?;
    let branch = init.upload_bytes(&owner.leaf.branch, BufferRole::Activation)?;
    let residual = init.upload_bytes(&owner.leaf.residual, BufferRole::Activation)?;
    drop(init.complete_sync().await?);
    let prepared = FusedRmsPrepared::prepare_fixed_prompt_final_leaf(
        service.clone(),
        owner.source.clone(),
        branch.clone(),
        residual.clone(),
        3,
        cancelled,
    )
    .await?;
    slot.finish()?;
    // Existing fused owner prepays four reads; admit both extra leaf checks.
    let input_reads = Arc::new(
        owner
            .source
            .source()
            .reserve_workspace((branch.bytes() + residual.bytes()) * 3)?,
    );
    let mut reads = Vec::from(prepared.readbacks()?);
    reads.push(branch.readback(0, branch.bytes())?);
    reads.push(residual.readback(0, residual.bytes())?);
    let outputs: Vec<ResidentLease> = prepared
        .accesses()
        .iter()
        .filter(|(_, access)| *access == DeviceAccess::Write)
        .map(|(lease, _)| lease.clone())
        .collect();
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    not_cancelled(cancelled)?;
    let mut flight = service.submit(2, prepared.accesses())?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(prepared.completion_owner())?;
    flight.retain_completion_owner(input_reads.clone())?;
    prepared.enqueue(cancelled)?;
    let (ready, raw) = flight
        .complete_with_readback(&outputs, &reads, |raw: &[Bytes]| {
            if raw.len() != 6
                || &raw[4][..] != owner.leaf.branch.as_slice()
                || &raw[5][..] != owner.leaf.residual.as_slice()
            {
                return Err(Error::State(
                    "fixed_prompt dual leaf operands changed".into(),
                ));
            }
            prepared.validate_readback(&raw[..4])?;
            validate(FixedPromptRmsReadback::FinalDualLeaf {
                sum_f32: &raw[1][..],
                normalized: &raw[2][..],
                residual: &raw[3][..],
            })
        })
        .await?;
    slot.finish()?;
    drop((
        ready,
        raw,
        outputs,
        reads,
        prepared,
        branch,
        residual,
        input_reads,
    ));
    Ok(())
}

/// Original source embedding/input gamma, then explicit final dual LEAF.
/// Expected words may be opened only in `validate`, after actual completion.
#[expect(clippy::too_many_arguments)]
pub async fn qualify_fixed_prompt_rms(
    device: Device,
    source: RmsLeafSource,
    teachers: [u32; COUNT],
    leaf: FixedPromptDualRmsLeaf,
    device_cap: u64,
    mut validate: impl FnMut(FixedPromptRmsReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
    validate_ids: impl Fn(&[u32]) -> Result<()>,
) -> Result<FixedPromptRmsReceipt> {
    not_cancelled(cancelled)?;
    let owner = Inputs::new(source, teachers, leaf, validate_ids)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for fixed_prompt RMS leaf".into(),
        ));
    }
    not_cancelled(cancelled)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        owner.source.source().source_identity(),
        ResourceOwner::new("qwen08-cuda-source", "fixed_prompt-rms-leaves")?,
        PROVIDER,
        device_cap,
    )?;
    let key = service.physical_key();
    ordinary(&service, &owner, &mut validate, cancelled).await?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "fixed_prompt ordinary retained allocations/events".into(),
        ));
    }
    dual_leaf(&service, &owner, &mut validate, cancelled).await?;
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "fixed_prompt dual retained allocations/events".into(),
        ));
    }
    drop(service);
    let physical = ResourceArbiter::process_shared().snapshot(&key)?;
    if physical.reserved != 0 || physical.fenced {
        return Err(Error::State(
            "fixed_prompt RMS original envelope did not retire cleanly".into(),
        ));
    }
    drop(owner);
    Ok(FixedPromptRmsReceipt {
        source_embedding_bf16_words: WORDS,
        ordinary_bf16_words: WORDS,
        dual_sum_f32_words: WORDS,
        dual_normalized_bf16_words: WORDS,
        dual_residual_bf16_words: WORDS,
        actual_stage_dispatches: 20,
        readback_calls: 2,
    })
}
