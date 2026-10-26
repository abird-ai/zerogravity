//! Actual physical leaves through the original ResidentService; expected-only callback.
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
    teachers: [Vec<u32>; 5],
    leaves: [Qwen2bDualRmsLeaf; 5],
    _payload: WindowWorkspace,
    _metadata: WindowWorkspace,
}
impl Inputs {
    fn new(
        source: RmsLeafSource,
        teachers: [Vec<u32>; 5],
        leaves: [Qwen2bDualRmsLeaf; 5],
        validate_ids: impl Fn(&[u32]) -> Result<()>,
    ) -> Result<Arc<Self>> {
        let metadata = source.source().reserve_workspace(64 * 1024)?;
        let mut bytes = 64 * 1024usize;
        for ((ids, leaf), count) in teachers.iter().zip(&leaves).zip(COUNTS) {
            validate_ids(ids)?;
            validate_fixture(ids, leaf, count)?;
            bytes = bytes
                .checked_add(
                    ids.capacity()
                        .checked_mul(4)
                        .ok_or_else(|| Error::Limit("2B teacher storage overflow".into()))?,
                )
                .and_then(|n| n.checked_add(leaf.branch.capacity()))
                .and_then(|n| n.checked_add(leaf.residual.capacity()))
                .ok_or_else(|| Error::Limit("2B RMS leaf payload overflow".into()))?;
        }
        let payload = source.source().reserve_workspace(bytes)?;
        Ok(Arc::new(Self {
            source,
            teachers,
            leaves,
            _payload: payload,
            _metadata: metadata,
        }))
    }
}
pub(super) fn validate_fixture(ids: &[u32], leaf: &Qwen2bDualRmsLeaf, count: usize) -> Result<()> {
    Rows::new(count)?;
    let bytes = count
        .checked_mul(4096)
        .ok_or_else(|| Error::Limit("2B RMS physical width overflow".into()))?;
    if ids.len() != count
        || ids.iter().any(|id| *id >= 248320)
        || leaf.branch.len() != bytes
        || leaf.residual.len() != bytes
        || !finite_bf16(&leaf.branch)
        || !finite_bf16(&leaf.residual)
    {
        return Err(Error::Malformed(
            "original2B rows/IDs/finite BF16 dual-leaf extents".into(),
        ));
    }
    Ok(())
}

async fn ordinary(
    service: &Arc<ResidentService>,
    owner: &Arc<Inputs>,
    case: usize,
    validate: &mut impl FnMut(Qwen2bRmsReadback<'_>) -> Result<()>,
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
    let prepared = RmsPrepared::prepare_2b(
        service.clone(),
        owner.source.clone(),
        &owner.teachers[case],
        case as u64 + 1,
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
    let mut flight = service.submit(case as u64 + 1, prepared.accesses())?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(prepared.completion_owner())?;
    prepared.enqueue()?;
    let (ready, raw) = flight
        .complete_with_readback(prepared.outputs(), &reads, |raw: &[Bytes]| {
            let normalized = prepared.validate_readback(raw)?;
            validate(Qwen2bRmsReadback::SourceInput {
                case,
                teacher_ids: &owner.teachers[case],
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
    case: usize,
    validate: &mut impl FnMut(Qwen2bRmsReadback<'_>) -> Result<()>,
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
    let mut init = service.begin_initialization(case as u64 + 11)?;
    init.retain_completion_owner(owner.clone())?;
    let leaf = &owner.leaves[case];
    let branch = init.upload_bytes(&leaf.branch, BufferRole::Activation)?;
    let residual = init.upload_bytes(&leaf.residual, BufferRole::Activation)?;
    drop(init.complete_sync().await?);
    let prepared = FusedRmsPrepared::prepare_2b_final_leaf(
        service.clone(),
        owner.source.clone(),
        COUNTS[case],
        branch.clone(),
        residual.clone(),
        case as u64 + 21,
        cancelled,
    )
    .await?;
    slot.finish()?;
    // Original fused budget covers its four readbacks. Admit the two borrowed
    // leaf-input checks separately, including original staging/readback copies.
    let input_reads = Arc::new(
        owner.source.source().reserve_workspace(
            branch
                .bytes()
                .checked_add(residual.bytes())
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(|| Error::Limit("2B dual input readback staging overflow".into()))?,
        )?,
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
    let mut flight = service.submit(case as u64 + 11, prepared.accesses())?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(prepared.completion_owner())?;
    flight.retain_completion_owner(input_reads.clone())?;
    prepared.enqueue(cancelled)?;
    let (ready, raw) = flight
        .complete_with_readback(&outputs, &reads, |raw: &[Bytes]| {
            if raw.len() != 6
                || &raw[4][..] != leaf.branch.as_slice()
                || &raw[5][..] != leaf.residual.as_slice()
            {
                return Err(Error::State("2B actual dual-leaf operands changed".into()));
            }
            prepared.validate_readback(&raw[..4])?;
            validate(Qwen2bRmsReadback::FinalDualLeaf {
                case,
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

/// Ten bounded leaf calls, each with a genuine original physical completion.
/// Ordinary inputs are ORIGINAL source rows/gamma; final dual inputs explicitly
/// remain observed leaves. No StateRoot/PreparedInference/model support follows.
#[expect(clippy::too_many_arguments)]
pub async fn qualify_qwen2b_rms(
    device: Device,
    source: RmsLeafSource,
    teachers: [Vec<u32>; 5],
    leaves: [Qwen2bDualRmsLeaf; 5],
    device_cap: u64,
    mut validate: impl FnMut(Qwen2bRmsReadback<'_>) -> Result<()>,
    cancelled: Option<&AtomicBool>,
    validate_ids: impl Fn(&[u32]) -> Result<()>,
) -> Result<Qwen2bRmsReceipt> {
    // True pre-cancel wins before source metadata, service or provider work.
    not_cancelled(cancelled)?;
    let owner = Inputs::new(source, teachers, leaves, validate_ids)?;
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for2B RMS diagnostic".into(),
        ));
    }
    not_cancelled(cancelled)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        owner.source.source().source_identity(),
        ResourceOwner::new("qwen2b-cuda-source", "ordinary-and-dual-rms-leaves")?,
        PROVIDER,
        device_cap,
    )?;
    let key = service.physical_key();
    for dual in [false, true] {
        for case in 0..COUNTS.len() {
            if dual {
                dual_leaf(&service, &owner, case, &mut validate, cancelled).await?;
            } else {
                ordinary(&service, &owner, case, &mut validate, cancelled).await?;
            }
            if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
                return Err(Error::State(
                    "2B RMS leaves retained allocations/events after completion".into(),
                ));
            }
        }
    }
    drop(service);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "2B RMS physical envelope did not retire".into(),
        ));
    }
    drop(owner);
    let words = COUNTS.iter().sum::<usize>() * 2048;
    Ok(Qwen2bRmsReceipt {
        source_embedding_bf16_words: words,
        ordinary_bf16_words: words,
        dual_sum_f32_words: words,
        dual_normalized_bf16_words: words,
        dual_residual_bf16_words: words,
        actual_stage_dispatches: 100,
    })
}
