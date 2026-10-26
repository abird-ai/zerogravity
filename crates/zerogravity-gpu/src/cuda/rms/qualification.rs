//! Original source rows → ten CUDA RMS stores → independent native BF16 bits.
//! A diagnostic source gate; no StateRoot, model route or numerical permit.
use super::{RmsPrepared, resident::finite_bf16};
use crate::resident_leases::ResidentService;
use cubecl::{Device, bytes::Bytes};
use std::sync::Arc;
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, ResourceArbiter, ResourceOwner,
};
use zerogravity_core::{Error, Result};

const PROVIDER_CAP: usize = 128 * 1024 * 1024;
const METADATA_CAP: usize = 2 * 1024 * 1024;
const VOCABULARY: usize = 248_320;
const COUNTS: [usize; 5] = [1, 1, 64, 2, 1];

/// ROOT pins genuine scheduled IDs/output records before invoking this seam.
/// There is deliberately no observed embedding/normalized input field.
pub struct InputRmsFixture {
    pub teacher_ids: Vec<u32>,
    pub expected: Vec<u8>,
}

struct FixtureOwner {
    source: VerifiedTensorSource,
    fixtures: [InputRmsFixture; 5],
    _metadata: WindowWorkspace,
    _payload: WindowWorkspace,
}
impl FixtureOwner {
    fn new(source: VerifiedTensorSource, fixtures: [InputRmsFixture; 5]) -> Result<Arc<Self>> {
        let metadata = source.reserve_workspace(METADATA_CAP)?;
        let mut payload_bytes: usize = 64 * 1024;
        for (fixture, count) in fixtures.iter().zip(COUNTS) {
            if fixture.teacher_ids.len() != count
                || fixture.expected.len() != count * 2048
                || !finite_bf16(&fixture.expected)
                || fixture
                    .teacher_ids
                    .iter()
                    .any(|id| *id as usize >= VOCABULARY)
            {
                return Err(Error::Malformed(
                    "actual input RMS teacher/output geometry".into(),
                ));
            }
            payload_bytes = payload_bytes
                .checked_add(fixture.expected.capacity())
                .and_then(|bytes| bytes.checked_add(fixture.teacher_ids.capacity().checked_mul(4)?))
                .ok_or_else(|| Error::Limit("input RMS fixture host envelope overflow".into()))?;
        }
        if fixtures[0].teacher_ids != [11]
            || fixtures[1].teacher_ids != [353]
            || fixtures[4].teacher_ids != [353]
        {
            return Err(Error::Malformed("genuine source teachers11/353".into()));
        }
        let payload = source.reserve_workspace(payload_bytes)?;
        Ok(Arc::new(Self {
            source,
            fixtures,
            _metadata: metadata,
            _payload: payload,
        }))
    }
}

async fn source_rms(
    service: Arc<ResidentService>,
    owner: Arc<FixtureOwner>,
    index: usize,
) -> Result<usize> {
    let fixture = &owner.fixtures[index];
    let requests = [AllocationRequest {
        kind: AllocationClass::Other,
        bytes: PROVIDER_CAP as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &requests, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let prepared = RmsPrepared::prepare(
        service.clone(),
        owner.source.clone(),
        &fixture.teacher_ids,
        index as u64 + 1,
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
    let mut flight = service.submit(index as u64 + 1, prepared.accesses())?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(prepared.completion_owner())?;
    prepared.enqueue()?;
    let (ready, readback) = flight
        .complete_with_readback(prepared.outputs(), &reads, |actual: &[Bytes]| {
            let actual = prepared.validate_readback(actual)?;
            if actual != fixture.expected.as_slice() {
                let first = actual
                    .chunks_exact(2)
                    .zip(fixture.expected.chunks_exact(2))
                    .position(|(a, b)| a != b)
                    .unwrap_or(0);
                return Err(Error::State(format!(
                    "original source input RMS call{index} differs at BF16 word{first}"
                )));
            }
            Ok(())
        })
        .await?;
    slot.finish()?;
    drop((ready, readback, reads, prepared));
    Ok(COUNTS[index] * 1024)
}

/// Five original V1 calls in observed order. GPU inputs derive exclusively from
/// authenticated original source windows. Native words are compared only in
/// the completion validator after all ten real operators have executed.
pub async fn qualify_input_rms(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [InputRmsFixture; 5],
    device_cap: u64,
) -> Result<usize> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for input RMS source gate".into(),
        ));
    }
    let owner = FixtureOwner::new(source.clone(), fixtures)?;
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("qwen08-cuda-source", "embedding-input-rms")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    let mut compared = 0;
    for index in 0..COUNTS.len() {
        compared += source_rms(service.clone(), owner.clone(), index).await?;
        if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
            return Err(Error::State(
                "input RMS retained ordinary completed device allocations/events".into(),
            ));
        }
    }
    drop(service);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "input RMS original physical envelope did not retire".into(),
        ));
    }
    drop(owner);
    Ok(compared)
}
