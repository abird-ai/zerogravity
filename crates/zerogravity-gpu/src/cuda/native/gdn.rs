//! Connected GDN leaf with explicitly observed QKVZ/BA projection inputs.
//! Original parameters and computed caches; no upstream/model-route claim.
use super::layer_common::{LayerExpectedRecord, LayerOutput, Snapshot, load_parameters};
use super::*;
use crate::cuda::gdn::{GdnCacheSlot, GdnInputs, GdnPrepared};
use crate::resident_leases::ResidentReadback;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};

mod faults;
pub use faults::{GdnFault, GdnFaultOutcome, qualify_gdn_fault};

const HOST_BYTES: usize = 32 * 1024 * 1024;
const METADATA_BYTES: usize = 4 * 1024 * 1024;
const PARAMETER_BYTES: usize = 6144 * 4 * 2 + 16 * 4 + 16 * 2 + 128 * 2;
const PHASES: [GdnPhase; 5] = [
    GdnPhase::InitialOne,
    GdnPhase::CachedDecode,
    GdnPhase::Initial64,
    GdnPhase::ContinuedTwo,
    GdnPhase::CachedDecode,
];

/// ROOT authenticates the captured projection boundary and expected records.
/// Only these explicit QKVZ/BA inputs enter computation; expectations never do.
pub struct GdnFixture {
    pub qkvz: Vec<u8>,
    pub ba: Vec<u8>,
    pub expected: Vec<LayerExpectedRecord>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GdnComparison {
    pub calls: usize,
    pub gdn_launches: usize,
    pub bf16_words: usize,
    pub f32_words: usize,
}

struct GdnOwner {
    _source: Arc<VerifiedTensorSource>,
    _parameters: Vec<Arc<ParameterWindow>>,
    parameters: [Vec<u8>; 4],
    fixtures: [GdnFixture; 5],
    _host: WindowWorkspace,
}

struct Call {
    gdn: Arc<GdnPrepared>,
    phase: GdnPhase,
    slot: GdnCacheSlot,
    snapshot: Option<Snapshot>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
    reads: Vec<ResidentReadback>,
}
impl Call {
    fn readback(&self, output: LayerOutput) -> Result<ResidentReadback> {
        let bytes = output.bytes(self.phase);
        let cache = self.gdn.lease(GdnBuffer::CacheArena)?;
        let (lease, offset) = match output {
            LayerOutput::Qkvz => (
                match &self.snapshot {
                    Some(snapshot) => &snapshot.lease,
                    None => self.gdn.lease(GdnBuffer::PackedQkvz)?,
                },
                0,
            ),
            LayerOutput::Convolution => (
                self.gdn.lease(if self.phase == GdnPhase::CachedDecode {
                    GdnBuffer::PackedQkvz
                } else {
                    GdnBuffer::ConvOutput
                })?,
                0,
            ),
            LayerOutput::ConvolutionState => (cache, self.slot.convolution_offset()),
            LayerOutput::RecurrentState => (cache, self.slot.recurrent_offset()),
            _ => (
                self.gdn.lease(
                    output
                        .role()
                        .ok_or_else(|| Error::State("closed GDN leaf output role".into()))?,
                )?,
                0,
            ),
        };
        lease.readback(offset, bytes)
    }
}

fn required(phase: GdnPhase) -> BTreeSet<LayerOutput> {
    LayerOutput::required(phase)
        .into_iter()
        .filter(|output| !matches!(output, LayerOutput::Normalized | LayerOutput::Projected))
        .collect()
}

fn finite_f32(raw: &[u8]) -> bool {
    raw.len().is_multiple_of(4)
        && raw
            .chunks_exact(4)
            .all(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()).is_finite())
}

fn validate_fixtures(fixtures: &[GdnFixture; 5]) -> Result<()> {
    let mut retained = 0usize;
    let mut readback_peak = 0usize;
    for (fixture, phase) in fixtures.iter().zip(PHASES) {
        if fixture.qkvz.len() != GdnBuffer::PackedQkvz.bytes(phase)
            || fixture.ba.len() != GdnBuffer::PackedBa.bytes(phase)
            || !finite_bf16(&fixture.qkvz)
            || !finite_bf16(&fixture.ba)
        {
            return Err(Error::Malformed(
                "actual GDN projection input geometry".into(),
            ));
        }
        retained = retained
            .checked_add(fixture.qkvz.capacity())
            .and_then(|n| n.checked_add(fixture.ba.capacity()))
            .and_then(|n| {
                n.checked_add(
                    fixture
                        .expected
                        .capacity()
                        .checked_mul(std::mem::size_of::<LayerExpectedRecord>())?,
                )
            })
            .ok_or_else(|| Error::Limit("GDN leaf retained fixture overflow".into()))?;
        let mut outputs = BTreeSet::new();
        let mut readback = 0usize;
        for record in &fixture.expected {
            let finite = if record.output.word_bytes() == 4 {
                finite_f32(&record.bytes)
            } else {
                finite_bf16(&record.bytes)
            };
            if !outputs.insert(record.output)
                || record.bytes.len() != record.output.bytes(phase)
                || !finite
                || record.output == LayerOutput::Qkvz && record.bytes != fixture.qkvz
                || record.output == LayerOutput::Ba && record.bytes != fixture.ba
            {
                return Err(Error::Malformed(
                    "complete finite GDN leaf records/boundary".into(),
                ));
            }
            retained = retained
                .checked_add(record.bytes.capacity())
                .ok_or_else(|| Error::Limit("GDN leaf expected capacity overflow".into()))?;
            readback = readback
                .checked_add(record.bytes.len())
                .ok_or_else(|| Error::Limit("GDN leaf readback overflow".into()))?;
        }
        if outputs != required(phase) {
            return Err(Error::Malformed(
                "exact GDN leaf prefill18/decode7 records".into(),
            ));
        }
        // Four unchanged source parameter buffers accompany every completion.
        readback_peak = readback_peak.max(readback + PARAMETER_BYTES);
    }
    // A preparation barrier separates the two cache uploads: at most one
    // original zero buffer + one cache staging copy. Other inputs may queue.
    // The zero host buffer drops before specs/flight readbacks are prepared.
    let input_bytes = fixtures
        .iter()
        .map(|f| f.qkvz.len() + f.ba.len())
        .sum::<usize>();
    let cache_peak =
        GdnBuffer::CacheArena.bytes(GdnPhase::InitialOne) * 2 + input_bytes + PARAMETER_BYTES * 2;
    let required = retained
        .checked_add(METADATA_BYTES)
        .and_then(|n| n.checked_add(cache_peak.max(readback_peak.checked_mul(2)?)))
        .ok_or_else(|| Error::Limit("GDN leaf host envelope overflow".into()))?;
    if required > HOST_BYTES {
        return Err(Error::Limit(
            "GDN leaf source/fixture/staging exceeds32MiB".into(),
        ));
    }
    Ok(())
}

/// Five genuine phases through the same original GdnPrepared operator chain.
/// QKVZ/BA are observed leaf inputs; no embedding/RMS/projection qualification.
/// Each cache starts zero once and every continuation consumes its computed
/// predecessor. All expected bytes are used only after actual arithmetic.
pub async fn qualify_gdn(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [GdnFixture; 5],
    device_cap: u64,
) -> Result<GdnComparison> {
    match qualify_gdn_inner(device, source, fixtures, device_cap, Mode::Compare, None).await? {
        Outcome::Compared(compared) => Ok(compared),
        Outcome::Fault(_) => Err(Error::State("GDN comparison returned fault outcome".into())),
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Compare,
    Fault(GdnFault),
}
enum Outcome {
    Compared(GdnComparison),
    Fault(GdnFaultOutcome),
}

async fn qualify_gdn_inner(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [GdnFixture; 5],
    device_cap: u64,
    mode: Mode,
    cancelled: Option<&AtomicBool>,
) -> Result<Outcome> {
    // FIRST inner operation: refusal cannot reserve host/GPU resources.
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("GDN qualification pre-cancelled".into()));
    }
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for GDN leaf".into(),
        ));
    }
    let host = source.reserve_workspace(HOST_BYTES)?;
    validate_fixtures(&fixtures)?;
    let (windows, parameters) = load_parameters(&source)?;
    let source = Arc::new(source);
    let owner = Arc::new(GdnOwner {
        _source: source.clone(),
        _parameters: windows,
        _host: host,
        parameters,
        fixtures,
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("qwen08-cuda-gdn-leaf", "observed-projection-boundary")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    if mode == Mode::Fault(GdnFault::AdmissionPressure) {
        let witness = faults::admission_pressure(&source, &owner, &service)?;
        drop((owner, source, service));
        return faults::pressure_retired(witness).map(Outcome::Fault);
    }
    let request = [AllocationRequest {
        kind: AllocationClass::Other,
        bytes: PROVIDER_CAP as u64,
    }];
    let ticket =
        service
            .queue()
            .submit(service.budget(), service.owner().clone(), &request, None)?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(400)?;
    init.retain_completion_owner(owner.clone())?;
    let parameters = owner
        .parameters
        .iter()
        .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
        .collect::<Result<Vec<_>>>()?;
    let zero_cache = vec![0; GdnBuffer::CacheArena.bytes(GdnPhase::InitialOne)];
    let cache_one = init.upload_bytes(&zero_cache, BufferRole::RecurrentState)?;
    drop(init.complete_sync().await?);
    let mut init = service.begin_initialization(401)?;
    init.retain_completion_owner(owner.clone())?;
    let cache_four = init.upload_bytes(&zero_cache, BufferRole::RecurrentState)?;
    if cache_one.id() == cache_four.id() {
        return Err(Error::State("independent GDN leaf caches alias".into()));
    }
    let mut inputs = Vec::new();
    for (fixture, phase) in owner.fixtures.iter().zip(PHASES) {
        inputs.push((
            init.upload_bytes(&fixture.qkvz, BufferRole::Activation)?,
            init.upload_bytes(&fixture.ba, BufferRole::Activation)?,
            if phase == GdnPhase::CachedDecode {
                Some(init.allocate_empty(16384, BufferRole::Output)?)
            } else {
                None
            },
        ));
    }
    drop(init.complete_sync().await?);
    drop(zero_cache);
    let mut calls = Vec::new();
    for (index, ((qkvz, ba, snapshot), phase)) in inputs.into_iter().zip(PHASES).enumerate() {
        let cache = if index < 2 {
            cache_one.clone()
        } else {
            cache_four.clone()
        };
        let cache_slot = if index < 2 {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let snapshot = snapshot
            .map(|lease| Snapshot::prepare(&service, &qkvz, lease))
            .transpose()?;
        let gdn = GdnPrepared::prepare(
            service.clone(),
            phase,
            cache_slot,
            GdnInputs {
                qkvz,
                ba,
                cache,
                conv_weights: parameters[0].clone(),
                a_log: parameters[1].clone(),
                dt_bias: parameters[2].clone(),
                gamma: parameters[3].clone(),
            },
        )
        .await?;
        let mut accesses = gdn.accesses();
        if let Some(snapshot) = &snapshot {
            accesses.push((snapshot.lease.clone(), DeviceAccess::Write));
        }
        let outputs = accesses
            .iter()
            .filter(|(_, access)| *access == DeviceAccess::Write)
            .map(|(lease, _)| lease.clone())
            .collect();
        let mut call = Call {
            gdn,
            phase,
            slot: cache_slot,
            snapshot,
            accesses,
            outputs,
            reads: Vec::new(),
        };
        if mode == Mode::Compare {
            for parameter in &parameters {
                call.reads.push(parameter.readback(0, parameter.bytes())?);
            }
            for record in &owner.fixtures[index].expected {
                call.reads.push(call.readback(record.output)?);
            }
        }
        calls.push(Arc::new(call));
    }
    slot.finish()?;
    let mut compared = GdnComparison::default();
    let mut index = 0;
    while let Some(call) = calls.get(index).cloned() {
        let abandoned = if mode == Mode::Fault(GdnFault::AbandonAfterLaunch) && index == 0 {
            Some(faults::UnknownWitness::new(
                &source, &owner, &call, &service,
            )?)
        } else {
            None
        };
        let ticket =
            service
                .queue()
                .submit(service.budget(), service.owner().clone(), &request, None)?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(500 + index as u64, &call.accesses)?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(call.clone())?;
        flight.retain_completion_owner(call.gdn.clone())?;
        if let Some(snapshot) = &call.snapshot {
            snapshot.launch(&service)?;
        }
        let launches = call.gdn.launch(None)?;
        if let Some(witness) = abandoned {
            // No readback, synchronization or mathematical Ready. Original
            // armed drops quarantine the actual submitted owners/handles.
            drop(flight);
            drop(slot);
            drop((
                call, calls, cache_one, cache_four, parameters, owner, source, service,
            ));
            return witness.inspect(launches).map(Outcome::Fault);
        }
        let (ready, raw) = flight
            .complete_with_readback(&call.outputs, &call.reads, |actual: &[Bytes]| {
                for (parameter, raw) in owner.parameters.iter().zip(&actual[..4]) {
                    if &raw[..] != parameter.as_slice() {
                        return Err(Error::State(
                            "GDN leaf modified original source parameter".into(),
                        ));
                    }
                }
                for (record, actual) in owner.fixtures[index].expected.iter().zip(&actual[4..]) {
                    if let Some(first) =
                        record
                            .output
                            .first_difference(call.phase, actual, &record.bytes)?
                    {
                        return Err(Error::State(format!(
                            "GDN leaf call{index} {:?} differs at word{first}",
                            record.output
                        )));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.gdn_launches += launches;
        for record in &owner.fixtures[index].expected {
            if record.output.word_bytes() == 4 {
                compared.f32_words += record.bytes.len() / 4;
            } else {
                compared.bf16_words += record.bytes.len() / 2;
            }
        }
        drop((ready, raw));
        if service.pending_events()? != 0 {
            return Err(Error::State(
                "GDN leaf completed event pins retained".into(),
            ));
        }
        index += 1;
    }
    drop((calls, cache_one, cache_four, parameters, owner, source));
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "GDN leaf source/cache/device buffers not retired".into(),
        ));
    }
    drop(service);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "GDN leaf original physical envelope not retired".into(),
        ));
    }
    Ok(Outcome::Compared(compared))
}
