//! Closed original-source layer0 chain; no intermediate operand override.
use super::blas::{operation, source_weights};
use super::layer_common::{Snapshot, finite_f32, load_parameters};
use super::*;
use crate::resident_leases::{ForeignProviderCharge, ResidentReadback};
use crate::vllm_cuda_gdn::{GdnCacheSlot, GdnInputs, GdnPrepared};
use crate::vllm_cuda_input_rms::RmsPrepared;
use cubecl::cuda::{Bf16GemmBindings, OwnedBf16BlasProvider};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use zerogravity_artifacts::source_weights::TensorWindow;
use zerogravity_core::device_events::BufferLeaseId;

const FOREIGN_BYTES: usize = 64 * 1024 * 1024;
const PHASES: [VllmGdnPhase; 5] = [
    VllmGdnPhase::InitialOne,
    VllmGdnPhase::CachedDecode,
    VllmGdnPhase::Initial64,
    VllmGdnPhase::ContinuedTwo,
    VllmGdnPhase::CachedDecode,
];
const INITIAL64_IDS: [u32; 64] = [
    6269, 220, 15, 15, 15, 25, 357, 22486, 10695, 449, 75124, 908, 25659, 3294, 279, 8964, 13, 561,
    22839, 7189, 2234, 11, 14984, 11, 9039, 321, 9849, 1324, 220, 16, 15, 15, 15, 13, 357, 28336,
    13205, 1754, 1622, 579, 3555, 1528, 9209, 1345, 491, 944, 28712, 13, 79091, 864, 20934, 513,
    4221, 1518, 9722, 279, 9412, 13, 198, 8556, 5887, 62, 15, 19490,
];

#[derive(Clone, Copy, Debug, Default)]
pub struct LayerComparison {
    pub calls: usize,
    pub producer_flights: usize,
    pub gemm_calls: usize,
    pub rms_stages: usize,
    pub gdn_launches: usize,
    pub bf16_words: usize,
    pub f32_words: usize,
}
struct LayerOwner {
    _source: VerifiedTensorSource,
    _windows: Vec<Arc<TensorWindow>>,
    _parameters: Vec<Arc<ParameterWindow>>,
    weights: [Vec<u8>; 3],
    parameters: [Vec<u8>; 4],
    fixtures: [LayerFixture; 5],
    zero_cache: Vec<u8>,
    _host: [WindowWorkspace; 3],
}
struct Call {
    rms: Arc<RmsPrepared>,
    gdn: Arc<GdnPrepared>,
    qkvz: ResidentLease,
    ba: ResidentLease,
    projected: ResidentLease,
    cache: ResidentLease,
    slot: GdnCacheSlot,
    snapshot: Option<Snapshot>,
    phase: VllmGdnPhase,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    outputs: Vec<ResidentLease>,
    reads: Vec<ResidentReadback>,
}
fn native_error(error: impl std::fmt::Display) -> Error {
    Error::State(format!("connected original cuBLAS: {error}"))
}
fn validate_fixtures(fixtures: &[LayerFixture; 5]) -> Result<()> {
    let ids: [&[u32]; 5] = [&[11], &[353], &INITIAL64_IDS, &[25, 43112], &[353]];
    for ((fixture, phase), ids) in fixtures.iter().zip(PHASES).zip(ids) {
        if fixture.teacher_ids != ids {
            return Err(Error::Malformed(
                "closed actual layer teacher IDs/phase".into(),
            ));
        }
        let required = LayerOutput::required(phase)
            .into_iter()
            .collect::<BTreeSet<_>>();
        let mut actual = BTreeSet::new();
        for record in &fixture.expected {
            let finite = if record.output.word_bytes() == 4 {
                finite_f32(&record.bytes)
            } else {
                finite_bf16(&record.bytes)
            };
            if !actual.insert(record.output)
                || record.bytes.len() != record.output.bytes(phase)
                || !finite
            {
                return Err(Error::Malformed(
                    "complete finite unique layer output records".into(),
                ));
            }
        }
        if actual != required {
            return Err(Error::Malformed(
                "exact native prefill20/decode9 layer records".into(),
            ));
        }
    }
    Ok(())
}
fn insert_access(
    all: &mut BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)>,
    lease: ResidentLease,
    access: DeviceAccess,
) -> Result<()> {
    if let Some((prior, mode)) = all.get_mut(&lease.id()) {
        if !Arc::ptr_eq(prior.service(), lease.service()) || prior.bytes() != lease.bytes() {
            return Err(Error::State(
                "connected original lease ID changed owner/extent".into(),
            ));
        }
        if access == DeviceAccess::Write {
            *mode = DeviceAccess::Write;
        }
    } else {
        all.insert(lease.id(), (lease, access));
    }
    Ok(())
}
impl Call {
    fn prepare_accesses(
        &self,
        weights: &[ResidentLease],
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let mut all = BTreeMap::new();
        for (lease, access) in self
            .rms
            .accesses()
            .iter()
            .cloned()
            .chain(self.gdn.accesses())
        {
            insert_access(&mut all, lease, access)?;
        }
        for lease in weights {
            insert_access(&mut all, lease.clone(), DeviceAccess::Read)?;
        }
        for lease in [&self.qkvz, &self.ba, &self.projected, workspace] {
            insert_access(&mut all, lease.clone(), DeviceAccess::Write)?;
        }
        if let Some(snapshot) = &self.snapshot {
            insert_access(&mut all, snapshot.lease.clone(), DeviceAccess::Write)?;
        }
        Ok(all.into_values().collect())
    }
    fn readback(&self, output: LayerOutput) -> Result<ResidentReadback> {
        let bytes = output.bytes(self.phase);
        let (lease, offset) = match output {
            LayerOutput::Normalized => (self.rms.normalized(), 0),
            LayerOutput::Qkvz => (
                self.snapshot
                    .as_ref()
                    .map(|copy| &copy.lease)
                    .unwrap_or(&self.qkvz),
                0,
            ),
            LayerOutput::Convolution => (
                if self.phase == VllmGdnPhase::CachedDecode {
                    &self.qkvz
                } else {
                    self.gdn.lease(GdnBuffer::ConvOutput)?
                },
                0,
            ),
            LayerOutput::Projected => (&self.projected, 0),
            LayerOutput::ConvolutionState => (&self.cache, self.slot.convolution_offset()),
            LayerOutput::RecurrentState => (&self.cache, self.slot.recurrent_offset()),
            _ => (
                self.gdn.lease(
                    output
                        .role()
                        .ok_or_else(|| Error::State("closed layer output role".into()))?,
                )?,
                0,
            ),
        };
        lease.readback(offset, bytes)
    }
}

/// The five authentic calls execute computed embedding→RMS→cuBLAS→GDN→cuBLAS.
/// Every source/buffer/spec is prepared before the one-shot native pool starts.
/// One original producer flight per call; expected bytes exist only in its
/// completion validator. This proves one trained layer, never a model route.
pub async fn qualify_layer(
    device: Device,
    source: VerifiedTensorSource,
    fixtures: [LayerFixture; 5],
    cublas: &Path,
    cublas_lt: &Path,
    device_cap: u64,
) -> Result<LayerComparison> {
    if cfg!(debug_assertions) || !matches!(device, Device::Cuda(_)) {
        return Err(Error::Unsupported(
            "release original CUDA required for connected layer".into(),
        ));
    }
    validate_fixtures(&fixtures)?;
    // Separate same-source envelopes: concatenated weights, H2D staging,
    // completion readbacks. Each original workspace stays within its 32MiB cap.
    let host = [
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
        source.reserve_workspace(32 * 1024 * 1024)?,
    ];
    let (windows, weights) = source_weights(&source)?;
    let (parameters, parameter_values) = load_parameters(&source)?;
    let owner = Arc::new(LayerOwner {
        _source: source.clone(),
        _windows: windows,
        _parameters: parameters,
        _host: host,
        weights,
        parameters: parameter_values,
        fixtures,
        zero_cache: vec![0; GdnBuffer::CacheArena.bytes(VllmGdnPhase::InitialOne)],
    });
    let service = ResidentService::new_pinned_cuda(
        device,
        source.source_identity(),
        ResourceOwner::new("vllm08-cuda-layer", "connected-original-layer0")?,
        PROVIDER_CAP,
        device_cap,
    )?;
    let key = service.physical_key();
    let queue = service.pinned_cuda_queue()?.clone();
    let sources = queue
        .verify_bf16_blas_sources(cublas, cublas_lt)
        .map_err(native_error)?;
    let charge: Arc<ForeignProviderCharge> = service.reserve_foreign_provider(FOREIGN_BYTES)?;
    if charge.bytes() != FOREIGN_BYTES {
        return Err(Error::State(
            "connected exact original opaque charge".into(),
        ));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: (PROVIDER_CAP - FOREIGN_BYTES) as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut init = service.begin_initialization(100)?;
    init.retain_completion_owner(owner.clone())?;
    let weights = owner
        .weights
        .iter()
        .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
        .collect::<Result<Vec<_>>>()?;
    let parameters = owner
        .parameters
        .iter()
        .map(|raw| init.upload_bytes(raw, BufferRole::ImmutableWeights))
        .collect::<Result<Vec<_>>>()?;
    let cache_one = init.upload_bytes(&owner.zero_cache, BufferRole::RecurrentState)?;
    let cache_four = init.upload_bytes(&owner.zero_cache, BufferRole::RecurrentState)?;
    if cache_one.id() == cache_four.id() {
        return Err(Error::State(
            "independent authentic sequence caches alias".into(),
        ));
    }
    let workspace = init.allocate_empty(8_519_680, BufferRole::Scratch)?;
    let status = init.upload_bytes(&0x13579bdfu32.to_le_bytes(), BufferRole::ImmutableWeights)?;
    let mut ordinary = Vec::new();
    for phase in PHASES {
        ordinary.push((
            init.allocate_empty(
                operation(phase.tokens(), 0).output_bytes(),
                BufferRole::Output,
            )?,
            init.allocate_empty(
                operation(phase.tokens(), 1).output_bytes(),
                BufferRole::Output,
            )?,
            init.allocate_empty(
                operation(phase.tokens(), 2).output_bytes(),
                BufferRole::Output,
            )?,
            if phase == VllmGdnPhase::CachedDecode {
                Some(init.allocate_empty(16384, BufferRole::Output)?)
            } else {
                None
            },
        ));
    }
    drop(init.complete_sync().await?);
    let mut calls = Vec::new();
    for (index, (phase, (qkvz, ba, projected, snapshot))) in
        PHASES.into_iter().zip(ordinary).enumerate()
    {
        let cache_slot = if index < 2 {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        };
        let cache = if index < 2 {
            cache_one.clone()
        } else {
            cache_four.clone()
        };
        let rms = RmsPrepared::prepare(
            service.clone(),
            source.clone(),
            &owner.fixtures[index].teacher_ids,
            101 + index as u64,
        )
        .await?;
        let gdn = GdnPrepared::prepare(
            service.clone(),
            phase,
            cache_slot,
            GdnInputs {
                qkvz: qkvz.clone(),
                ba: ba.clone(),
                cache: cache.clone(),
                conv_weights: parameters[0].clone(),
                a_log: parameters[1].clone(),
                dt_bias: parameters[2].clone(),
                gamma: parameters[3].clone(),
            },
        )
        .await?;
        let snapshot = snapshot
            .map(|lease| Snapshot::prepare(&service, &qkvz, lease))
            .transpose()?;
        let mut call = Call {
            rms,
            gdn,
            qkvz,
            ba,
            projected,
            cache,
            slot: cache_slot,
            snapshot,
            phase,
            accesses: Vec::new(),
            outputs: Vec::new(),
            reads: Vec::new(),
        };
        call.accesses = call.prepare_accesses(&weights, &workspace)?;
        call.outputs = call
            .accesses
            .iter()
            .filter(|(_, access)| *access == DeviceAccess::Write)
            .map(|(lease, _)| lease.clone())
            .collect();
        call.reads = call.rms.readbacks()?.into_iter().collect();
        for record in &owner.fixtures[index].expected {
            let read = call.readback(record.output)?;
            call.reads.push(read);
        }
        calls.push(Arc::new(call));
    }
    slot.finish()?;
    let pool = unsafe { queue.activate_blas_allocation_pool(FOREIGN_BYTES as u64, charge.clone()) }
        .map_err(native_error)?;
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(200, &[(status.clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(owner.clone())?;
    flight.retain_completion_owner(pool.clone())?;
    flight.retain_completion_owner(charge.clone())?;
    let provider: Arc<OwnedBf16BlasProvider> =
        unsafe { queue.create_bf16_blas_provider(sources.clone(), pool.clone()) }.map_err(
            |error| {
                Error::State(format!(
                    "connected cuBLAS creation: {error}; pool={:?}",
                    pool.snapshot()
                ))
            },
        )?;
    flight.retain_completion_owner(provider.clone())?;
    flight
        .complete_read_only_readback(&[status.readback(0, 4)?], |_| Ok(()))
        .await?;
    slot.finish()?;
    let mut compared = LayerComparison::default();
    for (index, call) in calls.iter().enumerate() {
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Other,
                bytes: (PROVIDER_CAP - FOREIGN_BYTES) as u64,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        let mut flight = service.submit(300 + index as u64, &call.accesses)?;
        flight.retain_completion_owner(owner.clone())?;
        flight.retain_completion_owner(call.clone())?;
        flight.retain_completion_owner(call.rms.completion_owner())?;
        flight.retain_completion_owner(call.gdn.clone())?;
        flight.retain_completion_owner(provider.clone())?;
        flight.retain_completion_owner(pool.clone())?;
        flight.retain_completion_owner(charge.clone())?;
        call.rms.enqueue()?;
        for (projection, output) in [(0, &call.qkvz), (1, &call.ba)] {
            unsafe {
                provider.enqueue(
                    operation(call.phase.tokens(), projection),
                    Bf16GemmBindings {
                        weights: weights[projection].handle(),
                        input: call.rms.normalized().handle(),
                        output: output.handle(),
                        workspace: workspace.handle(),
                    },
                )
            }
            .map_err(native_error)?;
        }
        if let Some(snapshot) = &call.snapshot {
            snapshot.launch(&service)?;
        }
        let launches = call.gdn.launch(None)?;
        unsafe {
            provider.enqueue(
                operation(call.phase.tokens(), 2),
                Bf16GemmBindings {
                    weights: weights[2].handle(),
                    input: call.gdn.lease(GdnBuffer::Gated)?.handle(),
                    output: call.projected.handle(),
                    workspace: workspace.handle(),
                },
            )
        }
        .map_err(native_error)?;
        let (ready, raw) = flight
            .complete_with_readback(&call.outputs, &call.reads, |raw: &[Bytes]| {
                call.rms.validate_readback(&raw[..3])?;
                for (record, actual) in owner.fixtures[index].expected.iter().zip(&raw[3..]) {
                    if let Some(first) =
                        record
                            .output
                            .first_difference(call.phase, actual, &record.bytes)?
                    {
                        return Err(Error::State(format!(
                            "connected call{index} {:?} differs at word{first}",
                            record.output
                        )));
                    }
                }
                Ok(())
            })
            .await?;
        slot.finish()?;
        compared.calls += 1;
        compared.producer_flights += 1;
        compared.gemm_calls += 3;
        compared.rms_stages += 10;
        compared.gdn_launches += launches;
        for record in &owner.fixtures[index].expected {
            if record.output.word_bytes() == 4 {
                compared.f32_words += record.bytes.len() / 4;
            } else {
                compared.bf16_words += record.bytes.len() / 2;
            }
        }
        drop((ready, raw));
    }
    let ticket = service.queue().submit(
        service.budget(),
        service.owner().clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: FOREIGN_BYTES as u64,
        }],
        None,
    )?;
    let mut slot = ticket.wait().await?;
    slot.begin_submission()?;
    let mut flight = service.submit(999, &[(status.clone(), DeviceAccess::Read)])?;
    flight.retain_completion_owner(provider.clone())?;
    flight.retain_completion_owner(pool.clone())?;
    flight.retain_completion_owner(charge.clone())?;
    unsafe { provider.destroy_after_physical_completion() }.map_err(native_error)?;
    flight
        .complete_read_only_readback(&[status.readback(0, 4)?], |_| Ok(()))
        .await?;
    slot.finish()?;
    let retired = unsafe { pool.retire_after_physical_completion() }.map_err(native_error)?;
    if !retired.retired
        || retired.provider_live
        || retired.charged_bytes != 0
        || retired.live_allocations != 0
        || retired.inflight_allocations != 0
    {
        return Err(Error::State(
            "connected actual provider pool retirement incomplete".into(),
        ));
    }
    drop((
        calls, cache_one, cache_four, weights, parameters, workspace, status, provider, pool,
        sources, queue, charge, owner,
    ));
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "connected original buffers/events not retired".into(),
        ));
    }
    drop(service);
    if ResourceArbiter::process_shared().snapshot(&key)?.reserved != 0 {
        return Err(Error::State(
            "connected original physical envelope did not retire".into(),
        ));
    }
    Ok(compared)
}
