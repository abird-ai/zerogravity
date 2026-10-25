//! Opt-in physical source/one-state/completion probe; no learned-model permit.
#![allow(dead_code)]
use super::*;
#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
#[path = "portable_qualification/shared_physical.rs"]
mod shared_physical;
#[cfg(feature = "vulkan")]
use cubecl::device::WgpuBackend;
#[cfg(feature = "hip")]
use cubecl::ir::amd::GfxArch;
use sha2::{Digest, Sha256};
use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
    task::{Context, Poll, Waker},
};
use zerogravity_artifacts::quant::{QuantKind, dequantize};
use zerogravity_core::{
    accelerator::{
        SourceIdentity,
        program::{
            ConvDomain, MathematicalBackend, MatrixAxes, ParameterRef, PlanBuilder, SourceTensor,
            SourceVector,
        },
    },
    resource_arbiter::{GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter},
    source::SourceTensorRef,
};

const WIDTH: usize = 32;
const VOCAB: usize = 32;
const HOST: u64 = 16 * 1024 * 1024 * 1024;
const SOURCE_HOST: u64 = 1024 * 1024;
const PROVIDER: usize = 64 * 1024 * 1024;
const DEVICE: u64 = 64 * 1024 * 1024 * 1024;
const PCI: &str = "0000:c6:00.0";

struct Source {
    identity: SourceIdentity,
    shape: [u64; 2],
    packed: Vec<u8>,
    weights: Vec<f32>,
    sha256: String,
    reads: AtomicUsize,
    host_key: PhysicalDeviceKey,
    // Fixture/reference allocations drop before their source admission.
    _host: GlobalResourceLease,
}
impl Source {
    fn new() -> Result<Arc<Self>> {
        Self::new_with_salt(0)
    }
    fn new_with_salt(salt: u8) -> Result<Arc<Self>> {
        let identity = SourceIdentity::for_new_allocation()?;
        let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
        let arbiter = ResourceArbiter::process_shared();
        arbiter.register_matching(host_key.clone(), HOST)?;
        let host = arbiter.admit(
            &host_key,
            ResourceOwner::new("synthetic-q8-conv", &format!("fixture-{identity:?}"))?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: SOURCE_HOST,
            }],
        )?;
        let mut packed = Vec::with_capacity(VOCAB * 34);
        for row in 0..VOCAB {
            packed.extend_from_slice(&0x3c00u16.to_le_bytes()); // Q8_0 scale1.
            for column in 0..WIDTH {
                packed.push((((row * 13 + column * 7 + usize::from(salt)) % 29) as i8 - 14) as u8);
            }
        }
        let weights = (0..WIDTH)
            .flat_map(|_| [0.125, 0.25, 0.5, 1.0])
            .collect::<Vec<_>>();
        let mut digest = Sha256::new();
        digest.update(&packed);
        for word in &weights {
            digest.update(f32::to_le_bytes(*word));
        }
        Ok(Arc::new(Self {
            identity,
            shape: [WIDTH as u64, VOCAB as u64],
            packed,
            weights,
            sha256: format!("{:x}", digest.finalize()),
            reads: AtomicUsize::new(0),
            host_key,
            _host: host,
        }))
    }
    fn plan(&self) -> Result<ForwardPlan> {
        let domain = ConvDomain(0);
        let mut builder =
            PlanBuilder::new(self.identity, 256, vec![StateSlot::Conv(domain)], true)?;
        let input = builder.embed(
            SourceTensor::from_source(self, "embedding", MatrixAxes::InputOutput)?,
            ParameterRef(ParameterSlot::Token),
        )?;
        builder.observe(0, "portable.embedding", &input)?;
        let output = builder.conv_step(
            "portable.conv",
            domain,
            &input,
            SourceVector {
                source: self.identity,
                name: "conv.weight",
                values: &self.weights,
            },
        )?;
        builder.observe(0, "portable.convolution", &output)?;
        builder.observe_state(0, "portable.history", StateSlot::Conv(domain))?;
        builder.advance_frontier()?;
        builder.finish(Some(output))
    }
}
impl TensorSource for Source {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }
    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        if name != "embedding" {
            return Err(Error::Malformed("portable source role".into()));
        }
        Ok(SourceTensorRef {
            kind: QuantKind::Q8_0,
            shape: &self.shape,
            packed_bytes: &self.packed,
        })
    }
}

#[derive(Clone)]
struct Oracle {
    source: Arc<Source>,
    embedding: Vec<f32>,
    history: Vec<f32>,
}
struct Expected {
    _source: Arc<Source>,
    embedding: Vec<f32>,
    output: Vec<f32>,
    history: Vec<f32>,
}
impl Oracle {
    fn new(source: Arc<Source>) -> Result<Self> {
        let embedding = dequantize(QuantKind::Q8_0, &source.packed, VOCAB * WIDTH)?;
        Ok(Self {
            source,
            embedding,
            history: vec![0.0; WIDTH * 3],
        })
    }
    fn advance(&mut self, tokens: &[u32]) -> Result<Expected> {
        if tokens.is_empty() || tokens.len() > 4 {
            return Err(Error::Limit("portable reference token bound".into()));
        }
        let mut embedding = vec![0.0; WIDTH];
        let mut output = vec![0.0; WIDTH];
        for &token in tokens {
            let token = token as usize;
            if token >= VOCAB {
                return Err(Error::Malformed("portable source token".into()));
            }
            embedding.copy_from_slice(&self.embedding[token * WIDTH..(token + 1) * WIDTH]);
            zerogravity_model_qwen35::source_gdn_conv_step(
                &embedding,
                &self.source.weights,
                &mut self.history,
                &mut output,
            )?;
        }
        Ok(Expected {
            _source: self.source.clone(),
            embedding,
            output,
            history: self.history.clone(),
        })
    }
}
fn exact_words(expected: &[f32], actual: &[f32], label: &str) -> Result<()> {
    if expected.len() != actual.len() {
        return Err(Error::State(format!("portable {label} extent")));
    }
    if let Some(at) = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(Error::State(format!(
            "portable {label} word{at}: expected={:08x} actual={:08x}",
            expected[at].to_bits(),
            actual[at].to_bits()
        )));
    }
    Ok(())
}
impl Expected {
    fn compare(&self, read: &InferenceReadback, position: usize) -> Result<()> {
        exact_words(&self.output, &read.logits, "output")?;
        if read.domains.len() != 1 || read.domains[0].0 != StateSlot::Conv(ConvDomain(0)) {
            return Err(Error::State("portable complete Conv domain".into()));
        }
        exact_words(&self.history, &read.domains[0].1, "history")?;
        if read.observations.len() != 3 || read.dispatches == 0 || read.bytes == 0 {
            return Err(Error::State(
                "portable actual observations/dispatch/readback".into(),
            ));
        }
        for (actual, (stage, expected)) in read.observations.iter().zip([
            ("portable.embedding", self.embedding.as_slice()),
            ("portable.convolution", self.output.as_slice()),
            ("portable.history", self.history.as_slice()),
        ]) {
            if actual.layer != 0 || actual.stage != stage || actual.position != position {
                return Err(Error::State("portable actual producer order".into()));
            }
            exact_words(expected, &actual.values, stage)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
enum Backend {
    #[cfg(feature = "hip")]
    Hip,
    #[cfg(feature = "vulkan")]
    Vulkan,
}
impl Backend {
    fn device(self) -> Device {
        #[cfg(feature = "hip")]
        use cubecl::device::AmdDevice;
        #[cfg(feature = "vulkan")]
        use cubecl::device::{WgpuDevice, WgpuDeviceKind};
        match self {
            #[cfg(feature = "hip")]
            Self::Hip => Device::Hip(AmdDevice::new(0)),
            #[cfg(feature = "vulkan")]
            Self::Vulkan => Device::Wgpu(
                WgpuDevice::new(WgpuDeviceKind::IntegratedGpu(0)).on(WgpuBackend::Vulkan),
            ),
        }
    }
    fn validate(self, device: &Device) -> Result<()> {
        crate::backend::validate_provider(device)?;
        // Metadata query only; reject another card/architecture before fixtures
        // create managed buffers or enqueue the original engine's kernels.
        let client = device.client();
        let properties = client.properties();
        let pci = properties
            .identity
            .physical
            .as_ref()
            .and_then(|p| p.pci_address);
        if pci.map(|p| p.to_string()).as_deref() != Some(PCI)
            || properties.memory.max_memory().is_none_or(|n| n < DEVICE)
        {
            return Err(Error::Unsupported(
                "portable probe requires attested pvl-x2 PCI/capacity".into(),
            ));
        }
        match self {
            #[cfg(feature = "hip")]
            Self::Hip => {
                let name = properties
                    .identity
                    .fingerprint
                    .strip_prefix("hip-kernel_")
                    .ok_or_else(|| {
                        Error::Unsupported("portable probe HIP compiler identity".into())
                    })?;
                let gfx = GfxArch::parse(name);
                if !matches!(device, Device::Hip(_))
                    || gfx.name() != "gfx1151"
                    || properties.hardware.plane_size_min != 32
                    || properties.hardware.plane_size_max != 32
                {
                    return Err(Error::Unsupported(
                        "portable probe requires actual gfx1151/wave32 HIP".into(),
                    ));
                }
            }
            #[cfg(feature = "vulkan")]
            Self::Vulkan => {
                if !matches!(device, Device::Wgpu(d) if d.backend == WgpuBackend::Vulkan)
                    || properties.hardware.plane_size_min < 32
                    || properties.hardware.plane_size_max > 64
                    || properties.hardware.plane_size_min > properties.hardware.plane_size_max
                {
                    return Err(Error::Unsupported(
                        "portable probe explicit Vulkan/subgroup metadata".into(),
                    ));
                }
            }
        }
        ResidentProgram::ensure_capability(&client)?;
        eprintln!(
            "PORTABLE_DEVICE backend={self:?} pci={PCI} fingerprint={} plane_min={} plane_max={} reported={} provider={PROVIDER} operator_cap={DEVICE}",
            properties.identity.fingerprint,
            properties.hardware.plane_size_min,
            properties.hardware.plane_size_max,
            properties.memory.max_memory().unwrap_or(0)
        );
        Ok(())
    }
}

async fn load(backend: Backend) -> Result<(Arc<Source>, Arc<ResidentModel>)> {
    let device = backend.device();
    backend.validate(&device)?;
    let source = Source::new()?;
    let plan = source.plan()?;
    let (_, _, preflight) = preflight_profile(source.as_ref(), &plan)?;
    // Original preflight owns the source/value/workspace/status allocation;
    // eight state sets conservatively cover the normal/fault cohort (actual
    // normal maximum4), including uncommitted and retained snapshot owners.
    let greedy = crate::resident_greedy::workspace_bytes(VOCAB)?
        .iter()
        .sum::<usize>();
    let owned = preflight.owned_bytes()?;
    let required = preflight
        .prior_state_bytes
        .checked_mul(8)
        .and_then(|n| n.checked_add(owned))
        .and_then(|n| n.checked_add(greedy))
        .ok_or_else(|| Error::Limit("portable complete managed bound".into()))?;
    if required > PROVIDER {
        return Err(Error::Limit("portable derived model envelope".into()));
    }
    let model = ResidentModel::load(device, source.clone(), plan, PROVIDER, DEVICE).await?;
    if model.service.physical_key() != PhysicalDeviceKey::new("pci", PCI)?
        || model.service.source_identity() != source.identity
        || model.service.pending_events()? != 0
    {
        return Err(Error::State(
            "portable original source/service/device join".into(),
        ));
    }
    eprintln!(
        "PORTABLE_SOURCE synthetic=true sha256={} packed={} conv={} derived_preflight={required} resident={}",
        source.sha256,
        source.packed.len(),
        source.weights.len() * 4,
        model.service.resident_bytes()?
    );
    Ok((source, model))
}

async fn checked_prepare(
    session: &ResidentSession,
    oracle: &Oracle,
    tokens: &[u32],
) -> Result<(PreparedInference, Oracle)> {
    let before = session.binding();
    let prepared = session.prepare(tokens, true).await?;
    if session.binding() != before {
        return Err(Error::State(
            "portable private Ready published early".into(),
        ));
    }
    // Independent source primitive runs only after the actual proposal exists;
    // these values never become GPU inputs or alter the source traversal.
    let mut next = oracle.clone();
    next.advance(tokens)?.compare(
        &prepared.readback,
        before.frontier() as usize + tokens.len() - 1,
    )?;
    if prepared.ready.event_ids().len() != 1 || prepared.ready.outputs().len() != 1 {
        return Err(Error::State(
            "portable complete original Ready cohort".into(),
        ));
    }
    Ok((prepared, next))
}

fn queue_cancel(model: &ResidentModel) -> Result<()> {
    let queue = model.service.queue();
    let before = queue.stats()?;
    let ticket = || {
        queue.submit(
            model.service.budget(),
            model.service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 4,
            }],
            None,
        )
    };
    let mut active = Box::pin(ticket()?.wait());
    let mut cx = Context::from_waker(Waker::noop());
    let acquired = match active.as_mut().poll(&mut cx) {
        Poll::Ready(Ok(slot)) => slot,
        _ => return Err(Error::State("portable original fair queue acquire".into())),
    };
    drop(active);
    let waiting = ticket()?;
    drop(waiting); // Genuine queued request, never begun/submitted.
    drop(acquired); // Acquired but unsubmitted, no fake finish/begin.
    let after = queue.stats()?;
    if after.active
        || after.queued != 0
        || after.completed != before.completed
        || after.cancelled_queued != before.cancelled_queued + 1
    {
        return Err(Error::State("portable queued/acquired cancellation".into()));
    }
    model.service.ensure_usable()
}

async fn healthy(backend: Backend) -> Result<()> {
    let (source, model) = load(backend).await?;
    let host_key = source.host_key.clone();
    let device_key = model.service.physical_key();
    let mut a = model.session().await?;
    let mut b = model.session().await?;
    let mut reference_a = Oracle::new(source.clone())?;
    let mut reference_b = Oracle::new(source.clone())?;
    let b_initial = b.binding();
    let mut dispatches = 0;
    for tokens in [&[1, 7][..], &[11][..], &[17][..]] {
        let (ready, next) = checked_prepare(&a, &reference_a, tokens).await?;
        dispatches += ready.readback.dispatches;
        a.commit(ready)?;
        reference_a = next;
        if b.binding() != b_initial {
            return Err(Error::State("portable sibling root changed".into()));
        }
    }
    let (ready, next) = checked_prepare(&b, &reference_b, &[3]).await?;
    b.commit(ready)?;
    reference_b = next;
    exact_words(
        &reference_b.history,
        &b.snapshot().read_domains().await?[0].1,
        "sibling history",
    )?;
    queue_cancel(&model)?;
    let before = a.binding();
    let before_state = a.snapshot().read_domains().await?;
    let bytes = model.service.resident_bytes()?;
    let pending = model.service.pending_events()?;
    if model
        .service
        .allocate_uninitialized(model.service.remaining_capacity()? + 1, BufferRole::Scratch)
        .is_ok()
        || model.service.resident_bytes()? != bytes
        || model.service.pending_events()? != pending
        || a.binding() != before
    {
        return Err(Error::State(
            "portable real cold allocation pressure".into(),
        ));
    }
    model.service.ensure_usable()?;
    let unpolled = a.prepare(&[5], true);
    drop(unpolled); // No poll, allocation, enqueue or cancellation claim beyond this.
    let (abandoned, _) = checked_prepare(&a, &reference_a, &[5]).await?;
    drop(abandoned); // Completed private Ready, no commit.
    if a.binding() != before {
        return Err(Error::State("portable known Ready abandonment".into()));
    }
    exact_words(
        &before_state[0].1,
        &a.snapshot().read_domains().await?[0].1,
        "abandoned prior",
    )?;
    let snapshot = a.snapshot();
    let (stale, _) = checked_prepare(&a, &reference_a, &[9]).await?;
    a.restore(&snapshot)?;
    let restored = a.binding();
    if a.commit(stale).is_ok() || a.binding() != restored {
        return Err(Error::State("portable stale Ready published".into()));
    }
    let (ready, next) = checked_prepare(&a, &reference_a, &[9]).await?;
    a.commit(ready)?;
    reference_a = next;
    if b.restore(&snapshot).is_ok() {
        return Err(Error::State("portable foreign snapshot accepted".into()));
    }
    let retained = a.snapshot();
    let model_weak = Arc::downgrade(&model);
    let service_weak = Arc::downgrade(&model.service);
    drop((snapshot, a, b, model));
    if model_weak.strong_count() != 0 || service_weak.strong_count() == 0 {
        return Err(Error::State(
            "portable completed snapshot owner/backedge".into(),
        ));
    }
    exact_words(
        &reference_a.history,
        &retained.read_domains().await?[0].1,
        "retained committed state",
    )?;
    let service = service_weak
        .upgrade()
        .ok_or_else(|| Error::State("portable original service vanished".into()))?;
    drop(retained);
    if service.resident_bytes()? != 0 || service.pending_events()? != 0 {
        return Err(Error::State(
            "portable final resident/event ownership".into(),
        ));
    }
    service.ensure_usable()?;
    drop(service);
    let budget = ResourceArbiter::process_shared().snapshot(&device_key)?;
    if budget.reserved != 0 || budget.fenced {
        return Err(Error::State("portable physical reservation final0".into()));
    }
    drop((before_state, reference_a, reference_b, source));
    if ResourceArbiter::process_shared()
        .snapshot(&host_key)?
        .reserved
        != 0
    {
        return Err(Error::State(
            "portable fixture/reference host final0".into(),
        ));
    }
    eprintln!(
        "PORTABLE_HEALTHY_PASS backend={backend:?} initial=2 cached=2 state_domains=1 history_words=96 outputs=32 actual_dispatches={dispatches} sibling=true stale_refused=true queued_cancel=true pressure=true model_dropped=true resident=0 pending=0 physical_reserved=0 host_reserved=0 learned=false perf=false"
    );
    Ok(())
}

#[cfg(feature = "hardware-abort-probe")]
async fn drop_after_cached_launch(
    model: &ResidentModel,
    session: &ResidentSession,
) -> Result<usize> {
    model.abort_after_launch.store(1, Ordering::Release);
    let mut future = Box::pin(session.prepare(&[11], true));
    // Drive the SAME original future through asynchronous initialization/queue
    // waits; a first Pending alone proves no launch. The owned supervisor is
    // the hard deadline; no busy loop or timeout-based resource release.
    let retained = std::future::poll_fn(|cx| match future.as_mut().poll(cx) {
        Poll::Ready(_) => Poll::Ready(Err(Error::State(
            "portable fault completed without the real launch pause".into(),
        ))),
        Poll::Pending => {
            let retained = model.launch_seen.load(Ordering::Acquire);
            if retained == 0 {
                Poll::Pending
            } else {
                Poll::Ready(Ok(retained))
            }
        }
    })
    .await?;
    drop(future); // Original armed flight/queue RAII owns the uncertainty.
    Ok(retained)
}

#[cfg(feature = "hardware-abort-probe")]
async fn unknown(backend: Backend) -> Result<()> {
    let (source, model) = load(backend).await?;
    let host_key = source.host_key.clone();
    let device_key = model.service.physical_key();
    let mut session = model.session().await?;
    let prior_reference = Oracle::new(source.clone())?;
    let (ready, reference) = checked_prepare(&session, &prior_reference, &[1, 7]).await?;
    drop(prior_reference);
    session.commit(ready)?;
    let before = session.binding();
    let service_weak = Arc::downgrade(&model.service);
    let source_weak = Arc::downgrade(&source);
    let quarantined_before =
        crate::physical_quarantine::quarantined_handles_for(model.service.client())?;
    let retained = drop_after_cached_launch(&model, &session).await?;
    if session.binding() != before
        || model.service.ensure_usable().is_ok()
        || model.service.pending_events()? == 0
        || crate::physical_quarantine::quarantined_handles_for(model.service.client())?
            < quarantined_before + retained
    {
        return Err(Error::State(
            "portable actual unknown completion escaped quarantine".into(),
        ));
    }
    drop((reference, session, source, model));
    // The original source admission accompanies the quarantined service until
    // actual retirement, even after its uploaded payload stops being an operand.
    if source_weak.strong_count() == 0 || service_weak.strong_count() == 0 {
        return Err(Error::State(
            "portable authentic source / GPU service retention".into(),
        ));
    }
    let service = service_weak
        .upgrade()
        .ok_or_else(|| Error::State("portable quarantined service vanished".into()))?;
    if crate::physical_quarantine::quarantined_handles_for(service.client())?
        < quarantined_before + retained
    {
        return Err(Error::State("portable handles dropped with caller".into()));
    }
    drop(service);
    let budget = ResourceArbiter::process_shared().snapshot(&device_key)?;
    if !budget.fenced
        || budget.reserved != PROVIDER as u64
        || ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
            != SOURCE_HOST
    {
        return Err(Error::State(
            "portable terminal physical/source-host retention".into(),
        ));
    }
    eprintln!(
        "PORTABLE_UNKNOWN_PASS backend={backend:?} genuine_cached_frontier={} actual_retained_handles={retained} ready_observed=false commit=false root_unchanged=true pending=true fenced=true provider_retained={PROVIDER} original_source_retained=true learned=false perf=false; process retirement required",
        before.frontier()
    );
    Ok(())
}

#[cfg(all(feature = "hip", feature = "vulkan", feature = "hardware-abort-probe"))]
#[derive(Clone, Copy, Debug)]
pub struct SamePciPeerReceipt {
    pub same_process: bool,
    pub same_pci: bool,
    pub same_arbiter: bool,
    pub contexts_pre_admitted: bool,
    pub submitted_fault: bool,
    pub opposite_api_refused: bool,
    pub original_source_retained: bool,
    pub original_native_retained: bool,
    pub original_handle_retained: bool,
    pub physical_fenced: bool,
    pub physical_limit: u64,
    pub physical_reserved: u64,
    pub direction: &'static str,
}

/// Finite same-process peer: both providers are admitted before the submitter
/// launches a real unknown, then peer work must refuse against the shared fence.
#[cfg(all(feature = "hip", feature = "vulkan", feature = "hardware-abort-probe"))]
pub fn same_pci_peer(direction: &str) -> Result<SamePciPeerReceipt> {
    let (submitter, peer) = match direction {
        "vulkan-then-hip" => (Backend::Vulkan, Backend::Hip),
        "hip-then-vulkan" => (Backend::Hip, Backend::Vulkan),
        _ => return Err(Error::Malformed("same-PCI peer direction".into())),
    };
    futures_lite::future::block_on(cross_provider_unknown(submitter, peer))
}

#[cfg(all(feature = "hip", feature = "vulkan", feature = "hardware-abort-probe"))]
async fn cross_provider_unknown(submitter: Backend, peer: Backend) -> Result<SamePciPeerReceipt> {
    let (source, model) = load(submitter).await?;
    let (peer_source, peer_model) = load(peer).await?;
    let physical = model.service.physical_key();
    let host_key = source.host_key.clone();
    let budget = ResourceArbiter::process_shared().snapshot(&physical)?;
    if physical != peer_model.service.physical_key()
        || !Arc::ptr_eq(model.service.queue(), peer_model.service.queue())
        || model.service.client().service_id() == peer_model.service.client().service_id()
        || budget.limit != DEVICE
        || budget.reserved != 2 * PROVIDER as u64
        || budget.reservations != 2
        || budget.fenced
    {
        return Err(Error::State(
            "portable actual cross-provider PCI/queue/admission".into(),
        ));
    }
    // This is a real third envelope request against the SAME physical cap;
    // refusal needs no managed GPU allocation or fabricated provider key.
    if crate::physical_service::PhysicalServiceBudget::attach(
        peer_model.service.client(),
        DEVICE,
        budget.remaining + 1,
        ResourceOwner::new("portable-cross-provider", "over-physical-cap")?,
    )
    .is_ok()
        || ResourceArbiter::process_shared()
            .snapshot(&physical)?
            .reserved
            != budget.reserved
    {
        return Err(Error::State(
            "portable shared physical cap was bypassed".into(),
        ));
    }
    let mut session = model.session().await?;
    let mut peer_session = peer_model.session().await?;
    let prior_reference = Oracle::new(source.clone())?;
    let prior_peer_reference = Oracle::new(peer_source.clone())?;
    let (ready, reference) = checked_prepare(&session, &prior_reference, &[1, 7]).await?;
    drop(prior_reference);
    session.commit(ready)?;
    let (ready, peer_reference) =
        checked_prepare(&peer_session, &prior_peer_reference, &[3, 9]).await?;
    drop(prior_peer_reference);
    peer_session.commit(ready)?;
    let before = session.binding();
    let peer_before = peer_session.binding();
    let peer_pending = peer_model.service.pending_events()?;
    let peer_bytes = peer_model.service.resident_bytes()?;
    let peer_uploads = peer_model.service.uploaded_value_bytes();
    let weak = Arc::downgrade(&model.service);
    let peer_weak = Arc::downgrade(&peer_model.service);
    let source_weak = Arc::downgrade(&source);
    let peer_source_weak = Arc::downgrade(&peer_source);
    let quarantine = crate::physical_quarantine::quarantined_handles_for(model.service.client())?;
    let peer_quarantine =
        crate::physical_quarantine::quarantined_handles_for(peer_model.service.client())?;
    let retained = drop_after_cached_launch(&model, &session).await?;
    let queue = peer_model.service.queue().stats()?;
    if session.binding() != before
        || model.service.ensure_usable().is_ok()
        || model.service.pending_events()? == 0
        || peer_model.service.ensure_usable().is_ok()
        || peer_session.prepare(&[11], true).await.is_ok()
        || peer_session.binding() != peer_before
        || peer_model.service.pending_events()? != peer_pending
        || peer_model.service.resident_bytes()? != peer_bytes
        || peer_model.service.uploaded_value_bytes() != peer_uploads
        || peer_model.service.queue().stats()? != queue
        || ResidentService::new(
            peer.device(),
            peer_source.identity,
            ResourceOwner::new("portable-cross-provider", "new-after-fence")?,
            PROVIDER,
            DEVICE,
        )
        .is_ok()
        || crate::physical_quarantine::quarantined_handles_for(model.service.client())?
            < quarantine + retained
    {
        return Err(Error::State(
            "portable actual cross-provider fence/publication refusal".into(),
        ));
    }
    drop((
        reference,
        peer_reference,
        session,
        peer_session,
        source,
        peer_source,
        model,
        peer_model,
    ));
    if source_weak.strong_count() == 0
        || peer_source_weak.strong_count() == 0
        || weak.strong_count() == 0
        || peer_weak.strong_count() == 0
    {
        return Err(Error::State(
            "portable cross-provider caller-drop ownership".into(),
        ));
    }
    let service = weak
        .upgrade()
        .ok_or_else(|| Error::State("portable primary quarantine absent".into()))?;
    let peer_service = peer_weak
        .upgrade()
        .ok_or_else(|| Error::State("portable peer quarantine absent".into()))?;
    // Original LeaseInner Drop conservatively quarantines its OWN handles
    // after the shared physical fence, even for the completed peer service.
    let native_retained = crate::physical_quarantine::quarantined_handles_for(service.client())?
        >= quarantine + retained
        && crate::physical_quarantine::quarantined_handles_for(peer_service.client())?
            > peer_quarantine;
    if !native_retained {
        return Err(Error::State(
            "portable each-provider actual handle retention".into(),
        ));
    }
    drop((service, peer_service));
    let budget = ResourceArbiter::process_shared().snapshot(&physical)?;
    if !budget.fenced
        || budget.reserved != 2 * PROVIDER as u64
        || ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
            != 2 * SOURCE_HOST
    {
        return Err(Error::State(
            "portable cross-provider terminal disposition".into(),
        ));
    }
    eprintln!(
        "PORTABLE_CROSS_PROVIDER_PASS submitter={submitter:?} peer={peer:?} pci={PCI} physical_cap={DEVICE} shared_queue=true actual_retained_handles={retained} peer_work_refused=true new_service_refused=true roots_unchanged=true both_provider_envelopes_retained={} host_source_retained=true learned=false bf16_hf=false perf=false; process retirement required",
        2 * PROVIDER
    );
    Ok(SamePciPeerReceipt {
        same_process: true,
        same_pci: true,
        same_arbiter: true,
        contexts_pre_admitted: true,
        submitted_fault: retained > 0,
        opposite_api_refused: true,
        original_source_retained: source_weak.strong_count() > 0
            && peer_source_weak.strong_count() > 0,
        original_native_retained: native_retained,
        original_handle_retained: native_retained,
        physical_fenced: budget.fenced,
        physical_limit: budget.limit,
        physical_reserved: budget.reserved,
        direction: match (submitter, peer) {
            (Backend::Vulkan, Backend::Hip) => "vulkan-then-hip",
            (Backend::Hip, Backend::Vulkan) => "hip-then-vulkan",
            _ => return Err(Error::State("same-PCI peer direction join".into())),
        },
    })
}

#[test]
fn tiny_source_plan_and_shared_reference_are_closed_without_a_device() -> Result<()> {
    let source = Source::new()?;
    let host_key = source.host_key.clone();
    let plan = source.plan()?;
    let (capacity, vocab, preflight) = preflight_profile(source.as_ref(), &plan)?;
    if capacity != 256 || vocab != VOCAB as u64 || preflight.states.len() != 1 {
        return Err(Error::State("portable tiny source/plan geometry".into()));
    }
    let mut oracle = Oracle::new(source.clone())?;
    let expected = oracle.advance(&[1, 7, 11, 17])?;
    if expected.output.iter().all(|x| *x == 0.0) || expected.history.iter().all(|x| *x == 0.0) {
        return Err(Error::State("portable fixture is vacuous".into()));
    }
    let foreign = Source::new()?;
    let host = ResourceArbiter::process_shared().snapshot(&host_key)?;
    if host.reserved != 2 * SOURCE_HOST
        || host.reservations != 2
        || Source::new().is_ok()
        || ResourceArbiter::process_shared()
            .snapshot(&host_key)?
            .reserved
            != host.reserved
    {
        return Err(Error::State(
            "portable shared physical host admission".into(),
        ));
    }
    if preflight_profile(foreign.as_ref(), &plan).is_ok() {
        return Err(Error::State("portable foreign source plan accepted".into()));
    }
    if oracle.advance(&[]).is_ok() || oracle.advance(&[VOCAB as u32]).is_ok() {
        return Err(Error::State("portable source token bounds".into()));
    }
    drop((expected, oracle, source, foreign));
    if ResourceArbiter::process_shared()
        .snapshot(&host_key)?
        .reserved
        != 0
    {
        return Err(Error::State("portable cold fixture host final0".into()));
    }
    Ok(())
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "pvl-x2 gfx1151 physical HIP only; owned isolated process; no local gfx90c launch"]
fn physical_hip_gfx1151_healthy() -> Result<()> {
    futures_lite::future::block_on(healthy(Backend::Hip))
}

#[cfg(all(feature = "hip", feature = "hardware-abort-probe"))]
#[test]
#[ignore = "pvl-x2 gfx1151 HIP actual unknown completion; ONE fresh owned process"]
fn physical_hip_gfx1151_unknown() -> Result<()> {
    futures_lite::future::block_on(unknown(Backend::Hip))
}

#[cfg(feature = "vulkan")]
#[test]
#[ignore = "pvl-x2 c6:00.0 explicit Vulkan physical only; distinct from HIP"]
fn physical_vulkan_pvl_x2_healthy() -> Result<()> {
    futures_lite::future::block_on(healthy(Backend::Vulkan))
}

#[cfg(all(feature = "vulkan", feature = "hardware-abort-probe"))]
#[test]
#[ignore = "pvl-x2 explicit Vulkan actual unknown completion; ONE fresh owned process"]
fn physical_vulkan_pvl_x2_unknown() -> Result<()> {
    futures_lite::future::block_on(unknown(Backend::Vulkan))
}

#[cfg(all(feature = "hip", feature = "vulkan", feature = "hardware-abort-probe"))]
#[test]
#[ignore = "pvl-x2 actual HIP submission fences same-PCI Vulkan; ONE fresh owned process"]
fn physical_hip_fences_vulkan_pvl_x2() -> Result<()> {
    futures_lite::future::block_on(cross_provider_unknown(Backend::Hip, Backend::Vulkan))
        .map(|_| ())
}

#[cfg(all(feature = "hip", feature = "vulkan", feature = "hardware-abort-probe"))]
#[test]
#[ignore = "pvl-x2 actual Vulkan submission fences same-PCI HIP; ONE fresh owned process"]
fn physical_vulkan_fences_hip_pvl_x2() -> Result<()> {
    futures_lite::future::block_on(cross_provider_unknown(Backend::Vulkan, Backend::Hip))
        .map(|_| ())
}
