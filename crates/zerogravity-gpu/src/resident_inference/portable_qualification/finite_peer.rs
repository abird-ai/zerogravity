//! Finite HIP peer used only by the learned Gemma Vulkan same-PCI probe.
//!
//! The fixture is the existing portable Q8 source/plan. Every peer context,
//! compiler and source cost is admitted in the original process arbiter before
//! its own client/model work starts. The peer never becomes a serving owner.
use crate::resident_inference::*;
use cubecl::device::AmdDevice;
use cubecl_common::device::ServiceId;
use std::sync::atomic::Ordering;
use zerogravity_artifacts::quant::QuantKind;
use zerogravity_core::accelerator::{
    SourceIdentity,
    program::{
        ConvDomain, ForwardPlan, MathematicalBackend, MatrixAxes, ParameterRef, ParameterSlot,
        PlanBuilder, SourceTensor, SourceVector,
    },
};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_core::source::{SourceTensorRef, TensorSource};

const PEER_HOST_LIMIT: u64 = 16 * 1024 * 1024 * 1024;
const PEER_DEVICE_LIMIT: u64 = 64 * 1024 * 1024 * 1024;
const PEER_WIDTH: usize = 32;
const PEER_VOCAB: usize = 32;
const PEER_SOURCE_HOST: u64 = 1024 * 1024;
const PEER_PROVIDER: usize = 64 * 1024 * 1024;
const PEER_CONTEXT_HOST: u64 = 2 * 1024 * 1024 * 1024;
const PEER_COMPILER_HOST: u64 = 1024 * 1024 * 1024;
const PEER_CONTEXT_DEVICE: u64 = 1024 * 1024 * 1024;
const PEER_COMPILER_DEVICE: u64 = 1024 * 1024 * 1024;

/// A peer charge owned by the probe for the rest of the process. The learned
/// unknown/fence path cannot know when native cleanup is safe, so these leases
/// intentionally outlive every Rust caller without creating a second arbiter.
struct PersistentCharge(Arc<GlobalResourceLease>);

impl PersistentCharge {
    fn new(lease: GlobalResourceLease) -> Self {
        Self(Arc::new(lease))
    }
}

impl Drop for PersistentCharge {
    fn drop(&mut self) {
        std::mem::forget(self.0.clone());
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PeerCharges {
    pub provider: u64,
    pub source_host: u64,
    pub context_host: u64,
    pub compiler_host: u64,
    pub context_device: u64,
    pub compiler_device: u64,
}

impl PeerCharges {
    pub fn total_device(self) -> u64 {
        self.provider + self.context_device + self.compiler_device
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PeerRefusal {
    pub prepare_refused: bool,
    pub new_service_refused: bool,
    pub root_unchanged: bool,
    pub pending_unchanged: bool,
    pub resident_unchanged: bool,
    pub uploads_unchanged: bool,
    pub queue_unchanged: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PeerRetention {
    pub service_retained: bool,
    pub source_retained: bool,
    pub handles: usize,
    pub handles_added: usize,
}

struct PeerSource {
    identity: SourceIdentity,
    shape: [u64; 2],
    packed: Vec<u8>,
    weights: Vec<f32>,
    // Peer source allocations drop before their host admission.
    _host: GlobalResourceLease,
}

impl PeerSource {
    fn new() -> Result<Arc<Self>> {
        let identity = SourceIdentity::for_new_allocation()?;
        let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
        let arbiter = ResourceArbiter::process_shared();
        arbiter.register_matching(host_key.clone(), PEER_HOST_LIMIT)?;
        let host = arbiter.admit(
            &host_key,
            ResourceOwner::new("finite-hip-peer", &format!("source-{identity:?}"))?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: PEER_SOURCE_HOST,
            }],
        )?;
        let mut packed = Vec::with_capacity(PEER_VOCAB * 34);
        for row in 0..PEER_VOCAB {
            packed.extend_from_slice(&0x3c00u16.to_le_bytes()); // Q8_0 scale1.
            for column in 0..PEER_WIDTH {
                packed.push((((row * 13 + column * 7) % 29) as i8 - 14) as u8);
            }
        }
        let weights = (0..PEER_WIDTH)
            .flat_map(|_| [0.125, 0.25, 0.5, 1.0])
            .collect::<Vec<_>>();
        Ok(Arc::new(Self {
            identity,
            shape: [PEER_WIDTH as u64, PEER_VOCAB as u64],
            packed,
            weights,
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

impl TensorSource for PeerSource {
    fn source_identity(&self) -> SourceIdentity {
        self.identity
    }

    fn tensor(&self, name: &str) -> Result<SourceTensorRef<'_>> {
        if name != "embedding" {
            return Err(Error::Malformed("finite peer source role".into()));
        }
        Ok(SourceTensorRef {
            kind: QuantKind::Q8_0,
            shape: &self.shape,
            packed_bytes: &self.packed,
        })
    }
}

pub(super) struct FiniteHipPeer {
    source: Arc<PeerSource>,
    model: Arc<ResidentModel>,
    session: ResidentSession,
    _host: [PersistentCharge; 2],
    _device: PersistentCharge,
}

impl FiniteHipPeer {
    /// Pre-admit peer source/context/compiler/device costs on the primary's
    /// actual physical key, then load the existing finite HIP fixture.
    pub(super) async fn load(physical: PhysicalDeviceKey) -> Result<Self> {
        let arbiter = ResourceArbiter::process_shared();
        let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
        arbiter.register_matching(host_key.clone(), PEER_HOST_LIMIT)?;
        let device = Device::Hip(AmdDevice::new(0));
        crate::backend::validate_provider(&device)?;
        // Bind the selected ordinal-0 adapter before any context exists. A
        // mismatch or absent PCI identity is an admission refusal, not a
        // reason to charge the shared physical card speculatively.
        let selected = cubecl::hip::preflight_physical(AmdDevice::new(0))
            .ok_or_else(|| Error::State("finite HIP selected adapter PCI unavailable".into()))?;
        let actual = crate::physical_service::physical_key(Some(&selected))?;
        if actual != physical {
            return Err(Error::State(
                "finite HIP selected adapter PCI differs".into(),
            ));
        }
        // The context/compiler owners are retained before any awaited model
        // or session effect. They intentionally outlive every request shell.
        let context_host = PersistentCharge::new(arbiter.admit(
            &host_key,
            ResourceOwner::new("finite-hip-peer", "context-host")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: PEER_CONTEXT_HOST,
            }],
        )?);
        let compiler_host = PersistentCharge::new(arbiter.admit(
            &host_key,
            ResourceOwner::new("finite-hip-peer", "compiler-host")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: PEER_COMPILER_HOST,
            }],
        )?);
        arbiter.register_matching(actual.clone(), PEER_DEVICE_LIMIT)?;
        let device_charge = PersistentCharge::new(arbiter.admit(
            &actual,
            ResourceOwner::new("finite-hip-peer", "context-compiler-device")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: PEER_CONTEXT_DEVICE + PEER_COMPILER_DEVICE,
            }],
        )?);
        let source = PeerSource::new()?;
        let plan = source.plan()?;
        let model = ResidentModel::load(
            device,
            source.clone(),
            plan,
            PEER_PROVIDER,
            PEER_DEVICE_LIMIT,
        )
        .await?;
        if model.service.physical_key() != actual
            || model.service.source_identity() != source.identity
        {
            let service = Arc::clone(&model.service);
            service.fence_and_retain_owners(&[], Vec::new());
            return Err(Error::State("finite HIP peer source/device join".into()));
        }
        let session = match model.session().await {
            Ok(session) => session,
            Err(error) => {
                let service = Arc::clone(&model.service);
                service.fence_and_retain_owners(&[], Vec::new());
                return Err(error);
            }
        };
        Ok(Self {
            source,
            model,
            session,
            _host: [context_host, compiler_host],
            _device: device_charge,
        })
    }

    pub(super) fn charges(&self) -> PeerCharges {
        PeerCharges {
            provider: PEER_PROVIDER as u64,
            source_host: PEER_SOURCE_HOST,
            context_host: PEER_CONTEXT_HOST,
            compiler_host: PEER_COMPILER_HOST,
            context_device: PEER_CONTEXT_DEVICE,
            compiler_device: PEER_COMPILER_DEVICE,
        }
    }

    pub(super) fn physical_key(&self) -> PhysicalDeviceKey {
        self.model.service.physical_key()
    }

    pub(super) fn queue(&self) -> &Arc<crate::physical_queue::PhysicalQueue> {
        self.model.service.queue()
    }

    pub(super) fn service_id(&self) -> ServiceId {
        self.model.service.client().service_id()
    }

    pub(super) fn resident_bytes(&self) -> Result<usize> {
        self.model.service.resident_bytes()
    }

    pub(super) fn pending_events(&self) -> Result<usize> {
        self.model.service.pending_events()
    }

    pub(super) fn uploaded_value_bytes(&self) -> u64 {
        self.model.service.uploaded_value_bytes()
    }

    pub(super) fn queued(&self) -> Result<crate::physical_queue::QueueStats> {
        self.model.service.queue().stats()
    }

    pub(super) fn quarantine_handles(&self) -> Result<usize> {
        crate::physical_quarantine::quarantined_handles_for(self.model.service.client())
    }

    /// One known, committed peer step before the shared fault. It is finite
    /// source work only and does not claim learned-Gemma native execution.
    pub(super) async fn submit_known(&mut self) -> Result<()> {
        let before = self.session.binding();
        let ready = self.session.prepare(&[3, 9], true).await?;
        self.session.commit(ready)?;
        if self.session.binding().frontier() <= before.frontier() {
            return Err(Error::State(
                "finite HIP peer known completion unchanged".into(),
            ));
        }
        Ok(())
    }

    /// Submit the actual finite unknown through HIP and leave its original
    /// armed flight/queue RAII owning the uncertainty.
    pub(super) async fn abandon_after_launch(&self) -> Result<usize> {
        let before = self.quarantine_handles()?;
        self.model.launch_seen.store(0, Ordering::Release);
        self.model.abort_after_launch.store(1, Ordering::Release);
        let mut future = Box::pin(self.session.prepare(&[11], true));
        let launch = std::future::poll_fn(|cx| match future.as_mut().poll(cx) {
            std::task::Poll::Ready(_) => std::task::Poll::Ready(Err(Error::State(
                "finite HIP fault completed without the real launch pause".into(),
            ))),
            std::task::Poll::Pending => {
                let launch = self.model.launch_seen.load(Ordering::Acquire);
                if launch == 0 {
                    std::task::Poll::Pending
                } else {
                    std::task::Poll::Ready(Ok(launch))
                }
            }
        })
        .await?;
        if launch == 0 {
            return Err(Error::State("finite HIP launch count absent".into()));
        }
        drop(future);
        let handles = self.quarantine_handles()?;
        if handles <= before {
            return Err(Error::State(
                "finite HIP unknown retained no new managed handle".into(),
            ));
        }
        Ok(handles - before)
    }

    /// The opposite provider must refuse work and new service admission after
    /// the shared physical fence, without changing peer roots/quotas.
    pub(super) async fn refuse_after_fence(&self) -> Result<PeerRefusal> {
        let before = self.session.binding();
        let pending = self.pending_events()?;
        let resident = self.resident_bytes()?;
        let uploads = self.uploaded_value_bytes();
        let queue = self.queued()?;
        let prepare = self.session.prepare(&[11], true).await;
        let prepare_refused = prepare.is_err();
        drop(prepare);
        let new_service = ResidentService::new(
            self.model.service.physical_owner().device(),
            self.source.identity,
            ResourceOwner::new("finite-hip-peer", "new-after-shared-fence")?,
            PEER_PROVIDER,
            PEER_DEVICE_LIMIT,
        );
        let refusal = PeerRefusal {
            prepare_refused,
            new_service_refused: new_service.is_err(),
            root_unchanged: self.session.binding() == before,
            pending_unchanged: self.pending_events()? == pending,
            resident_unchanged: self.resident_bytes()? == resident,
            uploads_unchanged: self.uploaded_value_bytes() == uploads,
            queue_unchanged: self.queued()? == queue,
        };
        if !refusal.prepare_refused
            || !refusal.new_service_refused
            || !refusal.root_unchanged
            || !refusal.pending_unchanged
            || !refusal.resident_unchanged
            || !refusal.uploads_unchanged
            || !refusal.queue_unchanged
            || self.model.service.ensure_usable().is_ok()
        {
            return Err(Error::State(
                "finite HIP peer accepted work after shared fence".into(),
            ));
        }
        Ok(refusal)
    }

    /// Retire the peer object without releasing its conservative charges.
    /// Quarantine must retain the source/service and every uncertain handle.
    pub(super) fn retire_and_observe(self) -> Result<PeerRetention> {
        let source_weak = Arc::downgrade(&self.source);
        let service_weak = Arc::downgrade(&self.model.service);
        let client = self.model.service.client().clone();
        let before = crate::physical_quarantine::quarantined_handles_for(&client)?;
        // No observer fence/retain. The ordinary fixture owners drop through
        // their original production chain: an abandoned ResidentSubmission::drop
        // and every managed LeaseInner::drop fence the ORIGINAL service, while
        // ModelAdmission::Drop retains the authentic source/envelope on an
        // uncertain close. The observer only records that result.
        drop(self);
        let handles = if let Some(service) = service_weak.upgrade() {
            crate::physical_quarantine::quarantined_handles_for(service.client())?
        } else {
            crate::physical_quarantine::quarantined_handles_for(&client)?
        };
        let source_retained = source_weak.strong_count() > 0;
        let service_retained = service_weak.strong_count() > 0;
        // The original runtime/quarantine must retain BOTH the service and the
        // authentic source admission. The fixture shell itself owns neither.
        if !service_retained || !source_retained || handles < before {
            return Err(Error::State(
                "finite HIP peer service/source/handle retention".into(),
            ));
        }
        Ok(PeerRetention {
            service_retained,
            source_retained,
            handles,
            handles_added: handles.saturating_sub(before),
        })
    }
}
