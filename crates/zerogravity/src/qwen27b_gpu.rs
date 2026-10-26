//! Managed exact source facade for the finite HIP teacher diagnostic.
//! It owns SnapshotLease/lifecycle; all math, flight and roots use the engine.
mod source;
use crate::cache::{SnapshotLease, TransferAdmission};
use crate::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use crate::{Error, Result};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use zerogravity_model_qwen35::source::binding::OriginalWindowOperators;

pub struct Qwen27bGpuContext {
    host: TransferAdmission,
    _control: GlobalResourceLease,
}
impl Qwen27bGpuContext {
    pub fn new() -> Result<Arc<Self>> {
        Self::with_host_limit(zerogravity_gpu::Bf16TextSourceCandidate::qualification_host_bytes())
    }
    #[cfg(feature = "shared-hip-qualification")]
    pub fn new_shared_hip_qualification() -> Result<Arc<Self>> {
        Self::with_host_limit(zerogravity_gpu::SharedHipNative::host_bytes())
    }
    fn with_host_limit(host_bytes: u64) -> Result<Arc<Self>> {
        let host = TransferAdmission {
            arbiter: ResourceArbiter::process_shared(),
            resource: PhysicalDeviceKey::new("host", "system-ram")?,
            owner: ResourceOwner::new("Qwen27B", "finite-original-HIP-source")?,
        };
        host.arbiter
            .register_matching(host.resource.clone(), host_bytes)?;
        let control = host.arbiter.admit(
            &host.resource,
            host.owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 4 * 1024 * 1024,
            }],
        )?;
        Ok(Arc::new(Self {
            host,
            _control: control,
        }))
    }
    pub fn reserved_bytes(&self) -> Result<u64> {
        Ok(self.host.arbiter.snapshot(&self.host.resource)?.reserved)
    }
    pub fn fenced(&self) -> Result<bool> {
        Ok(self.host.arbiter.snapshot(&self.host.resource)?.fenced)
    }
}
struct Life {
    native_owners: usize,
    closed: bool,
}
pub struct Qwen27bGpuLoaded {
    source: Arc<OriginalWindowOperators>,
    snapshot: Arc<SnapshotLease>,
    context: Arc<Qwen27bGpuContext>,
    life: Mutex<Life>,
    attempted: AtomicBool,
}
struct SourceRetention(Arc<Qwen27bGpuLoaded>);
impl Drop for SourceRetention {
    fn drop(&mut self) {
        match self.0.life.lock() {
            Ok(mut life) => life.native_owners -= 1,
            Err(_) => std::mem::forget(self.0.clone()),
        }
    }
}
enum NativeTarget {
    Single(zerogravity_gpu::Device),
    #[cfg(feature = "shared-hip-qualification")]
    Shared(Arc<zerogravity_gpu::SharedHipNative>),
}
impl Qwen27bGpuLoaded {
    pub fn open(
        snapshot: Arc<SnapshotLease>,
        context: Arc<Qwen27bGpuContext>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        check_cancel(cancelled)?;
        let source = Arc::new(source::load(&snapshot, &context.host, cancelled)?);
        Ok(Arc::new(Self {
            source,
            snapshot,
            context,
            life: Mutex::new(Life {
                native_owners: 0,
                closed: false,
            }),
            attempted: AtomicBool::new(false),
        }))
    }
    /// Only this validated facade crosses the low-level retained-source contract.
    /// One owned process attempt; exact two-position source scope.
    /// Teacher11/353 and natural selected-ID diagnostics share these owners;
    /// neither loading nor a successful diagnostic grants serving permission.
    pub async fn load_hip_qualification(
        self: &Arc<Self>,
        device: zerogravity_gpu::Device,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<zerogravity_gpu::Bf16TextSourceCandidate>> {
        self.load_native(NativeTarget::Single(device), false, cancelled)
            .await
    }
    /// Distinct natural qualification catalogue; original teacher path stays117.
    /// No numerical/serving grant follows from preparing this candidate.
    pub async fn load_hip_natural_qualification(
        self: &Arc<Self>,
        device: zerogravity_gpu::Device,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<zerogravity_gpu::Bf16TextSourceCandidate>> {
        self.load_native(NativeTarget::Single(device), true, cancelled)
            .await
    }
    /// Original managed source/close guards, pinned to one cold shared actor.
    #[cfg(feature = "shared-hip-qualification")]
    pub async fn load_shared_hip_qualification(
        self: &Arc<Self>,
        shared: Arc<zerogravity_gpu::SharedHipNative>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<zerogravity_gpu::Bf16TextSourceCandidate>> {
        self.load_native(NativeTarget::Shared(shared), false, cancelled)
            .await
    }
    /// Complete authentic source-shaped costs before creating the shared actor.
    #[cfg(feature = "shared-hip-qualification")]
    pub fn shared_hip_cold_plan(
        &self,
        gemma: &crate::gemma4::execution::Model,
        input: &crate::gemma4::execution::Input,
        host: &zerogravity_gpu::TextHostOwners,
    ) -> Result<zerogravity_gpu::SharedHipColdPlan> {
        source::require(&self.snapshot)?;
        let life = self
            .life
            .lock()
            .map_err(|_| Error::State("Qwen source lifecycle poisoned".into()))?;
        if life.closed || life.native_owners != 0 || self.attempted.load(Ordering::Acquire) {
            return Err(Error::State(
                "Qwen cold plan requires unused open source".into(),
            ));
        }
        let (gemma_source, gemma_bindings) = gemma.shared_hip_cold_bindings(input)?;
        let steps = [
            Arc::new(self.source.record_full_source_token(0)?),
            Arc::new(self.source.record_full_source_token(1)?),
        ];
        zerogravity_gpu::SharedHipColdPlan::derive(
            self.source.source(),
            &steps,
            &gemma_source,
            gemma_bindings,
            host,
        )
    }
    async fn load_native(
        self: &Arc<Self>,
        target: NativeTarget,
        natural: bool,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<zerogravity_gpu::Bf16TextSourceCandidate>> {
        #[cfg(feature = "shared-hip-qualification")]
        if natural && matches!(&target, NativeTarget::Shared(_)) {
            return Err(Error::Unsupported(
                "shared Qwen natural readout is not an admitted route".into(),
            ));
        }
        check_cancel(cancelled)?;
        source::require(&self.snapshot)?;
        let steps = [
            Arc::new(self.source.record_full_source_token(0)?),
            Arc::new(self.source.record_full_source_token(1)?),
        ];
        {
            let mut life = self
                .life
                .lock()
                .map_err(|_| Error::State("Qwen source lifecycle poisoned".into()))?;
            if life.closed || life.native_owners != 0 {
                return Err(Error::State(
                    "Qwen source closed or native owner retained".into(),
                ));
            }
            if matches!(&target, NativeTarget::Single(_)) {
                self.attempted
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .map_err(|_| {
                        Error::State("Qwen cold process attempt already consumed".into())
                    })?;
            }
            life.native_owners += 1;
        }
        let retention = Arc::new(SourceRetention(self.clone()));
        // SAFETY: source::require/load authenticates SAME managed31-member snapshot,
        // source issuer/config/18-shard binding, fixed host domain and source profile.
        // Private retention owns that exact SnapshotLease and close counter through
        // every prepared module, initialization, flight and unknown/quarantine owner.
        unsafe {
            match target {
                NativeTarget::Single(device) if natural => {
                    zerogravity_gpu::Bf16TextSourceCandidate::load_hip_natural_qualification(
                        device,
                        self.source.source().clone(),
                        steps,
                        retention,
                        cancelled,
                    )
                    .await
                }
                NativeTarget::Single(device) => {
                    zerogravity_gpu::Bf16TextSourceCandidate::load_hip_qualification(
                        device,
                        self.source.source().clone(),
                        steps,
                        retention,
                        cancelled,
                    )
                    .await
                }
                #[cfg(feature = "shared-hip-qualification")]
                NativeTarget::Shared(shared) => {
                    zerogravity_gpu::Bf16TextSourceCandidate::load_shared_hip_qualification(
                        shared,
                        self.source.source().clone(),
                        steps,
                        retention,
                        cancelled,
                    )
                    .await
                }
            }
        }
    }
    pub fn native_owner_count(&self) -> Result<usize> {
        Ok(self
            .life
            .lock()
            .map_err(|_| Error::State("Qwen source lifecycle poisoned".into()))?
            .native_owners)
    }
    pub fn source_identity(&self) -> crate::accelerator::SourceIdentity {
        self.source.source().source_identity()
    }
    pub fn host_reserved_bytes(&self) -> Result<u64> {
        self.context.reserved_bytes()
    }
    pub fn close(self: Arc<Self>, cancelled: Option<&AtomicBool>) -> Result<()> {
        check_cancel(cancelled)?;
        {
            let mut life = self
                .life
                .lock()
                .map_err(|_| Error::State("Qwen source lifecycle poisoned".into()))?;
            if life.closed || life.native_owners != 0 {
                return Err(Error::State(
                    "Qwen source close requires all original native/readers retired".into(),
                ));
            }
            life.closed = true;
        }
        source::after(&self.snapshot, cancelled)
    }
}
fn check_cancel(flag: Option<&AtomicBool>) -> Result<()> {
    if flag.is_some_and(|f| f.load(Ordering::Acquire)) {
        return Err(Error::State("Qwen managed source cancelled".into()));
    }
    Ok(())
}
