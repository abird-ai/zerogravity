//! Concrete cold Qwen-streamed/Gemma-resident candidate on the original owner.
use super::*;
use crate::resident_program::bf16_text::{Bf16TextBody, TextHostOwners, TextResourceInventory};
use crate::resident_program::bf16_text_source::{TextSourceReadout, TextSourceResourceInventory};
use crate::resident_program::text_source::TextSourcePlan;
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::accelerator::program::{TextPlanBindings, TextStepBindings};

pub const SHARED_HOST_BYTES: u64 = (32 * GIB) as u64;
const SHARED_PROVIDER: usize = 32 * GIB;
const QWEN_QUOTA: usize = 12 * GIB;
const GEMMA_QUOTA: usize = 16 * GIB;
const CONTROL_HOST: usize = 64 * 1024;
// Two initial sets +one admitted reload; retired compiler caches get no credit.
const MAX_PREPARATIONS: usize = 3;
const POLICY: &str = "x2-gfx1151-scoped-qwen12-gemma-resident16-provider32-host32-v1";

/// Source-derived cold limits, not native arithmetic/fit authority.
#[derive(Clone, Debug)]
pub struct SharedHipColdPlan {
    qwen_source: SourceIdentity,
    gemma_source: SourceIdentity,
    qwen: TextSourceResourceInventory,
    gemma: TextResourceInventory,
    complete_provider_upper_bytes: usize,
    complete_host_upper_bytes: usize,
}
impl SharedHipColdPlan {
    pub fn qwen(&self) -> &TextSourceResourceInventory {
        &self.qwen
    }
    pub fn gemma(&self) -> &TextResourceInventory {
        &self.gemma
    }
    pub fn complete_provider_upper_bytes(&self) -> usize {
        self.complete_provider_upper_bytes
    }
    pub fn complete_host_upper_bytes(&self) -> usize {
        self.complete_host_upper_bytes
    }
    /// No Client/device allocation: derive both authentic family layouts before
    /// the one cold physical bootstrap. Record owners retire before return.
    pub fn derive(
        qwen: &VerifiedTensorSource,
        qwen_steps: &[Arc<TextStepBindings>; 2],
        gemma: &VerifiedTensorSource,
        gemma_bindings: Arc<TextPlanBindings>,
        gemma_host: &TextHostOwners,
    ) -> Result<Self> {
        qwen.require_process_host_workspace(SHARED_HOST_BYTES)?;
        gemma.require_process_host_workspace(SHARED_HOST_BYTES)?;
        let qwen_source = qwen.source_identity();
        let gemma_source = gemma.source_identity();
        let qwen = crate::resident_program::bf16_text_source::scoped_preparation_inventory(
            qwen,
            qwen_steps,
            TextSourceReadout::TeacherLogits,
        )?;
        let recorded = TextSourcePlan::bind(gemma.clone(), gemma_bindings)?;
        let gemma = Bf16TextBody::scoped_cold_resource_inventory(&recorded, gemma_host)?;
        let complete_provider_upper_bytes = qwen
            .complete_device_bytes
            .checked_add(gemma.device_with_six_roots_bytes)
            .and_then(|n| n.checked_add(CONTEXT_DEVICE as usize))
            .ok_or_else(|| Error::Limit("shared HIP complete device sum".into()))?;
        // BOTH existing reports explicitly include the same2GiB context host.
        // Price it once, retain both per-set compiler/opaque/module/source owners.
        let complete_host_upper_bytes = qwen
            .host_upper_bytes
            .checked_sub(CONTEXT_HOST as usize)
            .and_then(|q| {
                gemma
                    .host_upper_bytes
                    .checked_sub(gemma.persistent_context_host_bytes)
                    .and_then(|g| q.checked_add(g))
            })
            .and_then(|n| n.checked_add(CONTEXT_HOST as usize))
            .and_then(|n| n.checked_add(CONTROL_HOST))
            .and_then(|n| n.checked_add((MAX_PREPARATIONS - 2) * COMPILER_HOST))
            .ok_or_else(|| Error::Limit("shared HIP complete host sum".into()))?;
        let plan = Self {
            qwen_source,
            gemma_source,
            qwen,
            gemma,
            complete_provider_upper_bytes,
            complete_host_upper_bytes,
        };
        plan.require_fit()?;
        Ok(plan)
    }
    fn require_fit(&self) -> Result<()> {
        if self.qwen.complete_device_bytes > QWEN_QUOTA
            || self.gemma.device_with_six_roots_bytes > GEMMA_QUOTA
            || self.gemma.persistent_context_device_bytes != CONTEXT_DEVICE as usize
            || self.gemma.persistent_context_host_bytes != CONTEXT_HOST as usize
            || self.complete_provider_upper_bytes > SHARED_PROVIDER
            || self.complete_host_upper_bytes as u64 > SHARED_HOST_BYTES
        {
            return Err(Error::Limit(
                "shared HIP complete source/host/native/layout cold profile".into(),
            ));
        }
        Ok(())
    }
}

/// Actual compiler/process cache leases on the SAME original persistent context.
/// No source, Client, Admission or model backedge; no unload credit or tombstones.
pub(super) struct PersistentCompilerBudget {
    arbiter: ResourceArbiter,
    host: PhysicalDeviceKey,
    leases: Mutex<Vec<Arc<GlobalResourceLease>>>,
}
impl PersistentCompilerBudget {
    fn new(arbiter: ResourceArbiter, host: PhysicalDeviceKey) -> Self {
        Self {
            arbiter,
            host,
            leases: Mutex::new(Vec::with_capacity(MAX_PREPARATIONS)),
        }
    }
    pub(super) fn reserve(&self) -> Result<Arc<GlobalResourceLease>> {
        let mut leases = self
            .leases
            .lock()
            .map_err(|_| Error::State("HIP persistent compiler ledger poisoned".into()))?;
        if leases.len() >= MAX_PREPARATIONS {
            return Err(Error::Limit(
                "HIP cold cumulative compiler preparation generations exhausted".into(),
            ));
        }
        let owner = Arc::new(self.arbiter.admit(
            &self.host,
            ResourceOwner::new("backend/hip", "persistent-cumulative-compiler-cache")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: COMPILER_HOST as u64,
            }],
        )?);
        // Serialized reserve+retain BEFORE any original compiler/RTC submission.
        // This admitted attempt stays charged even if later preparation cancels.
        leases.push(owner.clone());
        Ok(owner)
    }
    fn bytes(&self) -> Result<u64> {
        let count = self
            .leases
            .lock()
            .map_err(|_| Error::State("HIP persistent compiler ledger poisoned".into()))?
            .len();
        Ok((count * COMPILER_HOST) as u64)
    }
}

/// One original physical actor/arbiter/fair queue with two fixed model quotas.
/// No source object or loaded model is retained by this source-neutral owner.
pub struct SharedHipNative {
    physical: Arc<crate::resident_leases::ResidentPhysical>,
    context: Arc<Context>,
    plan: SharedHipColdPlan,
    qwen_live: Arc<AtomicBool>,
    gemma_live: Arc<AtomicBool>,
}
struct RoleRetention {
    live: Arc<AtomicBool>,
    _source_owner: crate::physical_quarantine::CompletionOwner,
}
impl Drop for RoleRetention {
    fn drop(&mut self) {
        self.live.store(false, Ordering::Release);
    }
}
impl SharedHipNative {
    pub fn host_bytes() -> u64 {
        SHARED_HOST_BYTES
    }
    pub fn cold_plan(&self) -> &SharedHipColdPlan {
        &self.plan
    }
    /// This is a NEW fixed cold candidate. No runtime config/cap may already be
    /// established differently; no set or old single-source proof is upgraded.
    pub fn bootstrap(device: Device, plan: SharedHipColdPlan) -> Result<Arc<Self>> {
        plan.require_fit()?;
        if !matches!(device, Device::Hip(_)) {
            return Err(Error::Unsupported(
                "shared native candidate requires original HIP".into(),
            ));
        }
        configure_single_stream()?;
        let arbiter = ResourceArbiter::process_shared();
        let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
        arbiter.register_matching(host_key.clone(), SHARED_HOST_BYTES)?;
        let control = Arc::new(arbiter.admit(
            &host_key,
            ResourceOwner::new("backend/hip", "shared-cold-source-cost")?,
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: CONTROL_HOST as u64,
            }],
        )?);
        let persistent_host = persistent_context_host(&arbiter, &host_key)?;
        let (physical, context_device) =
            crate::resident_leases::ResidentPhysical::new_with_native_context(
                device,
                SHARED_PROVIDER,
                DEVICE_CAP,
                Arc::from(POLICY),
                CONTEXT_DEVICE,
            )?;
        let context = Arc::new(Context {
            device: context_device,
            policy: POLICY,
            _host: persistent_host,
            compiler: Some(Arc::new(PersistentCompilerBudget::new(
                arbiter.clone(),
                host_key,
            ))),
            _shared_control: Some(control),
        });
        require_environment(physical.client(), physical.physical_key())?;
        // SAFETY: same just-admitted cold actor; fix neutral-client refusal
        // before publishing this source-neutral bootstrap owner.
        unsafe { RegisteredHipNative::select_scoped(physical.client()) }.map_err(native)?;
        Ok(Arc::new(Self {
            physical,
            context,
            plan,
            qwen_live: Arc::new(AtomicBool::new(false)),
            gemma_live: Arc::new(AtomicBool::new(false)),
        }))
    }
    /// Process-lifetime HOST charge; source/model retirement never credits it.
    pub fn compiler_host_reserved_bytes(&self) -> Result<u64> {
        self.context
            .compiler
            .as_ref()
            .ok_or_else(|| Error::State("HIP shared compiler ledger absent".into()))?
            .bytes()
    }
    pub fn inspect_native(&self) -> Result<cubecl::hip::HipScopedObservation> {
        RegisteredHipNative::inspect_scoped(self.physical.client()).map_err(native)
    }
    pub fn stream_preflight_usable(&self, logical: u64) -> Result<bool> {
        RegisteredHipNative::scoped_stream_usable(self.physical.client(), logical).map_err(native)
    }
    pub fn available_model_capacity(&self) -> Result<usize> {
        self.physical.available_model_capacity()
    }
    pub fn resident_bytes(&self) -> Result<usize> {
        self.physical.resident_bytes()
    }
    pub(crate) fn device(&self) -> Device {
        self.physical.device()
    }
    pub(crate) fn begin_qwen(
        &self,
        source: &VerifiedTensorSource,
        source_owner: crate::physical_quarantine::CompletionOwner,
        inventory: &TextSourceResourceInventory,
    ) -> Result<ColdNativePreparation> {
        if source.source_identity() != self.plan.qwen_source
            || inventory.complete_device_bytes > self.plan.qwen.complete_device_bytes
            || inventory.host_upper_bytes > self.plan.qwen.host_upper_bytes
        {
            return Err(Error::Limit("shared Qwen source cold cost changed".into()));
        }
        self.attach(
            source,
            source_owner,
            QWEN_QUOTA,
            self.qwen_live.clone(),
            "Qwen27B",
        )
    }
    pub(crate) fn begin_gemma(
        &self,
        source: &VerifiedTensorSource,
        source_owner: crate::physical_quarantine::CompletionOwner,
        inventory: &TextResourceInventory,
    ) -> Result<ColdNativePreparation> {
        if source.source_identity() != self.plan.gemma_source
            || inventory.device_with_six_roots_bytes > self.plan.gemma.device_with_six_roots_bytes
            || inventory.host_upper_bytes > self.plan.gemma.host_upper_bytes
            || inventory.resident_matrices != self.plan.gemma.resident_matrices
            || inventory.resident_matrix_bytes != self.plan.gemma.resident_matrix_bytes
        {
            return Err(Error::Limit(
                "shared Gemma resident source cold cost changed".into(),
            ));
        }
        self.attach(
            source,
            source_owner,
            GEMMA_QUOTA,
            self.gemma_live.clone(),
            "GemmaE2BResident",
        )
    }
    fn attach(
        &self,
        source: &VerifiedTensorSource,
        source_owner: crate::physical_quarantine::CompletionOwner,
        cap: usize,
        live: Arc<AtomicBool>,
        role: &str,
    ) -> Result<ColdNativePreparation> {
        source.require_process_host_workspace(SHARED_HOST_BYTES)?;
        live.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                Error::State("shared HIP role still has original source/native readers".into())
            })?;
        // Detached admission retains only a flag + actual source guard; never a
        // SharedHipNative/Client/actor/cache backedge.
        let retention = Arc::new(RoleRetention {
            live,
            _source_owner: source_owner,
        });
        // SAFETY: original pre-admitted fixed cold context, one physical stream;
        // pinned view is installed in the new source service before any enqueue.
        let scoped = unsafe { cubecl::hip::RegisteredHipNative::register(self.physical.client()) }
            .map_err(native)?;
        let service = self.physical.attach_native_model(
            crate::resident_leases::ResidentSource::Windows(source.clone()),
            ResourceOwner::new(role, POLICY)?,
            cap,
            &scoped,
        )?;
        Ok(ColdNativePreparation {
            service,
            context: self.context.clone(),
            source: source.clone(),
            source_owner: retention,
            scoped: Some(scoped),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compiler_generations_remain_charged_after_callers_drop_and_limit_is_preeffect() {
        let arbiter = ResourceArbiter::new();
        let host = PhysicalDeviceKey::new("host", "scoped-compiler-lifetime-test").unwrap();
        arbiter
            .register_matching(host.clone(), SHARED_HOST_BYTES)
            .unwrap();
        let compiler = PersistentCompilerBudget::new(arbiter.clone(), host.clone());
        for generation in 1..=MAX_PREPARATIONS {
            let caller = compiler.reserve().unwrap();
            assert_eq!(
                compiler.bytes().unwrap(),
                (generation * COMPILER_HOST) as u64
            );
            drop(caller);
            assert_eq!(
                arbiter.snapshot(&host).unwrap().reserved,
                (generation * COMPILER_HOST) as u64
            );
        }
        assert!(compiler.reserve().is_err());
        let after = arbiter.snapshot(&host).unwrap();
        assert_eq!(after.reserved, (MAX_PREPARATIONS * COMPILER_HOST) as u64);
        assert!(!after.fenced);
    }
}
