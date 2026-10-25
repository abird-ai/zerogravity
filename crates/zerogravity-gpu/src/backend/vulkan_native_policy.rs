//! Fixed finite original-Vulkan native ownership policy; no numerical permit.
//!
//! This is the Vulkan analogue of `hip_native_policy`: one cold source, one
//! prepared SPIR-V catalogue, one persistent context slice. It never reports a
//! HIP fingerprint or `HipSourceSetIdentity`, and never retags Vulkan work.
use crate::physical_service::NativeContextBudget;
use crate::resident_leases::{
    NativeCacheBudget, ResidentPhysical, ResidentService, ResidentSource,
};
use cubecl::client::Client;
use cubecl::wgpu::{
    PreparedWgpuNative, WgpuAdapterPreflight, WgpuKernelRecipe, WgpuNativeAdmission,
    WgpuNativeInventory, WgpuNativeRetention,
};
use cubecl::{Device, server::ServerError};
use cubecl_common::device::ServiceId;
use cubecl_core::CubeKernel;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use std::any::Any;
use std::sync::{Arc, Mutex, Weak};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_core::{Error, Result};

const GIB: usize = 1024 * 1024 * 1024;
pub(crate) const SINGLE_SOURCE_QUALIFICATION_HOST_BYTES: u64 = (16 * GIB) as u64;
const DEVICE_CAP: u64 = (64 * GIB) as u64;
const CONTEXT_DEVICE: u64 = (4 * GIB) as u64;
const CONTEXT_HOST: u64 = (2 * GIB) as u64;
const COMPILER_HOST: usize = 8 * GIB;
const MODULE_OPAQUE: usize = 8 * GIB;
const MAX_REPRESENTATION: usize = GIB;
const MAX_MODULES: usize = 1024;
const CONTROL: usize = 2 * 1024 * 1024;
const POLICY: &str = "x2-radv-gemma-e2b-full-catalogue-provisional-v1";
const RESIDENT_POLICY: &str = "x2-radv-gemma-e2b-resident-projections20-provisional-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VulkanColdProfile {
    Streamed16,
    Resident20,
}
pub(crate) struct VulkanColdCost {
    pub provider: usize,
    pub host: usize,
    pub context_device: usize,
    pub context_host: usize,
    pub native_device: usize,
    pub native_host: usize,
}
impl VulkanColdProfile {
    pub(crate) fn cost(self) -> VulkanColdCost {
        VulkanColdCost {
            provider: match self {
                Self::Streamed16 => 16 * GIB,
                Self::Resident20 => 20 * GIB,
            },
            host: SINGLE_SOURCE_QUALIFICATION_HOST_BYTES as usize,
            context_device: CONTEXT_DEVICE as usize,
            context_host: CONTEXT_HOST as usize,
            native_device: MODULE_OPAQUE + MAX_REPRESENTATION,
            native_host: COMPILER_HOST + MAX_REPRESENTATION + CONTROL,
        }
    }
    fn policy(self) -> &'static str {
        match self {
            Self::Streamed16 => POLICY,
            Self::Resident20 => RESIDENT_POLICY,
        }
    }
}

fn native(error: impl std::fmt::Display) -> Error {
    Error::State(format!("original Vulkan native owner: {error}"))
}

/// Selected adapter admitted before any Vulkan device/context exists.
#[cfg(all(feature = "vulkan", target_os = "linux"))]
pub(crate) struct VulkanDevicePreflight {
    key: PhysicalDeviceKey,
    max_memory: u64,
    setup: WgpuAdapterPreflight,
}

#[cfg(all(feature = "vulkan", target_os = "linux"))]
impl VulkanDevicePreflight {
    pub(crate) fn key(&self) -> &PhysicalDeviceKey {
        &self.key
    }
    pub(crate) fn max_memory(&self) -> u64 {
        self.max_memory
    }
    /// Register this exact adapter and create the one original client. The
    /// physical lease is already in the arbiter, so no second charge is made.
    pub(crate) fn register(self, device: &Device) -> Result<()> {
        let Device::Wgpu(wgpu) = device else {
            return Err(Error::State(
                "Vulkan preflighted context device kind differs".into(),
            ));
        };
        if wgpu.backend != cubecl::device::WgpuBackend::Vulkan {
            return Err(Error::State(
                "Vulkan preflighted context backend differs".into(),
            ));
        }
        cubecl::wgpu::try_init_preflight::<cubecl::wgpu::Vulkan>(
            wgpu,
            self.setup,
            cubecl::wgpu::RuntimeOptions::default(),
        )
        .map(|_| ())
        .map_err(|error| Error::State(format!("Vulkan preflighted context: {error}")))
    }
}

/// Select the actual adapter without a device or queue, so its PCI identity
/// and driver-reported capacity can be admitted first.
#[cfg(all(feature = "vulkan", target_os = "linux"))]
pub(crate) fn preflight_vulkan_device(device: &Device) -> Result<VulkanDevicePreflight> {
    let Device::Wgpu(wgpu) = device else {
        return Err(Error::State(
            "Vulkan adapter preflight requires a wgpu device".into(),
        ));
    };
    if wgpu.backend != cubecl::device::WgpuBackend::Vulkan {
        return Err(Error::State(
            "Vulkan adapter preflight backend differs".into(),
        ));
    }
    let setup = cubecl::wgpu::try_preflight_adapter::<cubecl::wgpu::Vulkan>(wgpu)
        .map_err(|error| Error::State(format!("Vulkan adapter preflight: {error}")))?;
    if setup.backend() != cubecl::wgpu::wgpu::Backend::Vulkan {
        return Err(Error::State(
            "Vulkan adapter preflight selected another backend".into(),
        ));
    }
    let physical = setup.physical().ok_or_else(|| {
        Error::State("Vulkan adapter preflight has no physical PCI identity".into())
    })?;
    let key = crate::physical_service::physical_key(Some(&physical))?;
    let max_memory = setup.max_memory().ok_or_else(|| {
        Error::State("Vulkan adapter preflight has no device-local capacity".into())
    })?;
    Ok(VulkanDevicePreflight {
        key,
        max_memory,
        setup,
    })
}
fn server(error: Error) -> ServerError {
    ServerError::graph_state(error.to_string())
}
struct PersistentHost(Arc<GlobalResourceLease>);
impl Drop for PersistentHost {
    fn drop(&mut self) {
        std::mem::forget(self.0.clone());
    }
}
/// The cumulative compiler/cache backing is process-lifetime, exactly like
/// the HIP native context's compiler lease. It is independent of the source
/// workspace, so losing or releasing the source closure cannot credit it.
struct PersistentCompiler(Arc<GlobalResourceLease>);
impl Drop for PersistentCompiler {
    fn drop(&mut self) {
        std::mem::forget(self.0.clone());
    }
}
struct Context {
    device: Arc<NativeContextBudget>,
    policy: &'static str,
    _host: Arc<PersistentHost>,
    compiler: Arc<PersistentCompiler>,
}
struct Totals {
    modules: usize,
    representation: usize,
    host: usize,
    device: usize,
    host_owners: Vec<WindowWorkspace>,
}
impl Default for Totals {
    fn default() -> Self {
        Self {
            modules: 0,
            representation: 0,
            host: 0,
            device: 0,
            host_owners: Vec::new(),
        }
    }
}
struct Admission {
    budget: Arc<NativeCacheBudget>,
    context: Arc<Context>,
    source: VerifiedTensorSource,
    _source_owner: Arc<dyn Any + Send + Sync>,
    physical: PhysicalDeviceKey,
    service: ServiceId,
    kernels: usize,
    totals: Mutex<Totals>,
    _catalogue_metadata: WindowWorkspace,
    #[cfg(feature = "hardware-abort-probe")]
    fault_after_modules: Option<usize>,
}
impl WgpuNativeAdmission for Admission {
    fn context_owner(&self) -> std::result::Result<WgpuNativeRetention, ServerError> {
        self.context
            .device
            .require(&self.physical, self.service, self.context.policy)
            .map_err(server)?;
        Ok(self.context.clone())
    }
    fn require_usable(&self) -> std::result::Result<(), ServerError> {
        self.context
            .device
            .require(&self.physical, self.service, self.context.policy)
            .map_err(server)?;
        self.budget.require_usable().map_err(server)
    }
    fn fence_unknown(&self) {
        self.context.device.fence_unknown();
        self.budget.fence_unknown();
    }
    fn abort_after_modules(&self) -> Option<usize> {
        #[cfg(feature = "hardware-abort-probe")]
        {
            return self.fault_after_modules;
        }
        #[cfg(not(feature = "hardware-abort-probe"))]
        None
    }
    fn reserve_preparation(
        &self,
        kernels: usize,
    ) -> std::result::Result<WgpuNativeRetention, ServerError> {
        self.require_usable()?;
        if kernels != self.kernels || kernels == 0 || kernels > MAX_MODULES {
            return Err(ServerError::graph_state(
                "Gemma complete Vulkan specialization count",
            ));
        }
        Ok(self.context.compiler.clone() as WgpuNativeRetention)
    }
    fn reserve_module(
        &self,
        inventory: WgpuNativeInventory,
    ) -> std::result::Result<WgpuNativeRetention, ServerError> {
        self.require_usable()?;
        let mut totals = self
            .totals
            .lock()
            .map_err(|_| ServerError::graph_state("Vulkan native totals poisoned"))?;
        let modules = totals
            .modules
            .checked_add(inventory.modules)
            .filter(|n| *n <= self.kernels)
            .ok_or_else(|| ServerError::graph_state("Gemma finite native module count"))?;
        let representation = totals
            .representation
            .checked_add(inventory.retained_host_bytes() as usize)
            .filter(|n| *n <= MAX_REPRESENTATION)
            .ok_or_else(|| ServerError::graph_state("Gemma finite Vulkan representation extent"))?;
        let host = totals
            .host
            .checked_add(inventory.retained_host_bytes() as usize)
            .filter(|n| *n <= MAX_REPRESENTATION)
            .ok_or_else(|| ServerError::graph_state("Gemma retained Vulkan host extent"))?;
        let device_bytes = if totals.modules == 0 {
            MODULE_OPAQUE
        } else {
            0
        }
        .checked_add(inventory.retained_host_bytes() as usize)
        .ok_or_else(|| ServerError::graph_state("Gemma Vulkan device allowance overflow"))?;
        let device = self.budget.reserve(device_bytes).map_err(server)?;
        let host_owner = self
            .source
            .reserve_backend_workspace(inventory.retained_host_bytes() as usize)
            .map_err(server)?;
        let device_total = totals
            .device
            .checked_add(device.bytes())
            .ok_or_else(|| ServerError::graph_state("Gemma native device sum"))?;
        totals.host_owners.push(host_owner);
        let host_owners = std::mem::take(&mut totals.host_owners);
        *totals = Totals {
            modules,
            representation,
            host,
            device: device_total,
            host_owners,
        };
        eprintln!(
            "GEMMA_VULKAN_NATIVE module={modules} retained_host={} representation={} device={}",
            inventory.retained_host_bytes(),
            representation,
            device_total
        );
        Ok(device)
    }
}

/// Cold source-bound original Vulkan service. Only this backend creates its
/// policy; family code derives the concrete catalogue using the original
/// client.
pub(crate) struct ColdVulkanPreparation {
    service: Arc<ResidentService>,
    context: Arc<Context>,
    source: VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
}

pub(crate) fn begin_gemma_qualification(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    profile: VulkanColdProfile,
) -> Result<ColdVulkanPreparation> {
    source.require_process_host_workspace(SINGLE_SOURCE_QUALIFICATION_HOST_BYTES)?;
    if !matches!(&device, Device::Wgpu(wgpu) if wgpu.backend == cubecl::device::WgpuBackend::Vulkan)
    {
        return Err(Error::State(
            "Gemma exact pre-bound Vulkan host/provider policy".into(),
        ));
    }
    configure_single_stream()?;
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let arbiter = ResourceArbiter::process_shared();
    arbiter.register_matching(host_key.clone(), SINGLE_SOURCE_QUALIFICATION_HOST_BYTES)?;
    let persistent_host = persistent_context_host(&arbiter, &host_key)?;
    let compiler = persistent_compiler_host(&arbiter, &host_key)?;
    let (physical, context_device) = ResidentPhysical::new_with_native_context(
        device,
        profile.cost().provider,
        DEVICE_CAP,
        Arc::from(profile.policy()),
        CONTEXT_DEVICE,
    )?;
    let service = physical.attach_model(
        ResidentSource::Windows(source.clone()),
        ResourceOwner::new("source-bf16-candidate", profile.policy())?,
        physical.available_model_capacity()?,
    )?;
    let context = Arc::new(Context {
        device: context_device,
        policy: profile.policy(),
        _host: persistent_host,
        compiler,
    });
    if context.device.bytes() != CONTEXT_DEVICE {
        return Err(Error::State(
            "Gemma actual Vulkan context partition extent differs".into(),
        ));
    }
    require_environment(service.client(), service.physical_key())?;
    Ok(ColdVulkanPreparation {
        service,
        context,
        source: source.clone(),
        source_owner,
    })
}

fn configure_single_stream() -> Result<()> {
    let mut config = CubeClRuntimeConfig::default();
    config.streaming.max_streams = 1;
    config.streaming.policy = cubecl_runtime::config::streaming::StreamPolicy::Single;
    if !CubeClRuntimeConfig::try_set(config.clone()) {
        return Err(Error::State(
            "Gemma Vulkan qualifier requires cold runtime configuration".into(),
        ));
    }
    eprintln!("GEMMA_VULKAN_NATIVE_RUNTIME_CONFIG {config:?}");
    Ok(())
}
fn persistent_context_host(
    arbiter: &ResourceArbiter,
    host_key: &PhysicalDeviceKey,
) -> Result<Arc<PersistentHost>> {
    Ok(Arc::new(PersistentHost(Arc::new(arbiter.admit(
        host_key,
        ResourceOwner::new("backend/vulkan", "persistent-finite-qualification-context")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: CONTEXT_HOST,
        }],
    )?))))
}
fn persistent_compiler_host(
    arbiter: &ResourceArbiter,
    host_key: &PhysicalDeviceKey,
) -> Result<Arc<PersistentCompiler>> {
    Ok(Arc::new(PersistentCompiler(Arc::new(arbiter.admit(
        host_key,
        ResourceOwner::new("backend/vulkan", "persistent-cumulative-compiler-cache")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: COMPILER_HOST as u64,
        }],
    )?))))
}
fn require_environment(client: &Client, physical: PhysicalDeviceKey) -> Result<()> {
    let environment = PreparedWgpuNative::environment(client).map_err(native)?;
    let expected_physical = PhysicalDeviceKey::new("pci", "0000:c6:00.0")?;
    if physical != expected_physical
        || environment.physical.as_deref() != Some("0000:c6:00.0")
        || environment.backend != "Vulkan"
        || !environment.supports_vulkan_compiler
        || environment.name != "wgpu<spirv>"
        || !environment.fingerprint.starts_with("spirv_")
        || environment.effective_storage_buffer_binding_size == 0
        || environment.max_bindings < 7
        || environment.max_cube_count == 0
        || environment.max_units_per_cube < 64
        || environment.max_shared_memory_size < 512
    {
        return Err(Error::Unsupported(
            "Gemma provisional exact x2 Vulkan/SPIR-V environment".into(),
        ));
    }
    eprintln!("GEMMA_VULKAN_NATIVE_ENVIRONMENT {environment:?}");
    Ok(())
}

/// Actual prepared cache token; last completion-owner Drop asks the original
/// actor for all-stream completion and known source-owned retirement. The
/// persistent context/compiler baseline stays a distinct owner.
pub(crate) struct PreparedVulkanSourceNative {
    source: SourceIdentity,
    source_service_generation: u64,
    prepared: Arc<PreparedWgpuNative>,
    host_bytes: usize,
    device_bytes: usize,
    _metadata: WindowWorkspace,
    #[cfg(feature = "hardware-abort-probe")]
    final_owner_admission: std::sync::Weak<Admission>,
}
#[cfg(feature = "hardware-abort-probe")]
pub(crate) struct FinalNativeProbe(std::sync::Weak<Admission>);
#[cfg(feature = "hardware-abort-probe")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FinalNativeObservation {
    pub(crate) retained: bool,
    pub(crate) modules: usize,
    pub(crate) code_bytes: usize,
    pub(crate) host_bytes: usize,
    pub(crate) device_bytes: usize,
}
#[cfg(feature = "hardware-abort-probe")]
impl FinalNativeProbe {
    pub(crate) fn observe(&self) -> Result<FinalNativeObservation> {
        let Some(admission) = self.0.upgrade() else {
            return Ok(FinalNativeObservation::default());
        };
        let totals = admission
            .totals
            .lock()
            .map_err(|_| Error::State("final-owner Vulkan native totals poisoned".into()))?;
        Ok(FinalNativeObservation {
            retained: true,
            modules: totals.modules,
            code_bytes: totals.representation,
            host_bytes: totals.host,
            device_bytes: totals.device,
        })
    }
}
#[cfg(feature = "hardware-abort-probe")]
static LAST_PARTIAL_ADMISSION: Mutex<Option<Weak<Admission>>> = Mutex::new(None);
#[cfg(feature = "hardware-abort-probe")]
fn last_partial_admission() -> &'static Mutex<Option<Weak<Admission>>> {
    &LAST_PARTIAL_ADMISSION
}
/// Actual retained partial-module count from the last Vulkan preparation.
/// Returns `None` once the admission (and every actual partial module) is
/// known-retired; an unretired partial/full owner keeps it observable.
#[cfg(feature = "hardware-abort-probe")]
pub fn retained_partial_native_modules() -> Option<usize> {
    let admission = last_partial_admission().lock().ok()?.as_ref()?.upgrade()?;
    let totals = admission.totals.lock().ok()?;
    (totals.modules > 0).then_some(totals.modules)
}
#[derive(Clone, Copy, Debug)]
pub struct VulkanSourceSetIdentity {
    pub source: SourceIdentity,
    pub source_service_generation: u64,
    pub actor: ServiceId,
    pub set_key: u64,
}
impl PreparedVulkanSourceNative {
    pub(crate) fn identity(&self, service: &ResidentService) -> Result<VulkanSourceSetIdentity> {
        if service.generation() != self.source_service_generation
            || service.source_identity() != self.source
            || service.client().service_id() != self.prepared.service_id()
        {
            return Err(Error::State(
                "Vulkan native original source/service generation differs".into(),
            ));
        }
        Ok(VulkanSourceSetIdentity {
            source: self.source,
            source_service_generation: self.source_service_generation,
            actor: self.prepared.service_id(),
            set_key: self.prepared.set_key(),
        })
    }
    pub(crate) fn host_bytes(&self) -> usize {
        self.host_bytes
    }
    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes
    }
    pub(crate) fn require_live(&self, client: &Client, source: SourceIdentity) -> Result<()> {
        if source != self.source {
            return Err(Error::State("Vulkan native original source differs".into()));
        }
        self.prepared.require_live(client).map_err(native)
    }
    #[cfg(feature = "hardware-abort-probe")]
    pub(crate) fn final_owner_probe(&self) -> FinalNativeProbe {
        FinalNativeProbe(self.final_owner_admission.clone())
    }
}
impl Drop for PreparedVulkanSourceNative {
    fn drop(&mut self) {
        #[cfg(feature = "hardware-abort-probe")]
        crate::resident_leases::final_owner_probe::native_retirement();
        if let Err(error) = self.prepared.retire() {
            eprintln!("Vulkan native retirement retained after uncertainty: {error}");
        }
    }
}
impl ColdVulkanPreparation {
    pub(crate) fn client(&self) -> &Client {
        self.service.client()
    }
    pub(crate) fn service(&self) -> Arc<ResidentService> {
        self.service.clone()
    }
    /// Internal engine adapter: genuine complete catalogue and its already
    /// admitted source metadata; no caller opaque allowance or policy argument.
    pub(crate) async fn prepare(
        self,
        kernels: Vec<Box<dyn CubeKernel>>,
        catalogue_metadata: WindowWorkspace,
    ) -> Result<Arc<PreparedVulkanSourceNative>> {
        let Self {
            service,
            context,
            source,
            source_owner,
        } = self;
        if !catalogue_metadata.source().same_issuer(&source)
            || catalogue_metadata.bytes() != 1024 * 1024
        {
            return Err(Error::State(
                "Vulkan original complete source catalogue metadata owner".into(),
            ));
        }
        let mut kernels = kernels;
        kernels.push(crate::source_float::qualification::kernel(service.client()));
        let kernel_count = kernels.len();
        if kernel_count == 0 || kernel_count > MAX_MODULES {
            return Err(Error::Limit(
                "Gemma complete Vulkan specialization bound".into(),
            ));
        }
        let recipes: Vec<WgpuKernelRecipe> = kernels
            .iter()
            .map(|kernel| WgpuKernelRecipe::new(kernel.id()))
            .collect();
        for recipe in &recipes {
            eprintln!("GEMMA_VULKAN_NATIVE_KERNEL {:?}", recipe.kernel_id);
        }
        let metadata = source.reserve_workspace(1024 * 1024)?;
        let budget = service.native_cache_budget()?;
        budget.require_service(&service)?;
        let admission = Arc::new(Admission {
            budget,
            context,
            source: source.clone(),
            _source_owner: source_owner,
            physical: service.physical_key(),
            service: service.client().service_id(),
            kernels: kernel_count,
            totals: Mutex::new(Totals::default()),
            _catalogue_metadata: catalogue_metadata,
            #[cfg(feature = "hardware-abort-probe")]
            fault_after_modules: std::env::var("GEMMA_VULKAN_NATIVE_FAULT_AFTER_MODULES")
                .ok()
                .and_then(|value| value.parse().ok()),
        });
        #[cfg(feature = "hardware-abort-probe")]
        {
            *last_partial_admission()
                .lock()
                .map_err(|_| Error::State("Vulkan partial admission registry poisoned".into()))? =
                Some(Arc::downgrade(&admission));
        }
        let ticket = service.queue().submit(
            service.budget(),
            service.owner().clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: 1,
            }],
            None,
        )?;
        let mut slot = ticket.wait().await?;
        slot.begin_submission()?;
        // SAFETY: complete original generated catalogue; backend-fixed explicit
        // allowances; SAME detached service/global/ledger; original fair slot.
        let prepared = unsafe {
            PreparedWgpuNative::prepare(service.client(), kernels, recipes, admission.clone())
        }
        .map_err(native)?;
        if prepared.inventory().modules != kernel_count {
            return Err(Error::State(
                "Gemma Vulkan complete module inventory differs".into(),
            ));
        }
        prepared.require_live(service.client()).map_err(native)?;
        let (host_bytes, device_bytes) = {
            let totals = admission
                .totals
                .lock()
                .map_err(|_| Error::State("Gemma Vulkan native totals poisoned".into()))?;
            let host_bytes = COMPILER_HOST
                .checked_add(totals.host)
                .and_then(|n| n.checked_add(CONTROL))
                .ok_or_else(|| Error::Limit("Gemma Vulkan native host inventory".into()))?;
            (host_bytes, totals.device)
        };
        eprintln!(
            "GEMMA_VULKAN_NATIVE_INVENTORY kernels={kernel_count} host={host_bytes} device={device_bytes} persistent_device={CONTEXT_DEVICE} persistent_host={CONTEXT_HOST} {:?}",
            prepared.inventory()
        );
        slot.finish()?;
        #[cfg(feature = "hardware-abort-probe")]
        let final_owner_admission = Arc::downgrade(&admission);
        let owner = Arc::new(PreparedVulkanSourceNative {
            source: source.source_identity(),
            source_service_generation: service.generation(),
            prepared,
            host_bytes,
            device_bytes,
            _metadata: metadata,
            #[cfg(feature = "hardware-abort-probe")]
            final_owner_admission,
        });
        owner.identity(&service)?;
        crate::source_float::qualification::run(&service, &source).await?;
        Ok(owner)
    }
}
