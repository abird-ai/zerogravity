//! Original HIP native ownership; current cold policies remain finite.
//! Preparation itself grants no numerical or ordinary-model eligibility.
use crate::physical_service::NativeContextBudget;
use crate::resident_leases::{ForeignProviderCharge, NativeCacheBudget, ResidentService};
use cubecl::client::Client;
use cubecl::hip::{
    HipNativeAdmission, HipNativeInventory, NativeRetention, PreparedHipNative,
    RegisteredHipNative, SourcePageAuthority,
};
use cubecl::{Device, server::ServerError};
use cubecl_common::device::ServiceId;
use cubecl_runtime::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl_server::kernel::CubeKernel;
use std::sync::{Arc, Mutex};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};
use zerogravity_core::{Error, Result};
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
mod diagnostic;
#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
mod shared;
#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
pub use shared::{SharedHipColdPlan, SharedHipNative};

#[cfg(any(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
pub(crate) enum HipColdOwner {
    Single(Device),
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    Shared(Arc<SharedHipNative>),
}
#[cfg(any(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
impl HipColdOwner {
    pub(crate) fn device(&self) -> Device {
        match self {
            Self::Single(device) => device.clone(),
            #[cfg(all(
                feature = "qwen-source-qualification",
                feature = "gemma-bf16-qualification"
            ))]
            Self::Shared(shared) => shared.device(),
        }
    }
    pub(crate) fn host_bytes(&self) -> u64 {
        match self {
            Self::Single(_) => SINGLE_SOURCE_QUALIFICATION_HOST_BYTES,
            #[cfg(all(
                feature = "qwen-source-qualification",
                feature = "gemma-bf16-qualification"
            ))]
            Self::Shared(_) => SharedHipNative::host_bytes(),
        }
    }
}

#[cfg(feature = "hip-source")]
mod reusable;
#[cfg(feature = "hip-source")]
mod source_budget;
#[cfg(feature = "hip-source")]
pub use source_budget::HipSourceBudget;

const GIB: usize = 1024 * 1024 * 1024;
pub(crate) const SOURCE_HOST_BYTES: u64 = (16 * GIB) as u64;
#[cfg(any(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
pub(crate) const SINGLE_SOURCE_QUALIFICATION_HOST_BYTES: u64 = SOURCE_HOST_BYTES;
#[cfg(feature = "gemma-bf16-qualification")]
pub(crate) const GEMMA_QUALIFICATION_HOST_BYTES: u64 = SINGLE_SOURCE_QUALIFICATION_HOST_BYTES;
#[cfg(any(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
const PROVIDER_CAP: usize = 16 * GIB;
const DEVICE_CAP: u64 = (64 * GIB) as u64;
const CONTEXT_DEVICE: u64 = (4 * GIB) as u64;
const CONTEXT_HOST: u64 = (2 * GIB) as u64;
const COMPILER_HOST: usize = 8 * GIB;
const MODULE_OPAQUE: usize = 8 * GIB;
// Integrated gfx1150/gfx1151, ROCm+driver 7.2.53211 / RTC 9.0:
// four cold lifecycle cycles and complete real Qwen catalogues stayed below
// 194 MB first-use device delta. Keep 1 GiB opaque plus the unchanged code bound.
const QUALIFIED_MODULE_OPAQUE: usize = GIB;
const QUALIFIED_VERSIONS: (i32, i32, i32, i32) = (70253211, 70253211, 9, 0);
const MAX_CODE: usize = GIB;
const MAX_MODULES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HipSourceNativePolicy {
    Fallback,
    Gfx1150,
    Gfx1151,
}
impl HipSourceNativePolicy {
    fn module_opaque_bytes(self) -> usize {
        match self {
            Self::Fallback => MODULE_OPAQUE,
            Self::Gfx1150 | Self::Gfx1151 => QUALIFIED_MODULE_OPAQUE,
        }
    }

    fn required_environment(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Fallback => None,
            Self::Gfx1150 => Some(("hip-kernel_gfx1150", GFX1150_OPTIONS)),
            Self::Gfx1151 => Some(("hip-kernel_gfx1151", GFX1151_OPTIONS)),
        }
    }

    /// Resident full-K64 packed-index traversal wins on this qualified actor.
    #[cfg(feature = "hip-source")]
    pub(crate) fn prefers_hoisted_projection(self) -> bool {
        matches!(self, Self::Gfx1150 | Self::Gfx1151)
    }

    /// Exact guarded FP32 triangular arithmetic wins on qualified gfx1151.
    #[cfg(feature = "hip-source")]
    pub(crate) fn prefers_guarded_triangular(self) -> bool {
        self == Self::Gfx1151
    }

    #[cfg(feature = "hip-source")]
    fn preparation_upper_bytes(self) -> (usize, usize) {
        (
            COMPILER_HOST + 2 * MAX_CODE + 2 * 1024 * 1024,
            self.module_opaque_bytes() + MAX_CODE,
        )
    }

    #[cfg(feature = "hip-source")]
    fn context_policy(self) -> &'static str {
        match self {
            Self::Gfx1150 => "hip-source-gfx1150-rocm7.2.53211-opaque1g-v1",
            Self::Gfx1151 => "hip-source-gfx1151-rocm7.2.53211-opaque1g-v1",
            Self::Fallback => "hip-source-bf16-v1",
        }
    }
}
#[cfg(feature = "hip-source")]
pub(crate) fn source_native_policy(device: &Device) -> Result<HipSourceNativePolicy> {
    let Device::Hip(device) = device else {
        return Err(Error::State(
            "HIP source native policy requires HIP device".into(),
        ));
    };
    let observed = cubecl::hip::observe_device(device.clone())
        .ok_or_else(|| Error::Unsupported("HIP source native device observation".into()))?;
    // Nominate only the qualified integrated device/runtime tuple. The
    // prepared actor revalidates its exact environment before compilation.
    use HipSourceNativePolicy::*;
    let candidate = match (observed.integrated, observed.arch_name.as_str()) {
        (true, "gfx1150") => Gfx1150,
        (true, "gfx1151") => Gfx1151,
        _ => Fallback,
    };
    Ok(
        if candidate != Fallback
            && cubecl::hip::observe_native_versions().ok() == Some(QUALIFIED_VERSIONS)
        {
            candidate
        } else {
            Fallback
        },
    )
}
#[cfg(feature = "hip-source")]
#[allow(dead_code)]
pub(crate) fn source_preparation_upper_bytes() -> (usize, usize) {
    HipSourceNativePolicy::Fallback.preparation_upper_bytes()
}
#[cfg(feature = "hip-source")]
pub(crate) fn source_preparation_upper_bytes_for(policy: HipSourceNativePolicy) -> (usize, usize) {
    let (host, device) = policy.preparation_upper_bytes();
    // Scoped metadata is per live native set, not a retained context cache.
    let metadata =
        cubecl::hip::scoped_metadata_bytes(MAX_MODULES).expect("bounded native catalogue");
    (host + metadata, device)
}

#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
pub(crate) fn scoped_metadata_upper_bytes() -> Result<usize> {
    cubecl::hip::scoped_metadata_bytes(MAX_MODULES).map_err(native)
}

#[cfg(feature = "gemma-bf16-qualification")]
const GEMMA_POLICY: &str = "x2-gfx1151-gemma-e2b-full-catalogue-provisional-v2";
#[derive(Clone, Copy)]
enum SingleSourceProfile {
    #[cfg(feature = "gemma-bf16-qualification")]
    Gemma(GemmaColdProfile),
    #[cfg(feature = "qwen-source-qualification")]
    Qwen,
    #[cfg(feature = "hip-source")]
    ResidentText(HipSourceBudget, HipSourceNativePolicy),
}
impl SingleSourceProfile {
    fn provider_cap(self) -> usize {
        match self {
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::Gemma(profile) => profile.cost().provider,
            #[cfg(feature = "qwen-source-qualification")]
            Self::Qwen => PROVIDER_CAP,
            #[cfg(feature = "hip-source")]
            Self::ResidentText(budget, _) => budget.total_device_bytes(),
        }
    }
    fn device_cap(self) -> u64 {
        match self {
            #[cfg(feature = "hip-source")]
            Self::ResidentText(budget, _) => budget.physical_admission_bytes(),
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::Gemma(_) => DEVICE_CAP,
            #[cfg(feature = "qwen-source-qualification")]
            Self::Qwen => DEVICE_CAP,
        }
    }
    fn resident_text_native_policy(self) -> Option<HipSourceNativePolicy> {
        match self {
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::Gemma(_) => None,
            #[cfg(feature = "qwen-source-qualification")]
            Self::Qwen => None,
            #[cfg(feature = "hip-source")]
            Self::ResidentText(_, native) => Some(native),
        }
    }
    fn module_opaque_bytes(self) -> usize {
        self.resident_text_native_policy()
            .map_or(MODULE_OPAQUE, HipSourceNativePolicy::module_opaque_bytes)
    }
    fn policy(self) -> &'static str {
        match self {
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::Gemma(profile) => profile.policy(),
            #[cfg(feature = "qwen-source-qualification")]
            Self::Qwen => "x2-gfx1151-qwen27b-initial-cached-full-catalogue-provisional-v1",
            #[cfg(feature = "hip-source")]
            Self::ResidentText(_, native) => native.context_policy(),
        }
    }
}

#[cfg(feature = "gemma-bf16-qualification")]
/// Distinct cold finite candidates; existing16GiB evidence never grants resident20.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GemmaColdProfile {
    Streamed16,
    Resident20,
}
#[cfg(feature = "gemma-bf16-qualification")]
pub(crate) struct GemmaColdCost {
    pub provider: usize,
    pub host: usize,
    pub context_device: usize,
    pub context_host: usize,
    pub native_device: usize,
    pub native_host: usize,
}
#[cfg(feature = "gemma-bf16-qualification")]
impl GemmaColdProfile {
    pub(crate) fn cost(self) -> GemmaColdCost {
        GemmaColdCost {
            provider: match self {
                Self::Streamed16 => PROVIDER_CAP,
                Self::Resident20 => 20 * GIB,
            },
            host: GEMMA_QUALIFICATION_HOST_BYTES as usize,
            context_device: CONTEXT_DEVICE as usize,
            context_host: CONTEXT_HOST as usize,
            native_device: MODULE_OPAQUE + MAX_CODE,
            native_host: COMPILER_HOST + 2 * MAX_CODE + 2 * 1024 * 1024,
        }
    }
    fn policy(self) -> &'static str {
        match self {
            Self::Streamed16 => GEMMA_POLICY,
            Self::Resident20 => "x2-gfx1151-gemma-e2b-resident-projections20-provisional-v1",
        }
    }
}
const GFX1151_OPTIONS: &str = "HipCompilationOptions { cpp: CompilationOptions { warp_size: 32, supports_features: CppSupportedFeatures { grid_constants: false, clusters: false, fast_math: true, fast_tanh: false, elect_sync: false, dp4a: false }, amd_wmma: Some(Rdna3) }, arch: Some(GfxArch { name: \"gfx1151\", family: GFX11 }) }";
const GFX1150_OPTIONS: &str = "HipCompilationOptions { cpp: CompilationOptions { warp_size: 32, supports_features: CppSupportedFeatures { grid_constants: false, clusters: false, fast_math: true, fast_tanh: false, elect_sync: false, dp4a: false }, amd_wmma: Some(Rdna3) }, arch: Some(GfxArch { name: \"gfx1150\", family: GFX11 }) }";

fn native(error: impl std::fmt::Display) -> Error {
    Error::State(format!("original HIP native owner: {error}"))
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
struct Context {
    device: Arc<NativeContextBudget>,
    policy: &'static str,
    module_opaque_bytes: usize,
    _host: Arc<PersistentHost>,
    #[cfg(feature = "hip-source")]
    reusable_compiler: Option<Arc<GlobalResourceLease>>,
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    compiler: Option<Arc<shared::PersistentCompilerBudget>>,
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    _shared_control: Option<Arc<GlobalResourceLease>>,
}
#[cfg(feature = "hip-source")]
impl Drop for Context {
    fn drop(&mut self) {
        if let Some(compiler) = &self.reusable_compiler {
            std::mem::forget(compiler.clone());
        }
    }
}

struct Module {
    _device: Arc<ForeignProviderCharge>,
    _host: WindowWorkspace,
}
#[derive(Default)]
struct Totals {
    modules: usize,
    code: usize,
    host: usize,
    device: usize,
    metadata: usize,
}
struct Admission {
    budget: Arc<NativeCacheBudget>,
    context: Arc<Context>,
    source: VerifiedTensorSource,
    _source_owner: Arc<dyn std::any::Any + Send + Sync>,
    physical: PhysicalDeviceKey,
    service: ServiceId,
    kernels: usize,
    totals: Mutex<Totals>,
    _catalogue_metadata: WindowWorkspace,
    compiler_preparation: Option<NativeRetention>,
}
impl HipNativeAdmission for Admission {
    fn context_owner(&self) -> std::result::Result<NativeRetention, ServerError> {
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
    fn reserve_preparation(
        &self,
        kernels: usize,
    ) -> std::result::Result<NativeRetention, ServerError> {
        self.require_usable()?;
        if kernels != self.kernels || kernels == 0 || kernels > MAX_MODULES {
            return Err(ServerError::graph_state(
                "Gemma complete native specialization count",
            ));
        }
        if let Some(owner) = &self.compiler_preparation {
            return Ok(owner.clone());
        }
        Ok(Arc::new(
            self.source
                .reserve_backend_workspace(COMPILER_HOST)
                .map_err(server)?,
        ))
    }
    fn reserve_scoped_metadata(
        &self,
        bytes: usize,
    ) -> std::result::Result<NativeRetention, ServerError> {
        self.require_usable()?;
        if bytes != cubecl::hip::scoped_metadata_bytes(self.kernels)? {
            return Err(ServerError::graph_state("HIP exact scoped metadata extent"));
        }
        let mut totals = self
            .totals
            .lock()
            .map_err(|_| ServerError::graph_state("HIP scoped totals poisoned"))?;
        if totals.metadata != 0 {
            return Err(ServerError::graph_state(
                "HIP scoped metadata already admitted",
            ));
        }
        let owner = self
            .source
            .reserve_backend_workspace(bytes)
            .map_err(server)?;
        totals.metadata = bytes;
        Ok(Arc::new(owner))
    }
    fn reserve_module(
        &self,
        inventory: HipNativeInventory,
        code: &[i8],
    ) -> std::result::Result<NativeRetention, ServerError> {
        self.require_usable()?;
        let mut totals = self
            .totals
            .lock()
            .map_err(|_| ServerError::graph_state("Gemma native totals poisoned"))?;
        let modules = totals
            .modules
            .checked_add(inventory.modules)
            .filter(|n| *n <= self.kernels)
            .ok_or_else(|| ServerError::graph_state("Gemma finite native module count"))?;
        let load = totals
            .code
            .checked_add(inventory.code_object_load_bytes)
            .filter(|n| *n <= MAX_CODE)
            .ok_or_else(|| ServerError::graph_state("Gemma finite native code extent"))?;
        let host = totals
            .host
            .checked_add(inventory.retained_host_bytes)
            .filter(|n| *n <= 2 * MAX_CODE)
            .ok_or_else(|| ServerError::graph_state("Gemma finite retained native host extent"))?;
        let device_bytes = inventory
            .code_object_load_bytes
            .checked_add(if totals.modules == 0 {
                self.context.module_opaque_bytes
            } else {
                0
            })
            .ok_or_else(|| ServerError::graph_state("Gemma native allowance overflow"))?;
        let device = self.budget.reserve(device_bytes).map_err(server)?;
        let host_owner = self
            .source
            .reserve_backend_workspace(inventory.retained_host_bytes)
            .map_err(server)?;
        let device_total = totals
            .device
            .checked_add(device.bytes())
            .ok_or_else(|| ServerError::graph_state("Gemma native device sum"))?;
        *totals = Totals {
            modules,
            code: load,
            host,
            device: device_total,
            metadata: totals.metadata,
        };
        #[cfg(any(
            feature = "qwen-source-qualification",
            feature = "gemma-bf16-qualification"
        ))]
        {
            use sha2::{Digest, Sha256};
            let mut hash = Sha256::new();
            let mut bytes = [0u8; 65536];
            for chunk in code.chunks(bytes.len()) {
                for (out, input) in bytes.iter_mut().zip(chunk) {
                    *out = *input as u8;
                }
                hash.update(&bytes[..chunk.len()]);
            }
            eprintln!(
                "GEMMA_NATIVE_CODE module={modules} bytes={} load={} retained_host={} sha256={:x}",
                code.len(),
                inventory.code_object_load_bytes,
                inventory.retained_host_bytes,
                hash.clone().finalize()
            );
            if self.compiler_preparation.is_some() {
                eprintln!(
                    "HIP_SCOPED_NATIVE_CODE source={:?} module={modules} bytes={} load={} retained_host={} sha256={:x}",
                    self.source.source_identity(),
                    code.len(),
                    inventory.code_object_load_bytes,
                    inventory.retained_host_bytes,
                    hash.clone().finalize()
                );
            }
        }
        #[cfg(not(any(
            feature = "qwen-source-qualification",
            feature = "gemma-bf16-qualification"
        )))]
        let _ = code;
        Ok(Arc::new(Module {
            _device: device,
            _host: host_owner,
        }))
    }
}

/// Cold source-bound original service. Only this backend creates its policy;
/// family code may derive the concrete catalogue using the original client.
pub(crate) struct ColdNativePreparation {
    service: Arc<ResidentService>,
    context: Arc<Context>,
    source: VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    scoped: Option<RegisteredHipNative>,
}

#[cfg(feature = "gemma-bf16-qualification")]
pub(crate) fn begin_gemma_qualification(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    profile: GemmaColdProfile,
) -> Result<ColdNativePreparation> {
    begin_single_source(
        device,
        source,
        source_owner,
        SingleSourceProfile::Gemma(profile),
    )
}
#[cfg(feature = "qwen-source-qualification")]
pub(crate) fn begin_qwen_qualification(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
) -> Result<ColdNativePreparation> {
    begin_single_source(device, source, source_owner, SingleSourceProfile::Qwen)
}
#[cfg(feature = "qwen-source-qualification")]
pub(crate) fn resident_source_device_inventory(model_upper: usize) -> Result<(usize, usize)> {
    HipSourceBudget::default().inventory(model_upper)
}
#[cfg(feature = "qwen-source-qualification")]
pub(crate) fn begin_resident_source(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    model_device_upper: usize,
) -> Result<ColdNativePreparation> {
    begin_resident_source_with_budget(
        device,
        source,
        source_owner,
        model_device_upper,
        HipSourceBudget::default(),
        HipSourceNativePolicy::Fallback,
    )
}
#[cfg(feature = "hip-source")]
pub(crate) fn begin_resident_source_with_budget(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    model_device_upper: usize,
    budget: HipSourceBudget,
    native_policy: HipSourceNativePolicy,
) -> Result<ColdNativePreparation> {
    budget.inventory(model_device_upper)?;
    reusable::begin(
        device,
        source,
        source_owner,
        SingleSourceProfile::ResidentText(budget, native_policy),
    )
}
#[cfg(any(
    feature = "gemma-bf16-qualification",
    feature = "qwen-source-qualification"
))]
fn begin_single_source(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    profile: SingleSourceProfile,
) -> Result<ColdNativePreparation> {
    let (physical, context) = bootstrap_context(device, source, profile)?;
    let service = physical.attach_model(
        crate::resident_leases::ResidentSource::Windows(source.clone()),
        ResourceOwner::new("source-bf16-candidate", profile.policy())?,
        physical.available_model_capacity()?,
    )?;
    Ok(ColdNativePreparation {
        service,
        context,
        source: source.clone(),
        source_owner,
        scoped: None,
    })
}

fn bootstrap_context(
    device: Device,
    source: &VerifiedTensorSource,
    profile: SingleSourceProfile,
) -> Result<(Arc<crate::resident_leases::ResidentPhysical>, Arc<Context>)> {
    source.require_process_host_workspace(SOURCE_HOST_BYTES)?;
    if !matches!(device, Device::Hip(_)) {
        return Err(Error::State(
            "Gemma exact pre-bound HIP host/provider policy".into(),
        ));
    }
    let pre = crate::resident_leases::ResidentPhysical::preadmit_hip_device(
        &device,
        profile.provider_cap(),
        profile.device_cap(),
    )?;
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let arbiter = ResourceArbiter::process_shared();
    // Register_matching never upgrades an existing128MiB CPU contract.
    arbiter.register_matching(host_key.clone(), SOURCE_HOST_BYTES)?;
    let host_lease = context_host_lease(&arbiter, &host_key)?;
    #[cfg(feature = "hip-source")]
    let reusable_compiler = if matches!(profile, SingleSourceProfile::ResidentText(..)) {
        // Peak compiler + opaque retained compiler state share one persistent
        // allowance. Actual module/metadata leases remain per source. All
        // compilation is serialized by the original physical queue.
        Some(Arc::new(source.reserve_detached_host_workspace(
            SOURCE_HOST_BYTES,
            ResourceOwner::new("backend/hip", "compiler-context")?,
            COMPILER_HOST as u64,
        )?))
    } else {
        None
    };
    configure_single_stream()?;
    // Only retain permanently once the irreversible bootstrap is about to start.
    let persistent_host = Arc::new(PersistentHost(Arc::new(host_lease)));
    let (physical, context_device) =
        crate::resident_leases::ResidentPhysical::new_with_native_context(
            device,
            profile.provider_cap(),
            profile.device_cap(),
            Arc::from(profile.policy()),
            CONTEXT_DEVICE,
            Some(pre),
        )?;
    // Attach the persistent host before ANY later fallible policy/source step.
    let module_opaque_bytes = profile.module_opaque_bytes();
    let context = Arc::new(Context {
        device: context_device,
        policy: profile.policy(),
        module_opaque_bytes,
        _host: persistent_host,
        #[cfg(feature = "hip-source")]
        reusable_compiler,
        #[cfg(all(
            feature = "qwen-source-qualification",
            feature = "gemma-bf16-qualification"
        ))]
        compiler: None,
        #[cfg(all(
            feature = "qwen-source-qualification",
            feature = "gemma-bf16-qualification"
        ))]
        _shared_control: None,
    });
    if context.device.bytes() != CONTEXT_DEVICE {
        return Err(Error::State(
            "Gemma actual context partition extent differs".into(),
        ));
    }
    if let Err(error) = require_environment(
        physical.client(),
        physical.physical_key(),
        profile.resident_text_native_policy(),
    ) {
        // The original context exists. No new bootstrap is allowed after an
        // incompatible or unobservable runtime was constructed.
        context.device.fence_unknown();
        return Err(error);
    }
    Ok((physical, context))
}

fn configure_single_stream() -> Result<()> {
    let mut config = CubeClRuntimeConfig::default();
    config.streaming.max_streams = 1;
    config.streaming.policy = cubecl_runtime::config::streaming::StreamPolicy::Single;
    if !CubeClRuntimeConfig::try_set(config.clone()) {
        return Err(Error::State(
            "Gemma qualifier requires cold runtime configuration".into(),
        ));
    }
    #[cfg(feature = "qualification")]
    eprintln!("GEMMA_NATIVE_RUNTIME_CONFIG {config:?}");
    Ok(())
}
fn context_host_lease(
    arbiter: &ResourceArbiter,
    host_key: &PhysicalDeviceKey,
) -> Result<GlobalResourceLease> {
    // The source-shaped lifetime upper bound remains a logical fit, not
    // incremental physical growth. Check the actual next context reservation.
    arbiter.check_host_cold_allocation(CONTEXT_HOST)?;
    arbiter.admit(
        host_key,
        ResourceOwner::new("backend/hip", "persistent-finite-qualification-context")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: CONTEXT_HOST,
        }],
    )
}
fn require_environment(
    client: &Client,
    physical: PhysicalDeviceKey,
    source_policy: Option<HipSourceNativePolicy>,
) -> Result<()> {
    let environment = PreparedHipNative::environment(client).map_err(native)?;
    let versions = (
        environment.runtime_version,
        environment.driver_version,
        environment.rtc_major,
        environment.rtc_minor,
    );

    if let Some(native) = source_policy {
        if physical.backend != "pci" {
            return Err(Error::Unsupported(
                "HIP source runtime environment is incomplete".into(),
            ));
        }
        if let Some((fingerprint, compiler_options)) = native.required_environment() {
            if environment.fingerprint != fingerprint
                || environment.compiler_options != compiler_options
                || versions != QUALIFIED_VERSIONS
            {
                return Err(Error::Unsupported(
                    "qualified HIP native policy environment differs".into(),
                ));
            }
        } else if !environment.fingerprint.starts_with("hip-kernel_gfx")
            || environment.compiler_options.is_empty()
            || environment.runtime_version <= 0
            || environment.driver_version <= 0
            || environment.rtc_major <= 0
        {
            return Err(Error::Unsupported(
                "HIP source runtime environment is incomplete".into(),
            ));
        }
        return Ok(());
    }

    if physical != PhysicalDeviceKey::new("pci", "0000:c6:00.0")?
        || environment.fingerprint != "hip-kernel_gfx1151"
        || environment.compiler_options != GFX1151_OPTIONS
        || versions != QUALIFIED_VERSIONS
    {
        return Err(Error::Unsupported(
            "qualification requires the exact x2 native environment".into(),
        ));
    }
    #[cfg(feature = "qualification")]
    eprintln!("HIP_NATIVE_ENVIRONMENT {environment:?}");
    Ok(())
}

#[cfg(all(
    feature = "qwen-source-qualification",
    feature = "gemma-bf16-qualification"
))]
#[derive(Clone, Copy, Debug)]
pub struct HipSourceSetIdentity {
    pub source: SourceIdentity,
    pub source_service_generation: u64,
    pub actor: ServiceId,
    pub set_key: u64,
    pub logical_stream: Option<u64>,
}
/// Actual prepared cache token; last completion-owner Drop asks original actor
/// for actual all-stream completion/unload. Context remains a distinct baseline.
pub(crate) struct PreparedSourceNative {
    source: SourceIdentity,
    /// Weak live service; per-operation authority, never a bare generation.
    service: std::sync::Weak<ResidentService>,
    prepared: Arc<PreparedHipNative>,
    host_bytes: usize,
    device_bytes: usize,
    _metadata: WindowWorkspace,
    #[cfg(all(
        any(feature = "hip-source", feature = "gemma-bf16-qualification"),
        feature = "hardware-abort-probe"
    ))]
    final_owner_admission: std::sync::Weak<Admission>,
}
#[cfg(all(
    any(feature = "hip-source", feature = "gemma-bf16-qualification"),
    feature = "hardware-abort-probe"
))]
#[derive(Clone)]
pub(crate) struct FinalNativeProbe(std::sync::Weak<Admission>);
#[cfg(all(
    any(feature = "hip-source", feature = "gemma-bf16-qualification"),
    feature = "hardware-abort-probe"
))]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FinalNativeObservation {
    pub(crate) retained: bool,
    pub(crate) modules: usize,
    pub(crate) code_bytes: usize,
    pub(crate) host_bytes: usize,
    pub(crate) device_bytes: usize,
}
#[cfg(all(
    any(feature = "hip-source", feature = "gemma-bf16-qualification"),
    feature = "hardware-abort-probe"
))]
impl FinalNativeProbe {
    pub(crate) fn observe(&self) -> Result<FinalNativeObservation> {
        let Some(admission) = self.0.upgrade() else {
            return Ok(FinalNativeObservation::default());
        };
        let totals = admission
            .totals
            .lock()
            .map_err(|_| Error::State("final-owner original native totals poisoned".into()))?;
        Ok(FinalNativeObservation {
            retained: true,
            modules: totals.modules,
            code_bytes: totals.code,
            host_bytes: totals.host,
            device_bytes: totals.device,
        })
    }
}
/// The original resident service is the only source of live generation/fence
/// authority for a source page. It is not the native SET key and it is not the
/// page-allocation identity.
struct ResidentPageAuthority {
    service: std::sync::Weak<ResidentService>,
    source: SourceIdentity,
}
impl SourcePageAuthority for ResidentPageAuthority {
    fn generation(&self) -> u64 {
        self.service
            .upgrade()
            .map_or(0, |service| service.generation())
    }
    fn require_usable(&self) -> std::result::Result<(), ServerError> {
        let Some(service) = self.service.upgrade() else {
            return Err(ServerError::graph_state(
                "HIP source page original service retired",
            ));
        };
        if service.source_identity() != self.source {
            return Err(ServerError::graph_state(
                "HIP source page original source differs",
            ));
        }
        service.ensure_usable().map_err(server)
    }
    fn fence_unknown(&self) {
        // Existing physical fence only: poison the original service and its
        // prepaid envelope. Page/event/window owners stay with the caller's
        // quarantine; nothing is released or reused on this path.
        if let Some(service) = self.service.upgrade() {
            service.fence_and_retain_owners(&[], Vec::new());
        }
    }
}
impl PreparedSourceNative {
    #[cfg(all(
        any(feature = "hip-source", feature = "gemma-bf16-qualification"),
        feature = "hardware-abort-probe"
    ))]
    pub(crate) fn final_owner_probe(&self) -> FinalNativeProbe {
        FinalNativeProbe(self.final_owner_admission.clone())
    }
    #[cfg(all(
        feature = "qwen-source-qualification",
        feature = "gemma-bf16-qualification"
    ))]
    pub(crate) fn identity(&self, service: &ResidentService) -> HipSourceSetIdentity {
        HipSourceSetIdentity {
            source: self.source,
            source_service_generation: service.generation(),
            actor: self.prepared.service_id(),
            set_key: self.prepared.set_key(),
            logical_stream: self.prepared.scoped_logical_stream(),
        }
    }
    pub(crate) fn host_bytes(&self) -> usize {
        self.host_bytes
    }
    pub(crate) fn device_bytes(&self) -> usize {
        self.device_bytes
    }
    /// Shared source-page transport on the original actor, if this set owns a
    /// scoped logical stream and is still live. The retained weak service is the
    /// per-operation generation/fence authority, never a bare generation value.
    pub(crate) fn source_page_actor(
        &self,
        service: &ResidentService,
    ) -> Option<cubecl::hip::HipSourcePageActor> {
        if self
            .require_live(service.client(), service.source_identity())
            .is_err()
        {
            return None;
        }
        let authority: Arc<dyn SourcePageAuthority> = Arc::new(ResidentPageAuthority {
            service: self.service.clone(),
            source: self.source,
        });
        self.prepared.source_page_actor(authority)
    }
    #[cfg(feature = "hip-source")]
    pub(crate) fn prepare_rocblas(&self, service: &Arc<ResidentService>) -> Result<()> {
        self.require_live(service.client(), service.source_identity())?;
        // The qualified original persistent context already has a 4 GiB
        // device / 2 GiB host allowance; workspace is separately leased.
        unsafe { self.prepared.prepare_rocblas(service.client()) }.map_err(native)
    }

    #[cfg(feature = "hip-source")]
    pub(crate) fn enqueue_rocblas(
        &self,
        service: &Arc<ResidentService>,
        op: cubecl::hip::RocBlasGemm,
        handles: [cubecl::server::Handle; 4],
    ) -> Result<()> {
        self.require_rocblas_service(service)?;
        // Caller retains exact read/write leases in the original flight.
        unsafe { self.prepared.enqueue_rocblas(service.client(), op, handles) }.map_err(native)
    }

    #[cfg(feature = "hip-source")]
    pub(crate) fn enqueue_rocblas_solve(
        &self,
        service: &Arc<ResidentService>,
        op: cubecl::hip::RocBlasUnitLowerSolve,
        handles: [cubecl::server::Handle; 3],
    ) -> Result<()> {
        self.require_rocblas_service(service)?;
        unsafe {
            self.prepared
                .enqueue_rocblas_solve(service.client(), op, handles)
        }
        .map_err(native)
    }

    #[cfg(feature = "hip-source")]
    fn require_rocblas_service(&self, service: &Arc<ResidentService>) -> Result<()> {
        if !self
            .service
            .upgrade()
            .is_some_and(|owner| Arc::ptr_eq(&owner, service))
            || self.source != service.source_identity()
        {
            return Err(Error::State(
                "native dense original service/source differs".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn require_live(&self, client: &Client, source: SourceIdentity) -> Result<()> {
        if source != self.source {
            return Err(Error::State("HIP native original source differs".into()));
        }
        self.prepared.require_live(client).map_err(native)
    }
}
impl Drop for PreparedSourceNative {
    fn drop(&mut self) {
        #[cfg(all(
            any(
                feature = "qwen-source-qualification",
                feature = "gemma-bf16-qualification"
            ),
            feature = "hardware-abort-probe"
        ))]
        crate::resident_leases::final_owner_probe::native_retirement();
        if let Err(error) = self.prepared.retire() {
            eprintln!("HIP native retirement retained after uncertainty: {error}");
        }
    }
}
impl ColdNativePreparation {
    pub(crate) fn client(&self) -> &Client {
        self.service.client()
    }
    pub(crate) fn service(&self) -> Arc<ResidentService> {
        self.service.clone()
    }
    fn require_catalogue_metadata(&self, metadata: &WindowWorkspace) -> Result<()> {
        if !metadata.source().same_issuer(&self.source) || metadata.bytes() != 1024 * 1024 {
            return Err(Error::State(
                "HIP original complete source catalogue metadata owner".into(),
            ));
        }
        Ok(())
    }
    /// Prepare only the supplied kernel set under its original source/queue.
    /// No diagnostic kernels, probes or numerical eligibility are added here.
    pub(crate) async fn prepare(
        self,
        kernels: Vec<Box<dyn CubeKernel>>,
        catalogue_metadata: WindowWorkspace,
    ) -> Result<Arc<PreparedSourceNative>> {
        self.require_catalogue_metadata(&catalogue_metadata)?;
        let Self {
            service,
            context,
            source,
            source_owner,
            scoped,
        } = self;
        let kernel_count = kernels.len();
        if kernel_count == 0 || kernel_count > MAX_MODULES {
            return Err(Error::Limit(
                "Gemma complete native specialization bound".into(),
            ));
        }
        #[cfg(feature = "qualification")]
        for kernel in &kernels {
            eprintln!("GEMMA_NATIVE_KERNEL {:?}", kernel.id());
        }
        let metadata = source.reserve_workspace(1024 * 1024)?;
        let budget = service.native_cache_budget()?;
        budget.require_service(&service)?;
        // A finite precompiler refusal precedes queue submission. Dropping an
        // already submitted ActiveSlot would fence a healthy sibling needlessly.
        #[cfg(all(
            feature = "qwen-source-qualification",
            feature = "gemma-bf16-qualification"
        ))]
        let compiler_preparation = context
            .compiler
            .as_ref()
            .map(|compiler| compiler.reserve().map(|owner| owner as NativeRetention))
            .transpose()?;
        #[cfg(not(all(
            feature = "qwen-source-qualification",
            feature = "gemma-bf16-qualification"
        )))]
        let compiler_preparation: Option<NativeRetention> = None;
        #[cfg(feature = "hip-source")]
        let compiler_preparation = match &context.reusable_compiler {
            Some(owner) => {
                if source.detached_host_workspace_bytes(owner, SOURCE_HOST_BYTES)?
                    != COMPILER_HOST as u64
                {
                    return Err(Error::State(
                        "HIP reusable compiler workspace extent differs".into(),
                    ));
                }
                Some(owner.clone() as NativeRetention)
            }
            None => compiler_preparation,
        };
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
            compiler_preparation,
        });
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
        let prepared = match scoped {
            // SAFETY: SAME trusted source/admission/catalogue; the source service
            // has used this registered immutable view since attachment.
            Some(scoped) => unsafe { scoped.prepare(kernels, admission.clone()) },
            None => unsafe {
                PreparedHipNative::prepare(service.client(), kernels, admission.clone())
            },
        }
        .map_err(native)?;
        if prepared.inventory().modules != kernel_count {
            return Err(Error::State(
                "Gemma native complete module inventory differs".into(),
            ));
        }
        prepared.require_live(service.client()).map_err(native)?;
        let (host_bytes, device_bytes) = {
            let totals = admission
                .totals
                .lock()
                .map_err(|_| Error::State("Gemma native totals poisoned".into()))?;
            let host_bytes = COMPILER_HOST
                .checked_add(totals.host)
                .and_then(|n| n.checked_add(totals.metadata))
                .and_then(|n| n.checked_add(2 * 1024 * 1024))
                .ok_or_else(|| Error::Limit("Gemma native host inventory".into()))?;
            (host_bytes, totals.device)
        };
        #[cfg(feature = "qualification")]
        eprintln!(
            "GEMMA_NATIVE_INVENTORY kernels={kernel_count} host={host_bytes} device={device_bytes} persistent_device={CONTEXT_DEVICE} persistent_host={CONTEXT_HOST} {:?}",
            prepared.inventory()
        );
        slot.finish()?;
        let service_ref = Arc::downgrade(&service);
        let owner = Arc::new(PreparedSourceNative {
            source: source.source_identity(),
            service: service_ref,
            prepared,
            host_bytes,
            device_bytes,
            _metadata: metadata,
            #[cfg(all(
                any(feature = "hip-source", feature = "gemma-bf16-qualification"),
                feature = "hardware-abort-probe"
            ))]
            final_owner_admission: Arc::downgrade(&admission),
        });
        Ok(owner)
    }
}
