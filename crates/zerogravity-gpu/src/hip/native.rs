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
// A1 gfx1150 / ROCm+driver 7.2.53211 / RTC 9.0: four cold native
// prepare/load/launch/unload cycles plus the complete 92-module Qwen0.8B
// catalogue stayed below 158 MiB observed first-use device delta. Keep a
// 1 GiB fixed policy margin; mismatched actor environments refuse before load.
const QUALIFIED_GFX1150_MODULE_OPAQUE: usize = GIB;
const MAX_CODE: usize = GIB;
const MAX_MODULES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HipSourceNativePolicy {
    module_opaque_bytes: usize,
}
impl HipSourceNativePolicy {
    #[cfg(feature = "hip-source")]
    pub(crate) const FALLBACK: Self = Self {
        module_opaque_bytes: MODULE_OPAQUE,
    };
    const GFX1150_ROCM_7_2_53211: Self = Self {
        module_opaque_bytes: QUALIFIED_GFX1150_MODULE_OPAQUE,
    };

    /// Resident full-K64 packed-index traversal wins on this qualified actor.
    #[cfg(feature = "hip-source")]
    pub(crate) fn prefers_hoisted_projection(self) -> bool {
        self == Self::GFX1150_ROCM_7_2_53211
    }

    #[cfg(feature = "hip-source")]
    fn preparation_upper_bytes(self) -> (usize, usize) {
        (
            COMPILER_HOST + 2 * MAX_CODE + 2 * 1024 * 1024,
            self.module_opaque_bytes + MAX_CODE,
        )
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
    // Cold device facts only nominate the qualified candidate. The actual
    // prepared actor revalidates the exact ROCm/driver/RTC/compiler tuple
    // before any module is compiled or loaded.
    Ok(if observed.arch_name == "gfx1150" && observed.integrated {
        HipSourceNativePolicy::GFX1150_ROCM_7_2_53211
    } else {
        HipSourceNativePolicy::FALLBACK
    })
}
#[cfg(feature = "hip-source")]
#[allow(dead_code)]
pub(crate) fn source_preparation_upper_bytes() -> (usize, usize) {
    HipSourceNativePolicy::FALLBACK.preparation_upper_bytes()
}
#[cfg(feature = "hip-source")]
pub(crate) fn source_preparation_upper_bytes_for(policy: HipSourceNativePolicy) -> (usize, usize) {
    policy.preparation_upper_bytes()
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
            .map_or(MODULE_OPAQUE, |native| native.module_opaque_bytes)
    }
    fn policy(self) -> &'static str {
        match self {
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::Gemma(profile) => profile.policy(),
            #[cfg(feature = "qwen-source-qualification")]
            Self::Qwen => "x2-gfx1151-qwen27b-initial-cached-full-catalogue-provisional-v1",
            #[cfg(feature = "hip-source")]
            Self::ResidentText(budget, native) => match (
                native == HipSourceNativePolicy::GFX1150_ROCM_7_2_53211,
                budget == HipSourceBudget::default(),
            ) {
                (true, true) => "hip-source-gfx1150-rocm7.2.53211-opaque1g-default-v1",
                (true, false) => "hip-source-gfx1150-rocm7.2.53211-opaque1g-explicit-v1",
                (false, true) => "hip-source-bf16-resident-projections-v1",
                (false, false) => "hip-source-bf16-explicit-total-device-budget-v1",
            },
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
const OPTIONS: &str = "HipCompilationOptions { cpp: CompilationOptions { warp_size: 32, supports_features: CppSupportedFeatures { grid_constants: false, clusters: false, fast_math: true, fast_tanh: false, elect_sync: false, dp4a: false }, amd_wmma: Some(Rdna3) }, arch: Some(GfxArch { name: \"gfx1151\", family: GFX11 }) }";
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
        HipSourceNativePolicy::FALLBACK,
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
    begin_single_source(
        device,
        source,
        source_owner,
        SingleSourceProfile::ResidentText(budget, native_policy),
    )
}
fn begin_single_source(
    device: Device,
    source: &VerifiedTensorSource,
    source_owner: crate::physical_quarantine::CompletionOwner,
    profile: SingleSourceProfile,
) -> Result<ColdNativePreparation> {
    source.require_process_host_workspace(SOURCE_HOST_BYTES)?;
    if !matches!(device, Device::Hip(_)) {
        return Err(Error::State(
            "Gemma exact pre-bound HIP host/provider policy".into(),
        ));
    }
    configure_single_stream()?;
    let host_key = PhysicalDeviceKey::new("host", "system-ram")?;
    let arbiter = ResourceArbiter::process_shared();
    // Register_matching never upgrades an existing128MiB CPU contract.
    arbiter.register_matching(host_key.clone(), SOURCE_HOST_BYTES)?;
    let persistent_host = persistent_context_host(&arbiter, &host_key)?;
    let (physical, context_device) =
        crate::resident_leases::ResidentPhysical::new_with_native_context(
            device,
            profile.provider_cap(),
            profile.device_cap(),
            Arc::from(profile.policy()),
            CONTEXT_DEVICE,
        )?;
    let service = physical.attach_model(
        crate::resident_leases::ResidentSource::Windows(source.clone()),
        ResourceOwner::new("source-bf16-candidate", profile.policy())?,
        physical.available_model_capacity()?,
    )?;
    // Attach the persistent host before ANY later fallible policy/source step.
    let module_opaque_bytes = profile.module_opaque_bytes();
    let context = Arc::new(Context {
        device: context_device,
        policy: profile.policy(),
        module_opaque_bytes,
        _host: persistent_host,
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
    require_environment(
        service.client(),
        service.physical_key(),
        profile.resident_text_native_policy(),
    )?;

    Ok(ColdNativePreparation {
        service,
        context,
        source: source.clone(),
        source_owner,
        scoped: None,
    })
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
fn persistent_context_host(
    arbiter: &ResourceArbiter,
    host_key: &PhysicalDeviceKey,
) -> Result<Arc<PersistentHost>> {
    let persistent_host = Arc::new(PersistentHost(Arc::new(arbiter.admit(
        host_key,
        ResourceOwner::new("backend/hip", "persistent-finite-qualification-context")?,
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: CONTEXT_HOST,
        }],
    )?)));
    Ok(persistent_host)
}
fn require_environment(
    client: &Client,
    physical: PhysicalDeviceKey,
    source_policy: Option<HipSourceNativePolicy>,
) -> Result<()> {
    let environment = PreparedHipNative::environment(client).map_err(native)?;

    if let Some(native) = source_policy {
        if physical.backend != "pci" {
            return Err(Error::Unsupported(
                "HIP source runtime environment is incomplete".into(),
            ));
        }
        if native == HipSourceNativePolicy::GFX1150_ROCM_7_2_53211 {
            if environment.fingerprint != "hip-kernel_gfx1150"
                || environment.compiler_options != GFX1150_OPTIONS
                || environment.runtime_version != 70253211
                || environment.driver_version != 70253211
                || environment.rtc_major != 9
                || environment.rtc_minor != 0
            {
                return Err(Error::Unsupported(
                    "qualified gfx1150 HIP native policy environment differs".into(),
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
        || environment.compiler_options != OPTIONS
        || environment.runtime_version != 70253211
        || environment.driver_version != 70253211
        || environment.rtc_major != 9
        || environment.rtc_minor != 0
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
        let compiler_preparation = None;
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
        #[cfg(feature = "qualification")]
        let native_vram_path = format!(
            "/sys/bus/pci/devices/{}/mem_info_vram_used",
            service.physical_key().device
        );
        #[cfg(feature = "qualification")]
        let native_vram_before = std::fs::read_to_string(&native_vram_path)
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
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
        #[cfg(feature = "qualification")]
        eprintln!(
            "HIP_NATIVE_VRAM before={native_vram_before:?} after={:?}",
            std::fs::read_to_string(&native_vram_path)
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
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
