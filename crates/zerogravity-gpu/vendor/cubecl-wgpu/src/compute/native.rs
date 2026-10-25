//! Sealed source-owned native catalogue for the Vulkan/SPIR-V target.
//!
//! A prepared set is the only path by which a kernel may run from the
//! source-issued catalogue: every full compilation key is enumerated before
//! construction, all actual modules are retained here, and retirement is
//! gated on the original actor's drain. An uncertain preparation or
//! retirement retains its owners and modules to the process boundary, the
//! same way the original HIP context does.
use super::server::{NativeSource, PipelineEntry};
use crate::AutoCompiler;
use crate::compute::WgpuServer;
use cubecl_common::hash::{StableHash, StableHasher};
use cubecl_common::{device::ServiceId, device_handle::DeviceHandle};
use cubecl_core::WgpuCompilationOptions;
use cubecl_core::client::Client;
use cubecl_core::prelude::{KernelId, Visibility};
use cubecl_core::server::ServerError;
use cubecl_server::compiler::KernelCacheKey;
use std::any::Any;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

/// Opaque process-lifetime retention, mirroring the HIP native owner type.
pub type WgpuNativeRetention = Arc<dyn Any + Send + Sync>;

const PREPARATION_METADATA_BYTES: u64 = 1024 * 1024;
static NEXT_SET: AtomicU64 = AtomicU64::new(1);

/// A full catalogue key. The id is the complete `KernelId`; the backend
/// seals it with its own properties hash and build hash in the set map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuKernelRecipe {
    pub kernel_id: KernelId,
}
impl WgpuKernelRecipe {
    pub fn new(kernel_id: KernelId) -> Self {
        Self { kernel_id }
    }
}

/// Immutable identity of the actor, source build, compiler and device ABI a
/// catalogue was prepared under. A hit must present the same seal; a lazy
/// pipeline compiled under an unknown seal is not authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuNativeSeal {
    pub service: ServiceId,
    pub properties_hash: u64,
    pub build_id: StableHash,
    pub backend: wgpu::Backend,
    pub compiler_options: StableHash,
}
impl WgpuNativeSeal {
    pub fn new(
        service: ServiceId,
        properties_hash: u64,
        build_id: StableHash,
        backend: wgpu::Backend,
        compilation_options: &WgpuCompilationOptions,
    ) -> Self {
        Self {
            service,
            properties_hash,
            build_id,
            backend,
            compiler_options: StableHasher::hash_one(&format!("{compilation_options:?}")),
        }
    }
}

/// Actual retained modules and their source/descriptor/IO/key bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgpuNativeInventory {
    pub modules: usize,
    pub entry_bytes: u64,
    pub spirv_bytes: u64,
    pub binding_bytes: u64,
    pub io_bytes: u64,
    pub key_bytes: u64,
}
impl WgpuNativeInventory {
    pub fn retained_host_bytes(self) -> u64 {
        self.entry_bytes
            .saturating_add(self.spirv_bytes)
            .saturating_add(self.binding_bytes)
            .saturating_add(self.io_bytes)
            .saturating_add(self.key_bytes)
    }
    fn checked_add(self, other: Self) -> Result<Self, ServerError> {
        Ok(Self {
            modules: self
                .modules
                .checked_add(other.modules)
                .ok_or_else(|| ServerError::graph_state("wgpu native module count"))?,
            entry_bytes: self
                .entry_bytes
                .checked_add(other.entry_bytes)
                .ok_or_else(|| ServerError::graph_state("wgpu native entry extent"))?,
            spirv_bytes: self
                .spirv_bytes
                .checked_add(other.spirv_bytes)
                .ok_or_else(|| ServerError::graph_state("wgpu native SPIR-V extent"))?,
            binding_bytes: self
                .binding_bytes
                .checked_add(other.binding_bytes)
                .ok_or_else(|| ServerError::graph_state("wgpu native binding extent"))?,
            io_bytes: self
                .io_bytes
                .checked_add(other.io_bytes)
                .ok_or_else(|| ServerError::graph_state("wgpu native IO extent"))?,
            key_bytes: self
                .key_bytes
                .checked_add(other.key_bytes)
                .ok_or_else(|| ServerError::graph_state("wgpu native key extent"))?,
        })
    }
    pub fn key_bytes() -> u64 {
        (std::mem::size_of::<WgpuKernelRecipe>()
            + std::mem::size_of::<KernelCacheKey>()
            + std::mem::size_of::<KernelId>()) as u64
    }
    pub fn preparation_metadata_bytes() -> u64 {
        PREPARATION_METADATA_BYTES
    }
}

impl From<&NativeSource> for WgpuNativeInventory {
    fn from(source: &NativeSource) -> Self {
        Self {
            modules: 1,
            entry_bytes: (std::mem::size_of::<NativeModule>() as u64)
                .saturating_add(source.entrypoint.len() as u64),
            spirv_bytes: (source.spirv.len() as u64).saturating_mul(4),
            binding_bytes: (source.bindings.len() as u64)
                .saturating_mul(std::mem::size_of::<Visibility>() as u64),
            io_bytes: source
                .io
                .as_ref()
                .map(|io| {
                    (io.len() as u64).saturating_mul(std::mem::size_of::<
                        cubecl_server::kernel::BufferIOAttr,
                    >() as u64)
                })
                .unwrap_or(0),
            key_bytes: Self::key_bytes(),
        }
    }
}

/// What the original Vulkan/RADV actor actually reports. No invented
/// pipeline counts or device names: every field is read from the live server.
#[derive(Clone, Debug)]
pub struct WgpuNativeEnvironment {
    pub service: ServiceId,
    pub name: &'static str,
    pub adapter: String,
    pub fingerprint: String,
    pub backend: String,
    pub compiler_options: String,
    pub supports_vulkan_compiler: bool,
    pub properties_hash: u64,
    pub effective_storage_buffer_binding_size: u64,
    pub max_bindings: usize,
    pub max_cube_count: u32,
    pub max_units_per_cube: u32,
    pub max_shared_memory_size: usize,
    pub physical: Option<String>,
}

/// Backend-owned admission for one complete source catalogue. The backend
/// derives every module request from the actual compiled source and never
/// accepts caller byte totals.
pub trait WgpuNativeAdmission: Send + Sync + 'static {
    /// The same pre-admitted primary-context proof for every set on this
    /// actor. It must not retain the model, Client or an actor-owned cache.
    fn context_owner(&self) -> Result<WgpuNativeRetention, ServerError>;
    /// Check the original physical/service fence without creating new work.
    fn require_usable(&self) -> Result<(), ServerError>;
    /// Compiler/cache opaque preparation charge, separate from the modules.
    fn reserve_preparation(&self, kernels: usize) -> Result<WgpuNativeRetention, ServerError>;
    /// Reserve one actual module's retained source/descriptor/IO/key bytes.
    fn reserve_module(
        &self,
        inventory: WgpuNativeInventory,
    ) -> Result<WgpuNativeRetention, ServerError>;
    /// Unknown release: retain partial/full modules, source and admission.
    fn fence_unknown(&self);
    /// Probe-only actual mid-catalogue boundary. Production always returns `None`.
    fn abort_after_modules(&self) -> Option<usize> {
        None
    }
}

struct NativeModule {
    recipe: WgpuKernelRecipe,
    source: Option<Arc<NativeSource>>,
    entry: Option<PipelineEntry>,
}
pub(crate) struct NativeSet {
    owner: Option<WgpuNativeRetention>,
    preparation: Option<WgpuNativeRetention>,
    admission: Option<Arc<dyn WgpuNativeAdmission>>,
    recipes: Vec<WgpuKernelRecipe>,
    modules: Vec<NativeModule>,
    module_owners: Vec<WgpuNativeRetention>,
    inventory: WgpuNativeInventory,
    seal: WgpuNativeSeal,
    pub(crate) sealed: bool,
    retired: bool,
}
impl Drop for NativeSet {
    fn drop(&mut self) {
        if !self.retired {
            // A partial or unreaped set is an unknown retention: fence the
            // original physical admission before the actual pipelines,
            // SPIR-V, descriptors, IO and admission stay live to the process
            // boundary, exactly like the original context.
            if let Some(admission) = self.admission.as_ref() {
                admission.fence_unknown();
            }
            std::mem::forget(std::mem::take(&mut self.modules));
            std::mem::forget(std::mem::take(&mut self.module_owners));
            std::mem::forget(std::mem::take(&mut self.recipes));
            std::mem::forget(self.owner.take());
            std::mem::forget(self.preparation.take());
            std::mem::forget(self.admission.take());
        }
    }
}
#[derive(Default)]
pub(crate) struct NativeSets {
    active: Mutex<Option<(u64, NativeSet)>>,
}
impl std::fmt::Debug for NativeSets {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("NativeSets")
    }
}
impl NativeSets {
    pub(crate) fn install(&self, key: u64, set: NativeSet) -> Result<(), ServerError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| ServerError::graph_state("wgpu native sets poisoned"))?;
        if active.is_some() {
            return Err(ServerError::graph_state(
                "wgpu native set/logical identity already present",
            ));
        }
        *active = Some((key, set));
        Ok(())
    }
    pub(crate) fn has_active(&self) -> bool {
        self.active
            .lock()
            .map(|active| active.is_some())
            .unwrap_or(true)
    }
    pub(crate) fn sealed(&self) -> bool {
        self.active
            .lock()
            .map(|active| {
                active
                    .as_ref()
                    .is_some_and(|(_, set)| set.sealed && !set.retired)
            })
            .unwrap_or(true)
    }
    /// A hit is only authority when the set is sealed, the presented seal
    /// matches the one recorded at preparation, and the original admission is
    /// still usable. No lazy fallback for an unknown key.
    pub(crate) fn pipeline(
        &self,
        id: &KernelId,
        seal: &WgpuNativeSeal,
    ) -> Result<Option<PipelineEntry>, ServerError> {
        let active = self
            .active
            .lock()
            .map_err(|_| ServerError::graph_state("wgpu native sets poisoned"))?;
        let Some((_, set)) = active.as_ref() else {
            return Ok(None);
        };
        if set.retired || !set.sealed {
            return Err(ServerError::graph_state(
                "wgpu native catalogue is not a live sealed set",
            ));
        }
        if set.seal != *seal {
            return Err(ServerError::graph_state(
                "wgpu native catalogue actor/source/compiler seal differs",
            ));
        }
        set.admission().require_usable()?;
        set.modules
            .iter()
            .find(|module| module.recipe.kernel_id == *id)
            .and_then(|module| module.entry.clone())
            .map(Some)
            .ok_or_else(|| ServerError::graph_state("wgpu native catalogue refused unknown key"))
    }
    pub(crate) fn with_mut<R>(
        &self,
        key: u64,
        f: impl FnOnce(&mut NativeSet) -> Result<R, ServerError>,
    ) -> Result<R, ServerError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| ServerError::graph_state("wgpu native sets poisoned"))?;
        let (stored, set) = active
            .as_mut()
            .ok_or_else(|| ServerError::graph_state("wgpu native set absent"))?;
        if *stored != key {
            return Err(ServerError::graph_state("wgpu native set identity differs"));
        }
        f(set)
    }
    pub(crate) fn take(&self, key: u64) -> Result<NativeSet, ServerError> {
        let mut active = self
            .active
            .lock()
            .map_err(|_| ServerError::graph_state("wgpu native sets poisoned"))?;
        match active.as_ref() {
            Some((stored, _)) if *stored == key => Ok(active.take().expect("checked").1),
            _ => Err(ServerError::graph_state("wgpu native set identity differs")),
        }
    }
}

fn call<R>(
    actor: &DeviceHandle<WgpuServer<AutoCompiler>>,
    f: impl FnOnce(&mut WgpuServer<AutoCompiler>) -> R + Send,
) -> Result<R, ServerError>
where
    R: Send,
{
    actor
        .submit_blocking(f)
        .map_err(|error| ServerError::graph_state(format!("wgpu native actor: {error:?}")))
}

/// Fence the original admission if the actor call fails or unwinds before a
/// caller proves the new state. Disarm only after the actor returned a
/// definitive result and any partial set is owned by the server.
pub(crate) struct FenceOnDrop {
    admission: Option<Arc<dyn WgpuNativeAdmission>>,
}
impl FenceOnDrop {
    pub(crate) fn new(admission: Arc<dyn WgpuNativeAdmission>) -> Self {
        Self {
            admission: Some(admission),
        }
    }
    pub(crate) fn disarm(&mut self) {
        self.admission = None;
    }
}
impl Drop for FenceOnDrop {
    fn drop(&mut self) {
        if let Some(admission) = self.admission.take() {
            admission.fence_unknown();
        }
    }
}

/// Actual prepared catalogue. Its public surface is the live proof, not a
/// constructor: only the original actor can create or retire it.
pub struct PreparedWgpuNative {
    actor: DeviceHandle<WgpuServer<AutoCompiler>>,
    key: u64,
    inventory: WgpuNativeInventory,
    admission: Arc<dyn WgpuNativeAdmission>,
}
impl PreparedWgpuNative {
    /// Cold construction of the complete source-issued catalogue. No model
    /// buffer is allocated or launched.
    ///
    /// # Safety
    /// `recipes` must be the complete, genuine source-issued catalogue for
    /// `kernels`, bound to this exact source and service. The admission must
    /// retain the authentic pre-admitted context and source lifecycle.
    pub unsafe fn prepare(
        client: &Client,
        kernels: Vec<Box<dyn cubecl_core::CubeKernel>>,
        recipes: Vec<WgpuKernelRecipe>,
        admission: Arc<dyn WgpuNativeAdmission>,
    ) -> Result<Arc<Self>, ServerError> {
        let service = client.service_id();
        let actor = DeviceHandle::<WgpuServer<AutoCompiler>>::new(service.device);
        let key = NEXT_SET.fetch_add(1, Ordering::AcqRel);
        let mut fence = FenceOnDrop::new(admission.clone());
        let prepared = call(&actor, {
            let admission = admission.clone();
            move |server| server.prepare_native(key, kernels, recipes, admission)
        });
        let inventory = match prepared {
            Ok(Ok(inventory)) => {
                fence.disarm();
                inventory
            }
            Ok(Err(error)) => {
                // The server already fenced its installed partial set.
                fence.disarm();
                return Err(error);
            }
            Err(error) => return Err(error),
        };
        Ok(Arc::new(Self {
            actor,
            key,
            inventory,
            admission,
        }))
    }
    pub fn service_id(&self) -> ServiceId {
        self.actor.service_id()
    }
    pub fn set_key(&self) -> u64 {
        self.key
    }
    pub fn inventory(&self) -> WgpuNativeInventory {
        self.inventory
    }
    pub fn require_live(&self, client: &Client) -> Result<(), ServerError> {
        let mut fence = FenceOnDrop::new(self.admission.clone());
        match call(&self.actor, move |server| {
            server.native_live(self.key, client.service_id())
        }) {
            Ok(result) => {
                fence.disarm();
                result
            }
            Err(error) => Err(error),
        }
    }
    pub fn environment(client: &Client) -> Result<WgpuNativeEnvironment, ServerError> {
        let actor = DeviceHandle::<WgpuServer<AutoCompiler>>::new(client.service_id().device);
        call(&actor, |server| server.native_environment())
    }
    pub fn retire(&self) -> Result<WgpuNativeInventory, ServerError> {
        let mut fence = FenceOnDrop::new(self.admission.clone());
        match call(&self.actor, move |server| server.retire_native(self.key)) {
            Ok(Ok(inventory)) => {
                fence.disarm();
                Ok(inventory)
            }
            Ok(Err(error)) => {
                // The server fenced its own refusal; do not double-fence here.
                fence.disarm();
                Err(error)
            }
            Err(error) => Err(error),
        }
    }
}

impl NativeSet {
    pub(crate) fn new(
        owner: WgpuNativeRetention,
        preparation: WgpuNativeRetention,
        admission: Arc<dyn WgpuNativeAdmission>,
        recipes: Vec<WgpuKernelRecipe>,
        seal: WgpuNativeSeal,
    ) -> Self {
        Self {
            owner: Some(owner),
            preparation: Some(preparation),
            admission: Some(admission),
            recipes,
            modules: Vec::new(),
            module_owners: Vec::new(),
            inventory: WgpuNativeInventory::default(),
            seal,
            sealed: false,
            retired: false,
        }
    }
    /// Record the admitted module and its actual source bytes BEFORE the
    /// driver module/pipeline is created. A failure after this point retains
    /// the charge and the partial set.
    pub(crate) fn begin_module(
        &mut self,
        recipe: WgpuKernelRecipe,
        source: Arc<NativeSource>,
        inventory: WgpuNativeInventory,
    ) -> Result<(), ServerError> {
        if self.sealed || self.retired {
            return Err(ServerError::graph_state(
                "wgpu native module after seal/retirement",
            ));
        }
        if self
            .modules
            .iter()
            .any(|module| module.recipe.kernel_id == recipe.kernel_id)
        {
            return Err(ServerError::graph_state("wgpu native duplicate recipe"));
        }
        self.inventory = self.inventory.checked_add(inventory)?;
        self.modules.push(NativeModule {
            recipe,
            source: Some(source),
            entry: None,
        });
        Ok(())
    }
    /// Attach the actual driver entry to the already-admitted module.
    pub(crate) fn complete_module(
        &mut self,
        id: &KernelId,
        entry: PipelineEntry,
    ) -> Result<(), ServerError> {
        let module = self
            .modules
            .iter_mut()
            .find(|module| module.recipe.kernel_id == *id)
            .ok_or_else(|| ServerError::graph_state("wgpu native completed module absent"))?;
        if module.entry.is_some() {
            return Err(ServerError::graph_state(
                "wgpu native module completed twice",
            ));
        }
        module.entry = Some(entry);
        Ok(())
    }
    pub(crate) fn module_count(&self) -> usize {
        self.modules.len()
    }
    pub(crate) fn retain_module_owner(&mut self, owner: WgpuNativeRetention) {
        self.module_owners.push(owner);
    }
    /// The only successful seal has every enumerated recipe constructed.
    pub(crate) fn seal(&mut self) -> Result<WgpuNativeInventory, ServerError> {
        if self.sealed || self.retired {
            return Err(ServerError::graph_state("wgpu native set already sealed"));
        }
        if self.modules.len() != self.recipes.len()
            || self
                .modules
                .iter()
                .any(|module| module.entry.is_none() || module.source.is_none())
        {
            return Err(ServerError::graph_state(
                "wgpu native catalogue sealed before every module was created",
            ));
        }
        self.sealed = true;
        Ok(self.inventory)
    }
    pub(crate) fn admission(&self) -> Arc<dyn WgpuNativeAdmission> {
        self.admission
            .as_ref()
            .expect("live native set has an admission")
            .clone()
    }
    pub(crate) fn inventory(&self) -> WgpuNativeInventory {
        self.inventory
    }
    pub(crate) fn retire(&mut self) {
        self.retired = true;
    }
    pub(crate) fn retired(&self) -> bool {
        self.retired
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::device::{DeviceId, DeviceService, ServerUtilitiesHandle};
    use std::sync::atomic::{AtomicBool, Ordering};

    struct Admission {
        fenced: Arc<AtomicBool>,
    }
    impl WgpuNativeAdmission for Admission {
        fn context_owner(&self) -> Result<WgpuNativeRetention, ServerError> {
            Ok(Arc::new(()))
        }
        fn require_usable(&self) -> Result<(), ServerError> {
            Ok(())
        }
        fn reserve_preparation(&self, _: usize) -> Result<WgpuNativeRetention, ServerError> {
            Ok(Arc::new(()))
        }
        fn reserve_module(
            &self,
            _: WgpuNativeInventory,
        ) -> Result<WgpuNativeRetention, ServerError> {
            Ok(Arc::new(()))
        }
        fn fence_unknown(&self) {
            self.fenced.store(true, Ordering::Release);
        }
    }

    struct PanicActor;
    impl DeviceService for PanicActor {
        fn init(_: DeviceId) -> Self {
            Self
        }
        fn utilities(&self) -> ServerUtilitiesHandle {
            Arc::new(())
        }
    }

    #[test]
    fn definitive_result_disarms_without_double_fence() {
        let fenced = Arc::new(AtomicBool::new(false));
        let admission: Arc<dyn WgpuNativeAdmission> = Arc::new(Admission {
            fenced: fenced.clone(),
        });
        let mut fence = FenceOnDrop::new(admission);
        fence.disarm();
        drop(fence);
        assert!(
            !fenced.load(Ordering::Acquire),
            "server already proved refusal"
        );
    }

    #[test]
    fn actor_panic_fences_original_admission_before_retention() {
        let id = DeviceId {
            type_id: u16::MAX,
            index_id: u16::MAX,
        };
        let fenced = Arc::new(AtomicBool::new(false));
        let admission: Arc<dyn WgpuNativeAdmission> = Arc::new(Admission {
            fenced: fenced.clone(),
        });
        let mut fence = FenceOnDrop::new(admission.clone());
        let actor = DeviceHandle::<PanicActor>::new(id);
        let result: Result<Result<(), ServerError>, _> =
            actor.submit_blocking(|_| -> Result<(), ServerError> {
                panic!("actual wgpu actor panic");
            });
        assert!(result.is_err(), "the actor panic is a call error");
        assert!(
            !fenced.load(Ordering::Acquire),
            "not fenced while result is owned"
        );
        drop(fence);
        assert!(
            fenced.load(Ordering::Acquire),
            "actor failure fences admission"
        );
        drop(actor);
    }
}
