//! Finite native preparation on the original HIP actor. No source/math permit.
use crate::compute::{HipServer, context::HipCompiledKernel};
use cubecl_common::{device::ServiceId, device_handle::DeviceHandle};
use cubecl_core::{client::Client, prelude::KernelId, server::ServerError};
use cubecl_server::kernel::CubeKernel;
use std::{
    collections::HashMap,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

const MAX_KERNELS: usize = 4096;
mod artifact;
mod observation;
mod scoped;
pub use artifact::{HipNativeInventory, NativeRetention};
pub(crate) use artifact::{LoadedNativeOwner, PersistentNativeOwner, code_inventory};
pub use observation::{HipScopedObservation, HipScopedSetObservation};
pub use scoped::scoped_metadata_bytes;
pub(crate) use scoped::{NativeMode, NativeSets};

static NEXT_SET: AtomicU64 = AtomicU64::new(1);
fn next_key() -> Result<u64, ServerError> {
    NEXT_SET
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        .map_err(|_| failure("HIP native set/logical identity exhausted"))
}

pub(crate) fn failure(message: impl Into<String>) -> ServerError {
    ServerError::graph_state(message.into())
}

/// Trusted engine adapter. Returned owners must retain the actual source/service,
/// host admission and original physical envelope. No caller byte totals enter
/// preparation; each module request is derived by this backend from actual code.
/// Compiler/RTC opaque memory is a separate explicitly admitted preparation cost.
pub trait HipNativeAdmission: Send + Sync + 'static {
    /// SAME pre-admitted primary-context/RTC proof on every set for this actor.
    /// It must not retain the model, Client or an actor-owned cache.
    fn context_owner(&self) -> Result<NativeRetention, ServerError>;
    /// Checks the original physical/service fence without creating new work.
    fn require_usable(&self) -> Result<(), ServerError>;
    /// Unknown compiler/load/completion/unload retains and fences original proof.
    fn fence_unknown(&self);
    fn reserve_preparation(&self, kernels: usize) -> Result<NativeRetention, ServerError>;
    /// Source-owned host lease before the scoped key tables are allocated.
    fn reserve_scoped_metadata(&self, _bytes: usize) -> Result<NativeRetention, ServerError> {
        Err(failure(
            "HIP admission does not support scoped source metadata",
        ))
    }
    fn reserve_module(
        &self,
        artifact: HipNativeInventory,
        code: &[i8],
    ) -> Result<NativeRetention, ServerError>;
}

/// Exact runtime observation for a finite policy; no admission/fit authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HipNativeEnvironment {
    pub fingerprint: String,
    pub compiler_options: String,
    pub runtime_version: i32,
    pub driver_version: i32,
    pub rtc_major: i32,
    pub rtc_minor: i32,
}

/// Immutable pinned view reserved before source-service allocation/enqueue.
/// The trusted engine attaches its source/admission before preparing the catalogue.
pub struct RegisteredHipNative {
    actor: DeviceHandle<HipServer>,
    client: Arc<Client>,
    key: u64,
    registered: bool,
}
impl RegisteredHipNative {
    /// Fix Scoped mode before exposing the cold physical owner to any source.
    /// # Safety
    /// Same original pre-admitted cold context and one-stream obligations as
    /// `register`; no legacy allocation/compile/source work may precede this.
    pub unsafe fn select_scoped(client: &Client) -> Result<(), ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<HipServer>(service.device) {
            return Err(failure("HIP scoped mode requires original HIP service"));
        }
        DeviceHandle::<HipServer>::new(service.device)
            .submit_blocking(|server| server.select_scoped_native())
            .map_err(|e| failure(format!("HIP scoped mode actor: {e:?}")))?
    }
    /// # Safety
    /// Use only for the original cold, pre-admitted HIP actor with one physical
    /// stream. Install this exact Client in one source service before allocations;
    /// preserve that source's real admission and complete catalogue through owners.
    pub unsafe fn register(client: &Client) -> Result<Self, ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<HipServer>(service.device) {
            return Err(failure("HIP scoped view requires original HIP service"));
        }
        let key = next_key()?;
        // Checked nonzero identities are never reused, even after retirement.
        // Scoped mode requires Single policy: the neutral/default ID remains0.
        let logical = cubecl_environment::stream::StreamId { value: key };
        let actor = DeviceHandle::<HipServer>::new(service.device);
        actor
            .submit_blocking(move |server| server.register_native(key, logical))
            .map_err(|e| failure(format!("HIP native registration actor: {e:?}")))??;
        let mut pinned = client.clone();
        // SAFETY: exact actor registration precedes this immutable source view.
        unsafe { pinned.set_stream(logical) };
        Ok(Self {
            actor,
            client: Arc::new(pinned),
            key,
            registered: true,
        })
    }
    pub fn client(&self) -> &Client {
        &self.client
    }
    /// SAME immutable client pointee; source services clone only this Arc.
    pub fn source_client(&self) -> Arc<Client> {
        self.client.clone()
    }
    /// # Safety
    /// Same genuine complete catalogue/admission obligations as
    /// `PreparedHipNative::prepare`; source service must use this pinned view.
    pub unsafe fn prepare(
        mut self,
        kernels: Vec<Box<dyn CubeKernel>>,
        admission: Arc<dyn HipNativeAdmission>,
    ) -> Result<Arc<PreparedHipNative>, ServerError> {
        if kernels.is_empty() || kernels.len() > MAX_KERNELS {
            return Err(failure("HIP bounded scoped catalogue extent"));
        }
        let key = self.key;
        let inventory = self
            .actor
            .submit_blocking(move |server| {
                server.prepare_registered_native(key, kernels, admission)
            })
            .map_err(|e| failure(format!("HIP scoped preparation actor: {e:?}")))??;
        self.registered = false;
        Ok(Arc::new(PreparedHipNative {
            actor: self.actor.clone(),
            key,
            inventory,
            view: Some(Arc::downgrade(&self.client)),
            retired: AtomicBool::new(false),
        }))
    }
}
impl Drop for RegisteredHipNative {
    fn drop(&mut self) {
        if self.registered {
            let key = self.key;
            // Only an empty reservation may retire here. Failed loaded sets
            // stay on the original actor, globally fenced with their admissions.
            let _ = self
                .actor
                .submit_blocking(move |server| server.cancel_native_registration(key));
        }
    }
}

/// Owns the original actor. Dropping this token never releases a cache's charge.
/// Explicit retirement drains all original HIP streams before module unloading.
pub struct PreparedHipNative {
    actor: DeviceHandle<HipServer>,
    key: u64,
    inventory: HipNativeInventory,
    view: Option<Weak<Client>>,
    retired: AtomicBool,
}
impl PreparedHipNative {
    /// Prepare an exact finite set without executing a kernel or allocating an
    /// activation. An already used/unowned context refuses retroactive charging.
    ///
    /// # Safety
    /// `admission` must reserve the original service's host/native budgets before
    /// compiler/module allocation and retain their actual charged owner. Its
    /// distinct context proof must cover the original context constructor and
    /// outlive every module set; this API cannot retroactively admit it. The
    /// caller seals the COMPLETE concrete specialization list and retains this
    /// token on every original initialization/flight. This is not numerical,
    /// source-profile, Ready or publication authority.
    pub unsafe fn prepare(
        client: &Client,
        kernels: Vec<Box<dyn CubeKernel>>,
        admission: Arc<dyn HipNativeAdmission>,
    ) -> Result<Arc<Self>, ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<HipServer>(service.device)
            || kernels.is_empty()
            || kernels.len() > MAX_KERNELS
        {
            return Err(failure("HIP prepared native service/kernel count"));
        }
        let key = next_key()?;
        let actor = DeviceHandle::<HipServer>::new(service.device);
        let inventory = actor
            .submit_blocking(move |server| server.prepare_native(key, kernels, admission))
            .map_err(|e| failure(format!("HIP native preparation actor: {e:?}")))??;
        Ok(Arc::new(Self {
            actor,
            key,
            inventory,
            view: None,
            retired: AtomicBool::new(false),
        }))
    }
    /// Query on the SAME original actor; runtime/driver versions are HIP's
    /// reported versions, not a claim to identify every loaded shared library.
    pub fn environment(client: &Client) -> Result<HipNativeEnvironment, ServerError> {
        let service = client.service_id();
        if service != ServiceId::of::<HipServer>(service.device) {
            return Err(failure(
                "HIP native environment requires original HIP service",
            ));
        }
        DeviceHandle::<HipServer>::new(service.device)
            .submit_blocking(|server| server.native_environment())
            .map_err(|e| failure(format!("HIP native environment actor: {e:?}")))?
    }
    pub fn set_key(&self) -> u64 {
        self.key
    }
    pub fn scoped_logical_stream(&self) -> Option<u64> {
        // Registration assigns this exact never-reused key to the pinned stream.
        self.view.as_ref().map(|_| self.key)
    }
    pub fn service_id(&self) -> ServiceId {
        self.actor.service_id()
    }
    pub fn inventory(&self) -> HipNativeInventory {
        self.inventory
    }
    pub fn require_live(&self, client: &Client) -> Result<(), ServerError> {
        if self.retired.load(Ordering::Acquire)
            || client.service_id() != self.service_id()
            || self.view.as_ref().is_some_and(|view| {
                view.strong_count() == 0 || !std::ptr::eq(view.as_ptr(), client)
            })
        {
            return Err(failure("HIP native set retired/foreign service"));
        }
        self.actor
            .submit_blocking(|server| server.require_native(self.key))
            .map_err(|e| failure(format!("HIP native set actor: {e:?}")))?
    }
    /// Original all-stream completion then actual unload. Failure retains the
    /// failed/remaining modules and their charges. No context recovery is minted.
    pub fn retire(&self) -> Result<(), ServerError> {
        if self.retired.load(Ordering::Acquire) {
            return Err(failure("HIP native set already retired"));
        }
        self.actor
            .submit_blocking(|server| server.retire_native(self.key))
            .map_err(|e| failure(format!("HIP native retirement actor: {e:?}")))??;
        self.retired.store(true, Ordering::Release);
        Ok(())
    }
}

pub(crate) struct PreparedModules {
    pub key: u64,
    pub kernels: HashMap<KernelId, HipCompiledKernel>,
    pub admission: Arc<dyn HipNativeAdmission>,
    pub inventory: HipNativeInventory,
    pub sealed: bool,
    pub failed: bool,
    pub retired: bool,
    pub _preparation: NativeRetention,
    pub _scoped_metadata: Option<NativeRetention>,
}
impl std::fmt::Debug for PreparedModules {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedModules")
            .field("key", &self.key)
            .field("inventory", &self.inventory)
            .field("sealed", &self.sealed)
            .field("failed", &self.failed)
            .finish()
    }
}

impl Drop for PreparedModules {
    fn drop(&mut self) {
        if !self.retired {
            self.admission.fence_unknown();
            // Set preparation/source metadata remains with unresolved modules.
            std::mem::forget(self._preparation.clone());
            if let Some(metadata) = &self._scoped_metadata {
                std::mem::forget(metadata.clone());
            }
            std::mem::forget(self.admission.clone());
        }
    }
}
