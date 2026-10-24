//! Prepared-cache lifetime remains on the original server actor.
use super::*;
use crate::prepared_native::{
    HipNativeAdmission, HipNativeInventory, NativeMode, PreparedModules, failure,
};

impl HipServer {
    pub(crate) fn native_environment(
        &mut self,
    ) -> Result<crate::prepared_native::HipNativeEnvironment, ServerError> {
        self.require_known_storage_retirement()?;
        let (mut runtime_version, mut driver_version, mut rtc_major, mut rtc_minor) = (0, 0, 0, 0);
        // SAFETY: original actor has selected this actual device; each version
        // API writes only the supplied live stack integer(s).
        unsafe {
            crate::compute::status::checked(
                "hipRuntimeGetVersion",
                cubecl_hip_sys::hipRuntimeGetVersion(&mut runtime_version),
            )?;
            crate::compute::status::checked(
                "hipDriverGetVersion",
                cubecl_hip_sys::hipDriverGetVersion(&mut driver_version),
            )?;
            crate::compute::status::checked(
                "hiprtcVersion",
                cubecl_hip_sys::hiprtcVersion(&mut rtc_major, &mut rtc_minor),
            )?;
        }
        Ok(crate::prepared_native::HipNativeEnvironment {
            fingerprint: self.ctx.properties.identity.fingerprint.clone(),
            compiler_options: format!("{:?}", self.ctx.compilation_options),
            runtime_version,
            driver_version,
            rtc_major,
            rtc_minor,
        })
    }
    pub(crate) fn select_scoped_native(&mut self) -> Result<(), ServerError> {
        self.require_known_storage_retirement()?;
        let config = CubeClRuntimeConfig::get();
        if self.ctx.legacy_loaded != 0
            || self.native_graph_seen
            || config.streaming.max_streams != 1
            || !matches!(
                config.streaming.policy,
                cubecl_environment::stream::StreamPolicy::Single
            )
        {
            return Err(failure(
                "HIP scoped mode requires cold original actor/single physical stream",
            ));
        }
        self.ctx.native.select_scoped()
    }
    pub(crate) fn register_native(
        &mut self,
        key: u64,
        logical: StreamId,
    ) -> Result<(), ServerError> {
        self.require_known_storage_retirement()?;
        let config = CubeClRuntimeConfig::get();
        if self.ctx.legacy_loaded != 0
            || self.native_graph_seen
            || config.streaming.max_streams != 1
            || !matches!(
                config.streaming.policy,
                cubecl_environment::stream::StreamPolicy::Single
            )
        {
            return Err(failure(
                "HIP scoped sets require cold original actor/single physical stream",
            ));
        }
        self.ctx.native.register(key, logical, NativeMode::Scoped)
    }
    pub(crate) fn cancel_native_registration(&mut self, key: u64) -> Result<(), ServerError> {
        self.ctx.native.cancel_registration(key)
    }
    pub(crate) fn prepare_native(
        &mut self,
        key: u64,
        kernels: Vec<Box<dyn CubeKernel>>,
        admission: Arc<dyn HipNativeAdmission>,
    ) -> Result<HipNativeInventory, ServerError> {
        self.require_known_storage_retirement()?;
        if self.ctx.legacy_loaded != 0 || self.native_graph_seen {
            return Err(failure("HIP native preparation needs a cold/unowned cache"));
        }
        self.ctx
            .native
            .register(key, StreamId { value: 0 }, NativeMode::SingleCompat)?;
        let result = self.prepare_registered_native(key, kernels, admission);
        if result.is_err() {
            // Only a known empty pre-compilation reservation may be removed.
            let _ = self.ctx.native.cancel_registration(key);
        }
        result
    }
    pub(crate) fn prepare_registered_native(
        &mut self,
        key: u64,
        kernels: Vec<Box<dyn CubeKernel>>,
        admission: Arc<dyn HipNativeAdmission>,
    ) -> Result<HipNativeInventory, ServerError> {
        self.require_known_storage_retirement()?;
        admission.require_usable()?;
        let scoped_metadata = if self.ctx.native.is_scoped() {
            Some(admission.reserve_scoped_metadata(
                crate::prepared_native::scoped_metadata_bytes(kernels.len())?,
            )?)
        } else {
            None
        };
        let originals = kernels.iter().map(|kernel| kernel.id()).collect::<Vec<_>>();
        let routed = self.ctx.native.catalogue(key, &originals)?;
        admission.require_usable()?;
        let context = admission.context_owner()?;
        match &self.ctx.native_context {
            Some(existing) if !existing.same(&context) => {
                return Err(failure("HIP original context charge differs"));
            }
            Some(_) => {}
            None => {
                self.ctx.native_context =
                    Some(crate::prepared_native::PersistentNativeOwner::new(context))
            }
        }
        let preparation = admission.reserve_preparation(kernels.len())?;
        self.ctx.native.install(
            key,
            PreparedModules {
                key,
                kernels: HashMap::with_capacity(kernels.len()),
                admission,
                inventory: HipNativeInventory::default(),
                sealed: false,
                failed: false,
                retired: false,
                _preparation: preparation,
                _scoped_metadata: scoped_metadata,
            },
            &originals,
            &routed,
        );
        for (kernel, id) in kernels.into_iter().zip(routed) {
            if let Err(error) = self
                .ctx
                .compile_kernel(&id, kernel, self.streams.logger.clone())
            {
                // No sibling may enqueue after uncertain compiler/load ownership.
                self.ctx.native.fence_all();
                return Err(ServerError::Launch(error));
            }
        }
        let prepared = self
            .ctx
            .native
            .set_mut(key)
            .expect("installed original set");
        if prepared.inventory.modules == 0 || prepared.failed {
            self.ctx.native.fence_all();
            return Err(failure("HIP empty/failed prepared native cache"));
        }
        prepared.sealed = true;
        Ok(prepared.inventory)
    }
    pub(crate) fn inspect_scoped_native(
        &self,
    ) -> Result<crate::prepared_native::HipScopedObservation, ServerError> {
        self.ctx.native.observation()
    }
    pub(crate) fn scoped_stream_usable(&self, logical: u64) -> Result<bool, ServerError> {
        if !self.ctx.native.is_scoped() {
            return Err(failure("HIP scoped observation mode differs"));
        }
        Ok(self
            .ctx
            .native
            .require_stream(StreamId { value: logical })
            .is_ok())
    }
    pub(crate) fn require_native(&mut self, key: u64) -> Result<(), ServerError> {
        self.require_known_storage_retirement()?;
        match self.ctx.native.set(key) {
            Some(p) if p.key == key && p.sealed && !p.failed => p.admission.require_usable(),
            _ => Err(failure("HIP native cache identity/state differs")),
        }
    }
    pub(crate) fn retire_native(&mut self, key: u64) -> Result<(), ServerError> {
        self.require_known_storage_retirement()?;
        let prepared = self
            .ctx
            .native
            .set_mut(key)
            .ok_or_else(|| failure("HIP native retirement identity differs"))?;
        prepared.admission.require_usable()?;
        // No queued future launch may use this cache after this actor operation.
        prepared.sealed = true;
        // SAFETY: original device actor, after its preceding submissions. This
        // drains every physical stream; it does not publish Ready/model state.
        let completion = unsafe {
            crate::compute::status::checked(
                "hipDeviceSynchronize",
                cubecl_hip_sys::hipDeviceSynchronize(),
            )
        };
        if let Err(error) = completion {
            self.ctx.native.fence_all();
            return Err(error.into());
        }
        for kernel in prepared.kernels.values_mut() {
            if kernel.module.is_null() {
                self.ctx.native.fence_all();
                return Err(failure(
                    "HIP unresolved native load retains its original charge",
                ));
            }
            // SAFETY: successful all-stream completion precedes unload. The
            // cache keeps its charge until every unload has actually succeeded.
            let result = unsafe {
                crate::compute::status::checked(
                    "hipModuleUnload",
                    cubecl_hip_sys::hipModuleUnload(kernel.module),
                )
            };
            if let Err(error) = result {
                self.ctx.native.fence_all();
                return Err(error.into());
            }
            kernel.module = std::ptr::null_mut();
        }
        for kernel in prepared.kernels.values_mut() {
            kernel.mark_native_unloaded();
        }
        // Retain the SAME prepared admission/source while deferred managed
        // handles and unpinned metadata pages retire on every actual stream.
        let streams = self.stream_ids();
        for stream in streams {
            if let Err(error) = self.memory_cleanup(stream) {
                self.ctx.native.fence_all();
                return Err(error);
            }
        }
        self.require_known_storage_retirement()?;
        self.ctx.native.set_mut(key).unwrap().retired = true;
        self.ctx.native.retire(key);
        Ok(())
    }
}
