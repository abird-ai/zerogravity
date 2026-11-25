//! Reserve the original physical envelope before client bootstrap.
use super::*;

/// Physical device admission taken before the CubeCL Client is constructed. It
/// carries the one original lease into `PhysicalServiceBudget` without a
/// second charge.
pub(crate) struct PreAdmittedDevice {
    key: PhysicalDeviceKey,
    reported_max: u64,
    operator_limit: u64,
    owner_model: String,
    armed: bool,
    lease: Arc<GlobalResourceLease>,
    fence_flag: Arc<AtomicBool>,
    in_flight: Arc<Mutex<usize>>,
}

impl PreAdmittedDevice {
    /// Bootstrap uncertainty: retain the charge and fence the card instead of
    /// crediting an admission whose context may have been constructed.
    fn fence_and_retain(&self) {
        self.lease.fence_unknown();
        self.fence_flag.store(true, Ordering::Release);
        std::mem::forget(self.lease.clone());
        std::mem::forget(self.fence_flag.clone());
    }

    /// Any error or unwind after a bootstrap starts retains the original
    /// envelope. A refusal before this point drops only unused admission.
    pub(crate) fn bootstrap<T>(mut self, construct: impl FnOnce(&Self) -> Result<T>) -> Result<T> {
        self.armed = true;
        let result = construct(&self);
        if result.is_ok() {
            self.armed = false;
        }
        result
    }
}

impl Drop for PreAdmittedDevice {
    fn drop(&mut self) {
        if self.armed {
            self.fence_and_retain();
        }
    }
}

impl PhysicalServiceBudget {
    /// Admit the selected adapter's actual identity/device-local capacity
    /// before `device.client()` or native context construction.
    pub(crate) fn preadmit_device(
        key: PhysicalDeviceKey,
        reported_max: u64,
        operator_device_cap: u64,
        provider_cap: u64,
        owner: ResourceOwner,
    ) -> Result<PreAdmittedDevice> {
        checked_caps(Some(reported_max), operator_device_cap, provider_cap)?;
        let arbiter = ResourceArbiter::process_shared();
        arbiter.register_matching(key.clone(), operator_device_cap)?;
        let lease = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes: provider_cap,
            }],
        )?;
        Ok(PreAdmittedDevice {
            key,
            reported_max,
            operator_limit: operator_device_cap,
            owner_model: owner.model,
            armed: false,
            lease: Arc::new(lease),
            fence_flag: Arc::new(AtomicBool::new(false)),
            in_flight: Arc::new(Mutex::new(0)),
        })
    }

    /// The selected device's observed local capacity and live availability
    /// constrain the same original reservation. Neither is another pool.
    #[cfg(any(
        feature = "hip",
        all(feature = "vulkan", feature = "gemma-bf16-qualification")
    ))]
    pub(crate) fn preadmit_available_device(
        key: PhysicalDeviceKey,
        ceiling: u64,
        available: u64,
        operator_device_cap: u64,
        provider_cap: u64,
        owner: ResourceOwner,
    ) -> Result<PreAdmittedDevice> {
        checked_caps(Some(ceiling), operator_device_cap, provider_cap)?;
        if provider_cap > available {
            return Err(Error::resource(
                "device-local VRAM",
                "cold provider envelope",
                provider_cap,
                available,
            ));
        }
        Self::preadmit_device(key, ceiling, operator_device_cap, provider_cap, owner)
    }

    /// Attach the already-admitted physical charge to the actual client. The
    /// client's selected identity and capacity must match the preflight; no
    /// second arbiter admission is made.
    pub(crate) fn attach_preadmitted(
        client: &Client,
        pre: &PreAdmittedDevice,
        operator_device_cap: u64,
        provider_cap: u64,
    ) -> Result<Self> {
        let properties = client.properties();
        let key = physical_key(properties.identity.physical.as_ref())?;
        if !pre.armed
            || key != pre.key
            || operator_device_cap != pre.operator_limit
            || provider_cap != pre.lease.reserved_bytes()?
            || properties
                .memory
                .max_memory()
                .is_none_or(|n| n < pre.reported_max || n < operator_device_cap)
            || operator_device_cap == 0
            || provider_cap == 0
            || provider_cap > operator_device_cap
        {
            return Err(Error::State(
                "pre-admitted physical identity/capacity changed before context".into(),
            ));
        }
        crate::physical_quarantine::ensure_service_safe(client)?;
        Ok(Self {
            client: client.clone(),
            key: pre.key.clone(),
            total_limit: operator_device_cap,
            provider_cap,
            owner_model: pre.owner_model.clone(),
            model_closed: None,
            lease: pre.lease.clone(),
            fence_flag: pre.fence_flag.clone(),
            in_flight: pre.in_flight.clone(),
        })
    }
    /// Release only unused cold credit after actual live memory observation.
    /// The provider retains its original reservation/identity; no
    /// retire-and-readmit race or work in flight is permitted.
    #[cfg(all(
        feature = "cuda",
        any(feature = "qualification", feature = "cuda-native")
    ))]
    pub(crate) fn narrow_unused_provider(&mut self, effective: u64) -> Result<()> {
        if effective == 0 || effective > self.provider_cap {
            return Err(Error::Limit("invalid cold provider narrowing".into()));
        }
        if effective == self.provider_cap {
            return Ok(());
        }
        let lease = Arc::get_mut(&mut self.lease).ok_or_else(|| {
            Error::State("cold provider narrowing requires exclusive envelope".into())
        })?;
        let unused = lease.partition_reserved(
            ResourceOwner::new("backend/physical", "unused-cold-capacity")?,
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes: self.provider_cap - effective,
            }],
        )?;
        unused.release()?;
        self.provider_cap = effective;
        Ok(())
    }
}
