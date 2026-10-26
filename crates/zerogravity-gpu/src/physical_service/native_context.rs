//! Persistent backend slice has no source/model/Client/cache backedge.
use super::*;
use cubecl_common::device::ServiceId;

pub(crate) struct NativeContextBudget {
    envelope: Arc<GlobalResourceLease>,
    physical: PhysicalDeviceKey,
    service: ServiceId,
    fence: Arc<AtomicBool>,
    policy: Arc<str>,
    bytes: u64,
}
impl NativeContextBudget {
    #[cfg(any(test, feature = "hip", feature = "vulkan"))]
    pub(super) fn partition(
        original: &mut GlobalResourceLease,
        physical: PhysicalDeviceKey,
        service: ServiceId,
        fence: Arc<AtomicBool>,
        backend: &'static str,
        policy: Arc<str>,
        bytes: u64,
    ) -> Result<Arc<Self>> {
        if policy.is_empty() || fence.load(Ordering::Acquire) {
            return Err(Error::State("native context policy/fence".into()));
        }
        let child = original.partition_reserved(
            ResourceOwner::new(backend, &format!("context/{policy}"))?,
            &[AllocationRequest {
                kind: AllocationClass::ProviderEnvelope,
                bytes,
            }],
        )?;
        Ok(Arc::new(Self {
            envelope: Arc::new(child),
            physical,
            service,
            fence,
            policy,
            bytes,
        }))
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    ))]
    pub(crate) fn require(
        &self,
        physical: &PhysicalDeviceKey,
        service: ServiceId,
        policy: &str,
    ) -> Result<()> {
        if physical != &self.physical
            || service != self.service
            || policy != self.policy.as_ref()
            || self.fence.load(Ordering::Acquire)
            || ResourceArbiter::process_shared().is_fenced(physical)?
        {
            return Err(Error::State(
                "original native context identity/policy/fence differs".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }
    #[cfg(all(
        target_os = "linux",
        any(feature = "hip", feature = "vulkan"),
        any(
            feature = "gemma-bf16-qualification",
            feature = "qwen-source-qualification",
            all(test, feature = "qualification", feature = "hip")
        )
    ))]
    pub(crate) fn fence_unknown(&self) {
        self.envelope.fence_unknown();
        self.fence.store(true, Ordering::Release);
    }
}

impl Drop for NativeContextBudget {
    fn drop(&mut self) {
        // Client/context bootstrap can already have happened before partition.
        // No later model failure or module unload proves primary-context death.
        // This initial policy deliberately has process lifetime; it never keeps
        // the source/model remainder alive and introduces no actor backedge.
        std::mem::forget(self.envelope.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_common::device::DeviceId;
    #[test]
    fn persistent_slice_is_source_neutral_and_does_not_keep_model_remainder() {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("host", "native-context-owner").unwrap();
        arbiter.register(key.clone(), 128).unwrap();
        let mut model = arbiter
            .admit(
                &key,
                ResourceOwner::new("source", "model").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::ProviderEnvelope,
                    bytes: 128,
                }],
            )
            .unwrap();
        let context = NativeContextBudget::partition(
            &mut model,
            key.clone(),
            ServiceId::of::<()>(DeviceId::new(0, 0)),
            Arc::new(AtomicBool::new(false)),
            "backend/hip",
            Arc::from("host-fixture"),
            32,
        )
        .unwrap();
        assert_eq!(context.bytes(), 32);
        assert_eq!(context.physical, key);
        assert_eq!(context.service, ServiceId::of::<()>(DeviceId::new(0, 0)));
        assert_eq!(context.policy.as_ref(), "host-fixture");
        assert!(!context.fence.load(Ordering::Acquire));
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 128);
        drop(model);
        let retained = arbiter.snapshot(&key).unwrap();
        assert_eq!(retained.reserved, 32);
        assert_eq!(
            retained.active_models,
            std::collections::BTreeSet::from(["backend/hip".into()])
        );
        // Even failure before actor attachment retains the source-neutral slice.
        // Process retirement is the sole established destruction boundary here.
        drop(context);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 32);
    }
}
