//! Consume an existing host reservation; no allocator or source authenticator.
use super::*;
use crate::accelerator::SourceIdentity;
use std::any::Any;

/// One charged host allocation scope. Non-Clone: separate consumers cannot
/// spend this reservation twice. Share it only within the admitted allocation.
pub struct HostWorkspace {
    // Release the actual source owner before returning its host reservation.
    _source_owner: Arc<dyn Any + Send + Sync>,
    source: SourceIdentity,
    bytes: usize,
    _lease: GlobalResourceLease,
}
impl HostWorkspace {
    pub fn source_identity(&self) -> SourceIdentity {
        self.source
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl GlobalResourceLease {
    /// Retain the original source owner under this existing host charge. The
    /// source adapter establishes authenticity; this only checks admission and
    /// source binding metadata. No caller-supplied byte count establishes fit.
    pub fn into_host_workspace<T: Any + Send + Sync>(
        self,
        source: SourceIdentity,
        source_owner: Arc<T>,
    ) -> Result<HostWorkspace> {
        if !source.is_bound() || self.device.backend != "host" || self.released {
            return Err(Error::State(
                "host workspace source/resource binding".into(),
            ));
        }
        let bytes = {
            let state = self
                .shared
                .lock()
                .map_err(|_| Error::State("host workspace arbiter poisoned".into()))?;
            let budget = state
                .budgets
                .get(&self.device)
                .ok_or_else(|| Error::State("host workspace resource absent".into()))?;
            let reservation = budget
                .reservations
                .get(&self.id)
                .ok_or_else(|| Error::State("host workspace reservation absent".into()))?;
            if budget.fenced || reservation.phase != LeasePhase::Reserved {
                return Err(Error::State(
                    "host workspace reservation is not idle".into(),
                ));
            }
            usize::try_from(reservation.bytes)
                .map_err(|_| Error::Limit("host workspace byte extent".into()))?
        };
        Ok(HostWorkspace {
            _source_owner: source_owner,
            source,
            bytes,
            _lease: self,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn admission(backend: &str) -> (ResourceArbiter, PhysicalDeviceKey, GlobalResourceLease) {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new(backend, "existing-domain").unwrap();
        arbiter.register(key.clone(), 4096).unwrap();
        let lease = arbiter
            .admit(
                &key,
                ResourceOwner::new("actual-source", "metadata").unwrap(),
                &[AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 3072,
                }],
            )
            .unwrap();
        (arbiter, key, lease)
    }

    #[test]
    fn transferred_charge_keeps_original_bytes_and_owner_until_last_shared_use() {
        let (arbiter, key, lease) = admission("host");
        let source = SourceIdentity::for_new_allocation().unwrap();
        let owner = Arc::new(vec![1u8, 2, 3]);
        let witness = Arc::downgrade(&owner);
        let workspace = Arc::new(lease.into_host_workspace(source, owner).unwrap());
        assert_eq!(workspace.bytes(), 3072);
        assert_eq!(workspace.source_identity(), source);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 3072);
        let reader = workspace.clone();
        drop(workspace);
        assert!(witness.upgrade().is_some());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 3072);
        drop(reader);
        assert!(witness.upgrade().is_none());
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }

    #[test]
    fn non_host_or_unbound_metadata_cannot_relabel_a_reservation() {
        for (backend, source) in [
            ("cuda", SourceIdentity::for_new_allocation().unwrap()),
            ("host", SourceIdentity::unbound()),
        ] {
            let (arbiter, key, lease) = admission(backend);
            assert!(lease.into_host_workspace(source, Arc::new(())).is_err());
            assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
        }
    }

    #[test]
    fn submitted_reservation_cannot_become_idle_host_metadata() {
        let (arbiter, key, lease) = admission("host");
        lease.mark_submitted().unwrap();
        assert!(
            lease
                .into_host_workspace(SourceIdentity::for_new_allocation().unwrap(), Arc::new(()))
                .is_err()
        );
        let snapshot = arbiter.snapshot(&key).unwrap();
        assert!(snapshot.fenced);
        assert_eq!(snapshot.reserved, 3072);
    }
}
