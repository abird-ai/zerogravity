//! Transfer part of one idle admission; no new physical capacity or authority.
use super::*;

impl GlobalResourceLease {
    /// Split an exact positive portion from this original Reserved lease into a
    /// distinct lifetime/owner on the SAME physical domain and arbiter.
    ///
    /// Global committed bytes and per-class sums stay unchanged atomically. The
    /// remainder keeps this lease's identity/owner. Failure changes neither.
    /// Classes cannot be relabeled, the full lease cannot be extracted, and any
    /// prior submission or device fence refuses partition. Requires exclusive
    /// mutable ownership before Arc sharing; creates no device/Ready/source proof.
    pub fn partition_reserved(
        &mut self,
        owner: ResourceOwner,
        allocations: &[AllocationRequest],
    ) -> Result<GlobalResourceLease> {
        if self.released
            || owner.model.is_empty()
            || owner.session.is_empty()
            || allocations.is_empty()
        {
            return Err(Error::State(
                "reserved lease partition owner/request".into(),
            ));
        }
        let mut requested = BTreeMap::<AllocationClass, u64>::new();
        let mut bytes = 0u64;
        for part in allocations {
            if part.bytes == 0 {
                return Err(Error::Limit("reserved lease partition zero extent".into()));
            }
            bytes = bytes
                .checked_add(part.bytes)
                .ok_or_else(|| Error::Limit("reserved lease partition byte overflow".into()))?;
            let total = requested.entry(part.kind).or_default();
            *total = total
                .checked_add(part.bytes)
                .ok_or_else(|| Error::Limit("reserved lease partition class overflow".into()))?;
        }
        let child_allocations = allocations.to_vec();
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("reserved lease partition arbiter poisoned".into()))?;
        let id = state.next_id;
        let next_id = id
            .checked_add(1)
            .ok_or_else(|| Error::Limit("resource reservation ID exhausted".into()))?;
        let (remaining, retained_allocations) = {
            let budget = state
                .budgets
                .get(&self.device)
                .ok_or_else(|| Error::State("reserved lease partition device absent".into()))?;
            let original = budget.reservations.get(&self.id).ok_or_else(|| {
                Error::State("reserved lease partition reservation absent".into())
            })?;
            if budget.fenced || original.phase != LeasePhase::Reserved {
                return Err(Error::State(
                    "reserved lease partition requires unfenced unused admission".into(),
                ));
            }
            let remaining = original
                .bytes
                .checked_sub(bytes)
                .filter(|n| *n != 0)
                .ok_or_else(|| {
                    Error::Limit("reserved lease partition must be strictly partial".into())
                })?;
            let mut retained = BTreeMap::<AllocationClass, u64>::new();
            for part in &original.allocations {
                let total = retained.entry(part.kind).or_default();
                *total = total
                    .checked_add(part.bytes)
                    .ok_or_else(|| Error::State("original resource class sum overflow".into()))?;
            }
            for (kind, taken) in requested {
                let available = retained
                    .get_mut(&kind)
                    .ok_or_else(|| Error::Limit("reserved lease partition class absent".into()))?;
                *available = available.checked_sub(taken).ok_or_else(|| {
                    Error::Limit("reserved lease partition class exceeds original".into())
                })?;
            }
            let retained = retained
                .into_iter()
                .filter(|(_, n)| *n != 0)
                .map(|(kind, bytes)| AllocationRequest { kind, bytes })
                .collect::<Vec<_>>();
            (remaining, retained)
        };
        // Every fallible check completed. No retire/re-admit gap exists and no
        // resource-owning value drops while this original arbiter lock is held.
        state.next_id = next_id;
        let budget = state
            .budgets
            .get_mut(&self.device)
            .expect("validated original physical domain");
        let original = budget
            .reservations
            .get_mut(&self.id)
            .expect("validated original reservation");
        original.bytes = remaining;
        original.allocations = retained_allocations;
        budget.reservations.insert(
            id,
            Reservation {
                owner,
                bytes,
                allocations: child_allocations,
                phase: LeasePhase::Reserved,
            },
        );
        let child = GlobalResourceLease {
            shared: self.shared.clone(),
            device: self.device.clone(),
            id,
            released: false,
        };
        drop(state);
        Ok(child)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (ResourceArbiter, PhysicalDeviceKey, GlobalResourceLease) {
        let arbiter = ResourceArbiter::new();
        let key = PhysicalDeviceKey::new("host", "original-partition-domain").unwrap();
        arbiter.register(key.clone(), 128).unwrap();
        let lease = arbiter
            .admit(
                &key,
                ResourceOwner::new("model", "source-provider").unwrap(),
                &[
                    AllocationRequest {
                        kind: AllocationClass::ProviderEnvelope,
                        bytes: 96,
                    },
                    AllocationRequest {
                        kind: AllocationClass::Scratch,
                        bytes: 32,
                    },
                ],
            )
            .unwrap();
        (arbiter, key, lease)
    }
    fn backend() -> ResourceOwner {
        ResourceOwner::new("backend/hip", "original-context").unwrap()
    }
    fn part(bytes: u64) -> AllocationRequest {
        AllocationRequest {
            kind: AllocationClass::ProviderEnvelope,
            bytes,
        }
    }
    fn unchanged(a: &DeviceBudgetSnapshot, b: &DeviceBudgetSnapshot) {
        assert_eq!(a.limit, b.limit);
        assert_eq!(a.reserved, b.reserved);
        assert_eq!(a.remaining, b.remaining);
        assert_eq!(a.reservations, b.reservations);
        assert_eq!(a.fenced, b.fenced);
        assert_eq!(a.per_class, b.per_class);
        assert_eq!(a.active_models, b.active_models);
    }
    #[test]
    fn context_slice_survives_model_drop_without_second_admission() {
        let (arbiter, key, mut model) = fixture();
        let before = arbiter.snapshot(&key).unwrap();
        let context = model.partition_reserved(backend(), &[part(32)]).unwrap();
        let split = arbiter.snapshot(&key).unwrap();
        assert_eq!(split.reserved, before.reserved);
        assert_eq!(split.remaining, 0);
        assert_eq!(split.per_class, before.per_class);
        assert_eq!(split.reservations, 2);
        assert!(split.active_models.contains("model"));
        assert!(split.active_models.contains("backend/hip"));
        // Independent model/module retirement does not release persistent context.
        drop(model);
        let held = arbiter.snapshot(&key).unwrap();
        assert_eq!(held.reserved, 32);
        assert_eq!(held.remaining, 96);
        assert_eq!(
            held.active_models,
            BTreeSet::from(["backend/hip".to_string()])
        );
        // This host-only fixture is genuinely idle; real backend lifetime owner
        // may drop its slice only after actual context/process retirement.
        drop(context);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    #[test]
    fn invalid_partition_preserves_original_identity_owner_and_capacity() {
        let (arbiter, key, mut model) = fixture();
        let id = model.id;
        let before = arbiter.snapshot(&key).unwrap();
        let invalid = [
            vec![],
            vec![part(0)],
            vec![part(129)],
            vec![
                part(96),
                AllocationRequest {
                    kind: AllocationClass::Scratch,
                    bytes: 32,
                },
            ],
            vec![AllocationRequest {
                kind: AllocationClass::ImmutableWeights,
                bytes: 1,
            }],
            vec![part(97)],
            vec![part(u64::MAX), part(1)],
        ];
        for allocations in invalid {
            assert!(model.partition_reserved(backend(), &allocations).is_err());
            assert_eq!(model.id, id);
            unchanged(&before, &arbiter.snapshot(&key).unwrap());
        }
        assert!(
            model
                .partition_reserved(
                    ResourceOwner {
                        model: String::new(),
                        session: "context".into()
                    },
                    &[part(1)]
                )
                .is_err()
        );
        unchanged(&before, &arbiter.snapshot(&key).unwrap());
        drop(model);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
    #[test]
    fn submitted_ready_or_unknown_work_cannot_transfer_ownership() {
        for unknown in [false, true] {
            let (arbiter, key, mut model) = fixture();
            model.mark_submitted().unwrap();
            let submitted = arbiter.snapshot(&key).unwrap();
            assert!(model.partition_reserved(backend(), &[part(32)]).is_err());
            unchanged(&submitted, &arbiter.snapshot(&key).unwrap());
            if unknown {
                model.fence_unknown();
            } else {
                model.mark_ready().unwrap();
            }
            let before = arbiter.snapshot(&key).unwrap();
            assert!(model.partition_reserved(backend(), &[part(32)]).is_err());
            unchanged(&before, &arbiter.snapshot(&key).unwrap());
            drop(model);
            assert_eq!(
                arbiter.snapshot(&key).unwrap().reserved,
                if unknown { 128 } else { 0 }
            );
        }
    }
    #[test]
    fn fenced_or_exhausted_arbiter_refuses_without_releasing_original() {
        let (arbiter, key, mut model) = fixture();
        arbiter.shared.lock().unwrap().next_id = u64::MAX;
        let before = arbiter.snapshot(&key).unwrap();
        assert!(model.partition_reserved(backend(), &[part(32)]).is_err());
        unchanged(&before, &arbiter.snapshot(&key).unwrap());
        arbiter.shared.lock().unwrap().next_id = 2;
        arbiter.fence_unknown_device(&key).unwrap();
        let before = arbiter.snapshot(&key).unwrap();
        assert!(model.partition_reserved(backend(), &[part(32)]).is_err());
        unchanged(&before, &arbiter.snapshot(&key).unwrap());
    }
    #[test]
    fn concurrent_admission_cannot_spend_a_partition_gap() {
        use std::sync::Barrier;
        let (arbiter, key, mut model) = fixture();
        let start = Arc::new(Barrier::new(3));
        let done = Arc::new(Barrier::new(3));
        let (model, context) = std::thread::scope(|scope| {
            let split = scope.spawn(|| {
                start.wait();
                let context = model.partition_reserved(backend(), &[part(32)]).unwrap();
                done.wait();
                (model, context)
            });
            let contender = scope.spawn(|| {
                start.wait();
                let result = arbiter.admit(
                    &key,
                    ResourceOwner::new("other", "concurrent").unwrap(),
                    &[part(1)],
                );
                done.wait();
                result
            });
            start.wait();
            done.wait();
            assert!(contender.join().unwrap().is_err());
            split.join().unwrap()
        });
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 128);
        drop(model);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 32);
        drop(context);
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
}
