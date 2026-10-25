//! Model-backed fixture HOST policy and independently priced permanent baseline.
use std::{collections::BTreeMap, ops::Deref};
use zerogravity::{
    Error, Result,
    cache::TransferAdmission,
    resource_arbiter::{
        AllocationClass, DeviceBudgetSnapshot, PhysicalDeviceKey, ResourceOwner,
        lease_actual_inventory_bytes,
    },
    short_hf,
};

const MIB: u64 = 1024 * 1024;
const BOOTSTRAP_SESSION: &str = "short-hf-fixture-bootstrap";

pub(super) struct FixtureHost {
    admission: TransferAdmission,
    baseline: DeviceBudgetSnapshot,
    declared_host_bytes: u64,
}

impl Deref for FixtureHost {
    type Target = TransferAdmission;

    fn deref(&self) -> &Self::Target {
        &self.admission
    }
}

fn ensure(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::State(message.into()))
    }
}

fn same(a: &DeviceBudgetSnapshot, b: &DeviceBudgetSnapshot) -> bool {
    a.limit == b.limit
        && a.reserved == b.reserved
        && a.remaining == b.remaining
        && a.reservations == b.reservations
        && a.fenced == b.fenced
        && a.per_class == b.per_class
        && a.active_models == b.active_models
}

/// Only these model-backed fixtures select the declared finite 8GiB policy.
/// The unchanged loader authenticates full source and inventory fit afterward.
pub(super) fn declared_limit(host: &TransferAdmission) -> Result<u64> {
    let limit = short_hf::HOST_MIB_MAX as u64 * MIB;
    ensure(
        host.arbiter.is_process_shared()
            && host.resource == PhysicalDeviceKey::new("host", "zerogravity-serve-cpu")?
            && host.arbiter.snapshot(&host.resource)?.limit == limit,
        "model fixture original process HOST domain and declared8GiB policy",
    )?;
    Ok(limit)
}

impl FixtureHost {
    pub(super) fn new(owner: ResourceOwner) -> Result<Self> {
        let admission = short_hf::host_admission_mib(short_hf::HOST_MIB_MAX, owner)?;
        let limit = declared_limit(&admission)?;
        let empty = DeviceBudgetSnapshot {
            limit,
            reserved: 0,
            remaining: limit,
            reservations: 0,
            fenced: false,
            per_class: BTreeMap::new(),
            active_models: Default::default(),
        };
        ensure(
            same(&empty, &admission.arbiter.snapshot(&admission.resource)?),
            "fresh model fixture HOST before reference/tokenizer/model owners",
        )?;
        let bootstrap =
            zerogravity::tokenize_admission::approve_arbiter(&admission, BOOTSTRAP_SESSION)?;
        bootstrap.validate_domain(&admission)?;
        let plan = bootstrap.plan();
        let owner = ResourceOwner::new(&admission.owner.model, BOOTSTRAP_SESSION)?;
        let inventory = lease_actual_inventory_bytes(&admission.resource, &owner, 2)?;
        let reserved = plan
            .bootstrap_bytes
            .checked_add(inventory)
            .ok_or_else(|| Error::Limit("fixture bootstrap baseline sum".into()))?;
        let baseline = DeviceBudgetSnapshot {
            limit,
            reserved,
            remaining: limit
                .checked_sub(reserved)
                .ok_or_else(|| Error::Limit("fixture bootstrap baseline remaining".into()))?,
            reservations: 1,
            fenced: false,
            per_class: BTreeMap::from([
                (AllocationClass::ImmutableWeights, plan.bootstrap_bytes),
                (AllocationClass::Other, inventory),
            ]),
            active_models: [admission.owner.model.clone()].into(),
        };
        let actual = admission.arbiter.snapshot(&admission.resource)?;
        if !same(&baseline, &actual) {
            return Err(Error::State(format!(
                "fixture tokenizer bootstrap baseline differs: expected={baseline:?} actual={actual:?}"
            )));
        }
        eprintln!("SHORT_HF_FIXTURE_HOST_BASELINE {baseline:?}");
        Ok(Self {
            admission,
            baseline,
            declared_host_bytes: limit,
        })
    }

    fn after_backend(&self, backend_bytes: u64) -> Result<DeviceBudgetSnapshot> {
        ensure(
            backend_bytes == 384 * MIB,
            "fixture exact original CUDA backend HOST384MiB",
        )?;
        ensure(
            declared_limit(self)? == self.declared_host_bytes,
            "fixture authenticated HOST declaration unchanged",
        )?;
        let mut expected = self.baseline.clone();
        expected.reserved = expected
            .reserved
            .checked_add(backend_bytes)
            .ok_or_else(|| Error::Limit("fixture backend baseline sum".into()))?;
        expected.remaining = expected
            .limit
            .checked_sub(expected.reserved)
            .ok_or_else(|| Error::Limit("fixture backend baseline remaining".into()))?;
        expected.reservations += 1;
        expected
            .per_class
            .insert(AllocationClass::Scratch, backend_bytes);
        expected.active_models.insert("backend/cuda".into());
        Ok(expected)
    }

    pub(super) fn require_final(&self, backend_bytes: u64) -> Result<()> {
        let expected = self.after_backend(backend_bytes)?;
        let actual = self.arbiter.snapshot(&self.resource)?;
        if !same(&expected, &actual) {
            return Err(Error::State(format!(
                "fixture final HOST differs from exact tokenizer bootstrap+backend: expected={expected:?} actual={actual:?}"
            )));
        }
        Ok(())
    }

    pub(super) fn require_retained(&self, backend_bytes: u64) -> Result<()> {
        let baseline = self.after_backend(backend_bytes)?;
        let actual = self.arbiter.snapshot(&self.resource)?;
        ensure(
            actual.limit == self.declared_host_bytes
                && actual.reserved > baseline.reserved
                && actual.reserved <= actual.limit
                && actual.remaining == actual.limit - actual.reserved
                && actual.reservations > baseline.reservations
                && !actual.fenced
                && actual.active_models == baseline.active_models
                && actual
                    .per_class
                    .get(&AllocationClass::Scratch)
                    .copied()
                    .unwrap_or(0)
                    > backend_bytes
                && actual.per_class.get(&AllocationClass::ImmutableWeights)
                    >= baseline.per_class.get(&AllocationClass::ImmutableWeights)
                && actual.per_class.get(&AllocationClass::Other)
                    >= baseline.per_class.get(&AllocationClass::Other)
                && actual.per_class.keys().all(|class| {
                    matches!(
                        class,
                        AllocationClass::ImmutableWeights
                            | AllocationClass::Scratch
                            | AllocationClass::Other
                            | AllocationClass::TransferStaging
                            | AllocationClass::Output
                    )
                })
                && actual
                    .per_class
                    .values()
                    .try_fold(0u64, |sum, bytes| sum.checked_add(*bytes))
                    == Some(actual.reserved),
            "fault retains real source HOST beyond exact bootstrap+backend baseline",
        )
    }
}
