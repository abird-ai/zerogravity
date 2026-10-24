//! Exact source-plan ValueId→original physical BF16 lease sealing.
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_program::{DeviceBuffer, ResidentBufferLayout, ResidentPhysicalType};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_core::accelerator::program::{
    ActivationPrecision, RecordedVllm08Model, ValueId, ValueShape,
};
use zerogravity_core::device_events::{BufferLeaseId, BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

pub(crate) mod bounded;

pub(crate) struct ModelValues {
    record: Arc<RecordedVllm08Model>,
    values: Vec<ResidentLease>,
    service: Arc<ResidentService>,
    status: ResidentLease,
    storage: Option<Arc<bounded::BoundedStorage>>,
}
impl ModelValues {
    /// Caller is the original ResidentProgram workspace allocator. This seal
    /// cannot manufacture source rows, a RootAttempt or a completion permit.
    pub(crate) fn bind(
        record: Arc<RecordedVllm08Model>,
        values: &[DeviceBuffer],
        service: Arc<ResidentService>,
        status: ResidentLease,
    ) -> Result<Arc<Self>> {
        Self::bind_count(record, values, service, status, 327)
    }
    pub(crate) fn bind_canonical47_prefix(
        cut: &Arc<super::canonical47_model_prefix::Canonical47PrefixCut>,
        values: &[DeviceBuffer],
        service: Arc<ResidentService>,
        status: ResidentLease,
    ) -> Result<Arc<Self>> {
        let count = cut.kind().values();
        if cut.values().len() != count || values.len() != count {
            return Err(Error::State(
                "canonical47 exact39 actual source value bindings".into(),
            ));
        }
        Self::bind_count(cut.scope().record().clone(), values, service, status, count)
    }
    fn bind_count(
        record: Arc<RecordedVllm08Model>,
        values: &[DeviceBuffer],
        service: Arc<ResidentService>,
        status: ResidentLease,
        count: usize,
    ) -> Result<Arc<Self>> {
        record.call().validate()?;
        let plan = record.plan();
        if plan.values().len() != 327
            || values.len() != count
            || plan.source() != service.source_identity()
            || plan.source() != record.call().source()
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
        {
            return Err(Error::State(
                "full24 original unpublished plan/value/source coverage".into(),
            ));
        }
        if !Arc::ptr_eq(status.service(), &service)
            || status.source_identity() != plan.source()
            || status.service_generation() != service.generation()
            || status.role() != BufferRole::Scratch
            || status.bytes() != 4
        {
            return Err(Error::State(
                "full24 original attempt control lease/source/service".into(),
            ));
        }
        let mut originals = BTreeMap::<BufferLeaseId, ValueId>::new();
        let mut sealed = Vec::with_capacity(values.len());
        for (index, (buffer, shape)) in values.iter().zip(plan.values()).enumerate() {
            let ValueShape::Vector(elements) = *shape else {
                return Err(Error::State(
                    "full24 physical BF16 dense vector only".into(),
                ));
            };
            let id = ValueId(index as u32);
            if plan.precisions().get(index) != Some(&ActivationPrecision::Bf16WidenedF32) {
                return Err(Error::State(
                    "full24 physical precision differs from plan".into(),
                ));
            }
            plan.vllm_value_provenance(id)?
                .ok_or_else(|| Error::State("full24 real teacher provenance absent".into()))?
                .require_invocation(record.call().request())?;
            buffer.require_layout(ResidentBufferLayout::new(
                ResidentPhysicalType::Bf16,
                elements,
            )?)?;
            let lease = buffer.original_lease()?;
            if !Arc::ptr_eq(lease.service(), &service)
                || lease.source_identity() != plan.source()
                || lease.service_generation() != service.generation()
                || lease.bytes()
                    != elements.checked_mul(2).ok_or_else(|| {
                        Error::Limit("full24 physical BF16 extent overflow".into())
                    })?
                || !matches!(
                    lease.role(),
                    BufferRole::Activation | BufferRole::Output | BufferRole::Scratch
                )
                || lease.id() == status.id()
                || originals.insert(lease.id(), id).is_some()
            {
                return Err(Error::State(
                    "full24 distinct original producer lease/service/extent".into(),
                ));
            }
            sealed.push(lease.clone());
        }
        Ok(Arc::new(Self {
            record,
            values: sealed,
            service,
            status,
            storage: None,
        }))
    }
    pub(crate) fn record(&self) -> &Arc<RecordedVllm08Model> {
        &self.record
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        &self.status
    }
    pub(crate) fn lease(&self, value: ValueId) -> Result<&ResidentLease> {
        self.values
            .get(value.0 as usize)
            .ok_or_else(|| Error::State("full24 sealed original ValueId absent".into()))
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        if self.storage.is_some() {
            // One original physical owner per slot; logical aliases do not
            // multiply flight events, admission or quarantine handles.
            let mut unique = BTreeMap::new();
            for lease in self.values.iter().chain([&self.status]) {
                unique
                    .entry(lease.id())
                    .or_insert((lease.clone(), DeviceAccess::Write));
            }
            return unique.into_values().collect();
        }
        // Original plan may read a producer later in the SAME flight. Write
        // dominance covers the actual producer; no historical input is invented.
        self.values
            .iter()
            .map(|lease| (lease.clone(), DeviceAccess::Write))
            .chain([(self.status.clone(), DeviceAccess::Write)])
            .collect()
    }
}
