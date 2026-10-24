//! One physical B axis over genuine per-request semantic ValueIds.
use super::super::*;
use super::bindings::CohortBindings;
use crate::resident_leases::backing_pool::{ResidentWorkspaceGeometry, ResidentWorkspaceRun};
use crate::resident_leases::{ResidentLease, ResidentService};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{ActivationPrecision, RecordedVllm08Model};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

pub(crate) struct CohortValues {
    records: Box<[Arc<RecordedVllm08Model>]>,
    _source: Arc<zerogravity_model_qwen35::original_binding::OriginalWindowModelSource>,
    service: Arc<ResidentService>,
    geometry: Arc<ResidentWorkspaceGeometry>,
    _run: Arc<ResidentWorkspaceRun>,
    buffers: Vec<DeviceBuffer>,
    _metadata: WindowWorkspace,
}
impl CohortValues {
    /// Conservative327 dedicated value views. No liveness alias/reuse is
    /// inferred from B or borrowed from the six original N1 geometries.
    pub(super) fn bind(
        binding: &CohortBindings,
        service: Arc<ResidentService>,
        geometry: Arc<ResidentWorkspaceGeometry>,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        let records = binding.records().into_boxed_slice();
        let plan = records[0].plan();
        let width = binding.width();
        if geometry.leases().len() != 327
            || plan.values().len() != 327
            || plan.precisions().len() != 327
            || records.iter().any(|record| {
                record.plan().values() != plan.values()
                    || record.plan().precisions() != plan.precisions()
                    || record.call().source() != service.source_identity()
            })
            || plan
                .precisions()
                .iter()
                .any(|precision| *precision != ActivationPrecision::Bf16WidenedF32)
        {
            return Err(Error::State(
                "cohort actual327 logical BF16 source values".into(),
            ));
        }
        let metadata = records[0].call().source();
        if metadata != service.source_identity() {
            return Err(Error::State("cohort source/service identity".into()));
        }
        // Source admission belongs to the actual binding, not an independent
        // host allocator. Immutable records retain that same original owner.
        let _metadata = binding.source().source().reserve_workspace(128 * 1024)?;
        let mut buffers = Vec::with_capacity(327);
        let mut ranges = Vec::with_capacity(327);
        for (shape, lease) in plan.values().iter().zip(geometry.leases()) {
            let row_elements = shape.capacity_elements()?;
            let elements = row_elements
                .checked_mul(width)
                .ok_or_else(|| Error::Limit("cohort physical BF16 extent overflow".into()))?;
            let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, elements)?;
            if !Arc::ptr_eq(lease.service(), &service)
                || lease.bytes() != layout.bytes()?
                || lease.role() != BufferRole::Activation
            {
                return Err(Error::State(
                    "cohort exact B workspace producer lease".into(),
                ));
            }
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(layout.bytes()?)?;
            if ranges
                .iter()
                .any(|prior: &BufferRange| range.overlaps(prior))
            {
                return Err(Error::State(
                    "cohort conservative value views overlap".into(),
                ));
            }
            ranges.push(range);
            buffers.push(DeviceBuffer::from_lease(
                lease.clone(),
                layout,
                BufferRole::Activation,
                DeviceAccess::Write,
            )?);
        }
        // One real workspace epoch; the call-local body retains it through the
        // original flight/readback or quarantine and never installs in N1 pool.
        let run = geometry.try_activate()?;
        Ok(Arc::new(Self {
            records,
            _source: binding.source().clone(),
            service,
            geometry,
            _run: run,
            buffers,
            _metadata,
        }))
    }
    pub(crate) fn require_inputs(
        &self,
        inputs: &[Arc<crate::resident_inference::native_fragment::short_hf::ShortHfCallInput>],
    ) -> Result<()> {
        if inputs.len() != self.records.len()
            || inputs.iter().zip(&self.records).any(|(input, record)| {
                !Arc::ptr_eq(input.record(), record)
                    || !Arc::ptr_eq(input.source().source(), &self._source)
            })
        {
            return Err(Error::State(
                "cohort values require every actual row record".into(),
            ));
        }
        self.service.ensure_usable()
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub(crate) fn buffers(&self) -> &[DeviceBuffer] {
        &self.buffers
    }
    pub(crate) fn lease(&self, value: ValueId) -> Result<&ResidentLease> {
        self.geometry
            .leases()
            .get(value.0 as usize)
            .ok_or_else(|| Error::State("cohort original ValueId absent".into()))
    }
}
