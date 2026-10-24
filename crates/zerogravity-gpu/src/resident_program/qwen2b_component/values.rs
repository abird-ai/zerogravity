//! Exact cut-sized original SSA/lease bindings; no independent graph/value issuer.
use super::recipe::ComponentRecipe;
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_program::{DeviceBuffer, ResidentBufferLayout, ResidentPhysicalType};
use std::{collections::BTreeSet, sync::Arc};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{ActivationPrecision, ValueId, ValueShape};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

pub(crate) struct ComponentValues {
    recipe: Arc<ComponentRecipe>,
    values: Vec<ResidentLease>,
    status: ResidentLease,
    service: Arc<ResidentService>,
    _metadata: WindowWorkspace,
}
impl ComponentValues {
    /// Called only by SAME ResidentProgram physical workspace preparation.
    pub(super) fn bind(
        recipe: Arc<ComponentRecipe>,
        buffers: &[DeviceBuffer],
        status: ResidentLease,
        service: Arc<ResidentService>,
    ) -> Result<Arc<Self>> {
        let metadata = recipe
            .source()
            .source()
            .source()
            .reserve_workspace(64 * 1024)?;
        let plan = recipe.record().plan();
        if buffers.len() != recipe.value_end()
            || plan.source() != service.source_identity()
            || !Arc::ptr_eq(status.service(), &service)
            || status.bytes() != 4
            || status.role() != BufferRole::Scratch
            || status.source_identity() != plan.source()
            || status.service_generation() != service.generation()
        {
            return Err(Error::State(
                "source2B same original workspace/status service".into(),
            ));
        }
        let mut seen = BTreeSet::from([status.id()]);
        let mut actual = Vec::with_capacity(recipe.value_end());
        for (index, buffer) in buffers.iter().enumerate() {
            let ValueShape::Vector(elements) = plan.values()[index] else {
                return Err(Error::State("source2B dense original BF16 vector".into()));
            };
            if plan.precisions()[index] != ActivationPrecision::Bf16WidenedF32 {
                return Err(Error::State("source2B original physical precision".into()));
            }
            plan.vllm_value_provenance(ValueId(index as u32))?
                .ok_or_else(|| Error::State("source2B original SSA provenance absent".into()))?
                .require_invocation(recipe.record().call().request())?;
            let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, elements)?;
            buffer.require_layout(layout)?;
            // This validates the public handle against the retained original
            // lease; no same-width replacement or offset view is accepted.
            let lease = buffer.original_lease()?;
            if !Arc::ptr_eq(lease.service(), &service)
                || lease.source_identity() != plan.source()
                || lease.service_generation() != service.generation()
                || lease.bytes() != layout.bytes()?
                || !matches!(lease.role(), BufferRole::Activation | BufferRole::Output)
                || !seen.insert(lease.id())
            {
                return Err(Error::State(
                    "source2B exact distinct original seven SSA leases".into(),
                ));
            }
            actual.push(lease.clone());
        }
        let values = actual;
        Ok(Arc::new(Self {
            recipe,
            values,
            status,
            service,
            _metadata: metadata,
        }))
    }
    pub(crate) fn recipe(&self) -> &Arc<ComponentRecipe> {
        &self.recipe
    }
    pub(crate) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub(crate) fn status(&self) -> &ResidentLease {
        &self.status
    }
    pub(crate) fn lease(&self, id: ValueId) -> Result<&ResidentLease> {
        self.values
            .get(id.0 as usize)
            .ok_or_else(|| Error::Unsupported("unprepared source2B value beyond Gated".into()))
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, l)| {
                (
                    l.clone(),
                    if i == 0 {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    },
                )
            })
            .chain(std::iter::once((self.status.clone(), DeviceAccess::Write)))
            .collect()
    }
}
