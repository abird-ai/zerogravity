//! Bind actual new value/state issuers to an UNUSED original GDN storage slot.
use super::*;
use crate::resident_program::natural_binding::{same_lease, same_recipe};
impl ModelGdnPrepared {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        values: Arc<ModelValues>,
        domains: Arc<BoundModelState>,
    ) -> Result<Arc<Self>> {
        if self.fixed_prompt.is_some() || self.text.is_some() {
            return Err(Error::Unsupported(
                "fixed_prompt model GDN cached/natural binding remains closed".into(),
            ));
        }
        same_recipe(self.binding.values.record(), values.record())?;
        if self
            .pending
            .lock()
            .map_err(|_| Error::State("cold GDN split mutex poisoned".into()))?
            .is_some()
            || self.binding.layer() != domains.layer()
            || domains.call() != values.record().call()
        {
            return Err(Error::State(
                "natural unused GDN original source/call/root slot".into(),
            ));
        }
        let record = values.record();
        let conv = &record.plan().nodes()[self.binding.nodes[0]];
        let recurrent = &record.plan().nodes()[self.binding.nodes[1]];
        let (Some(cd), Some(rd)) = (
            &record.operator(self.binding.nodes[0])?,
            &record.operator(self.binding.nodes[1])?,
        ) else {
            return Err(Error::State("natural actual GDN operators absent".into()));
        };
        self.binding.parameters.require(cd, rd)?;
        domains.require_effect(
            conv.state
                .ok_or_else(|| Error::State("natural Conv effect absent".into()))?,
            false,
        )?;
        domains.require_effect(
            recurrent
                .state
                .ok_or_else(|| Error::State("natural recurrent effect absent".into()))?,
            true,
        )?;
        let ([qkvz, ba], [activated, z], [ra, rz, rb], [gated]) = (
            conv.inputs.as_slice(),
            conv.outputs.as_slice(),
            recurrent.inputs.as_slice(),
            recurrent.outputs.as_slice(),
        ) else {
            return Err(Error::State("natural GDN SSA arity".into()));
        };
        if activated != ra || z != rz || ba != rb {
            return Err(Error::State("natural GDN same actual Conv cohort".into()));
        }
        let operands: [ResidentLease; 5] = [*qkvz, *ba, *activated, *z, *gated]
            .into_iter()
            .map(|id| values.lease(id).cloned())
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| Error::State("natural exact five original GDN values".into()))?;
        for (old, new) in self.binding.operands.iter().zip(&operands) {
            same_lease(old, new)?;
        }
        let metadata = self.binding.source().reserve_workspace(64 * 1024)?;
        let binding = Arc::new(ModelGdnBinding {
            arithmetic: self.binding.arithmetic.clone(),
            values,
            parameters: self.binding.parameters.clone(),
            nodes: self.binding.nodes,
            operands,
        });
        let adapter =
            self.adapter
                .bind_natural(binding.clone(), domains.clone(), &self.prepared)?;
        Ok(Arc::new(Self {
            binding,
            prepared: self.prepared.clone(),
            adapter,
            domains,
            _arena: self._arena.clone(),
            pending: Mutex::new(None),
            _metadata: metadata,
            fixed_prompt: None,
            text: None,
        }))
    }
}
