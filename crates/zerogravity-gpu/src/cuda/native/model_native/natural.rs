//! Fresh immutable descriptors/maps; original native records/W/storage unchanged.
use super::*;
use crate::resident_program::natural_binding::{same_lease, same_recipe};
impl ModelNativePrograms {
    pub(crate) fn bind_natural(
        self: &Arc<Self>,
        phase: usize,
        values: Arc<ModelValues>,
    ) -> Result<Arc<Self>> {
        let old = self
            .values
            .get(phase)
            .ok_or_else(|| Error::State("natural native physical slot absent".into()))?;
        same_recipe(old.record(), values.record())?;
        for index in 0..327 {
            same_lease(
                old.lease(zerogravity_core::accelerator::program::ValueId(index))?,
                values.lease(zerogravity_core::accelerator::program::ValueId(index))?,
            )?;
        }
        same_lease(old.status(), values.status())?;
        let metadata = self.weights.source().reserve_workspace(2 * 1024 * 1024)?;
        let mut nodes = BTreeMap::new();
        for (&key, bound) in &self.nodes {
            let descriptor = if key.0 == phase {
                let Some(descriptor) = values.record().operator(key.1)? else {
                    return Err(Error::State("natural native source node missing".into()));
                };
                if operation(values.record(), key.1, descriptor)? != Some(bound.operation)
                    || descriptor.kind() != bound.descriptor.kind()
                    || descriptor.numerical_contract() != bound.descriptor.numerical_contract()
                    || descriptor.matrices() != bound.descriptor.matrices()
                    || descriptor.parameters() != bound.descriptor.parameters()
                {
                    return Err(Error::State(
                        "natural native original recipe/weights differs".into(),
                    ));
                }
                descriptor.clone()
            } else {
                bound.descriptor.clone()
            };
            nodes.insert(
                key,
                Node {
                    operation: bound.operation,
                    descriptor,
                    input: bound.input.clone(),
                    output: bound.output.clone(),
                },
            );
        }
        let mut maps = self.values.clone();
        maps[phase] = values;
        Ok(Arc::new(Self {
            nodes,
            projections: self.projections.clone(),
            mlp: self.mlp.clone(),
            activation: self.activation.clone(),
            weights: self.weights.clone(),
            values: maps,
            workspace: self.workspace.clone(),
            accesses: self.accesses.clone(),
            _images: self._images.clone(),
            _metadata: metadata,
        }))
    }
}
