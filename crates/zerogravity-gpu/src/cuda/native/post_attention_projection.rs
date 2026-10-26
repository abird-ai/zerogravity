//! Finite node6 producer, sharing the prefix's original native set/initializer.
use super::cubin_manifests::ProjectionImage;
use super::projection_cubin::{native_error, not_cancelled, resident::PreparedImages};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use crate::resident_program::qwen2b_component::{
    ComponentCut, ComponentKind, ComponentRecipe, ComponentValues,
};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, OwnedCudaKernel,
};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{TensorWindow, WindowWorkspace};
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
mod manifests;
mod weights;
use weights::OutWeights;
pub(crate) use weights::PendingOutWeights;
pub(crate) fn require_recipe(recipe: &ComponentRecipe) -> Result<()> {
    if recipe.cut() != ComponentCut::PostAttention {
        return Err(Error::Unsupported(
            "postAttention source cut required".into(),
        ));
    }
    require_source_recipe(recipe)
}
pub(crate) fn require_mlp_recipe(recipe: &ComponentRecipe) -> Result<()> {
    if recipe.cut() != ComponentCut::Mlp || !matches!(recipe.kind(), ComponentKind::FixedPrompt(_))
    {
        return Err(Error::Unsupported(
            "fixed_prompt MLP source cut required".into(),
        ));
    }
    require_source_recipe(recipe)
}
pub(crate) fn require_qwen2b_mlp_recipe(recipe: &ComponentRecipe) -> Result<()> {
    if recipe.cut() != ComponentCut::Qwen2bMlp
        || recipe.kind() != ComponentKind::Qwen2b
        || recipe.qwen2b_arithmetic()
            != Some(crate::cuda::gdn::Qwen2bGdnArithmetic::OwnedSolve4Recompute3)
    {
        return Err(Error::Unsupported(
            "explicit owned2B MLP source cut required".into(),
        ));
    }
    require_source_recipe(recipe)
}
fn require_bound_recipe(recipe: &ComponentRecipe) -> Result<()> {
    match recipe.cut() {
        ComponentCut::PostAttention => require_recipe(recipe),
        ComponentCut::Mlp => require_mlp_recipe(recipe),
        ComponentCut::Qwen2bMlp => require_qwen2b_mlp_recipe(recipe),
        _ => Err(Error::Unsupported(
            "source-bound postAttention producer cut required".into(),
        )),
    }
}
fn require_source_recipe(recipe: &ComponentRecipe) -> Result<()> {
    recipe.kind().require_call(recipe.bindings().call())?;
    let op = recipe.operator(6)?;
    let node = &recipe.nodes()[6];
    if op.call() != &recipe.bindings().call().clone()
        || op.kind() != (zerogravity_core::accelerator::program::SourceOperatorKind::Projection { selected_row: None })
        || op.numerical_contract() != zerogravity_core::accelerator::program::SourceNumericalContract::ReferenceCudaSm86Bf16V1
        || op.matrices().len() != 1
        || !op.parameters().is_empty()
        || node.inputs != [ValueId(6)]
        || node.outputs != [ValueId(7)]
        || node.state.is_some()
        || !matches!(
            &node.operation,
            PlanOperation::SourceOperator { state: None, .. }
        )
    {
        return Err(Error::State(
            "actual node6 GdnOut source/profile/SSA required".into(),
        ));
    }
    require_matrix_source(recipe.source(), &op.matrices()[0])
}
pub(crate) fn require_coverage(recipe: &ComponentRecipe) -> Result<()> {
    require_recipe(recipe)?;
    manifests::launch(recipe.kind(), recipe.bindings().call().rows().len())?;
    manifests::require_memory_semantics()
}
pub(crate) fn required_image(recipe: &ComponentRecipe) -> Result<ProjectionImage> {
    require_coverage(recipe)?;
    Ok(manifests::launch(recipe.kind(), recipe.bindings().call().rows().len())?.0)
}
pub(crate) fn require_mlp_coverage(recipe: &ComponentRecipe) -> Result<()> {
    require_mlp_recipe(recipe)?;
    manifests::launch(recipe.kind(), recipe.bindings().call().rows().len())?;
    manifests::require_memory_semantics()
}
pub(crate) fn required_mlp_image(recipe: &ComponentRecipe) -> Result<ProjectionImage> {
    require_mlp_coverage(recipe)?;
    Ok(manifests::launch(recipe.kind(), recipe.bindings().call().rows().len())?.0)
}
pub(crate) fn require_qwen2b_mlp_coverage(recipe: &ComponentRecipe) -> Result<()> {
    require_qwen2b_mlp_recipe(recipe)?;
    manifests::launch(recipe.kind(), recipe.bindings().call().rows().len())?;
    manifests::require_memory_semantics()
}
pub(crate) fn required_qwen2b_mlp_image(recipe: &ComponentRecipe) -> Result<ProjectionImage> {
    require_qwen2b_mlp_coverage(recipe)?;
    Ok(manifests::launch(recipe.kind(), recipe.bindings().call().rows().len())?.0)
}
pub(crate) struct OutPrograms {
    values: Arc<ComponentValues>,
    weights: Arc<OutWeights>,
    kernel: Arc<OwnedCudaKernel>,
    _images: Arc<PreparedImages>,
}
impl OutPrograms {
    pub(crate) fn prepare(
        images: Arc<PreparedImages>,
        pending: PendingOutWeights,
        values: Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        require_coverage(values.recipe())?;
        Self::prepare_checked(images, pending, values, init, cancelled)
    }
    pub(crate) fn prepare_mlp(
        images: Arc<PreparedImages>,
        pending: PendingOutWeights,
        values: Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        require_mlp_coverage(values.recipe())?;
        Self::prepare_checked(images, pending, values, init, cancelled)
    }
    pub(crate) fn prepare_qwen2b_mlp(
        images: Arc<PreparedImages>,
        pending: PendingOutWeights,
        values: Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        require_qwen2b_mlp_coverage(values.recipe())?;
        Self::prepare_checked(images, pending, values, init, cancelled)
    }
    fn prepare_checked(
        images: Arc<PreparedImages>,
        pending: PendingOutWeights,
        values: Arc<ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        init.require_service(values.service())?;
        images.require_service(values.service())?;
        let weights = pending.bind(init)?;
        weights.require_source(&values, false)?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(weights.clone())?;
        init.retain_completion_owner(images.completion_owner())?;
        let (image, manifest) = manifests::launch(
            values.recipe().kind(),
            values.recipe().bindings().call().rows().len(),
        )?;
        // SAFETY: only reachable after the independent finite memory gate closes.
        let kernel = unsafe { images.set().prepare(images.image(image)?, manifest) }
            .map_err(native_error)?;
        Ok(Arc::new(Self {
            values,
            weights,
            kernel,
            _images: images,
        }))
    }
    pub(crate) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.weights.require_source(&self.values, true)?;
        let input = self.values.lease(ValueId(6))?;
        let output = self.values.lease(ValueId(7))?;
        let rows = self.values.recipe().bindings().call().rows().len();
        if input.id() == output.id()
            || input.id() == self.weights.lease.id()
            || output.id() == self.weights.lease.id()
            || input.bytes() != rows * 4096
            || output.bytes() != rows * self.values.recipe().kind().columns() * 2
        {
            return Err(Error::State("GdnOut exact sealed nonalias operands".into()));
        }
        for lease in [input, output] {
            if !Arc::ptr_eq(lease.service(), self.values.service())
                || lease.service_generation() != self.values.service().generation()
                || lease.source_identity() != self.values.service().source_identity()
            {
                return Err(Error::State(
                    "GdnOut original source/service/generation".into(),
                ));
            }
        }
        Ok(vec![
            (self.weights.lease.clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ])
    }
    pub(crate) fn require_operator(
        &self,
        op: &zerogravity_core::accelerator::program::SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<()> {
        if op != self.values.recipe().operator(6)?
            || input.id() != self.values.lease(ValueId(6))?.id()
            || output.id() != self.values.lease(ValueId(7))?.id()
            || !Arc::ptr_eq(input.service(), self.values.service())
            || !Arc::ptr_eq(output.service(), self.values.service())
        {
            return Err(Error::State(
                "GdnOut exact original operator/producer map".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn enqueue(&self, cancelled: Option<&AtomicBool>) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.accesses()?;
        let handles = vec![
            self.weights.lease.handle(),
            self.values.lease(ValueId(6))?.handle(),
            self.values.lease(ValueId(7))?.handle(),
        ];
        // SAFETY: exact retained W and SAME immutable producer map checked above.
        unsafe { self.kernel.enqueue(handles) }.map_err(native_error)?;
        Ok(1)
    }
    pub(crate) fn retained_values(&self, values: &Arc<ComponentValues>) -> bool {
        Arc::ptr_eq(&self.values, values)
            && Arc::ptr_eq(self.weights.lease.service(), values.service())
            && self.weights.lease.source_identity() == values.recipe().bindings().call().source()
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        vec![self.weights.immutable()]
    }
    pub(crate) unsafe fn retire_kernels_after_physical_completion(&self) -> Result<()> {
        // Caller retires prefix kernels and the ONE shared native set afterward.
        unsafe { self.kernel.retire_after_physical_completion() }.map_err(native_error)
    }
}

pub(crate) fn validate_images(
    recipe: &ComponentRecipe,
    inputs: &super::CubinQualificationInputs,
) -> Result<()> {
    use super::projection_cubin::validate_image_set;
    if inputs.images.capacity() > 8 {
        return Err(Error::Limit("postAttention image controls".into()));
    }
    let mut required = match recipe.kind() {
        ComponentKind::Qwen2b => crate::cuda::native::qwen2b_projection::required_images(
            std::slice::from_ref(recipe.bindings()),
        )?,
        ComponentKind::FixedPrompt(_) => {
            crate::cuda::native::fixed_prompt_projection::component::required_post_attention_images(
                recipe,
            )?
        }
    };
    required.insert(required_image(recipe)?);
    validate_image_set(recipe.source(), &required, &inputs.images)
}

/// Original47 Out packet only; no ComponentRecipe or layer is retagged.
pub(crate) fn model_fixed_prompt_packet() -> Result<super::fixed_prompt_model_native::Packet> {
    use super::fixed_prompt_model_native::{Binding, Packet};
    manifests::require_memory_semantics()?;
    let (image, manifest) = manifests::model_fixed_prompt_packet();
    Ok(Packet {
        image: Some(image),
        manifest,
        roles: vec![Binding::W, Binding::X, Binding::Y],
    })
}

// Test-only finite packet collector; old production coverage/issuer gates unchanged.
#[cfg(test)]
pub(crate) fn qwen2b_mlp_diagnostic_manifest(
    rows: usize,
) -> Result<(ProjectionImage, CubinKernelManifest)> {
    manifests::launch(ComponentKind::Qwen2b, rows)
}

pub(crate) fn require_matrix_source(
    source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    member: &zerogravity_core::accelerator::program::SourceTensor,
) -> Result<()> {
    let actual = source.matrix(&member.name, member.axes)?;
    if member.source != source.source_identity()
        || actual.tensor().kind != member.kind
        || actual.rows() != member.rows
        || actual.columns() != member.columns
        || actual.axes() != member.axes
    {
        return Err(Error::State(
            "component actual source matrix header differs".into(),
        ));
    }
    source.require_matrix_binding(&actual)
}
