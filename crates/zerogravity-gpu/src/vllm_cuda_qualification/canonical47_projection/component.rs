//! Actual computed-operand bridge to the same three original47 native records.
use super::*;
use crate::physical_quarantine::CompletionOwner;
use crate::resident_program::qwen2b_component::{ComponentKind, ComponentRecipe};
pub(crate) struct ConnectedProjectionPrograms(Arc<ProjectionPrograms>);
impl ConnectedProjectionPrograms {
    pub(crate) fn accesses(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.0.accesses(op, input, output, workspace)
    }
    pub(crate) fn enqueue(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.0.enqueue(op, input, output, workspace, cancelled)
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.0.weights.immutable_weights()
    }
    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        unsafe { self.0.retire_after_physical_completion() }
    }
}
pub(crate) fn validate_component_images(
    source: &BoundFullModelSource,
    record: &Arc<RecordedVllm08Model>,
    inputs: &CubinQualificationInputs,
) -> Result<()> {
    manifests::require_memory_semantics()?;
    if inputs.images.capacity() > 8 {
        return Err(Error::Limit("47 component image controls".into()));
    }
    validate_image_set(
        source.source().source(),
        &required_images(std::slice::from_ref(record))?,
        &inputs.images,
    )
}
pub(crate) fn prepare_projections(
    recipe: Arc<ComponentRecipe>,
    service: &Arc<ResidentService>,
    inputs: CubinQualificationInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<ConnectedProjectionPrograms>> {
    not_cancelled(cancelled)?;
    let ComponentKind::Canonical47(arithmetic) = recipe.kind() else {
        return Err(Error::Unsupported(
            "47 component source scope required".into(),
        ));
    };
    arithmetic.require_owned()?;
    init.require_service(service)?;
    if recipe.record().call().source() != service.source_identity()
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "47 component original source/service/envelope".into(),
        ));
    }
    validate_component_images(recipe.source(), recipe.record(), &inputs)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    // Recipe/source only: never Body/Producer, which owns these programs.
    let image_owner: CompletionOwner = recipe.clone();
    let admitted = AdmittedImages::reserve(service, image_owner, inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let pending = PendingProjectionWeights::upload(
        recipe.source().clone(),
        recipe.record(),
        service,
        init,
        cancelled,
    )?;
    let images = admitted.prepare_images(sources, init, cancelled)?;
    let weights = pending.bind_pending_original_upload(init)?;
    let records = [recipe.record().clone()];
    let programs =
        ProjectionPrograms::prepare_original(images, weights, &records, init, cancelled)?;
    Ok(Arc::new(ConnectedProjectionPrograms(programs)))
}

/// One union of original image identities; never a second native set.
pub(crate) fn prepare_post_attention_projections(
    values: Arc<crate::resident_program::qwen2b_component::ComponentValues>,
    service: &Arc<ResidentService>,
    inputs: CubinQualificationInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<(
    Arc<ConnectedProjectionPrograms>,
    Arc<super::super::post_attention_projection::OutPrograms>,
)> {
    let recipe = values.recipe();
    not_cancelled(cancelled)?;
    init.require_service(service)?;
    let ComponentKind::Canonical47(arithmetic) = recipe.kind() else {
        return Err(Error::Unsupported("postAttention47 source scope".into()));
    };
    arithmetic.require_owned()?;
    if !Arc::ptr_eq(values.service(), service)
        || inputs.images.capacity() > 8
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "postAttention47 original service/envelope".into(),
        ));
    }
    manifests::require_memory_semantics()?;
    let records = [recipe.record().clone()];
    let mut required = required_images(&records)?;
    required.insert(super::super::post_attention_projection::required_image(
        recipe,
    )?);
    validate_image_set(recipe.source().source().source(), &required, &inputs.images)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let image_owner: CompletionOwner = recipe.clone();
    let admitted = AdmittedImages::reserve(service, image_owner, inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let pending = PendingProjectionWeights::upload(
        recipe.source().clone(),
        recipe.record(),
        service,
        init,
        cancelled,
    )?;
    let out_pending = super::super::post_attention_projection::PendingOutWeights::upload(
        &values, init, cancelled,
    )?;
    let images = admitted.prepare_images(sources, init, cancelled)?;
    let weights = pending.bind_pending_original_upload(init)?;
    let base =
        ProjectionPrograms::prepare_original(images.clone(), weights, &records, init, cancelled)?;
    let out = super::super::post_attention_projection::OutPrograms::prepare(
        images,
        out_pending,
        values,
        init,
        cancelled,
    )?;
    Ok((Arc::new(ConnectedProjectionPrograms(base)), out))
}

pub(crate) fn required_post_attention_images(
    recipe: &ComponentRecipe,
) -> Result<std::collections::BTreeSet<ProjectionImage>> {
    manifests::require_memory_semantics()?;
    required_images(std::slice::from_ref(recipe.record()))
}

pub(crate) struct PendingMlpPrefixWeights {
    base: PendingProjectionWeights,
    out: super::super::post_attention_projection::PendingOutWeights,
}
impl PendingMlpPrefixWeights {
    pub(crate) fn upload(
        values: &Arc<crate::resident_program::qwen2b_component::ComponentValues>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Self> {
        not_cancelled(cancelled)?;
        let recipe = values.recipe();
        super::super::post_attention_projection::require_mlp_coverage(recipe)?;
        manifests::require_memory_semantics()?;
        init.require_service(values.service())?;
        let base = PendingProjectionWeights::upload(
            recipe.source().clone(),
            recipe.record(),
            values.service(),
            init,
            cancelled,
        )?;
        let out = super::super::post_attention_projection::PendingOutWeights::upload_mlp(
            values, init, cancelled,
        )?;
        Ok(Self { base, out })
    }
}

/// The caller activates one finite MLP union after all managed W uploads.
pub(crate) fn prepare_mlp_prefix_projections(
    values: Arc<crate::resident_program::qwen2b_component::ComponentValues>,
    images: Arc<PreparedImages>,
    pending: PendingMlpPrefixWeights,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<(
    Arc<ConnectedProjectionPrograms>,
    Arc<super::super::post_attention_projection::OutPrograms>,
)> {
    not_cancelled(cancelled)?;
    let recipe = values.recipe();
    super::super::post_attention_projection::require_mlp_coverage(recipe)?;
    init.require_service(values.service())?;
    images.require_service(values.service())?;
    let weights = pending.base.bind_pending_original_upload(init)?;
    let base = ProjectionPrograms::prepare_original(
        images.clone(),
        weights,
        std::slice::from_ref(recipe.record()),
        init,
        cancelled,
    )?;
    let out = super::super::post_attention_projection::OutPrograms::prepare_mlp(
        images,
        pending.out,
        values,
        init,
        cancelled,
    )?;
    Ok((Arc::new(ConnectedProjectionPrograms(base)), out))
}
