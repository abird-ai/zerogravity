//! Actual source/SSA/learned-lease seal for the unpublished six-node cut.
use crate::cuda::gdn::{
    ComponentConvEnqueued, ComponentGdnAdapter, GdnArithmetic, GdnBuffer, GdnCacheSlot,
    GdnFamilyIdentity, GdnInputs, GdnPhase, GdnPrepared, GdnTypedOutputs,
};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_inference::gdn_state::BoundGdnDomains;
use crate::resident_inference::native_fragment::PrefixPhaseEnqueue;
use crate::resident_leases::{ResidentLease, ResidentService};
use crate::resident_program::qwen2b_component::{ComponentKind, ComponentValues};
use std::sync::{Arc, Mutex, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{ParameterWindow, WindowWorkspace};
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

struct SourcePayload {
    values: Arc<ComponentValues>,
    windows: Vec<ParameterWindow>,
    bytes: [Vec<u8>; 4],
    cache_zero: Vec<u8>,
    // Real source/RMS/native producer owner; never the enclosing body/model.
    _upstream: CompletionOwner,
    _host: Vec<WindowWorkspace>,
    _metadata: WindowWorkspace,
}
pub(crate) struct ComponentGdnBinding {
    // Declared source family only; no backedge to preparation or its actual Arc.
    arithmetic_family: GdnFamilyIdentity,
    payload: Arc<SourcePayload>,
    parameters: [ResidentLease; 4],
    cache: ResidentLease,
    domains: Arc<BoundGdnDomains>,
    _code_host: WindowWorkspace,
}
pub(crate) struct ComponentGdnPrepared {
    arithmetic: Arc<GdnArithmetic>,
    binding: Arc<ComponentGdnBinding>,
    prepared: Arc<GdnPrepared>,
    adapter: Arc<ComponentGdnAdapter>,
    pending: Mutex<Option<ComponentConvEnqueued>>,
}
fn phase(values: &ComponentValues) -> Result<GdnPhase> {
    if let ComponentKind::FixedPrompt(arithmetic) = values.recipe().kind() {
        arithmetic.require_owned()?;
        values
            .recipe()
            .kind()
            .require_call(values.recipe().bindings().call())?;
        return Ok(GdnPhase::FixedPrompt);
    }
    match values.recipe().bindings().call().rows().len() {
        1 => Ok(GdnPhase::InitialOne),
        64 => Ok(GdnPhase::Initial64),
        _ => Err(Error::Unsupported(
            "source2B component initial1/64 only".into(),
        )),
    }
}
impl ComponentGdnBinding {
    pub(crate) fn arithmetic_family(&self) -> &GdnFamilyIdentity {
        &self.arithmetic_family
    }
    pub(crate) fn values(&self) -> &Arc<ComponentValues> {
        &self.payload.values
    }
    pub(crate) fn domains(&self) -> &Arc<BoundGdnDomains> {
        &self.domains
    }
    pub(crate) fn cache(&self) -> &ResidentLease {
        &self.cache
    }
    pub(crate) fn phase(&self) -> Result<GdnPhase> {
        phase(self.values())
    }
    pub(crate) fn slot(&self) -> GdnCacheSlot {
        if self.values().recipe().bindings().call().rows().len() == 1
            || matches!(self.values().recipe().kind(), ComponentKind::FixedPrompt(_))
        {
            GdnCacheSlot::One
        } else {
            GdnCacheSlot::Four
        }
    }
    pub(crate) fn require_source(&self, service: &Arc<ResidentService>) -> Result<()> {
        let values = self.values();
        let recipe = values.recipe();
        recipe.kind().require_call(recipe.bindings().call())?;
        if !Arc::ptr_eq(values.service(), service)
            || recipe.bindings().call() != self.domains.invocation()
            || self.domains.conv_domain()? != zerogravity_core::accelerator::program::ConvDomain(0)
            || self.domains.recurrent_domain()?
                != zerogravity_core::accelerator::program::RecurrentDomain(0)
            || self.domains.base().source() != service.source_identity()
            || !Arc::ptr_eq(self.cache.service(), service)
            || self.cache.bytes() != GdnBuffer::CacheArena.bytes(self.phase()?)
            || !self.cache.is_initialized()
        {
            return Err(Error::State(
                "source2B original GDN source/producer/domain issuer".into(),
            ));
        }
        let conv = recipe.operator(4)?;
        let recurrent = recipe.operator(5)?;
        for (index, (lease, window)) in self
            .parameters
            .iter()
            .zip(&self.payload.windows)
            .enumerate()
        {
            let descriptor = if index == 0 {
                &conv.parameters()[0]
            } else {
                &recurrent.parameters()[index - 1]
            };
            let actual = if index == 0 {
                recipe.source().conv4_parameter(&descriptor.name)?
            } else {
                recipe.source().parameter(&descriptor.name)?
            };
            actual.validate_source_parameter(descriptor)?;
            if !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != service.source_identity()
                || lease.service_generation() != service.generation()
                || lease.role() != BufferRole::ImmutableWeights
                || !lease.is_initialized()
                || lease.bytes() != self.payload.bytes[index].len()
                || window.parameter().source_parameter() != *descriptor
            {
                return Err(Error::State(
                    "source2B actual learned parameter lease/header seal".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn require_prepared(&self, prepared: &GdnPrepared) -> Result<()> {
        self.require_source(self.values().service())?;
        self.arithmetic_family
            .require_same(prepared.arithmetic().family())?;
        prepared
            .arithmetic()
            .require_instance(self.phase()?, self.slot(), true)?;
        if let ComponentKind::FixedPrompt(arithmetic) = self.values().recipe().kind() {
            arithmetic.require_prepared(prepared.arithmetic())?;
        }
        if let Some(arithmetic) = self.values().recipe().qwen2b_arithmetic() {
            arithmetic.require_prepared(prepared.arithmetic(), self.phase()?)?;
        }
        let operands = [2, 3, 4, 5, 6]
            .into_iter()
            .map(|id| self.values().lease(ValueId(id)))
            .collect::<Result<Vec<_>>>()?;
        for (role, expected) in [
            GdnBuffer::PackedQkvz,
            GdnBuffer::PackedBa,
            GdnBuffer::ConvOutput,
            GdnBuffer::ZCompact,
            GdnBuffer::Gated,
        ]
        .into_iter()
        .zip(operands)
        .chain(
            [
                GdnBuffer::ConvWeights,
                GdnBuffer::ALog,
                GdnBuffer::DtBias,
                GdnBuffer::GatedGamma,
            ]
            .into_iter()
            .zip(self.parameters.iter()),
        )
        .chain(std::iter::once((GdnBuffer::CacheArena, &self.cache)))
        {
            let actual = prepared.lease(role)?;
            if actual.id() != expected.id()
                || !Arc::ptr_eq(actual.service(), expected.service())
                || actual.service_generation() != expected.service_generation()
                || actual.bytes() != expected.bytes()
            {
                return Err(Error::State(
                    "source2B prepared actual producer/parameter/cache IDs".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.parameters
            .iter()
            .cloned()
            .zip(self.payload.bytes.iter().map(Vec::as_slice))
            .collect()
    }
}
impl ComponentGdnPrepared {
    /// All source, caches, public values and canonical roots precede native activation.
    pub(crate) async fn prepare(
        values: Arc<ComponentValues>,
        domains: Arc<BoundGdnDomains>,
        upstream: CompletionOwner,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        super::projection_cubin::not_cancelled(cancelled)?;
        let source = values.recipe().source();
        let metadata = source.reserve_workspace(256 * 1024)?;
        let host = vec![
            source.reserve_workspace(512 * 1024)?,
            source.reserve_workspace(3 * GdnBuffer::CacheArena.bytes(phase(&values)?))?,
        ];
        let conv = values.recipe().operator(4)?;
        let recurrent = values.recipe().operator(5)?;
        if domains.invocation() != values.recipe().bindings().call()
            || domains.conv_domain()? != zerogravity_core::accelerator::program::ConvDomain(0)
            || domains.recurrent_domain()?
                != zerogravity_core::accelerator::program::RecurrentDomain(0)
            || domains.base().source() != source.source_identity()
            || domains.phase() != phase(&values)?
        {
            return Err(Error::State(
                "source2B genuine original component domains".into(),
            ));
        }
        let mut windows = Vec::with_capacity(4);
        let mut bytes: [Vec<u8>; 4] = std::array::from_fn(|_| Vec::new());
        for index in 0..4 {
            let descriptor = if index == 0 {
                &conv.parameters()[0]
            } else {
                &recurrent.parameters()[index - 1]
            };
            let parameter = if index == 0 {
                source.conv4_parameter(&descriptor.name)?
            } else {
                source.parameter(&descriptor.name)?
            };
            parameter.validate_source_parameter(descriptor)?;
            if parameter.source_parameter() != *descriptor {
                return Err(Error::State("source2B exact parameter header".into()));
            }
            let window = match cancelled {
                Some(flag) => {
                    source.read_parameter_cancellable(&parameter, 0..parameter.elements(), flag)?
                }
                None => source.read_parameter(&parameter, 0..parameter.elements())?,
            };
            if !window.is_complete() {
                return Err(Error::State(
                    "source2B complete learned parameter source".into(),
                ));
            }
            bytes[index] = if index == 3 {
                super::loaded_gamma(&window)?
            } else {
                window.as_slice().to_vec()
            };
            if !(if index == 1 {
                super::layer_common::finite_f32(&bytes[index])
            } else {
                super::finite_bf16(&bytes[index])
            }) {
                return Err(Error::Malformed(
                    "source2B finite actual learned GDN parameter".into(),
                ));
            }
            windows.push(window);
        }
        let code_host = source.reserve_workspace(match values.recipe().kind() {
            ComponentKind::Qwen2b => GdnPrepared::qwen2b_component_code_bytes(phase(&values)?)?,
            ComponentKind::FixedPrompt(arithmetic) => {
                GdnPrepared::fixed_prompt_component_code_bytes(arithmetic)?
            }
        })?;
        let payload = Arc::new(SourcePayload {
            cache_zero: vec![0; GdnBuffer::CacheArena.bytes(phase(&values)?)],
            values,
            windows,
            bytes,
            _upstream: upstream,
            _host: host,
            _metadata: metadata,
        });
        let service = payload.values.service();
        let mut init = service.begin_initialization(domains.base().root())?;
        init.retain_completion_owner(payload.clone())?;
        init.retain_completion_owner(domains.clone())?;
        let mut leases = Vec::with_capacity(4);
        for raw in &payload.bytes {
            leases.push(init.upload_bytes(raw, BufferRole::ImmutableWeights)?);
        }
        let parameters = leases
            .try_into()
            .map_err(|_| Error::State("source2B four actual parameter leases".into()))?;
        let cache = init.upload_bytes(&payload.cache_zero, BufferRole::Scratch)?;
        drop(init.complete_sync().await?);
        let binding = Arc::new(ComponentGdnBinding {
            arithmetic_family: match payload.values.recipe().kind() {
                ComponentKind::Qwen2b => GdnFamilyIdentity::original2b(),
                ComponentKind::FixedPrompt(_) => GdnFamilyIdentity::original08(),
            },
            payload,
            parameters,
            cache,
            domains,
            _code_host: code_host,
        });
        binding.require_source(binding.values().service())?;
        let inputs = GdnInputs {
            qkvz: binding.values().lease(ValueId(2))?.clone(),
            ba: binding.values().lease(ValueId(3))?.clone(),
            cache: binding.cache.clone(),
            conv_weights: binding.parameters[0].clone(),
            a_log: binding.parameters[1].clone(),
            dt_bias: binding.parameters[2].clone(),
            gamma: binding.parameters[3].clone(),
        };
        let outputs = GdnTypedOutputs {
            activated: binding.values().lease(ValueId(4))?.clone(),
            z: binding.values().lease(ValueId(5))?.clone(),
            gated: binding.values().lease(ValueId(6))?.clone(),
        };
        let prepared = match binding.values().recipe().kind() {
            ComponentKind::Qwen2b => {
                GdnPrepared::prepare_qwen2b_component_outputs(
                    binding.clone(),
                    inputs,
                    outputs,
                    cancelled,
                )
                .await?
            }
            ComponentKind::FixedPrompt(_) => {
                GdnPrepared::prepare_fixed_prompt_component_outputs(
                    binding.clone(),
                    inputs,
                    outputs,
                    cancelled,
                )
                .await?
            }
        };
        let adapter = ComponentGdnAdapter::prepare(binding.clone(), &prepared)?;
        Ok(Arc::new(Self {
            arithmetic: prepared.arithmetic().clone(),
            binding,
            prepared,
            adapter,
            pending: Mutex::new(None),
        }))
    }
    pub(crate) fn fixed_prompt_diagnostic_readbacks(
        &self,
    ) -> Result<
        Vec<(
            crate::cuda::gdn::FixedPromptGdnIntermediate,
            crate::resident_leases::ResidentReadback,
        )>,
    > {
        use crate::cuda::gdn::FixedPromptGdnIntermediate as I;
        let ComponentKind::FixedPrompt(arithmetic) = self.binding.values().recipe().kind() else {
            return Err(Error::Unsupported(
                "47 diagnostic cannot reuse old2B cohort".into(),
            ));
        };
        arithmetic.require_prepared(self.prepared.arithmetic())?;
        I::ALL
            .into_iter()
            .map(|key| {
                let lease = self.prepared.lease(key.role())?;
                if lease.bytes() != key.bytes() {
                    return Err(Error::State(
                        "47 original intermediate physical extent differs".into(),
                    ));
                }
                Ok((key, lease.readback(0, lease.bytes())?))
            })
            .collect()
    }
    pub(crate) fn domains(&self) -> &BoundGdnDomains {
        &self.binding.domains
    }
    pub(crate) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut accesses = self.prepared.accesses();
        for (lease, access) in self.binding.domains.leases() {
            if let Some((_, old)) = accesses.iter_mut().find(|(v, _)| v.id() == lease.id()) {
                if access == DeviceAccess::Write {
                    *old = access;
                }
            } else {
                accesses.push((lease, access));
            }
        }
        accesses.push((self.binding.values().status().clone(), DeviceAccess::Write));
        accesses
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.binding.immutable_weights()
    }
    pub(crate) fn enqueue(
        &self,
        node: usize,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        self.arithmetic.require_same(self.prepared.arithmetic())?;
        self.binding
            .domains
            .require_enqueue(self.binding.values().service(), Some(permit))?;
        let effects = self.binding.values().recipe().effects()?;
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| Error::State("source2B Conv token lock".into()))?;
        match node {
            4 if pending.is_none() => {
                let token = self.prepared.launch_component_convolution(
                    self.adapter.clone(),
                    effects[0],
                    cancelled,
                    permit,
                )?;
                let count = token.dispatches();
                *pending = Some(token);
                Ok(count)
            }
            5 => pending
                .take()
                .ok_or_else(|| Error::State("source2B recurrence before actual Conv".into()))?
                .launch_recurrent(effects[1], cancelled, permit),
            _ => Err(Error::State(
                "source2B exact Conv/recurrent node cursor".into(),
            )),
        }
    }
}

pub(crate) fn finite_component_words(bytes: &[u8]) -> bool {
    super::finite_bf16(bytes)
}
pub(crate) fn prepare_projections(
    recipe: Arc<crate::resident_program::qwen2b_component::ComponentRecipe>,
    service: &Arc<ResidentService>,
    inputs: super::CubinQualificationInputs,
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<super::qwen2b_projection::ProjectionPrograms>> {
    use super::projection_cubin::{
        native_error, not_cancelled, resident::AdmittedImages, validate_image_set,
    };
    use super::qwen2b_projection::{PendingProjectionWeights, ProjectionPrograms, required_images};
    not_cancelled(cancelled)?;
    init.require_service(service)?;
    if recipe.bindings().call().source() != service.source_identity()
        || inputs.images.capacity() > 8
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "source2B original image/service/device envelope".into(),
        ));
    }
    let records = [recipe.bindings().clone()];
    validate_image_set(recipe.source(), &required_images(&records)?, &inputs.images)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let image_owner: CompletionOwner = recipe.clone();
    let admitted = AdmittedImages::reserve(service, image_owner, inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let pending = PendingProjectionWeights::upload(
        recipe.source().clone(),
        recipe.bindings(),
        service,
        init,
        cancelled,
    )?;
    let images = admitted.prepare_images(sources, init, cancelled)?;
    ProjectionPrograms::prepare_pending_original_upload(images, pending, &records, init, cancelled)
}

pub(crate) fn validate_component_images(
    source: &zerogravity_artifacts::source_weights::VerifiedTensorSource,
    record: &Arc<crate::resident_program::source_bindings::SourceBindings>,
    inputs: &super::CubinQualificationInputs,
) -> Result<()> {
    if inputs.images.capacity() > 8 {
        return Err(Error::Limit("source2B image control capacity".into()));
    }
    super::projection_cubin::validate_image_set(
        source,
        &super::qwen2b_projection::required_images(std::slice::from_ref(record))?,
        &inputs.images,
    )
}

/// Additive eight-node factory; original six-node factory remains unchanged.
pub(crate) fn prepare_post_attention_projections(
    values: Arc<ComponentValues>,
    service: &Arc<ResidentService>,
    inputs: super::CubinQualificationInputs,
    init: &mut crate::resident_leases::ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<(
    Arc<super::qwen2b_projection::ProjectionPrograms>,
    Arc<super::post_attention_projection::OutPrograms>,
)> {
    use super::projection_cubin::{
        native_error, not_cancelled, resident::AdmittedImages, validate_image_set,
    };
    use super::qwen2b_projection::{PendingProjectionWeights, ProjectionPrograms, required_images};
    let recipe = values.recipe();
    not_cancelled(cancelled)?;
    init.require_service(service)?;
    if recipe.kind() != ComponentKind::Qwen2b
        || !Arc::ptr_eq(values.service(), service)
        || inputs.images.capacity() > 8
        || zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.device_cap
    {
        return Err(Error::State(
            "postAttention2B source/service/device envelope".into(),
        ));
    }
    let records = [recipe.bindings().clone()];
    let mut required = required_images(&records)?;
    required.insert(super::post_attention_projection::required_image(recipe)?);
    validate_image_set(recipe.source(), &required, &inputs.images)?;
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.cublas, &inputs.cublas_lt)
        .map_err(native_error)?;
    let image_owner: CompletionOwner = recipe.clone();
    let admitted = AdmittedImages::reserve(service, image_owner, inputs.images)?;
    init.retain_completion_owner(admitted.completion_owner())?;
    let pending = PendingProjectionWeights::upload(
        recipe.source().clone(),
        recipe.bindings(),
        service,
        init,
        cancelled,
    )?;
    let out_pending =
        super::post_attention_projection::PendingOutWeights::upload(&values, init, cancelled)?;
    let images = admitted.prepare_images(sources, init, cancelled)?;
    let base = ProjectionPrograms::prepare_pending_original_upload(
        images.clone(),
        pending,
        &records,
        init,
        cancelled,
    )?;
    let out = super::post_attention_projection::OutPrograms::prepare(
        images,
        out_pending,
        values,
        init,
        cancelled,
    )?;
    Ok((base, out))
}
