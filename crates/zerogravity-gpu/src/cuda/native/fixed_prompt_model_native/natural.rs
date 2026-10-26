//! Root-free pooled records; fresh immutable original-call operand bindings.
use super::*;
use crate::cuda::native::{
    activation_cubin::resident::ActivationPrograms,
    activation_manifest::Activation,
    cubin_manifests::Projection,
    mlp_cubin::resident::{MlpBindings, MlpPrograms},
    mlp_manifests::MlpProjection,
    projection_cubin::resident::{ProjectionBindings, ProjectionPrograms},
};
use crate::cuda::rms::FusedRmsPrepared;
use crate::resident_inference::{
    ResidentSession,
    native_fragment::{
        PrefixPhaseEnqueue,
        fixed_prompt_natural::{FixedPromptCallInput, FixedPromptCallState},
    },
};
use crate::resident_leases::TiedWeight;
use crate::resident_program::fixed_prompt_binding::same_recipe;
use cubecl::cuda::{Bf16VocabularyHead, VocabularyHeadSelection};
pub(crate) mod attention;
pub(crate) mod images;
pub(crate) use attention::{AttentionCall, AttentionStorage};

pub(crate) struct FixedPromptNativeStorage {
    scopes: [Arc<super::super::fixed_prompt_model::FixedPromptModelBinding>; 2],
    _cold_values: [Arc<ModelValues>; 2],
    weights: Arc<ModelWeights>,
    workspace: ResidentLease,
    images: Arc<PreparedImages>,
    initial: BTreeMap<Operation, Vec<Kernel>>,
    one_projection: Arc<ProjectionPrograms>,
    one_mlp: Arc<MlpPrograms>,
    one_activation: Arc<ActivationPrograms>,
    head: Arc<OwnedCudaKernel>,
    attention: Arc<AttentionStorage>,
    _metadata: Arc<WindowWorkspace>,
}
pub(crate) struct FixedPromptNativeCall {
    storage: Arc<FixedPromptNativeStorage>,
    input: Arc<FixedPromptCallInput>,
    state: Arc<FixedPromptCallState>,
    values: Arc<ModelValues>,
    nodes: BTreeMap<usize, Node>,
    attention: Arc<AttentionCall>,
    tied: Arc<TiedWeight>,
    final_normalizer: Arc<FusedRmsPrepared>,
    counts: AtomicUsize,
    head_count: AtomicUsize,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    _metadata: WindowWorkspace,
}
#[derive(Clone, Copy, Default)]
pub(crate) struct Submitted {
    pub(crate) native: usize,
    pub(crate) copies: usize,
    pub(crate) clears: usize,
}
impl FixedPromptNativeStorage {
    #[expect(
        clippy::too_many_arguments,
        reason = "one source/two geometry storage/set initializer"
    )]
    pub(crate) fn prepare(
        scopes: [Arc<super::super::fixed_prompt_model::FixedPromptModelBinding>; 2],
        cold_values: [Arc<ModelValues>; 2],
        weights: Arc<ModelWeights>,
        workspace: ResidentLease,
        images: Arc<PreparedImages>,
        attention: Arc<AttentionStorage>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        weights.require_initialized()?;
        for (scope, values) in scopes.iter().zip(&cold_values) {
            scope.require_values(values)?;
            init.require_service(values.service())?;
        }
        if scopes[0].kind() != super::super::fixed_prompt_model::FixedPromptCallKind::Prefill
            || !matches!(
                scopes[1].kind(),
                super::super::fixed_prompt_model::FixedPromptCallKind::Decode { base: 47 }
            )
            || !scopes[0].source().same_issuer(scopes[1].source())
            || workspace.bytes() != WORKSPACE_BYTES
            || workspace.role() != BufferRole::Scratch
            || !Arc::ptr_eq(workspace.service(), cold_values[0].service())
        {
            return Err(Error::State(
                "canonical two immutable source geometry templates".into(),
            ));
        }
        images.require_service(cold_values[0].service())?;
        let metadata = Arc::new(weights.source().reserve_workspace(2 * 1024 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(weights.clone())?;
        init.retain_completion_owner(attention.clone())?;
        for v in &cold_values {
            init.retain_completion_owner(v.clone())?;
        }
        let mut initial = BTreeMap::new();
        for op in Operation::ALL.into_iter().chain([Operation::AttentionQkv]) {
            let ps = if op == Operation::AttentionQkv {
                let binding = scopes[0].bind_projection(cold_values[0].clone(), &weights, 32)?;
                super::super::fixed_prompt_attention_projection::model_packets(&binding)?
            } else {
                packets(op)?
            };
            let mut prepared = Vec::new();
            for packet in ps {
                let image = if let Some(kind) = packet.image {
                    images.image(kind)?
                } else {
                    images.activation()?
                };
                not_cancelled(cancelled)?;
                // SAFETY: finite source packet and same admitted single module set.
                let kernel = unsafe { images.set().prepare(image, packet.manifest) }
                    .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                prepared.push(Kernel {
                    roles: packet.roles,
                    kernel,
                });
            }
            initial.insert(op, prepared);
        }
        let one_projection = ProjectionPrograms::prepare(
            images.clone(),
            &[Projection::Qkvz1, Projection::Ba1, Projection::Out1],
            init,
            cancelled,
        )?;
        let one_mlp = MlpPrograms::prepare(
            images.clone(),
            &[MlpProjection::GateUp1, MlpProjection::Down1],
            init,
            cancelled,
        )?;
        let one_activation =
            ActivationPrograms::prepare(images.clone(), &[Activation::Rows1], init, cancelled)?;
        let image = images.image(ProjectionImage::Bf16Gemv2TSm86)?;
        let head = unsafe { images.set().prepare(image, Bf16VocabularyHead::manifest()) }
            .map_err(native_error)?;
        init.retain_completion_owner(head.clone())?;
        let result = Arc::new(Self {
            scopes,
            _cold_values: cold_values,
            weights,
            workspace,
            images,
            initial,
            one_projection,
            one_mlp,
            one_activation,
            head,
            attention,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn bind(
        self: &Arc<Self>,
        input: Arc<FixedPromptCallInput>,
        state: Arc<FixedPromptCallState>,
        values: Arc<ModelValues>,
        positions: Arc<
            crate::resident_program::text_positions::fixed_prompt_natural::FixedPromptPositions,
        >,
        tied: Arc<TiedWeight>,
        final_normalizer: Arc<FusedRmsPrepared>,
    ) -> Result<Arc<FixedPromptNativeCall>> {
        state.require_submission_claimed()?;
        if !Arc::ptr_eq(state.input(), &input)
            || !tied.source().same_issuer(input.source())
            || !self.weights.source().same_issuer(input.source())
        {
            return Err(Error::State(
                "canonical fresh native source/state map differs".into(),
            ));
        }
        let which = usize::from(input.rows() == 1);
        same_recipe(input.source(), self.scopes[which].record(), input.record())?;
        input.scope().require_values(&values)?;
        self.images.require_service(values.service())?;
        let metadata = input.source().reserve_workspace(2 * 1024 * 1024)?;
        let mut nodes = BTreeMap::new();
        let mut all = BTreeMap::new();
        for (index, node) in input.record().plan().nodes().iter().enumerate() {
            let Some(descriptor) = input.record().operator(index)? else {
                continue;
            };
            let Ok(op) = Operation::of(descriptor) else {
                continue;
            };
            let ([x], [y]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                return Err(Error::State("canonical unary projection SSA".into()));
            };
            let projection = if op == Operation::Activation {
                None
            } else {
                Some(
                    input
                        .scope()
                        .bind_projection(values.clone(), &self.weights, index)?,
                )
            };
            if let Some(b) = &projection {
                b.require(&values, descriptor, values.lease(*x)?, values.lease(*y)?)?;
                insert(&mut all, b.weight().clone(), DeviceAccess::Read);
            }
            insert(&mut all, values.lease(*x)?.clone(), DeviceAccess::Read);
            insert(&mut all, values.lease(*y)?.clone(), DeviceAccess::Write);
            nodes.insert(
                index,
                Node {
                    operation: op,
                    descriptor: descriptor.clone(),
                    gdn_output: op == Operation::Out && is_gdn_output(input.record(), *x)?,
                    input: *x,
                    output: *y,
                    projection,
                },
            );
        }
        if nodes.len() != 138
            || tied.lease().bytes() != 508_559_360
            || !tied.lease().is_initialized()
        {
            return Err(Error::State(
                "canonical exact138 source sites/tied extent".into(),
            ));
        }
        let attention = self.attention.bind(
            input.clone(),
            state.clone(),
            values.clone(),
            positions,
            &self.weights,
            &self.workspace,
        )?;
        for (l, a) in attention.accesses()? {
            insert(&mut all, l, a);
        }
        for (l, a) in [
            (self.workspace.clone(), DeviceAccess::Write),
            (tied.lease().clone(), DeviceAccess::Read),
            (values.lease(ValueId(324))?.clone(), DeviceAccess::Read),
            (values.lease(ValueId(326))?.clone(), DeviceAccess::Write),
        ] {
            insert(&mut all, l, a);
        }
        final_normalizer
            .require_full_model_operator(&input.scope().rms(253)?.operator().clone())?;
        Ok(Arc::new(FixedPromptNativeCall {
            storage: self.clone(),
            input,
            state,
            values,
            nodes,
            attention,
            tied,
            final_normalizer,
            counts: AtomicUsize::new(0),
            head_count: AtomicUsize::new(0),
            accesses: all.into_values().collect(),
            _metadata: metadata,
        }))
    }
    pub(crate) fn weights(&self) -> &Arc<ModelWeights> {
        &self.weights
    }
}
impl FixedPromptNativeCall {
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn require_values(
        &self,
        input: &Arc<FixedPromptCallInput>,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        if !Arc::ptr_eq(input, &self.input)
            || !Arc::ptr_eq(values, &self.values)
            || !Arc::ptr_eq(self.state.input(), input)
        {
            return Err(Error::State(
                "canonical SAME immutable native call/value/state".into(),
            ));
        }
        input.scope().require_values(values)
    }
    pub(crate) fn enqueue(
        &self,
        index: usize,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Submitted> {
        not_cancelled(cancelled)?;
        self.input.require_session(session)?;
        self.input.require_permit(permit, self.values.service())?;
        let mut out = Submitted::default();
        if index == 254 {
            if self.head_count.load(Ordering::Acquire) != 0 {
                return Err(Error::State("canonical head reused within one call".into()));
            }
            let selection = if self.input.rows() == 47 {
                VocabularyHeadSelection::FixedPromptLast
            } else {
                VocabularyHeadSelection::WholeSelected
            };
            unsafe {
                self.storage.head.enqueue_vocabulary_head_selected(
                    self.tied.lease().handle(),
                    self.values.lease(ValueId(324))?.handle(),
                    self.values.lease(ValueId(326))?.handle(),
                    selection,
                )
            }
            .map_err(native_error)?;
            self.head_count.fetch_add(1, Ordering::AcqRel);
            out.native = 1;
        } else if let Some(node) = self.nodes.get(&index) {
            if self.input.rows() == 47 {
                for launch in &self.storage.initial[&node.operation] {
                    let handles = launch
                        .roles
                        .iter()
                        .map(|r| match r {
                            Binding::W => node
                                .projection
                                .as_ref()
                                .ok_or_else(|| Error::State("canonical W binding absent".into()))
                                .map(|b| b.weight().handle()),
                            Binding::X => self.values.lease(node.input).map(ResidentLease::handle),
                            Binding::Y => self.values.lease(node.output).map(ResidentLease::handle),
                            Binding::Workspace => Ok(self.storage.workspace.handle()),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)?;
                    out.native += 1;
                }
            } else {
                use Operation::*;
                match node.operation {
                    Qkvz | Ba | Out if node.operation != Out || node.gdn_output => {
                        let p = match node.operation {
                            Qkvz => Projection::Qkvz1,
                            Ba => Projection::Ba1,
                            _ => Projection::Out1,
                        };
                        out.native = self.storage.one_projection.enqueue(
                            p,
                            ProjectionBindings {
                                weights: node.projection.as_ref().unwrap().weight(),
                                input: self.values.lease(node.input)?,
                                output: self.values.lease(node.output)?,
                                workspace: Some(&self.storage.workspace),
                            },
                            cancelled,
                        )?;
                    }
                    GateUp | Down => {
                        let p = if node.operation == GateUp {
                            MlpProjection::GateUp1
                        } else {
                            MlpProjection::Down1
                        };
                        let s = self.storage.one_mlp.enqueue(
                            p,
                            MlpBindings {
                                weights: node.projection.as_ref().unwrap().weight(),
                                input: self.values.lease(node.input)?,
                                output: self.values.lease(node.output)?,
                                workspace: Some(&self.storage.workspace),
                            },
                            cancelled,
                        )?;
                        out.native = s.kernels;
                        out.clears = s.clears;
                    }
                    Activation => {
                        out.native = self.storage.one_activation.enqueue(
                            crate::cuda::native::activation_manifest::Activation::Rows1,
                            self.values.lease(node.input)?,
                            self.values.lease(node.output)?,
                            cancelled,
                        )?;
                    }
                    AttentionQkv | Out => {
                        out = self.attention.enqueue(index, session, permit, cancelled)?;
                    }
                    _ => return Err(Error::State("canonical finite native node mismatch".into())),
                }
            }
        } else {
            out = self.attention.enqueue(index, session, permit, cancelled)?;
        }
        self.counts.fetch_add(out.native, Ordering::AcqRel);
        Ok(out)
    }
    pub(crate) fn extra_readbacks(&self) -> Result<Vec<crate::resident_leases::ResidentReadback>> {
        self.attention.extra_readbacks()
    }
    pub(crate) fn validate_extra(&self, raw: &[cubecl::bytes::Bytes]) -> Result<()> {
        self.attention.validate_extra(raw)
    }
    pub(crate) fn retained(&self) -> Result<bool> {
        self.require_values(&self.input, &self.values)?;
        Ok(self.attention.retained()?
            && self
                .storage
                .weights
                .source()
                .same_issuer(self.input.source())
            && self.values.lease(ValueId(324))?.id() == self.final_normalizer.normalized().id()
            && self.tied.lease().source_identity() == self.input.call().source())
    }
}

impl FixedPromptNativeStorage {
    pub(crate) fn immutable_controls(&self) -> Vec<(ResidentLease, &[u8])> {
        self.attention.immutable_controls()
    }
    pub(crate) unsafe fn retire(&self) -> Result<()> {
        for l in self.initial.values().flatten() {
            unsafe { l.kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        unsafe {
            self.one_projection
                .retire_kernels_after_physical_completion()
        }
        .map_err(native_error)?;
        unsafe { self.one_mlp.retire_kernels_after_physical_completion() }.map_err(native_error)?;
        unsafe {
            self.one_activation
                .retire_kernels_after_physical_completion()
        }
        .map_err(native_error)?;
        unsafe { self.attention.retire() }?;
        unsafe { self.head.retire_after_physical_completion() }.map_err(native_error)?;
        unsafe { self.images.set().retire_after_physical_completion() }.map_err(native_error)
    }
}

impl FixedPromptNativeStorage {
    pub(crate) fn storage_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut all = self.weights.accesses();
        all.push((self.workspace.clone(), DeviceAccess::Write));
        all.extend(self.attention.storage_accesses());
        all
    }
    pub(crate) fn retention(&self) -> Result<(usize, usize, usize, usize)> {
        let mut k = self
            .initial
            .values()
            .flatten()
            .map(|l| l.kernel.clone())
            .collect::<Vec<_>>();
        k.push(self.head.clone());
        k.extend(self.attention.all_kernels());
        let shared = self.images.retention_witness(&k)?.inspect()?;
        let p = self.one_projection.retention_witness()?.inspect()?;
        let m = self.one_mlp.retention_witness()?.inspect()?;
        let a = self.one_activation.retention_witness()?.inspect()?;
        if [
            p.native_charge_bytes,
            m.native_charge_bytes,
            a.native_charge_bytes,
        ]
        .iter()
        .any(|n| *n != shared.native_charge_bytes)
        {
            return Err(Error::State(
                "canonical all original native charge owners differ".into(),
            ));
        }
        Ok((
            shared.image_owners,
            shared.kernel_owners + p.kernel_owners + m.kernel_owners + a.kernel_owners,
            shared.native_charge_bytes,
            shared.image_admission_bytes,
        ))
    }
}

#[cfg(test)]
mod inventory;
