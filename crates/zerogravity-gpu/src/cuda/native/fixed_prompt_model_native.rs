//! Six finite47 operations pooled once, bound to actual original first3 source nodes.
use super::cubin_manifests::{ProjectionImage, WORKSPACE_BYTES};
use super::fixed_prompt_model::FixedPromptProjectionBinding;
use super::fixed_prompt_model_prefix::FixedPromptPrefixCut;
use super::model_values::ModelValues;
use super::model_weights::ModelWeights;
use super::projection_cubin::{
    native_error, not_cancelled,
    resident::{AdmittedImages, PreparedImages},
    validate_activation_image, validate_image_set,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use crate::resident_program::source_bindings::SourceBindings;
use cubecl::cuda::{CubinKernelManifest, OwnedCudaKernel};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    SourceGdnContract, SourceNumericalContract, SourceOperator, SourceOperatorKind, ValueId,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Operation {
    Qkvz,
    Ba,
    Out,
    GateUp,
    Activation,
    Down,
    AttentionQkv,
}
impl Operation {
    const ALL: [Self; 6] = [
        Self::Qkvz,
        Self::Ba,
        Self::Out,
        Self::GateUp,
        Self::Activation,
        Self::Down,
    ];
    fn of(op: &SourceOperator) -> Result<Self> {
        op.validate()?;
        if op.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1 {
            return Err(Error::Unsupported(
                "fixed prompt native arithmetic contract".into(),
            ));
        }
        match op.kind() {
            SourceOperatorKind::Projection { selected_row: None } => {
                let shapes = op
                    .matrices()
                    .iter()
                    .map(|m| (m.rows, m.columns))
                    .collect::<Vec<_>>();
                match shapes.as_slice() {
                    [(6144, 1024), (2048, 1024)] => Ok(Self::Qkvz),
                    [(16, 1024), (16, 1024)] => Ok(Self::Ba),
                    [(1024, 2048)] => Ok(Self::Out),
                    [(3584, 1024), (3584, 1024)] => Ok(Self::GateUp),
                    [(1024, 3584)] => Ok(Self::Down),
                    [(4096, 1024), (512, 1024), (512, 1024)] => Ok(Self::AttentionQkv),
                    _ => Err(Error::Unsupported(
                        "fixed prompt native ordered matrix extents".into(),
                    )),
                }
            }
            SourceOperatorKind::SiluAndMul { width: 3584 } => Ok(Self::Activation),
            _ => Err(Error::Unsupported(
                "fixed_prompt first3 native GDN/MLP/ACT operations only".into(),
            )),
        }
    }
}
fn layer_of(record: &SourceBindings, index: usize) -> Result<usize> {
    for (layer, boundary) in record.selection().boundaries().iter().enumerate() {
        let producer = record
            .plan()
            .source_value_provenance(boundary[0])?
            .ok_or_else(|| Error::State("fixed prompt actual boundary producer absent".into()))?;
        if producer.call() != record.call()
            || record.plan().nodes()[producer.node_index()]
                .outputs
                .get(producer.output_index())
                != Some(&boundary[0])
        {
            return Err(Error::State(
                "fixed prompt actual boundary issuer/SSA differs".into(),
            ));
        }
        if index <= producer.node_index() {
            return Ok(layer);
        }
    }
    Err(Error::State(
        "fixed_prompt native actual layer absent".into(),
    ))
}
fn is_gdn_output(record: &SourceBindings, input: ValueId) -> Result<bool> {
    let producer = record
        .plan()
        .source_value_provenance(input)?
        .ok_or_else(|| Error::State("fixed prompt actual output producer absent".into()))?;
    let previous = record
        .operator(producer.node_index())?
        .ok_or_else(|| Error::State("fixed prompt actual output descriptor absent".into()))?;
    if producer.call() != record.call()
        || previous.call() != record.call()
        || record.plan().nodes()[producer.node_index()]
            .outputs
            .get(producer.output_index())
            != Some(&input)
    {
        return Err(Error::State(
            "fixed prompt actual output producer issuer/SSA differs".into(),
        ));
    }
    match previous.kind() {
        SourceOperatorKind::GatedDelta { geometry, .. }
            if geometry.contract == SourceGdnContract::ReferenceCudaSm86L2SiluBf16V1
                && previous.numerical_contract()
                    == SourceNumericalContract::ReferenceCudaSm86Bf16V1 =>
        {
            Ok(true)
        }
        SourceOperatorKind::Multiply { .. } => Ok(false),
        _ => Err(Error::State(
            "fixed prompt actual output arithmetic producer differs".into(),
        )),
    }
}
#[derive(Clone, Copy)]
pub(crate) enum Binding {
    W,
    X,
    Y,
    Workspace,
}
pub(crate) struct Packet {
    pub(crate) image: Option<ProjectionImage>,
    pub(crate) roles: Vec<Binding>,
    pub(crate) manifest: CubinKernelManifest,
}
struct Kernel {
    roles: Vec<Binding>,
    kernel: Arc<OwnedCudaKernel>,
}
struct Node {
    operation: Operation,
    descriptor: SourceOperator,
    gdn_output: bool,
    input: ValueId,
    output: ValueId,
    projection: Option<FixedPromptProjectionBinding>,
}
pub(crate) mod attention;
pub(crate) mod head;
mod tail;
mod whole;

pub(crate) struct FixedPromptModelNativePrograms {
    whole: Option<Arc<whole::Programs>>,
    attention: Option<Arc<attention::Programs>>,
    tail: Option<Arc<tail::Programs>>,
    tail_counts: [AtomicUsize; 4],
    cut: Arc<FixedPromptPrefixCut>,
    values: Arc<ModelValues>,
    weights: Arc<ModelWeights>,
    workspace: ResidentLease,
    nodes: BTreeMap<usize, Node>,
    kernels: BTreeMap<Operation, Vec<Kernel>>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    images: Arc<PreparedImages>,
    enqueued_records: AtomicUsize,
    qkv_enqueues: AtomicUsize,
    _metadata: Option<WindowWorkspace>,
}
/// Actual immutable shared-set ownership metadata; no completion/replay permit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FixedPromptNativeRetention {
    pub(crate) images: usize,
    pub(crate) kernels: usize,
    pub(crate) native_charge_bytes: usize,
    pub(crate) image_admission_bytes: usize,
}

fn packets(op: Operation) -> Result<Vec<Packet>> {
    match op {
        Operation::Qkvz => super::fixed_prompt_projection::model_packets(2),
        Operation::Ba => super::fixed_prompt_projection::model_packets(3),
        Operation::Out => Ok(vec![
            super::post_attention_projection::model_fixed_prompt_packet()?,
        ]),
        Operation::GateUp => super::fixed_prompt_mlp::model_packets(8),
        Operation::Down => super::fixed_prompt_mlp::model_packets(10),
        Operation::Activation => Ok(vec![Packet {
            image: None,
            roles: vec![Binding::X, Binding::Y],
            manifest: super::activation_manifest::fixed_prompt_manifest(),
        }]),
        Operation::AttentionQkv => Err(Error::State(
            "fixed_prompt QKV requires the actual model projection binding".into(),
        )),
    }
}
fn after_qkv_main(fault: usize, late_cancel: &AtomicBool) -> Result<()> {
    match fault {
        0 => Ok(()),
        3 => Err(Error::State(
            "deliberate fixed_prompt unknown after actual QKV main".into(),
        )),
        4 => {
            late_cancel.store(true, Ordering::Release);
            Ok(())
        }
        _ => Err(Error::State(
            "fixed_prompt QKV main fault mode differs".into(),
        )),
    }
}
fn insert(
    all: &mut BTreeMap<
        zerogravity_core::device_events::BufferLeaseId,
        (ResidentLease, DeviceAccess),
    >,
    lease: ResidentLease,
    access: DeviceAccess,
) {
    let v = all.entry(lease.id()).or_insert((lease, access));
    if access == DeviceAccess::Write {
        v.1 = access;
    }
}
pub(crate) fn validate_inputs(
    cut: &FixedPromptPrefixCut,
    inputs: &super::fixed_prompt_mlp::FixedPromptMlpQualificationInputs,
) -> Result<()> {
    let mut required = BTreeSet::new();
    for operation in Operation::ALL {
        required.extend(
            packets(operation)?
                .into_iter()
                .filter_map(|packet| packet.image),
        );
    }
    let max_images = if cut.includes_layer3() { 10 } else { 8 };
    if required != super::fixed_prompt_mlp::required_images()
        || inputs.projections.images.capacity() > max_images
    {
        return Err(Error::State(
            "fixed_prompt exact pooled model native image union".into(),
        ));
    }
    if cut.includes_qkv() {
        required.insert(ProjectionImage::AttentionAmpere256x128Sm80);
    }
    if cut.includes_kv() {
        required.extend([
            ProjectionImage::AttentionQkSm86,
            ProjectionImage::AttentionKvSm86,
        ]);
    }
    if cut.includes_fa() {
        required.insert(ProjectionImage::AttentionFaCausalCombineSm80);
        if inputs
            .projections
            .images
            .iter()
            .filter(|i| i.kind == ProjectionImage::AttentionFaCausalCombineSm80)
            .any(|i| i.bytes.capacity() > 4 * 1024 * 1024 || i.admission.bytes() > 4 * 1024 * 1024)
        {
            return Err(Error::Limit(
                "fixed_prompt FA exact4MiB source image admission".into(),
            ));
        }
    }
    if cut.includes_layer3() {
        required.extend([
            ProjectionImage::AttentionSigmoidSm86,
            ProjectionImage::AttentionMultiplySm86,
        ]);
        if inputs
            .projections
            .images
            .iter()
            .filter(|i| {
                matches!(
                    i.kind,
                    ProjectionImage::AttentionSigmoidSm86 | ProjectionImage::AttentionMultiplySm86
                )
            })
            .any(|i| i.bytes.capacity() > 4 * 1024 * 1024 || i.admission.bytes() > 4 * 1024 * 1024)
        {
            return Err(Error::Limit(
                "fixed_prompt tail exact4MiB per source image admission".into(),
            ));
        }
    }
    if cut.includes_kv()
        && inputs
            .projections
            .images
            .iter()
            .filter(|image| {
                matches!(
                    image.kind,
                    ProjectionImage::AttentionQkSm86 | ProjectionImage::AttentionKvSm86
                )
            })
            .any(|image| {
                image.bytes.capacity() > 2 * 1024 * 1024
                    || image.admission.bytes() > 2 * 1024 * 1024
            })
    {
        return Err(Error::Limit(
            "fixed_prompt QK/KV exact two2MiB image-admission ceiling".into(),
        ));
    }
    let activation = inputs
        .activation
        .as_ref()
        .ok_or_else(|| Error::State("fixed_prompt model actual activation image absent".into()))?;
    if activation.bytes.capacity() > 2 * 1024 * 1024
        || activation.admission.bytes() > 2 * 1024 * 1024
    {
        return Err(Error::Limit(
            "fixed_prompt finite model activation image admission".into(),
        ));
    }
    let source = cut.scope().source();
    if cut.includes_layer3() {
        super::projection_cubin::validate_fixed_prompt_layer3_image_set(
            source,
            &required,
            &inputs.projections.images,
        )?;
    } else if cut.includes_fa() {
        super::projection_cubin::validate_fixed_prompt_fa_image_set(
            source,
            &required,
            &inputs.projections.images,
        )?;
    } else if cut.includes_kv() {
        super::projection_cubin::validate_fixed_prompt_qkkv_image_set(
            source,
            &required,
            &inputs.projections.images,
        )?;
    } else {
        validate_image_set(source, &required, &inputs.projections.images)?;
    }
    validate_activation_image(source, activation)
}
pub(crate) fn prepare_images(
    cut: &Arc<FixedPromptPrefixCut>,
    service: &Arc<crate::resident_leases::ResidentService>,
    inputs: super::fixed_prompt_mlp::FixedPromptMlpQualificationInputs,
    init: &mut ResidentInitialization<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<Arc<PreparedImages>> {
    not_cancelled(cancelled)?;
    validate_inputs(cut, &inputs)?;
    init.require_service(service)?;
    if zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
        .snapshot(&service.physical_key())?
        .limit
        != inputs.projections.device_cap
    {
        return Err(Error::State(
            "fixed_prompt model original physical native envelope".into(),
        ));
    }
    let sources = service
        .pinned_cuda_queue()?
        .verify_bf16_blas_sources(&inputs.projections.cublas, &inputs.projections.cublas_lt)
        .map_err(native_error)?;
    let admitted = AdmittedImages::reserve_with_activation(
        service,
        cut.clone(),
        inputs.projections.images,
        inputs.activation,
    )?;
    init.retain_completion_owner(admitted.completion_owner())?;
    if cut.includes_layer3() {
        admitted.prepare_fixed_prompt_layer3_images(sources, init, cancelled)
    } else if cut.includes_fa() {
        admitted.prepare_fixed_prompt_fa_images(sources, init, cancelled)
    } else if cut.includes_kv() {
        admitted.prepare_fixed_prompt_qkkv_images(sources, init, cancelled)
    } else if cut.includes_qkv() {
        admitted.prepare_fixed_prompt_qkv_images(sources, init, cancelled)
    } else {
        admitted.prepare_fixed_prompt_mlp_images(sources, init, cancelled)
    }
}
impl FixedPromptModelNativePrograms {
    /// All real managed values/cache/parameters/W exist before these8 native records.
    pub(crate) fn prepare(
        cut: Arc<FixedPromptPrefixCut>,
        values: Arc<ModelValues>,
        weights: Arc<ModelWeights>,
        workspace: ResidentLease,
        images: Arc<PreparedImages>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        Self::prepare_with_attention(
            cut, values, weights, workspace, images, None, init, cancelled,
        )
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "one original initializer and same pooled native set"
    )]
    pub(crate) fn prepare_with_attention(
        cut: Arc<FixedPromptPrefixCut>,
        values: Arc<ModelValues>,
        weights: Arc<ModelWeights>,
        workspace: ResidentLease,
        images: Arc<PreparedImages>,
        attention: Option<Arc<attention::Binding>>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        if cut.is_whole() {
            return Err(Error::Unsupported(
                "wholeInitial47 requires its distinct pooled constructor".into(),
            ));
        }
        if cut.includes_kv() != attention.is_some() {
            return Err(Error::State(
                "fixed_prompt QK/KV exact additional owner cohort".into(),
            ));
        }
        if let Some(binding) = &attention {
            binding.require_values(&cut, &values)?;
        }
        not_cancelled(cancelled)?;
        cut.scope().require_values(&values)?;
        weights.require_fixed_prompt_prefix(&cut)?;
        init.require_service(values.service())?;
        images.require_service(values.service())?;
        if !weights.source().same_issuer(cut.scope().source())
            || !Arc::ptr_eq(workspace.service(), values.service())
            || workspace.service_generation() != values.service().generation()
            || workspace.source_identity() != values.service().source_identity()
            || workspace.bytes() != WORKSPACE_BYTES
            || workspace.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "fixed_prompt model native source/service/workspace owner".into(),
            ));
        }
        let metadata = cut.scope().source().reserve_workspace(2 * 1024 * 1024)?;
        let mut nodes = BTreeMap::new();
        let mut all = BTreeMap::new();
        let mut per_layer = BTreeMap::<usize, BTreeSet<Operation>>::new();
        for (index, node) in cut.nodes().iter().enumerate() {
            let Some(descriptor) = cut.scope().record().operator(index)? else {
                continue;
            };
            let Ok(operation) = Operation::of(descriptor) else {
                continue;
            };
            let layer = layer_of(cut.scope().record(), index)?;
            let ([x], [y]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                return Err(Error::State("fixed_prompt native actual unary SSA".into()));
            };
            let finite_site = (layer < 3 && operation != Operation::AttentionQkv)
                || (cut.includes_qkv()
                    && index == 32
                    && layer == 3
                    && operation == Operation::AttentionQkv)
                || (cut.includes_layer3()
                    && layer == 3
                    && matches!(
                        (index, operation),
                        (38, Operation::Out)
                            | (40, Operation::GateUp)
                            | (41, Operation::Activation)
                            | (42, Operation::Down)
                    ));
            if !finite_site
                || descriptor.call() != cut.scope().record().call()
                || !per_layer.entry(layer).or_default().insert(operation)
            {
                return Err(Error::State(
                    "fixed_prompt native actual source/node/layer cohort".into(),
                ));
            }
            let (input, output) = (values.lease(*x)?, values.lease(*y)?);
            let projection = if operation == Operation::Activation {
                None
            } else {
                Some(
                    cut.scope()
                        .bind_projection(values.clone(), &weights, index)?,
                )
            };
            if input.id() == output.id()
                || input.id() == workspace.id()
                || output.id() == workspace.id()
                || input.bytes() != descriptor.input_elements()?[0] * 2
                || output.bytes() != descriptor.output_elements()?[0] * 2
            {
                return Err(Error::State(
                    "fixed_prompt native full actual producer extents/nonalias".into(),
                ));
            }
            if let Some(binding) = &projection {
                binding.require(&values, descriptor, input, output)?;
                if binding.weight().id() == workspace.id() {
                    return Err(Error::State("fixed_prompt W/workspace alias".into()));
                }
                insert(&mut all, binding.weight().clone(), DeviceAccess::Read);
            }
            insert(&mut all, input.clone(), DeviceAccess::Read);
            insert(&mut all, output.clone(), DeviceAccess::Write);
            nodes.insert(
                index,
                Node {
                    operation,
                    descriptor: descriptor.clone(),
                    gdn_output: operation == Operation::Out
                        && is_gdn_output(cut.scope().record(), *x)?,
                    input: *x,
                    output: *y,
                    projection,
                },
            );
        }
        let expected = Operation::ALL.into_iter().collect::<BTreeSet<_>>();
        let qkv_layers = usize::from(cut.includes_qkv());
        let mut layer3_expected = BTreeSet::from([Operation::AttentionQkv]);
        if cut.includes_layer3() {
            layer3_expected.extend([
                Operation::Out,
                Operation::GateUp,
                Operation::Activation,
                Operation::Down,
            ]);
        }
        if nodes.len() != 18 + qkv_layers + usize::from(cut.includes_layer3()) * 4
            || per_layer.len() != 3 + qkv_layers
            || (0..3).any(|layer| per_layer.get(&layer) != Some(&expected))
            || (cut.includes_qkv() && per_layer.get(&3) != Some(&layer3_expected))
        {
            return Err(Error::State(
                "fixed_prompt exactly6 native source operations in each actual first3 layer".into(),
            ));
        }
        insert(&mut all, workspace.clone(), DeviceAccess::Write);
        init.retain_completion_owner(cut.clone())?;
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(weights.clone())?;
        let mut kernels = BTreeMap::new();
        let mut operations = Operation::ALL.to_vec();
        if cut.includes_qkv() {
            operations.push(Operation::AttentionQkv);
        }
        for operation in operations {
            let mut prepared = Vec::new();
            let records = if operation == Operation::AttentionQkv {
                let binding = nodes
                    .get(&32)
                    .and_then(|node| node.projection.as_ref())
                    .ok_or_else(|| {
                        Error::State("fixed_prompt actual node32 projection owner absent".into())
                    })?;
                super::fixed_prompt_attention_projection::model_packets(binding)?
            } else {
                packets(operation)?
            };
            for packet in records {
                not_cancelled(cancelled)?;
                let image = match packet.image {
                    Some(kind) => images.image(kind)?,
                    None => images.activation()?,
                };
                // SAFETY: unchanged accepted finite47 packet, same original image/set/init.
                let kernel = unsafe { images.set().prepare(image, packet.manifest) }
                    .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                prepared.push(Kernel {
                    roles: packet.roles,
                    kernel,
                });
            }
            kernels.insert(operation, prepared);
        }
        let attention = attention
            .map(|binding| attention::Programs::prepare(binding, &images, init, cancelled))
            .transpose()?;
        if let Some(programs) = &attention {
            for (lease, access) in programs.binding().accesses() {
                insert(&mut all, lease, access);
            }
        }
        let tail = if cut.includes_layer3() {
            Some(tail::Programs::prepare(
                cut.clone(),
                values.clone(),
                &images,
                init,
                cancelled,
            )?)
        } else {
            None
        };
        if let Some(t) = &tail {
            for (lease, access) in t.accesses()? {
                insert(&mut all, lease, access);
            }
        }
        if kernels.values().map(Vec::len).sum::<usize>()
            + attention.as_ref().map_or(0, |p| p.record_count())
            + usize::from(tail.is_some()) * 2
            != cut.native_records()
        {
            return Err(Error::State(
                "fixed_prompt exactly8 pooled native records".into(),
            ));
        }
        let result = Arc::new(Self {
            whole: None,
            attention,
            tail,
            tail_counts: [
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
                AtomicUsize::new(0),
            ],
            cut,
            values,
            weights,
            workspace,
            nodes,
            kernels,
            accesses: all.into_values().collect(),
            images,
            enqueued_records: AtomicUsize::new(0),
            qkv_enqueues: AtomicUsize::new(0),
            _metadata: Some(metadata),
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(crate) fn require_values(
        &self,
        cut: &Arc<FixedPromptPrefixCut>,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        if !Arc::ptr_eq(cut, &self.cut) || !Arc::ptr_eq(values, &self.values) {
            return Err(Error::State(
                "fixed_prompt native actual cut/value issuer differs".into(),
            ));
        }
        if let Some(whole) = &self.whole {
            whole.require_values(cut, values)?;
        }
        if let Some(programs) = &self.attention {
            programs.binding().require_values(cut, values)?;
        }
        self.cut.scope().require_values(values)
    }
    pub(crate) fn attention_binding(&self) -> Option<&Arc<attention::Binding>> {
        self.attention.as_ref().map(|programs| programs.binding())
    }
    pub(crate) fn fa_enqueues(&self) -> usize {
        self.attention.as_ref().map_or(0, |p| p.fa_enqueues())
    }
    pub(crate) fn fa_output_unpromoted(&self) -> Result<bool> {
        self.attention
            .as_ref()
            .map_or(Ok(false), |p| p.binding().fa_output_unpromoted())
    }
    pub(crate) fn attention_counts(&self) -> (usize, usize, usize) {
        self.attention
            .as_ref()
            .map_or((0, 0, 0), |programs| programs.counts())
    }
    pub(crate) fn enqueue_attention(
        &self,
        index: usize,
        permit: &crate::resident_inference::native_fragment::PrefixPhaseEnqueue<'_>,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        self.attention
            .as_ref()
            .ok_or_else(|| Error::State("fixed_prompt QK/KV programs absent".into()))?
            .enqueue(index, permit, cancelled, fault)
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn weights(&self) -> &Arc<ModelWeights> {
        &self.weights
    }
    pub(crate) fn enqueue(&self, index: usize, cancelled: Option<&AtomicBool>) -> Result<usize> {
        if self.cut.is_whole() {
            return Err(Error::Unsupported(
                "wholeInitial47 requires its exact permit dispatcher".into(),
            ));
        }
        self.enqueue_inner(index, cancelled, 0)
    }
    pub(crate) fn enqueue_qkv(
        &self,
        index: usize,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        if self.cut.is_whole()
            || !self.cut.includes_qkv()
            || index != 32
            || !matches!(fault, 0 | 3 | 4)
        {
            return Err(Error::State(
                "fixed_prompt actual QKV group/fault scope".into(),
            ));
        }
        self.enqueue_inner(index, cancelled, fault)
    }
    pub(crate) fn enqueue_layer3(
        &self,
        index: usize,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        if !self.cut.includes_layer3()
            || !matches!(index, 36 | 37 | 38 | 40 | 41 | 42)
            || !matches!(fault, 0 | 11..=16)
        {
            return Err(Error::State(
                "fixed_prompt layer3 finite tail cursor/fault scope".into(),
            ));
        }
        self.cut.require_layer3_tail()?;
        if matches!(index, 36 | 37) {
            let t = self
                .tail
                .as_ref()
                .ok_or_else(|| Error::State("fixed_prompt original gate programs absent".into()))?;
            let before = t.counts().into_iter().sum::<usize>();
            let result = t.enqueue(
                index,
                cancelled,
                if matches!(fault, 11 | 12) { fault } else { 0 },
            );
            let after = t.counts().into_iter().sum::<usize>();
            self.enqueued_records.fetch_add(
                after
                    .checked_sub(before)
                    .ok_or_else(|| Error::State("fixed_prompt gate counter regressed".into()))?,
                Ordering::AcqRel,
            );
            return result;
        }
        self.enqueue_inner(
            index,
            cancelled,
            if matches!(fault, 13..=16) { fault } else { 0 },
        )
    }
    pub(crate) fn tail_native_enqueues(&self) -> [usize; 6] {
        let gates = self.tail.as_ref().map_or([0, 0], |t| t.counts());
        let rest = self
            .tail_counts
            .each_ref()
            .map(|v| v.load(Ordering::Acquire));
        [gates[0], gates[1], rest[0], rest[1], rest[2], rest[3]]
    }
    pub(crate) fn tail_outputs_unpromoted(&self) -> Result<bool> {
        if !self.cut.includes_layer3() {
            return Ok(false);
        }
        for id in 46..54 {
            if self.values.lease(ValueId(id))?.is_initialized() {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(crate) fn enqueued_records(&self) -> usize {
        self.enqueued_records.load(Ordering::Acquire)
    }
    pub(crate) fn qkv_enqueues(&self) -> usize {
        self.qkv_enqueues.load(Ordering::Acquire)
    }
    pub(crate) fn qkv_output_unpromoted(&self) -> Result<bool> {
        Ok(self.cut.includes_qkv() && !self.values.lease(ValueId(41))?.is_initialized())
    }
    fn enqueue_inner(
        &self,
        index: usize,
        cancelled: Option<&AtomicBool>,
        fault: usize,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        let node = self
            .nodes
            .get(&index)
            .ok_or_else(|| Error::State("fixed_prompt bound source native node absent".into()))?;
        let input = self.values.lease(node.input)?;
        let output = self.values.lease(node.output)?;
        if let Some(binding) = &node.projection {
            binding.require(&self.values, &node.descriptor, input, output)?;
        }
        let kernels = self
            .kernels
            .get(&node.operation)
            .ok_or_else(|| Error::State("fixed_prompt pooled kernel absent".into()))?;
        let late_cancel = AtomicBool::new(index == 41 && fault == 14);
        for (ordinal, kernel) in kernels.iter().enumerate() {
            not_cancelled(if late_cancel.load(Ordering::Acquire) {
                Some(&late_cancel)
            } else {
                cancelled
            })?;
            if (node.operation == Operation::AttentionQkv
                || (self.cut.includes_layer3() && index >= 38))
                && output.is_initialized()
            {
                return Err(Error::State(
                    "fixed_prompt QKV output promoted inside native group".into(),
                ));
            }
            let handles = kernel
                .roles
                .iter()
                .map(|role| {
                    Ok(match role {
                        Binding::W => node
                            .projection
                            .as_ref()
                            .ok_or_else(|| {
                                Error::State("fixed_prompt learned source W absent".into())
                            })?
                            .weight()
                            .handle(),
                        Binding::X => input.handle(),
                        Binding::Y => output.handle(),
                        Binding::Workspace => self.workspace.handle(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: original caller's armed body flight retains all exact accesses.
            unsafe { kernel.kernel.enqueue(handles) }.map_err(native_error)?;
            self.enqueued_records.fetch_add(1, Ordering::AcqRel);
            if self.cut.includes_layer3()
                && let Some(slot) = match index {
                    38 => Some(0),
                    40 => Some(1),
                    41 => Some(2),
                    42 => Some(3),
                    _ => None,
                }
            {
                self.tail_counts[slot].fetch_add(1, Ordering::AcqRel);
                if output.is_initialized() {
                    return Err(Error::State(
                        "fixed_prompt layer3 tail prematurely promoted output".into(),
                    ));
                }
                if index == 38 && fault == 13 {
                    return Err(Error::State(
                        "deliberate fixed_prompt unknown after actual AttentionOut".into(),
                    ));
                }
                if index == 42 && ordinal == 0 {
                    if fault == 15 {
                        return Err(Error::State(
                            "deliberate fixed_prompt unknown after actual layer3 Down main".into(),
                        ));
                    }
                    if fault == 16 {
                        late_cancel.store(true, Ordering::Release);
                    }
                }
            }
            if node.operation == Operation::AttentionQkv {
                self.qkv_enqueues.fetch_add(1, Ordering::AcqRel);
                if ordinal == 0 {
                    // This main packet writes partial workspace; dead-C Y binding
                    // is conservative access only, never an initialization permit.
                    if output.is_initialized() {
                        return Err(Error::State(
                            "fixed_prompt QKV main initialized dead-C output".into(),
                        ));
                    }
                    after_qkv_main(fault, &late_cancel)?;
                }
            }
        }
        Ok(kernels.len())
    }
    /// Host-only original Weak inspection; safe after the service is fenced.
    pub(crate) fn retention_snapshot(&self) -> Result<FixedPromptNativeRetention> {
        let mut kernels = self
            .kernels
            .values()
            .flatten()
            .map(|k| k.kernel.clone())
            .collect::<Vec<_>>();
        if let Some(attention) = &self.attention {
            kernels.extend(attention.kernels());
        }
        if let Some(tail) = &self.tail {
            kernels.extend(tail.kernels().iter().cloned());
        }
        if let Some(whole) = &self.whole {
            kernels.extend(whole.kernels());
        }
        let actual = self.images.retention_witness(&kernels)?.inspect()?;
        Ok(FixedPromptNativeRetention {
            images: actual.image_owners,
            kernels: actual.kernel_owners,
            native_charge_bytes: actual.native_charge_bytes,
            image_admission_bytes: actual.image_admission_bytes,
        })
    }

    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        if let Some(whole) = &self.whole {
            unsafe { whole.retire() }?;
        }
        for kernel in self.kernels.values().flatten() {
            unsafe { kernel.kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        if let Some(attention) = &self.attention {
            unsafe { attention.retire() }?;
        }
        if let Some(tail) = &self.tail {
            unsafe { tail.retire() }?;
        }
        unsafe { self.images.set().retire_after_physical_completion() }.map_err(native_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_main_cuts_use_actual_original_cancellation_gate() {
        let cancel = AtomicBool::new(false);
        after_qkv_main(0, &cancel).unwrap();
        not_cancelled(Some(&cancel)).unwrap();
        assert!(after_qkv_main(3, &cancel).is_err());
        assert!(!cancel.load(Ordering::Acquire));
        after_qkv_main(4, &cancel).unwrap();
        assert!(
            matches!(not_cancelled(Some(&cancel)),Err(Error::State(message))
            if message == "closed original cubin qualification cancelled")
        );
        assert!(after_qkv_main(2, &cancel).is_err());
    }
    #[test]
    fn qkv_never_uses_the_unbound_generic_packet_factory() {
        assert!(packets(Operation::AttentionQkv).is_err());
    }
    #[test]
    fn pooled_eight_records_reuse_finite47_partial_and_parallel_paths() {
        let q = packets(Operation::Qkvz).unwrap();
        assert_eq!(q.len(), 1);
        assert_eq!(q[0].manifest.grid, [64, 1, 1]);
        assert_eq!(q[0].manifest.block, [64, 1, 1]);
        let ba = packets(Operation::Ba).unwrap();
        assert_eq!(ba.len(), 2);
        assert_eq!(ba[0].manifest.grid, [1, 1, 16]);
        assert_eq!(ba[0].manifest.dynamic_shared_bytes, 49152);
        assert_eq!(ba[1].manifest.grid, [1, 3, 1]);
        let down = packets(Operation::Down).unwrap();
        assert_eq!(down.len(), 2);
        assert_eq!(down[0].manifest.grid, [8, 1, 2]);
        let all = Operation::ALL
            .into_iter()
            .flat_map(|op| packets(op).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(all.len(), 8);
        assert_eq!(
            all.iter().filter_map(|p| p.image).collect::<BTreeSet<_>>(),
            super::super::fixed_prompt_mlp::required_images()
        );
    }
    #[test]
    fn source_bound_packet_io_preserves_real_47_weight_and_public_extents() {
        for (op, expected) in [
            (Operation::Qkvz, [16777216, 96256, 770048]),
            (Operation::Ba, [65536, 96256, 3008]),
            (Operation::Out, [4194304, 192512, 96256]),
            (Operation::GateUp, [14680064, 96256, 673792]),
            (Operation::Down, [7340032, 336896, 96256]),
        ] {
            let actual = packets(op).unwrap();
            for (role, bytes) in [Binding::W, Binding::X, Binding::Y]
                .into_iter()
                .zip(expected)
            {
                let key = |r: &Binding| {
                    matches!(
                        (r, role),
                        (Binding::W, Binding::W)
                            | (Binding::X, Binding::X)
                            | (Binding::Y, Binding::Y)
                    )
                };
                let extent = actual
                    .iter()
                    .flat_map(|p| p.roles.iter().zip(&p.manifest.bindings))
                    .find(|(r, _)| key(r))
                    .unwrap()
                    .1
                    .bytes;
                assert_eq!(extent, bytes);
            }
        }
    }

    #[test]
    #[ignore = "ROOT compiled metadata collector only; no native permission"]
    fn actual_fixed_prompt_pooled_out_for_root_comparison() {
        let packets = packets(Operation::Out).unwrap();
        assert_eq!(packets.len(), 1);
        println!("LAYER3_OUT47 manifest={:?}", packets[0].manifest);
    }
}

pub(crate) mod natural;
