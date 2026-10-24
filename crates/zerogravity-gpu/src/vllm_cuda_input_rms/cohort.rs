//! Existing RMS stages over physical188/4; every logical row stays47/1.
use super::{Columns, Rows, Spec, Stage, specification_for};
use crate::{
    physical_quarantine::CompletionOwner,
    resident_inference::native_fragment::{PrefixPhaseEnqueue, short_hf::ShortHfCallInput},
    resident_leases::{ResidentInitialization, ResidentLease, ResidentService},
    resident_program::{BufferRange, native_cohort::values::CohortValues},
    vllm_ptx_inline::PreparedPinnedLaunch,
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{FullModelProfile, PlanOperation, Vllm08Kind, Vllm08Operator},
    device_events::{BufferRole, DeviceAccess},
};

/// Immutable gamma ownership only; current invocation/root permits stay per row.
pub(crate) struct CohortGamma {
    descriptor: Vllm08Operator,
    source: VerifiedTensorSource,
    lease: ResidentLease,
    _owner: CompletionOwner,
}
impl CohortGamma {
    pub(super) fn bind(
        descriptor: Vllm08Operator,
        source: VerifiedTensorSource,
        lease: ResidentLease,
        owner: CompletionOwner,
    ) -> Result<Self> {
        descriptor.validate()?;
        if !matches!(
            descriptor.kind(),
            Vllm08Kind::InputRms | Vllm08Kind::ResidualRms(_)
        ) || descriptor.parameters().len() != 1
            || lease.bytes() != descriptor.call().model_profile().hidden_size() * 2
            || !lease.is_initialized()
            || lease.role() != BufferRole::ImmutableWeights
            || descriptor.call().source() != source.source_identity()
            || lease.source_identity() != source.source_identity()
        {
            return Err(Error::State(
                "B4 existing immutable source-width gamma owner".into(),
            ));
        }
        Ok(Self {
            descriptor,
            source,
            lease,
            _owner: owner,
        })
    }
    fn require(&self, inputs: &[Arc<ShortHfCallInput>], node: usize) -> Result<bool> {
        let mut fused = None;
        for input in inputs {
            let actual = input
                .record()
                .plan()
                .nodes()
                .get(node)
                .ok_or_else(|| Error::State("B4 RMS source node absent".into()))?;
            let PlanOperation::TypedVllm08 { descriptor, .. } = &actual.operation else {
                return Err(Error::State("B4 typed source RMS absent".into()));
            };
            descriptor.validate()?;
            if descriptor.call() != input.call()
                || descriptor.kind() != self.descriptor.kind()
                || descriptor.layer() != self.descriptor.layer()
                || descriptor.parameters() != self.descriptor.parameters()
                || descriptor.matrices() != self.descriptor.matrices()
                || !self.source.same_issuer(input.source().source().source())
                || descriptor.call().model_profile() != self.descriptor.call().model_profile()
            {
                return Err(Error::State(
                    "B4 every real row immutable RMS role/source".into(),
                ));
            }
            let current = matches!(descriptor.kind(), Vllm08Kind::ResidualRms(_));
            if fused.is_some_and(|first| first != current) {
                return Err(Error::State("B4 source RMS kind differs".into()));
            }
            fused = Some(current);
        }
        fused.ok_or_else(|| Error::State("B4 RMS row population absent".into()))
    }
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    GammaF32,
    GammaShifted,
    Sum,
    Squares,
    Mean,
    Epsilon,
    Inverse,
    Normalized,
    Weighted,
}
impl Role {
    const ALL: [Self; 9] = [
        Self::GammaF32,
        Self::GammaShifted,
        Self::Sum,
        Self::Squares,
        Self::Mean,
        Self::Epsilon,
        Self::Inverse,
        Self::Normalized,
        Self::Weighted,
    ];
    fn bytes(self, m: usize, columns: Columns) -> usize {
        match self {
            Self::GammaF32 | Self::GammaShifted => columns.count() * 4,
            Self::Mean | Self::Epsilon | Self::Inverse => m * 4,
            _ => m * columns.count() * 4,
        }
    }
}
pub(crate) struct CohortRmsStorage {
    source: VerifiedTensorSource,
    service: Arc<ResidentService>,
    initial: bool,
    columns: Columns,
    leases: BTreeMap<Role, ResidentLease>,
    _metadata: Arc<WindowWorkspace>,
}
fn geometry(
    inputs: &[Arc<ShortHfCallInput>],
    service: &Arc<ResidentService>,
) -> Result<(bool, usize, Columns)> {
    service.ensure_usable()?;
    let shape = crate::vllm_cuda_qualification::cohort_native::input_geometry(inputs)?;
    for input in inputs {
        if input.rows() != shape.logical_rows
            || input.workload().prompt_rows() != 47
            || input.call().model_profile() != shape.profile
            || input.call().source() != service.source_identity()
            || !Arc::ptr_eq(input.source().source(), inputs[0].source().source())
        {
            return Err(Error::State("B4 every real RMS logical span/source".into()));
        }
    }
    Ok((
        shape.initial,
        shape.physical_rows,
        Columns::short_hf(shape.profile),
    ))
}
impl CohortRmsStorage {
    pub(crate) fn allocation_layout(initial: bool, profile: FullModelProfile) -> Vec<usize> {
        Role::ALL
            .into_iter()
            .map(|role| role.bytes(if initial { 188 } else { 4 }, Columns::short_hf(profile)))
            .collect()
    }
    pub(crate) fn allocate(
        inputs: &[Arc<ShortHfCallInput>],
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        let (initial, m, columns) = geometry(inputs, &service)?;
        let source = inputs[0].source().source().source().clone();
        let metadata = Arc::new(source.reserve_workspace(128 * 1024)?);
        init.retain_completion_owner(metadata.clone())?;
        let mut leases = BTreeMap::new();
        for role in Role::ALL {
            leases.insert(
                role,
                init.allocate_empty(role.bytes(m, columns), BufferRole::Scratch)?,
            );
        }
        let result = Arc::new(Self {
            source,
            service,
            initial,
            columns,
            leases,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    fn lease(&self, role: Role) -> &ResidentLease {
        &self.leases[&role]
    }
}
pub(crate) struct CohortRmsNode {
    inputs: Box<[Arc<ShortHfCallInput>]>,
    storage: Arc<CohortRmsStorage>,
    values: Arc<CohortValues>,
    _gamma: CohortGamma,
    node: usize,
    stages: Box<[PreparedPinnedLaunch]>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    submitted: AtomicBool,
    _metadata: WindowWorkspace,
}
impl CohortRmsNode {
    pub(crate) fn metadata_bound(
        initial: bool,
        fused: bool,
        profile: FullModelProfile,
    ) -> Result<usize> {
        Stage::ORDER
            .into_iter()
            .try_fold(64 * 1024usize, |sum, stage| {
                let rows = Rows::CohortB4 { initial };
                let Spec { kernel, .. } = if fused {
                    super::fused::fused_specification(stage, rows, Columns::short_hf(profile))?
                } else {
                    specification_for(stage, rows, Columns::short_hf(profile))?
                };
                sum.checked_add(kernel.retained_metadata_bytes()?)
                    .and_then(|n| n.checked_add(4096))
                    .ok_or_else(|| Error::Limit("B4 RMS source/ABI/key/wrapper metadata".into()))
            })
    }
    /// Binder supplies the exact actual B SSA producers, with independent
    /// original row provenance. Scratch is reused only under the body cursor.
    pub(crate) fn bind(
        inputs: &[Arc<ShortHfCallInput>],
        storage: Arc<CohortRmsStorage>,
        node: usize,
        gamma: CohortGamma,
        values: &Arc<CohortValues>,
    ) -> Result<Arc<Self>> {
        values.require_inputs(inputs)?;
        if !Arc::ptr_eq(values.service(), &storage.service) {
            return Err(Error::State("B4 RMS original value/service owner".into()));
        }
        let actual = inputs[0]
            .record()
            .plan()
            .nodes()
            .get(node)
            .ok_or_else(|| Error::State("B4 RMS original node absent".into()))?;
        let operands = actual
            .inputs
            .iter()
            .map(|id| values.lease(*id).cloned())
            .collect::<Result<Vec<_>>>()?;
        let outputs = actual
            .outputs
            .iter()
            .map(|id| values.lease(*id).cloned())
            .collect::<Result<Vec<_>>>()?;
        let (initial, m, columns) = geometry(inputs, &storage.service)?;
        if storage.initial != initial
            || storage.columns != columns
            || !storage.source.same_issuer(&gamma.source)
        {
            return Err(Error::State("B4 RMS scratch phase/source".into()));
        }
        let fused = gamma.require(inputs, node)?;
        let arity = if fused { 2 } else { 1 };
        if operands.len() != arity || outputs.len() != arity {
            return Err(Error::State("B4 RMS original SSA arity".into()));
        }
        for input in inputs {
            let actual = &input.record().plan().nodes()[node];
            if actual.inputs.len() != arity
                || actual.outputs.len() != arity
                || actual.inputs != inputs[0].record().plan().nodes()[node].inputs
                || actual.outputs != inputs[0].record().plan().nodes()[node].outputs
            {
                return Err(Error::State("B4 every source RMS arity".into()));
            }
        }
        for lease in operands.iter().chain(&outputs) {
            if !Arc::ptr_eq(lease.service(), &storage.service)
                || lease.bytes() != m * columns.count() * 2
                || lease.role() != BufferRole::Activation
            {
                return Err(Error::State(
                    "B4 RMS actual physical BF16 producer/output".into(),
                ));
            }
        }
        let all = operands
            .iter()
            .chain(&outputs)
            .chain(storage.leases.values())
            .chain([&gamma.lease])
            .collect::<Vec<_>>();
        for (index, lease) in all.iter().enumerate() {
            let range = BufferRange::new(&lease.handle())?;
            for other in &all[..index] {
                if range.overlaps(&BufferRange::new(&other.handle())?) {
                    return Err(Error::State("B4 RMS physical role alias".into()));
                }
            }
        }
        let rows = Rows::CohortB4 { initial };
        let mut accesses = Vec::new();
        let stages = Stage::ORDER
            .into_iter()
            .map(|stage| {
                let Spec { kernel, grid } = if fused {
                    super::fused::fused_specification(stage, rows, columns)?
                } else {
                    specification_for(stage, rows, columns)?
                };
                let leases: Vec<&ResidentLease> = match stage {
                    Stage::GammaWiden => vec![&gamma.lease, storage.lease(Role::GammaF32)],
                    Stage::GammaShift => vec![
                        storage.lease(Role::GammaF32),
                        storage.lease(Role::GammaShifted),
                    ],
                    Stage::InputWiden => {
                        if fused {
                            vec![
                                &operands[0],
                                &operands[1],
                                storage.lease(Role::Sum),
                                &outputs[1],
                            ]
                        } else {
                            vec![&operands[0], storage.lease(Role::Sum)]
                        }
                    }
                    Stage::Square => vec![storage.lease(Role::Sum), storage.lease(Role::Squares)],
                    Stage::Mean => vec![storage.lease(Role::Squares), storage.lease(Role::Mean)],
                    Stage::Epsilon => vec![storage.lease(Role::Mean), storage.lease(Role::Epsilon)],
                    Stage::Inverse => {
                        vec![storage.lease(Role::Epsilon), storage.lease(Role::Inverse)]
                    }
                    Stage::Normalize => vec![
                        storage.lease(Role::Sum),
                        storage.lease(Role::Inverse),
                        storage.lease(Role::Normalized),
                    ],
                    Stage::Weight => vec![
                        storage.lease(Role::Normalized),
                        storage.lease(Role::GammaShifted),
                        storage.lease(Role::Weighted),
                    ],
                    Stage::Bf16Store => vec![storage.lease(Role::Weighted), &outputs[0]],
                };
                let last = leases.len() - 1;
                for (index, lease) in leases.iter().enumerate() {
                    accesses.push((
                        (*lease).clone(),
                        if index == last || fused && stage == Stage::InputWiden && index == 2 {
                            DeviceAccess::Write
                        } else {
                            DeviceAccess::Read
                        },
                    ));
                }
                kernel.prepare(
                    storage.service.client(),
                    grid,
                    leases.iter().map(|l| l.handle()).collect(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let metadata = storage.source.reserve_workspace(Self::metadata_bound(
            initial,
            fused,
            inputs[0].call().model_profile(),
        )?)?;
        Ok(Arc::new(Self {
            inputs: inputs.to_vec().into_boxed_slice(),
            storage,
            values: values.clone(),
            _gamma: gamma,
            node,
            stages: stages.into_boxed_slice(),
            accesses,
            submitted: AtomicBool::new(false),
            _metadata: metadata,
        }))
    }
    pub(crate) fn enqueue(&self, node: usize, permits: &[PrefixPhaseEnqueue<'_>]) -> Result<usize> {
        if node != self.node || permits.len() != 4 {
            return Err(Error::State("B4 RMS actual node/all-row permits".into()));
        }
        self.values.require_inputs(&self.inputs)?;
        geometry(&self.inputs, &self.storage.service)?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, &self.storage.service)?;
        }
        if self.submitted.swap(true, Ordering::AcqRel) {
            return Err(Error::State("B4 RMS node already consumed".into()));
        }
        for stage in &self.stages {
            stage.launch_on(self.storage.service.client())?;
        }
        Ok(self.stages.len())
    }
    pub(crate) fn sum(&self) -> &ResidentLease {
        self.storage.lease(Role::Sum)
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
}

pub(crate) fn prepare_code(
    source: &VerifiedTensorSource,
    profile: FullModelProfile,
    queue: &cubecl::cuda::OwnedCudaQueue,
) -> Result<usize> {
    let mut count = 0;
    for initial in
        crate::vllm_cuda_qualification::cohort_native::CohortPhases::for_profile(profile).iter()
    {
        let definitions = super::definitions::RmsDefinitions::prepare(
            source,
            Rows::CohortB4 { initial },
            Columns::short_hf(profile),
        )?;
        for kernel in definitions.kernels() {
            kernel.prepare_code(queue)?;
            count += 1;
        }
    }
    Ok(count)
}
