//! Projection/ACT lowering for the SAME original full-model node loop.
//! All inputs are sealed actual ValueId leases; no math, token loop or Ready.
use super::activation_cubin::resident::ActivationPrograms;
use super::activation_manifest::Activation;
use super::cubin_manifests::{Projection, WORKSPACE_BYTES};
use super::mlp_cubin::resident::{MlpBindings, MlpPrograms};
use super::mlp_manifests::MlpProjection;
use super::model_values::ModelValues;
use super::model_weights::{ModelWeightKey, ModelWeights};
use super::native_plan::{NativeOperation, activation_operation, projection_operation};
use super::projection_cubin::{
    not_cancelled,
    resident::{PreparedImages, ProjectionBindings, ProjectionPrograms},
};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use crate::resident_program::source_bindings::SourceBindings;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    SourceGdnContract, SourceNumericalContract, SourceOperator, SourceOperatorKind,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

fn operation(
    record: &SourceBindings,
    index: usize,
    operator: &SourceOperator,
) -> Result<Option<NativeOperation>> {
    let selected = match operator.kind() {
        SourceOperatorKind::SiluAndMul { width: 3584 } => {
            Some(activation_operation(operator.call().rows().len())?)
        }
        SourceOperatorKind::Projection { selected_row: None } => {
            let shapes = operator
                .matrices()
                .iter()
                .map(|m| (m.rows, m.columns))
                .collect::<Vec<_>>();
            if matches!(shapes.as_slice(), [(1024, 2048)]) {
                // Equal output matrices do not imply equal arithmetic: only the
                // actual GatedDelta SSA producer selects this GDN output path.
                let node = record
                    .plan()
                    .nodes()
                    .get(index)
                    .ok_or_else(|| Error::State("native actual source node absent".into()))?;
                let [input] = node.inputs.as_slice() else {
                    return Err(Error::State(
                        "native projection actual producer arity".into(),
                    ));
                };
                let producer = record
                    .plan()
                    .source_value_provenance(*input)?
                    .ok_or_else(|| Error::State("native output actual producer absent".into()))?;
                let previous = record.operator(producer.node_index())?.ok_or_else(|| {
                    Error::State("native output actual producer descriptor absent".into())
                })?;
                if producer.call() != operator.call() || previous.call() != operator.call() {
                    return Err(Error::State(
                        "native output actual producer issuer differs".into(),
                    ));
                }
                let SourceOperatorKind::GatedDelta { geometry, .. } = previous.kind() else {
                    return Ok(None);
                };
                if geometry.contract != SourceGdnContract::ReferenceCudaSm86L2SiluBf16V1
                    || previous.numerical_contract()
                        != SourceNumericalContract::ReferenceCudaSm86Bf16V1
                    || record.plan().nodes()[producer.node_index()]
                        .outputs
                        .get(producer.output_index())
                        != Some(input)
                {
                    return Err(Error::State(
                        "native GDN output actual producer arithmetic/SSA differs".into(),
                    ));
                }
            } else if !matches!(
                shapes.as_slice(),
                [(6144, 1024), (2048, 1024)]
                    | [(16, 1024), (16, 1024)]
                    | [(3584, 1024), (3584, 1024)]
                    | [(1024, 3584)]
            ) {
                return Ok(None);
            }
            Some(projection_operation(operator)?)
        }
        _ => None,
    };
    if selected.is_some()
        && (operator.call() != record.call()
            || operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1)
    {
        return Err(Error::State(
            "native actual source/call/arithmetic contract differs".into(),
        ));
    }
    Ok(selected)
}
struct Node {
    operation: NativeOperation,
    descriptor: SourceOperator,
    input: ResidentLease,
    output: ResidentLease,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ModelNativeSubmitted {
    pub(crate) dispatches: usize,
    pub(crate) workspace_clears: usize,
}
pub(crate) struct ModelNativePrograms {
    nodes: BTreeMap<(usize, usize), Node>,
    projections: Arc<ProjectionPrograms>,
    mlp: Arc<MlpPrograms>,
    activation: Arc<ActivationPrograms>,
    weights: Arc<ModelWeights>,
    values: Vec<Arc<ModelValues>>,
    workspace: ResidentLease,
    accesses: Vec<Vec<(ResidentLease, DeviceAccess)>>,
    _images: Arc<PreparedImages>,
    _metadata: WindowWorkspace,
}
impl ModelNativePrograms {
    /// Managed producers, all114 W groups, parameters/controls/cache/status and
    /// readbacks already exist. No allocations occur after these native loads.
    pub(crate) fn prepare(
        images: Arc<PreparedImages>,
        weights: Arc<ModelWeights>,
        values: Vec<Arc<ModelValues>>,
        workspace: ResidentLease,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        weights.require_initialized()?;
        if values.len() != 5 || values.capacity() > 8 {
            return Err(Error::Unsupported(
                "full24 closed five genuine phases".into(),
            ));
        }
        let service = values[0].service();
        init.require_service(service)?;
        images.require_service(service)?;
        if workspace.bytes() != WORKSPACE_BYTES
            || workspace.role() != BufferRole::Scratch
            || !Arc::ptr_eq(workspace.service(), service)
            || workspace.source_identity() != weights.source().source_identity()
        {
            return Err(Error::State(
                "full24 original shared workspace/source/service".into(),
            ));
        }
        let metadata = weights.source().reserve_workspace(2 * 1024 * 1024)?;
        let mut nodes = BTreeMap::new();
        let mut phase_accesses = Vec::with_capacity(5);
        let mut phases = Vec::with_capacity(5);
        let mut weight_keys = BTreeSet::new();
        for (phase_index, bound) in values.iter().enumerate() {
            let mut accesses = BTreeMap::new();
            let record = bound.record();
            if !Arc::ptr_eq(bound.service(), service)
                || record.call().source() != weights.source().source_identity()
                || !record.source().same_issuer(weights.source())
            {
                return Err(Error::State(
                    "full24 actual native request cohort source/service".into(),
                ));
            }
            phases.push((
                record.call().mode(),
                record.call().rows().base(),
                record.call().rows().len(),
            ));
            for (node_index, node) in record.plan().nodes().iter().enumerate() {
                let Some(descriptor) = record.operator(node_index)? else {
                    continue;
                };
                let Some(operation) = operation(record, node_index, descriptor)? else {
                    continue;
                };
                if descriptor.call() != record.call() || node.state.is_some() {
                    return Err(Error::State(
                        "full24 native descriptor call/state differs".into(),
                    ));
                }
                let ([input_id], [output_id]) = (node.inputs.as_slice(), node.outputs.as_slice())
                else {
                    return Err(Error::State("full24 native unary producer arity".into()));
                };
                let (input, output) = (
                    bound.lease(*input_id)?.clone(),
                    bound.lease(*output_id)?.clone(),
                );
                if input.id() == output.id()
                    || input.id() == workspace.id()
                    || output.id() == workspace.id()
                {
                    return Err(Error::State(
                        "full24 independent native producer/output/workspace".into(),
                    ));
                }
                if matches!(
                    descriptor.kind(),
                    SourceOperatorKind::Projection { selected_row: None }
                ) {
                    let weight = weights.require_operator(descriptor)?;
                    let key = ModelWeightKey::of(descriptor)?;
                    weight_keys.insert(key);
                    if [input.id(), output.id(), workspace.id()].contains(&weight.lease().id()) {
                        return Err(Error::State(
                            "full24 source W aliases actual producer".into(),
                        ));
                    }
                    accesses.insert(
                        weight.lease().id(),
                        (weight.lease().clone(), DeviceAccess::Read),
                    );
                }
                for (lease, access) in
                    [(&input, DeviceAccess::Read), (&output, DeviceAccess::Write)]
                {
                    let entry = accesses
                        .entry(lease.id())
                        .or_insert((lease.clone(), access));
                    if access == DeviceAccess::Write {
                        entry.1 = access;
                    }
                }
                nodes.insert(
                    (phase_index, node_index),
                    Node {
                        operation,
                        descriptor: descriptor.clone(),
                        input,
                        output,
                    },
                );
            }
            accesses.insert(workspace.id(), (workspace.clone(), DeviceAccess::Write));
            phase_accesses.push(accesses.into_values().collect());
        }
        use zerogravity_core::accelerator::program::SourceCallMode::{Decode, Prefill};
        if phases
            != [
                (Prefill, 0, 1),
                (Decode, 1, 1),
                (Prefill, 0, 64),
                (Prefill, 64, 2),
                (Decode, 66, 1),
            ]
            || nodes.len() != 630
            || weight_keys.len() != 102
        {
            return Err(Error::State(
                "full24 exact five-phase126-native-node/102-W subset".into(),
            ));
        }
        // Positive and fault submissions retain the SAME bound source graph.
        init.retain_completion_owner(weights.clone())?;
        for bound in &values {
            init.retain_completion_owner(bound.clone())?;
        }
        init.retain_completion_owner(images.clone())?;
        let projections = ProjectionPrograms::prepare(
            images.clone(),
            &[
                Projection::Qkvz1,
                Projection::Ba1,
                Projection::Out1,
                Projection::Qkvz64,
                Projection::Ba64,
                Projection::Out64,
                Projection::Qkvz2,
                Projection::Ba2,
                Projection::Out2,
            ],
            init,
            cancelled,
        )?;
        let mlp = MlpPrograms::prepare(
            images.clone(),
            &[
                MlpProjection::GateUp1,
                MlpProjection::Down1,
                MlpProjection::GateUp64,
                MlpProjection::Down64,
                MlpProjection::GateUp2,
                MlpProjection::Down2,
            ],
            init,
            cancelled,
        )?;
        let activation = ActivationPrograms::prepare(
            images.clone(),
            &[Activation::Rows1, Activation::Rows64, Activation::Rows2],
            init,
            cancelled,
        )?;
        let result = Arc::new(Self {
            nodes,
            projections,
            mlp,
            activation,
            weights,
            values,
            workspace,
            accesses: phase_accesses,
            _images: images,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    /// Original current-phase flight only; retaining future owners never marks
    /// their uncomputed values initialized or includes them in current Ready.
    pub(crate) fn accesses(&self, phase: usize) -> Result<&[(ResidentLease, DeviceAccess)]> {
        self.accesses
            .get(phase)
            .map(Vec::as_slice)
            .ok_or_else(|| Error::State("full24 current native access phase absent".into()))
    }
    pub(crate) fn require_phase(&self, phase: usize, values: &Arc<ModelValues>) -> Result<()> {
        let prepared = self
            .values
            .get(phase)
            .ok_or_else(|| Error::State("full24 native sealed phase absent".into()))?;
        if !Arc::ptr_eq(prepared, values) || !Arc::ptr_eq(prepared.service(), values.service()) {
            return Err(Error::State(
                "full24 native library differs from SAME original phase value map".into(),
            ));
        }
        Ok(())
    }
    /// Called by the original ResidentProgram's existing node-dispatch loop.
    /// No caller-supplied operands, traversal, readback or completion here.
    pub(crate) fn enqueue(
        &self,
        phase: usize,
        node_index: usize,
        cancelled: Option<&AtomicBool>,
    ) -> Result<ModelNativeSubmitted> {
        not_cancelled(cancelled)?;
        let node = self
            .nodes
            .get(&(phase, node_index))
            .ok_or_else(|| Error::State("full24 unbound native node/phase".into()))?;
        self.values
            .get(phase)
            .ok_or_else(|| Error::State("full24 phase absent".into()))?
            .service()
            .ensure_usable()?;
        let mut submitted = ModelNativeSubmitted::default();
        match node.operation {
            NativeOperation::Projection(projection) => {
                let weight = self.weights.require_operator(&node.descriptor)?;
                submitted.dispatches = self.projections.enqueue(
                    projection,
                    ProjectionBindings {
                        weights: weight.lease(),
                        input: &node.input,
                        output: &node.output,
                        workspace: Some(&self.workspace),
                    },
                    cancelled,
                )?;
            }
            NativeOperation::Mlp(projection) => {
                let weight = self.weights.require_operator(&node.descriptor)?;
                let actual = self.mlp.enqueue(
                    projection,
                    MlpBindings {
                        weights: weight.lease(),
                        input: &node.input,
                        output: &node.output,
                        workspace: Some(&self.workspace),
                    },
                    cancelled,
                )?;
                submitted.dispatches = actual.kernels;
                submitted.workspace_clears = actual.clears;
            }
            NativeOperation::Activation(activation) => {
                submitted.dispatches =
                    self.activation
                        .enqueue(activation, &node.input, &node.output, cancelled)?;
            }
        }
        Ok(submitted)
    }
    /// SAME original initialized-W read-only retirement flight owns every record.
    /// Body retires attention+head records too, then unloads the ONE common set.
    pub(crate) unsafe fn retire_kernels_after_physical_completion(
        &self,
    ) -> std::result::Result<(), cubecl::server::ServerError> {
        unsafe { self.projections.retire_kernels_after_physical_completion() }?;
        unsafe { self.mlp.retire_kernels_after_physical_completion() }?;
        unsafe { self.activation.retire_kernels_after_physical_completion() }
    }
}

#[cfg(all(feature = "cuda-native", target_os = "linux"))]
mod natural;
