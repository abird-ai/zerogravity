//! Projection/ACT lowering for the SAME original full-model node loop.
//! All inputs are sealed actual ValueId leases; no math, token loop or Ready.
use super::activation_cubin::resident::ActivationPrograms;
use super::activation_manifest::Activation;
use super::cubin_manifests::{Projection, WORKSPACE_BYTES};
use super::mlp_cubin::resident::{MlpBindings, MlpPrograms};
use super::mlp_manifests::MlpProjection;
use super::model_values::ModelValues;
use super::model_weights::{ModelWeightKey, ModelWeights};
use super::projection_cubin::{
    not_cancelled,
    resident::{PreparedImages, ProjectionBindings, ProjectionPrograms},
};
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::{
    PlanOperation, Vllm08Kind, Vllm08Operator, Vllm08Projection,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Projection(Projection),
    Mlp(MlpProjection),
    Activation(Activation),
}
fn operation(operator: &Vllm08Operator) -> Result<Operation> {
    operator.validate()?;
    use Vllm08Kind::{Projection as P, SiluAndMul};
    use Vllm08Projection::{Ba, Down, GateUp, GdnOut, Qkvz};
    Ok(match (operator.call().rows(), operator.kind()) {
        (1, P(Qkvz)) => Operation::Projection(Projection::Qkvz1),
        (1, P(Ba)) => Operation::Projection(Projection::Ba1),
        (1, P(GdnOut)) => Operation::Projection(Projection::Out1),
        (64, P(Qkvz)) => Operation::Projection(Projection::Qkvz64),
        (64, P(Ba)) => Operation::Projection(Projection::Ba64),
        (64, P(GdnOut)) => Operation::Projection(Projection::Out64),
        (2, P(Qkvz)) => Operation::Projection(Projection::Qkvz2),
        (2, P(Ba)) => Operation::Projection(Projection::Ba2),
        (2, P(GdnOut)) => Operation::Projection(Projection::Out2),
        (1, P(GateUp)) => Operation::Mlp(MlpProjection::GateUp1),
        (1, P(Down)) => Operation::Mlp(MlpProjection::Down1),
        (64, P(GateUp)) => Operation::Mlp(MlpProjection::GateUp64),
        (64, P(Down)) => Operation::Mlp(MlpProjection::Down64),
        (2, P(GateUp)) => Operation::Mlp(MlpProjection::GateUp2),
        (2, P(Down)) => Operation::Mlp(MlpProjection::Down2),
        (1, SiluAndMul) => Operation::Activation(Activation::Rows1),
        (64, SiluAndMul) => Operation::Activation(Activation::Rows64),
        (2, SiluAndMul) => Operation::Activation(Activation::Rows2),
        // AttentionOut has different source arithmetic at N64 even when its
        // matrix dimensions equal GdnOut. Its original factory owns that node.
        _ => {
            return Err(Error::Unsupported(
                "full24 exact GDN/MLP/ACT native operation".into(),
            ));
        }
    })
}
fn supported(kind: Vllm08Kind) -> bool {
    matches!(
        kind,
        Vllm08Kind::Projection(
            Vllm08Projection::Qkvz
                | Vllm08Projection::Ba
                | Vllm08Projection::GdnOut
                | Vllm08Projection::GateUp
                | Vllm08Projection::Down
        ) | Vllm08Kind::SiluAndMul
    )
}
struct Node {
    operation: Operation,
    descriptor: Vllm08Operator,
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
        let metadata = weights
            .source()
            .source()
            .reserve_workspace(2 * 1024 * 1024)?;
        let mut nodes = BTreeMap::new();
        let mut phase_accesses = Vec::with_capacity(5);
        let mut phases = Vec::with_capacity(5);
        let mut weight_keys = BTreeSet::new();
        for (phase_index, bound) in values.iter().enumerate() {
            let mut accesses = BTreeMap::new();
            let record = bound.record();
            if !Arc::ptr_eq(bound.service(), service)
                || record.call().source() != weights.source().source_identity()
            {
                return Err(Error::State(
                    "full24 actual native request cohort source/service".into(),
                ));
            }
            phases.push((
                record.call().request().legacy_phase()?,
                record.call().request().teachers().base(),
                record.call().rows(),
            ));
            for (node_index, node) in record.plan().nodes().iter().enumerate() {
                let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                    continue;
                };
                if !supported(descriptor.kind()) {
                    continue;
                }
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
                if let Vllm08Kind::Projection(_) = descriptor.kind() {
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
                        operation: operation(descriptor)?,
                        descriptor: descriptor.as_ref().clone(),
                        input,
                        output,
                    },
                );
            }
            accesses.insert(workspace.id(), (workspace.clone(), DeviceAccess::Write));
            phase_accesses.push(accesses.into_values().collect());
        }
        use zerogravity_core::accelerator::program::VllmBf16Phase as Phase;
        if phases
            != [
                (Phase::InitialOne, 0, 1),
                (Phase::CachedOne, 1, 1),
                (Phase::Initial64, 0, 64),
                (Phase::ContinuedTwo, 64, 2),
                (Phase::CachedOne, 66, 1),
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
            Operation::Projection(projection) => {
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
            Operation::Mlp(projection) => {
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
            Operation::Activation(activation) => {
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

#[cfg(all(feature = "vllm-cuda-qualification", target_os = "linux"))]
mod natural;
