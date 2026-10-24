//! Source-owned finite47 QKVZ/BA producers; no full-model/state permission.
use super::cubin_manifests::{ProjectionImage, WORKSPACE_BYTES};
use super::projection_cubin::{
    CubinQualificationInputs, native_error, not_cancelled,
    resident::{AdmittedImages, PreparedImages},
    validate_image_set,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, OwnedCudaKernel,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{TensorWindow, WindowWorkspace};
use zerogravity_core::accelerator::program::VllmBf16Mode;
use zerogravity_core::accelerator::program::{
    FullModelProfile, PlanOperation, RecordedVllm08Model, Vllm08Kind, Vllm08Operator,
    Vllm08Projection,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
use zerogravity_model_qwen35::original_binding::{
    BoundFullModelSource, OriginalWindowModelSource, QwenMatrixRole,
};

pub(crate) mod component;
mod faults;
mod manifests;
mod qualification;
#[cfg(test)]
mod tests;
mod weights;
pub use faults::{
    Canonical47ProjectionFault, Canonical47ProjectionFaultOutcome,
    qualify_canonical47_projection_fault,
};
pub use qualification::{
    Canonical47ProjectionFixture, Canonical47ProjectionReadback, Canonical47ProjectionReceipt,
    qualify_canonical47_projections,
};
use weights::{PendingProjectionWeights, ProjectionWeights};

const PROMPT: [u32; 47] = [
    248045, 846, 198, 22791, 411, 1103, 314, 24959, 11, 18101, 539, 73982, 11, 364, 1599, 777,
    3665, 25, 220, 16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11, 220, 21, 11, 220,
    22, 11, 220, 23, 11, 248046, 198, 248045, 74455, 198,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    Qkvz,
    Ba,
}
impl Role {
    fn of(operator: &Vllm08Operator) -> Result<Self> {
        operator.validate()?;
        let request = operator.call().request();
        if request.mode() != VllmBf16Mode::Prefill
            || request.teachers().short_hf_span().is_some()
            || request.teachers().base() != 0
            || request.teachers().capacity() != 512
            || request.rows() != 47
            || request.teachers().ids() != PROMPT
            || request.legacy_phase().is_ok()
        {
            return Err(Error::Unsupported(
                "exact original47 Prefill0/span47/prompt".into(),
            ));
        }
        if operator.call().model_profile() != FullModelProfile::Qwen08
            || operator.layer().map(|layer| layer.index()) != Some(0)
        {
            return Err(Error::Unsupported(
                "exact original47 layer0 learned producer".into(),
            ));
        }
        match operator.kind() {
            Vllm08Kind::Projection(Vllm08Projection::Qkvz) => Ok(Self::Qkvz),
            Vllm08Kind::Projection(Vllm08Projection::Ba) => Ok(Self::Ba),
            _ => Err(Error::Unsupported(
                "original47 QKVZ/BA component only".into(),
            )),
        }
    }
    fn rows(self) -> usize {
        match self {
            Self::Qkvz => 8192,
            Self::Ba => 32,
        }
    }
    fn bytes(self) -> usize {
        self.rows() * 1024 * 2
    }
}
#[derive(Clone, Copy)]
enum Binding {
    W,
    X,
    Y,
    Workspace,
}
struct Launch {
    image: ProjectionImage,
    roles: Vec<Binding>,
    manifest: CubinKernelManifest,
}
struct PreparedLaunch {
    roles: Vec<Binding>,
    kernel: Arc<OwnedCudaKernel>,
}
fn descriptors(record: &RecordedVllm08Model) -> Result<[Vllm08Operator; 2]> {
    if record.call().model_profile() != FullModelProfile::Qwen08
        || record.plan().nodes().len() != 255
        || record.plan().values().len() != 327
    {
        return Err(Error::Unsupported(
            "source-recorded exact47 recipe required".into(),
        ));
    }
    let descriptor = |index: usize, role: Role| -> Result<Vllm08Operator> {
        let node = &record.plan().nodes()[index];
        let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
            return Err(Error::State("actual47 projection node absent".into()));
        };
        if Role::of(descriptor)? != role
            || descriptor.call() != record.call()
            || node.inputs.as_slice() != [zerogravity_core::accelerator::program::ValueId(1)]
            || node.outputs.as_slice()
                != [zerogravity_core::accelerator::program::ValueId(
                    index as u32,
                )]
            || node.state.is_some()
        {
            return Err(Error::State(
                "original47 source projection recipe/producer".into(),
            ));
        }
        Ok(descriptor.as_ref().clone())
    };
    Ok([descriptor(2, Role::Qkvz)?, descriptor(3, Role::Ba)?])
}
fn require_source(bound: &BoundFullModelSource, operator: &Vllm08Operator) -> Result<Role> {
    let role = Role::of(operator)?;
    if bound.model_profile() != FullModelProfile::Qwen08
        || operator.call().source() != bound.source().source_identity()
        || !operator.parameters().is_empty()
        || operator.matrices().len() != 2
    {
        return Err(Error::State(
            "actual47 projection bound source/profile/ordered members".into(),
        ));
    }
    let roles = match role {
        Role::Qkvz => [QwenMatrixRole::GdnQkv(0), QwenMatrixRole::GdnGate(0)],
        Role::Ba => [QwenMatrixRole::GdnBeta(0), QwenMatrixRole::GdnAlpha(0)],
    };
    for (member, expected) in operator.matrices().iter().zip(roles) {
        bound.source().validate_matrix(expected, member)?;
    }
    Ok(role)
}
fn require_same_operator(expected: &Vllm08Operator, actual: &Vllm08Operator) -> Result<()> {
    if expected.call() != actual.call()
        || expected.kind() != actual.kind()
        || expected.layer() != actual.layer()
        || expected.matrices() != actual.matrices()
    {
        return Err(Error::State(
            "original47 exact source/teacher issuer/ordered learned members".into(),
        ));
    }
    Ok(())
}
fn required_images(records: &[Arc<RecordedVllm08Model>]) -> Result<BTreeSet<ProjectionImage>> {
    let mut required = BTreeSet::new();
    if records.len() != 1 {
        return Err(Error::Limit(
            "bounded original47 producer recipe cohort".into(),
        ));
    }
    for record in records {
        for operator in descriptors(record)? {
            required.extend(
                manifests::launches(record.call().rows(), Role::of(&operator)?)?
                    .into_iter()
                    .map(|launch| launch.image),
            );
        }
    }
    Ok(required)
}

/// Finite explicit leaf only. Holds immutable
/// W/images only; no body/Root/Session/Ready backedge. All operands are retained
/// in the original caller's armed init/flight before the first native enqueue.
struct ProjectionPrograms {
    launches: BTreeMap<(usize, Role), Vec<PreparedLaunch>>,
    weights: Arc<ProjectionWeights>,
    images: Arc<PreparedImages>,
    _metadata: Arc<WindowWorkspace>,
}
impl ProjectionPrograms {
    fn prepare(
        images: Arc<PreparedImages>,
        weights: Arc<ProjectionWeights>,
        records: &[Arc<RecordedVllm08Model>],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        manifests::require_memory_semantics()?;
        let service = weights.service();
        init.require_service(service)?;
        images.require_service(service)?;
        weights.require_initialized()?;
        Self::prepare_original(images, weights, records, init, cancelled)
    }
    fn prepare_original(
        images: Arc<PreparedImages>,
        weights: Arc<ProjectionWeights>,
        records: &[Arc<RecordedVllm08Model>],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let service = weights.service();
        init.require_service(service)?;
        images.require_service(service)?;
        let metadata = Arc::new(
            weights
                .source()
                .source()
                .reserve_workspace(2 * 1024 * 1024)?,
        );
        // The exact original graph is armed before any partial load/reflection.
        init.retain_completion_owner(weights.clone())?;
        init.retain_completion_owner(images.completion_owner())?;
        init.retain_completion_owner(metadata.clone())?;
        let mut launches = BTreeMap::new();
        for record in records {
            for operator in descriptors(record)? {
                let role = weights.require_operator_source(&operator)?;
                let key = (record.call().rows(), role);
                if launches.contains_key(&key) {
                    continue;
                }
                let mut prepared = Vec::new();
                for launch in manifests::launches(key.0, role)? {
                    not_cancelled(cancelled)?;
                    let image = images.image(launch.image)?;
                    // SAFETY: literal exact47 original packets, three original
                    // sealed code images and SAME prepaid64MiB original owner.
                    let kernel = unsafe { images.set().prepare(image, launch.manifest) }
                        .map_err(native_error)?;
                    prepared.push(PreparedLaunch {
                        roles: launch.roles,
                        kernel,
                    });
                }
                launches.insert(key, prepared);
            }
        }
        Ok(Arc::new(Self {
            launches,
            weights,
            images,
            _metadata: metadata,
        }))
    }
    fn accesses(
        &self,
        operator: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let role = self.weights.require_operator(operator)?;
        let service = self.weights.service();
        let rows = operator.call().rows();
        let weight = self.weights.lease(role);
        let leases = [
            (weight, role.bytes(), BufferRole::ImmutableWeights),
            (input, rows * 2048, input.role()),
            (output, rows * role.rows() * 2, output.role()),
            (workspace, WORKSPACE_BYTES, BufferRole::Scratch),
        ];
        let mut ids = BTreeSet::new();
        for (lease, bytes, role) in leases {
            if !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != self.weights.source().source_identity()
                || lease.service_generation() != service.generation()
                || lease.bytes() != bytes
                || lease.role() != role
                || !ids.insert(lease.id())
            {
                return Err(Error::State(
                    "original47 native exact source/service/extent/nonalias".into(),
                ));
            }
        }
        if !matches!(
            input.role(),
            BufferRole::Activation | BufferRole::Output | BufferRole::Scratch
        ) || !matches!(output.role(), BufferRole::Activation | BufferRole::Output)
            || !self.launches.contains_key(&(rows, role))
        {
            return Err(Error::State(
                "original47 prepared output/finite record".into(),
            ));
        }
        let mut accesses = vec![
            (weight.clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        if role == Role::Ba {
            accesses.push((workspace.clone(), DeviceAccess::Write));
        }
        Ok(accesses)
    }
    fn enqueue(
        &self,
        operator: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.accesses(operator, input, output, workspace)?;
        let role = self.weights.require_operator(operator)?;
        let mut count = 0;
        for launch in &self.launches[&(operator.call().rows(), role)] {
            not_cancelled(cancelled)?;
            let handles = launch
                .roles
                .iter()
                .map(|role| match role {
                    Binding::W => self
                        .weights
                        .lease(Role::of(operator).expect("sealed role"))
                        .handle(),
                    Binding::X => input.handle(),
                    Binding::Y => output.handle(),
                    Binding::Workspace => workspace.handle(),
                })
                .collect();
            // SAFETY: original retained source-W/X/Y/workspace leases above.
            unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)?;
            count += 1;
        }
        Ok(count)
    }
    unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        for launches in self.launches.values() {
            for launch in launches {
                // SAFETY: caller's original armed read-only W guard; all real
                // flights completed, no prepare/enqueue concurrent or retry.
                unsafe { launch.kernel.retire_after_physical_completion() }
                    .map_err(native_error)?;
            }
        }
        unsafe { self.images.set().retire_after_physical_completion() }.map_err(native_error)
    }
}

/// Unchanged accepted finite packet primitive; actual model source/SSA binds elsewhere.
pub(crate) fn model_packets(
    projection: Vllm08Projection,
) -> Result<Vec<super::canonical47_model_native::Packet>> {
    use super::canonical47_model_native::{Binding as B, Packet};
    manifests::require_memory_semantics()?;
    let role = match projection {
        Vllm08Projection::Qkvz => Role::Qkvz,
        Vllm08Projection::Ba => Role::Ba,
        _ => {
            return Err(Error::Unsupported(
                "finite model47 QKVZ/BA packet primitive only".into(),
            ));
        }
    };
    manifests::launches(47, role)?
        .into_iter()
        .map(|p| {
            Ok(Packet {
                image: Some(p.image),
                manifest: p.manifest,
                roles: p
                    .roles
                    .into_iter()
                    .map(|r| match r {
                        Binding::W => B::W,
                        Binding::X => B::X,
                        Binding::Y => B::Y,
                        Binding::Workspace => B::Workspace,
                    })
                    .collect(),
            })
        })
        .collect()
}
