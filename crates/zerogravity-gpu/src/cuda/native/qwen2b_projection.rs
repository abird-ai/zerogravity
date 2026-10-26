//! Source-owned finite2B QKVZ/BA producers; no full-model/state permission.
use super::cubin_manifests::{ProjectionImage, WORKSPACE_BYTES};
use super::projection_cubin::{
    CubinQualificationInputs, native_error, not_cancelled,
    resident::{AdmittedImages, PreparedImages},
    validate_image_set,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::resident_program::source_bindings::SourceBindings;
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, OwnedCudaKernel,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, atomic::AtomicBool};
use zerogravity_artifacts::source_weights::{TensorWindow, VerifiedTensorSource, WindowWorkspace};
use zerogravity_core::accelerator::program::{
    SourceNumericalContract, SourceOperator, SourceOperatorKind,
};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};

mod faults;
mod manifests;
mod qualification;
#[cfg(test)]
mod tests;
mod weights;
pub use faults::{
    Qwen2bProjectionFault, Qwen2bProjectionFaultOutcome, qualify_qwen2b_projection_fault,
};
pub use qualification::{
    Qwen2bProjectionFixture, Qwen2bProjectionReadback, Qwen2bProjectionReceipt,
    qualify_qwen2b_projections,
};
pub(crate) use weights::{PendingProjectionWeights, ProjectionWeights};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Role {
    Qkvz,
    Ba,
}
impl Role {
    fn of(operator: &SourceOperator) -> Result<Self> {
        operator.validate()?;
        crate::cuda::gdn::GdnPhase::for_reference_call(operator.call())?;
        if operator.kind() != (SourceOperatorKind::Projection { selected_row: None })
            || operator.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1
            || !operator.parameters().is_empty()
            || operator.matrices().len() != 2
            || operator.matrices().iter().any(|m| m.columns != 2048)
        {
            return Err(Error::Unsupported(
                "projection native mathematical/source layout differs".into(),
            ));
        }
        match operator
            .matrices()
            .iter()
            .map(|m| m.rows)
            .collect::<Vec<_>>()
            .as_slice()
        {
            [6144, 2048] => Ok(Self::Qkvz),
            [16, 16] => Ok(Self::Ba),
            _ => Err(Error::Unsupported(
                "projection exact ordered native packing differs".into(),
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
        self.rows() * 2048 * 2
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
#[derive(Clone, Copy)]
enum PacketScope {
    Original,
    Qwen2bMlp,
}
impl PacketScope {
    fn launches(self, rows: usize, role: Role) -> Result<Vec<Launch>> {
        match self {
            Self::Original => manifests::launches(rows, role),
            Self::Qwen2bMlp => manifests::mlp_launches(rows, role),
        }
    }
}
struct PreparedLaunch {
    roles: Vec<Binding>,
    kernel: Arc<OwnedCudaKernel>,
}
pub(crate) fn descriptors(record: &SourceBindings) -> Result<[SourceOperator; 2]> {
    if record.plan().nodes().len() != 255 || record.plan().values().len() != 327 {
        return Err(Error::Unsupported(
            "source-recorded exact2B recipe required".into(),
        ));
    }
    let descriptor = |index: usize, role: Role| -> Result<SourceOperator> {
        let node = &record.plan().nodes()[index];
        let Some(descriptor) = record.operator(index)? else {
            return Err(Error::State("actual2B projection node absent".into()));
        };
        if Role::of(descriptor)? != role
            || descriptor.call() != record.call()
            || node.inputs.as_slice() != [zerogravity_core::accelerator::program::ValueId(1)]
            || node.outputs.len() != 1
            || node.state.is_some()
        {
            return Err(Error::State(
                "original2B source projection recipe/producer".into(),
            ));
        }
        Ok(descriptor.clone())
    };
    Ok([descriptor(2, Role::Qkvz)?, descriptor(3, Role::Ba)?])
}
fn require_source(bound: &VerifiedTensorSource, operator: &SourceOperator) -> Result<Role> {
    let role = Role::of(operator)?;
    if operator.call().source() != bound.source_identity() {
        return Err(Error::State(
            "projection actual source/call issuer differs".into(),
        ));
    }
    for member in operator.matrices() {
        let actual = bound.matrix(&member.name, member.axes)?;
        if member.source != bound.source_identity()
            || member.rows != actual.rows()
            || member.columns != actual.columns()
            || member.kind != actual.tensor().kind
            || member.kind != zerogravity_core::quant::QuantKind::Bf16
            || member.axes != actual.axes()
        {
            return Err(Error::State(
                "projection actual ordered source header differs".into(),
            ));
        }
        bound.require_matrix_binding(&actual)?;
    }
    Ok(role)
}
pub(crate) fn required_images(
    records: &[Arc<SourceBindings>],
) -> Result<BTreeSet<ProjectionImage>> {
    let mut required = BTreeSet::new();
    if records.is_empty() || records.len() > 5 {
        return Err(Error::Limit(
            "bounded original2B producer recipe cohort".into(),
        ));
    }
    for record in records {
        for operator in descriptors(record)? {
            required.extend(
                manifests::launches(record.call().rows().len(), Role::of(&operator)?)?
                    .into_iter()
                    .map(|launch| launch.image),
            );
        }
    }
    Ok(required)
}

/// Reused by explicit leaf and original recorded-node consumer. Holds immutable
/// W/images only; no body/Root/Session/Ready backedge. All operands are retained
/// in the original caller's armed init/flight before the first native enqueue.
pub(crate) struct ProjectionPrograms {
    launches: BTreeMap<(usize, Role), Vec<PreparedLaunch>>,
    weights: Arc<ProjectionWeights>,
    images: Arc<PreparedImages>,
    _metadata: Arc<WindowWorkspace>,
}
impl ProjectionPrograms {
    pub(crate) fn prepare(
        images: Arc<PreparedImages>,
        weights: Arc<ProjectionWeights>,
        records: &[Arc<SourceBindings>],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let service = weights.service();
        init.require_service(service)?;
        images.require_service(service)?;
        weights.require_initialized()?;
        Self::prepare_original(
            images,
            weights,
            records,
            PacketScope::Original,
            init,
            cancelled,
        )
    }
    /// Distinct source component path: no eventless promotion of pending W.
    pub(crate) fn prepare_pending_original_upload(
        images: Arc<PreparedImages>,
        pending: PendingProjectionWeights,
        records: &[Arc<SourceBindings>],
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let weights = pending.bind_pending_original_upload(init)?;
        Self::prepare_original(
            images,
            weights,
            records,
            PacketScope::Original,
            init,
            cancelled,
        )
    }
    /// Captured MLP union packets; legacy factories retain their original bytes.
    pub(crate) fn prepare_pending_qwen2b_mlp_upload(
        images: Arc<PreparedImages>,
        pending: PendingProjectionWeights,
        recipe: &crate::resident_program::qwen2b_component::ComponentRecipe,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        super::qwen2b_mlp::require_recipe(recipe)?;
        let weights = pending.bind_pending_original_upload(init)?;
        Self::prepare_original(
            images,
            weights,
            std::slice::from_ref(recipe.bindings()),
            PacketScope::Qwen2bMlp,
            init,
            cancelled,
        )
    }
    fn prepare_original(
        images: Arc<PreparedImages>,
        weights: Arc<ProjectionWeights>,
        records: &[Arc<SourceBindings>],
        packet_scope: PacketScope,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        let service = weights.service();
        init.require_service(service)?;
        images.require_service(service)?;
        let metadata = Arc::new(weights.source().reserve_workspace(2 * 1024 * 1024)?);
        // The exact original graph is armed before any partial load/reflection.
        init.retain_completion_owner(weights.clone())?;
        init.retain_completion_owner(images.completion_owner())?;
        init.retain_completion_owner(metadata.clone())?;
        let mut launches = BTreeMap::new();
        for record in records {
            for operator in descriptors(record)? {
                let role = weights.require_operator_source(&operator)?;
                let key = (record.call().rows().len(), role);
                if launches.contains_key(&key) {
                    continue;
                }
                let mut prepared = Vec::new();
                for launch in packet_scope.launches(key.0, role)? {
                    not_cancelled(cancelled)?;
                    let image = images.image(launch.image)?;
                    // SAFETY: literal exact2B original packets, four original
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
    pub(crate) fn accesses(
        &self,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        let role = self.weights.require_operator(operator)?;
        let service = self.weights.service();
        let rows = operator.call().rows().len();
        let weight = self.weights.lease(role);
        let leases = [
            (weight, role.bytes(), BufferRole::ImmutableWeights),
            (input, rows * 4096, input.role()),
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
                    "original2B native exact source/service/extent/nonalias".into(),
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
                "original2B prepared output/finite record".into(),
            ));
        }
        let mut accesses = vec![
            (weight.clone(), DeviceAccess::Read),
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        if rows == 64 && role == Role::Ba {
            accesses.push((workspace.clone(), DeviceAccess::Write));
        }
        Ok(accesses)
    }
    pub(crate) fn enqueue(
        &self,
        operator: &zerogravity_core::accelerator::program::SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.accesses(operator, input, output, workspace)?;
        let role = self.weights.require_operator(operator)?;
        let mut count = 0;
        for launch in &self.launches[&(operator.call().rows().len(), role)] {
            not_cancelled(cancelled)?;
            let handles = launch
                .roles
                .iter()
                .map(|binding| match binding {
                    Binding::W => self.weights.lease(role).handle(),
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
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        self.weights.immutable_weights()
    }
    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
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

// Test-only finite packet collector; no image/set/driver handles or new issuer.
#[cfg(test)]
pub(crate) fn mlp_diagnostic_manifests(
    rows: usize,
) -> Result<Vec<(ProjectionImage, CubinKernelManifest)>> {
    if !matches!(rows, 1 | 64) {
        return Err(Error::Unsupported(
            "2B MLP diagnostic initial1/64 only".into(),
        ));
    }
    let mut out = Vec::new();
    for role in [Role::Qkvz, Role::Ba] {
        out.extend(
            PacketScope::Qwen2bMlp
                .launches(rows, role)?
                .into_iter()
                .map(|p| (p.image, p.manifest)),
        );
    }
    Ok(out)
}
