//! Finite InitialOne/Initial64 2B source-produced eleven-node native union.
use super::cubin_manifests::{ProjectionImage, WORKSPACE_BYTES};
use super::projection_cubin::{
    CubinActivationImage, CubinQualificationInputs, native_error, not_cancelled,
    resident::{AdmittedImages, PreparedImages},
    validate_activation_image, validate_image_set,
};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentInitialization, ResidentLease};
use crate::resident_program::qwen2b_component::{
    ComponentCut, ComponentKind, ComponentRecipe, ComponentValues,
};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, OwnedCudaKernel,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use zerogravity_core::accelerator::program::{ValueId, Vllm08Operator};
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
mod manifests;
#[cfg(test)]
mod tests;
use crate::resident_leases::Qwen2bMlpWeights;
use crate::vllm_cuda_gdn::Qwen2bGdnArithmetic;

pub struct Qwen2bMlpQualificationInputs {
    pub projections: CubinQualificationInputs,
    /// None is accepted only by entry cancellation or cold admission-pressure probes.
    pub activation: Option<CubinActivationImage>,
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

pub(crate) fn require_recipe(recipe: &ComponentRecipe) -> Result<()> {
    super::post_attention_projection::require_qwen2b_mlp_recipe(recipe)?;
    if recipe.cut() != ComponentCut::Qwen2bMlp
        || recipe.kind() != ComponentKind::Qwen2b
        || recipe.qwen2b_arithmetic() != Some(Qwen2bGdnArithmetic::OwnedSolve4Recompute3)
        || recipe.nodes().len() != 11
        || !recipe.qwen2b_mlp_record_admission_retained()
    {
        return Err(Error::Unsupported(
            "explicit owned2B eleven-node MLP source cut required".into(),
        ));
    }
    Ok(())
}
pub(crate) fn required_images(rows: usize) -> Result<BTreeSet<ProjectionImage>> {
    use ProjectionImage as I;
    Ok(match rows {
        1 => BTreeSet::from([
            I::CutlassWmmaSm80,
            I::Bf16GemvxSm86,
            I::Bf16Gemv2TSm86,
            I::Bf16SplitKReduceSm86,
        ]),
        64 => BTreeSet::from([
            I::AmpereBf16Gemm128x64Sm80,
            I::CutlassWmmaSm80,
            I::Bf16SplitKReduceSm86,
            I::Canonical47CutlassBf16Sm80,
        ]),
        _ => {
            return Err(Error::Unsupported(
                "closed2B MLP initial1/64 rows only".into(),
            ));
        }
    })
}
pub(crate) fn validate_inputs(
    recipe: &ComponentRecipe,
    inputs: &Qwen2bMlpQualificationInputs,
) -> Result<()> {
    require_recipe(recipe)?;
    manifests::require_memory_semantics()?;
    let rows = recipe.record().call().rows();
    let mut required = super::qwen2b_projection::required_images(&[recipe.record().clone()])?;
    required.insert(super::post_attention_projection::required_qwen2b_mlp_image(
        recipe,
    )?);
    for index in [8, 10] {
        required.extend(
            manifests::launches(rows, index)?
                .into_iter()
                .map(|launch| launch.image),
        );
    }
    if required != required_images(rows)? {
        return Err(Error::State(
            "2B source-derived exact native image union".into(),
        ));
    }
    let activation = inputs
        .activation
        .as_ref()
        .ok_or_else(|| Error::State("2B actual activation image required".into()))?;
    if inputs.projections.images.capacity() > 8
        || inputs.projections.images.iter().any(|image| {
            image.bytes.capacity() > 16 * 1024 * 1024 || image.admission.bytes() > 16 * 1024 * 1024
        })
        || activation.bytes.capacity() > 2 * 1024 * 1024
        || activation.admission.bytes() > 2 * 1024 * 1024
    {
        return Err(Error::Limit("2B finite native image admissions".into()));
    }
    let source = recipe.source().source().source();
    validate_image_set(source, &required, &inputs.projections.images)?;
    validate_activation_image(source, activation)
}

pub(crate) struct Qwen2bMlpPrograms {
    values: Arc<ComponentValues>,
    weights: Arc<Qwen2bMlpWeights>,
    prefix: Arc<super::qwen2b_projection::ProjectionPrograms>,
    out: Arc<super::post_attention_projection::OutPrograms>,
    launches: BTreeMap<usize, Vec<PreparedLaunch>>,
    _images: Arc<PreparedImages>,
    native_launches: AtomicUsize,
    down1_fault: AtomicUsize,
    down1_enqueues: AtomicUsize,
    _metadata: Arc<zerogravity_artifacts::source_weights::WindowWorkspace>,
}
impl Qwen2bMlpPrograms {
    pub(crate) fn prepare(
        values: Arc<ComponentValues>,
        weights: Arc<Qwen2bMlpWeights>,
        inputs: Qwen2bMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        validate_inputs(values.recipe(), &inputs)?;
        weights.require_record(values.recipe().record(), values.service())?;
        let service = values.service();
        init.require_service(service)?;
        if zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.projections.device_cap
        {
            return Err(Error::State(
                "qwen2b original physical native envelope".into(),
            ));
        }
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(weights.clone())?;
        let metadata = Arc::new(
            values
                .recipe()
                .source()
                .source()
                .source()
                .reserve_workspace(2 * 1024 * 1024)?,
        );
        init.retain_completion_owner(metadata.clone())?;
        // Actual complete source packs and copies are retained on this SAME init.
        let pending = super::qwen2b_projection::PendingProjectionWeights::upload(
            values.recipe().source().clone(),
            values.recipe().record(),
            service,
            init,
            cancelled,
        )?;
        let pending_out = super::post_attention_projection::PendingOutWeights::upload_qwen2b_mlp(
            &values, init, cancelled,
        )?;
        let sources = service
            .pinned_cuda_queue()?
            .verify_bf16_blas_sources(&inputs.projections.cublas, &inputs.projections.cublas_lt)
            .map_err(native_error)?;
        let owner: CompletionOwner = values.recipe().clone();
        let admitted = AdmittedImages::reserve_with_activation(
            service,
            owner,
            inputs.projections.images,
            inputs.activation,
        )?;
        init.retain_completion_owner(admitted.completion_owner())?;
        let images = admitted.prepare_qwen2b_mlp_images(
            values.recipe().record().call().rows(),
            sources,
            init,
            cancelled,
        )?;
        let prefix =
            super::qwen2b_projection::ProjectionPrograms::prepare_pending_qwen2b_mlp_upload(
                images.clone(),
                pending,
                values.recipe(),
                init,
                cancelled,
            )?;
        init.retain_completion_owner(prefix.clone())?;
        let out = super::post_attention_projection::OutPrograms::prepare_qwen2b_mlp(
            images.clone(),
            pending_out,
            values.clone(),
            init,
            cancelled,
        )?;
        init.retain_completion_owner(out.clone())?;
        let rows = values.recipe().record().call().rows();
        let mut launches = BTreeMap::new();
        for index in [8, 10] {
            let mut prepared = Vec::new();
            for launch in manifests::launches(rows, index)? {
                not_cancelled(cancelled)?;
                // SAFETY: finite complete packet, same source/charge and armed init.
                let kernel = unsafe {
                    images
                        .set()
                        .prepare(images.image(launch.image)?, launch.manifest)
                }
                .map_err(native_error)?;
                init.retain_completion_owner(kernel.clone())?;
                prepared.push(PreparedLaunch {
                    roles: launch.roles,
                    kernel,
                });
            }
            launches.insert(index, prepared);
        }
        let activation = manifests::activation(rows)?;
        // SAFETY: separate finite6144 ABI gate and same sealed captured resource.
        let kernel = unsafe { images.set().prepare(images.activation()?, activation) }
            .map_err(native_error)?;
        init.retain_completion_owner(kernel.clone())?;
        launches.insert(
            9,
            vec![PreparedLaunch {
                roles: vec![Binding::X, Binding::Y],
                kernel,
            }],
        );
        let programs = Arc::new(Self {
            values,
            weights,
            prefix,
            out,
            launches,
            _images: images,
            native_launches: AtomicUsize::new(0),
            down1_fault: AtomicUsize::new(0),
            down1_enqueues: AtomicUsize::new(0),
            _metadata: metadata,
        });
        init.retain_completion_owner(programs.clone())?;
        Ok(programs)
    }
    pub(crate) fn arm_down1_fault(&self, cancel: bool) -> Result<()> {
        require_recipe(self.values.recipe())?;
        self.weights
            .require_record(self.values.recipe().record(), self.values.service())?;
        require_down1_arm(
            self.values.recipe().record().call().rows(),
            self.native_launches(),
            self.values.lease(ValueId(12))?.is_initialized(),
        )?;
        self.down1_fault
            .compare_exchange(
                0,
                if cancel { 2 } else { 1 },
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map_err(|_| Error::State("2B Down1 partial fault already armed".into()))?;
        Ok(())
    }
    pub(crate) fn native_launches(&self) -> usize {
        self.native_launches.load(Ordering::Acquire)
    }
    pub(crate) fn down1_output_unpromoted(&self) -> Result<bool> {
        Ok(self.values.recipe().record().call().rows() == 1
            && !self.values.lease(ValueId(12))?.is_initialized())
    }
    pub(crate) fn down1_enqueues(&self) -> usize {
        self.down1_enqueues.load(Ordering::Acquire)
    }
    fn node(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
    ) -> Result<usize> {
        let index = [2, 3, 6, 8, 9, 10]
            .into_iter()
            .find(|&index| {
                self.values
                    .recipe()
                    .operator(index)
                    .is_ok_and(|expected| expected == op)
            })
            .ok_or_else(|| Error::State("qwen2b exact original native node issuer".into()))?;
        let (x, y) = match index {
            2 => (1, 2),
            3 => (1, 3),
            6 => (6, 7),
            8 => (8, 10),
            9 => (10, 11),
            10 => (11, 12),
            _ => unreachable!(),
        };
        if input.id() != self.values.lease(ValueId(x))?.id()
            || output.id() != self.values.lease(ValueId(y))?.id()
        {
            return Err(Error::State(
                "qwen2b SAME native producer/consumer leases".into(),
            ));
        }
        Ok(index)
    }
    pub(crate) fn accesses(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.weights
            .require_record(self.values.recipe().record(), self.values.service())?;
        let index = self.node(op, input, output)?;
        if index == 2 || index == 3 {
            return self.prefix.accesses(op, input, output, workspace);
        }
        if index == 6 {
            self.out.require_operator(op, input, output)?;
            return self.out.accesses();
        }
        let mut ids = BTreeSet::new();
        for lease in [
            input,
            output,
            workspace,
            self.weights.lease(8)?,
            self.weights.lease(10)?,
        ] {
            if !ids.insert(lease.id())
                || !Arc::ptr_eq(lease.service(), self.values.service())
                || lease.source_identity() != self.values.service().source_identity()
                || lease.service_generation() != self.values.service().generation()
            {
                return Err(Error::State(
                    "qwen2b finite nonalias original native operands".into(),
                ));
            }
        }
        let rows = self.values.recipe().record().call().rows();
        let (x, y) = match index {
            8 => (rows * 4096, rows * 24_576),
            9 => (rows * 24_576, rows * 12_288),
            10 => (rows * 12_288, rows * 4096),
            _ => unreachable!(),
        };
        if input.bytes() != x
            || output.bytes() != y
            || workspace.bytes() != WORKSPACE_BYTES
            || workspace.role() != BufferRole::Scratch
        {
            return Err(Error::State("qwen2b full native operand extents".into()));
        }
        let mut accesses = vec![
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        if index != 9 {
            accesses.push((self.weights.lease(index)?.clone(), DeviceAccess::Read));
        }
        if index == 10 && rows == 1 {
            accesses.push((workspace.clone(), DeviceAccess::Write));
        }
        Ok(accesses)
    }
    pub(crate) fn enqueue(
        &self,
        op: &Vllm08Operator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.accesses(op, input, output, workspace)?;
        let index = self.node(op, input, output)?;
        if index == 2 || index == 3 {
            let count = self
                .prefix
                .enqueue(op, input, output, workspace, cancelled)?;
            self.native_launches.fetch_add(count, Ordering::AcqRel);
            return Ok(count);
        }
        if index == 6 {
            let count = self.out.enqueue(cancelled)?;
            self.native_launches.fetch_add(count, Ordering::AcqRel);
            return Ok(count);
        }
        let launches = self
            .launches
            .get(&index)
            .ok_or_else(|| Error::State("qwen2b prepared native node absent".into()))?;
        let down1 = index == 10 && self.values.recipe().record().call().rows() == 1;
        let late_cancel = AtomicBool::new(false);
        for (ordinal, launch) in launches.iter().enumerate() {
            not_cancelled(if late_cancel.load(Ordering::Acquire) {
                Some(&late_cancel)
            } else {
                cancelled
            })?;
            if down1 && output.is_initialized() {
                return Err(Error::State(
                    "2B Down1 output promoted inside native group".into(),
                ));
            }
            let handles = launch
                .roles
                .iter()
                .map(|role| {
                    Ok(match role {
                        Binding::W => self.weights.lease(index)?.handle(),
                        Binding::X => input.handle(),
                        Binding::Y => output.handle(),
                        Binding::Workspace => workspace.handle(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // SAFETY: original armed graph retains these exact nonalias operands.
            unsafe { launch.kernel.enqueue(handles) }.map_err(native_error)?;
            self.native_launches.fetch_add(1, Ordering::AcqRel);
            if down1 {
                self.down1_enqueues.fetch_add(1, Ordering::AcqRel);
            }
            if down1 && ordinal == 0 {
                // Conservative dead-C Write is no Y initialization certificate.
                if output.is_initialized() {
                    return Err(Error::State(
                        "2B Down1 main initialized dead-C output".into(),
                    ));
                }
                after_down1_main(self.down1_fault.load(Ordering::Acquire), &late_cancel)?;
            }
        }
        Ok(launches.len())
    }
    pub(crate) fn immutable_weights(&self) -> Vec<(ResidentLease, &[u8])> {
        let mut weights = self.prefix.immutable_weights();
        weights.extend(self.out.immutable_weights());
        weights
    }
    pub(crate) fn retained_values(&self, values: &Arc<ComponentValues>) -> bool {
        Arc::ptr_eq(&self.values, values)
            && self.weights.retained_record(
                values.recipe().source().source(),
                values.recipe().record(),
                values.service(),
            )
            && self.out.retained_values(values)
    }
    pub(crate) unsafe fn retire_after_physical_completion(&self) -> Result<()> {
        for launch in self.launches.values().flatten() {
            unsafe { launch.kernel.retire_after_physical_completion() }.map_err(native_error)?;
        }
        unsafe { self.out.retire_kernels_after_physical_completion() }?;
        // The prefix retires its records, then the ONE shared set last.
        unsafe { self.prefix.retire_after_physical_completion() }
    }
}

fn require_down1_arm(rows: usize, launches: usize, initialized: bool) -> Result<()> {
    if rows != 1 || launches != 0 || initialized {
        return Err(Error::State(
            "2B Down1 fault requires fresh actualInitial1 source cut".into(),
        ));
    }
    Ok(())
}
fn after_down1_main(fault: usize, late_cancel: &AtomicBool) -> Result<()> {
    match fault {
        0 => Ok(()),
        1 => Err(Error::State(
            "deliberate qwen2b MLP unknown after actual Down main".into(),
        )),
        2 => {
            late_cancel.store(true, Ordering::Release);
            Ok(())
        }
        _ => Err(Error::State("2B Down1 partial fault mode differs".into())),
    }
}
