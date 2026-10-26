//! One finite native union for the original47 source-produced layer0 MLP cut.
use super::cubin_manifests::{ProjectionImage, WORKSPACE_BYTES};
use super::projection_cubin::{
    CubinActivationImage, CubinQualificationInputs, native_error, not_cancelled,
    resident::{AdmittedImages, PreparedImages},
    validate_activation_image, validate_image_set,
};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::resident_program::qwen2b_component::{
    ComponentCut, ComponentKind, ComponentRecipe, ComponentValues,
};
use cubecl::cuda::{
    CubinBinding, CubinIo, CubinKernelManifest, CubinLaunchAttributes, CubinParameter,
    CubinPointerFixup, OwnedCudaKernel,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_core::accelerator::program::ValueId;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};
use zerogravity_core::{Error, Result};
mod manifests;
mod weights;
pub(crate) use weights::FixedPromptMlpWeights;

pub struct FixedPromptMlpQualificationInputs {
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

pub(crate) fn required_images() -> BTreeSet<ProjectionImage> {
    BTreeSet::from([
        ProjectionImage::AmpereBf16Gemm128x64Sm80,
        ProjectionImage::AmpereBf16Gemm64x64Stages64x5Sm80,
        ProjectionImage::Bf16SplitKReduceSm86,
        ProjectionImage::FixedPromptCutlassBf16Sm80,
    ])
}
pub(crate) fn validate_inputs(
    recipe: &ComponentRecipe,
    inputs: &FixedPromptMlpQualificationInputs,
) -> Result<()> {
    weights::require_recipe(recipe)?;
    manifests::require_memory_semantics()?;
    let mut required =
        super::fixed_prompt_projection::component::required_post_attention_images(recipe)?;
    required.insert(super::post_attention_projection::required_mlp_image(
        recipe,
    )?);
    for index in [8, 10] {
        required.extend(
            manifests::launches(index)?
                .into_iter()
                .map(|launch| launch.image),
        );
    }
    if required != required_images() {
        return Err(Error::State(
            "fixed_prompt source-derived exact native image union".into(),
        ));
    }
    let activation = inputs
        .activation
        .as_ref()
        .ok_or_else(|| Error::State("fixed_prompt actual activation image required".into()))?;
    if inputs.projections.images.capacity() > 8
        || activation.bytes.capacity() > 2 * 1024 * 1024
        || activation.admission.bytes() > 2 * 1024 * 1024
    {
        return Err(Error::Limit(
            "fixed_prompt finite native image admissions".into(),
        ));
    }
    let source = recipe.source();
    validate_image_set(source, &required, &inputs.projections.images)?;
    validate_activation_image(source, activation)
}

pub(crate) struct FixedPromptMlpPrograms {
    values: Arc<ComponentValues>,
    weights: Arc<FixedPromptMlpWeights>,
    prefix: Arc<super::fixed_prompt_projection::component::ConnectedProjectionPrograms>,
    out: Arc<super::post_attention_projection::OutPrograms>,
    launches: BTreeMap<usize, Vec<PreparedLaunch>>,
    _images: Arc<PreparedImages>,
    _metadata: Arc<zerogravity_artifacts::source_weights::WindowWorkspace>,
}
impl FixedPromptMlpPrograms {
    pub(crate) fn prepare(
        values: Arc<ComponentValues>,
        weights: Arc<FixedPromptMlpWeights>,
        inputs: FixedPromptMlpQualificationInputs,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        not_cancelled(cancelled)?;
        validate_inputs(values.recipe(), &inputs)?;
        weights.require_values(&values)?;
        let service = values.service();
        init.require_service(service)?;
        if zerogravity_core::resource_arbiter::ResourceArbiter::process_shared()
            .snapshot(&service.physical_key())?
            .limit
            != inputs.projections.device_cap
        {
            return Err(Error::State(
                "fixed_prompt original physical native envelope".into(),
            ));
        }
        init.retain_completion_owner(values.clone())?;
        init.retain_completion_owner(weights.clone())?;
        let metadata = Arc::new(
            values
                .recipe()
                .source()
                .reserve_workspace(2 * 1024 * 1024)?,
        );
        init.retain_completion_owner(metadata.clone())?;
        // The existing initializer retains all pending source windows before loading.
        let pending = super::fixed_prompt_projection::component::PendingMlpPrefixWeights::upload(
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
        let images = admitted.prepare_fixed_prompt_mlp_images(sources, init, cancelled)?;
        let (prefix, out) =
            super::fixed_prompt_projection::component::prepare_mlp_prefix_projections(
                values.clone(),
                images.clone(),
                pending,
                init,
                cancelled,
            )?;
        let mut launches = BTreeMap::new();
        for index in [8, 10] {
            let mut prepared = Vec::new();
            for launch in manifests::launches(index)? {
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
        let activation = super::activation_manifest::fixed_prompt_manifest();
        // SAFETY: the separately accepted original47 full ABI and same sealed resource.
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
            _metadata: metadata,
        });
        init.retain_completion_owner(programs.clone())?;
        Ok(programs)
    }
    fn node(
        &self,
        op: &zerogravity_core::accelerator::program::SourceOperator,
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
            .ok_or_else(|| Error::State("fixed_prompt exact original native node issuer".into()))?;
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
                "fixed_prompt SAME native producer/consumer leases".into(),
            ));
        }
        Ok(index)
    }
    pub(crate) fn accesses(
        &self,
        op: &zerogravity_core::accelerator::program::SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.weights.require_values(&self.values)?;
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
                    "fixed_prompt finite nonalias original native operands".into(),
                ));
            }
        }
        let (x, y) = match index {
            8 => (96_256, 673_792),
            9 => (673_792, 336_896),
            10 => (336_896, 96_256),
            _ => unreachable!(),
        };
        if input.bytes() != x
            || output.bytes() != y
            || workspace.bytes() != WORKSPACE_BYTES
            || workspace.role() != BufferRole::Scratch
        {
            return Err(Error::State(
                "fixed_prompt full native operand extents".into(),
            ));
        }
        let mut accesses = vec![
            (input.clone(), DeviceAccess::Read),
            (output.clone(), DeviceAccess::Write),
        ];
        if index != 9 {
            accesses.push((self.weights.lease(index)?.clone(), DeviceAccess::Read));
        }
        if index == 10 {
            accesses.push((workspace.clone(), DeviceAccess::Write));
        }
        Ok(accesses)
    }
    pub(crate) fn enqueue(
        &self,
        op: &zerogravity_core::accelerator::program::SourceOperator,
        input: &ResidentLease,
        output: &ResidentLease,
        workspace: &ResidentLease,
        cancelled: Option<&AtomicBool>,
    ) -> Result<usize> {
        not_cancelled(cancelled)?;
        self.accesses(op, input, output, workspace)?;
        let index = self.node(op, input, output)?;
        if index == 2 || index == 3 {
            return self.prefix.enqueue(op, input, output, workspace, cancelled);
        }
        if index == 6 {
            return self.out.enqueue(cancelled);
        }
        let launches = self
            .launches
            .get(&index)
            .ok_or_else(|| Error::State("fixed_prompt prepared native node absent".into()))?;
        for launch in launches {
            not_cancelled(cancelled)?;
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
            && self.weights.retained_values(values)
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

/// Existing GateUp/parallelDown+reducer packet primitives; actual model source nodes bind separately.
pub(crate) fn model_packets(
    projection_node: usize,
) -> Result<Vec<super::fixed_prompt_model_native::Packet>> {
    use super::fixed_prompt_model_native::{Binding as B, Packet};
    manifests::require_memory_semantics()?;
    let key = match projection_node {
        8 => 8,
        10 => 10,
        _ => {
            return Err(Error::Unsupported(
                "finite model47 GateUp/Down packet primitive only".into(),
            ));
        }
    };
    manifests::launches(key)?
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
