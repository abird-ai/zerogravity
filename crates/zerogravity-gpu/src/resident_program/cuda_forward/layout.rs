//! Cold forward-Qwen CUDA physical/resource analysis.
//!
//! This consumes only the immutable call-free ForwardPlan. It performs no
//! source reads, GPU allocations, provider creation, request binding or launch.

use super::rotary;
use crate::{
    cuda::gdn::{
        GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel,
        GdnRuntimeBufferOwnership, GdnRuntimeConfig, GdnRuntimeKernelGeometry,
        GdnRuntimeWorkspaceLayout, GdnStateKernel,
    },
    resident_leases::forward_parameter_upload::ForwardParameterStorage,
    resident_program::{
        ResidentBufferLayout, ResidentPhysicalType, ResidentPreflight, StateLayout,
        StateRequirement,
    },
};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, NumericalContract, PlanOperation, SourceOperatorKind, SourceParameter,
        SourceTensor, StateSlot, ValueId, ValueStorageBinding, ValueStorageLimits,
        ValueStorageRetention, ValueStorageReusePolicy,
    },
};

pub(super) const MAX_ROWS: usize = 64;
const MAX_VALUES: usize = 8192;
const MAX_NODES: usize = 8192;
const VALUE_ALIGNMENT: usize = 512;
const VALUE_MAX_SLOT_BYTES: usize = 16 * 1024 * 1024;
const VALUE_MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;
// Keep the original conservative admission envelope while qualifying the new
// executor. GEMM itself has no device scratch or opaque-provider allocation;
// reclaiming this headroom is a separate admission-policy change.
const ADMISSION_HEADROOM_BYTES: usize = 64 * 1024 * 1024 + 8_519_680 + 4;
const TOKEN_BYTES: usize = MAX_ROWS * 4;
const INVERSE_FREQUENCY_BYTES: usize = 32 * 4;
const STATUS_BYTES: usize = 4;

#[derive(Clone)]
pub(super) struct MatrixPackSpec {
    members: Arc<[SourceTensor]>,
    bytes: usize,
}
impl MatrixPackSpec {
    pub(super) fn members(&self) -> &[SourceTensor] {
        &self.members
    }
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Clone)]
pub(super) struct ParameterSpec {
    descriptor: SourceParameter,
    storage: ForwardParameterStorage,
    bytes: usize,
}
impl ParameterSpec {
    pub(super) fn descriptor(&self) -> &SourceParameter {
        &self.descriptor
    }
    pub(super) fn storage(&self) -> ForwardParameterStorage {
        self.storage
    }
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Clone)]
pub(super) struct CudaForwardLayout {
    plan: Arc<ForwardPlan>,
    geometry: crate::cuda::text_forward::Geometry,
    groups: crate::cuda::gdn::head_tile::HeadGroups,
    value_layouts: Arc<[ResidentBufferLayout]>,
    value_bindings: Arc<[ValueStorageBinding]>,
    slot_bytes: Arc<[usize]>,
    states: Arc<[StateRequirement]>,
    matrices: Arc<[MatrixPackSpec]>,
    parameters: Arc<[ParameterSpec]>,
    gdn: GdnRuntimeWorkspaceLayout,
    preflight: ResidentPreflight,
    resident_weights: bool,
}
impl CudaForwardLayout {
    pub(super) fn analyze(plan: Arc<ForwardPlan>) -> Result<Arc<Self>> {
        require_plan_shape(&plan)?;
        let (geometry, groups) = super::geometry::derive(&plan)?;
        let value_layouts = plan
            .values()
            .iter()
            .map(|shape| {
                ResidentBufferLayout::new(ResidentPhysicalType::Bf16, shape.capacity_elements()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let residual_aliases = residual_aliases(&plan)?;
        let mut pinned = residual_aliases
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        pinned.sort_unstable_by_key(|value| value.0);
        pinned.dedup();
        let storage = ResidentBufferLayout::plan_values(
            &plan,
            &value_layouts,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &pinned,
                taps: &[],
            },
            ValueStorageLimits {
                max_nodes: MAX_NODES,
                max_values: MAX_VALUES,
                max_slots: MAX_VALUES,
                max_slot_bytes: VALUE_MAX_SLOT_BYTES,
                max_total_bytes: VALUE_MAX_TOTAL_BYTES,
                alignment: VALUE_ALIGNMENT,
                reuse_policy: ValueStorageReusePolicy::ExactAlignedSize,
            },
        )?;
        storage.require_plan(&plan)?;
        let mut slot_bytes = vec![0usize; storage.slots().len()];
        for binding in storage.bindings() {
            let reserved = storage.slots()[binding.slot].reserved_bytes;
            if binding.bytes != reserved {
                return Err(Error::Unsupported(
                    "CUDA forward value slot requires padded/subview ownership".into(),
                ));
            }
            let prior = &mut slot_bytes[binding.slot];
            if *prior != 0 && *prior != binding.bytes {
                return Err(Error::Unsupported(
                    "CUDA forward value slot logical extents differ".into(),
                ));
            }
            *prior = binding.bytes;
        }
        if slot_bytes.contains(&0) || slot_bytes.iter().sum::<usize>() != storage.total_bytes() {
            return Err(Error::State(
                "CUDA forward value slot coverage differs".into(),
            ));
        }

        let states = state_requirements(&plan)?;
        let state_bytes = checked_sum(states.iter().map(|state| state.bytes))?;
        let (matrices, parameters) = immutable_specs(&plan)?;
        let matrix_bytes = checked_sum(matrices.iter().map(MatrixPackSpec::bytes))?;
        let parameter_bytes = checked_sum(parameters.iter().map(ParameterSpec::bytes))?;

        let gdn = GdnRuntimeWorkspaceLayout::new(gdn_runtime_config()?)?;
        let gdn_workspace_bytes = checked_sum(
            gdn.buffers()
                .iter()
                .filter(|buffer| {
                    owns_gdn_buffer(buffer.role(), buffer.ownership(), !groups.direct())
                })
                .map(|buffer| buffer.bytes()),
        )?;
        let workspace_bytes = checked_sum([
            gdn_workspace_bytes,
            if groups.direct() {
                0
            } else {
                crate::cuda::gdn::state_transport::CONV_ELEMENTS * 4
            },
            TOKEN_BYTES,
            geometry.vocabulary * 4,
            rotary::scratch_bytes(geometry.query_heads, geometry.kv_heads),
        ])?;
        let mut value_bindings = storage.bindings().to_vec();
        for (value, alias) in residual_aliases.iter().enumerate() {
            let Some(root) = alias else {
                continue;
            };
            let root = root.0 as usize;
            if value_layouts[value] != value_layouts[root]
                || value_bindings[value].bytes != value_bindings[root].bytes
            {
                return Err(Error::State(
                    "CUDA forward residual alias physical layout differs".into(),
                ));
            }
            value_bindings[value] = value_bindings[root];
        }
        let value_bindings: Arc<[ValueStorageBinding]> = value_bindings.into();
        let value_total_bytes = storage.total_bytes();
        drop(storage);

        let preflight = ResidentPreflight {
            source: plan.source(),
            source_bytes: matrix_bytes,
            value_bytes: value_total_bytes,
            workspace_bytes,
            status_bytes: STATUS_BYTES,
            parameter_bytes: checked_sum([
                parameter_bytes,
                INVERSE_FREQUENCY_BYTES,
                rotary::immutable_bytes(),
            ])?,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states: states.to_vec(),
            scratch: [0; super::super::SCRATCH_SLOTS],
        };
        // Exercise every checked sum before returning cold authority metadata.
        preflight.total_with_state()?;
        Ok(Arc::new(Self {
            plan,
            geometry,
            groups,
            value_layouts: value_layouts.into(),
            value_bindings,
            slot_bytes: slot_bytes.into(),
            states: states.into(),
            matrices: matrices.into(),
            parameters: parameters.into(),
            gdn,
            preflight,
            resident_weights: true,
        }))
    }

    pub(super) fn resident_weights(&self) -> bool {
        self.resident_weights
    }
    pub(super) fn stream_weights(&mut self) -> Result<()> {
        if self.resident_weights {
            self.preflight.source_bytes = 0;
            self.preflight.workspace_bytes = self
                .preflight
                .workspace_bytes
                .checked_add(crate::resident_leases::source_page::PAGE_BYTES)
                .ok_or_else(|| Error::Limit("CUDA forward source page inventory".into()))?;
            self.resident_weights = false;
        }
        Ok(())
    }

    pub(super) fn groups(&self) -> crate::cuda::gdn::head_tile::HeadGroups {
        self.groups
    }

    pub(super) fn geometry(&self) -> crate::cuda::text_forward::Geometry {
        self.geometry
    }

    pub(super) fn plan(&self) -> &Arc<ForwardPlan> {
        &self.plan
    }
    pub(super) fn value_layouts(&self) -> &[ResidentBufferLayout] {
        &self.value_layouts
    }
    pub(super) fn value_bindings(&self) -> &[ValueStorageBinding] {
        &self.value_bindings
    }
    pub(super) fn slot_bytes(&self) -> &[usize] {
        &self.slot_bytes
    }
    pub(super) fn states(&self) -> &[StateRequirement] {
        &self.states
    }
    pub(super) fn matrices(&self) -> &[MatrixPackSpec] {
        &self.matrices
    }
    pub(super) fn parameters(&self) -> &[ParameterSpec] {
        &self.parameters
    }
    pub(super) fn gdn(&self) -> &GdnRuntimeWorkspaceLayout {
        &self.gdn
    }
    pub(super) fn preflight(&self) -> &ResidentPreflight {
        &self.preflight
    }
    pub(super) fn total_admission_bytes(&self) -> Result<usize> {
        checked_sum([self.preflight.total_with_state()?, ADMISSION_HEADROOM_BYTES])
    }
}

pub(super) fn owns_gdn_buffer(
    role: crate::cuda::gdn::GdnRuntimeBuffer,
    ownership: GdnRuntimeBufferOwnership,
    grouped: bool,
) -> bool {
    use crate::cuda::gdn::GdnRuntimeBuffer::*;
    matches!(
        ownership,
        GdnRuntimeBufferOwnership::CacheState
            | GdnRuntimeBufferOwnership::Control
            | GdnRuntimeBufferOwnership::Scratch
    ) || matches!(role, Rstd | InitialState | FinalState)
        || grouped
            && matches!(
                role,
                PackedQkvz | PackedBa | ConvOutput | ZCompact | Gated | ConvWeights
            )
}

pub(super) fn gdn_runtime_config() -> Result<GdnRuntimeConfig> {
    Ok(GdnRuntimeConfig::new(
        GdnKernelConfig::from_stages(
            GdnKktKernel::Bk32W8S3,
            GdnRecomputeKernel::Bk64Bv64W8S3,
            GdnStateKernel::Bv64W4S2,
            GdnOutputKernel::Bk32Bv64W4S3,
        )?,
        GdnRuntimeKernelGeometry::new(6144, 2048, 32, 4, 16, 128, 64, 5)?,
    ))
}

fn require_plan_shape(plan: &ForwardPlan) -> Result<()> {
    if plan.text_invocation().is_some()
        || plan.frontier_delta() != 0
        || plan.nodes().is_empty()
        || plan.nodes().len() > MAX_NODES
        || plan.values().is_empty()
        || plan.values().len() > MAX_VALUES
        || plan.logits().is_none()
        || plan.prefix_capacity() < 512
        || plan.prefix_capacity() > 32_768
        || plan.required_domains().is_empty()
        || plan.required_domains().len() > MAX_NODES
        || plan.proposed_domains().len() != plan.required_domains().len()
    {
        return Err(Error::Unsupported(
            "CUDA forward bounded invocation-free source plan".into(),
        ));
    }
    for (index, node) in plan.nodes().iter().enumerate() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(Error::Unsupported(
                "CUDA forward plan contains non-template operation".into(),
            ));
        };
        if descriptor.source() != plan.source()
            || descriptor.max_rows() != MAX_ROWS
            || descriptor.numerical_contract() != NumericalContract::CudaSm86Bf16ReferenceV1
        {
            return Err(Error::State(
                "CUDA forward template source/rows/vocabulary differs".into(),
            ));
        }
        if matches!(descriptor.kind(), SourceOperatorKind::Projection { .. }) {
            super::arithmetic::projection_use(plan, index)?;
        }
        use SourceOperatorKind::*;
        if !matches!(
            descriptor.kind(),
            EmbeddingRows
                | Projection { selected_row: None }
                | Rms(_)
                | ResidualRms(_)
                | CausalConvSplit { .. }
                | GatedDelta { .. }
                | QkRmsRotaryGated(_)
                | KvAppend { .. }
                | FlashAttention { .. }
                | Sigmoid { .. }
                | Multiply { .. }
                | SiluAndMul { .. }
        ) {
            return Err(Error::Unsupported(
                "CUDA forward source operator implementation".into(),
            ));
        }
    }
    Ok(())
}

fn residual_aliases(plan: &ForwardPlan) -> Result<Vec<Option<ValueId>>> {
    let mut aliases = vec![None; plan.values().len()];
    for node in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(Error::State(
                "CUDA forward residual alias requires source template plan".into(),
            ));
        };
        if !matches!(descriptor.kind(), SourceOperatorKind::ResidualRms(_)) {
            continue;
        }
        let ([_, residual], [_, carried]) = (node.inputs.as_slice(), node.outputs.as_slice())
        else {
            return Err(Error::State(
                "CUDA forward residual RMS exact two-input/two-output arity".into(),
            ));
        };
        let residual_index = residual.0 as usize;
        let carried_index = carried.0 as usize;
        let root = aliases
            .get(residual_index)
            .copied()
            .flatten()
            .unwrap_or(*residual);
        let root_index = root.0 as usize;
        if residual.0 >= carried.0
            || aliases
                .get(carried_index)
                .is_none_or(|current| current.is_some())
            || plan.values().get(root_index) != plan.values().get(carried_index)
        {
            return Err(Error::State(
                "CUDA forward residual in-place alias chain differs".into(),
            ));
        }
        aliases[carried_index] = Some(root);
    }
    Ok(aliases)
}

fn state_requirements(plan: &ForwardPlan) -> Result<Vec<StateRequirement>> {
    let mut writers = BTreeMap::new();
    for node in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(Error::Unsupported(
                "CUDA forward state source template".into(),
            ));
        };
        let state = match descriptor.kind() {
            SourceOperatorKind::CausalConvSplit { domain, geometry } => {
                let elements = geometry
                    .kernel
                    .checked_sub(1)
                    .and_then(|history| history.checked_mul(geometry.qkv_channels))
                    .ok_or_else(|| Error::Limit("CUDA forward convolution state extent".into()))?;
                state_vector(StateSlot::Conv(domain), elements, ResidentPhysicalType::F32)?
            }
            SourceOperatorKind::GatedDelta { domain, geometry } => {
                let elements = geometry
                    .value_heads
                    .checked_mul(geometry.key_width)
                    .and_then(|n| n.checked_mul(geometry.value_width))
                    .ok_or_else(|| Error::Limit("CUDA forward recurrent state extent".into()))?;
                state_vector(
                    StateSlot::Recurrent(domain),
                    elements,
                    ResidentPhysicalType::F32,
                )?
            }
            SourceOperatorKind::KvAppend {
                domain, key_width, ..
            } => {
                let elements = key_width
                    .checked_mul(plan.prefix_capacity())
                    .ok_or_else(|| Error::Limit("CUDA forward KV state extent".into()))?;
                let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, elements)?;
                StateRequirement {
                    slot: StateSlot::Kv(domain),
                    layout: StateLayout::Kv {
                        slab: key_width,
                        capacity: plan.prefix_capacity(),
                    },
                    physical: ResidentPhysicalType::Bf16,
                    bytes: layout
                        .bytes()?
                        .checked_mul(2)
                        .ok_or_else(|| Error::Limit("CUDA forward KV state bytes".into()))?,
                }
            }
            _ => continue,
        };
        state.plane_layout()?;
        if writers.insert(state.slot, state).is_some() {
            return Err(Error::State(
                "CUDA forward state has multiple writers".into(),
            ));
        }
    }
    if writers.len() != plan.required_domains().len()
        || plan.proposed_domains().len() != plan.required_domains().len()
    {
        return Err(Error::State(
            "CUDA forward state writer/domain coverage".into(),
        ));
    }
    plan.required_domains()
        .iter()
        .zip(plan.proposed_domains())
        .map(|(&slot, version)| {
            if version.slot != slot || version.ordinal != 1 {
                return Err(Error::State(
                    "CUDA forward proposed state slot/version differs".into(),
                ));
            }
            writers
                .remove(&slot)
                .ok_or_else(|| Error::State("CUDA forward state writer absent".into()))
        })
        .collect()
}

fn state_vector(
    slot: StateSlot,
    elements: usize,
    physical: ResidentPhysicalType,
) -> Result<StateRequirement> {
    let layout = ResidentBufferLayout::new(physical, elements)?;
    Ok(StateRequirement {
        slot,
        layout: StateLayout::Vector { elements },
        physical,
        bytes: layout.bytes()?,
    })
}

fn immutable_specs(plan: &ForwardPlan) -> Result<(Vec<MatrixPackSpec>, Vec<ParameterSpec>)> {
    let mut matrices: Vec<MatrixPackSpec> = Vec::new();
    let mut parameters: Vec<ParameterSpec> = Vec::new();
    for node in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            unreachable!("validated template plan")
        };
        if !descriptor.matrices().is_empty()
            && !matrices
                .iter()
                .any(|spec| spec.members() == descriptor.matrices())
        {
            let bytes = descriptor
                .matrices()
                .iter()
                .try_fold(0usize, |sum, matrix| {
                    let bytes = usize::try_from(matrix.packed_bytes()?)
                        .map_err(|_| Error::Limit("CUDA forward source matrix bytes".into()))?;
                    sum.checked_add(bytes)
                        .ok_or_else(|| Error::Limit("CUDA forward source matrix sum".into()))
                })?;
            matrices.push(MatrixPackSpec {
                members: descriptor.matrices().to_vec().into(),
                bytes,
            });
        }
        for (ordinal, parameter) in descriptor.parameters().iter().enumerate() {
            let storage = parameter_storage(descriptor.kind(), ordinal)?;
            if parameters
                .iter()
                .any(|spec| spec.descriptor() == parameter && spec.storage() == storage)
            {
                continue;
            }
            let bytes = parameter_resident_bytes(parameter, storage)?;
            parameters.push(ParameterSpec {
                descriptor: parameter.clone(),
                storage,
                bytes,
            });
        }
    }
    if matrices.is_empty() || parameters.is_empty() {
        return Err(Error::State(
            "CUDA forward immutable source population empty".into(),
        ));
    }
    Ok((matrices, parameters))
}

pub(super) fn parameter_storage(
    kind: SourceOperatorKind,
    ordinal: usize,
) -> Result<ForwardParameterStorage> {
    use ForwardParameterStorage::{Bf16Rne, F32Exact, Original};
    use SourceOperatorKind::*;
    match kind {
        Rms(_) | ResidualRms(_) if ordinal == 0 => Ok(Bf16Rne),
        QkRmsRotaryGated(_) if ordinal < 2 => Ok(Original),
        CausalConvSplit { .. } if ordinal == 0 => Ok(Original),
        GatedDelta { .. } => match ordinal {
            0 => Ok(F32Exact),
            1 => Ok(Original),
            2 => Ok(Bf16Rne),
            _ => Err(Error::State("CUDA forward GDN parameter ordinal".into())),
        },
        _ => Err(Error::State(
            "CUDA forward unexpected learned parameter role".into(),
        )),
    }
}

fn parameter_resident_bytes(
    parameter: &SourceParameter,
    storage: ForwardParameterStorage,
) -> Result<usize> {
    storage.resident_bytes(parameter)
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| Error::Limit("CUDA forward resource byte sum".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerogravity_core::accelerator::program::{ConvDomain, KvDomain, RecurrentDomain};

    #[test]
    fn typed_state_layout_is_exact_f32_f32_bf16_and_two_root_cost_is_derived() {
        let capacity = 32_768usize;
        let mut slots = Vec::new();
        for layer in 0..24 {
            if layer % 4 == 3 {
                slots.push(StateSlot::Kv(KvDomain(layer)));
            } else {
                slots.push(StateSlot::Conv(ConvDomain(layer)));
                slots.push(StateSlot::Recurrent(RecurrentDomain(layer)));
            }
        }
        assert_eq!(slots.len(), 42);
        let mut bytes = 0usize;
        for slot in slots {
            let requirement = match slot {
                StateSlot::Conv(_) => {
                    state_vector(slot, 6144 * 3, ResidentPhysicalType::F32).unwrap()
                }
                StateSlot::Recurrent(_) => {
                    state_vector(slot, 16 * 128 * 128, ResidentPhysicalType::F32).unwrap()
                }
                StateSlot::Kv(_) => {
                    let layout =
                        ResidentBufferLayout::new(ResidentPhysicalType::Bf16, 512 * capacity)
                            .unwrap();
                    StateRequirement {
                        slot,
                        layout: StateLayout::Kv {
                            slab: 512,
                            capacity,
                        },
                        physical: ResidentPhysicalType::Bf16,
                        bytes: layout.bytes().unwrap() * 2,
                    }
                }
            };
            requirement.plane_layout().unwrap();
            bytes += requirement.bytes;
        }
        assert_eq!(
            bytes,
            18 * (6144 * 3 * 4 + 16 * 128 * 128 * 4) + 6 * (512 * capacity * 2 * 2)
        );
        assert_eq!(bytes * 2, 845_709_312);
    }

    #[test]
    fn generic_gdn_workspace_charges_only_internal_owned_roles() {
        let layout = GdnRuntimeWorkspaceLayout::new(gdn_runtime_config().unwrap()).unwrap();
        let internal = layout
            .buffers()
            .iter()
            .filter(|buffer| {
                matches!(
                    buffer.ownership(),
                    GdnRuntimeBufferOwnership::CacheState
                        | GdnRuntimeBufferOwnership::Control
                        | GdnRuntimeBufferOwnership::Scratch
                ) || buffer.role() == crate::cuda::gdn::GdnRuntimeBuffer::Rstd
            })
            .map(|buffer| buffer.bytes())
            .sum::<usize>();
        assert!(internal > 0);
        assert!(
            layout
                .buffers()
                .iter()
                .any(|buffer| { buffer.ownership() == GdnRuntimeBufferOwnership::InputOutput })
        );
        assert!(
            layout
                .buffers()
                .iter()
                .any(|buffer| { buffer.ownership() == GdnRuntimeBufferOwnership::Parameters })
        );
        assert!(
            layout
                .buffers()
                .iter()
                .any(|buffer| { buffer.ownership() == GdnRuntimeBufferOwnership::RecurrentState })
        );
    }
}
