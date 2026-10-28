//! Cold ordinary-Qwen CUDA physical/resource analysis.
//!
//! This consumes only the immutable call-free ForwardPlan. It performs no
//! source reads, GPU allocations, provider creation, request binding or launch.

use super::attention;
use crate::{
    cuda::gdn::{
        GdnKernelConfig, GdnKktKernel, GdnOutputKernel, GdnRecomputeKernel,
        GdnRuntimeBufferOwnership, GdnRuntimeConfig, GdnRuntimeKernelGeometry,
        GdnRuntimeWorkspaceLayout, GdnStateKernel,
    },
    resident_leases::ordinary_parameter_upload::OrdinaryParameterStorage,
    resident_program::{
        ResidentBufferLayout, ResidentPhysicalType, ResidentPreflight, StateLayout,
        StateRequirement,
    },
};
use std::sync::Arc;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ConvDomain, ForwardPlan, KvDomain, PlanOperation, RecurrentDomain, SourceOperatorKind,
        SourceParameter, SourceTensor, StateSlot, ValueStorageBinding, ValueStorageLimits,
        ValueStorageRetention, ValueStorageReusePolicy,
    },
};

pub(super) const MAX_ROWS: usize = 64;
pub(super) const VOCABULARY: usize = 248_320;
const VALUES: usize = 327;
const NODES: usize = 255;
const VALUE_ALIGNMENT: usize = 512;
const VALUE_MAX_SLOT_BYTES: usize = 2 * 1024 * 1024;
const VALUE_MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;
const FOREIGN_PROVIDER_BYTES: usize = 64 * 1024 * 1024;
const BLAS_WORKSPACE_BYTES: usize = 8_519_680;
const BLAS_SENTINEL_BYTES: usize = 4;
const TOKEN_BYTES: usize = MAX_ROWS * 4;
const POSITION_BYTES: usize = MAX_ROWS * 4;
const INVERSE_FREQUENCY_BYTES: usize = 32 * 4;
const F32_LOGITS_BYTES: usize = VOCABULARY * 4;
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
    storage: OrdinaryParameterStorage,
    bytes: usize,
}
impl ParameterSpec {
    pub(super) fn descriptor(&self) -> &SourceParameter {
        &self.descriptor
    }
    pub(super) fn storage(&self) -> OrdinaryParameterStorage {
        self.storage
    }
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
}

#[derive(Clone)]
pub(super) struct OrdinaryCudaLayout {
    plan: Arc<ForwardPlan>,
    value_layouts: Arc<[ResidentBufferLayout]>,
    value_bindings: Arc<[ValueStorageBinding]>,
    slot_bytes: Arc<[usize]>,
    states: Arc<[StateRequirement]>,
    matrices: Arc<[MatrixPackSpec]>,
    parameters: Arc<[ParameterSpec]>,
    gdn: GdnRuntimeWorkspaceLayout,
    preflight: ResidentPreflight,
    foreign_provider_bytes: usize,
    blas_bytes: usize,
}
impl OrdinaryCudaLayout {
    pub(super) fn analyze(plan: Arc<ForwardPlan>) -> Result<Arc<Self>> {
        require_plan_shape(&plan)?;
        let value_layouts = plan
            .values()
            .iter()
            .map(|shape| {
                ResidentBufferLayout::new(ResidentPhysicalType::Bf16, shape.capacity_elements()?)
            })
            .collect::<Result<Vec<_>>>()?;
        let storage = ResidentBufferLayout::plan_values(
            &plan,
            &value_layouts,
            ValueStorageRetention::Reuse {
                escaped: &[],
                pinned: &[],
                taps: &[],
            },
            ValueStorageLimits {
                max_nodes: NODES,
                max_values: VALUES,
                max_slots: VALUES,
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
                    "ordinary CUDA value slot requires padded/subview ownership".into(),
                ));
            }
            let prior = &mut slot_bytes[binding.slot];
            if *prior != 0 && *prior != binding.bytes {
                return Err(Error::Unsupported(
                    "ordinary CUDA value slot logical extents differ".into(),
                ));
            }
            *prior = binding.bytes;
        }
        if slot_bytes.contains(&0) || slot_bytes.iter().sum::<usize>() != storage.total_bytes() {
            return Err(Error::State(
                "ordinary CUDA value slot coverage differs".into(),
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
                    matches!(
                        buffer.ownership(),
                        GdnRuntimeBufferOwnership::CacheState
                            | GdnRuntimeBufferOwnership::Control
                            | GdnRuntimeBufferOwnership::Scratch
                    )
                })
                .map(|buffer| buffer.bytes()),
        )?;
        let attention_bytes = attention::managed_bytes(plan.prefix_capacity())?;
        let workspace_bytes = checked_sum([
            attention_bytes,
            gdn_workspace_bytes,
            TOKEN_BYTES,
            POSITION_BYTES,
            F32_LOGITS_BYTES,
        ])?;
        let value_bindings: Arc<[ValueStorageBinding]> = storage.bindings().into();
        let value_total_bytes = storage.total_bytes();
        drop(storage);

        let preflight = ResidentPreflight {
            source: plan.source(),
            source_bytes: matrix_bytes,
            value_bytes: value_total_bytes,
            workspace_bytes,
            status_bytes: STATUS_BYTES,
            parameter_bytes: parameter_bytes
                .checked_add(INVERSE_FREQUENCY_BYTES)
                .ok_or_else(|| Error::Limit("ordinary CUDA parameter byte sum".into()))?,
            prior_state_bytes: state_bytes,
            candidate_state_bytes: state_bytes,
            states: states.to_vec(),
            scratch: [0; super::super::SCRATCH_SLOTS],
        };
        // Exercise every checked sum before returning cold authority metadata.
        preflight.total_with_state()?;
        Ok(Arc::new(Self {
            plan,
            value_layouts: value_layouts.into(),
            value_bindings,
            slot_bytes: slot_bytes.into(),
            states: states.into(),
            matrices: matrices.into(),
            parameters: parameters.into(),
            gdn,
            preflight,
            foreign_provider_bytes: FOREIGN_PROVIDER_BYTES,
            blas_bytes: BLAS_WORKSPACE_BYTES + BLAS_SENTINEL_BYTES,
        }))
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
    pub(super) fn foreign_provider_bytes(&self) -> usize {
        self.foreign_provider_bytes
    }
    pub(super) fn blas_bytes(&self) -> usize {
        self.blas_bytes
    }
    pub(super) fn total_admission_bytes(&self) -> Result<usize> {
        checked_sum([
            self.preflight.total_with_state()?,
            self.foreign_provider_bytes,
            self.blas_bytes,
        ])
    }
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
    if plan.source_call().is_some()
        || plan.frontier_delta() != 0
        || plan.nodes().len() != NODES
        || plan.values().len() != VALUES
        || plan.logits().is_none()
        || plan.prefix_capacity() < 512
        || plan.prefix_capacity() > 32_768
        || plan.required_domains().len() != 42
        || plan.proposed_domains().len() != 42
    {
        return Err(Error::Unsupported(
            "ordinary CUDA exact call-free Qwen plan shape".into(),
        ));
    }
    let mut counts = OperationCounts::default();
    for node in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            return Err(Error::Unsupported(
                "ordinary CUDA plan contains non-template operation".into(),
            ));
        };
        if descriptor.source() != plan.source()
            || descriptor.max_rows() != MAX_ROWS
            || descriptor.vocabulary() != VOCABULARY
        {
            return Err(Error::State(
                "ordinary CUDA template source/rows/vocabulary differs".into(),
            ));
        }
        counts.add(descriptor.kind());
    }
    counts.require_exact()
}

#[derive(Default)]
struct OperationCounts {
    embedding: usize,
    projection: usize,
    rms: usize,
    residual_rms: usize,
    conv: usize,
    recurrent: usize,
    qk: usize,
    kv: usize,
    attention: usize,
    sigmoid: usize,
    multiply: usize,
    silu: usize,
}
impl OperationCounts {
    fn add(&mut self, kind: SourceOperatorKind) {
        use SourceOperatorKind::*;
        match kind {
            EmbeddingRows => self.embedding += 1,
            Projection { .. } => self.projection += 1,
            Rms(_) => self.rms += 1,
            ResidualRms(_) => self.residual_rms += 1,
            CausalConvSplit { .. } => self.conv += 1,
            GatedDelta { .. } => self.recurrent += 1,
            QkRmsRotaryGated(_) => self.qk += 1,
            KvAppend { .. } => self.kv += 1,
            FlashAttention { .. } => self.attention += 1,
            Sigmoid { .. } => self.sigmoid += 1,
            Multiply { .. } => self.multiply += 1,
            SiluAndMul { .. } => self.silu += 1,
            _ => {}
        }
    }
    fn require_exact(&self) -> Result<()> {
        let actual = [
            self.embedding,
            self.projection,
            self.rms,
            self.residual_rms,
            self.conv,
            self.recurrent,
            self.qk,
            self.kv,
            self.attention,
            self.sigmoid,
            self.multiply,
            self.silu,
        ];
        let expected = [1, 115, 1, 48, 18, 18, 6, 6, 6, 6, 6, 24];
        if actual != expected || actual.iter().sum::<usize>() != NODES {
            return Err(Error::Unsupported(
                "ordinary CUDA exact source operator population".into(),
            ));
        }
        Ok(())
    }
}

fn state_requirements(plan: &ForwardPlan) -> Result<Vec<StateRequirement>> {
    let mut result = Vec::with_capacity(42);
    let proposed = plan.proposed_domains();
    for (index, &slot) in plan.required_domains().iter().enumerate() {
        let version = proposed
            .get(index)
            .ok_or_else(|| Error::State("ordinary CUDA proposed state missing".into()))?;
        if version.slot != slot || version.ordinal != 1 {
            return Err(Error::State(
                "ordinary CUDA proposed state slot/version differs".into(),
            ));
        }
        let requirement = match slot {
            StateSlot::Conv(ConvDomain(layer)) if layer < 24 && layer % 4 != 3 => {
                state_vector(slot, 6144 * 3, ResidentPhysicalType::F32)?
            }
            StateSlot::Recurrent(RecurrentDomain(layer)) if layer < 24 && layer % 4 != 3 => {
                state_vector(slot, 16 * 128 * 128, ResidentPhysicalType::F32)?
            }
            StateSlot::Kv(KvDomain(layer)) if layer < 24 && layer % 4 == 3 => {
                let elements = 512usize
                    .checked_mul(plan.prefix_capacity())
                    .ok_or_else(|| Error::Limit("ordinary CUDA KV state extent".into()))?;
                let layout = ResidentBufferLayout::new(ResidentPhysicalType::Bf16, elements)?;
                StateRequirement {
                    slot,
                    layout: StateLayout::Kv {
                        slab: 512,
                        capacity: plan.prefix_capacity(),
                    },
                    physical: ResidentPhysicalType::Bf16,
                    bytes: layout
                        .bytes()?
                        .checked_mul(2)
                        .ok_or_else(|| Error::Limit("ordinary CUDA KV state bytes".into()))?,
                }
            }
            _ => {
                return Err(Error::Unsupported(
                    "ordinary CUDA state domain population".into(),
                ));
            }
        };
        requirement.plane_layout()?;
        result.push(requirement);
    }
    if result
        .iter()
        .filter(|state| matches!(state.slot, StateSlot::Conv(_)))
        .count()
        != 18
        || result
            .iter()
            .filter(|state| matches!(state.slot, StateSlot::Recurrent(_)))
            .count()
            != 18
        || result
            .iter()
            .filter(|state| matches!(state.slot, StateSlot::Kv(_)))
            .count()
            != 6
    {
        return Err(Error::State(
            "ordinary CUDA exact 18/18/6 state population differs".into(),
        ));
    }
    Ok(result)
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
                        .map_err(|_| Error::Limit("ordinary CUDA source matrix bytes".into()))?;
                    sum.checked_add(bytes)
                        .ok_or_else(|| Error::Limit("ordinary CUDA source matrix sum".into()))
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
            "ordinary CUDA immutable source population empty".into(),
        ));
    }
    Ok((matrices, parameters))
}

fn parameter_storage(kind: SourceOperatorKind, ordinal: usize) -> Result<OrdinaryParameterStorage> {
    use OrdinaryParameterStorage::{Bf16Rne, Original};
    use SourceOperatorKind::*;
    match kind {
        Rms(_) | ResidualRms(_) if ordinal == 0 => Ok(Bf16Rne),
        QkRmsRotaryGated(_) if ordinal < 2 => Ok(Original),
        CausalConvSplit { .. } if ordinal == 0 => Ok(Original),
        GatedDelta { .. } => match ordinal {
            0 | 1 => Ok(Original),
            2 => Ok(Bf16Rne),
            _ => Err(Error::State("ordinary CUDA GDN parameter ordinal".into())),
        },
        _ => Err(Error::State(
            "ordinary CUDA unexpected learned parameter role".into(),
        )),
    }
}

fn parameter_resident_bytes(
    parameter: &SourceParameter,
    storage: OrdinaryParameterStorage,
) -> Result<usize> {
    match storage {
        OrdinaryParameterStorage::Original => Ok(parameter.bytes),
        OrdinaryParameterStorage::Bf16Rne => {
            if !matches!(
                parameter.kind,
                zerogravity_core::quant::QuantKind::Bf16 | zerogravity_core::quant::QuantKind::F32
            ) {
                return Err(Error::Unsupported(
                    "ordinary CUDA converted parameter codec".into(),
                ));
            }
            let elements = parameter
                .shape
                .iter()
                .try_fold(1usize, |product, &dimension| {
                    let dimension = usize::try_from(dimension)
                        .map_err(|_| Error::Limit("ordinary CUDA parameter dimension".into()))?;
                    product
                        .checked_mul(dimension)
                        .ok_or_else(|| Error::Limit("ordinary CUDA parameter elements".into()))
                })?;
            elements
                .checked_mul(2)
                .ok_or_else(|| Error::Limit("ordinary CUDA converted parameter bytes".into()))
        }
    }
}

fn checked_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0usize, |sum, value| {
        sum.checked_add(value)
            .ok_or_else(|| Error::Limit("ordinary CUDA resource byte sum".into()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
                )
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
