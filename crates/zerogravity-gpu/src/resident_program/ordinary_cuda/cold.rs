//! Cold ordinary CUDA source ownership and reusable resident storage.
//! No request binding, launch, completion, root attempt, or publication lives here.

use super::{OrdinaryCudaProjection, attention::OrdinaryCudaAttention, layout};
use crate::{
    cuda::{
        gdn::{GdnRuntimeBuffer, GdnRuntimeBufferOwnership},
        ordinary_text::{Geometry, Program},
    },
    resident_leases::{
        ResidentInitialization, ResidentLease, ResidentService,
        ordinary_parameter_upload::OrdinaryParameter,
        ordinary_projection_upload::PackedSourceMatrices,
    },
    resident_program::DeviceBuffer,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
};
use zerogravity_artifacts::source_weights::VerifiedTensorSource;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{
        ForwardPlan, PlanOperation, SourceNumericalContract, SourceOperatorKind, SourceParameter,
        SourceTensor,
    },
    device_events::{BufferRole, DeviceAccess},
};

pub(super) struct OrdinaryCudaCold {
    layout: Arc<layout::OrdinaryCudaLayout>,
    source: VerifiedTensorSource,
    service: Arc<ResidentService>,
    program: Program,
    values: Arc<[DeviceBuffer]>,
    value_slots: Arc<[ResidentLease]>,
    status: ResidentLease,
    token_ids: ResidentLease,
    positions: ResidentLease,
    logits_f32: ResidentLease,
    inverse_frequency: ResidentLease,
    attention: Arc<OrdinaryCudaAttention>,
    gdn_internal: BTreeMap<GdnRuntimeBuffer, ResidentLease>,
    matrix_packs: Arc<[Arc<PackedSourceMatrices>]>,
    parameters: Arc<[Arc<OrdinaryParameter>]>,
    node_parameters: BTreeMap<usize, Arc<[Arc<OrdinaryParameter>]>>,
    projections: BTreeMap<usize, Arc<OrdinaryCudaProjection>>,
    embedding: Arc<PackedSourceMatrices>,
}

impl OrdinaryCudaCold {
    pub(super) async fn prepare(
        layout: Arc<layout::OrdinaryCudaLayout>,
        source: VerifiedTensorSource,
        service: Arc<ResidentService>,
        init: &mut ResidentInitialization<'_>,
        cancelled: Option<&AtomicBool>,
    ) -> Result<Arc<Self>> {
        init.require_service(&service)?;
        service.ensure_usable()?;
        if source.source_identity() != layout.plan().source()
            || service.source_identity() != layout.plan().source()
            || layout.preflight().source != layout.plan().source()
        {
            return Err(Error::State(
                "ordinary CUDA cold source/service/plan differs".into(),
            ));
        }
        init.retain_completion_owner(Arc::new(source.clone()))?;

        let geometry = geometry(layout.plan())?;
        let program = Program::new(geometry)?;

        let value_slots = layout
            .slot_bytes()
            .iter()
            .map(|&bytes| init.allocate_empty(bytes, BufferRole::Activation))
            .collect::<Result<Vec<_>>>()?;
        let values = layout
            .value_bindings()
            .iter()
            .zip(layout.value_layouts())
            .map(|(binding, &physical)| {
                let lease = value_slots
                    .get(binding.slot)
                    .ok_or_else(|| Error::State("ordinary CUDA value slot absent".into()))?
                    .clone();
                if lease.bytes() != binding.bytes || physical.bytes()? != binding.bytes {
                    return Err(Error::State(
                        "ordinary CUDA value slot/binding/layout differs".into(),
                    ));
                }
                DeviceBuffer::from_lease(
                    lease,
                    physical,
                    BufferRole::Activation,
                    DeviceAccess::Write,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        let status = init.upload_zeroed(4, BufferRole::Scratch)?;
        let token_ids = init.upload_zeroed(layout::MAX_ROWS * 4, BufferRole::Activation)?;
        let positions = init.upload_zeroed(layout::MAX_ROWS * 4, BufferRole::Activation)?;
        let logits_f32 = init.allocate_empty(layout::VOCABULARY * 4, BufferRole::Output)?;
        let inverse_frequency =
            init.upload_bytes(&inverse_frequency_bytes(), BufferRole::ImmutableWeights)?;
        let attention =
            OrdinaryCudaAttention::prepare(service.clone(), init, layout.plan().prefix_capacity())?;

        let mut gdn_internal = BTreeMap::new();
        for buffer in layout.gdn().buffers() {
            if !matches!(
                buffer.ownership(),
                GdnRuntimeBufferOwnership::CacheState
                    | GdnRuntimeBufferOwnership::Control
                    | GdnRuntimeBufferOwnership::Scratch
            ) && !matches!(
                buffer.role(),
                GdnRuntimeBuffer::Rstd
                    | GdnRuntimeBuffer::InitialState
                    | GdnRuntimeBuffer::FinalState
            ) {
                continue;
            }
            if gdn_internal
                .insert(
                    buffer.role(),
                    init.allocate_empty(buffer.bytes(), BufferRole::Scratch)?,
                )
                .is_some()
            {
                return Err(Error::State(
                    "ordinary CUDA duplicate GDN internal role".into(),
                ));
            }
        }

        let mut matrix_packs = Vec::with_capacity(layout.matrices().len());
        for spec in layout.matrices() {
            if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
                return Err(Error::State(
                    "ordinary CUDA cold matrix preparation cancelled".into(),
                ));
            }
            let packed = init
                .upload_ordinary_matrices(&service, source.clone(), spec.members(), cancelled)
                .await?;
            if packed.lease().bytes() != spec.bytes() {
                return Err(Error::State(
                    "ordinary CUDA packed source matrix preflight bytes differ".into(),
                ));
            }
            matrix_packs.push(packed);
        }

        let mut parameters = Vec::with_capacity(layout.parameters().len());
        for spec in layout.parameters() {
            if cancelled.is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
                return Err(Error::State(
                    "ordinary CUDA cold parameter preparation cancelled".into(),
                ));
            }
            let parameter = init.upload_ordinary_parameter(
                &service,
                source.clone(),
                spec.descriptor(),
                spec.storage(),
                cancelled,
            )?;
            if parameter.lease().bytes() != spec.bytes() {
                return Err(Error::State(
                    "ordinary CUDA parameter preflight bytes differ".into(),
                ));
            }
            parameters.push(parameter);
        }

        let find_pack = |members: &[SourceTensor]| -> Result<Arc<PackedSourceMatrices>> {
            let mut matches = matrix_packs
                .iter()
                .filter(|pack| pack.require_members(members).is_ok());
            let pack = matches
                .next()
                .ok_or_else(|| Error::State("ordinary CUDA source matrix pack absent".into()))?
                .clone();
            if matches.next().is_some() {
                return Err(Error::State(
                    "ordinary CUDA source matrix pack identity duplicated".into(),
                ));
            }
            Ok(pack)
        };
        let find_parameter = |descriptor: &SourceParameter,
                              storage|
         -> Result<Arc<OrdinaryParameter>> {
            let mut matches = parameters.iter().filter(|parameter| {
                parameter.storage() == storage && parameter.require_descriptor(descriptor).is_ok()
            });
            let parameter = matches
                .next()
                .ok_or_else(|| Error::State("ordinary CUDA authenticated parameter absent".into()))?
                .clone();
            if matches.next().is_some() {
                return Err(Error::State(
                    "ordinary CUDA authenticated parameter duplicated".into(),
                ));
            }
            Ok(parameter)
        };

        let mut embedding = None;
        let mut projections = BTreeMap::new();
        let mut node_parameters = BTreeMap::new();
        for (index, node) in layout.plan().nodes().iter().enumerate() {
            let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
                unreachable!("layout accepted only source templates")
            };
            if descriptor.numerical_contract() != SourceNumericalContract::ReferenceCudaSm86Bf16V1 {
                return Err(Error::State(
                    "ordinary CUDA source numerical contract differs".into(),
                ));
            }
            if !descriptor.parameters().is_empty() {
                let bound = descriptor
                    .parameters()
                    .iter()
                    .enumerate()
                    .map(|(ordinal, parameter)| {
                        find_parameter(
                            parameter,
                            layout::parameter_storage(descriptor.kind(), ordinal)?,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                node_parameters.insert(index, bound.into());
            }
            match descriptor.kind() {
                SourceOperatorKind::EmbeddingRows => {
                    if embedding.is_some()
                        || descriptor.matrices().len() != 1
                        || !descriptor.parameters().is_empty()
                    {
                        return Err(Error::State(
                            "ordinary CUDA exact embedding source binding differs".into(),
                        ));
                    }
                    embedding = Some(find_pack(descriptor.matrices())?);
                }
                SourceOperatorKind::Projection { .. } => {
                    let projection = OrdinaryCudaProjection::prepare_with_weights(
                        descriptor,
                        find_pack(descriptor.matrices())?,
                    )?;
                    if projections.insert(index, projection).is_some() {
                        return Err(Error::State(
                            "ordinary CUDA projection node duplicated".into(),
                        ));
                    }
                }
                _ => {}
            }
        }
        if projections.len() != 115 || node_parameters.is_empty() {
            return Err(Error::State(
                "ordinary CUDA cold projection/parameter population differs".into(),
            ));
        }

        let result = Arc::new(Self {
            layout,
            source,
            service,
            program,
            values: values.into(),
            value_slots: value_slots.into(),
            status,
            token_ids,
            positions,
            logits_f32,
            inverse_frequency,
            attention,
            gdn_internal,
            matrix_packs: matrix_packs.into(),
            parameters: parameters.into(),
            node_parameters,
            projections,
            embedding: embedding
                .ok_or_else(|| Error::State("ordinary CUDA embedding pack absent".into()))?,
        });
        init.retain_completion_owner(result.clone())?;
        result.require_cold_ownership()?;
        Ok(result)
    }

    fn require_cold_ownership(&self) -> Result<()> {
        if self.source.source_identity() != self.layout.plan().source()
            || self.service.source_identity() != self.layout.plan().source()
            || self.values.len() != self.layout.plan().values().len()
            || self.value_slots.len() != self.layout.slot_bytes().len()
            || self.status.bytes() != 4
            || self.token_ids.bytes() != layout::MAX_ROWS * 4
            || self.positions.bytes() != layout::MAX_ROWS * 4
            || self.logits_f32.bytes() != layout::VOCABULARY * 4
            || self.inverse_frequency.bytes() != 32 * 4
            || self.matrix_packs.len() != self.layout.matrices().len()
            || self.parameters.len() != self.layout.parameters().len()
        {
            return Err(Error::State(
                "ordinary CUDA cold owned resource population differs".into(),
            ));
        }
        for (parameter, spec) in self.parameters.iter().zip(self.layout.parameters()) {
            if parameter.descriptor() != spec.descriptor()
                || parameter.storage() != spec.storage()
                || parameter.lease().bytes() != spec.bytes()
            {
                return Err(Error::State(
                    "ordinary CUDA cold parameter/spec ownership differs".into(),
                ));
            }
            let expected_elements =
                spec.descriptor()
                    .shape
                    .iter()
                    .try_fold(1usize, |product, &dimension| {
                        product
                            .checked_mul(usize::try_from(dimension).map_err(|_| {
                                Error::Limit("ordinary CUDA parameter element dimension".into())
                            })?)
                            .ok_or_else(|| {
                                Error::Limit("ordinary CUDA parameter element count".into())
                            })
                    })?;
            if parameter.elements() != expected_elements
                || (parameter.storage()
                    == crate::resident_leases::ordinary_parameter_upload::OrdinaryParameterStorage::Bf16Rne
                    && parameter.kind() != zerogravity_core::quant::QuantKind::Bf16)
            {
                return Err(Error::State(
                    "ordinary CUDA cold parameter converted physical type differs".into(),
                ));
            }
        }
        for (value, (&layout, binding)) in self
            .layout
            .value_layouts()
            .iter()
            .zip(self.layout.value_bindings())
            .enumerate()
        {
            self.values[value].require_layout(layout)?;
            let lease = self.values[value].original_lease()?;
            if lease.id() != self.value_slots[binding.slot].id()
                || !Arc::ptr_eq(lease.service(), &self.service)
            {
                return Err(Error::State(
                    "ordinary CUDA logical value/original slot differs".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn layout(&self) -> &Arc<layout::OrdinaryCudaLayout> {
        &self.layout
    }
    pub(super) fn plan(&self) -> &Arc<ForwardPlan> {
        self.layout.plan()
    }
    pub(super) fn source(&self) -> &VerifiedTensorSource {
        &self.source
    }
    pub(super) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub(super) fn program(&self) -> &Program {
        &self.program
    }
    pub(super) fn values(&self) -> &[DeviceBuffer] {
        &self.values
    }
    pub(super) fn value(&self, index: usize) -> Result<&DeviceBuffer> {
        self.values
            .get(index)
            .ok_or_else(|| Error::State("ordinary CUDA value index absent".into()))
    }
    pub(super) fn value_lease(&self, index: usize) -> Result<&ResidentLease> {
        self.value(index)?.original_lease()
    }
    pub(super) fn status(&self) -> &ResidentLease {
        &self.status
    }
    pub(super) fn token_ids(&self) -> &ResidentLease {
        &self.token_ids
    }
    pub(super) fn positions(&self) -> &ResidentLease {
        &self.positions
    }
    pub(super) fn logits_f32(&self) -> &ResidentLease {
        &self.logits_f32
    }
    pub(super) fn inverse_frequency(&self) -> &ResidentLease {
        &self.inverse_frequency
    }
    pub(super) fn attention(&self) -> &Arc<OrdinaryCudaAttention> {
        &self.attention
    }
    pub(super) fn projection(&self, node: usize) -> Result<&Arc<OrdinaryCudaProjection>> {
        self.projections
            .get(&node)
            .ok_or_else(|| Error::State("ordinary CUDA projection binding absent".into()))
    }
    pub(super) fn embedding(&self) -> &Arc<PackedSourceMatrices> {
        &self.embedding
    }
    pub(super) fn node_parameters(&self, node: usize) -> Result<&[Arc<OrdinaryParameter>]> {
        self.node_parameters
            .get(&node)
            .map(AsRef::as_ref)
            .ok_or_else(|| Error::State("ordinary CUDA node parameters absent".into()))
    }
    pub(super) fn gdn_internal(&self, role: GdnRuntimeBuffer) -> Result<&ResidentLease> {
        self.gdn_internal
            .get(&role)
            .ok_or_else(|| Error::State("ordinary CUDA GDN internal role absent".into()))
    }

    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut unique = BTreeMap::new();
        let mut add = |lease: &ResidentLease, access| {
            unique
                .entry(lease.id())
                .and_modify(|(_, old)| {
                    if access == DeviceAccess::Write {
                        *old = DeviceAccess::Write;
                    }
                })
                .or_insert((lease.clone(), access));
        };
        for lease in self.value_slots.iter() {
            add(lease, DeviceAccess::Write);
        }
        for lease in [
            &self.status,
            &self.token_ids,
            &self.positions,
            &self.logits_f32,
        ] {
            add(lease, DeviceAccess::Write);
        }
        add(&self.inverse_frequency, DeviceAccess::Read);
        for (lease, access) in self.attention.accesses() {
            add(&lease, access);
        }
        for lease in self.gdn_internal.values() {
            add(lease, DeviceAccess::Write);
        }
        for pack in self.matrix_packs.iter() {
            add(pack.lease(), DeviceAccess::Read);
        }
        for parameter in self.parameters.iter() {
            add(parameter.lease(), DeviceAccess::Read);
        }
        unique.into_values().collect()
    }
}

fn geometry(plan: &ForwardPlan) -> Result<Geometry> {
    let mut hidden = None;
    let mut intermediate = None;
    for node in plan.nodes() {
        let PlanOperation::SourceOperatorTemplate { descriptor, .. } = &node.operation else {
            continue;
        };
        match descriptor.kind() {
            SourceOperatorKind::Rms(g) | SourceOperatorKind::ResidualRms(g)
                if hidden
                    .replace(g.width)
                    .is_some_and(|prior| prior != g.width) =>
            {
                return Err(Error::State(
                    "ordinary CUDA hidden width differs across RMS".into(),
                ));
            }
            SourceOperatorKind::SiluAndMul { width }
                if intermediate
                    .replace(width)
                    .is_some_and(|prior| prior != width) =>
            {
                return Err(Error::State(
                    "ordinary CUDA intermediate width differs".into(),
                ));
            }
            _ => {}
        }
    }
    let hidden = hidden.ok_or_else(|| Error::State("ordinary CUDA hidden width absent".into()))?;
    let intermediate = intermediate
        .ok_or_else(|| Error::State("ordinary CUDA intermediate width absent".into()))?;
    if !matches!((hidden, intermediate), (1024, 3584) | (2048, 6144)) {
        return Err(Error::Unsupported(
            "ordinary CUDA Qwen .8B/2B hidden/intermediate geometry".into(),
        ));
    }
    Geometry {
        hidden_width: hidden,
        intermediate_width: intermediate,
        vocabulary: layout::VOCABULARY,
        query_heads: 8,
        kv_heads: 2,
        head_width: 256,
        rotary_width: 64,
    }
    .validate()
}

fn inverse_frequency_bytes() -> Vec<u8> {
    let scale = 10_000_000f32.powf(-2.0 / 64.0);
    let mut frequency = 1.0f32;
    let mut bytes = Vec::with_capacity(32 * 4);
    for _ in 0..32 {
        bytes.extend_from_slice(&frequency.to_bits().to_le_bytes());
        frequency *= scale;
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_inverse_frequency_is_source_formula_not_a_capture_table() {
        let bytes = inverse_frequency_bytes();
        assert_eq!(bytes.len(), 128);
        let values = bytes
            .chunks_exact(4)
            .map(|word| f32::from_bits(u32::from_le_bytes(word.try_into().unwrap())))
            .collect::<Vec<_>>();
        let scale = 10_000_000f32.powf(-2.0 / 64.0);
        assert_eq!(values[0].to_bits(), 1.0f32.to_bits());
        for i in 1..32 {
            assert_eq!(values[i].to_bits(), (values[i - 1] * scale).to_bits());
        }
    }
}
