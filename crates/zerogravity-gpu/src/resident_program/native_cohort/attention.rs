//! Actual B4 attention producer/state/control roles for the SAME source loop.
use super::super::*;
use super::{kv::CohortKvTransport, values::CohortValues};
use crate::resident_inference::native_fragment::{
    PrefixPhaseEnqueue, model_state::BoundModelState, short_hf::ShortHfCallInput,
};
use crate::resident_leases::{ResidentInitialization, ResidentLease, ResidentService};
use crate::vllm_cuda_qualification::{
    cohort_native::Role, model_rotary::ModelRotary, short_hf_native::ShortHfNativeStorage,
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::device_events::{BufferRole, DeviceAccess};

pub(super) struct AttentionOwners {
    pub(super) native: Arc<ShortHfNativeStorage>,
    pub(super) rotary: Arc<ModelRotary>,
    pub(super) cache: ResidentLease,
    pub(super) status: ResidentLease,
    pub(super) controls: Arc<CohortAttentionControls>,
    pub(super) transport: Arc<super::CohortTransportLibrary>,
}
pub(super) struct CohortAttention {
    inputs: Box<[Arc<ShortHfCallInput>]>,
    values: Arc<CohortValues>,
    _native: Arc<ShortHfNativeStorage>,
    rotary: Arc<ModelRotary>,
    nodes: [usize; 7],
    roles: BTreeMap<Role, ResidentLease>,
    transport: Arc<CohortKvTransport>,
    _metadata: Arc<WindowWorkspace>,
}
fn bytes32(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
pub(super) struct CohortAttentionControls {
    inputs: Box<[Arc<ShortHfCallInput>]>,
    roles: BTreeMap<Role, ResidentLease>,
}
impl CohortAttentionControls {
    pub(super) fn prepare(
        inputs: &[Arc<ShortHfCallInput>],
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        crate::vllm_cuda_qualification::cohort_native::require_input_shape(inputs)?;
        let initial = inputs[0].rows() == 47;
        let m = 4 * inputs[0].rows();
        let mut roles = BTreeMap::new();
        // Positions are the concatenation of all four actual source spans;
        // pitch513 is the pinned original graph-position ABI, not dense m.
        let mut positions = vec![0u8; 3 * 513 * 8];
        for axis in 0..3 {
            for (row, input) in inputs.iter().enumerate() {
                for token in 0..input.rows() {
                    let position = i64::try_from(input.base() + token)
                        .map_err(|_| Error::Limit("B4 position".into()))?;
                    let offset = (axis * 513 + row * input.rows() + token) * 8;
                    positions[offset..offset + 8].copy_from_slice(&position.to_le_bytes());
                }
            }
        }
        let slots: Vec<u8> = inputs
            .iter()
            .enumerate()
            .flat_map(|(row, input)| {
                (input.base()..input.base() + input.rows()).flat_map(move |position| {
                    (((4 + 4 * row) * 544 + position) as i64).to_le_bytes()
                })
            })
            .collect();
        if slots.len() != m * 8 {
            return Err(Error::State("B4 source exact i64 slot controls".into()));
        }
        let mut table = vec![0i32; 16];
        for row in 0..4 {
            table[row * 4] = (4 + 4 * row) as i32;
        }
        for (role, bytes) in [
            (Role::Positions, positions),
            (Role::Slots, slots),
            (Role::DummyCuK, vec![0; 20]),
            (
                Role::SeqUsed,
                bytes32(
                    inputs
                        .iter()
                        .map(|input| (input.base() + input.rows()) as i32),
                ),
            ),
            (Role::Table, bytes32(table)),
            (Role::KScale, 1f32.to_le_bytes().to_vec()),
            (Role::VScale, 1f32.to_le_bytes().to_vec()),
        ] {
            roles.insert(
                role,
                init.upload_bytes(&bytes, BufferRole::ImmutableWeights)?,
            );
        }
        if initial {
            roles.insert(
                Role::CuQ,
                init.upload_bytes(
                    &bytes32([0, 47, 94, 141, 188]),
                    BufferRole::ImmutableWeights,
                )?,
            );
        }
        let result = Arc::new(Self {
            inputs: inputs.to_vec().into_boxed_slice(),
            roles,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn kv_sources(
        &self,
        inputs: &[Arc<ShortHfCallInput>],
        service: &Arc<ResidentService>,
    ) -> Result<[ResidentLease; 3]> {
        self.require_inputs(inputs)?;
        let sources = [Role::Positions, Role::SeqUsed, Role::Table]
            .map(|role| self.roles.get(&role).cloned());
        let [Some(positions), Some(ends), Some(pages)] = sources else {
            return Err(Error::State("B4 original KV controls absent".into()));
        };
        let sources = [positions, ends, pages];
        for (lease, bytes) in sources.iter().zip([3 * 513 * 8, 16, 64]) {
            if !Arc::ptr_eq(lease.service(), service)
                || lease.source_identity() != service.source_identity()
                || lease.role() != BufferRole::ImmutableWeights
                || lease.bytes() != bytes
            {
                return Err(Error::State("B4 original KV control owner/extent".into()));
            }
        }
        // Initialization completes on the enclosing original flight, before
        // transport enqueue; these are its already retained readonly owners.
        Ok(sources)
    }
    fn require_inputs(&self, inputs: &[Arc<ShortHfCallInput>]) -> Result<()> {
        if inputs.len() != self.inputs.len()
            || inputs
                .iter()
                .zip(&self.inputs)
                .any(|(a, b)| !Arc::ptr_eq(a, b))
        {
            return Err(Error::State(
                "B4 readonly controls belong to another original input epoch".into(),
            ));
        }
        Ok(())
    }
}
impl CohortAttention {
    pub(super) fn prepare(
        inputs: &[Arc<ShortHfCallInput>],
        states: &[Arc<BoundModelState>],
        values: Arc<CohortValues>,
        layer: usize,
        owners: AttentionOwners,
        init: &mut ResidentInitialization<'_>,
    ) -> Result<Arc<Self>> {
        let AttentionOwners {
            native,
            rotary,
            cache,
            status,
            controls,
            transport,
        } = owners;
        values.require_inputs(inputs)?;
        crate::vllm_cuda_qualification::cohort_native::require_input_shape(inputs)?;
        init.require_service(values.service())?;
        if inputs.len() != 4 || states.len() != 4 || layer % 4 != 3 {
            return Err(Error::State(
                "B4 actual attention layer/all-row state population".into(),
            ));
        }
        let initial = (inputs[0].rows(), inputs[0].base()) == (47, 0);
        let site = inputs[0].scope().attention_site(layer)?;
        for input in inputs {
            let actual = input.scope().attention_site(layer)?;
            if actual.nodes != site.nodes
                || actual.operands != site.operands
                || input.rows() != inputs[0].rows()
                || !matches!((input.rows(), input.base()), (47, 0) | (1, 47..=53))
                || input.workload().prompt_rows() != 47
            {
                return Err(Error::State(
                    "B4 every original attention producer/span".into(),
                ));
            }
            rotary.require_call(input.source().source(), input.call(), values.service())?;
        }
        let parameters = native.cohort_qk_parameters(inputs, layer)?;
        let metadata = Arc::new(
            inputs[0]
                .source()
                .source()
                .source()
                .reserve_workspace(512 * 1024)?,
        );
        init.retain_completion_owner(metadata.clone())?;
        init.retain_completion_owner(native.clone())?;
        init.retain_completion_owner(rotary.clone())?;
        let transport = CohortKvTransport::prepare(
            values.service().clone(),
            inputs,
            states,
            cache.clone(),
            status,
            controls.clone(),
            transport,
            init,
        )?;
        let mut roles = BTreeMap::from([
            (Role::Packed, values.lease(site.operands[1])?.clone()),
            (Role::Q, values.lease(site.operands[2])?.clone()),
            (Role::K, values.lease(site.operands[3])?.clone()),
            (Role::Gate, values.lease(site.operands[4])?.clone()),
            (Role::O, values.lease(site.operands[5])?.clone()),
            (Role::QGamma, parameters[0].clone()),
            (Role::KGamma, parameters[1].clone()),
            (Role::Rotary, rotary.lease().clone()),
            (Role::Cache, cache),
        ]);
        controls.require_inputs(inputs)?;
        for (role, lease) in &controls.roles {
            if roles.insert(*role, lease.clone()).is_some() {
                return Err(Error::State("B4 repeated source control role".into()));
            }
        }
        roles.insert(
            Role::Lse,
            init.allocate_empty(
                if initial { 4 * 8 * 47 * 4 } else { 128 },
                BufferRole::Scratch,
            )?,
        );
        if !initial {
            for (role, bytes) in [
                (Role::PrivateQ, 16384),
                (Role::PrivateO, 16384),
                (Role::PublicLse, 128),
            ] {
                roles.insert(role, init.allocate_empty(bytes, BufferRole::Scratch)?);
            }
        }
        let original = roles.iter().collect::<Vec<_>>();
        for (index, (_, lease)) in original.iter().enumerate() {
            let range = BufferRange::new(&lease.handle())?;
            for (_, other) in &original[..index] {
                if range.overlaps(&BufferRange::new(&other.handle())?) {
                    return Err(Error::State(
                        "B4 attention private/source roles overlap".into(),
                    ));
                }
            }
        }
        let nodes = site.nodes;
        let result = Arc::new(Self {
            inputs: inputs.to_vec().into_boxed_slice(),
            values,
            _native: native,
            rotary,
            nodes,
            roles,
            transport,
            _metadata: metadata,
        });
        init.retain_completion_owner(result.clone())?;
        Ok(result)
    }
    pub(super) fn roles(&self, node: usize) -> Result<BTreeMap<Role, ResidentLease>> {
        let site = self.inputs[0].scope().attention_site(
            // Site source descriptor supplies the genuine model layer.
            match &self.inputs[0].record().plan().nodes()[self.nodes[1]].operation {
                PlanOperation::TypedVllm08 { descriptor, .. } => descriptor
                    .layer()
                    .ok_or_else(|| Error::State("B4 attention layer descriptor".into()))?
                    .index(),
                _ => return Err(Error::State("B4 actual QK source".into())),
            },
        )?;
        let mut roles = self.roles.clone();
        if node == self.nodes[4] {
            roles.insert(Role::X, self.values.lease(site.operands[4])?.clone());
            roles.insert(Role::Y, self.values.lease(site.operands[6])?.clone());
        } else if node == self.nodes[5] {
            roles.insert(Role::X, self.values.lease(site.operands[5])?.clone());
            roles.insert(Role::Gate, self.values.lease(site.operands[6])?.clone());
            roles.insert(Role::Y, self.values.lease(site.operands[7])?.clone());
        } else if !self.nodes.contains(&node) {
            return Err(Error::State("B4 attention actual source node".into()));
        }
        Ok(roles)
    }
    pub(super) fn before(
        &self,
        node: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<(usize, usize)> {
        if node != self.nodes[2] {
            return Ok((0, 0));
        }
        self.require_current(permits)?;
        if node == self.nodes[2] {
            self.transport.before_append(permits, effects)
        } else {
            Ok((0, 0))
        }
    }
    pub(super) fn after(
        &self,
        node: usize,
        permits: &[PrefixPhaseEnqueue<'_>],
        effects: &[StateEffect],
    ) -> Result<(usize, usize)> {
        if node != self.nodes[2] {
            return Ok((0, 0));
        }
        self.require_current(permits)?;
        if node == self.nodes[2] {
            self.transport.after_append(permits, effects)
        } else {
            Ok((0, 0))
        }
    }
    /// The KV-append source node owning both grouped transport directions.
    pub(super) fn kv_append_node(&self) -> usize {
        self.nodes[2]
    }
    /// Both grouped KV transport directions' census attribution (import and
    /// export both execute at the KvAppend source node).
    pub(super) fn kv_transport_accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        let mut accesses = self.transport.direction_accesses(true);
        accesses.extend(self.transport.direction_accesses(false));
        accesses
    }
    fn require_current(&self, permits: &[PrefixPhaseEnqueue<'_>]) -> Result<()> {
        if permits.len() != 4 {
            return Err(Error::State("B4 attention all row permits".into()));
        }
        self.values.require_inputs(&self.inputs)?;
        for (input, permit) in self.inputs.iter().zip(permits) {
            input.require_permit(permit, self.values.service())?;
            self.rotary.require_call(
                input.source().source(),
                input.call(),
                self.values.service(),
            )?;
        }
        Ok(())
    }
    pub(super) fn accesses(&self) -> Vec<(ResidentLease, DeviceAccess)> {
        self.transport.accesses()
    }
}
