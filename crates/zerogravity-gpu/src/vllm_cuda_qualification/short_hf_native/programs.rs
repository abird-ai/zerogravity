//! Fresh current-record operands over one admitted immutable packet/image pool.
use super::super::{
    model_values::ModelValues, projection_cubin::not_cancelled, short_hf_head::StreamedHead,
    short_hf_model::ShortHfProjectionBinding,
};
use super::{
    ShortHfNativeStorage, ShortHfSubmission,
    packets::{self, Key, Role},
};
use crate::{
    resident_inference::{
        ResidentSession,
        native_fragment::{
            PrefixPhaseEnqueue,
            short_hf::{ShortHfCallInput, ShortHfCallState},
        },
    },
    resident_leases::{ResidentLease, ResidentReadback},
    resident_program::text_positions::short_hf::ShortHfPositions,
    vllm_cuda_input_rms::{FullModelRmsScratch, FusedRmsPrepared},
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use zerogravity_artifacts::source_weights::WindowWorkspace;
use zerogravity_core::{
    Error, Result,
    accelerator::program::{PlanOperation, Vllm08Kind, Vllm08Operator},
    device_events::{BufferLeaseId, DeviceAccess},
};

struct Node {
    descriptor: Vllm08Operator,
    projection: Option<ShortHfProjectionBinding>,
    roles: BTreeMap<Role, ResidentLease>,
    packet: super::pool::BoundSite,
    submitted: AtomicUsize,
}
pub(crate) struct ShortHfPrograms {
    storage: Arc<ShortHfNativeStorage>,
    input: Arc<ShortHfCallInput>,
    state: Arc<ShortHfCallState>,
    values: Arc<ModelValues>,
    positions: Arc<ShortHfPositions>,
    attention: Arc<super::attention::Current>,
    head: Arc<StreamedHead>,
    nodes: BTreeMap<usize, Node>,
    accesses: Vec<(ResidentLease, DeviceAccess)>,
    counts: [AtomicUsize; 3],
    _metadata: WindowWorkspace,
}
pub(super) fn insert(
    all: &mut BTreeMap<BufferLeaseId, (ResidentLease, DeviceAccess)>,
    lease: ResidentLease,
    access: DeviceAccess,
) {
    let pair = all.entry(lease.id()).or_insert((lease, access));
    if access == DeviceAccess::Write {
        pair.1 = access;
    }
}
impl ShortHfPrograms {
    pub(crate) fn native_provider_bytes(&self) -> Result<usize> {
        self.storage.images.provider_bytes()
    }

    pub(super) fn bind(
        storage: Arc<ShortHfNativeStorage>,
        input: Arc<ShortHfCallInput>,
        state: Arc<ShortHfCallState>,
        values: Arc<ModelValues>,
        rms_scope: Arc<FullModelRmsScratch>,
        final_rms: Arc<FusedRmsPrepared>,
        positions: Arc<ShortHfPositions>,
    ) -> Result<Arc<Self>> {
        state.require_submission_claimed()?;
        input.scope().require_values(&values)?;
        storage.require_call(&input, &values)?;
        if !Arc::ptr_eq(state.input(), &input)
            || !Arc::ptr_eq(rms_scope.record(), input.record())
            || !positions.retained(&input, values.service())
        {
            return Err(Error::State(
                "short-HF native SAME actual current source/state/positions/RMS".into(),
            ));
        }
        let metadata = input
            .source()
            .source()
            .source()
            .reserve_workspace(2 * 1024 * 1024)?;
        let attention = storage.attention.bind(
            input.clone(),
            state.clone(),
            values.clone(),
            positions.clone(),
        )?;
        let head =
            storage
                .head
                .bind_call(input.scope().clone(), values.clone(), rms_scope, final_rms)?;
        let mut nodes = BTreeMap::new();
        let mut all = BTreeMap::new();
        let mut projections = input
            .scope()
            .bind_projections(values.clone(), &storage.weights)?;
        for (index, node) in input.record().plan().nodes().iter().enumerate() {
            let PlanOperation::TypedVllm08 { descriptor, .. } = &node.operation else {
                continue;
            };
            if !packets::is_native(descriptor.kind()) {
                continue;
            }
            let key = Key::current(input.scope(), index)?;
            let projection =
                if matches!(descriptor.kind(), Vllm08Kind::Projection(_)) {
                    Some(projections.remove(&index).ok_or_else(|| {
                        Error::State("short-HF sealed learned site absent".into())
                    })?)
                } else {
                    None
                };
            let mut roles = attention.roles(index)?;
            if let Some(binding) = &projection {
                roles.insert(Role::W, binding.weight().lease().clone());
                roles.insert(Role::X, binding.input().clone());
                roles.insert(Role::Y, binding.output().clone());
            } else if descriptor.kind() == Vllm08Kind::SiluAndMul {
                let ([x], [y]) = (node.inputs.as_slice(), node.outputs.as_slice()) else {
                    return Err(Error::State("short-HF ACT actual unary source SSA".into()));
                };
                roles.insert(Role::X, values.lease(*x)?.clone());
                roles.insert(Role::Y, values.lease(*y)?.clone());
            }
            roles.insert(Role::Workspace, storage.workspace.clone());
            let packet = storage.pool.bind_site(key, &roles, values.service())?;
            for (lease, access) in packet.accesses() {
                insert(&mut all, lease.clone(), *access);
            }
            nodes.insert(
                index,
                Node {
                    descriptor: descriptor.as_ref().clone(),
                    projection,
                    roles,
                    packet,
                    submitted: AtomicUsize::new(0),
                },
            );
        }
        if nodes.len() != 168 || !projections.is_empty() {
            return Err(Error::State(
                "short-HF exact114+24+30 native source sites".into(),
            ));
        }
        for (lease, access) in head.accesses() {
            insert(&mut all, lease, access);
        }
        for (lease, access) in attention.accesses() {
            insert(&mut all, lease, access);
        }
        let result = Arc::new(Self {
            storage,
            input,
            state,
            values,
            positions,
            attention,
            head,
            nodes,
            accesses: all.into_values().collect(),
            counts: std::array::from_fn(|_| AtomicUsize::new(0)),
            _metadata: metadata,
        });
        result.require_values(&result.input, &result.values)?;
        Ok(result)
    }
    pub(crate) fn require_values(
        &self,
        input: &Arc<ShortHfCallInput>,
        values: &Arc<ModelValues>,
    ) -> Result<()> {
        input.scope().require_values(values)?;
        if !Arc::ptr_eq(input, &self.input)
            || !Arc::ptr_eq(values, &self.values)
            || !Arc::ptr_eq(self.state.input(), input)
            || !Arc::ptr_eq(self.head.record(), input.record())
            || !Arc::ptr_eq(self.storage.weights.source(), input.source().source())
        {
            return Err(Error::State(
                "short-HF immutable current native source/value/state seal".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.accesses
    }
    pub(crate) fn storage_accesses(
        &self,
        node: usize,
    ) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        if node == 254 {
            return Ok(self.head.accesses());
        }
        let mut all = BTreeMap::new();
        for (l, a) in self
            .nodes
            .get(&node)
            .ok_or_else(|| Error::State("short-HF native envelope node absent".into()))?
            .packet
            .accesses()
        {
            insert(&mut all, l.clone(), *a);
        }
        for (l, a) in self.attention.storage_accesses(node)? {
            insert(&mut all, l, a);
        }
        Ok(all.into_values().collect())
    }
    pub(crate) fn enqueue(
        &self,
        node: usize,
        session: &ResidentSession,
        permit: &PrefixPhaseEnqueue<'_>,
        cancelled: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<ShortHfSubmission> {
        not_cancelled(cancelled)?;
        self.input.require_session(session)?;
        self.input.require_permit(permit, self.values.service())?;
        self.values.require_storage_node(node)?;
        self.values.service().ensure_usable()?;
        self.require_values(&self.input, &self.values)?;
        if node == 254 {
            let before = self.head.counts()?;
            let result = self.head.enqueue(node, cancelled);
            let after = self.head.counts()?;
            let actual = ShortHfSubmission {
                native: after[1]
                    .checked_sub(before[1])
                    .ok_or_else(|| Error::State("short-HF Head counter regressed".into()))?,
                copies: after[0]
                    .checked_sub(before[0])
                    .ok_or_else(|| Error::State("short-HF Head upload counter regressed".into()))?,
                clears: 0,
            };
            self.add(actual);
            result?;
            if !self.head.complete()? {
                return Err(Error::State(
                    "short-HF Head returned before actual window coverage".into(),
                ));
            }
            return Ok(actual);
        }
        let bound = self
            .nodes
            .get(&node)
            .ok_or_else(|| Error::State("short-HF native source node absent".into()))?;
        if bound
            .submitted
            .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(Error::State("short-HF native current node replay".into()));
        }
        if let Some(binding) = &bound.projection {
            binding.require(
                &self.values,
                &bound.descriptor,
                binding.input(),
                binding.output(),
            )?;
        }
        let before = self.submission_counts();
        self.attention
            .before(node, session, permit, cancelled, &self.counts)?;
        // The original flight retains self, every role and pooled native owners.
        // Each successful command updates the same counters before any later error.
        bound
            .packet
            .enqueue(self.values.service(), cancelled, &self.counts)?;
        self.attention
            .after(node, session, permit, cancelled, &self.counts)?;
        let after = self.submission_counts();
        after.difference(before)
    }
    fn add(&self, n: ShortHfSubmission) {
        for (counter, n) in self.counts.iter().zip([n.native, n.clears, n.copies]) {
            counter.fetch_add(n, Ordering::Release);
        }
    }
    pub(crate) fn submission_counts(&self) -> ShortHfSubmission {
        ShortHfSubmission {
            native: self.counts[0].load(Ordering::Acquire),
            clears: self.counts[1].load(Ordering::Acquire),
            copies: self.counts[2].load(Ordering::Acquire),
        }
    }
    pub(crate) fn enqueued_records(&self) -> usize {
        self.submission_counts().native
    }
    pub(crate) fn head(&self) -> &Arc<StreamedHead> {
        &self.head
    }
    pub(crate) fn extra_readbacks(&self) -> Result<Vec<ResidentReadback>> {
        let mut all = vec![self.positions.readback()?];
        all.extend(self.attention.status_readbacks()?);
        Ok(all)
    }
    pub(crate) fn validate_extra(&self, raw: &[cubecl::bytes::Bytes]) -> Result<()> {
        if raw.len() != 7 {
            return Err(Error::State(
                "short-HF positions+six actual KV statuses".into(),
            ));
        }
        self.positions.validate(&raw[0])?;
        self.attention.validate_statuses(&raw[1..])
    }
    pub(crate) fn retained_values(&self) -> Result<bool> {
        self.require_values(&self.input, &self.values)?;
        if !self.positions.retained(&self.input, self.values.service())
            || !self
                .attention
                .retained(&self.input, &self.state, &self.values)?
            || !self.head.retained_graph()?
            || !self.storage.retained_source_graph()?
        {
            return Ok(false);
        }
        // No initialized/usable/Ready guard here: unknown completion fences the service.
        for bound in self.nodes.values() {
            for lease in bound.roles.values() {
                if !Arc::ptr_eq(lease.service(), self.values.service())
                    || lease.source_identity() != self.input.call().source()
                    || lease.service_generation() != self.values.service().generation()
                {
                    return Ok(false);
                }
            }
            if let Some(p) = &bound.projection
                && (p.weight().lease().source_identity() != self.input.call().source()
                    || p.input().id() == p.output().id())
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
