//! Original full24 attempt/domain borrow only; no root allocation/publication.
use super::super::*;
use super::continuation::{AttemptPhaseBinding, PrefixPhaseEnqueue};
use zerogravity_core::accelerator::program::{
    ConvDomain, KvDomain, PlanOperation, RecordedVllm08Model, RecurrentDomain, StateEffect,
    Vllm08Call, Vllm08Kind, Vllm08Layer,
};

#[derive(Clone)]
enum ModelStateScope {
    Legacy,
    ShortHf(Arc<crate::vllm_cuda_qualification::short_hf_model::ShortHfModelBinding>),
    Canonical47Prefix,
    Canonical47Published(
        Arc<crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding>,
    ),
    Canonical47Whole(
        Arc<crate::vllm_cuda_qualification::canonical47_model_prefix::Canonical47PrefixCut>,
    ),
}
pub(crate) struct BoundModelState {
    base: RootBinding,
    call: Vllm08Call,
    layer: Vllm08Layer,
    effects: Vec<StateEffect>,
    // GDN: Conv prior/candidate, H prior/candidate. KV: K prior/candidate,
    // V prior/candidate. Actual candidate allocations belong to RootAttempt.
    leases: [ResidentLease; 4],
    phase: Arc<AttemptPhaseBinding>,
    scope: ModelStateScope,
}

fn canonical_vector(domain: &Domain, slot: StateSlot, elements: usize) -> Result<ResidentLease> {
    let StateBuffers::Vector(buffer) = &domain.buffers else {
        return Err(Error::State("full24 canonical vector domain layout".into()));
    };
    if domain.slot != slot || domain.leases.len() != 1 || domain.generations.len() != 1 {
        return Err(Error::State("full24 unique original vector domain".into()));
    }
    let lease = &domain.leases[0];
    buffer.require_layout(ResidentBufferLayout::new(
        ResidentPhysicalType::F32,
        elements,
    )?)?;
    if lease.bytes() != elements * 4
        || lease.role() != BufferRole::RecurrentState
        || buffer.original_lease()?.id() != lease.id()
        || lease.allocation_generation() != domain.generations[0]
        || !same_handle(&buffer.handle, &lease.handle())
    {
        return Err(Error::State(
            "full24 vector exact canonical lease/generation".into(),
        ));
    }
    Ok(lease.clone())
}
fn canonical_kv(domain: &Domain, slot: StateSlot) -> Result<[ResidentLease; 2]> {
    let StateBuffers::Kv { keys, values } = &domain.buffers else {
        return Err(Error::State(
            "full24 canonical paired KV domain layout".into(),
        ));
    };
    if domain.slot != slot || domain.leases.len() != 2 || domain.generations.len() != 2 {
        return Err(Error::State(
            "full24 unique original paired KV domain".into(),
        ));
    }
    for ((buffer, lease), generation) in [keys, values]
        .into_iter()
        .zip(&domain.leases)
        .zip(&domain.generations)
    {
        buffer.require_layout(ResidentBufferLayout::new(
            ResidentPhysicalType::F32,
            512 * 512,
        )?)?;
        if lease.bytes() != 512 * 512 * 4
            || lease.role() != BufferRole::RecurrentState
            || buffer.original_lease()?.id() != lease.id()
            || lease.allocation_generation() != *generation
            || !same_handle(&buffer.handle, &lease.handle())
        {
            return Err(Error::State(
                "full24 KV exact canonical lease/generation".into(),
            ));
        }
    }
    Ok([domain.leases[0].clone(), domain.leases[1].clone()])
}
fn domain(domains: &[Domain], slot: StateSlot) -> Result<&Domain> {
    let mut found = domains.iter().filter(|domain| domain.slot == slot);
    let value = found
        .next()
        .ok_or_else(|| Error::State("full24 original domain absent".into()))?;
    if found.next().is_some() {
        return Err(Error::State("full24 original domain repeated".into()));
    }
    Ok(value)
}

impl ResidentSession {
    /// The existing transaction allocates/carries all42 original domains.
    /// A genuine previous phase may be prepared unready; enqueue requires its
    /// actual original completion through the same borrowed session permit.
    pub(super) fn bind_full_model_state(
        &self,
        attempt: &RootAttempt,
        record: &RecordedVllm08Model,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        self.bind_model_state_impl(attempt, record, layer, candidate, phase, false)
    }
    pub(super) fn bind_canonical47_model_prefix_state(
        &self,
        attempt: &RootAttempt,
        scope: &crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if scope.kind()
            != crate::vllm_cuda_qualification::canonical47_model::Canonical47CallKind::Prefill
            || layer.index() >= 3
            || layer.is_attention()
        {
            return Err(Error::Unsupported(
                "canonical47 first3 actual model GDN domains only".into(),
            ));
        }
        scope.gdn_site(layer.index())?;
        self.bind_model_state_impl(attempt, scope.record(), layer, candidate, phase, true)
    }
    fn bind_model_state_impl(
        &self,
        attempt: &RootAttempt,
        record: &RecordedVllm08Model,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
        canonical47_prefix: bool,
    ) -> Result<Arc<BoundModelState>> {
        self.bind_model_state_scoped(
            attempt,
            record,
            layer,
            candidate,
            phase,
            if canonical47_prefix {
                ModelStateScope::Canonical47Prefix
            } else {
                ModelStateScope::Legacy
            },
        )
    }
    fn bind_model_state_scoped(
        &self,
        attempt: &RootAttempt,
        record: &RecordedVllm08Model,
        layer: Vllm08Layer,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
        scope: ModelStateScope,
    ) -> Result<Arc<BoundModelState>> {
        let call = record.call();
        call.validate()?;
        match &scope {
            ModelStateScope::Legacy => phase.require_full_model_binding(self, attempt, call)?,
            ModelStateScope::ShortHf(scope) => {
                if !std::ptr::eq(scope.record().as_ref(), record) {
                    return Err(Error::State("short-HF exact state record differs".into()));
                }
                phase.require_short_hf_binding(self, attempt, call)?;
            }
            ModelStateScope::Canonical47Prefix => {
                phase.require_canonical47_model_prefix_binding(self, attempt, call)?
            }
            ModelStateScope::Canonical47Published(scope) => {
                if !std::ptr::eq(scope.record().as_ref(), record) {
                    return Err(Error::State(
                        "canonical published exact record differs".into(),
                    ));
                }
                phase.require_canonical47_published_binding(self, attempt, call)?;
            }
            ModelStateScope::Canonical47Whole(cut) => {
                cut.require_whole()?;
                if !std::ptr::eq(cut.scope().record().as_ref(), record) {
                    return Err(Error::State("whole state exact record differs".into()));
                }
                phase.require_canonical47_whole_binding(self, attempt, call)?;
            }
        }
        let base = self.binding();
        let plan = record.plan();
        if self.model.capacity != 512
            || plan.source() != base.source()
            || call.source() != base.source()
            || layer != call.layer(layer.index())?
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
            || plan.required_domains().len() != 42
            || plan.nodes().len() != 255
        {
            return Err(Error::Unsupported(
                "original full24 complete unpublished source/root recipe".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        let prior = phase.prior_domains(self);
        self.model.validate_domains(prior, phase.prior_is_ready())?;
        self.model.validate_domains(candidate, false)?;
        let slots = if layer.is_attention() {
            vec![(StateSlot::Kv(KvDomain(layer.index())), Vllm08Kind::KvAppend)]
        } else {
            vec![
                (
                    StateSlot::Conv(ConvDomain(layer.index())),
                    Vllm08Kind::GdnConv,
                ),
                (
                    StateSlot::Recurrent(RecurrentDomain(layer.index())),
                    Vllm08Kind::GdnRecurrent,
                ),
            ]
        };
        let mut effects = Vec::with_capacity(slots.len());
        for (slot, kind) in &slots {
            let mut matching = plan.nodes().iter().filter(|node| {
                matches!(&node.operation,
                PlanOperation::TypedVllm08 { descriptor, .. }
                    if descriptor.layer() == Some(layer) && descriptor.kind() == *kind)
            });
            let node = matching
                .next()
                .ok_or_else(|| Error::State("full24 actual indexed state node absent".into()))?;
            if matching.next().is_some() || !plan.required_domains().contains(slot) {
                return Err(Error::State(
                    "full24 one actual indexed state effect".into(),
                ));
            }
            let PlanOperation::TypedVllm08 { descriptor, state } = &node.operation else {
                unreachable!()
            };
            descriptor.validate()?;
            let effect = node
                .state
                .ok_or_else(|| Error::State("full24 actual state effect absent".into()))?;
            if descriptor.call() != call
                || state.is_some()
                || effect.read.slot != *slot
                || effect.write.slot != *slot
                || effect.read.ordinal != 0
                || effect.write.ordinal != 1
            {
                return Err(Error::State(
                    "full24 source/teacher/state version differs".into(),
                ));
            }
            for &id in node.inputs.iter().chain(&node.outputs) {
                plan.vllm_value_provenance(id)?
                    .ok_or_else(|| {
                        Error::State("full24 state producer teacher provenance absent".into())
                    })?
                    .require_invocation(call.request())?;
            }
            effects.push(effect);
        }
        let leases = if layer.is_attention() {
            let slot = slots[0].0;
            let [pk, pv] = canonical_kv(domain(prior, slot)?, slot)?;
            let [ck, cv] = canonical_kv(domain(candidate, slot)?, slot)?;
            [pk, ck, pv, cv]
        } else {
            let [conv, recurrent] = [slots[0].0, slots[1].0];
            [
                canonical_vector(domain(prior, conv)?, conv, 6144 * 3)?,
                canonical_vector(domain(candidate, conv)?, conv, 6144 * 3)?,
                canonical_vector(domain(prior, recurrent)?, recurrent, 16 * 128 * 128)?,
                canonical_vector(domain(candidate, recurrent)?, recurrent, 16 * 128 * 128)?,
            ]
        };
        let mut ids = std::collections::BTreeSet::new();
        for (index, lease) in leases.iter().enumerate() {
            if !Arc::ptr_eq(lease.service(), &self.model.service)
                || lease.source_identity() != base.source()
                || lease.service_generation() != self.model.service.generation()
                || !ids.insert(lease.id())
            {
                return Err(Error::State(
                    "full24 independent original prior/candidate domain owners".into(),
                ));
            }
            if index % 2 == 0 {
                phase.require_prepared_prior(lease)?;
            }
        }
        Ok(Arc::new(BoundModelState {
            base,
            call: call.clone(),
            layer,
            effects,
            leases,
            phase,
            scope,
        }))
    }
}
impl BoundModelState {
    pub(crate) fn call(&self) -> &Vllm08Call {
        &self.call
    }
    pub(crate) fn layer(&self) -> Vllm08Layer {
        self.layer
    }
    pub(crate) fn base(&self) -> RootBinding {
        self.base
    }
    pub(crate) fn same_request(&self, other: &Self) -> bool {
        match (&self.scope, &other.scope) {
            (ModelStateScope::ShortHf(a), ModelStateScope::ShortHf(b)) => {
                Arc::ptr_eq(a, b) && self.phase.same_short_hf_request(&other.phase)
            }
            (ModelStateScope::Legacy, ModelStateScope::Legacy) => {
                self.phase.same_full_model_request(&other.phase)
            }
            (ModelStateScope::Canonical47Prefix, ModelStateScope::Canonical47Prefix) => self
                .phase
                .same_canonical47_model_prefix_request(&other.phase),
            (
                ModelStateScope::Canonical47Published(a),
                ModelStateScope::Canonical47Published(b),
            ) => Arc::ptr_eq(a, b) && self.phase.same_canonical47_published_request(&other.phase),
            (ModelStateScope::Canonical47Whole(a), ModelStateScope::Canonical47Whole(b)) => {
                Arc::ptr_eq(a, b) && self.phase.same_canonical47_whole_request(&other.phase)
            }
            _ => false,
        }
    }
    pub(crate) fn leases(&self) -> [(ResidentLease, DeviceAccess); 4] {
        std::array::from_fn(|index| {
            (
                self.leases[index].clone(),
                if index % 2 == 0 {
                    DeviceAccess::Read
                } else {
                    DeviceAccess::Write
                },
            )
        })
    }
    pub(crate) fn require_enqueue(
        &self,
        service: &Arc<ResidentService>,
        permit: &PrefixPhaseEnqueue<'_>,
    ) -> Result<()> {
        match &self.scope {
            ModelStateScope::Legacy => self
                .phase
                .require_full_model_permit(permit, service, &self.call)?,
            ModelStateScope::ShortHf(_) => self
                .phase
                .require_short_hf_permit(permit, service, &self.call)?,
            ModelStateScope::Canonical47Prefix => self
                .phase
                .require_canonical47_model_prefix_permit(permit, service, &self.call)?,
            ModelStateScope::Canonical47Published(_) => self
                .phase
                .require_canonical47_published_permit(permit, service, &self.call)?,
            ModelStateScope::Canonical47Whole(_) => self
                .phase
                .require_canonical47_whole_permit(permit, service, &self.call)?,
        }
        self.phase.require_launch_ready(service)?;
        for prior in [&self.leases[0], &self.leases[2]] {
            if !prior.is_initialized() {
                return Err(Error::State(
                    "full24 canonical prior lacks actual physical completion".into(),
                ));
            }
        }
        Ok(())
    }
    pub(crate) fn require_effect(&self, effect: StateEffect, recurrent: bool) -> Result<()> {
        let index = usize::from(recurrent);
        if self.effects.get(index) != Some(&effect) {
            return Err(Error::State(
                "full24 node differs from actual bound indexed effect".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_canonical47_model_prefix(
        &self,
        scope: &crate::vllm_cuda_qualification::canonical47_model::Canonical47ModelBinding,
    ) -> Result<()> {
        if !matches!(&self.scope, ModelStateScope::Canonical47Prefix)
            || self.call != *scope.record().call()
            || self.layer.index() >= 3
            || self.layer.is_attention()
            || self.base.source() != scope.source().source().source_identity()
            || self.base.frontier() != 0
        {
            return Err(Error::State(
                "canonical47 genuine model source/layer/root domains differ".into(),
            ));
        }
        scope.gdn_site(self.layer.index())?;
        Ok(())
    }
    pub(crate) fn effect(&self, recurrent: bool) -> Result<StateEffect> {
        self.effects
            .get(usize::from(recurrent))
            .copied()
            .ok_or_else(|| Error::State("full24 bound effect kind absent".into()))
    }
}

mod canonical47_kv;

mod canonical47_published;
mod canonical47_whole;

mod short_hf;
