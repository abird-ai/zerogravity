//! Original full24 attempt/domain borrow only; no root allocation/publication.
use super::super::*;
use super::continuation::{AttemptPhaseBinding, PrefixPhaseEnqueue};
use crate::resident_program::source_bindings::SourceBindings;
use zerogravity_core::accelerator::program::ConvDomain;
use zerogravity_core::accelerator::program::KvDomain;
use zerogravity_core::accelerator::program::PlanOperation;
use zerogravity_core::accelerator::program::RecurrentDomain;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::accelerator::program::{SourceCall, SourceOperatorKind};

#[derive(Clone)]
enum ModelStateScope {
    Legacy,
    Text(Arc<crate::cuda::native::text_model::TextModelBinding>),
    FixedPromptPrefix,
    FixedPromptPublished(Arc<crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding>),
    FixedPromptWhole(Arc<crate::cuda::native::fixed_prompt_model_prefix::FixedPromptPrefixCut>),
}
pub(crate) struct BoundModelState {
    base: RootBinding,
    call: SourceCall,
    layer: usize,
    effects: Vec<StateEffect>,
    // GDN: Conv prior/candidate, H prior/candidate. KV: K prior/candidate,
    // V prior/candidate. Actual candidate allocations belong to RootAttempt.
    leases: [ResidentLease; 4],
    phase: Arc<AttemptPhaseBinding>,
    scope: ModelStateScope,
}

fn canonical_vector(domain: &Domain, slot: StateSlot, elements: usize) -> Result<ResidentLease> {
    let bytes = elements
        .checked_mul(4)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source F32 vector state byte extent".into()))?;
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
    if lease.bytes() != bytes
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
fn canonical_kv(domain: &Domain, slot: StateSlot, elements: usize) -> Result<[ResidentLease; 2]> {
    let bytes = elements
        .checked_mul(4)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source F32 KV state plane byte extent".into()))?;
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
            elements,
        )?)?;
        if lease.bytes() != bytes
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
        record: &SourceBindings,
        layer: usize,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        self.bind_model_state_impl(attempt, record, layer, candidate, phase, false)
    }
    pub(super) fn bind_fixed_prompt_model_prefix_state(
        &self,
        attempt: &RootAttempt,
        scope: &crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding,
        layer: usize,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundModelState>> {
        if scope.kind() != crate::cuda::native::fixed_prompt_model::FixedPromptCallKind::Prefill
            || layer >= 3
        {
            return Err(Error::Unsupported(
                "canonical47 first3 actual model GDN domains only".into(),
            ));
        }
        scope.gdn_site(layer)?;
        self.bind_model_state_impl(attempt, scope.record(), layer, candidate, phase, true)
    }
    fn bind_model_state_impl(
        &self,
        attempt: &RootAttempt,
        record: &SourceBindings,
        layer: usize,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
        fixed_prompt_prefix: bool,
    ) -> Result<Arc<BoundModelState>> {
        self.bind_model_state_scoped(
            attempt,
            record,
            layer,
            candidate,
            phase,
            if fixed_prompt_prefix {
                ModelStateScope::FixedPromptPrefix
            } else {
                ModelStateScope::Legacy
            },
        )
    }
    fn bind_model_state_scoped(
        &self,
        attempt: &RootAttempt,
        record: &SourceBindings,
        layer: usize,
        candidate: &[Domain],
        phase: Arc<AttemptPhaseBinding>,
        scope: ModelStateScope,
    ) -> Result<Arc<BoundModelState>> {
        let call = record.call();
        match &scope {
            ModelStateScope::Legacy => phase.require_full_model_binding(self, attempt, call)?,
            ModelStateScope::Text(scope) => {
                if !std::ptr::eq(scope.record().as_ref(), record) {
                    return Err(Error::State("short-HF exact state record differs".into()));
                }
                phase.require_text_binding(self, attempt, call)?;
            }
            ModelStateScope::FixedPromptPrefix => {
                phase.require_fixed_prompt_model_prefix_binding(self, attempt, call)?
            }
            ModelStateScope::FixedPromptPublished(scope) => {
                if !std::ptr::eq(scope.record().as_ref(), record) {
                    return Err(Error::State(
                        "canonical published exact record differs".into(),
                    ));
                }
                phase.require_fixed_prompt_published_binding(self, attempt, call)?;
            }
            ModelStateScope::FixedPromptWhole(cut) => {
                cut.require_whole()?;
                if !std::ptr::eq(cut.scope().record().as_ref(), record) {
                    return Err(Error::State("whole state exact record differs".into()));
                }
                phase.require_fixed_prompt_whole_binding(self, attempt, call)?;
            }
        }
        let base = self.binding();
        let plan = record.plan();
        if self.model.capacity != call.rows().capacity()
            || plan.source() != base.source()
            || call.source() != base.source()
            || plan.source_call() != Some(call)
            || plan.frontier_delta() != 0
            || plan.logits().is_some()
            || self.model.states.len() != record.state_requirements().len()
            || record.state_requirements().iter().any(|required| {
                self.model
                    .states
                    .iter()
                    .find(|actual| actual.slot == required.slot)
                    .is_none_or(|actual| {
                        actual.bytes != required.bytes
                            || actual.layout.planes().ok() != required.layout.planes().ok()
                    })
            })
        {
            return Err(Error::Unsupported(
                "original full24 complete unpublished source/root recipe".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        let prior = phase.prior_domains(self);
        self.model.validate_domains(prior, phase.prior_is_ready())?;
        self.model.validate_domains(candidate, false)?;
        // Domain kind comes from actual operators, never layer position or names.
        let mut slots = Vec::with_capacity(2);
        for (index, node) in plan.nodes().iter().enumerate() {
            let PlanOperation::SourceOperator {
                descriptor,
                state: None,
            } = &node.operation
            else {
                continue;
            };
            let slot = match descriptor.kind() {
                SourceOperatorKind::CausalConvSplit { domain, .. } if domain.0 == layer => {
                    StateSlot::Conv(domain)
                }
                SourceOperatorKind::GatedDelta { domain, .. } if domain.0 == layer => {
                    StateSlot::Recurrent(domain)
                }
                SourceOperatorKind::KvAppend { domain, .. } if domain.0 == layer => {
                    StateSlot::Kv(domain)
                }
                _ => continue,
            };
            if slots.iter().any(|(old, _)| *old == slot) {
                return Err(Error::State("source state operator repeats domain".into()));
            }
            slots.push((slot, index));
        }
        slots.sort_by_key(|(slot, _)| *slot);
        let is_kv = matches!(slots.as_slice(), [(StateSlot::Kv(_), _)]);
        if !is_kv
            && !matches!(
                slots.as_slice(),
                [(StateSlot::Conv(_), _), (StateSlot::Recurrent(_), _)]
            )
        {
            return Err(Error::Unsupported(
                "source state requires actual GDN pair or KV operator".into(),
            ));
        }
        let mut effects = Vec::with_capacity(slots.len());
        for &(slot, index) in &slots {
            let node = &plan.nodes()[index];
            let PlanOperation::SourceOperator {
                descriptor,
                state: None,
            } = &node.operation
            else {
                unreachable!()
            };
            let effect = node
                .state
                .ok_or_else(|| Error::State("source state effect absent".into()))?;
            if descriptor.call() != call
                || !plan.required_domains().contains(&slot)
                || effect.read.slot != slot
                || effect.write.slot != slot
                || effect.read.ordinal != 0
                || effect.write.ordinal != 1
            {
                return Err(Error::State(
                    "source state call/domain/version differs".into(),
                ));
            }
            for &id in node.inputs.iter().chain(&node.outputs) {
                let tag = plan.source_value_provenance(id)?.ok_or_else(|| {
                    Error::State("source state producer provenance absent".into())
                })?;
                if tag.call() != call {
                    return Err(Error::State(
                        "source state producer call issuer differs".into(),
                    ));
                }
            }
            effects.push(effect);
        }
        let required = |slot| {
            record
                .state_requirements()
                .iter()
                .find(|r| r.slot == slot)
                .ok_or_else(|| Error::State("source state storage requirement absent".into()))
        };
        let leases = if is_kv {
            let slot = slots[0].0;
            let (planes, words) = required(slot)?.layout.planes()?;
            if planes != 2 {
                return Err(Error::State("source KV storage planes".into()));
            }
            let [pk, pv] = canonical_kv(domain(prior, slot)?, slot, words)?;
            let [ck, cv] = canonical_kv(domain(candidate, slot)?, slot, words)?;
            [pk, ck, pv, cv]
        } else {
            let conv = slots[0].0;
            let recurrent = slots[1].0;
            let (cp, cw) = required(conv)?.layout.planes()?;
            let (rp, rw) = required(recurrent)?.layout.planes()?;
            if cp != 1 || rp != 1 {
                return Err(Error::State("source vector storage planes".into()));
            }
            [
                canonical_vector(domain(prior, conv)?, conv, cw)?,
                canonical_vector(domain(candidate, conv)?, conv, cw)?,
                canonical_vector(domain(prior, recurrent)?, recurrent, rw)?,
                canonical_vector(domain(candidate, recurrent)?, recurrent, rw)?,
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
    pub(crate) fn call(&self) -> &SourceCall {
        &self.call
    }
    pub(crate) fn layer(&self) -> usize {
        self.layer
    }
    pub(crate) fn conv_domain(&self) -> Result<ConvDomain> {
        match self.effects.first().map(|effect| effect.write.slot) {
            Some(StateSlot::Conv(domain)) => Ok(domain),
            _ => Err(Error::State(
                "source state is not a convolution domain".into(),
            )),
        }
    }
    pub(crate) fn recurrent_domain(&self) -> Result<RecurrentDomain> {
        match self.effects.get(1).map(|effect| effect.write.slot) {
            Some(StateSlot::Recurrent(domain)) => Ok(domain),
            _ => Err(Error::State(
                "source state is not a recurrent domain".into(),
            )),
        }
    }
    pub(crate) fn kv_domain(&self) -> Result<KvDomain> {
        match self.effects.first().map(|effect| effect.write.slot) {
            Some(StateSlot::Kv(domain)) => Ok(domain),
            _ => Err(Error::State("source state is not a KV domain".into())),
        }
    }
    pub(crate) fn base(&self) -> RootBinding {
        self.base
    }
    pub(crate) fn same_request(&self, other: &Self) -> bool {
        match (&self.scope, &other.scope) {
            (ModelStateScope::Text(a), ModelStateScope::Text(b)) => {
                Arc::ptr_eq(a, b) && self.phase.same_text_request(&other.phase)
            }
            (ModelStateScope::Legacy, ModelStateScope::Legacy) => {
                self.phase.same_full_model_request(&other.phase)
            }
            (ModelStateScope::FixedPromptPrefix, ModelStateScope::FixedPromptPrefix) => self
                .phase
                .same_fixed_prompt_model_prefix_request(&other.phase),
            (
                ModelStateScope::FixedPromptPublished(a),
                ModelStateScope::FixedPromptPublished(b),
            ) => Arc::ptr_eq(a, b) && self.phase.same_fixed_prompt_published_request(&other.phase),
            (ModelStateScope::FixedPromptWhole(a), ModelStateScope::FixedPromptWhole(b)) => {
                Arc::ptr_eq(a, b) && self.phase.same_fixed_prompt_whole_request(&other.phase)
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
            ModelStateScope::Text(_) => self
                .phase
                .require_text_permit(permit, service, &self.call)?,
            ModelStateScope::FixedPromptPrefix => self
                .phase
                .require_fixed_prompt_model_prefix_permit(permit, service, &self.call)?,
            ModelStateScope::FixedPromptPublished(_) => self
                .phase
                .require_fixed_prompt_published_permit(permit, service, &self.call)?,
            ModelStateScope::FixedPromptWhole(_) => self
                .phase
                .require_fixed_prompt_whole_permit(permit, service, &self.call)?,
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
    pub(crate) fn require_fixed_prompt_model_prefix(
        &self,
        scope: &crate::cuda::native::fixed_prompt_model::FixedPromptModelBinding,
    ) -> Result<()> {
        if !matches!(&self.scope, ModelStateScope::FixedPromptPrefix)
            || self.call != *scope.record().call()
            || self.layer >= 3
            || self.kv_domain().is_ok()
            || self.base.source() != scope.source().source_identity()
            || self.base.frontier() != 0
        {
            return Err(Error::State(
                "canonical47 genuine model source/layer/root domains differ".into(),
            ));
        }
        scope.gdn_site(self.layer)?;
        Ok(())
    }
    pub(crate) fn effect(&self, recurrent: bool) -> Result<StateEffect> {
        self.effects
            .get(usize::from(recurrent))
            .copied()
            .ok_or_else(|| Error::State("full24 bound effect kind absent".into()))
    }
}

mod fixed_prompt_kv;

mod fixed_prompt_published;
mod fixed_prompt_whole;

mod text;
