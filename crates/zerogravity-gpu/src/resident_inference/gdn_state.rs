//! Staged original session-domain borrow. No state allocation/publication.
use super::native_fragment::continuation::{AttemptPhaseBinding, PrefixPhaseEnqueue};
use super::*;
use crate::cuda::gdn::GdnPhase;
use zerogravity_core::accelerator::program::ConvDomain;
use zerogravity_core::accelerator::program::ProgramNumericalProfile;
use zerogravity_core::accelerator::program::RecurrentDomain;
use zerogravity_core::accelerator::program::StateEffect;
use zerogravity_core::accelerator::program::{
    PlanOperation, SourceCall, SourceCallMode, SourceOperatorKind,
};

/// Sealed original root + original prior/private candidate leases. Only the
/// existing ResidentSession can construct this binding; not an execution permit.
pub(crate) struct BoundGdnDomains {
    base: RootBinding,
    invocation: SourceCall,
    phase: GdnPhase,
    conv_effect: StateEffect,
    recurrent_effect: StateEffect,
    prior_conv: ResidentLease,
    proposed_conv: ResidentLease,
    prior_recurrent: ResidentLease,
    proposed_recurrent: ResidentLease,
    attempt_phase: Option<Arc<AttemptPhaseBinding>>,
}

fn validate_phase(current: RootBinding, attempt: &RootAttempt, phase: GdnPhase) -> Result<()> {
    if current != attempt.base()
        || attempt.frontier().checked_sub(current.frontier()) != Some(phase.tokens() as u64)
        || !match phase {
            GdnPhase::InitialOne | GdnPhase::Initial64 => current.frontier() == 0,
            GdnPhase::ContinuedTwo => current.frontier() == 64,
            GdnPhase::CachedDecode => matches!(current.frontier(), 1 | 66),
            GdnPhase::FixedPrompt | GdnPhase::TextPrefill(_) => false,
        }
    {
        return Err(Error::State(
            "original GDN root/attempt/phase frontier".into(),
        ));
    }
    Ok(())
}
fn vector_lease(domains: &[Domain], slot: StateSlot, elements: usize) -> Result<ResidentLease> {
    let bytes = elements
        .checked_mul(4)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Limit("source F32 vector state byte extent".into()))?;
    let mut found = domains.iter().filter(|domain| domain.slot == slot);
    let domain = found
        .next()
        .ok_or_else(|| Error::State("missing original GDN domain".into()))?;
    if found.next().is_some() || domain.leases.len() != 1 || domain.generations.len() != 1 {
        return Err(Error::State("unique original GDN vector domain".into()));
    }
    let StateBuffers::Vector(buffer) = &domain.buffers else {
        return Err(Error::State("original GDN vector layout".into()));
    };
    let lease = &domain.leases[0];
    if buffer.elements != elements
        || lease.bytes() != bytes
        || !same_handle(&buffer.handle, &lease.handle())
        || lease.allocation_generation() != domain.generations[0]
    {
        return Err(Error::State(
            "original GDN canonical F32 domain extent/generation".into(),
        ));
    }
    Ok(lease.clone())
}
pub(super) struct GdnPhasePlan<'a> {
    pub candidate: &'a [Domain],
    pub plan: &'a ForwardPlan,
    pub invocation: &'a SourceCall,
    pub phase: GdnPhase,
    pub nodes: [usize; 2],
}
impl ResidentSession {
    /// Called by the SAME existing execution path after its original root
    /// attempt/candidate admission. Full model guards stay closed meanwhile.
    pub(super) fn bind_gdn_prefix_state(
        &self,
        attempt: &RootAttempt,
        candidate: &[Domain],
        plan: &ForwardPlan,
        invocation: &SourceCall,
        phase: GdnPhase,
        node_indices: [usize; 2],
    ) -> Result<Arc<BoundGdnDomains>> {
        let base = self.binding();
        validate_phase(base, attempt, phase)?;
        if !matches!(phase, GdnPhase::InitialOne | GdnPhase::Initial64)
            || u64::try_from(invocation.rows().base()).ok() != Some(base.frontier())
        {
            return Err(Error::Unsupported(
                "original initial prefix binder only".into(),
            ));
        }
        self.bind_gdn_phase_inner(
            GdnPhasePlan {
                candidate,
                plan,
                invocation,
                phase,
                nodes: node_indices,
            },
            &self.domains,
            None,
        )
    }
    pub(super) fn bind_gdn_attempt_phase_state(
        &self,
        attempt: &RootAttempt,
        request: GdnPhasePlan<'_>,
        binding: Arc<AttemptPhaseBinding>,
    ) -> Result<Arc<BoundGdnDomains>> {
        let node = request
            .plan
            .nodes()
            .get(request.nodes[0])
            .ok_or_else(|| Error::State("GDN phase actual conv node absent".into()))?;
        let PlanOperation::SourceOperator { descriptor, .. } = &node.operation else {
            return Err(Error::State(
                "GDN phase actual source operator absent".into(),
            ));
        };
        let SourceOperatorKind::CausalConvSplit { domain, .. } = descriptor.kind() else {
            return Err(Error::State("GDN phase actual conv domain absent".into()));
        };
        if descriptor.call() != request.invocation {
            return Err(Error::State("GDN phase actual conv issuer differs".into()));
        }
        binding.require_binding(self, attempt, request.invocation, domain.0)?;
        let prior = binding.prior_domains(self);
        self.model
            .validate_domains(prior, binding.prior_is_ready())?;
        self.bind_gdn_phase_inner(request, prior, Some(binding.clone()))
    }
    fn bind_gdn_phase_inner(
        &self,
        request: GdnPhasePlan<'_>,
        prior: &[Domain],
        attempt_phase: Option<Arc<AttemptPhaseBinding>>,
    ) -> Result<Arc<BoundGdnDomains>> {
        let GdnPhasePlan {
            candidate,
            plan,
            invocation,
            phase,
            nodes: node_indices,
        } = request;
        let base = self.binding();
        let exact_phase = match (
            phase,
            invocation.mode(),
            invocation.rows().base(),
            invocation.rows().len(),
        ) {
            (GdnPhase::InitialOne, SourceCallMode::Prefill, 0, 1)
            | (GdnPhase::Initial64, SourceCallMode::Prefill, 0, 64)
            | (GdnPhase::ContinuedTwo, SourceCallMode::Prefill, 64, 2)
            | (GdnPhase::CachedDecode, SourceCallMode::Decode, 1 | 66, 1) => true,
            (GdnPhase::FixedPrompt, SourceCallMode::Prefill, 0, 47) => {
                attempt_phase
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "fixed prompt component requires original attempt-bound state".into(),
                        )
                    })?
                    .require_initial_component_call(invocation)?;
                true
            }
            _ => false,
        };
        if !exact_phase
            || invocation.rows().source() != base.source()
            || invocation.rows().capacity() != self.model.capacity
            || u64::try_from(invocation.rows().vocabulary()).ok() != Some(self.model.vocab)
            || plan.source() != base.source()
            || base.source() != self.model.service.source_identity()
            || plan.numerical_profile() != ProgramNumericalProfile::SourceCudaSm86Bf16V1
            || plan.is_operator_slice()
            || plan.frontier_delta() != 0
            || plan.source_call() != Some(invocation)
        {
            return Err(Error::Unsupported(
                "GDN actual source/call/root phase differs".into(),
            ));
        }
        let actual =
            |index: usize| -> Result<&zerogravity_core::accelerator::program::SourceOperator> {
                let node = plan
                    .nodes()
                    .get(index)
                    .ok_or_else(|| Error::State("GDN actual state node absent".into()))?;
                let PlanOperation::SourceOperator {
                    descriptor,
                    state: None,
                } = &node.operation
                else {
                    return Err(Error::State("GDN actual state operator absent".into()));
                };
                if descriptor.call() != invocation {
                    return Err(Error::State("GDN actual state call issuer differs".into()));
                }
                Ok(descriptor.as_ref())
            };
        let (
            SourceOperatorKind::CausalConvSplit {
                domain: conv_domain,
                geometry: cg,
            },
            SourceOperatorKind::GatedDelta {
                domain: recurrent_domain,
                geometry: rg,
            },
        ) = (
            actual(node_indices[0])?.kind(),
            actual(node_indices[1])?.kind(),
        )
        else {
            return Err(Error::State(
                "GDN actual state mathematical domain kinds".into(),
            ));
        };
        if conv_domain.0 != recurrent_domain.0 {
            return Err(Error::State(
                "GDN actual paired domain coordinates differ".into(),
            ));
        }
        let elements = [
            cg.qkv_channels
                .checked_mul(
                    cg.kernel
                        .checked_sub(1)
                        .ok_or_else(|| Error::State("GDN convolution history geometry".into()))?,
                )
                .ok_or_else(|| Error::Limit("GDN convolution history extent".into()))?,
            rg.value_heads
                .checked_mul(rg.key_width)
                .and_then(|n| n.checked_mul(rg.value_width))
                .ok_or_else(|| Error::Limit("GDN recurrent state extent".into()))?,
        ];
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&self.domains, true)?;
        self.model.validate_domains(candidate, false)?;
        let conv = StateSlot::Conv(conv_domain);
        let recurrent = StateSlot::Recurrent(recurrent_domain);
        if [conv, recurrent]
            .iter()
            .any(|slot| !plan.required_domains().contains(slot))
        {
            return Err(Error::State("GDN domains absent from same plan".into()));
        }
        if node_indices[0] >= node_indices[1] {
            return Err(Error::State(
                "ordered distinct GDN convolution/recurrent nodes".into(),
            ));
        }
        let effect = |slot: StateSlot, index: usize| -> Result<StateEffect> {
            // Direct immutable records from the lowerer; no producer graph walk.
            let node = plan
                .nodes()
                .get(index)
                .ok_or_else(|| Error::State("missing original GDN node index".into()))?;
            let value = node
                .state
                .ok_or_else(|| Error::State("missing GDN domain effect".into()))?;
            if node.inputs.is_empty() || node.outputs.is_empty() {
                return Err(Error::State(
                    "GDN typed producer/consumer provenance arity".into(),
                ));
            }
            for &id in node.inputs.iter().chain(&node.outputs) {
                let tag = plan.source_value_provenance(id)?.ok_or_else(|| {
                    Error::State("GDN value lost original teacher provenance".into())
                })?;
                if tag.call() != invocation {
                    return Err(Error::State(
                        "GDN value phase differs from original root invocation".into(),
                    ));
                }
            }
            if value.write.slot != slot
                || value.read.slot != slot
                || value.read.ordinal != 0
                || value.write.ordinal != 1
            {
                return Err(Error::State("one original GDN domain proposal".into()));
            }
            Ok(value)
        };
        let effects = [
            effect(conv, node_indices[0])?,
            effect(recurrent, node_indices[1])?,
        ];
        let bound = Self::gdn_domain_binding(
            base,
            invocation.clone(),
            phase,
            prior,
            candidate,
            effects,
            elements,
            attempt_phase,
        )?;
        Ok(Arc::new(bound))
    }
    fn gdn_domain_binding(
        base: RootBinding,
        invocation: SourceCall,
        phase: GdnPhase,
        prior: &[Domain],
        candidate: &[Domain],
        effects: [StateEffect; 2],
        elements: [usize; 2],
        attempt_phase: Option<Arc<AttemptPhaseBinding>>,
    ) -> Result<BoundGdnDomains> {
        let [conv, recurrent] = [effects[0].write.slot, effects[1].write.slot];
        if !matches!((conv, recurrent),
            (StateSlot::Conv(c),StateSlot::Recurrent(r)) if c.0 == r.0)
        {
            return Err(Error::State(
                "GDN retained effect domain kinds differ".into(),
            ));
        }
        let value = BoundGdnDomains {
            base,
            invocation,
            phase,
            conv_effect: effects[0],
            recurrent_effect: effects[1],
            prior_conv: vector_lease(prior, conv, elements[0])?,
            proposed_conv: vector_lease(candidate, conv, elements[0])?,
            prior_recurrent: vector_lease(prior, recurrent, elements[1])?,
            proposed_recurrent: vector_lease(candidate, recurrent, elements[1])?,
            attempt_phase,
        };
        let leases = value.leases();
        if leases.iter().enumerate().any(|(i, (lease, _))| {
            leases[..i]
                .iter()
                .any(|(other, _)| other.id() == lease.id())
        }) {
            return Err(Error::State(
                "GDN proposal aliases committed/sibling domain".into(),
            ));
        }
        Ok(value)
    }
}
impl BoundGdnDomains {
    pub(crate) fn is_attempt_local(&self) -> bool {
        self.attempt_phase.is_some()
    }
    pub(crate) fn require_prepared_prior(&self, lease: &ResidentLease) -> Result<()> {
        if let Some(binding) = &self.attempt_phase {
            return binding.require_prepared_prior(lease);
        }
        if !lease.is_initialized() {
            return Err(Error::State(
                "original root prior is not initialized".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn require_enqueue(
        &self,
        service: &Arc<ResidentService>,
        permit: Option<&PrefixPhaseEnqueue<'_>>,
    ) -> Result<()> {
        match (&self.attempt_phase, permit) {
            (Some(binding), Some(permit)) => {
                binding.require_permit(permit, service, &self.invocation, self.conv_domain()?.0)?;
                binding.require_launch_ready(service)?;
            }
            (None, None) => {}
            _ => {
                return Err(Error::State(
                    "matching original phase enqueue permit required".into(),
                ));
            }
        }
        if !self.prior_conv.is_initialized() || !self.prior_recurrent.is_initialized() {
            return Err(Error::State(
                "actual canonical GDN prior is not physically completed".into(),
            ));
        }
        Ok(())
    }
    pub(crate) fn invocation(&self) -> &SourceCall {
        &self.invocation
    }
    pub(crate) fn base(&self) -> RootBinding {
        self.base
    }
    pub(crate) fn conv_domain(&self) -> Result<ConvDomain> {
        match self.conv_effect.write.slot {
            StateSlot::Conv(domain) => Ok(domain),
            _ => Err(Error::State("GDN convolution state domain differs".into())),
        }
    }
    pub(crate) fn recurrent_domain(&self) -> Result<RecurrentDomain> {
        match self.recurrent_effect.write.slot {
            StateSlot::Recurrent(domain) => Ok(domain),
            _ => Err(Error::State("GDN recurrent state domain differs".into())),
        }
    }
    pub(crate) fn phase(&self) -> GdnPhase {
        self.phase
    }
    pub(crate) fn effect(&self, convolution: bool) -> StateEffect {
        if convolution {
            self.conv_effect
        } else {
            self.recurrent_effect
        }
    }
    pub(crate) fn leases(&self) -> [(ResidentLease, DeviceAccess); 4] {
        [
            (self.prior_conv.clone(), DeviceAccess::Read),
            (self.proposed_conv.clone(), DeviceAccess::Write),
            (self.prior_recurrent.clone(), DeviceAccess::Read),
            (self.proposed_recurrent.clone(), DeviceAccess::Write),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn existing_authority_attempts_bind_exact_five_phase_frontiers() {
        let source = zerogravity_core::accelerator::SourceIdentity::for_new_allocation().unwrap();
        for (base, phase) in [
            (0, GdnPhase::InitialOne),
            (1, GdnPhase::CachedDecode),
            (0, GdnPhase::Initial64),
            (64, GdnPhase::ContinuedTwo),
            (66, GdnPhase::CachedDecode),
        ] {
            let mut authority = SessionAuthority::new(source).unwrap();
            if base > 0 {
                let initial = authority.begin_advance(base).unwrap();
                authority.prepare_publish(initial).unwrap().commit();
            }
            let attempt = authority.begin_advance(phase.tokens() as u64).unwrap();
            assert!(validate_phase(authority.binding(), &attempt, phase).is_ok());
            for other_phase in [
                GdnPhase::InitialOne,
                GdnPhase::Initial64,
                GdnPhase::ContinuedTwo,
                GdnPhase::CachedDecode,
                GdnPhase::FixedPrompt,
            ] {
                let other_attempt = authority
                    .begin_advance(other_phase.tokens() as u64)
                    .unwrap();
                let expected = match other_phase {
                    GdnPhase::InitialOne | GdnPhase::Initial64 => base == 0,
                    GdnPhase::ContinuedTwo => base == 64,
                    GdnPhase::CachedDecode => matches!(base, 1 | 66),
                    GdnPhase::FixedPrompt | GdnPhase::TextPrefill(_) => false,
                };
                assert_eq!(
                    validate_phase(authority.binding(), &other_attempt, other_phase).is_ok(),
                    expected
                );
            }
            let other = SessionAuthority::new(source).unwrap();
            assert!(validate_phase(other.binding(), &attempt, phase).is_err());
            let wrong_count = authority.begin_advance(3).unwrap();
            assert!(validate_phase(authority.binding(), &wrong_count, phase).is_err());
            let replacement = authority.begin_validated_replacement(base).unwrap();
            authority.prepare_publish(replacement).unwrap().commit();
            assert_eq!(authority.binding().frontier(), base);
            assert!(
                validate_phase(authority.binding(), &attempt, phase).is_err(),
                "stale same-frontier epoch"
            );
            let fresh = authority.begin_advance(phase.tokens() as u64).unwrap();
            authority.prepare_publish(fresh).unwrap().commit();
            assert!(validate_phase(authority.binding(), &attempt, phase).is_err());
        }
    }
}
