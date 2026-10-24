//! Staged original session-domain borrow. No state allocation/publication.
use super::native_fragment::continuation::{AttemptPhaseBinding, PrefixPhaseEnqueue};
use super::*;
use crate::vllm_cuda_gdn::VllmGdnPhase;
use zerogravity_core::accelerator::program::{
    ConvDomain, GdnPrefixLayer, ProgramNumericalProfile, RecurrentDomain, StateEffect,
    VllmBf16Invocation, VllmBf16Phase,
};

/// Sealed original root + original prior/private candidate leases. Only the
/// existing ResidentSession can construct this binding; not an execution permit.
pub(crate) struct BoundGdnDomains {
    base: RootBinding,
    invocation: VllmBf16Invocation,
    phase: VllmGdnPhase,
    conv_effect: StateEffect,
    recurrent_effect: StateEffect,
    prior_conv: ResidentLease,
    proposed_conv: ResidentLease,
    prior_recurrent: ResidentLease,
    proposed_recurrent: ResidentLease,
    attempt_phase: Option<Arc<AttemptPhaseBinding>>,
}
const CONV_ELEMENTS: usize = 6144 * 3;
const RECURRENT_ELEMENTS: usize = 16 * 128 * 128;

fn validate_phase(current: RootBinding, attempt: &RootAttempt, phase: VllmGdnPhase) -> Result<()> {
    if current != attempt.base()
        || attempt.frontier().checked_sub(current.frontier()) != Some(phase.tokens() as u64)
        || !match phase {
            VllmGdnPhase::InitialOne | VllmGdnPhase::Initial64 => current.frontier() == 0,
            VllmGdnPhase::ContinuedTwo => current.frontier() == 64,
            VllmGdnPhase::CachedDecode => matches!(current.frontier(), 1 | 66),
            VllmGdnPhase::Initial47 | VllmGdnPhase::ShortHfPrefill(_) => false,
        }
    {
        return Err(Error::State(
            "original GDN root/attempt/phase frontier".into(),
        ));
    }
    Ok(())
}
fn vector_lease(domains: &[Domain], slot: StateSlot, elements: usize) -> Result<ResidentLease> {
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
        || lease.bytes() != elements * 4
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
    pub invocation: &'a VllmBf16Invocation,
    pub phase: VllmGdnPhase,
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
        invocation: &VllmBf16Invocation,
        phase: VllmGdnPhase,
        node_indices: [usize; 2],
    ) -> Result<Arc<BoundGdnDomains>> {
        invocation.validate()?;
        let base = self.binding();
        validate_phase(base, attempt, phase)?;
        if !matches!(phase, VllmGdnPhase::InitialOne | VllmGdnPhase::Initial64)
            || u64::try_from(invocation.teachers().base()).ok() != Some(base.frontier())
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
        request.invocation.validate()?;
        binding.require_binding(self, attempt, request.invocation)?;
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
        let layer = invocation.layer();
        let base = self.binding();
        let expected_phase = match phase {
            VllmGdnPhase::ShortHfPrefill(_) => {
                return Err(Error::Unsupported(
                    "short-HF requires genuine whole-model current-root binding".into(),
                ));
            }
            VllmGdnPhase::InitialOne => Some(VllmBf16Phase::InitialOne),
            VllmGdnPhase::CachedDecode => Some(VllmBf16Phase::CachedOne),
            VllmGdnPhase::Initial64 => Some(VllmBf16Phase::Initial64),
            VllmGdnPhase::ContinuedTwo => Some(VllmBf16Phase::ContinuedTwo),
            VllmGdnPhase::Initial47 => {
                attempt_phase
                    .as_ref()
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "47 component requires original attempt-bound state".into(),
                        )
                    })?
                    .require_initial_canonical47_component(invocation)?;
                None
            }
        };
        let phase_matches = match expected_phase {
            Some(expected) => invocation.legacy_phase()? == expected,
            None => true, // The exact attempt-bound47 scope was checked above.
        };
        if !phase_matches
            || invocation.rows() != phase.tokens()
            || invocation.teachers().source() != base.source()
            || invocation.teachers().capacity() != self.model.capacity
        {
            return Err(Error::State(
                "same original GDN teacher invocation/root span".into(),
            ));
        }
        if !matches!(layer.index(), 0..=2)
            || layer.source() != base.source()
            || plan.source() != base.source()
            || base.source() != self.model.service.source_identity()
            || plan.numerical_profile() != ProgramNumericalProfile::Vllm031CudaSm86Bf16SourceV1
            || plan.is_operator_slice()
            || plan.frontier_delta() != 0
            || !plan.vllm_gdn_fragment().is_some_and(|base| {
                base.teachers() == invocation.teachers()
                    && base.mode() == invocation.mode()
                    && base.layer().source() == invocation.layer().source()
            })
        {
            return Err(Error::Unsupported(
                "same full CUDA GDN prefix plan/root source".into(),
            ));
        }
        self.model.service.ensure_usable()?;
        self.model.validate_domains(&self.domains, true)?;
        self.model.validate_domains(candidate, false)?;
        let conv = StateSlot::Conv(ConvDomain(layer.index()));
        let recurrent = StateSlot::Recurrent(RecurrentDomain(layer.index()));
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
                let tag = plan.vllm_value_provenance(id)?.ok_or_else(|| {
                    Error::State("GDN value lost original teacher provenance".into())
                })?;
                if expected_phase.is_some_and(|expected| tag.phase() != Some(expected)) {
                    return Err(Error::State(
                        "GDN value phase differs from original root invocation".into(),
                    ));
                }
                tag.require_invocation(invocation)?;
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
            attempt_phase,
        )?;
        Ok(Arc::new(bound))
    }
    fn gdn_domain_binding(
        base: RootBinding,
        invocation: VllmBf16Invocation,
        phase: VllmGdnPhase,
        prior: &[Domain],
        candidate: &[Domain],
        effects: [StateEffect; 2],
        attempt_phase: Option<Arc<AttemptPhaseBinding>>,
    ) -> Result<BoundGdnDomains> {
        let layer = invocation.layer();
        let conv = StateSlot::Conv(ConvDomain(layer.index()));
        let recurrent = StateSlot::Recurrent(RecurrentDomain(layer.index()));
        let value = BoundGdnDomains {
            base,
            invocation,
            phase,
            conv_effect: effects[0],
            recurrent_effect: effects[1],
            prior_conv: vector_lease(prior, conv, CONV_ELEMENTS)?,
            proposed_conv: vector_lease(candidate, conv, CONV_ELEMENTS)?,
            prior_recurrent: vector_lease(prior, recurrent, RECURRENT_ELEMENTS)?,
            proposed_recurrent: vector_lease(candidate, recurrent, RECURRENT_ELEMENTS)?,
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
                binding.require_permit(permit, service, &self.invocation)?;
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
    pub(crate) fn invocation(&self) -> &VllmBf16Invocation {
        &self.invocation
    }
    pub(crate) fn base(&self) -> RootBinding {
        self.base
    }
    pub(crate) fn layer(&self) -> GdnPrefixLayer {
        self.invocation.layer()
    }
    pub(crate) fn phase(&self) -> VllmGdnPhase {
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
            (0, VllmGdnPhase::InitialOne),
            (1, VllmGdnPhase::CachedDecode),
            (0, VllmGdnPhase::Initial64),
            (64, VllmGdnPhase::ContinuedTwo),
            (66, VllmGdnPhase::CachedDecode),
        ] {
            let mut authority = SessionAuthority::new(source).unwrap();
            if base > 0 {
                let initial = authority.begin_advance(base).unwrap();
                authority.prepare_publish(initial).unwrap().commit();
            }
            let attempt = authority.begin_advance(phase.tokens() as u64).unwrap();
            assert!(validate_phase(authority.binding(), &attempt, phase).is_ok());
            for other_phase in [
                VllmGdnPhase::InitialOne,
                VllmGdnPhase::Initial64,
                VllmGdnPhase::ContinuedTwo,
                VllmGdnPhase::CachedDecode,
                VllmGdnPhase::Initial47,
            ] {
                let other_attempt = authority
                    .begin_advance(other_phase.tokens() as u64)
                    .unwrap();
                let expected = match other_phase {
                    VllmGdnPhase::InitialOne | VllmGdnPhase::Initial64 => base == 0,
                    VllmGdnPhase::ContinuedTwo => base == 64,
                    VllmGdnPhase::CachedDecode => matches!(base, 1 | 66),
                    VllmGdnPhase::Initial47 | VllmGdnPhase::ShortHfPrefill(_) => false,
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
