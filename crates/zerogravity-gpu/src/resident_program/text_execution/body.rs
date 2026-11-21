//! Bind the original resident owner; no additional scheduler or root publisher.
use super::super::{
    BufferRange, OwnedRanges, ResidentProgram, ResidentProgramBody, ResidentProgramSource,
};
use super::{cold::Cold, *};
use zerogravity_core::accelerator::program::{InvocationPhase, TextInvocation};

impl Cold {
    pub(super) fn bind(self: &Arc<Self>) -> Result<ResidentProgram> {
        self.service.ensure_usable()?;
        self.native
            .require_live(self.service.client(), self.source.source_identity())?;
        let physical = self
            .accesses
            .iter()
            .map(|(lease, _)| lease.clone())
            .collect::<Vec<_>>();
        if !self.service.pairwise_disjoint(&physical)? {
            return Err(Error::State(
                "text-execution distinct physical owners".into(),
            ));
        }
        let mut owned_ranges = OwnedRanges::new(self.service.client().service_id());
        for (lease, _) in &self.accesses {
            if lease.source_identity() != self.layout.plan.source() {
                return Err(Error::State(
                    "text-execution original owned lease provenance".into(),
                ));
            }
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(lease.bytes())?;
            owned_ranges.insert(range)?;
        }
        if layout::sum(self.accesses.iter().map(|(l, _)| l.bytes()))?
            != self.layout.preflight.owned_bytes()?
        {
            return Err(Error::State(
                "text-execution bound allocation accounting".into(),
            ));
        }
        Ok(ResidentProgram {
            _source_owner: ResidentProgramSource::Windows(self.source.clone()),
            plan: self.layout.plan.clone(),
            preflight: self.layout.preflight.clone(),
            values: self.values.clone(),
            status: self.status.handle(),
            owned_ranges,
            body: ResidentProgramBody::TextExecution(self.clone()),
        })
    }
    pub(in crate::resident_program) fn validate_request<'a>(
        &self,
        request: &'a ForwardPlan,
    ) -> Result<&'a TextInvocation> {
        let cold = &self.layout.plan;
        if !request.shares_semantics(cold)
            || request.nodes().len() != cold.nodes().len()
            || request.values().len() != cold.values().len()
            || request.required_domains() != cold.required_domains()
            || request.proposed_domains() != cold.proposed_domains()
        {
            return Err(Error::State(
                "text-execution original cold plan issuer differs".into(),
            ));
        }
        for (actual, template) in request.nodes().iter().zip(cold.nodes()) {
            if !actual.shares_topology(template)
                || !matches!(actual.operation, PlanOperation::SourceOperator { .. })
            {
                return Err(Error::State(
                    "text-execution materialized original topology differs".into(),
                ));
            }
        }
        let call = request
            .text_invocation()
            .ok_or_else(|| Error::State("text-execution actual invocation absent".into()))?;
        if call.source() != cold.source()
            || call.tokens().capacity() != cold.prefix_capacity()
            || call.tokens().vocabulary() != self.layout.vocabulary
            || !(1..=layout::MAX_ROWS).contains(&call.tokens().len())
            || call.tokens().frontier() > cold.prefix_capacity()
            || (call.phase() == InvocationPhase::Decode)
                != (call.tokens().base() > 0 && call.tokens().len() == 1)
        {
            return Err(Error::State(
                "text-execution actual source/rows/frontier/phase differs".into(),
            ));
        }
        Ok(call)
    }
}
