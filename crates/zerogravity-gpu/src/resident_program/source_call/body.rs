//! Bind the original resident owner; no additional scheduler or root publisher.
use super::super::{
    BufferRange, OwnedRanges, ResidentProgram, ResidentProgramBody, ResidentProgramSource,
};
use super::{cold::Cold, *};
use zerogravity_core::accelerator::program::{SourceCall, SourceCallMode};

impl Cold {
    pub(super) fn bind(self: &Arc<Self>) -> Result<ResidentProgram> {
        self.service.ensure_usable()?;
        self.native
            .require_live(self.service.client(), self.source.source_identity())?;
        let mut owned_ranges = OwnedRanges::new(self.service.client().service_id());
        let mut ranges = Vec::new();
        let mut ids = std::collections::BTreeSet::new();
        for (lease, _) in &self.accesses {
            if !Arc::ptr_eq(lease.service(), &self.service)
                || lease.source_identity() != self.layout.plan.source()
                || lease.service_generation() != self.service.generation()
                || !ids.insert(lease.id())
            {
                return Err(Error::State(
                    "source-call original owned lease provenance".into(),
                ));
            }
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(lease.bytes())?;
            if ranges.iter().any(|r: &BufferRange| r.overlaps(&range)) {
                return Err(Error::State("source-call distinct physical owners".into()));
            }
            owned_ranges.insert(range.clone())?;
            ranges.push(range);
        }
        if layout::sum(self.accesses.iter().map(|(l, _)| l.bytes()))?
            != self.layout.preflight.owned_bytes()?
        {
            return Err(Error::State(
                "source-call bound allocation accounting".into(),
            ));
        }
        Ok(ResidentProgram {
            _source_owner: ResidentProgramSource::Windows(self.source.clone()),
            plan: self.layout.plan.clone(),
            preflight: self.layout.preflight.clone(),
            values: self.values.clone(),
            status: self.status.handle(),
            owned_ranges,
            body: ResidentProgramBody::SourceCall(self.clone()),
        })
    }
    pub(in crate::resident_program) fn validate_request<'a>(
        &self,
        request: &'a ForwardPlan,
    ) -> Result<&'a SourceCall> {
        let cold = &self.layout.plan;
        if !request.shares_semantics(cold)
            || request.nodes().len() != cold.nodes().len()
            || request.values().len() != cold.values().len()
            || request.required_domains() != cold.required_domains()
            || request.proposed_domains() != cold.proposed_domains()
        {
            return Err(Error::State(
                "source-call original cold plan issuer differs".into(),
            ));
        }
        for (actual, template) in request.nodes().iter().zip(cold.nodes()) {
            if !actual.shares_topology(template)
                || !matches!(actual.operation, PlanOperation::SourceOperator { .. })
            {
                return Err(Error::State(
                    "source-call materialized original topology differs".into(),
                ));
            }
        }
        let call = request
            .source_call()
            .ok_or_else(|| Error::State("source-call actual invocation absent".into()))?;
        if call.source() != cold.source()
            || call.rows().capacity() != cold.prefix_capacity()
            || call.rows().vocabulary() != self.layout.vocabulary
            || !(1..=layout::MAX_ROWS).contains(&call.rows().len())
            || call.rows().frontier() > cold.prefix_capacity()
            || (call.mode() == SourceCallMode::Decode)
                != (call.rows().base() > 0 && call.rows().len() == 1)
        {
            return Err(Error::State(
                "source-call actual source/rows/frontier/phase differs".into(),
            ));
        }
        Ok(call)
    }
}
