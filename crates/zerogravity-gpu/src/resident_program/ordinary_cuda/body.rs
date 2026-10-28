//! Bound ordinary CUDA ResidentProgram body.
//! Request-local execution is added separately; this binding only joins the
//! cold source owner with the one opaque BLAS owner under the existing program.

use super::{OrdinaryCudaBlas, cold::OrdinaryCudaCold};
use crate::{
    resident_leases::ResidentLease,
    resident_program::{
        BufferRange, OwnedRanges, ResidentProgram, ResidentProgramBody, ResidentProgramSource,
    },
};
use std::{collections::BTreeMap, sync::Arc};
use zerogravity_core::{
    Error, Result,
    device_events::{BufferLeaseId, DeviceAccess},
};

pub(in crate::resident_program) struct OrdinaryCudaBody {
    cold: Arc<OrdinaryCudaCold>,
    blas: Arc<OrdinaryCudaBlas>,
    owned_accesses: Arc<[(ResidentLease, DeviceAccess)]>,
    state_accesses: Arc<[(ResidentLease, DeviceAccess)]>,
    logit_accesses: Arc<[(ResidentLease, DeviceAccess)]>,
}
impl OrdinaryCudaBody {
    pub(super) fn bind(
        cold: Arc<OrdinaryCudaCold>,
        blas: Arc<OrdinaryCudaBlas>,
    ) -> Result<(ResidentProgram, Arc<Self>)> {
        cold.service().ensure_usable()?;
        blas.require_service(cold.service())?;
        if cold.layout().total_admission_bytes()?
            != cold
                .layout()
                .preflight()
                .total_with_state()?
                .checked_add(cold.layout().foreign_provider_bytes())
                .and_then(|n| n.checked_add(cold.layout().blas_bytes()))
                .ok_or_else(|| Error::Limit("ordinary CUDA bound admission sum".into()))?
        {
            return Err(Error::State(
                "ordinary CUDA cold/resource admission identity differs".into(),
            ));
        }

        let mut unique = BTreeMap::<BufferLeaseId, (ResidentLease, DeviceAccess)>::new();
        for (lease, access) in cold.accesses().into_iter().chain(blas.accesses()) {
            if !Arc::ptr_eq(lease.service(), cold.service())
                || lease.source_identity() != cold.plan().source()
                || lease.service_generation() != cold.service().generation()
            {
                return Err(Error::State(
                    "ordinary CUDA bound lease service/source/generation differs".into(),
                ));
            }
            unique
                .entry(lease.id())
                .and_modify(|(_, prior)| {
                    if access == DeviceAccess::Write {
                        *prior = DeviceAccess::Write;
                    }
                })
                .or_insert((lease, access));
        }
        let accesses = unique.into_values().collect::<Vec<_>>();
        let mut owned_ranges = OwnedRanges::new(cold.service().client().service_id());
        let mut ranges = Vec::with_capacity(accesses.len());
        for (lease, _) in &accesses {
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(lease.bytes())?;
            if ranges
                .iter()
                .any(|prior: &BufferRange| prior.overlaps(&range))
            {
                return Err(Error::State(
                    "ordinary CUDA bound physical allocations overlap".into(),
                ));
            }
            owned_ranges.insert(range.clone())?;
            ranges.push(range);
        }

        let mut preflight = cold.layout().preflight().clone();
        preflight.workspace_bytes = preflight
            .workspace_bytes
            .checked_add(cold.layout().blas_bytes())
            .ok_or_else(|| Error::Limit("ordinary CUDA BLAS resident preflight".into()))?;
        let expected_owned =
            accesses
                .iter()
                .map(|(lease, _)| lease.bytes())
                .try_fold(0usize, |sum, bytes| {
                    sum.checked_add(bytes)
                        .ok_or_else(|| Error::Limit("ordinary CUDA bound lease byte sum".into()))
                })?;
        if preflight.owned_bytes()? != expected_owned {
            return Err(Error::State(format!(
                "ordinary CUDA bound managed bytes differ: preflight={} actual={expected_owned}",
                preflight.owned_bytes()?
            )));
        }

        let logit_accesses = cold
            .accesses()
            .into_iter()
            .chain(blas.execution_accesses())
            .collect::<Vec<_>>();
        let logits_id = cold.logits_f32().id();
        let state_accesses = logit_accesses
            .iter()
            .filter(|(lease, _)| lease.id() != logits_id)
            .cloned()
            .collect::<Vec<_>>();
        let body = Arc::new(Self {
            cold: cold.clone(),
            blas,
            owned_accesses: accesses.into(),
            state_accesses: state_accesses.into(),
            logit_accesses: logit_accesses.into(),
        });
        let program = ResidentProgram {
            _source_owner: ResidentProgramSource::Windows(cold.source().clone()),
            plan: cold.plan().clone(),
            preflight,
            values: cold.values().to_vec(),
            status: cold.status().handle(),
            owned_ranges,
            body: ResidentProgramBody::OrdinaryCuda(body.clone()),
        };
        Ok((program, body))
    }

    pub(super) fn cold(&self) -> &Arc<OrdinaryCudaCold> {
        &self.cold
    }
    pub(super) fn blas(&self) -> &Arc<OrdinaryCudaBlas> {
        &self.blas
    }
    pub(in crate::resident_program) fn owned_accesses(&self) -> &[(ResidentLease, DeviceAccess)] {
        &self.owned_accesses
    }

    pub(in crate::resident_program) fn accesses(
        &self,
        output: crate::resident_program::ResidentOutput,
    ) -> &[(ResidentLease, DeviceAccess)] {
        match output {
            crate::resident_program::ResidentOutput::StateOnly => &self.state_accesses,
            crate::resident_program::ResidentOutput::Logits => &self.logit_accesses,
        }
    }
}

impl ResidentProgram {
    pub(crate) fn is_ordinary_cuda(&self) -> bool {
        matches!(self.body, ResidentProgramBody::OrdinaryCuda(_))
    }

    pub(crate) fn launch_ordinary_cuda(
        &self,
        flight: &mut crate::resident_leases::ResidentSubmission<'_>,
        request: &zerogravity_core::accelerator::program::ForwardPlan,
        states: &[crate::resident_program::StateBinding],
        output: crate::resident_program::ResidentOutput,
    ) -> Result<crate::resident_program::ResidentSubmission> {
        let ResidentProgramBody::OrdinaryCuda(body) = &self.body else {
            return Err(Error::State(
                "ordinary CUDA launch requested from another resident body".into(),
            ));
        };
        body.execute(flight, request, states, output)
    }
}
