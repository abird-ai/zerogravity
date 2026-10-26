//! Host-only Weak inspection; no enqueue, completion or release authority.
use super::*;
use std::{collections::BTreeSet, sync::Weak};
use zerogravity_core::{accelerator::SourceIdentity, device_events::BufferLeaseId};

pub(crate) struct FusedRmsRetentionWitness {
    prepared: Weak<FusedRmsPrepared>,
    owner: Weak<FusedRmsSource>,
    service: Weak<ResidentService>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FusedRmsRetentionState {
    pub(crate) source: SourceIdentity,
    pub(crate) rows: usize,
    pub(crate) input_ids: [BufferLeaseId; 2],
    pub(crate) output_ids: [BufferLeaseId; 3],
    pub(crate) output_initialized: [bool; 3],
    pub(crate) readback_ids: [BufferLeaseId; 4],
    pub(crate) readback_bytes: [usize; 4],
    pub(crate) host_workspace_bytes: usize,
}

impl FusedRmsPrepared {
    pub(crate) fn retention_witness(self: &Arc<Self>) -> FusedRmsRetentionWitness {
        FusedRmsRetentionWitness {
            prepared: Arc::downgrade(self),
            owner: Arc::downgrade(&self.owner),
            service: Arc::downgrade(&self.service),
        }
    }
}

impl FusedRmsRetentionWitness {
    pub(crate) fn inspect(&self) -> Result<FusedRmsRetentionState> {
        let prepared = self
            .prepared
            .upgrade()
            .ok_or_else(|| Error::State("quarantine lost original fused prepared".into()))?;
        let owner = self
            .owner
            .upgrade()
            .ok_or_else(|| Error::State("quarantine lost original fused source owner".into()))?;
        let service = self
            .service
            .upgrade()
            .ok_or_else(|| Error::State("quarantine lost original fused service".into()))?;
        if !Arc::ptr_eq(&prepared.owner, &owner)
            || !Arc::ptr_eq(&prepared.service, &service)
            || owner.source.source_identity() != service.source_identity()
            || !owner.source.same_issuer(owner._workspace.source())
            || !owner.source.same_issuer(owner._metadata.source())
            || !owner._parameter.is_complete()
            || owner._parameter.parameter().source_identity() != service.source_identity()
            || owner._parameter.parameter().tensor().name != GAMMA
            || owner._parameter.parameter().tensor().shape != [COLUMNS as u64]
            || owner.gamma.len() != COLUMNS * 2
            || owner.specs.len() != Stage::ORDER.len()
            || prepared.stages.len() != Stage::ORDER.len()
            || prepared.leases.len() != 14
            || prepared.accesses.len() != 14
        {
            return Err(Error::State(
                "retained fused source/charge/geometry changed".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for (role, lease) in prepared.leases.iter() {
            if !ids.insert(lease.id())
                || !Arc::ptr_eq(lease.service(), &service)
                || lease.source_identity() != service.source_identity()
                || !prepared.accesses.iter().any(|(accessed, access)| {
                    accessed.id() == lease.id()
                        && *access
                            == if role.read_only() {
                                DeviceAccess::Read
                            } else {
                                DeviceAccess::Write
                            }
                })
            {
                return Err(Error::State(
                    "retained fused original lease/access changed".into(),
                ));
            }
        }
        for (role, bytes) in Buffer::owned(owner.rows, Columns::Qwen08)
            .into_iter()
            .chain([
                (Buffer::Attention, owner.rows.count() * COLUMNS * 2),
                (Buffer::PriorResidual, owner.rows.count() * COLUMNS * 2),
            ])
        {
            if prepared.lease(role).bytes() != bytes {
                return Err(Error::State(
                    "retained fused original physical extent changed".into(),
                ));
            }
        }
        let readback_roles = [
            Buffer::GammaRaw,
            Buffer::SumF32,
            Buffer::NormalizedBf16,
            Buffer::ResidualBf16,
        ];
        Ok(FusedRmsRetentionState {
            source: owner.source.source_identity(),
            rows: owner.rows.count(),
            input_ids: [Buffer::Attention, Buffer::PriorResidual]
                .map(|role| prepared.lease(role).id()),
            output_ids: [Buffer::SumF32, Buffer::NormalizedBf16, Buffer::ResidualBf16]
                .map(|role| prepared.lease(role).id()),
            output_initialized: [Buffer::SumF32, Buffer::NormalizedBf16, Buffer::ResidualBf16]
                .map(|role| prepared.lease(role).is_initialized()),
            readback_ids: readback_roles.map(|role| prepared.lease(role).id()),
            readback_bytes: readback_roles.map(|role| prepared.lease(role).bytes()),
            host_workspace_bytes: owner._workspace.bytes() + owner._metadata.bytes(),
        })
    }
}
