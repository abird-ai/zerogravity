//! Captured leaf metadata only; no model, request or StateRoot authority.
use super::manifests::{self, AttentionCase, AttentionRole, AttentionStage};
use crate::physical_quarantine::CompletionOwner;
use crate::resident_leases::{ResidentLease, ResidentService};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
use zerogravity_core::device_events::{BufferLeaseId, DeviceAccess};
use zerogravity_core::{Error, Result, accelerator::SourceIdentity};

pub(super) const KV_BYTES: usize = 5 * 544 * 1024 * 2;
pub(super) const STAGES: [AttentionStage; 5] = [
    AttentionStage::Qkv,
    AttentionStage::Qk,
    AttentionStage::Kv,
    AttentionStage::Fa,
    AttentionStage::Out,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionScope {
    CapturedOutInput,
    ComputedGate,
}
impl AttentionScope {
    pub(super) fn stages(self) -> &'static [AttentionStage] {
        match self {
            Self::CapturedOutInput => &STAGES,
            Self::ComputedGate => &[
                AttentionStage::Qkv,
                AttentionStage::Qk,
                AttentionStage::Kv,
                AttentionStage::Fa,
                AttentionStage::Sigmoid,
                AttentionStage::Multiply,
                AttentionStage::Out,
            ],
        }
    }
    pub(super) fn records(self) -> usize {
        match self {
            Self::CapturedOutInput => 20,
            Self::ComputedGate => 26,
        }
    }
    pub(super) fn images(self) -> usize {
        match self {
            Self::CapturedOutInput => 9,
            Self::ComputedGate => 11,
        }
    }
}

#[derive(Clone)]
pub(super) struct AttentionDescriptor {
    source: SourceIdentity,
    case: AttentionCase,
    scope: AttentionScope,
}
impl AttentionDescriptor {
    pub(super) fn new(source: SourceIdentity, case: AttentionCase) -> Result<Self> {
        if !source.is_bound() {
            return Err(Error::State(
                "attention leaf immutable source absent".into(),
            ));
        }
        Ok(Self {
            source,
            case,
            scope: AttentionScope::CapturedOutInput,
        })
    }
    pub(super) fn with_scope(
        source: SourceIdentity,
        case: AttentionCase,
        scope: AttentionScope,
    ) -> Result<Self> {
        let mut descriptor = Self::new(source, case)?;
        descriptor.scope = scope;
        Ok(descriptor)
    }
    pub(super) fn scope(&self) -> AttentionScope {
        self.scope
    }
    pub(super) fn source(&self) -> SourceIdentity {
        self.source
    }
    pub(super) fn case(&self) -> AttentionCase {
        self.case
    }
    /// Original observed control metadata; no arithmetic output or heap snapshot.
    /// All bytes, including pitch gaps, are initialized before original H2D.
    pub(super) fn controls(&self) -> AttentionControls {
        let rows = self.case.rows();
        let base = self.case.base();
        let mut positions = [0i64; 195];
        let mut slots = [0i64; 64];
        for i in 0..rows {
            for axis in 0..3 {
                positions[axis * 65 + i] = (base + i) as i64;
            }
            slots[i] = (self.case.block() * 544 + base + i) as i64;
        }
        AttentionControls {
            positions,
            slots,
            slot_count: rows,
            cu_q: [0, rows as i32],
            seq_used: (base + rows) as i32,
            block_table: [self.case.block() as i32, 0, 0, 0],
            dummy_cu_k: [0, 0],
        }
    }
    pub(super) fn required_roles(&self) -> Result<BTreeMap<AttentionRole, (usize, DeviceAccess)>> {
        let mut roles = BTreeMap::new();
        for &stage in self.scope.stages() {
            for launch in manifests::launches(self.case, stage) {
                for (&role, binding) in launch.roles.iter().zip(&launch.manifest.bindings) {
                    if role == AttentionRole::OriginalKvState && binding.bytes != KV_BYTES as u64 {
                        return Err(Error::State(
                            "attention exact five-page original cache".into(),
                        ));
                    }
                    let access = if matches!(binding.io, cubecl::cuda::CubinIo::Read) {
                        DeviceAccess::Read
                    } else {
                        DeviceAccess::Write
                    };
                    match roles.get_mut(&role) {
                        Some((bytes, old)) => {
                            if *bytes != binding.bytes as usize {
                                return Err(Error::State(
                                    "attention role byte extent changes".into(),
                                ));
                            }
                            if access == DeviceAccess::Write {
                                *old = access;
                            }
                        }
                        None => {
                            roles.insert(role, (binding.bytes as usize, access));
                        }
                    }
                }
            }
        }
        if self.case.rows() == 1 {
            roles.insert(AttentionRole::AttentionOutput, (4096, DeviceAccess::Write));
        }
        Ok(roles)
    }
}

pub(super) struct AttentionControls {
    pub(super) positions: [i64; 195],
    pub(super) slots: [i64; 64],
    pub(super) slot_count: usize,
    pub(super) cu_q: [i32; 2],
    pub(super) seq_used: i32,
    pub(super) block_table: [i32; 4],
    // Source's nonnull dummy values are unused under nonnull table/seqUsed and
    // AppendKVfalse. Own initialized bytes; never import uninitialized heap.
    pub(super) dummy_cu_k: [i32; 2],
}

/// Exact original allocations and controls. CapturedOutInput retains the
/// explicit authentic leaf operand. ComputedGate uses fresh Sigmoid/OutInput
/// outputs, produced by two distinct native kernels in the SAME flight.
pub(super) struct AttentionBindings {
    descriptor: AttentionDescriptor,
    leases: BTreeMap<AttentionRole, ResidentLease>,
    service: Arc<ResidentService>,
    _source_owner: CompletionOwner,
}
impl AttentionBindings {
    /// # Safety
    /// Caller retains original source/window owners and uploads the descriptor's
    /// exact controls, source parameters/cosSin and authentic leaf operands in
    /// SAME init. Original KV storage is initialized; cached cases retain the
    /// actual prior cache or an explicitly labelled authentic leaf cache input.
    /// This seam never authorizes a model profile or publishes a StateRoot.
    pub(super) unsafe fn new(
        descriptor: AttentionDescriptor,
        service: Arc<ResidentService>,
        source_owner: CompletionOwner,
        leases: BTreeMap<AttentionRole, ResidentLease>,
    ) -> Result<Arc<Self>> {
        service.ensure_usable()?;
        if descriptor.source() != service.source_identity() {
            return Err(Error::State("attention original source/service".into()));
        }
        let expected = descriptor.required_roles()?;
        if leases.len() != expected.len() {
            return Err(Error::Malformed(
                "attention exact original lease roles".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for (role, (bytes, _)) in expected {
            let lease = leases
                .get(&role)
                .ok_or_else(|| Error::Malformed("attention original role absent".into()))?;
            if !Arc::ptr_eq(lease.service(), &service)
                || lease.source_identity() != descriptor.source()
                || lease.bytes() != bytes
                || !seen.insert(lease.id())
            {
                return Err(Error::State(
                    "attention distinct original source/service/extent".into(),
                ));
            }
        }
        Ok(Arc::new(Self {
            descriptor,
            leases,
            service,
            _source_owner: source_owner,
        }))
    }
    pub(super) fn descriptor(&self) -> &AttentionDescriptor {
        &self.descriptor
    }
    pub(super) fn service(&self) -> &Arc<ResidentService> {
        &self.service
    }
    pub(super) fn lease(&self, role: AttentionRole) -> Result<&ResidentLease> {
        self.leases
            .get(&role)
            .ok_or_else(|| Error::State("attention unbound original role".into()))
    }
    pub(super) fn accesses(&self) -> Result<Vec<(ResidentLease, DeviceAccess)>> {
        self.descriptor
            .required_roles()?
            .into_iter()
            .map(|(role, (_, access))| Ok((self.lease(role)?.clone(), access)))
            .collect()
    }
    pub(super) fn cohort(&self) -> Vec<BufferLeaseId> {
        self.leases.values().map(ResidentLease::id).collect()
    }
}
