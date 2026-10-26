//! Exact dedicated-per-run RMS views. Source gamma never enters this storage.
use super::*;
use crate::resident_leases::backing_pool::ResidentWorkspaceGeometry;
use crate::resident_program::BufferRange;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum RmsWorkspaceOutput {
    Embedding,
    Sum,
    Normalized,
    Residual,
}

/// Only source-bound cold preparation creates this map. It carries the actual
/// charged geometry owner; no larger lease, epoch flag or fake Handle size.
pub(crate) struct RmsWorkspaceViews {
    scratch: Box<[ResidentLease]>,
    outputs: BTreeMap<(usize, RmsWorkspaceOutput), ResidentLease>,
    _owner: Arc<ResidentWorkspaceGeometry>,
}
impl RmsWorkspaceViews {
    pub(crate) fn bind(
        record: &Arc<SourceBindings>,
        owner: Arc<ResidentWorkspaceGeometry>,
        scratch: Vec<ResidentLease>,
        outputs: BTreeMap<(usize, RmsWorkspaceOutput), ResidentLease>,
    ) -> Result<Arc<Self>> {
        let result = Arc::new(Self {
            scratch: scratch.into_boxed_slice(),
            outputs,
            _owner: owner,
        });
        let service = result
            .scratch
            .first()
            .ok_or_else(|| Error::State("RMS shared scratch missing".into()))?
            .service();
        result.require(record, service)?;
        Ok(result)
    }
    pub(super) fn require(
        &self,
        record: &Arc<SourceBindings>,
        service: &Arc<ResidentService>,
    ) -> Result<()> {
        let actual = bindings(record)?;
        let rows = record.call().rows().len();
        let width = actual
            .get(&1)
            .ok_or_else(|| Error::State("ordinary RMS absent".into()))?
            .geometry()
            .width;
        let row_bf16 = rows
            .checked_mul(width)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| Error::Limit("RMS shared row extent".into()))?;
        let private = FullModelRmsScratch::allocation_layout(rows, width)?;
        if self.scratch.len() != private.len() || private.len() != 9 || self.outputs.len() != 146 {
            return Err(Error::State(
                "RMS shared nine scratch/146 observed producers".into(),
            ));
        }
        let mut required = BTreeMap::from([
            (
                (1, RmsWorkspaceOutput::Embedding),
                (row_bf16, BufferRole::Activation),
            ),
            (
                (1, RmsWorkspaceOutput::Normalized),
                (row_bf16, BufferRole::Output),
            ),
        ]);
        for &node in actual.keys().filter(|&&node| node != 1) {
            required.insert(
                (node, RmsWorkspaceOutput::Sum),
                (row_bf16 * 2, BufferRole::Scratch),
            );
            required.insert(
                (node, RmsWorkspaceOutput::Normalized),
                (row_bf16, BufferRole::Output),
            );
            required.insert(
                (node, RmsWorkspaceOutput::Residual),
                (row_bf16, BufferRole::Output),
            );
        }
        let mut ids = BTreeSet::new();
        let mut ranges = Vec::with_capacity(155);
        let all = self
            .scratch
            .iter()
            .zip(private)
            .map(|(l, n)| (l, n, BufferRole::Scratch))
            .chain(
                required
                    .iter()
                    .map(|(key, &(n, role))| (self.outputs.get(key), n, role))
                    .map(|(l, n, role)| {
                        l.map(|l| (l, n, role))
                            .ok_or_else(|| Error::State("RMS shared producer role omitted".into()))
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        for (lease, bytes, role) in all {
            if !ids.insert(lease.id())
                || lease.bytes() != bytes
                || lease.role() != role
                || !Arc::ptr_eq(lease.service(), service)
                || lease.service_generation() != service.generation()
                || lease.source_identity() != record.call().rows().source()
                || !self._owner.leases().iter().any(|original| {
                    original.id() == lease.id()
                        && original.allocation_generation() == lease.allocation_generation()
                })
            {
                return Err(Error::State(
                    "RMS exact shared view source/service/role/owner differs".into(),
                ));
            }
            let range = BufferRange::new(&lease.handle())?;
            range.require_exact_len(bytes)?;
            if ranges
                .iter()
                .any(|prior: &BufferRange| prior.overlaps(&range))
            {
                return Err(Error::State("RMS actual actor view ranges overlap".into()));
            }
            ranges.push(range);
        }
        Ok(())
    }
    pub(super) fn scratch(&self, index: usize, bytes: usize) -> Result<ResidentLease> {
        self.scratch
            .get(index)
            .filter(|l| l.bytes() == bytes && l.role() == BufferRole::Scratch)
            .cloned()
            .ok_or_else(|| Error::State("RMS exact shared private role/extent".into()))
    }
    pub(super) fn output(
        &self,
        node: usize,
        role: RmsWorkspaceOutput,
        bytes: usize,
    ) -> Result<ResidentLease> {
        self.outputs
            .get(&(node, role))
            .filter(|l| l.bytes() == bytes)
            .cloned()
            .ok_or_else(|| Error::State("RMS exact shared public/sum producer extent".into()))
    }
}
