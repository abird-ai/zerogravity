//! Charged physical storage and independently retired exact logical views.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceBackingId(u64);

pub(super) struct BackingAllocation {
    bytes: usize,
    role: BufferRole,
    pub(super) views: BTreeSet<BufferLeaseId>,
    pub(super) fenced: bool,
}
#[derive(Clone, Copy)]
pub(super) struct BackingRange {
    pub(super) id: DeviceBackingId,
    offset: usize,
}

impl DeviceEventLedger {
    /// Reserve once, before the backend allocator. Idle storage stays charged.
    pub fn admit_backing(&mut self, bytes: usize, role: BufferRole) -> Result<DeviceBackingId> {
        if bytes == 0 || bytes > self.budget_bytes.saturating_sub(self.resident_bytes) {
            return Err(Error::Limit("device backing byte budget exceeded".into()));
        }
        let next = self
            .next_backing
            .checked_add(1)
            .ok_or_else(|| Error::Limit("device backing sequence exhausted".into()))?;
        let id = DeviceBackingId(self.next_backing);
        self.next_backing = next;
        self.backings.insert(
            id,
            BackingAllocation {
                bytes,
                role,
                views: BTreeSet::new(),
                fenced: false,
            },
        );
        self.resident_bytes += bytes;
        Ok(id)
    }

    /// Fresh logical identity; the same physical charge already covers its bytes.
    /// Range identity is control metadata, never a backend handle or source proof.
    pub fn admit_backing_view(
        &mut self,
        backing: DeviceBackingId,
        offset: usize,
        bytes: usize,
        role: BufferRole,
    ) -> Result<BufferLeaseId> {
        let storage = self
            .backings
            .get(&backing)
            .ok_or_else(|| Error::State("unknown device backing".into()))?;
        if storage.fenced
            || storage.role != role
            || bytes == 0
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > storage.bytes)
        {
            return Err(Error::State("device backing view range/role/fence".into()));
        }
        let next = self
            .next_lease
            .checked_add(1)
            .ok_or_else(|| Error::Limit("device lease sequence exhausted".into()))?;
        let id = BufferLeaseId(self.next_lease);
        self.next_lease = next;
        self.leases.insert(
            id,
            Lease {
                bytes,
                role,
                readers: 0,
                writer: false,
                fenced: false,
                backing: Some(BackingRange {
                    id: backing,
                    offset,
                }),
            },
        );
        self.backings
            .get_mut(&backing)
            .expect("observed backing")
            .views
            .insert(id);
        Ok(id)
    }

    pub fn require_idle_backing(&self, id: DeviceBackingId) -> Result<()> {
        let backing = self
            .backings
            .get(&id)
            .ok_or_else(|| Error::State("unknown or released device backing".into()))?;
        if backing.fenced || !backing.views.is_empty() {
            return Err(Error::State(
                "device backing has live or uncertain views".into(),
            ));
        }
        Ok(())
    }

    /// Only the original last backing owner releases its charge after its handle.
    pub fn release_backing(&mut self, id: DeviceBackingId) -> Result<()> {
        self.require_idle_backing(id)?;
        let storage = self.backings.remove(&id).expect("observed backing");
        self.resident_bytes -= storage.bytes;
        Ok(())
    }

    pub(super) fn require_backing_accesses(
        &self,
        accesses: &[(BufferLeaseId, DeviceAccess)],
        selected: &BTreeMap<BufferLeaseId, DeviceAccess>,
    ) -> Result<()> {
        for &(id, access) in accesses {
            let lease = &self.leases[&id];
            let Some(range) = lease.backing else { continue };
            let backing = &self.backings[&range.id];
            if backing.fenced {
                return Err(Error::State("device backing completion uncertain".into()));
            }
            for alias in &backing.views {
                if *alias == id {
                    continue;
                }
                let other = &self.leases[alias];
                if !overlaps(lease, other) {
                    continue;
                }
                if selected.get(alias).is_some_and(|&other_access| {
                    access == DeviceAccess::Write || other_access == DeviceAccess::Write
                }) {
                    return Err(Error::State(
                        "overlapping device backing aliases in submission".into(),
                    ));
                }
                if other.fenced
                    || other.writer
                    || (access == DeviceAccess::Write && other.readers != 0)
                {
                    return Err(Error::State(
                        "overlapping device backing access in flight".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn overlaps(a: &Lease, b: &Lease) -> bool {
    match (a.backing, b.backing) {
        (Some(a_range), Some(b_range)) => {
            a_range.id == b_range.id
                && a_range.offset < b_range.offset + b.bytes
                && b_range.offset < a_range.offset + a.bytes
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests;
