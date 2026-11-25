//! Cold OS memory observations and one host-backing constraint over the
//! original scoped leases. No allocator, per-request OS polling or OS promise.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct HostBackingSnapshot {
    /// Fixed first attested ceiling. Individual host scopes retain their caps.
    pub ceiling: Option<u64>,
    /// Last known effective OS/cgroup ceiling, never a free-byte counter.
    pub observed_ceiling: Option<u64>,
    pub reserved: u64,
}
impl HostBackingSnapshot {
    pub fn remaining(self) -> Option<u64> {
        self.limit().map(|n| n.saturating_sub(self.reserved))
    }

    pub fn limit(self) -> Option<u64> {
        match (self.ceiling, self.observed_ceiling) {
            (Some(fixed), Some(observed)) => Some(fixed.min(observed)),
            (fixed, observed) => fixed.or(observed),
        }
    }

    /// Intersect the logical remaining allowance and live availability.
    /// With no backing/charge mapping, conservatively protect all existing
    /// host commitments as potential growth (may underutilize resident RAM).
    pub fn conservative_available(self, ceiling: u64, available: u64, headroom: u64) -> u64 {
        let ceiling = self.limit().map_or(ceiling, |n| n.min(ceiling));
        ceiling.saturating_sub(self.reserved).min(
            available
                .min(ceiling)
                .saturating_sub(headroom)
                .saturating_sub(self.reserved),
        )
    }
}

#[derive(Clone, Debug)]
pub struct HostMemoryObservation {
    pub ceiling: Option<u64>,
    pub available: Option<u64>,
    pub provenance: String,
}
impl HostMemoryObservation {
    /// Observe the OS-visible host and complete visible cgroup-v2 ancestry.
    /// Unrecognized/inaccessible hierarchy means unknown effective availability.
    pub fn observe() -> Self {
        #[cfg(target_os = "linux")]
        {
            linux::observe()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Self {
                ceiling: None,
                available: None,
                provenance: "host memory observation unavailable on this OS".into(),
            }
        }
    }

    pub fn require_fit(
        &self,
        backing: HostBackingSnapshot,
        bytes: u64,
        headroom: u64,
    ) -> Result<()> {
        let (ceiling, available) = self.ceiling.zip(self.available).ok_or_else(|| {
            Error::State("effective HOST capacity/current availability unknown".into())
        })?;
        let usable = backing.conservative_available(ceiling, available, headroom);
        if bytes > usable {
            return Err(Error::resource(
                "host memory",
                "cold host fit after headroom and protected commitments",
                bytes,
                usable,
            ));
        }
        Ok(())
    }
}

impl ResourceArbiter {
    /// Establish the backing ceiling once, before cold preparation. A later
    /// observation may tighten the current fit, but cannot resize this ceiling.
    /// All existing host scopes still have their own immutable operator caps.
    pub fn establish_host_backing(&self, ceiling: u64) -> Result<HostBackingSnapshot> {
        let mut state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        if state.host_ceiling.is_none() && ceiling != 0 {
            state.host_ceiling = Some(ceiling);
        }
        // Keep the newest known hard constraint even on refusal; already
        // admitted owners retire normally, and no new growth may exceed it.
        state.host_observed_ceiling = Some(ceiling);
        let snapshot = state.host_backing();
        let effective = snapshot.limit().unwrap_or(ceiling);
        if effective == 0 || snapshot.reserved > effective {
            return Err(Error::resource(
                "host memory",
                "host backing ceiling",
                snapshot.reserved.max(1),
                effective,
            ));
        }
        Ok(snapshot)
    }

    /// Read-only observation of the same arbiter; not another reservation.
    pub fn host_backing(&self) -> Result<HostBackingSnapshot> {
        let state = self
            .shared
            .lock()
            .map_err(|_| Error::State("resource arbiter mutex poisoned".into()))?;
        Ok(state.host_backing())
    }

    /// Explicit operator routes can retain their existing scoped logical cap
    /// if OS observation is unavailable. This never claims a physical fit.
    pub fn observe_host_backing(&self) -> Result<HostMemoryObservation> {
        let observed = HostMemoryObservation::observe();
        if let Some(ceiling) = observed.ceiling {
            self.establish_host_backing(ceiling)?;
        }
        Ok(observed)
    }
}

#[cfg(target_os = "linux")]
mod linux;
