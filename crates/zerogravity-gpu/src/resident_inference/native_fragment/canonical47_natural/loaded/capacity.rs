//! Closed owner envelopes over one unchanged source/native mathematical graph.
use super::*;
pub(super) const SESSIONS: usize = 2;
const MIB: usize = 1024 * 1024;
const ROOT_BYTES: usize = 32_784_384;
const EIGHT_PROVIDER: usize = 4096 * MIB;
const EIGHT_HOST: usize = 4096 * MIB;
const EIGHT_METADATA: usize = 2 * MIB;
const EIGHT_DEVICE_UPPER: usize = DEVICE_UPPER + 6 * ROOT_BYTES;
const EIGHT_HOST_UPPER: usize = HOST_UPPER + 6 * 3 * ROOT_BYTES + EIGHT_METADATA;

/// Configured owner capacity, never new numerical/native eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Canonical47LifecycleProfile {
    FiniteTwo,
    BenchmarkEight,
}
impl Canonical47LifecycleProfile {
    pub const fn session_limit(self) -> usize {
        match self {
            Self::FiniteTwo => SESSIONS,
            Self::BenchmarkEight => 8,
        }
    }
    pub const fn provider_bytes(self) -> usize {
        match self {
            Self::FiniteTwo => PROVIDER_SCOPE,
            Self::BenchmarkEight => EIGHT_PROVIDER,
        }
    }
    pub const fn host_bytes(self) -> usize {
        match self {
            Self::FiniteTwo => HOST_SCOPE,
            Self::BenchmarkEight => EIGHT_HOST,
        }
    }
    pub(super) const fn metadata_bytes(self) -> usize {
        match self {
            Self::FiniteTwo => 0,
            Self::BenchmarkEight => EIGHT_METADATA,
        }
    }
    pub(super) fn require_device_cap(self, bytes: u64) -> Result<()> {
        if self == Self::BenchmarkEight && bytes != 5120 * MIB as u64 {
            return Err(Error::Limit(
                "canonical eight-session exact5GiB device envelope".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn require_profile(
    source: &OriginalWindowModelSource,
    profile: Canonical47LifecycleProfile,
) -> Result<(usize, usize)> {
    // The original immutable source, numerical and finite geometry gates still run.
    require(source)?;
    match profile {
        Canonical47LifecycleProfile::FiniteTwo => require_gates(PROVIDER, HOST),
        Canonical47LifecycleProfile::BenchmarkEight => {
            if source.source().workspace_limit_bytes()? != EIGHT_HOST as u64 {
                return Err(Error::Limit(
                    "canonical eight-session original source host4GiB required".into(),
                ));
            }
            let (device, host) = eight_bounds()?;
            if device > EIGHT_PROVIDER || host > EIGHT_HOST {
                return Err(Error::Limit(
                    "canonical eight-session derived envelope".into(),
                ));
            }
            Ok((EIGHT_PROVIDER, EIGHT_HOST))
        }
    }
}

fn eight_bounds() -> Result<(usize, usize)> {
    let root = (18usize * 128 * 128 * 16 * 4)
        .checked_add(18 * 3 * 6144 * 4)
        .and_then(|n| n.checked_add(6 * 2 * 512 * 512 * 4))
        .ok_or_else(|| Error::Limit("canonical root extent overflow".into()))?;
    // Eight current roots +two checkpoints +two candidates +two displaced roots.
    // The predecessor included eight roots; the existing4096 alignment slots
    // cover all1934 extra managed allocations, including another6*48 leases.
    if root != ROOT_BYTES || 327 + 49 * 16 + 18 * 8 + 12 * 48 + 64 + 3 + 4 + 16 + 16 > 4096 {
        return Err(Error::State(
            "canonical eight-session source allocation geometry".into(),
        ));
    }
    Ok((EIGHT_DEVICE_UPPER, EIGHT_HOST_UPPER))
}

pub(super) fn require_model_metadata(
    profile: Canonical47LifecycleProfile,
    model: &ResidentModel,
) -> Result<()> {
    if profile == Canonical47LifecycleProfile::FiniteTwo {
        return Ok(());
    }
    let physical = model
        .states
        .iter()
        .map(|s| match s.layout {
            crate::resident_program::StateLayout::Vector { .. } => 1usize,
            crate::resident_program::StateLayout::Kv { .. } => 2usize,
        })
        .sum::<usize>();
    if model.states.len() != 42 || physical != 48 || model.state_bytes != ROOT_BYTES {
        return Err(Error::State(
            "canonical eight-session authentic42domains/48planes".into(),
        ));
    }
    require_metadata_counts(
        64,
        48,
        std::mem::size_of::<crate::resident_inference::Domain>(),
    )
}

fn require_metadata_counts(
    domains_capacity: usize,
    planes: usize,
    domain_bytes: usize,
) -> Result<()> {
    // Fixed headers/Arc/ledger entries, separate from the charged state payload.
    // Fourteen complete root metadata sets fit the real extra2MiB ticket.
    if domains_capacity > 64 || planes != 48 || domain_bytes > 1024 {
        return Err(Error::Limit("canonical owner metadata geometry".into()));
    }
    let per_root = domains_capacity
        .checked_mul(domain_bytes)
        .and_then(|n| n.checked_add(planes * 1024 + 4096))
        .ok_or_else(|| Error::Limit("canonical owner metadata overflow".into()))?;
    if per_root > 128 * 1024 || per_root * 14 > EIGHT_METADATA {
        return Err(Error::Limit(
            "canonical owner metadata exceeds admitted2MiB".into(),
        ));
    }
    Ok(())
}

pub(super) fn require_session_metadata(
    profile: Canonical47LifecycleProfile,
    session: &ResidentSession,
) -> Result<()> {
    if profile == Canonical47LifecycleProfile::FiniteTwo {
        return Ok(());
    }
    let planes = session
        .domains
        .iter()
        .map(|d| d.leases.len())
        .sum::<usize>();
    if session.domains.len() != 42
        || session.domains.iter().any(|d| {
            d.leases.capacity() > 2
                || d.generations.capacity() > 2
                || d.leases.len() != d.generations.len()
        })
    {
        return Err(Error::State(
            "canonical session root metadata geometry".into(),
        ));
    }
    require_metadata_counts(
        session.domains.capacity(),
        planes,
        std::mem::size_of::<crate::resident_inference::Domain>(),
    )
}
pub(super) const PENDING: usize = 2;
const PROVIDER: Option<usize> = Some(PROVIDER_SCOPE);
const HOST: Option<usize> = Some(HOST_SCOPE);
pub(super) fn require(source: &OriginalWindowModelSource) -> Result<(usize, usize)> {
    source.require_vllm08_full_source()?;
    if source.source().staging_chunk_bytes() != 1024 * 1024 {
        return Err(Error::Unsupported(
            "canonical natural immutable1MiB source chunk geometry".into(),
        ));
    }
    require_gates(PROVIDER, HOST)
}
const DEVICE_UPPER: usize = 2_177_274_680;
const HOST_UPPER: usize = 2_874_102_168;
const PROVIDER_SCOPE: usize = 2304 * 1024 * 1024;
const HOST_SCOPE: usize = 3072 * 1024 * 1024;
fn require_gates(provider: Option<usize>, host: Option<usize>) -> Result<(usize, usize)> {
    let p = provider.ok_or_else(|| {
        Error::Unsupported(
            "canonical natural whole lifecycle provider envelope pending ROOT".into(),
        )
    })?;
    let h = host.ok_or_else(|| {
        Error::Unsupported("canonical natural whole lifecycle host envelope pending ROOT".into())
    })?;
    if p != PROVIDER_SCOPE || h != HOST_SCOPE || p < DEVICE_UPPER || h < HOST_UPPER {
        return Err(Error::Limit(
            "canonical source-derived whole lifecycle envelope differs".into(),
        ));
    }
    Ok((p, h))
}
pub(super) fn host() -> Result<usize> {
    HOST.ok_or_else(|| {
        Error::Unsupported("canonical natural whole lifecycle host envelope pending ROOT".into())
    })
}
#[cfg(test)]
mod active8_tests;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_capacity_never_borrows_predecessor_envelope() {
        assert!(require_gates(None, None).is_err());
        assert!(require_gates(Some(1), None).is_err());
        assert!(require_gates(None, Some(1)).is_err());
        assert_eq!((SESSIONS, PENDING), (2, 2));
        assert_eq!(
            require_gates(PROVIDER, HOST).unwrap(),
            (PROVIDER_SCOPE, HOST_SCOPE)
        );
        assert!(require_gates(Some(PROVIDER_SCOPE - 1), HOST).is_err());
        assert!(require_gates(PROVIDER, Some(HOST_SCOPE + 1)).is_err());
    }
}
