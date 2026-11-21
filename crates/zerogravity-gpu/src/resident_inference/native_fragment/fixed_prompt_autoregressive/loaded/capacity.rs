//! Closed owner envelopes over one unchanged source/native mathematical graph.
use super::*;
pub(super) const TWO_SESSION_LIMIT: usize = 2;
const MIB: usize = 1024 * 1024;
const ROOT_BYTES: usize = 32_784_384;
const EIGHT_SESSION_PROVIDER: usize = 4096 * MIB;
const EIGHT_SESSION_HOST: usize = 4096 * MIB;
const EIGHT_SESSION_METADATA: usize = 2 * MIB;
const EIGHT_SESSION_DEVICE_UPPER: usize = DEVICE_UPPER + 6 * ROOT_BYTES;
const EIGHT_SESSION_HOST_UPPER: usize = HOST_UPPER + 6 * 3 * ROOT_BYTES + EIGHT_SESSION_METADATA;

/// Configured owner capacity, never new numerical/native eligibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedPromptLifecycleProfile {
    TwoSession,
    EightSessionBenchmark,
}
impl FixedPromptLifecycleProfile {
    pub const fn session_limit(self) -> usize {
        match self {
            Self::TwoSession => TWO_SESSION_LIMIT,
            Self::EightSessionBenchmark => 8,
        }
    }
    pub const fn provider_bytes(self) -> usize {
        match self {
            Self::TwoSession => PROVIDER_SCOPE,
            Self::EightSessionBenchmark => EIGHT_SESSION_PROVIDER,
        }
    }
    pub const fn host_bytes(self) -> usize {
        match self {
            Self::TwoSession => HOST_SCOPE,
            Self::EightSessionBenchmark => EIGHT_SESSION_HOST,
        }
    }
    pub(super) const fn metadata_bytes(self) -> usize {
        match self {
            Self::TwoSession => 0,
            Self::EightSessionBenchmark => EIGHT_SESSION_METADATA,
        }
    }
    pub(super) fn require_device_cap(self, bytes: u64) -> Result<()> {
        if self == Self::EightSessionBenchmark && bytes != 5120 * MIB as u64 {
            return Err(Error::Limit(
                "eight-session benchmark requires exact 5 GiB device envelope".into(),
            ));
        }
        Ok(())
    }
}

pub(super) fn require_profile(
    source: &VerifiedTensorSource,
    profile: FixedPromptLifecycleProfile,
) -> Result<(usize, usize)> {
    // Model factory checks finite source policy; this gate checks physical storage.
    require(source)?;
    match profile {
        FixedPromptLifecycleProfile::TwoSession => require_gates(PROVIDER, HOST),
        FixedPromptLifecycleProfile::EightSessionBenchmark => {
            if source.workspace_limit_bytes()? != EIGHT_SESSION_HOST as u64 {
                return Err(Error::Limit(
                    "eight-session benchmark requires exact 4 GiB source host workspace".into(),
                ));
            }
            let (device, host) = eight_session_bounds()?;
            if device > EIGHT_SESSION_PROVIDER || host > EIGHT_SESSION_HOST {
                return Err(Error::Limit(
                    "eight-session benchmark derived envelope exceeded".into(),
                ));
            }
            Ok((EIGHT_SESSION_PROVIDER, EIGHT_SESSION_HOST))
        }
    }
}

fn eight_session_bounds() -> Result<(usize, usize)> {
    let root = (18usize * 128 * 128 * 16 * 4)
        .checked_add(18 * 3 * 6144 * 4)
        .and_then(|n| n.checked_add(6 * 2 * 512 * 512 * 4))
        .ok_or_else(|| Error::Limit("canonical root extent overflow".into()))?;
    // Eight current roots +two checkpoints +two candidates +two displaced roots.
    // The predecessor included eight roots; the existing4096 alignment slots
    // cover all1934 extra managed allocations, including another6*48 leases.
    if root != ROOT_BYTES || 327 + 49 * 16 + 18 * 8 + 12 * 48 + 64 + 3 + 4 + 16 + 16 > 4096 {
        return Err(Error::State(
            "eight-session benchmark source allocation geometry differs".into(),
        ));
    }
    Ok((EIGHT_SESSION_DEVICE_UPPER, EIGHT_SESSION_HOST_UPPER))
}

pub(super) fn require_model_metadata(
    profile: FixedPromptLifecycleProfile,
    model: &ResidentModel,
) -> Result<()> {
    if profile == FixedPromptLifecycleProfile::TwoSession {
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
            "eight-session benchmark requires all state domains and 48 physical planes".into(),
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
    if per_root > 128 * 1024 || per_root * 14 > EIGHT_SESSION_METADATA {
        return Err(Error::Limit(
            "canonical owner metadata exceeds admitted2MiB".into(),
        ));
    }
    Ok(())
}

pub(super) fn require_session_metadata(
    profile: FixedPromptLifecycleProfile,
    session: &ResidentSession,
) -> Result<()> {
    if profile == FixedPromptLifecycleProfile::TwoSession {
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
pub(super) fn require(source: &VerifiedTensorSource) -> Result<(usize, usize)> {
    if source.staging_chunk_bytes() != 1024 * 1024 {
        return Err(Error::Unsupported(
            "canonical autoregressive immutable1MiB source chunk geometry".into(),
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
            "canonical autoregressive whole lifecycle provider envelope pending ROOT".into(),
        )
    })?;
    let h = host.ok_or_else(|| {
        Error::Unsupported(
            "canonical autoregressive whole lifecycle host envelope pending ROOT".into(),
        )
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
        Error::Unsupported(
            "canonical autoregressive whole lifecycle host envelope pending ROOT".into(),
        )
    })
}
#[cfg(test)]
mod eight_session_profile_tests;
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
