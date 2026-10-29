//! Preparation choices from qualified implementations and full live byte costs.
//! Decisions grant no allocation, execution, source or completion authority.
use crate::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryFootprint {
    pub host_bytes: u64,
    pub device_bytes: u64,
}
impl MemoryFootprint {
    pub const fn fits(self, limit: Self) -> bool {
        self.host_bytes <= limit.host_bytes && self.device_bytes <= limit.device_bytes
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ResidencyCosts {
    pub streamed: MemoryFootprint,
    /// None: no qualified resident implementation. Costs include every live
    /// source, workspace, reader and candidate owner, not only weight bytes.
    pub resident: Option<MemoryFootprint>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyPreference {
    Auto,
    RequireResident,
    RequireStreamed,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyChoice {
    Resident,
    Streamed,
}

pub fn select_immutable_residency(
    costs: ResidencyCosts,
    budget: MemoryFootprint,
    preference: ResidencyPreference,
) -> Result<ResidencyChoice> {
    if preference != ResidencyPreference::RequireStreamed {
        match costs.resident {
            Some(cost) if cost.fits(budget) => return Ok(ResidencyChoice::Resident),
            None if preference == ResidencyPreference::RequireResident => {
                return Err(Error::Unsupported(
                    "resident source implementation unavailable".into(),
                ));
            }
            Some(cost) if preference == ResidencyPreference::RequireResident => {
                return Err(footprint_error("resident", cost, budget));
            }
            _ => {}
        }
    }
    if costs.streamed.fits(budget) {
        Ok(ResidencyChoice::Streamed)
    } else {
        Err(footprint_error("streamed", costs.streamed, budget))
    }
}

fn footprint_error(mode: &str, cost: MemoryFootprint, budget: MemoryFootprint) -> Error {
    Error::Limit(format!(
        "complete {mode} footprint exceeds budget: host_bytes={} host_limit={} device_bytes={} device_limit={}",
        cost.host_bytes, budget.host_bytes, cost.device_bytes, budget.device_bytes,
    ))
}

/// Bounded immutable small-window reuse inside an already paid metadata
/// envelope. The cache must price its entries AND outstanding evicted readers.
pub fn prepaid_window_cache_bytes(prepaid: u64, metadata_upper: u64) -> usize {
    let available = prepaid.saturating_sub(metadata_upper).min(8 * 1024 * 1024);
    if available < 128 * 1024 {
        0
    } else {
        available as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residency_requires_qualified_provider_and_both_complete_budgets() {
        let budget = MemoryFootprint {
            host_bytes: 100,
            device_bytes: 200,
        };
        let costs = ResidencyCosts {
            streamed: MemoryFootprint {
                host_bytes: 80,
                device_bytes: 90,
            },
            resident: Some(budget),
        };
        assert_eq!(
            select_immutable_residency(costs, budget, ResidencyPreference::Auto).unwrap(),
            ResidencyChoice::Resident
        );
        for resident in [
            None,
            Some(MemoryFootprint {
                host_bytes: 101,
                ..budget
            }),
            Some(MemoryFootprint {
                device_bytes: 201,
                ..budget
            }),
        ] {
            let costs = ResidencyCosts { resident, ..costs };
            assert_eq!(
                select_immutable_residency(costs, budget, ResidencyPreference::Auto).unwrap(),
                ResidencyChoice::Streamed
            );
            assert!(
                select_immutable_residency(costs, budget, ResidencyPreference::RequireResident)
                    .is_err()
            );
        }
        assert_eq!(
            select_immutable_residency(costs, budget, ResidencyPreference::RequireStreamed)
                .unwrap(),
            ResidencyChoice::Streamed
        );
        let too_small = MemoryFootprint {
            host_bytes: 79,
            device_bytes: 89,
        };
        assert!(select_immutable_residency(costs, too_small, ResidencyPreference::Auto).is_err());
    }
    #[test]
    fn residency_refusal_reports_both_required_and_available_extents() {
        let budget = MemoryFootprint {
            host_bytes: 100,
            device_bytes: 200,
        };
        let costs = ResidencyCosts {
            streamed: MemoryFootprint {
                host_bytes: 101,
                device_bytes: 90,
            },
            resident: Some(MemoryFootprint {
                host_bytes: 80,
                device_bytes: 201,
            }),
        };
        for (preference, expected) in [
            (
                ResidencyPreference::Auto,
                "complete streamed footprint exceeds budget: host_bytes=101 host_limit=100 device_bytes=90 device_limit=200",
            ),
            (
                ResidencyPreference::RequireResident,
                "complete resident footprint exceeds budget: host_bytes=80 host_limit=100 device_bytes=201 device_limit=200",
            ),
            (
                ResidencyPreference::RequireStreamed,
                "complete streamed footprint exceeds budget: host_bytes=101 host_limit=100 device_bytes=90 device_limit=200",
            ),
        ] {
            let Error::Limit(message) =
                select_immutable_residency(costs, budget, preference).unwrap_err()
            else {
                panic!("expected a capacity refusal");
            };
            assert_eq!(message, expected);
        }
    }
    #[test]
    fn cache_never_spends_unpaid_or_used_metadata_capacity() {
        assert_eq!(prepaid_window_cache_bytes(1, 2), 0);
        assert_eq!(prepaid_window_cache_bytes(128 * 1024 - 1, 0), 0);
        assert_eq!(prepaid_window_cache_bytes(128 * 1024, 0), 128 * 1024);
        assert_eq!(prepaid_window_cache_bytes(u64::MAX, 0), 8 * 1024 * 1024);
        assert_eq!(
            prepaid_window_cache_bytes(12 * 1024 * 1024, 7 * 1024 * 1024),
            5 * 1024 * 1024
        );
    }
}
