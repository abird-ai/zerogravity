//! Retained performance candidates that are deliberately not production-selected.
//!
//! These are reusable implementation primitives for cross-model A/B work. A negative
//! result on one model/shape must not delete the implementation or become a global
//! policy. Qualification callers own model/source/root semantics and decide whether a
//! candidate is eligible for a particular measurement.

use zerogravity_core::{Error, Result};

pub const MAX_CANDIDATE_LANES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneRegroupPlan {
    len: u8,
    physical_to_logical: [u8; MAX_CANDIDATE_LANES],
    logical_to_physical: [u8; MAX_CANDIDATE_LANES],
}

impl LaneRegroupPlan {
    pub fn identity(len: usize) -> Result<Self> {
        if len == 0 || len > MAX_CANDIDATE_LANES {
            return Err(Error::Malformed(
                "candidate lane regroup width must be in 1..=8".into(),
            ));
        }
        let identity = std::array::from_fn(|lane| lane as u8);
        Ok(Self {
            len: len as u8,
            physical_to_logical: identity,
            logical_to_physical: identity,
        })
    }

    /// Stable, allocation-free clustering by a caller-provided physical key.
    ///
    /// A key may represent prompt rows, expert class, attention shape, or any
    /// future model-specific physical compatibility. Logical request/fairness/
    /// publication order remains outside this plan.
    pub fn clustered(keys: &[usize]) -> Result<Self> {
        let mut plan = Self::identity(keys.len())?;
        if keys.contains(&0) {
            return Err(Error::Malformed(
                "candidate lane regroup keys must be positive".into(),
            ));
        }
        for physical in 1..keys.len() {
            let logical = plan.physical_to_logical[physical];
            let key = keys[usize::from(logical)];
            let mut cursor = physical;
            while cursor != 0 && keys[usize::from(plan.physical_to_logical[cursor - 1])] > key {
                plan.physical_to_logical[cursor] = plan.physical_to_logical[cursor - 1];
                cursor -= 1;
            }
            plan.physical_to_logical[cursor] = logical;
        }
        for physical in 0..keys.len() {
            let logical = usize::from(plan.physical_to_logical[physical]);
            plan.logical_to_physical[logical] = physical as u8;
        }
        Ok(plan)
    }

    pub fn len(self) -> usize {
        usize::from(self.len)
    }

    pub fn is_empty(self) -> bool {
        self.len == 0
    }

    pub fn physical_lane(self, physical: usize) -> Result<usize> {
        if physical >= self.len() {
            return Err(Error::Malformed(
                "candidate physical lane outside regroup plan".into(),
            ));
        }
        Ok(usize::from(self.physical_to_logical[physical]))
    }

    pub fn logical_lane(self, logical: usize) -> Result<usize> {
        if logical >= self.len() {
            return Err(Error::Malformed(
                "candidate logical lane outside regroup plan".into(),
            ));
        }
        Ok(usize::from(self.logical_to_physical[logical]))
    }

    pub fn apply<T>(self, values: &mut [T]) -> Result<()> {
        self.reorder(values, self.physical_to_logical)
    }

    pub fn restore<T>(self, values: &mut [T]) -> Result<()> {
        self.reorder(values, self.logical_to_physical)
    }

    fn reorder<T>(self, values: &mut [T], desired_at_position: [u8; 8]) -> Result<()> {
        if values.len() != self.len() {
            return Err(Error::Malformed(
                "candidate lane regroup value coverage differs".into(),
            ));
        }
        let mut current: [u8; MAX_CANDIDATE_LANES] = std::array::from_fn(|lane| lane as u8);
        for (target, &desired) in desired_at_position.iter().take(values.len()).enumerate() {
            let source = (target..values.len())
                .find(|&position| current[position] == desired)
                .ok_or_else(|| Error::State("candidate lane regroup is not bijective".into()))?;
            values.swap(target, source);
            current.swap(target, source);
        }
        Ok(())
    }

    /// Discover contiguous equal-key physical groups of exactly width after
    /// clustering. Results use a fixed array; no allocation enters the A/B path.
    pub fn equal_groups(self, logical_keys: &[usize], width: usize) -> Result<EqualGroups> {
        if logical_keys.len() != self.len() || width == 0 || width > self.len() {
            return Err(Error::Malformed(
                "candidate equal-group geometry differs".into(),
            ));
        }
        let mut starts = [0u8; MAX_CANDIDATE_LANES];
        let mut count = 0usize;
        let mut run_start = 0usize;
        while run_start < self.len() {
            let key = logical_keys[self.physical_lane(run_start)?];
            let mut run_end = run_start + 1;
            while run_end < self.len() && logical_keys[self.physical_lane(run_end)?] == key {
                run_end += 1;
            }
            let mut start = run_start;
            while start + width <= run_end {
                starts[count] = start as u8;
                count += 1;
                start += width;
            }
            run_start = run_end;
        }
        Ok(EqualGroups {
            starts,
            count: count as u8,
            width: width as u8,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EqualGroups {
    starts: [u8; MAX_CANDIDATE_LANES],
    count: u8,
    width: u8,
}

impl EqualGroups {
    pub fn len(self) -> usize {
        usize::from(self.count)
    }

    pub fn is_empty(self) -> bool {
        self.count == 0
    }

    pub fn group(self, index: usize) -> Option<std::ops::Range<usize>> {
        (index < self.len()).then(|| {
            let start = usize::from(self.starts[index]);
            start..start + usize::from(self.width)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_lane_regroup_candidate_clusters_and_restores_without_allocation() -> Result<()> {
        let keys = [49, 48, 49, 48, 49, 48, 49, 48];
        let plan = LaneRegroupPlan::clustered(&keys)?;
        let physical = (0..8)
            .map(|lane| plan.physical_lane(lane))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(physical, [1, 3, 5, 7, 0, 2, 4, 6]);
        let groups = plan.equal_groups(&keys, 4)?;
        assert_eq!(groups.group(0), Some(0..4));
        assert_eq!(groups.group(1), Some(4..8));

        let mut values = [10, 11, 12, 13, 14, 15, 16, 17];
        plan.apply(&mut values)?;
        assert_eq!(values, [11, 13, 15, 17, 10, 12, 14, 16]);
        plan.restore(&mut values)?;
        assert_eq!(values, [10, 11, 12, 13, 14, 15, 16, 17]);
        Ok(())
    }
}
