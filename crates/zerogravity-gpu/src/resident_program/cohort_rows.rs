//! Checked packed-row geometry for independent-session cohorts.
//!
//! This is scheduling/model agnostic metadata only. It owns no source identity,
//! root, session, numerical policy, allocation, launch or completion authority.

use std::ops::Range;
use zerogravity_core::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CohortRowSpans {
    rows: Box<[usize]>,
    offsets: Box<[usize]>,
    total_rows: usize,
}

impl CohortRowSpans {
    pub(crate) fn new(rows: &[usize]) -> Result<Self> {
        if rows.is_empty() || rows.contains(&0) {
            return Err(Error::Malformed(
                "cohort row spans require nonempty positive lanes".into(),
            ));
        }
        let mut offsets = Vec::with_capacity(rows.len() + 1);
        offsets.push(0usize);
        for &rows in rows {
            let next = offsets
                .last()
                .copied()
                .unwrap()
                .checked_add(rows)
                .ok_or_else(|| Error::Limit("cohort row span total overflow".into()))?;
            offsets.push(next);
        }
        let total_rows = *offsets.last().unwrap();
        Ok(Self {
            rows: rows.into(),
            offsets: offsets.into(),
            total_rows,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn total_rows(&self) -> usize {
        self.total_rows
    }

    pub(crate) fn lane_rows(&self, lane: usize) -> Result<usize> {
        self.rows
            .get(lane)
            .copied()
            .ok_or_else(|| Error::Malformed("cohort row lane outside spans".into()))
    }

    pub(crate) fn row_span(&self, lane: usize) -> Result<Range<usize>> {
        let start = self
            .offsets
            .get(lane)
            .copied()
            .ok_or_else(|| Error::Malformed("cohort row lane outside spans".into()))?;
        let end = self
            .offsets
            .get(lane + 1)
            .copied()
            .ok_or_else(|| Error::Malformed("cohort row lane outside spans".into()))?;
        Ok(start..end)
    }

    pub(crate) fn packed_elements(&self, elements_per_row: usize) -> Result<usize> {
        if elements_per_row == 0 {
            return Err(Error::Malformed(
                "cohort row element width must be positive".into(),
            ));
        }
        self.total_rows
            .checked_mul(elements_per_row)
            .ok_or_else(|| Error::Limit("cohort packed element extent overflow".into()))
    }

    pub(crate) fn element_span(
        &self,
        lane: usize,
        elements_per_row: usize,
    ) -> Result<Range<usize>> {
        if elements_per_row == 0 {
            return Err(Error::Malformed(
                "cohort row element width must be positive".into(),
            ));
        }
        let rows = self.row_span(lane)?;
        let start = rows
            .start
            .checked_mul(elements_per_row)
            .ok_or_else(|| Error::Limit("cohort lane element offset overflow".into()))?;
        let end = rows
            .end
            .checked_mul(elements_per_row)
            .ok_or_else(|| Error::Limit("cohort lane element extent overflow".into()))?;
        Ok(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_row_spans_preserve_lane_boundaries_without_policy() -> Result<()> {
        let spans = CohortRowSpans::new(&[47, 64, 3, 11])?;
        assert_eq!(spans.len(), 4);
        assert_eq!(spans.total_rows(), 125);
        assert_eq!(spans.row_span(0)?, 0..47);
        assert_eq!(spans.row_span(1)?, 47..111);
        assert_eq!(spans.row_span(2)?, 111..114);
        assert_eq!(spans.row_span(3)?, 114..125);
        assert_eq!(spans.lane_rows(1)?, 64);
        assert_eq!(spans.packed_elements(1024)?, 125 * 1024);
        assert_eq!(spans.element_span(2, 256)?, 111 * 256..114 * 256);
        assert!(spans.row_span(4).is_err());
        Ok(())
    }

    #[test]
    fn packed_row_spans_reject_only_invalid_metadata_or_overflow() {
        assert!(CohortRowSpans::new(&[]).is_err());
        assert!(CohortRowSpans::new(&[1, 0]).is_err());
        let spans = CohortRowSpans::new(&[usize::MAX]).unwrap();
        assert!(spans.packed_elements(2).is_err());
        assert!(spans.element_span(0, 2).is_err());
    }
}
