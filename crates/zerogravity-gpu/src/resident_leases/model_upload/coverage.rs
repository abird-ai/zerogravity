//! Exact ordered source rows; checkpoint credit is private to real DMA caller.
use cubecl::cuda::SourceWeightLayout;
use zerogravity_core::{Error, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Region {
    pub member: usize,
    pub first: usize,
    pub end: usize,
    pub destination_offset: usize,
    pub bytes: usize,
}
pub(super) struct Coverage {
    layout: SourceWeightLayout,
    member: usize,
    row: usize,
    completed_bytes: usize,
    pending: Option<Region>,
}
impl Coverage {
    pub(super) fn new(layout: SourceWeightLayout) -> Self {
        Self {
            layout,
            member: 0,
            row: 0,
            completed_bytes: 0,
            pending: None,
        }
    }
    pub(super) fn members(&self) -> &'static [usize] {
        match self.layout {
            SourceWeightLayout::Vocabulary08 | SourceWeightLayout::Vocabulary2b => &[248320],
            SourceWeightLayout::GdnQkvz | SourceWeightLayout::Qwen2bGdnQkvz => &[6144, 2048],
            SourceWeightLayout::GdnBa | SourceWeightLayout::Qwen2bGdnBa => &[16, 16],
            SourceWeightLayout::AttentionQkv | SourceWeightLayout::Qwen2bAttentionQkv => {
                &[4096, 512, 512]
            }
            SourceWeightLayout::MlpGateUp => &[3584, 3584],
            SourceWeightLayout::Qwen2bMlpGateUp => &[6144, 6144],
            SourceWeightLayout::Qwen2bMlpDown
            | SourceWeightLayout::Qwen2bGdnOutput
            | SourceWeightLayout::Qwen2bAttentionOutput => &[2048],
            _ => &[1024],
        }
    }
    pub(super) fn next(&self) -> Result<Option<Region>> {
        if self.pending.is_some() {
            return Err(Error::State(
                "original learned source window pending".into(),
            ));
        }
        let Some(&rows) = self.members().get(self.member) else {
            self.require_complete()?;
            return Ok(None);
        };
        let row_bytes = self
            .layout
            .columns()
            .checked_mul(2)
            .ok_or_else(|| Error::Limit("source row width".into()))?;
        let end = self
            .row
            .checked_add(1024 * 1024 / row_bytes)
            .map(|end| end.min(rows))
            .filter(|end| *end > self.row)
            .ok_or_else(|| Error::Limit("source row interval".into()))?;
        let bytes = (end - self.row)
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Limit("source interval bytes".into()))?;
        if self
            .completed_bytes
            .checked_add(bytes)
            .is_none_or(|end| end > self.layout.bytes())
        {
            return Err(Error::Limit(
                "original packed source destination extent".into(),
            ));
        }
        Ok(Some(Region {
            member: self.member,
            first: self.row,
            end,
            destination_offset: self.completed_bytes,
            bytes,
        }))
    }
    pub(super) fn begin(&mut self, region: Region) -> Result<()> {
        if self.next()? != Some(region) {
            return Err(Error::State(
                "exact next original member/row/offset required".into(),
            ));
        }
        self.pending = Some(region);
        Ok(())
    }
    // Only original upload method invokes this after sync_buffers succeeded.
    pub(super) fn checkpoint(&mut self, region: Region) -> Result<()> {
        if self.pending != Some(region) {
            return Err(Error::State(
                "actual learned upload checkpoint region differs".into(),
            ));
        }
        self.completed_bytes = self
            .completed_bytes
            .checked_add(region.bytes)
            .ok_or_else(|| Error::Limit("learned upload coverage bytes".into()))?;
        self.row = region.end;
        if self.row == self.members()[self.member] {
            self.member += 1;
            self.row = 0;
        }
        self.pending = None;
        Ok(())
    }
    pub(super) fn require_complete(&self) -> Result<()> {
        if self.pending.is_some()
            || self.row != 0
            || self.member != self.members().len()
            || self.completed_bytes != self.layout.bytes()
        {
            return Err(Error::State(
                "full original ordered learned upload incomplete".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_closed_member_windows_cover_original_rows_once_without_padding() {
        for (layout, count) in [
            (SourceWeightLayout::GdnQkvz, 16),
            (SourceWeightLayout::GdnBa, 2),
            (SourceWeightLayout::GdnOutput, 4),
            (SourceWeightLayout::AttentionQkv, 10),
            (SourceWeightLayout::AttentionOutput, 4),
            (SourceWeightLayout::MlpGateUp, 14),
            (SourceWeightLayout::MlpDown, 8),
        ] {
            let mut c = Coverage::new(layout);
            let mut regions = 0;
            let mut offset = 0;
            while let Some(region) = c.next().unwrap() {
                assert_eq!(region.destination_offset, offset);
                assert!(region.bytes > 0 && region.bytes <= 1024 * 1024);
                assert_eq!(
                    region.bytes,
                    (region.end - region.first) * layout.columns() * 2
                );
                c.begin(region).unwrap();
                assert!(c.require_complete().is_err());
                assert!(c.next().is_err());
                assert!(c.begin(region).is_err());
                c.checkpoint(region).unwrap();
                assert!(c.checkpoint(region).is_err());
                regions += 1;
                offset += region.bytes;
            }
            c.require_complete().unwrap();
            assert_eq!(regions, count);
            assert_eq!(offset, layout.bytes());
        }
    }
    #[test]
    fn member_reorder_gap_overlap_extent_and_uncredited_final_window_refuse() {
        let mut c = Coverage::new(SourceWeightLayout::AttentionQkv);
        let first = c.next().unwrap().unwrap();
        for bad in [
            Region { member: 1, ..first },
            Region { first: 1, ..first },
            Region {
                end: first.end - 1,
                ..first
            },
            Region {
                destination_offset: 1,
                ..first
            },
            Region {
                bytes: first.bytes - 2,
                ..first
            },
            Region {
                destination_offset: usize::MAX,
                ..first
            },
        ] {
            assert!(c.begin(bad).is_err());
            assert_eq!(c.next().unwrap(), Some(first));
        }
        assert!(c.checkpoint(first).is_err());
        while let Some(region) = c.next().unwrap() {
            c.begin(region).unwrap();
            if region.destination_offset + region.bytes == c.layout.bytes() {
                break;
            }
            c.checkpoint(region).unwrap();
        }
        assert!(c.require_complete().is_err());
        assert!(c.next().is_err());
    }
    #[test]
    fn qwen2b_ordered_members_and_non_mebibyte_down_tail_are_exact() {
        for (layout, members, expected_count, expected_tail) in [
            (
                SourceWeightLayout::Qwen2bMlpGateUp,
                &[6144, 6144][..],
                48,
                (5888, 6144, 1048576),
            ),
            (
                SourceWeightLayout::Qwen2bMlpDown,
                &[2048][..],
                25,
                (2040, 2048, 98304),
            ),
        ] {
            let mut c = Coverage::new(layout);
            assert_eq!(c.members(), members);
            let mut previous = 0;
            let mut count = 0;
            let mut last = None;
            while let Some(region) = c.next().unwrap() {
                assert_eq!(region.destination_offset, previous);
                assert_eq!(
                    region.bytes,
                    (region.end - region.first) * layout.columns() * 2
                );
                let mut wrong = region;
                wrong.destination_offset += 2;
                assert!(c.begin(wrong).is_err());
                c.begin(region).unwrap();
                assert!(c.require_complete().is_err());
                c.checkpoint(region).unwrap();
                previous += region.bytes;
                count += 1;
                last = Some(region);
            }
            c.require_complete().unwrap();
            assert_eq!(count, expected_count);
            assert_eq!(previous, layout.bytes());
            let last = last.unwrap();
            assert_eq!((last.first, last.end, last.bytes), expected_tail);
            assert_eq!(last.destination_offset + last.bytes, layout.bytes());
        }
    }
    #[test]
    fn full_2b_groups_use_authentic_member_order_and_2676_real_windows() {
        let layouts = [
            (SourceWeightLayout::Qwen2bGdnQkvz, 32),
            (SourceWeightLayout::Qwen2bGdnBa, 2),
            (SourceWeightLayout::Qwen2bGdnOutput, 8),
            (SourceWeightLayout::Qwen2bAttentionQkv, 20),
            (SourceWeightLayout::Qwen2bAttentionOutput, 8),
            (SourceWeightLayout::Qwen2bMlpGateUp, 48),
            (SourceWeightLayout::Qwen2bMlpDown, 25),
        ];
        let mut counts = Vec::new();
        for (layout, expected) in layouts {
            let mut coverage = Coverage::new(layout);
            let mut count = 0;
            let mut bytes = 0;
            while let Some(region) = coverage.next().unwrap() {
                assert_eq!(region.destination_offset, bytes);
                assert_eq!(
                    region.bytes,
                    (region.end - region.first) * layout.columns() * 2
                );
                assert!(region.bytes <= 1048576);
                coverage.begin(region).unwrap();
                assert!(coverage.require_complete().is_err());
                coverage.checkpoint(region).unwrap();
                count += 1;
                bytes += region.bytes;
            }
            coverage.require_complete().unwrap();
            assert_eq!(bytes, layout.bytes());
            assert_eq!(count, expected);
            counts.push(count);
        }
        assert_eq!(
            18 * (counts[0] + counts[1] + counts[2] + counts[5] + counts[6])
                + 6 * (counts[3] + counts[4] + counts[5] + counts[6]),
            2676
        );
    }
}
