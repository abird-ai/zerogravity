//! Complete output rows with unchanged full K; byte geometry, not a GPU lease.
use super::*;
pub(super) struct Rows {
    rows: usize,
    row_bytes: usize,
    per_page: usize,
    bytes: usize,
}
impl Rows {
    pub(super) fn new(rows: usize, columns: usize) -> Result<Self> {
        let row_bytes = columns
            .checked_mul(2)
            .filter(|n| *n != 0 && *n <= PAGE && n.is_multiple_of(16))
            .ok_or_else(|| Error::Limit("Gemma full-K aligned source row".into()))?;
        let bytes = rows
            .checked_mul(row_bytes)
            .filter(|_| rows != 0)
            .ok_or_else(|| Error::Limit("Gemma complete source matrix byte overflow".into()))?;
        Ok(Self {
            rows,
            row_bytes,
            per_page: PAGE / row_bytes,
            bytes,
        })
    }
    pub(super) fn bytes(&self) -> usize {
        self.bytes
    }
    pub(super) fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    pub(super) fn count(&self) -> usize {
        self.rows.div_ceil(self.per_page)
    }
    pub(super) fn tile(&self, ordinal: usize) -> Result<std::ops::Range<usize>> {
        if ordinal >= self.count() {
            return Err(Error::State(
                "Gemma source tile outside complete coverage".into(),
            ));
        }
        let first = ordinal * self.per_page;
        Ok(first..first + (self.rows - first).min(self.per_page))
    }
    pub(super) fn selected(&self, id: u32) -> Result<std::ops::Range<usize>> {
        let id = id as usize;
        if id >= self.rows {
            return Err(Error::State(
                "Gemma source selected ID outside matrix".into(),
            ));
        }
        Ok(id..id + 1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_ple_head_and_down_rows_cover_every_byte_without_k_splitting() {
        for (rows, columns, count, tail) in [
            (262144, 8960, 4520, 42),
            (262144, 1536, 769, 256),
            (1536, 12288, 37, 24),
        ] {
            let layout = Rows::new(rows, columns).unwrap();
            assert_eq!(layout.count(), count);
            let mut next = 0;
            for index in 0..count {
                let range = layout.tile(index).unwrap();
                assert_eq!(range.start, next);
                assert!(range.len() * columns * 2 <= PAGE);
                next = range.end;
            }
            assert_eq!(next, rows);
            assert_eq!(layout.tile(count - 1).unwrap().len(), tail);
            assert_eq!(layout.bytes(), rows * columns * 2);
            assert_eq!(layout.selected((rows - 1) as u32).unwrap(), rows - 1..rows);
            assert!(layout.selected(rows as u32).is_err());
            assert!(layout.tile(count).is_err());
        }
    }
    #[test]
    fn malformed_overflow_odd_or_partial_packed_rows_refuse() {
        for (rows, columns) in [
            (0, 1536),
            (1, 0),
            (1, 1535),
            (1, usize::MAX),
            (usize::MAX, 1536),
            (1, PAGE),
        ] {
            assert!(Rows::new(rows, columns).is_err());
        }
    }
}
