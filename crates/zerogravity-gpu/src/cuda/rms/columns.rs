//! Finite source-derived width selection; no dtype/shape reinterpretation.
use super::Rows;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Columns {
    Qwen08,
    Qwen2b,
}
impl Columns {
    pub(super) const fn count(self) -> usize {
        match self {
            Self::Qwen08 => 1024,
            Self::Qwen2b => 2048,
        }
    }
    pub(super) const fn mean_factor(self) -> &'static str {
        match self {
            Self::Qwen08 => "0x1p-10f",
            Self::Qwen2b => "0x1p-11f",
        }
    }
    pub(super) const fn mean_block(self, rows: Rows) -> (u32, u32) {
        match (self, rows) {
            (Self::Qwen08, _) => rows.mean_block(),
            (Self::Qwen2b, Rows::CohortB4 { initial: true }) => {
                panic!("2B INITIAL B4 RMS is outside source capability")
            }
            (Self::Qwen2b, Rows::CohortB4 { initial: false }) => (128, 4),
            (Self::Qwen2b, Rows::One) => (512, 1),
            (Self::Qwen2b, Rows::Two) => (256, 2),
            // Legacy FixedPrompt/2048 stays closed; typed text uses the same warp reduction.
            (Self::Qwen2b, Rows::FixedPrompt | Rows::SixtyFour | Rows::Text { .. }) => (32, 16),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn torch_reduce_vector4_geometry_has_no_y_or_global_reduction() {
        for columns in [Columns::Qwen08, Columns::Qwen2b] {
            for rows in [Rows::One, Rows::Two, Rows::SixtyFour] {
                let (width, height) = columns.mean_block(rows);
                assert!(width * height <= 512);
                assert_eq!(rows.count() % height as usize, 0);
                // ReduceConfig::values_per_thread counts scalar inputs. The
                // y-split predicate is false for every admitted configuration.
                let values = columns.count().div_ceil(width as usize);
                assert!(values < (height as usize * 16).min(256));
                // Vector loads cover complete aligned rows; no head/tail.
                let vectors = columns.count() / 4;
                let visited = (0..width as usize)
                    .flat_map(|x| (x..vectors).step_by(width as usize))
                    .count();
                assert_eq!(visited, vectors);
            }
        }
        assert_eq!(Columns::Qwen2b.mean_block(Rows::One), (512, 1));
        assert_eq!(Columns::Qwen08.mean_block(Rows::One), (256, 1));
        for columns in [Columns::Qwen08, Columns::Qwen2b] {
            let (x, y) = columns.mean_block(Rows::CohortB4 { initial: false });
            assert_eq!((x, y), (128, 4));
            let vector_count = columns.count() / 4;
            let visited = (0..x as usize)
                .flat_map(|lane| (lane..vector_count).step_by(x as usize))
                .collect::<Vec<_>>();
            assert_eq!(
                visited
                    .iter()
                    .copied()
                    .collect::<std::collections::BTreeSet<_>>(),
                (0..vector_count).collect::<std::collections::BTreeSet<_>>()
            );
            assert_eq!(visited.len(), vector_count);
        }
        assert_eq!(Columns::Qwen2b.mean_factor(), "0x1p-11f");
        assert_eq!(Columns::Qwen08.mean_factor(), "0x1p-10f");
    }
}
