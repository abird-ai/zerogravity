//! Model-neutral CUDA transport for two row-strided BF16 segments.
//!
//! The primitive copies raw BF16 bits from two independently-described
//! row-strided sources into two dense row-major outputs while certifying the
//! copied words for nonfinite BF16 values. It owns no model, source, session,
//! root, allocation, queue-completion or publication semantics.

use crate::resident_program::BufferRange;
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};

const LANES: u32 = 32;
const MAX_ROWS: usize = 64;
const MAX_TOTAL_ELEMENTS: usize = 8_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RowSegment {
    pub(crate) stride: usize,
    pub(crate) offset: usize,
    pub(crate) elements: usize,
}

impl RowSegment {
    fn validate(self, rows: usize) -> Result<(usize, usize, usize)> {
        if rows == 0
            || rows > MAX_ROWS
            || self.stride == 0
            || self.elements == 0
            || !self.stride.is_multiple_of(2)
            || !self.offset.is_multiple_of(2)
            || !self.elements.is_multiple_of(2)
            || self
                .offset
                .checked_add(self.elements)
                .is_none_or(|end| end > self.stride)
        {
            return Err(Error::Unsupported(
                "BF16 row-segment geometry/alignment".into(),
            ));
        }
        let source_elements = rows
            .checked_mul(self.stride)
            .ok_or_else(|| Error::Limit("BF16 row-segment source extent".into()))?;
        let output_elements = rows
            .checked_mul(self.elements)
            .ok_or_else(|| Error::Limit("BF16 row-segment output extent".into()))?;
        if source_elements > MAX_TOTAL_ELEMENTS || output_elements > MAX_TOTAL_ELEMENTS {
            return Err(Error::Limit("BF16 row-segment element bound".into()));
        }
        Ok((source_elements / 2, self.offset / 2, self.elements / 2))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TwoSegmentRows {
    pub(crate) rows: usize,
    pub(crate) first: RowSegment,
    pub(crate) second: RowSegment,
}

impl TwoSegmentRows {
    fn validate(self) -> Result<Validated> {
        let (source0_words, offset0_words, row0_words) = self.first.validate(self.rows)?;
        let (source1_words, offset1_words, row1_words) = self.second.validate(self.rows)?;
        let output0_words = self
            .rows
            .checked_mul(row0_words)
            .ok_or_else(|| Error::Limit("BF16 first dense output words".into()))?;
        let output1_words = self
            .rows
            .checked_mul(row1_words)
            .ok_or_else(|| Error::Limit("BF16 second dense output words".into()))?;
        let total_words = output0_words
            .checked_add(output1_words)
            .filter(|&n| n <= MAX_TOTAL_ELEMENTS / 2 && u32::try_from(n).is_ok())
            .ok_or_else(|| Error::Limit("BF16 row-split aggregate words".into()))?;
        Ok(Validated {
            rows: self.rows,
            source0_words,
            source1_words,
            stride0_words: self.first.stride / 2,
            stride1_words: self.second.stride / 2,
            offset0_words,
            offset1_words,
            row0_words,
            row1_words,
            output0_words,
            output1_words,
            total_words,
        })
    }
}

/// Dense BF16 matrix whose logical rows are independent values. The transpose
/// primitive changes only storage order from [rows, columns] to [columns, rows].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseRows {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
}
impl DenseRows {
    fn validate(self) -> Result<usize> {
        if self.rows == 0
            || self.rows > MAX_ROWS
            || !self.rows.is_multiple_of(2)
            || self.columns == 0
            || !self.columns.is_multiple_of(2)
        {
            return Err(Error::Unsupported(
                "BF16 dense-row transpose requires positive even geometry".into(),
            ));
        }
        self.rows
            .checked_mul(self.columns)
            .filter(|&elements| elements <= MAX_TOTAL_ELEMENTS)
            .ok_or_else(|| Error::Limit("BF16 dense-row transpose element bound".into()))
    }
}

#[derive(Clone, Copy)]
struct Validated {
    rows: usize,
    source0_words: usize,
    source1_words: usize,
    stride0_words: usize,
    stride1_words: usize,
    offset0_words: usize,
    offset1_words: usize,
    row0_words: usize,
    row1_words: usize,
    output0_words: usize,
    output1_words: usize,
    total_words: usize,
}

#[cube(launch_unchecked)]
fn copy_two_segments_certified(
    source0: &[u32],
    source1: &[u32],
    output0: &mut [u32],
    output1: &mut [u32],
    status: &[Atomic<u32>],
    rows: u32,
    stride0_words: u32,
    offset0_words: u32,
    row0_words: u32,
    stride1_words: u32,
    offset1_words: u32,
    row1_words: u32,
) {
    let i = ABSOLUTE_POS;
    let rows = usize::cast_from(rows);
    let stride0_words = usize::cast_from(stride0_words);
    let offset0_words = usize::cast_from(offset0_words);
    let row0_words = usize::cast_from(row0_words);
    let stride1_words = usize::cast_from(stride1_words);
    let offset1_words = usize::cast_from(offset1_words);
    let row1_words = usize::cast_from(row1_words);
    let first_total = rows * row0_words;
    let second_total = rows * row1_words;
    let total = first_total + second_total;
    if i < total {
        if i < first_total {
            let row = i / row0_words;
            let column = i % row0_words;
            let source_index = row * stride0_words + offset0_words + column;
            if source_index < source0.len() && i < output0.len() {
                let word = source0[source_index];
                let lo = word & 0xffffu32;
                let hi = word >> 16u32;
                if (lo & 0x7f80u32) == 0x7f80u32 || (hi & 0x7f80u32) == 0x7f80u32 {
                    status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
                }
                output0[i] = word;
            } else {
                status[0].fetch_or(crate::program_kernels::VIEW_OUT_OF_BOUNDS);
            }
        } else {
            let local = i - first_total;
            let row = local / row1_words;
            let column = local % row1_words;
            let source_index = row * stride1_words + offset1_words + column;
            if source_index < source1.len() && local < output1.len() {
                let word = source1[source_index];
                let lo = word & 0xffffu32;
                let hi = word >> 16u32;
                if (lo & 0x7f80u32) == 0x7f80u32 || (hi & 0x7f80u32) == 0x7f80u32 {
                    status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
                }
                output1[local] = word;
            } else {
                status[0].fetch_or(crate::program_kernels::VIEW_OUT_OF_BOUNDS);
            }
        }
    }
}

#[cube(launch_unchecked)]
fn row_major_to_inner_major_certified(
    source: &[u32],
    output: &mut [u32],
    status: &[Atomic<u32>],
    rows: u32,
    columns: u32,
) {
    let i = ABSOLUTE_POS;
    let rows = usize::cast_from(rows);
    let columns = usize::cast_from(columns);
    let row_pairs = rows / 2;
    let words = rows * columns / 2;
    if i < words {
        let column = i / row_pairs;
        let pair = i % row_pairs;
        let row0 = pair * 2;
        let row1 = row0 + 1;
        let words_per_row = columns / 2;
        let source_word0 = source[row0 * words_per_row + column / 2];
        let source_word1 = source[row1 * words_per_row + column / 2];
        let shift = u32::cast_from((column % 2) * 16);
        let lo = (source_word0 >> shift) & 0xffffu32;
        let hi = (source_word1 >> shift) & 0xffffu32;
        if (lo & 0x7f80u32) == 0x7f80u32 || (hi & 0x7f80u32) == 0x7f80u32 {
            status[0].fetch_or(crate::program_kernels::NONFINITE_VALUE);
        }
        output[i] = lo | (hi << 16u32);
    }
}

fn arg(handle: &Handle, words: usize) -> BufferArg {
    // SAFETY: launch() checks the exact visible byte extent for every binding.
    unsafe { BufferArg::from_raw_parts(handle.clone(), words) }
}

fn bytes(words: usize) -> Result<u64> {
    words
        .checked_mul(4)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or_else(|| Error::Limit("BF16 row-split byte extent".into()))
}

fn require_exact(client: &Client, handle: &Handle, words: usize, role: &str) -> Result<()> {
    if handle.service != client.service_id() || handle.size_in_used() != bytes(words)? {
        return Err(Error::State(format!(
            "BF16 row-split {role} service/extent"
        )));
    }
    Ok(())
}

fn require_nonaliasing(handles: &[&Handle]) -> Result<()> {
    let ranges = handles
        .iter()
        .map(|handle| BufferRange::new(handle))
        .collect::<Result<Vec<_>>>()?;
    for output in [2usize, 3, 4] {
        for other in 0..ranges.len() {
            if output != other && ranges[output].overlaps(&ranges[other]) {
                return Err(Error::State(
                    "BF16 row-split writable range aliases another binding".into(),
                ));
            }
        }
    }
    Ok(())
}

/// One launch, no allocation. The source handles expose exactly rows*stride
/// BF16 elements. Outputs expose exactly rows*segment.elements BF16 elements.
pub(crate) fn launch_two_segment_rows(
    client: &Client,
    geometry: TwoSegmentRows,
    source0: &Handle,
    source1: &Handle,
    output0: &Handle,
    output1: &Handle,
    status: &Handle,
) -> Result<usize> {
    let v = geometry.validate()?;
    require_exact(client, source0, v.source0_words, "source0")?;
    require_exact(client, source1, v.source1_words, "source1")?;
    require_exact(client, output0, v.output0_words, "output0")?;
    require_exact(client, output1, v.output1_words, "output1")?;
    require_exact(client, status, 1, "status")?;
    require_nonaliasing(&[source0, source1, output0, output1, status])?;

    let as_u32 = |n: usize, role: &str| {
        u32::try_from(n).map_err(|_| Error::Limit(format!("BF16 row-split {role} u32")))
    };
    let grid = CubeCount::Static(as_u32(v.total_words, "grid")?.div_ceil(LANES), 1, 1);
    let threads = CubeDim::new_1d(LANES);
    // SAFETY: geometry, exact visible extents, service identity and writable
    // non-aliasing were validated above. The kernel only copies/certifies.
    unsafe {
        copy_two_segments_certified::launch_unchecked(
            client,
            grid,
            threads,
            arg(source0, v.source0_words),
            arg(source1, v.source1_words),
            arg(output0, v.output0_words),
            arg(output1, v.output1_words),
            arg(status, 1),
            as_u32(v.rows, "rows")?,
            as_u32(v.stride0_words, "stride0")?,
            as_u32(v.offset0_words, "offset0")?,
            as_u32(v.row0_words, "row0")?,
            as_u32(v.stride1_words, "stride1")?,
            as_u32(v.offset1_words, "offset1")?,
            as_u32(v.row1_words, "row1")?,
        );
    }
    Ok(1)
}

/// Allocation-free storage transpose for dense BF16 rows. Source and output
/// expose the same exact extent and must not alias. The destination is laid out
/// as [column, row], making independent batch rows contiguous for one inner k.
pub(crate) fn launch_row_major_to_inner_major(
    client: &Client,
    geometry: DenseRows,
    source: &Handle,
    output: &Handle,
    status: &Handle,
) -> Result<usize> {
    let elements = geometry.validate()?;
    let words = elements / 2;
    require_exact(client, source, words, "transpose source")?;
    require_exact(client, output, words, "transpose output")?;
    require_exact(client, status, 1, "transpose status")?;
    let source_range = BufferRange::new(source)?;
    let output_range = BufferRange::new(output)?;
    let status_range = BufferRange::new(status)?;
    if source_range.overlaps(&output_range)
        || source_range.overlaps(&status_range)
        || output_range.overlaps(&status_range)
    {
        return Err(Error::State(
            "BF16 dense-row transpose writable range aliases another binding".into(),
        ));
    }
    let as_u32 = |n: usize, role: &str| {
        u32::try_from(n).map_err(|_| Error::Limit(format!("BF16 transpose {role} u32")))
    };
    let grid = CubeCount::Static(as_u32(words, "grid")?.div_ceil(LANES), 1, 1);
    let threads = CubeDim::new_1d(LANES);
    // SAFETY: exact service/extents and writable non-aliasing were checked.
    unsafe {
        row_major_to_inner_major_certified::launch_unchecked(
            client,
            grid,
            threads,
            arg(source, words),
            arg(output, words),
            arg(status, 1),
            as_u32(geometry.rows, "rows")?,
            as_u32(geometry.columns, "columns")?,
        );
    }
    Ok(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_requires_even_bf16_word_boundaries_and_bounded_rows() {
        let good = TwoSegmentRows {
            rows: 8,
            first: RowSegment {
                stride: 8192,
                offset: 0,
                elements: 6144,
            },
            second: RowSegment {
                stride: 8192,
                offset: 6144,
                elements: 2048,
            },
        };
        let v = good.validate().unwrap();
        assert_eq!(v.source0_words, 8 * 4096);
        assert_eq!(v.output0_words, 8 * 3072);
        assert_eq!(v.output1_words, 8 * 1024);
        assert_eq!(
            DenseRows {
                rows: 8,
                columns: 2048
            }
            .validate()
            .unwrap(),
            16_384
        );
        for dense in [
            DenseRows {
                rows: 0,
                columns: 2048,
            },
            DenseRows {
                rows: 3,
                columns: 2048,
            },
            DenseRows {
                rows: 66,
                columns: 2048,
            },
            DenseRows {
                rows: 8,
                columns: 0,
            },
            DenseRows {
                rows: 8,
                columns: 2047,
            },
        ] {
            assert!(dense.validate().is_err());
        }
        for bad in [
            TwoSegmentRows { rows: 0, ..good },
            TwoSegmentRows { rows: 65, ..good },
            TwoSegmentRows {
                first: RowSegment {
                    stride: 8191,
                    ..good.first
                },
                ..good
            },
            TwoSegmentRows {
                first: RowSegment {
                    offset: 1,
                    ..good.first
                },
                ..good
            },
            TwoSegmentRows {
                second: RowSegment {
                    offset: 6144,
                    elements: 2050,
                    ..good.second
                },
                ..good
            },
        ] {
            assert!(bad.validate().is_err());
        }
    }

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    #[test]
    #[ignore = "physical NVIDIA row-strided BF16 split/certificate qualification"]
    fn physical_split_matches_host_and_preserves_guards() -> Result<()> {
        futures_lite::future::block_on(physical())
    }

    #[cfg(all(
        feature = "qualification",
        feature = "cuda-cublas",
        target_os = "linux"
    ))]
    async fn physical() -> Result<()> {
        const GUARD: usize = 16;
        let diagnostic = crate::cuda::test_support::CudaDiagnostic::new("bf16-row-split")?;
        let service = diagnostic.service.clone();
        let client = service.client();
        let make_word = |row: usize, col: usize, seed: u16| -> u32 {
            let a = 0x3f00u16.wrapping_add(((row * 37 + col * 11) % 0x40) as u16) ^ seed;
            let b = 0xbf00u16.wrapping_add(((row * 17 + col * 7) % 0x40) as u16) ^ seed;
            u32::from(a) | (u32::from(b) << 16)
        };
        for rows in [2usize, 4, 8] {
            let geometry = TwoSegmentRows {
                rows,
                first: RowSegment {
                    stride: 18,
                    offset: 2,
                    elements: 8,
                },
                second: RowSegment {
                    stride: 20,
                    offset: 10,
                    elements: 6,
                },
            };
            let v = geometry.validate()?;
            let source0_words = (0..v.source0_words)
                .map(|i| make_word(i / 9, i % 9, 0))
                .collect::<Vec<_>>();
            let mut source1_words = (0..v.source1_words)
                .map(|i| make_word(i / 10, i % 10, 3))
                .collect::<Vec<_>>();
            if rows == 4 {
                let poison = 1 * 10 + 5 + 1;
                source1_words[poison] = 0x7f80_3f80;
            }
            let to_bytes = |words: &[u32]| {
                words
                    .iter()
                    .flat_map(|word| word.to_ne_bytes())
                    .collect::<Vec<_>>()
            };
            let guard_word = 0x55aa_a55au32;
            let guarded = |payload_words: usize| {
                let mut words = vec![guard_word; GUARD / 4];
                words.extend(std::iter::repeat_n(0xdead_beefu32, payload_words));
                words.extend(std::iter::repeat_n(guard_word, GUARD / 4));
                client.create_from_slice(&to_bytes(&words))
            };
            let source0 = client.create_from_slice(&to_bytes(&source0_words));
            let source1 = client.create_from_slice(&to_bytes(&source1_words));
            let raw0 = guarded(v.output0_words);
            let raw1 = guarded(v.output1_words);
            let output0 = raw0
                .clone()
                .offset_start(GUARD as u64)
                .offset_end(GUARD as u64);
            let output1 = raw1
                .clone()
                .offset_start(GUARD as u64)
                .offset_end(GUARD as u64);
            let status = client.create_from_slice(&0u32.to_ne_bytes());

            launch_two_segment_rows(
                client, geometry, &source0, &source1, &output0, &output1, &status,
            )?;
            client
                .sync()
                .await
                .map_err(|e| Error::State(format!("BF16 row-split sync: {e}")))?;

            let actual0 = client
                .read_one(output0)
                .map_err(|e| Error::State(format!("BF16 row-split output0: {e}")))?;
            let actual1 = client
                .read_one(output1)
                .map_err(|e| Error::State(format!("BF16 row-split output1: {e}")))?;
            let status_bytes = client
                .read_one(status)
                .map_err(|e| Error::State(format!("BF16 row-split status: {e}")))?;
            let raw0_bytes = client
                .read_one(raw0)
                .map_err(|e| Error::State(format!("BF16 row-split guard0: {e}")))?;
            let raw1_bytes = client
                .read_one(raw1)
                .map_err(|e| Error::State(format!("BF16 row-split guard1: {e}")))?;

            let expected0 = (0..rows)
                .flat_map(|row| {
                    let start = row * 9 + 1;
                    source0_words[start..start + 4].iter().copied()
                })
                .collect::<Vec<_>>();
            let expected1 = (0..rows)
                .flat_map(|row| {
                    let start = row * 10 + 5;
                    source1_words[start..start + 3].iter().copied()
                })
                .collect::<Vec<_>>();
            if &actual0[..] != to_bytes(&expected0).as_slice()
                || &actual1[..] != to_bytes(&expected1).as_slice()
            {
                return Err(Error::State(format!(
                    "BF16 row-split byte mismatch rows={rows}"
                )));
            }
            let expected_status = if rows == 4 {
                crate::program_kernels::NONFINITE_VALUE
            } else {
                0
            };
            if &status_bytes[..] != expected_status.to_ne_bytes().as_slice() {
                return Err(Error::State(format!(
                    "BF16 row-split status rows={rows}: {status_bytes:?}"
                )));
            }
            let guard = guard_word.to_ne_bytes().repeat(GUARD / 4);
            if raw0_bytes[..GUARD] != guard
                || raw0_bytes[raw0_bytes.len() - GUARD..] != guard
                || raw1_bytes[..GUARD] != guard
                || raw1_bytes[raw1_bytes.len() - GUARD..] != guard
            {
                return Err(Error::State(format!(
                    "BF16 row-split guard changed rows={rows}"
                )));
            }
            println!(
                "BF16_ROW_SPLIT_PASS rows={rows} exact=true status={expected_status} guards=true"
            );
        }

        {
            let geometry = DenseRows {
                rows: 8,
                columns: 16,
            };
            let mut row_major = (0..geometry.rows * geometry.columns)
                .map(|i| 0x3e00u16.wrapping_add((i % 0x80) as u16))
                .collect::<Vec<_>>();
            row_major[2 * geometry.columns + 5] = 0x7f80;
            let mut expected = Vec::with_capacity(row_major.len());
            for column in 0..geometry.columns {
                for row in 0..geometry.rows {
                    expected.push(row_major[row * geometry.columns + column]);
                }
            }
            let encode = |words: &[u16]| {
                words
                    .iter()
                    .flat_map(|word| word.to_ne_bytes())
                    .collect::<Vec<_>>()
            };
            let source = client.create_from_slice(&encode(&row_major));
            let output = client.empty(expected.len() * 2);
            let status = client.create_from_slice(&0u32.to_ne_bytes());
            launch_row_major_to_inner_major(client, geometry, &source, &output, &status)?;
            client
                .sync()
                .await
                .map_err(|e| Error::State(format!("BF16 dense transpose sync: {e}")))?;
            let actual = client
                .read_one(output)
                .map_err(|e| Error::State(format!("BF16 dense transpose output: {e}")))?;
            let flags = client
                .read_one(status)
                .map_err(|e| Error::State(format!("BF16 dense transpose status: {e}")))?;
            if &actual[..] != encode(&expected).as_slice()
                || &flags[..]
                    != crate::program_kernels::NONFINITE_VALUE
                        .to_ne_bytes()
                        .as_slice()
            {
                return Err(Error::State(
                    "BF16 dense transpose bits/status differ".into(),
                ));
            }
            println!("BF16_DENSE_TRANSPOSE_PASS rows=8 columns=16 exact=true status=nonfinite");
        }
        client
            .memory_cleanup()
            .map_err(|e| Error::State(format!("BF16 row-split cleanup: {e}")))?;
        service.close_model_namespace()?;
        Ok(())
    }
}
