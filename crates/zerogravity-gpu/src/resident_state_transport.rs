//! Bounded U32 transport across independent original state leases.
//!
//! No source math, allocation, upload, authority, Ready or publication. The
//! original caller validates source/generation and retains every physical lease
//! before launch, including captured-graph IO and unknown-completion quarantine.
use cubecl::{client::Client, prelude::*, server::Handle};
use cubecl_common::device::ServiceId;
use std::ops::Range;
use zerogravity_core::{Error, Result};

mod certified;
// Grouped B4 uses the pinned CUDA launch ABI. Keep it out of HIP/Vulkan-only
// builds: vllm_ptx_inline and the native_cohort callers share this feature.
#[cfg(all(feature = "cuda-native", target_os = "linux"))]
pub(crate) mod typed;
pub(crate) use certified::{PreparedCertifiedScatter, ensure_certified_capability};

const MAX_ROW_WORDS: usize = 1_000_000;
const MAX_TOTAL_WORDS: usize = 8_000_000;
const LANES: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shape {
    Vector { words: usize },
    Kv { slab: usize, capacity: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Direction {
    Gather,
    Scatter,
}

pub(crate) fn geometry(shape: Shape, rows: usize) -> Result<(usize, usize)> {
    if !matches!(rows, 2 | 4 | 8) {
        return Err(Error::Unsupported("state transport native rows".into()));
    }
    let words = match shape {
        Shape::Vector { words } => words,
        Shape::Kv { slab, capacity } => {
            if slab == 0 || capacity == 0 || capacity > 512 {
                return Err(Error::Limit("state transport KV geometry".into()));
            }
            slab.checked_mul(capacity)
                .ok_or_else(|| Error::Limit("state transport KV word overflow".into()))?
        }
    };
    let total = words
        .checked_mul(rows)
        .filter(|&n| n <= MAX_TOTAL_WORDS && u32::try_from(n).is_ok())
        .ok_or_else(|| Error::Limit("state transport aggregate word domain".into()))?;
    if words == 0 || words > MAX_ROW_WORDS {
        return Err(Error::Limit("state transport per-row word domain".into()));
    }
    Ok((words, total))
}

pub(crate) fn ensure_capability(client: &Client, shape: Shape, rows: usize) -> Result<()> {
    if !crate::quantization::activation_fused::is_cuda_service(client) {
        return Err(Error::Unsupported(
            "native state transport CUDA service".into(),
        ));
    }
    let (_, total) = geometry(shape, rows)?;
    let hardware = &client.properties().hardware;
    let bindings = if matches!(shape, Shape::Kv { .. }) {
        18
    } else {
        9
    };
    let cubes = (total as u32).div_ceil(LANES);
    if hardware.max_bindings < bindings
        || hardware.max_units_per_cube < LANES
        || hardware.max_cube_dim.0 < LANES
        || hardware.max_cube_dim.1 == 0
        || hardware.max_cube_dim.2 == 0
        || hardware.max_cube_count.0 < cubes
        || hardware.max_cube_count.1 == 0
        || hardware.max_cube_count.2 == 0
    {
        return Err(Error::Unsupported(
            "state transport CUDA binding/grid limits".into(),
        ));
    }
    Ok(())
}

fn checked_span(size: u64, start: u64, trim: u64, words: usize) -> Result<Range<u64>> {
    let end = size
        .checked_sub(trim)
        .ok_or_else(|| Error::State("state transport end exceeds allocation".into()))?;
    let used = end
        .checked_sub(start)
        .ok_or_else(|| Error::State("state transport start exceeds end".into()))?;
    let needed = u64::try_from(words)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Limit("state transport byte domain".into()))?;
    if needed == 0 || !start.is_multiple_of(4) || used < needed {
        return Err(Error::State("state transport buffer alignment/size".into()));
    }
    Ok(start..end)
}

fn buffer(client: &Client, handle: &Handle, words: usize) -> Result<Range<u64>> {
    if handle.service != client.service_id() {
        return Err(Error::State(
            "state transport foreign physical service".into(),
        ));
    }
    checked_span(
        handle.size(),
        handle.offset_start.unwrap_or(0),
        handle.offset_end.unwrap_or(0),
        words,
    )
}

/// Original handles only. `second` exists exactly for an original KV domain.
pub(crate) struct Row<'a> {
    pub first: &'a Handle,
    pub second: Option<&'a Handle>,
}

/// Checked metadata/handle clones, never a new allocation or lease capability.
pub(crate) struct PreparedTransport {
    service: ServiceId,
    first: Vec<Handle>,
    second: Vec<Handle>,
    dense_first: Handle,
    dense_second: Option<Handle>,
    shape: Shape,
    direction: Direction,
    words: usize,
    total: usize,
}

impl PreparedTransport {
    pub(crate) fn prepare(
        client: &Client,
        rows: &[Row<'_>],
        dense_first: &Handle,
        dense_second: Option<&Handle>,
        shape: Shape,
        direction: Direction,
    ) -> Result<Self> {
        ensure_capability(client, shape, rows.len())?;
        let (words, total) = geometry(shape, rows.len())?;
        let kv = matches!(shape, Shape::Kv { .. });
        if dense_second.is_some() != kv || rows.iter().any(|row| row.second.is_some() != kv) {
            return Err(Error::Malformed(
                "state transport domain buffer coverage".into(),
            ));
        }
        let mut ranges = Vec::with_capacity(if kv {
            2 * rows.len() + 2
        } else {
            rows.len() + 1
        });
        for row in rows {
            ranges.push((row.first, buffer(client, row.first, words)?));
            if let Some(second) = row.second {
                ranges.push((second, buffer(client, second, words)?));
            }
        }
        ranges.push((dense_first, buffer(client, dense_first, total)?));
        if let Some(second) = dense_second {
            ranges.push((second, buffer(client, second, total)?));
        }
        let ranges = ranges
            .into_iter()
            .map(|(handle, range)| (handle.clone().binding(), range))
            .collect::<Vec<_>>();
        for (i, (a, ar)) in ranges.iter().enumerate() {
            for (b, br) in &ranges[..i] {
                if a.memory.id() == b.memory.id() && ar.start < br.end && br.start < ar.end {
                    return Err(Error::State(
                        "state transport aliased independent buffers".into(),
                    ));
                }
            }
        }
        Ok(Self {
            service: client.service_id(),
            first: rows.iter().map(|row| row.first.clone()).collect(),
            second: rows.iter().filter_map(|row| row.second.cloned()).collect(),
            dense_first: dense_first.clone(),
            dense_second: dense_second.cloned(),
            shape,
            direction,
            words,
            total,
        })
    }

    /// One original-client launch per domain/direction, not per row. The
    /// enclosing physical submission already retains every original lease.
    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        if client.service_id() != self.service {
            return Err(Error::State(
                "prepared state transport service changed".into(),
            ));
        }
        // ABI-only inactive arguments reuse row zero. Compile-time row bounds
        // make them unreachable; no session, model row or state is padded.
        // SAFETY: prepare checked every original handle and visible word span.
        let first = |row: usize| unsafe {
            BufferArg::from_raw_parts(
                self.first.get(row).unwrap_or(&self.first[0]).clone(),
                self.words,
            )
        };
        // SAFETY: only KV branches call this with the checked second handles.
        let second = |row: usize| unsafe {
            BufferArg::from_raw_parts(
                self.second.get(row).unwrap_or(&self.second[0]).clone(),
                self.words,
            )
        };
        // SAFETY: the complete dense spans were checked against B times N.
        let dense_first =
            || unsafe { BufferArg::from_raw_parts(self.dense_first.clone(), self.total) };
        let dense_second = || unsafe {
            BufferArg::from_raw_parts(self.dense_second.as_ref().unwrap().clone(), self.total)
        };
        let grid = CubeCount::Static((self.total as u32).div_ceil(LANES), 1, 1);
        let threads = CubeDim::new_1d(LANES);
        // SAFETY: checked original service, disjoint visible ranges, exact
        // B×N layouts and all shader/binding/grid bounds were fixed prelaunch.
        unsafe {
            match (self.shape, self.direction) {
                (Shape::Vector { .. }, Direction::Gather) => gather_vector::launch_unchecked(
                    client,
                    grid,
                    threads,
                    first(0),
                    first(1),
                    first(2),
                    first(3),
                    first(4),
                    first(5),
                    first(6),
                    first(7),
                    dense_first(),
                    self.words,
                    self.first.len(),
                ),
                (Shape::Vector { .. }, Direction::Scatter) => scatter_vector::launch_unchecked(
                    client,
                    grid,
                    threads,
                    first(0),
                    first(1),
                    first(2),
                    first(3),
                    first(4),
                    first(5),
                    first(6),
                    first(7),
                    dense_first(),
                    self.words,
                    self.first.len(),
                ),
                (Shape::Kv { slab, capacity }, Direction::Gather) => gather_kv::launch_unchecked(
                    client,
                    grid,
                    threads,
                    first(0),
                    first(1),
                    first(2),
                    first(3),
                    first(4),
                    first(5),
                    first(6),
                    first(7),
                    second(0),
                    second(1),
                    second(2),
                    second(3),
                    second(4),
                    second(5),
                    second(6),
                    second(7),
                    dense_first(),
                    dense_second(),
                    slab,
                    capacity,
                    self.first.len(),
                ),
                (Shape::Kv { slab, capacity }, Direction::Scatter) => scatter_kv::launch_unchecked(
                    client,
                    grid,
                    threads,
                    first(0),
                    first(1),
                    first(2),
                    first(3),
                    first(4),
                    first(5),
                    first(6),
                    first(7),
                    second(0),
                    second(1),
                    second(2),
                    second(3),
                    second(4),
                    second(5),
                    second(6),
                    second(7),
                    dense_first(),
                    dense_second(),
                    slab,
                    capacity,
                    self.first.len(),
                ),
            }
        }
        Ok(1)
    }
}

#[cube(launch_unchecked)]
fn gather_vector(
    p0: &[u32],
    p1: &[u32],
    p2: &[u32],
    p3: &[u32],
    p4: &[u32],
    p5: &[u32],
    p6: &[u32],
    p7: &[u32],
    dense: &mut [u32],
    #[comptime] words: usize,
    #[comptime] rows: usize,
) {
    let at = ABSOLUTE_POS;
    if at < rows * words {
        let row = at / words;
        let local = at % words;
        let packed = at;
        if row == 0usize {
            dense[packed] = p0[local];
        } else if row == 1usize {
            dense[packed] = p1[local];
        } else if row == 2usize {
            dense[packed] = p2[local];
        } else if row == 3usize {
            dense[packed] = p3[local];
        } else if row == 4usize {
            dense[packed] = p4[local];
        } else if row == 5usize {
            dense[packed] = p5[local];
        } else if row == 6usize {
            dense[packed] = p6[local];
        } else if row == 7usize {
            dense[packed] = p7[local];
        }
    }
}

#[cube(launch_unchecked)]
fn scatter_vector(
    p0: &mut [u32],
    p1: &mut [u32],
    p2: &mut [u32],
    p3: &mut [u32],
    p4: &mut [u32],
    p5: &mut [u32],
    p6: &mut [u32],
    p7: &mut [u32],
    dense: &[u32],
    #[comptime] words: usize,
    #[comptime] rows: usize,
) {
    let at = ABSOLUTE_POS;
    if at < rows * words {
        let row = at / words;
        let local = at % words;
        let packed = at;
        if row == 0usize {
            p0[local] = dense[packed];
        } else if row == 1usize {
            p1[local] = dense[packed];
        } else if row == 2usize {
            p2[local] = dense[packed];
        } else if row == 3usize {
            p3[local] = dense[packed];
        } else if row == 4usize {
            p4[local] = dense[packed];
        } else if row == 5usize {
            p5[local] = dense[packed];
        } else if row == 6usize {
            p6[local] = dense[packed];
        } else if row == 7usize {
            p7[local] = dense[packed];
        }
    }
}

#[cube(launch_unchecked)]
fn gather_kv(
    p0: &[u32],
    p1: &[u32],
    p2: &[u32],
    p3: &[u32],
    p4: &[u32],
    p5: &[u32],
    p6: &[u32],
    p7: &[u32],
    v0: &[u32],
    v1: &[u32],
    v2: &[u32],
    v3: &[u32],
    v4: &[u32],
    v5: &[u32],
    v6: &[u32],
    v7: &[u32],
    dense: &mut [u32],
    dense_v: &mut [u32],
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] rows: usize,
) {
    let at = ABSOLUTE_POS;
    let words = slab * capacity;
    if at < rows * words {
        let row = at / words;
        let local = at % words;
        let packed = ((local / slab) * rows + row) * slab + local % slab;
        if row == 0usize {
            dense[packed] = p0[local];
            dense_v[packed] = v0[local];
        } else if row == 1usize {
            dense[packed] = p1[local];
            dense_v[packed] = v1[local];
        } else if row == 2usize {
            dense[packed] = p2[local];
            dense_v[packed] = v2[local];
        } else if row == 3usize {
            dense[packed] = p3[local];
            dense_v[packed] = v3[local];
        } else if row == 4usize {
            dense[packed] = p4[local];
            dense_v[packed] = v4[local];
        } else if row == 5usize {
            dense[packed] = p5[local];
            dense_v[packed] = v5[local];
        } else if row == 6usize {
            dense[packed] = p6[local];
            dense_v[packed] = v6[local];
        } else if row == 7usize {
            dense[packed] = p7[local];
            dense_v[packed] = v7[local];
        }
    }
}

#[cube(launch_unchecked)]
fn scatter_kv(
    p0: &mut [u32],
    p1: &mut [u32],
    p2: &mut [u32],
    p3: &mut [u32],
    p4: &mut [u32],
    p5: &mut [u32],
    p6: &mut [u32],
    p7: &mut [u32],
    v0: &mut [u32],
    v1: &mut [u32],
    v2: &mut [u32],
    v3: &mut [u32],
    v4: &mut [u32],
    v5: &mut [u32],
    v6: &mut [u32],
    v7: &mut [u32],
    dense: &[u32],
    dense_v: &[u32],
    #[comptime] slab: usize,
    #[comptime] capacity: usize,
    #[comptime] rows: usize,
) {
    let at = ABSOLUTE_POS;
    let words = slab * capacity;
    if at < rows * words {
        let row = at / words;
        let local = at % words;
        let packed = ((local / slab) * rows + row) * slab + local % slab;
        if row == 0usize {
            p0[local] = dense[packed];
            v0[local] = dense_v[packed];
        } else if row == 1usize {
            p1[local] = dense[packed];
            v1[local] = dense_v[packed];
        } else if row == 2usize {
            p2[local] = dense[packed];
            v2[local] = dense_v[packed];
        } else if row == 3usize {
            p3[local] = dense[packed];
            v3[local] = dense_v[packed];
        } else if row == 4usize {
            p4[local] = dense[packed];
            v4[local] = dense_v[packed];
        } else if row == 5usize {
            p5[local] = dense[packed];
            v5[local] = dense_v[packed];
        } else if row == 6usize {
            p6[local] = dense[packed];
            v6[local] = dense_v[packed];
        } else if row == 7usize {
            p7[local] = dense[packed];
            v7[local] = dense_v[packed];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_native_domain_includes_real_8x262144_state_words() {
        assert_eq!(
            geometry(Shape::Vector { words: 262_144 }, 8).unwrap(),
            (262_144, 2_097_152)
        );
        assert_eq!(
            geometry(
                Shape::Kv {
                    slab: 512,
                    capacity: 512
                },
                8
            )
            .unwrap(),
            (262_144, 2_097_152)
        );
        assert_eq!(
            geometry(
                Shape::Vector {
                    words: MAX_ROW_WORDS
                },
                8
            )
            .unwrap()
            .1,
            MAX_TOTAL_WORDS
        );
        for rows in [0, 1, 3, 5, 16, usize::MAX] {
            assert!(geometry(Shape::Vector { words: 1 }, rows).is_err());
        }
        for words in [0, MAX_ROW_WORDS + 1, usize::MAX] {
            assert!(geometry(Shape::Vector { words }, 8).is_err());
        }
        for (slab, capacity) in [(0, 512), (512, 0), (1, 513), (usize::MAX, 512)] {
            assert!(geometry(Shape::Kv { slab, capacity }, 8).is_err());
        }
    }

    #[test]
    fn checked_original_canary_views_refuse_underflow_and_wrapped_offsets() {
        assert_eq!(checked_span(64, 16, 16, 8).unwrap(), 16..48);
        for (size, start, trim, words) in [
            (64, 0, 65, 1),
            (64, 49, 16, 1),
            (64, u64::MAX, u64::MAX, 1),
            (64, 1, 0, 1),
            (64, 16, 16, 9),
            (64, 0, 0, 0),
            (64, 0, 0, usize::MAX),
        ] {
            assert!(checked_span(size, start, trim, words).is_err());
        }
    }
}
