//! Typed final-state scatter + original finite-status certificate, no FP math.
//! Opaque U32 transport stays separate; only the model owner selects this API.
use super::*;
use crate::program_kernels::NONFINITE_VALUE;

pub(crate) fn ensure_certified_capability(
    client: &Client,
    shape: Shape,
    rows: usize,
) -> Result<()> {
    ensure_capability(client, shape, rows)?;
    if client.properties().hardware.max_bindings < 10 {
        return Err(Error::Unsupported(
            "certified state scatter CUDA binding limits".into(),
        ));
    }
    Ok(())
}

/// Checked original handles only. Caller retains the existing status and ALL
/// state leases in its original physical flight before the first launch.
pub(crate) struct PreparedCertifiedScatter {
    transport: PreparedTransport,
    status: Handle,
}

impl PreparedCertifiedScatter {
    pub(crate) fn prepare(
        client: &Client,
        rows: &[Row<'_>],
        dense_first: &Handle,
        dense_second: Option<&Handle>,
        shape: Shape,
        status: &Handle,
    ) -> Result<Self> {
        ensure_certified_capability(client, shape, rows.len())?;
        let transport = PreparedTransport::prepare(
            client,
            rows,
            dense_first,
            dense_second,
            shape,
            Direction::Scatter,
        )?;
        // Each typed field uses8 row bindings +dense +status. KV fields are
        // separate launches, retaining the accepted <=18 binding ceiling.
        let status_range = buffer(client, status, 1)?;
        let status_binding = status.clone().binding();
        for handle in transport
            .first
            .iter()
            .chain(&transport.second)
            .chain(std::iter::once(&transport.dense_first))
            .chain(transport.dense_second.iter())
        {
            let range = checked_span(
                handle.size(),
                handle.offset_start.unwrap_or(0),
                handle.offset_end.unwrap_or(0),
                1,
            )?;
            if handle.clone().binding().memory.id() == status_binding.memory.id()
                && overlaps(&range, &status_range)
            {
                return Err(Error::State(
                    "certified state scatter status aliases state".into(),
                ));
            }
        }
        Ok(Self {
            transport,
            status: status.clone(),
        })
    }

    /// Vector1/KV2 actual source launches. Never clears existing status bits.
    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        let transport = &self.transport;
        if client.service_id() != transport.service {
            return Err(Error::State(
                "certified state scatter service changed".into(),
            ));
        }
        let slab = match transport.shape {
            Shape::Vector { .. } => 1,
            Shape::Kv { slab, .. } => slab,
        };
        let interleaved = matches!(transport.shape, Shape::Kv { .. });
        let channels = 1 + usize::from(interleaved);
        for channel in 0..channels {
            let (rows, dense) = if channel == 0 {
                (&transport.first, &transport.dense_first)
            } else {
                (
                    &transport.second,
                    transport.dense_second.as_ref().expect("checked original V"),
                )
            };
            // SAFETY: full service/range/alias/geometry preflight precedes
            // enclosing submission. Inactive ABI aliases cannot be addressed.
            let row = |index: usize| unsafe {
                BufferArg::from_raw_parts(
                    rows.get(index).unwrap_or(&rows[0]).clone(),
                    transport.words,
                )
            };
            // SAFETY: the original flight retains every checked state/status
            // lease; all visible spans and CUDA launch bounds are prevalidated.
            unsafe {
                scatter_field::launch_unchecked(
                    client,
                    CubeCount::Static((transport.total as u32).div_ceil(LANES), 1, 1),
                    CubeDim::new_1d(LANES),
                    row(0),
                    row(1),
                    row(2),
                    row(3),
                    row(4),
                    row(5),
                    row(6),
                    row(7),
                    BufferArg::from_raw_parts(dense.clone(), transport.total),
                    BufferArg::from_raw_parts(self.status.clone(), 1),
                    transport.words,
                    slab,
                    rows.len(),
                    interleaved,
                );
            }
        }
        Ok(channels)
    }
}

fn overlaps(a: &Range<u64>, b: &Range<u64>) -> bool {
    a.start < b.end && b.start < a.end
}

#[cube(launch_unchecked)]
fn scatter_field(
    p0: &mut [u32],
    p1: &mut [u32],
    p2: &mut [u32],
    p3: &mut [u32],
    p4: &mut [u32],
    p5: &mut [u32],
    p6: &mut [u32],
    p7: &mut [u32],
    dense: &[u32],
    status: &[Atomic<u32>],
    #[comptime] words: usize,
    #[comptime] slab: usize,
    #[comptime] rows: usize,
    #[comptime] interleaved: bool,
) {
    let at = ABSOLUTE_POS;
    if at < rows * words {
        let row = at / words;
        let local = at % words;
        let mut packed = at;
        if interleaved {
            packed = ((local / slab) * rows + row) * slab + local % slab;
        }
        let bits = dense[packed];
        if row == 0usize {
            p0[local] = bits;
        } else if row == 1usize {
            p1[local] = bits;
        } else if row == 2usize {
            p2[local] = bits;
        } else if row == 3usize {
            p3[local] = bits;
        } else if row == 4usize {
            p4[local] = bits;
        } else if row == 5usize {
            p5[local] = bits;
        } else if row == 6usize {
            p6[local] = bits;
        } else if row == 7usize {
            p7[local] = bits;
        }
        // EXACT original certify_finite_strided predicate/status bit. Reads
        // the same stored U32 value; no cast, FP operation or zero exception.
        if (bits & 0x7f800000u32) == 0x7f800000u32 {
            status[0].fetch_or(NONFINITE_VALUE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_status_word_span_and_disjoint_view_boundaries() {
        assert_eq!(checked_span(64, 16, 44, 1).unwrap(), 16..20);
        assert!(!overlaps(&(0..16), &(16..20)));
        assert!(!overlaps(&(20..64), &(16..20)));
        assert!(overlaps(&(0..64), &(16..20)));
        assert!(overlaps(&(16..20), &(16..20)));
        for (size, start, trim) in [(3, 0, 0), (64, 1, 0), (64, 64, 0), (64, 0, 65)] {
            assert!(checked_span(size, start, trim, 1).is_err());
        }
    }
}
