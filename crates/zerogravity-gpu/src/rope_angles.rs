//! Ordered source-FP32 angles in the original admitted RoPE scratch.
//!
//! Only explicitly selected CUDA uses this lowering. The enclosing original
//! attempt owns every handle, status, Ready and unknown-completion quarantine.
use crate::source_float::{round_source, widen_source};
use cubecl::{client::Client, prelude::*, server::Handle};
use cubecl_common::device::ServiceId;
use zerogravity_core::{Error, Result};

#[cfg(all(feature = "qualification", feature = "cuda"))]
pub(crate) mod qualification;

const MAX_HALF: usize = 32;

fn words(half: usize, rows: usize) -> Result<usize> {
    if !(1..=MAX_HALF).contains(&half) || ![1, 2, 4, 8].contains(&rows) {
        return Err(Error::Unsupported("ordered RoPE half/row geometry".into()));
    }
    Ok(half * rows)
}

fn scalars(theta_scale: f32, position: Option<f32>) -> Result<()> {
    if !theta_scale.is_finite()
        || theta_scale <= 0.0
        || theta_scale > 1.0
        || position.is_some_and(|p| !p.is_finite() || !(0.0..=i32::MAX as f32).contains(&p))
    {
        return Err(Error::Malformed("ordered RoPE scalar domain".into()));
    }
    Ok(())
}

fn checked_span(size: u64, start: u64, trim: u64, words: usize) -> Result<std::ops::Range<u64>> {
    let end = size
        .checked_sub(trim)
        .ok_or_else(|| Error::State("ordered RoPE end offset exceeds allocation".into()))?;
    let used = end
        .checked_sub(start)
        .ok_or_else(|| Error::State("ordered RoPE start exceeds checked end".into()))?;
    let needed = u64::try_from(words)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Limit("ordered RoPE byte domain".into()))?;
    if needed == 0 || !start.is_multiple_of(4) || used < needed {
        return Err(Error::State("ordered RoPE buffer alignment/size".into()));
    }
    Ok(start..end)
}

fn buffer(client: &Client, handle: &Handle, words: usize) -> Result<std::ops::Range<u64>> {
    if handle.service != client.service_id() {
        return Err(Error::State("ordered RoPE buffer service".into()));
    }
    checked_span(
        handle.size(),
        handle.offset_start.unwrap_or(0),
        handle.offset_end.unwrap_or(0),
        words,
    )
}

fn disjoint(
    a: &Handle,
    ar: std::ops::Range<u64>,
    b: &Handle,
    br: std::ops::Range<u64>,
) -> Result<()> {
    let a = a.clone().binding();
    let b = b.clone().binding();
    if a.memory.id() == b.memory.id() && ar.start < br.end && br.start < ar.end {
        return Err(Error::State("ordered RoPE positions alias angles".into()));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Positions<'a> {
    Single(f32),
    Rows(&'a Handle),
}

/// Borrowed views only; no allocation, source cache, lease or Ready capability.
pub(crate) struct PreparedAngles<'a> {
    service: ServiceId,
    angles: &'a Handle,
    positions: Positions<'a>,
    theta_scale: f32,
    half: usize,
    rows: usize,
}

impl<'a> PreparedAngles<'a> {
    /// None means an explicitly selected non-CUDA backend, never a CUDA error.
    pub(crate) fn single_for_cuda(
        client: &Client,
        angles: &'a Handle,
        position: f32,
        theta_scale: f32,
        half: usize,
    ) -> Result<Option<Self>> {
        if !crate::quantization::activation_fused::is_cuda_service(client) {
            return Ok(None);
        }
        let count = words(half, 1)?;
        scalars(theta_scale, Some(position))?;
        let _ = buffer(client, angles, count)?;
        crate::source_accumulation::ensure_capability(
            client,
            crate::source_accumulation::SourceAccumulation::WideF64,
        )?;
        Ok(Some(Self {
            service: client.service_id(),
            angles,
            positions: Positions::Single(position),
            theta_scale,
            half,
            rows: 1,
        }))
    }

    /// Runtime positions were validated by the original per-row parameter
    /// owner; this method preserves its immutable device positions handle.
    pub(crate) fn rows_for_cuda(
        client: &Client,
        angles: &'a Handle,
        positions: &'a Handle,
        theta_scale: f32,
        half: usize,
        rows: usize,
    ) -> Result<Option<Self>> {
        if !crate::quantization::activation_fused::is_cuda_service(client) {
            return Ok(None);
        }
        if ![2, 4, 8].contains(&rows) {
            return Err(Error::Unsupported("ordered RoPE native rows2/4/8".into()));
        }
        let count = words(half, rows)?;
        scalars(theta_scale, None)?;
        let angle_span = buffer(client, angles, count)?;
        let position_span = buffer(client, positions, rows)?;
        disjoint(angles, angle_span, positions, position_span)?;
        crate::source_accumulation::ensure_capability(
            client,
            crate::source_accumulation::SourceAccumulation::WideF64,
        )?;
        Ok(Some(Self {
            service: client.service_id(),
            angles,
            positions: Positions::Rows(positions),
            theta_scale,
            half,
            rows,
        }))
    }

    /// Exactly one original-client launch; all owning leases must already be
    /// pinned by the caller's armed physical submission.
    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        if self.service != client.service_id() {
            return Err(Error::State("ordered RoPE prepared service changed".into()));
        }
        // SAFETY: checked half/rows and unchanged original views cover every
        // index. One lane owns a whole row; no cross-lane dependency exists.
        unsafe {
            match self.positions {
                Positions::Single(position) => ordered_single::launch_unchecked(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(self.angles.clone(), self.half),
                    position,
                    self.theta_scale,
                    self.half,
                ),
                Positions::Rows(positions) => ordered_rows::launch_unchecked(
                    client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(32),
                    BufferArg::from_raw_parts(self.angles.clone(), self.rows * self.half),
                    BufferArg::from_raw_parts(positions.clone(), self.rows),
                    self.theta_scale,
                    self.half,
                    self.rows,
                ),
            }
        }
        Ok(1)
    }
}

#[cube(launch_unchecked)]
fn ordered_single(angles: &mut [f32], position: f32, theta_scale: f32, #[comptime] half: usize) {
    if ABSOLUTE_POS == 0usize {
        angles[0] = position;
        for stage in 1..half {
            angles[stage] = round_source(
                widen_source(angles[stage - 1]) * widen_source(theta_scale),
                0u32,
            );
        }
    }
}

#[cube(launch_unchecked)]
fn ordered_rows(
    angles: &mut [f32],
    positions: &[u32],
    theta_scale: f32,
    #[comptime] half: usize,
    #[comptime] rows: usize,
) {
    let row = ABSOLUTE_POS;
    if row < rows {
        let start = row * half;
        angles[start] = f32::cast_from(positions[row]);
        for stage in 1..half {
            angles[start + stage] = round_source(
                widen_source(angles[start + stage - 1]) * widen_source(theta_scale),
                0u32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_preserves_the_existing_bounded_rope_domain() {
        for half in 1..=32 {
            for rows in [1, 2, 4, 8] {
                assert_eq!(words(half, rows).unwrap(), half * rows);
            }
        }
        for (half, rows) in [(0, 1), (33, 1), (usize::MAX, 8), (32, 0), (32, 3)] {
            assert!(words(half, rows).is_err());
        }
    }

    #[test]
    fn scalar_guards_keep_signed_zero_and_refuse_invalid_source_domains() {
        for position in [0.0f32, -0.0, f32::from_bits(1), 32768.0, i32::MAX as f32] {
            scalars(0.5, Some(position)).unwrap();
        }
        for theta in [0.0, -0.0, -1.0, 1.000001, f32::INFINITY, f32::NAN] {
            assert!(scalars(theta, Some(0.0)).is_err());
        }
        for position in [-1.0, f32::INFINITY, f32::NAN, f32::MAX] {
            assert!(scalars(0.5, Some(position)).is_err());
        }
        scalars(f32::from_bits(1), None).unwrap();
    }

    #[test]
    fn malformed_offsets_refuse_before_unchecked_handle_range_arithmetic() {
        for (size, start, trim) in [
            (128, 0, 129),
            (128, 120, 16),
            (128, 129, 0),
            // Adding offsets would overflow; checked end/start subtraction
            // rejects them without ever forming their sum.
            (u64::MAX, u64::MAX - 8, 16),
            (u64::MAX, u64::MAX, u64::MAX),
            (128, 1, 0),
            (12, 4, 8),
        ] {
            assert!(checked_span(size, start, trim, 1).is_err());
        }
        #[cfg(target_pointer_width = "64")]
        assert!(checked_span(u64::MAX, 0, 0, usize::MAX).is_err());
        assert!(checked_span(128, 0, 0, 0).is_err());
    }

    #[test]
    fn checked_spans_preserve_original_full_and_guarded_views() {
        assert_eq!(checked_span(128, 0, 0, 32).unwrap(), 0..128);
        for words in [1, 2, 4, 8, 32, 64, 128, 256] {
            let total = (words as u64 + 8) * 4;
            assert_eq!(checked_span(total, 16, 16, words).unwrap(), 16..total - 16);
        }
    }
}
