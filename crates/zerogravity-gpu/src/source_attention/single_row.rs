//! Single-row source attention, with checked active geometry and borrowed owners.
//! No model policy, allocator, native admission or root publication.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result, accelerator::program::TorchCpuAttentionGeometry};
mod prepared;
pub(crate) use prepared::kernels;
#[cfg(all(
    test,
    feature = "hip",
    feature = "qwen-source-qualification",
    feature = "hardware-abort-probe"
))]
mod physical;

pub(crate) fn scratch_bytes(g: TorchCpuAttentionGeometry, position: usize) -> Result<[usize; 8]> {
    let (q, _, tokens) = g.validate(position)?;
    Ok([
        32 * 4,
        64 * 4,
        g.query_heads * 4,
        g.query_heads * tokens * 4,
        g.query_heads * 4,
        g.query_heads * 4,
        q * 8,
        q * 4,
    ])
}
fn check(h: &Handle, bytes: usize) -> Result<()> {
    let n = h
        .size()
        .checked_sub(h.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(h.offset_end.unwrap_or(0)));
    if bytes == 0 || n.is_none_or(|n| n < bytes as u64) {
        return Err(Error::State("source attention borrowed range".into()));
    }
    Ok(())
}
fn arg(h: &Handle, n: usize) -> BufferArg {
    // Extents checked before any launch; caller retains the original flight.
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}
fn require_ranges(
    client: &Client,
    scratch: &[Handle],
    input: &[&Handle],
    output: &[&Handle; 9],
    tables: [&Handle; 4],
    status: &Handle,
) -> Result<()> {
    use crate::resident_program::BufferRange;
    let mut writes = Vec::with_capacity(scratch.len() + output.len() + 1);
    for h in scratch.iter().chain(output.iter().copied()).chain([status]) {
        let range = BufferRange::new(h)?;
        if h.service != client.service_id()
            || writes.iter().any(|old: &BufferRange| old.overlaps(&range))
        {
            return Err(Error::State(
                "source attention distinct same-service writes".into(),
            ));
        }
        writes.push(range);
    }
    for h in input.iter().copied().chain(tables) {
        let range = BufferRange::new(h)?;
        if h.service != client.service_id() || writes.iter().any(|write| write.overlaps(&range)) {
            return Err(Error::State(
                "source attention original readonly operands".into(),
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &Client,
    scratch: &[Handle],
    input: &[&Handle],
    output: [&Handle; 9],
    frequencies: &Handle,
    trig_table: &Handle,
    scalar_table: &Handle,
    status: &Handle,
    zero: &Handle,
    geometry: zerogravity_core::accelerator::program::TorchCpuAttentionGeometry,
    position: usize,
) -> Result<usize> {
    let (q, kv, tokens) = geometry.validate(position)?;
    if input.len() != if position == 0 { 4 } else { 6 } {
        return Err(Error::Unsupported(
            "source attention current/prior input arity".into(),
        ));
    }
    let extents = scratch_bytes(geometry, position)?;
    if scratch.len() != extents.len() {
        return Err(Error::State("source attention scratch arity".into()));
    }
    for (h, bytes) in scratch.iter().zip(extents) {
        check(h, bytes)?;
    }
    require_ranges(
        client,
        scratch,
        input,
        &output,
        [frequencies, trig_table, scalar_table, zero],
        status,
    )?;
    let scores = geometry.query_heads * tokens;
    let mut extents = vec![q, kv, kv, q];
    if position > 0 {
        extents.extend([position * kv, position * kv]);
    }
    for (h, n) in input.iter().zip(extents) {
        check(h, n * 4)?;
    }
    for (h, n) in output.iter().zip([
        q,
        kv,
        tokens * kv,
        tokens * kv,
        scores,
        scores,
        scores,
        q,
        q,
    ]) {
        check(h, n * 4)?;
    }
    check(frequencies, 32 * 4)?;
    check(trig_table, 35 * 8)?;
    check(scalar_table, 72 * 8)?;
    check(status, 4)?;
    check(zero, 4)?;
    let mut launches = 0;
    unsafe {
        crate::source_rotary::kernels::angles::launch_unchecked(
            client,
            grid(32),
            threads(),
            arg(frequencies, 32),
            arg(&scratch[0], 32),
            position as u32,
            32usize,
        );
        crate::source_trig::sincos_angles::launch_unchecked(
            client,
            grid(32),
            threads(),
            arg(&scratch[0], 32),
            arg(trig_table, 35),
            arg(&scratch[1], 64),
            32usize,
        );
        launches += 2;
        for (original, rotated, count) in [(input[0], output[0], q), (input[1], output[1], kv)] {
            // Partial rotary leaves the remaining192 head lanes bit-identical.
            crate::program_kernels::copy_strided_words::launch_unchecked(
                client,
                grid(count),
                threads(),
                arg(original, count),
                arg(rotated, count),
                arg(status, 1),
                0u32,
                0u32,
                count as u32,
                count as u32,
                1usize,
                count,
                true,
            );
            crate::source_rotary::kernels::apply::launch_unchecked(
                client,
                grid((count / 256) * 32),
                threads(),
                arg(original, count),
                arg(&scratch[1], 64),
                arg(rotated, count),
                arg(status, 1),
                256usize,
                count,
                64usize,
            );
            launches += 2;
        }
        super::transport::append::launch_unchecked(
            client,
            grid(tokens * kv),
            threads(),
            arg(
                if position == 0 { zero } else { input[4] },
                if position == 0 { 1 } else { position * kv },
            ),
            arg(
                if position == 0 { zero } else { input[5] },
                if position == 0 { 1 } else { position * kv },
            ),
            arg(output[1], kv),
            arg(input[2], kv),
            arg(output[2], tokens * kv),
            arg(output[3], tokens * kv),
            kv,
            position,
        );
        crate::source_attention::kernels::qk::launch_unchecked(
            client,
            CubeCount::Static(scores as u32, 1, 1),
            CubeDim::new_1d(64),
            arg(output[0], q),
            arg(output[2], tokens * kv),
            arg(output[4], scores),
            arg(output[5], scores),
            arg(status, 1),
            arg(zero, 1),
            256usize,
            geometry.query_heads,
            geometry.kv_heads,
            tokens,
            0.0625f32,
            tokens as u32,
            0u32,
        );
        crate::source_attention::kernels::reduce::launch_unchecked(
            client,
            CubeCount::Static(geometry.query_heads as u32, 1, 1),
            threads(),
            arg(output[5], scores),
            arg(&scratch[2], geometry.query_heads),
            arg(status, 1),
            tokens as u32,
            true,
            tokens,
        );
        crate::source_attention::kernels::exponentials::launch_unchecked(
            client,
            grid(scores),
            threads(),
            arg(output[5], scores),
            arg(&scratch[2], geometry.query_heads),
            arg(&scratch[3], scores),
            arg(status, 1),
            tokens as u32,
            geometry.query_heads,
            tokens,
        );
        crate::source_attention::kernels::reduce::launch_unchecked(
            client,
            CubeCount::Static(geometry.query_heads as u32, 1, 1),
            threads(),
            arg(&scratch[3], scores),
            arg(&scratch[4], geometry.query_heads),
            arg(status, 1),
            tokens as u32,
            false,
            tokens,
        );
        crate::source_attention::kernels::inverse::launch_unchecked(
            client,
            grid(geometry.query_heads),
            threads(),
            arg(&scratch[4], geometry.query_heads),
            arg(&scratch[5], geometry.query_heads),
            arg(status, 1),
            geometry.query_heads,
        );
        crate::source_attention::kernels::probabilities::launch_unchecked(
            client,
            grid(scores),
            threads(),
            arg(&scratch[3], scores),
            arg(&scratch[5], geometry.query_heads),
            arg(output[6], scores),
            arg(status, 1),
            tokens as u32,
            geometry.query_heads,
            tokens,
        );
        crate::source_attention::kernels::pv::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(output[6], scores),
            arg(output[3], tokens * kv),
            arg(output[7], q),
            arg(status, 1),
            arg(zero, 1),
            256usize,
            geometry.query_heads,
            geometry.kv_heads,
            tokens,
            tokens as u32,
            1u32,
            tokens as u32,
        );
        // Original single-query BMM: four FP32 partials beyond the small path.
        // This is not the projection/QK64-lane dot reduction.
        crate::source_sigmoid::sigmoid_shift::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(input[3], q),
            arg(scalar_table, 72),
            arg(&scratch[6], q),
            q,
        );
        crate::source_sigmoid::sigmoid::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(input[3], q),
            arg(scalar_table, 72),
            arg(&scratch[6], q),
            arg(&scratch[7], q),
            arg(status, 1),
            q,
        );
        super::transport::gate::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(output[7], q),
            arg(&scratch[7], q),
            arg(output[8], q),
            arg(status, 1),
            q,
        );
        launches += 11;
    }
    Ok(launches)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn geometry(heads: usize, kv: usize) -> TorchCpuAttentionGeometry {
        TorchCpuAttentionGeometry {
            query_heads: heads,
            kv_heads: kv,
            width: 256,
            rotary: 64,
            theta: 10_000_000.0,
        }
    }
    #[test]
    fn scratch_covers_actual_active_scores_and_scalar_sigmoid() {
        assert_eq!(
            scratch_bytes(geometry(8, 2), 0).unwrap(),
            [128, 256, 32, 32, 32, 32, 16384, 8192]
        );
        assert_eq!(
            scratch_bytes(geometry(8, 2), 47).unwrap(),
            [128, 256, 32, 1536, 32, 32, 16384, 8192]
        );
        assert_eq!(
            scratch_bytes(geometry(24, 4), 63).unwrap(),
            [128, 256, 96, 6144, 96, 96, 49152, 24576]
        );
    }
    #[test]
    fn unimplemented_geometry_is_refused_before_any_handle_or_launch() {
        for (g, pos) in [
            (geometry(8, 2), 64),
            (geometry(25, 1), 0),
            (geometry(8, 3), 0),
            (geometry(8, 0), 0),
        ] {
            assert!(scratch_bytes(g, pos).is_err());
        }
    }
}
