//! Borrowed closed source operations; the original resident loop schedules nodes.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{StandaloneExpProfile, TorchCpuGdnContract, TorchCpuGdnPhase},
};
mod attention;
mod coefficients;
mod gdn;
pub(super) mod prepared;

// RMS squares,sum,mean,denominator,sqrt,inverse,normalized,weighted;
// repeated Q/K,softplus,Aexp,log_decay,decay,beta,scalar-shift,
// decayed key-major state,memory,delta,initialQK,head weighted.
const SIZES: [usize; 29] = [
    6144 * 4,
    48 * 4,
    48 * 4,
    48 * 4,
    48 * 4,
    48 * 4,
    6144 * 4,
    6144 * 4,
    6144 * 4,
    6144 * 4,
    48 * 4,
    48 * 4,
    48 * 4,
    48 * 4,
    48 * 4,
    48 * 8,
    48 * 128 * 128 * 4,
    6144 * 4,
    6144 * 4,
    48 * 4,
    6144 * 4,
    32 * 4,
    64 * 4,
    6144 * 8,
    6144 * 4,
    24 * 4,
    48 * 4,
    24 * 4,
    24 * 4,
];
pub(super) fn scratch_layout() -> &'static [usize] {
    &SIZES
}
pub(super) fn immutable_tables() -> Vec<Vec<u8>> {
    vec![
        zerogravity_core::accelerator::program::standalone_exp::MKL_EXHA_SSE2_EXP2_128
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        crate::source_transcendentals::TABLES
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        crate::source_trig::TRIG_TABLES
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        (0..32)
            .flat_map(|i| (1.0f32 / 10_000_000f32.powf((2 * i) as f32 / 64.0)).to_le_bytes())
            .collect(),
    ]
}
pub(super) struct Scratch<'a> {
    handles: &'a [Handle],
}
impl<'a> Scratch<'a> {
    pub(super) fn new(handles: &'a [Handle]) -> Result<Self> {
        if handles.len() != SIZES.len() {
            return Err(Error::State("source math scratch count".into()));
        }
        let mut ranges = Vec::with_capacity(SIZES.len());
        for (h, n) in handles.iter().zip(SIZES) {
            check(h, n)?;
            let r = super::super::BufferRange::new(h)?;
            if h.service != handles[0].service
                || ranges
                    .iter()
                    .any(|old: &super::super::BufferRange| old.overlaps(&r))
            {
                return Err(Error::State(
                    "source math distinct original scratch owners".into(),
                ));
            }
            ranges.push(r);
        }
        Ok(Self { handles })
    }
    fn arg(&self, i: usize, n: usize) -> BufferArg {
        arg(&self.handles[i], n)
    }
}
fn check(h: &Handle, bytes: usize) -> Result<()> {
    let n = h
        .size()
        .checked_sub(h.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(h.offset_end.unwrap_or(0)));
    if bytes == 0 || n.is_none_or(|n| n < bytes as u64) {
        return Err(Error::State("source math owner range".into()));
    }
    Ok(())
}
fn arg(h: &Handle, n: usize) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(h.clone(), n) }
}
fn grid(n: usize) -> CubeCount {
    CubeCount::Static((n as u32).div_ceil(32), 1, 1)
}
fn threads() -> CubeDim {
    CubeDim::new_1d(32)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ordinary_rms(
    client: &Client,
    s: &Scratch<'_>,
    input: &Handle,
    gamma: &Handle,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
) -> Result<usize> {
    // Eligibility is this exact source contract, not the shared primitive's limits.
    if !matches!(width, 256 | 1024 | 2048 | 5120)
        || groups == 0
        || groups > 24
        || width * groups > 6144
    {
        return Err(Error::Unsupported(
            "source ordinary RMS finite shape".into(),
        ));
    }
    crate::source_rms::launch(
        client,
        &s.handles[..8],
        input,
        Some(gamma),
        output,
        status,
        zero,
        width,
        groups,
        crate::source_rms::Gamma::OnePlusRaw,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn(
    client: &Client,
    s: &Scratch<'_>,
    c: &TorchCpuGdnContract,
    input: [&Handle; 6],
    parameters: [&Handle; 4],
    output: [&Handle; 6],
    mkl_table: &Handle,
    scalar_table: &Handle,
    status: &Handle,
    zero: &Handle,
    phase: TorchCpuGdnPhase,
) -> Result<usize> {
    let (channels, values, state) = c.validate(c.convolution.source)?;
    if c.standalone_exp != StandaloneExpProfile::MklExHaSse2HaFtzOffNormalV1
        || c.key_heads > 16
        || c.value_heads > 48
    {
        return Err(Error::Unsupported(
            "source GDN explicit normal-exp arithmetic".into(),
        ));
    }
    for (h, n) in input.iter().zip([
        channels,
        values,
        c.value_heads,
        c.value_heads,
        channels * 3,
        state,
    ]) {
        check(h, n * 4)?;
    }
    for (h, n) in output
        .iter()
        .zip([channels, channels, channels * 3, state, values, values])
    {
        check(h, n * 4)?;
    }
    for (h, n) in parameters
        .iter()
        .zip([channels * 4, c.value_heads, c.value_heads, 128])
    {
        check(h, n * 4)?;
    }
    check(mkl_table, 256 * 4)?;
    check(scalar_table, 72 * 8)?;
    check(status, 4)?;
    check(zero, 4)?;
    let heads = c.value_heads;
    let kh = c.key_heads;
    let initial = phase == TorchCpuGdnPhase::InitialOne;
    let mut launches = 0;
    unsafe {
        gdn::convolution::launch_unchecked(
            client,
            grid(channels),
            threads(),
            arg(input[0], channels),
            arg(input[4], channels * 3),
            arg(parameters[0], channels * 4),
            arg(output[0], channels),
            arg(output[1], channels),
            arg(output[2], channels * 3),
            arg(status, 1),
            channels,
            initial,
        );
        launches += 1;
        coefficients::biased_softplus::launch_unchecked(
            client,
            grid(heads),
            threads(),
            arg(input[2], heads),
            arg(parameters[2], heads),
            s.arg(10, heads),
            arg(status, 1),
            heads,
        );
        crate::source_standalone_exp::normal_vector::launch_unchecked(
            client,
            grid(heads),
            threads(),
            arg(parameters[1], heads),
            arg(mkl_table, 256),
            s.arg(11, heads),
            arg(status, 1),
            heads,
        );
        coefficients::log_decay::launch_unchecked(
            client,
            grid(heads),
            threads(),
            s.arg(11, heads),
            s.arg(10, heads),
            s.arg(12, heads),
            arg(status, 1),
            heads,
        );
        crate::source_standalone_exp::normal_vector::launch_unchecked(
            client,
            grid(heads),
            threads(),
            s.arg(12, heads),
            arg(mkl_table, 256),
            s.arg(13, heads),
            arg(status, 1),
            heads,
        );
        coefficients::sigmoid_shift::launch_unchecked(
            client,
            grid(heads),
            threads(),
            arg(input[3], heads),
            arg(scalar_table, 72),
            s.arg(15, heads),
            heads,
        );
        coefficients::sigmoid::launch_unchecked(
            client,
            grid(heads),
            threads(),
            arg(input[3], heads),
            arg(scalar_table, 72),
            s.arg(15, heads),
            s.arg(14, heads),
            arg(status, 1),
            heads,
        );
        launches += 6;
        gdn::normalize_qk::launch_unchecked(
            client,
            grid(heads),
            threads(),
            arg(output[1], channels),
            s.arg(8, values),
            s.arg(9, values),
            arg(status, 1),
            kh,
            heads,
            !initial,
        );
        gdn::decayed::launch_unchecked(
            client,
            grid(state),
            threads(),
            arg(input[5], state),
            s.arg(13, heads),
            s.arg(16, state),
            arg(status, 1),
            state,
            initial,
        );
        launches += 2;
        if initial {
            gdn::initial_dot::launch_unchecked(
                client,
                grid(heads),
                threads(),
                s.arg(8, values),
                s.arg(9, values),
                s.arg(19, heads),
                arg(status, 1),
                heads,
            );
            launches += 1;
        } else {
            gdn::key_reduce::launch_unchecked(
                client,
                grid(values),
                threads(),
                s.arg(16, state),
                s.arg(9, values),
                s.arg(17, values),
                arg(status, 1),
                values,
                false,
            );
            gdn::delta::launch_unchecked(
                client,
                grid(values),
                threads(),
                arg(output[1], channels),
                s.arg(17, values),
                s.arg(14, heads),
                s.arg(18, values),
                arg(status, 1),
                kh,
                values,
            );
            launches += 2;
        }
        gdn::state_update::launch_unchecked(
            client,
            grid(state),
            threads(),
            s.arg(16, state),
            s.arg(9, values),
            s.arg(18, values),
            arg(output[1], channels),
            s.arg(14, heads),
            arg(output[3], state),
            arg(status, 1),
            kh,
            state,
            initial,
        );
        if initial {
            gdn::initial_core::launch_unchecked(
                client,
                grid(values),
                threads(),
                arg(output[1], channels),
                s.arg(14, heads),
                s.arg(19, heads),
                arg(output[4], values),
                arg(status, 1),
                kh,
                values,
            );
        } else {
            gdn::key_reduce::launch_unchecked(
                client,
                grid(values),
                threads(),
                arg(output[3], state),
                s.arg(8, values),
                arg(output[4], values),
                arg(status, 1),
                values,
                true,
            );
        }
        launches += 2;
    }
    launches += crate::source_rms::launch(
        client,
        &s.handles[..8],
        output[4],
        Some(parameters[3]),
        &s.handles[20],
        status,
        zero,
        128,
        heads,
        crate::source_rms::Gamma::DirectF32AfterBf16,
    )?;
    unsafe {
        coefficients::gated::launch_unchecked(
            client,
            grid(values),
            threads(),
            s.arg(20, values),
            arg(input[1], values),
            arg(output[5], values),
            arg(status, 1),
            values,
            c.norm.kind == zerogravity_core::quant::QuantKind::Bf16,
        );
    }
    Ok(launches + 1)
}

pub(super) fn full_convolution(
    client: &Client,
    prior: &Handle,
    minimal: &Handle,
    input: &Handle,
    output: &Handle,
    status: &Handle,
    channels: usize,
) -> Result<usize> {
    if channels == 0 || channels > 10240 {
        return Err(Error::Unsupported("source Conv4 channels".into()));
    }
    for (h, n) in [
        (prior, channels * 4),
        (minimal, channels * 3),
        (input, channels),
        (output, channels * 4),
        (status, 1),
    ] {
        check(h, n * 4)?;
    }
    unsafe {
        gdn::full_convolution::launch_unchecked(
            client,
            grid(channels),
            threads(),
            arg(prior, channels * 4),
            arg(minimal, channels * 3),
            arg(input, channels),
            arg(output, channels * 4),
            arg(status, 1),
            channels,
        );
    }
    Ok(1)
}

/// Original per-token eager BF16 attention. Source contract fixes width256,
/// rotary64/theta1e7, QK BF16 then1/16, Vec8 softmax, and source BF16 PV.
#[allow(clippy::too_many_arguments)]
pub(super) fn attention(
    client: &Client,
    s: &Scratch<'_>,
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
    if position > 1 || input.len() != if position == 0 { 4 } else { 6 } {
        return Err(Error::Unsupported(
            "finite INITIAL1/cached1 source attention".into(),
        ));
    }
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
            s.arg(21, 32),
            position as u32,
            32usize,
        );
        crate::source_trig::sincos_angles::launch_unchecked(
            client,
            grid(32),
            threads(),
            s.arg(21, 32),
            arg(trig_table, 35),
            s.arg(22, 64),
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
                s.arg(22, 64),
                arg(rotated, count),
                arg(status, 1),
                256usize,
                count,
                64usize,
            );
            launches += 2;
        }
        attention::append::launch_unchecked(
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
            s.arg(25, geometry.query_heads),
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
            s.arg(25, geometry.query_heads),
            s.arg(26, scores),
            arg(status, 1),
            tokens as u32,
            geometry.query_heads,
            tokens,
        );
        crate::source_attention::kernels::reduce::launch_unchecked(
            client,
            CubeCount::Static(geometry.query_heads as u32, 1, 1),
            threads(),
            s.arg(26, scores),
            s.arg(27, geometry.query_heads),
            arg(status, 1),
            tokens as u32,
            false,
            tokens,
        );
        crate::source_attention::kernels::inverse::launch_unchecked(
            client,
            grid(geometry.query_heads),
            threads(),
            s.arg(27, geometry.query_heads),
            s.arg(28, geometry.query_heads),
            arg(status, 1),
            geometry.query_heads,
        );
        crate::source_attention::kernels::probabilities::launch_unchecked(
            client,
            grid(scores),
            threads(),
            s.arg(26, scores),
            s.arg(28, geometry.query_heads),
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
        // Contraction1/2 here equals the original dot's scalar tail. The shared
        // PV kernel's broader BMM schedule grants no larger Qwen contraction.
        coefficients::sigmoid_shift::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(input[3], q),
            arg(scalar_table, 72),
            s.arg(23, q),
            q,
        );
        coefficients::sigmoid::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(input[3], q),
            arg(scalar_table, 72),
            s.arg(23, q),
            s.arg(24, q),
            arg(status, 1),
            q,
        );
        attention::gate::launch_unchecked(
            client,
            grid(q),
            threads(),
            arg(output[7], q),
            s.arg(24, q),
            arg(output[8], q),
            arg(status, 1),
            q,
        );
        launches += 11;
    }
    Ok(launches)
}

pub(super) fn silu_mul(
    client: &Client,
    gate: &Handle,
    up: &Handle,
    output: &Handle,
    status: &Handle,
    elements: usize,
) -> Result<usize> {
    if elements == 0 || elements > 32768 || !elements.is_multiple_of(32) {
        return Err(Error::Unsupported(
            "source finite complete SiLU vector".into(),
        ));
    }
    for h in [gate, up, output] {
        check(h, elements * 4)?;
    }
    check(status, 4)?;
    unsafe {
        coefficients::ffn_silu_mul::launch_unchecked(
            client,
            grid(elements),
            threads(),
            arg(gate, elements),
            arg(up, elements),
            arg(output, elements),
            arg(status, 1),
            elements,
        );
    }
    Ok(1)
}
