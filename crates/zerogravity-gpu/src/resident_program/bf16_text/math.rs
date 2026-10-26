//! Borrowed arithmetic lowerers for the separately sealed E2B candidate.
//! No allocation, source/state authority, checkpoint, completion or promotion.
//! Every handle remains owned by the original admitted parent flight. Its
//! binder rejects operand/output/scratch aliases and supplies immutable +0.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{
    Error, Result,
    accelerator::program::{TextElementwise, TextRotary},
};

mod attention;
mod elementwise;
mod float;
pub(super) mod prepared;
mod rms;
mod rotary;
#[cfg(test)]
mod tests;
mod transcendentals;

const SIZES: [usize; 14] = [
    8960 * 4,
    35 * 4,
    35 * 4,
    35 * 4,
    35 * 4,
    35 * 4,
    8960 * 4,
    8960 * 4,
    512 * 4,
    8 * 4,
    8 * 4,
    8 * 4,
    256 * 4,
    512 * 4,
];
/// Logical bytes per distinct owner; parent separately applies physical alignment.
pub(super) fn scratch_layout() -> Vec<usize> {
    SIZES.to_vec()
}
/// Source-defined RoPE constants only. No input activation, sin/cos, expected
/// word, learned parameter or callback-derived value enters these tables.
/// Order: local inverse frequency128F32, full inverse frequency256F32, trig35F64.
pub(super) fn immutable_tables() -> Vec<Vec<u8>> {
    let local = (0..128)
        .flat_map(|i| (1.0f32 / 10000.0f32.powf((2 * i) as f32 / 256.0)).to_le_bytes())
        .collect();
    let full = (0..256)
        .flat_map(|i| {
            let x = if i < 64 {
                1.0f32 / 1000000.0f32.powf((2 * i) as f32 / 512.0)
            } else {
                0.0
            };
            x.to_le_bytes()
        })
        .collect();
    let trig = crate::source_trig::TRIG_TABLES
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    vec![local, full, trig]
}
pub(super) struct MathScratch<'a> {
    handles: &'a [Handle],
}
impl<'a> MathScratch<'a> {
    pub(super) fn new(handles: &'a [Handle]) -> Result<Self> {
        if handles.len() != SIZES.len() {
            return Err(Error::State(
                "BF16 text exact math scratch owner count".into(),
            ));
        }
        let mut ranges = Vec::with_capacity(SIZES.len());
        for (handle, bytes) in handles.iter().zip(SIZES) {
            check(handle, bytes)?;
            let range = crate::resident_program::BufferRange::new(handle)?;
            if ranges
                .iter()
                .any(|old: &crate::resident_program::BufferRange| old.overlaps(&range))
                || handle.service != handles[0].service
            {
                return Err(Error::State(
                    "BF16 text independent same-service math scratch owners".into(),
                ));
            }
            ranges.push(range);
        }
        Ok(Self { handles })
    }
    fn arg(&self, i: usize, elements: usize) -> BufferArg {
        arg(&self.handles[i], elements)
    }
}
fn check(h: &Handle, bytes: usize) -> Result<()> {
    let available = h
        .size()
        .checked_sub(h.offset_start.unwrap_or(0))
        .and_then(|v| v.checked_sub(h.offset_end.unwrap_or(0)));
    if bytes == 0 || available.is_none_or(|v| v < bytes as u64) {
        return Err(Error::State(
            "BF16 text arithmetic borrowed buffer range".into(),
        ));
    }
    Ok(())
}
fn words(h: &Handle, n: usize) -> Result<()> {
    check(
        h,
        n.checked_mul(4)
            .ok_or_else(|| Error::Limit("BF16 text arithmetic byte overflow".into()))?,
    )
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
pub(super) fn rms(
    client: &Client,
    s: &MathScratch<'_>,
    input: &Handle,
    gamma: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
) -> Result<usize> {
    let count = width
        .checked_mul(groups)
        .ok_or_else(|| Error::Limit("BF16 text RMS geometry overflow".into()))?;
    if !matches!(
        (width, groups),
        (256 | 512 | 1536, 1) | (256 | 512, 8) | (256, 35)
    ) || gamma.is_none() && !matches!((width, groups), (256 | 512, 1))
        || count > 8960
    {
        return Err(Error::Unsupported(
            "BF16 text direct/grouped source RMS extent".into(),
        ));
    }
    crate::source_rms::launch(
        client,
        &s.handles[..8],
        input,
        gamma,
        output,
        status,
        zero,
        width,
        groups,
        if gamma.is_none() {
            crate::source_rms::Gamma::Unit
        } else {
            crate::source_rms::Gamma::DirectBf16
        },
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn elementwise(
    client: &Client,
    input: &Handle,
    other: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    count: usize,
    op: TextElementwise,
) -> Result<usize> {
    op.validate(if other.is_some() { 2 } else { 1 })?;
    if count == 0 || count > 262144 {
        return Err(Error::Limit("BF16 text elementwise finite extent".into()));
    }
    let (kind, scalar) = match op {
        TextElementwise::Scale(x) => (0, x),
        TextElementwise::ScaleBf16Scalar(x) => (1, x),
        TextElementwise::Add => (2, 0.0),
        TextElementwise::Multiply => (3, 0.0),
        TextElementwise::GeluTanh => (4, 0.0),
        TextElementwise::TanhSoftcap(x) => (5, x),
    };
    if kind >= 4 && !count.is_multiple_of(16) {
        return Err(Error::Unsupported(
            "BF16 text source complete BF16 vector spans".into(),
        ));
    }
    words(input, count)?;
    words(output, count)?;
    words(status, 1)?;
    if let Some(other) = other {
        words(other, count)?;
    }
    unsafe {
        elementwise::operation::launch_unchecked(
            client,
            grid(count),
            threads(),
            arg(input, count),
            arg(other.unwrap_or(input), count),
            arg(output, count),
            arg(status, 1),
            scalar,
            kind,
            count,
        );
    }
    Ok(1)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn rotary(
    client: &Client,
    s: &MathScratch<'_>,
    tables: &[Handle],
    input: &Handle,
    output: &Handle,
    status: &Handle,
    count: usize,
    position: usize,
    g: TextRotary,
) -> Result<usize> {
    g.validate(count)?;
    if position >= 64
        || tables.len() != 3
        || !matches!(count / g.head_width, 1 | 8)
        || !((g.head_width == 256
            && g.rotated_pairs == 128
            && g.theta.to_bits() == 10000.0f32.to_bits())
            || (g.head_width == 512
                && g.rotated_pairs == 64
                && g.theta.to_bits() == 1000000.0f32.to_bits()))
    {
        return Err(Error::Unsupported(
            "BF16 text source split-half rotary geometry".into(),
        ));
    }
    let half = g.head_width / 2;
    let table = usize::from(g.head_width == 512);
    words(input, count)?;
    words(output, count)?;
    words(status, 1)?;
    check(&tables[table], half * 4)?;
    check(&tables[2], 35 * 8)?;
    unsafe {
        rotary::angles::launch_unchecked(
            client,
            grid(half),
            threads(),
            arg(&tables[table], half),
            s.arg(12, half),
            position as u32,
            half,
        );
        crate::source_trig::sincos_angles::launch_unchecked(
            client,
            grid(half),
            threads(),
            s.arg(12, half),
            arg(&tables[2], 35),
            s.arg(13, 2 * half),
            half,
        );
        rotary::apply::launch_unchecked(
            client,
            grid(count / 2),
            threads(),
            arg(input, count),
            s.arg(13, 2 * half),
            arg(output, count),
            arg(status, 1),
            g.head_width,
            count,
            g.head_width,
        );
    }
    Ok(3)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn qk(
    client: &Client,
    input: &Handle,
    keys: &Handle,
    dots: &Handle,
    scores: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    active: usize,
    window: Option<usize>,
) -> Result<usize> {
    if !matches!(width, 256 | 512) || !(1..=64).contains(&active) || window == Some(0) {
        return Err(Error::Unsupported(
            "BF16 text unit QK width/current prefix/window".into(),
        ));
    }
    words(input, 8 * width)?;
    words(keys, 64 * width)?;
    words(dots, 512)?;
    words(scores, 512)?;
    words(status, 1)?;
    words(zero, 1)?;
    let first = window.map_or(0, |w| active.saturating_sub(w));
    unsafe {
        attention::qk::launch_unchecked(
            client,
            CubeCount::Static(512, 1, 1),
            CubeDim::new_1d(64),
            arg(input, 8 * width),
            arg(keys, 64 * width),
            arg(dots, 512),
            arg(scores, 512),
            arg(status, 1),
            arg(zero, 1),
            width,
            8usize,
            1usize,
            64usize,
            1.0f32,
            active as u32,
            first as u32,
        );
    }
    Ok(1)
}
pub(super) fn softmax(
    client: &Client,
    s: &MathScratch<'_>,
    input: &Handle,
    output: &Handle,
    status: &Handle,
    active: usize,
) -> Result<usize> {
    if !(1..=64).contains(&active) {
        return Err(Error::Unsupported(
            "BF16 text source softmax active extent".into(),
        ));
    }
    words(input, 512)?;
    words(output, 512)?;
    words(status, 1)?;
    unsafe {
        attention::reduce::launch_unchecked(
            client,
            CubeCount::Static(8, 1, 1),
            threads(),
            arg(input, 512),
            s.arg(9, 8),
            arg(status, 1),
            active as u32,
            true,
            64usize,
        );
        attention::exponentials::launch_unchecked(
            client,
            grid(512),
            threads(),
            arg(input, 512),
            s.arg(9, 8),
            s.arg(8, 512),
            arg(status, 1),
            active as u32,
            8usize,
            64usize,
        );
        attention::reduce::launch_unchecked(
            client,
            CubeCount::Static(8, 1, 1),
            threads(),
            s.arg(8, 512),
            s.arg(10, 8),
            arg(status, 1),
            active as u32,
            false,
            64usize,
        );
        attention::inverse::launch_unchecked(
            client,
            grid(8),
            threads(),
            s.arg(10, 8),
            s.arg(11, 8),
            arg(status, 1),
            8usize,
        );
        attention::probabilities::launch_unchecked(
            client,
            grid(512),
            threads(),
            s.arg(8, 512),
            s.arg(11, 8),
            arg(output, 512),
            arg(status, 1),
            active as u32,
            8usize,
            64usize,
        );
    }
    Ok(5)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn pv(
    client: &Client,
    probabilities: &Handle,
    values: &Handle,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    active: usize,
    query_rows: usize,
    contraction: usize,
) -> Result<usize> {
    if !matches!(width, 256 | 512)
        || !(1..=64).contains(&contraction)
        || !(1..=contraction).contains(&query_rows)
        || !(1..=contraction).contains(&active)
    {
        return Err(Error::Unsupported(
            "BF16 text PV authentic full-call geometry".into(),
        ));
    }
    words(probabilities, 512)?;
    words(values, 64 * width)?;
    words(output, 8 * width)?;
    words(status, 1)?;
    words(zero, 1)?;
    unsafe {
        attention::pv::launch_unchecked(
            client,
            grid(8 * width),
            threads(),
            arg(probabilities, 512),
            arg(values, 64 * width),
            arg(output, 8 * width),
            arg(status, 1),
            arg(zero, 1),
            width,
            8usize,
            1usize,
            64usize,
            active as u32,
            query_rows as u32,
            contraction as u32,
        );
    }
    Ok(1)
}
