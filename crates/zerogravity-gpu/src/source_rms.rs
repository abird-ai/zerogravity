//! Borrowed cascade FP32 RMS; explicit source gamma/cast contract.
//! Caller retains authentic source, admitted owners and the original flight.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};
pub(crate) mod kernels;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gamma {
    Unit,
    #[cfg(feature = "gemma-bf16-qualification")]
    DirectBf16,
    #[cfg(feature = "qwen-source-qualification")]
    OnePlusRaw,
    DirectF32AfterBf16,
}
impl Gamma {
    pub(crate) fn mode(self) -> u32 {
        match self {
            Self::Unit => 0,
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::DirectBf16 => 1,
            #[cfg(feature = "qwen-source-qualification")]
            Self::OnePlusRaw => 2,
            Self::DirectF32AfterBf16 => 3,
        }
    }
}
fn arg(handle: &Handle, elements: usize) -> BufferArg {
    // Parent lowerer checks original leases/accesses; range checked below.
    unsafe { BufferArg::from_raw_parts(handle.clone(), elements) }
}
fn check(handle: &Handle, elements: usize) -> Result<()> {
    let bytes = elements
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("RMS byte extent".into()))?;
    let n = handle
        .size()
        .checked_sub(handle.offset_start.unwrap_or(0))
        .and_then(|n| n.checked_sub(handle.offset_end.unwrap_or(0)));
    if bytes == 0 || n.is_none_or(|n| n < bytes as u64) {
        return Err(Error::State("RMS borrowed owner range".into()));
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &Client,
    scratch: &[Handle],
    input: &Handle,
    gamma: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
    mode: Gamma,
) -> Result<usize> {
    let count = width
        .checked_mul(groups)
        .ok_or_else(|| Error::Limit("RMS geometry".into()))?;
    if width == 0
        || width > 5120
        || !width.is_multiple_of(32)
        || groups == 0
        || groups > 64
        || scratch.len() != 8
        || (mode == Gamma::Unit) != gamma.is_none()
    {
        return Err(Error::Unsupported(
            "RMS explicit source geometry/gamma".into(),
        ));
    }
    let extents = [count, groups, groups, groups, groups, groups, count, count];
    for (h, n) in scratch.iter().zip(extents) {
        check(h, n)?;
    }
    check(input, count)?;
    check(output, count)?;
    check(status, 1)?;
    check(zero, 1)?;
    if let Some(g) = gamma {
        check(g, width)?;
    }
    let mut unique = Vec::with_capacity(13);
    for handle in scratch
        .iter()
        .chain([input, output, status, zero])
        .chain(gamma)
    {
        let range = crate::resident_program::BufferRange::new(handle)?;
        if handle.service != input.service
            || unique
                .iter()
                .any(|old: &crate::resident_program::BufferRange| old.overlaps(&range))
        {
            return Err(Error::State(
                "RMS distinct same-service borrowed owners".into(),
            ));
        }
        unique.push(range);
    }
    let dim = CubeDim::new_1d(32);
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    unsafe {
        kernels::squares::launch_unchecked(
            client,
            grid(count),
            dim,
            arg(input, count),
            arg(&scratch[0], count),
            arg(status, 1),
            count,
        );
        kernels::sum::launch_unchecked(
            client,
            CubeCount::Static(groups as u32, 1, 1),
            dim,
            arg(&scratch[0], count),
            arg(zero, 1),
            arg(&scratch[1], groups),
            width,
        );
        for stage in 0..4u32 {
            kernels::scalar::launch_unchecked(
                client,
                grid(groups),
                dim,
                arg(&scratch[stage as usize + 1], groups),
                arg(&scratch[stage as usize + 2], groups),
                arg(status, 1),
                stage,
                width,
                groups,
            );
        }
        kernels::normalize::launch_unchecked(
            client,
            grid(count),
            dim,
            arg(input, count),
            arg(&scratch[5], groups),
            arg(&scratch[6], count),
            arg(status, 1),
            width,
            count,
        );
        let weighted = if mode == Gamma::DirectF32AfterBf16 {
            output
        } else {
            &scratch[7]
        };
        kernels::weight::launch_unchecked(
            client,
            grid(count),
            dim,
            arg(&scratch[6], count),
            arg(
                gamma.unwrap_or(zero),
                if gamma.is_some() { width } else { 1 },
            ),
            arg(weighted, count),
            arg(status, 1),
            width,
            count,
            mode.mode(),
        );
        if mode != Gamma::DirectF32AfterBf16 {
            kernels::store::launch_unchecked(
                client,
                grid(count),
                dim,
                arg(&scratch[7], count),
                arg(output, count),
                arg(status, 1),
                count,
            );
        }
    }
    Ok(if mode == Gamma::DirectF32AfterBf16 {
        8
    } else {
        9
    })
}
