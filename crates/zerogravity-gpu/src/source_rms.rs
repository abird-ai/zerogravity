//! Borrowed cascade FP32 RMS; explicit source gamma/cast contract.
//! Caller retains authentic source, admitted owners and the original flight.
use cubecl::{client::Client, prelude::*, server::Handle};
use zerogravity_core::{Error, Result};
pub(crate) mod kernels;
pub(crate) const MAX_GROUPS: usize = 64;
#[cfg(all(feature = "hip-source", feature = "hip"))]
pub(crate) mod prepared;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Gamma {
    Unit,
    #[cfg(feature = "gemma-bf16-qualification")]
    DirectBf16,
    #[cfg(feature = "hip-source")]
    OnePlusRaw,
    DirectF32AfterBf16,
}
impl Gamma {
    pub(crate) fn mode(self) -> u32 {
        match self {
            Self::Unit => 0,
            #[cfg(feature = "gemma-bf16-qualification")]
            Self::DirectBf16 => 1,
            #[cfg(feature = "hip-source")]
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
pub(crate) fn scratch_words(width: usize, groups: usize) -> Result<[usize; 8]> {
    if width == 0
        || width > 5120
        || !width.is_multiple_of(32)
        || !(1..=MAX_GROUPS).contains(&groups)
    {
        return Err(Error::Unsupported("RMS explicit source geometry".into()));
    }
    let count = width * groups;
    Ok([count, groups, groups, groups, groups, groups, count, count])
}
#[derive(Clone, Copy)]
enum RmsInput<'a> {
    Direct(&'a Handle),
    Residual {
        input: &'a Handle,
        prior: &'a Handle,
        residual: &'a Handle,
    },
}
impl<'a> RmsInput<'a> {
    fn normalized(self) -> &'a Handle {
        match self {
            Self::Direct(input) => input,
            Self::Residual { residual, .. } => residual,
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
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
    launch_input(
        client,
        scratch,
        RmsInput::Direct(input),
        gamma,
        output,
        status,
        zero,
        width,
        groups,
        mode,
        false,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_residual(
    client: &Client,
    scratch: &[Handle],
    input: &Handle,
    prior: &Handle,
    residual: &Handle,
    gamma: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
    mode: Gamma,
) -> Result<usize> {
    launch_input(
        client,
        scratch,
        RmsInput::Residual {
            input,
            prior,
            residual,
        },
        gamma,
        output,
        status,
        zero,
        width,
        groups,
        mode,
        false,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_fused_tail(
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
    launch_input(
        client,
        scratch,
        RmsInput::Direct(input),
        gamma,
        output,
        status,
        zero,
        width,
        groups,
        mode,
        true,
    )
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(crate) fn launch_residual_fused_tail(
    client: &Client,
    scratch: &[Handle],
    input: &Handle,
    prior: &Handle,
    residual: &Handle,
    gamma: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
    mode: Gamma,
) -> Result<usize> {
    launch_input(
        client,
        scratch,
        RmsInput::Residual {
            input,
            prior,
            residual,
        },
        gamma,
        output,
        status,
        zero,
        width,
        groups,
        mode,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_input(
    client: &Client,
    scratch: &[Handle],
    input: RmsInput<'_>,
    gamma: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
    mode: Gamma,
    fused_tail: bool,
) -> Result<usize> {
    let extents = scratch_words(width, groups)?;
    let count = extents[0];
    if scratch.len() != 8 || (mode == Gamma::Unit) != gamma.is_none() {
        return Err(Error::Unsupported(
            "RMS explicit source geometry/gamma".into(),
        ));
    }
    for (h, n) in scratch.iter().zip(extents) {
        check(h, n)?;
    }
    match input {
        RmsInput::Direct(input) => check(input, count)?,
        RmsInput::Residual {
            input,
            prior,
            residual,
        } => {
            check(input, count)?;
            check(prior, count)?;
            check(residual, count)?;
        }
    }
    check(output, count)?;
    check(status, 1)?;
    check(zero, 1)?;
    if let Some(g) = gamma {
        check(g, width)?;
    }

    let normalized = input.normalized();
    let mut unique = Vec::with_capacity(16);
    let mut require_distinct = |handle: &Handle| -> Result<()> {
        let range = crate::resident_program::BufferRange::new(handle)?;
        if handle.service != normalized.service
            || unique
                .iter()
                .any(|old: &crate::resident_program::BufferRange| old.overlaps(&range))
        {
            return Err(Error::State(
                "RMS distinct same-service borrowed owners".into(),
            ));
        }
        unique.push(range);
        Ok(())
    };
    for handle in scratch {
        require_distinct(handle)?;
    }
    match input {
        RmsInput::Direct(input) => require_distinct(input)?,
        RmsInput::Residual {
            input,
            prior,
            residual,
        } => {
            require_distinct(input)?;
            require_distinct(prior)?;
            require_distinct(residual)?;
        }
    }
    for handle in [output, status, zero] {
        require_distinct(handle)?;
    }
    if let Some(gamma) = gamma {
        require_distinct(gamma)?;
    }

    let dim = CubeDim::new_1d(32);
    let grid = |n: usize| CubeCount::Static((n as u32).div_ceil(32), 1, 1);
    unsafe {
        match input {
            RmsInput::Direct(input) => kernels::squares::launch_unchecked(
                client,
                grid(count),
                dim,
                arg(input, count),
                arg(&scratch[0], count),
                arg(status, 1),
            ),
            RmsInput::Residual {
                input,
                prior,
                residual,
            } => kernels::residual_squares::launch_unchecked(
                client,
                grid(count),
                dim,
                arg(input, count),
                arg(prior, count),
                arg(residual, count),
                arg(&scratch[0], count),
                arg(status, 1),
            ),
        }
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
            );
        }
        if fused_tail {
            kernels::normalize_weight_store::launch_unchecked(
                client,
                grid(count),
                dim,
                arg(normalized, count),
                arg(&scratch[5], groups),
                arg(
                    gamma.unwrap_or(zero),
                    if gamma.is_some() { width } else { 1 },
                ),
                arg(output, count),
                arg(status, 1),
                width,
                mode.mode(),
            );
        } else {
            kernels::normalize::launch_unchecked(
                client,
                grid(count),
                dim,
                arg(normalized, count),
                arg(&scratch[5], groups),
                arg(&scratch[6], count),
                arg(status, 1),
                width,
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
                );
            }
        }
    }
    Ok(if fused_tail {
        7
    } else if mode == Gamma::DirectF32AfterBf16 {
        8
    } else {
        9
    })
}

#[cfg(test)]
mod tests {
    use super::scratch_words;
    #[test]
    fn runtime_rows_price_distinct_vector_and_scalar_scratch() {
        for width in [32, 128, 256, 1024, 2048, 5120] {
            for groups in [1, 2, 8, 17, 47, 62, 64] {
                let words = width * groups;
                assert_eq!(
                    scratch_words(width, groups).unwrap(),
                    [words, groups, groups, groups, groups, groups, words, words]
                );
            }
        }
        for (width, groups) in [
            (0, 1),
            (33, 1),
            (5152, 1),
            (128, 0),
            (128, 65),
            (usize::MAX, 64),
            (32, usize::MAX),
        ] {
            assert!(scratch_words(width, groups).is_err());
        }
    }
}
