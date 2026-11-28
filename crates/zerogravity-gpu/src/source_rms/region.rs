//! Source-exact row region; caller retains the original resident flight.
use super::*;

/// Two launches irrespective of head-row count. Reuse one admitted inverse
/// arena and the existing source-exact normalize/gamma/store tail.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch(
    client: &Client,
    inverse: &Handle,
    input: RmsInput<'_>,
    gamma: Option<&Handle>,
    output: &Handle,
    status: &Handle,
    zero: &Handle,
    width: usize,
    groups: usize,
    mode: Gamma,
) -> Result<usize> {
    // Width shares the exact-source contract; unlike the staged path the
    // region needs one scalar per row rather than a tile of eight vectors.
    scratch_words(width, 1)?;
    let count = width
        .checked_mul(groups)
        .filter(|&n| n > 0 && u32::try_from(n).is_ok())
        .ok_or_else(|| Error::Limit("RMS region row extent".into()))?;
    let max_grid = client.properties().hardware.max_cube_count.0;
    let tail_grid = (count as u32).div_ceil(32);
    if groups > max_grid as usize || tail_grid > max_grid {
        return Err(Error::Limit("RMS region grid extent".into()));
    }
    validate_bindings(
        std::slice::from_ref(inverse),
        &[groups],
        input,
        gamma,
        output,
        status,
        zero,
        width,
        count,
        mode,
    )?;
    let dim = CubeDim::new_1d(32);
    let grid = CubeCount::Static(groups as u32, 1, 1);
    unsafe {
        match input {
            RmsInput::Direct(input) => kernels::region_inverse::launch_unchecked(
                client,
                grid,
                dim,
                arg(input, count),
                arg(zero, 1),
                arg(inverse, groups),
                arg(status, 1),
                width,
            ),
            RmsInput::Residual {
                input,
                prior,
                residual,
            } => kernels::region_residual_inverse::launch_unchecked(
                client,
                grid,
                dim,
                arg(input, count),
                arg(prior, count),
                arg(residual, count),
                arg(zero, 1),
                arg(inverse, groups),
                arg(status, 1),
                width,
            ),
        };
        kernels::normalize_weight_store::launch_unchecked(
            client,
            CubeCount::Static(tail_grid, 1, 1),
            dim,
            arg(input.normalized(), count),
            arg(inverse, groups),
            arg(
                gamma.unwrap_or(zero),
                if gamma.is_some() { width } else { 1 },
            ),
            arg(output, count),
            arg(status, 1),
            width,
            mode.mode(),
        );
    }
    Ok(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical HIP source-exact RMS region differential"]
    fn region_matches_staged_rms_on_hip() -> Result<()> {
        let client = cubecl::Device::Hip(Default::default()).client();
        // Include tiny/subnormal values, both signs of zero, ordinary values
        // and cancellation in the separately rounded residual.
        let pattern: [u32; 13] = [
            0, 0x80000000, 0x00010000, 0x80010000, 0x00800000, 0x1f000000, 0x9f000000, 0x3c800000,
            0xbc800000, 0x3f800000, 0xbf800000, 0x40800000, 0xc0800000,
        ];
        let upload = |v: &[f32]| client.create_from_slice(f32::as_bytes(v));
        let zero = upload(&[0.0]);
        let read = |h: Handle| {
            client
                .read_one(h)
                .map_err(|e| Error::State(format!("RMS region readback: {e}")))
        };
        // One test exercises the whole shared region at actual source widths,
        // including more than the staged path's 64-group tile.
        for (width, groups, residual, mode) in [
            (128, 129, false, Gamma::DirectF32AfterBf16),
            (256, 65, false, Gamma::OnePlusRaw),
            (1024, 1, true, Gamma::OnePlusRaw),
            (2048, 17, true, Gamma::OnePlusRaw),
            (5120, 2, true, Gamma::OnePlusRaw),
        ] {
            let n = width * groups;
            let input: Vec<_> = (0..n)
                .map(|i| f32::from_bits(pattern[(i * 7 + i / width) % pattern.len()]))
                .collect();
            let prior: Vec<_> = input
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    if i % 3 == 0 {
                        -x
                    } else {
                        f32::from_bits(pattern[(i * 3 + 2) % pattern.len()])
                    }
                })
                .collect();
            let gamma: Vec<_> = (0..width)
                .map(|i| (i % 11) as f32 / 128.0 - 0.125)
                .collect();
            let gamma = upload(&gamma);
            let source = upload(&input);
            let previous = upload(&prior);
            let residual_out = client.empty(n * 4);
            let output = client.empty(n * 4);
            let inv = client.empty(groups * 4);
            let status = client.create_from_slice(u32::as_bytes(&[0]));
            let binding = if residual {
                RmsInput::Residual {
                    input: &source,
                    prior: &previous,
                    residual: &residual_out,
                }
            } else {
                RmsInput::Direct(&source)
            };
            assert_eq!(
                launch(
                    &client,
                    &inv,
                    binding,
                    Some(&gamma),
                    &output,
                    &status,
                    &zero,
                    width,
                    groups,
                    mode
                )?,
                2
            );
            let actual = read(output)?;
            assert_eq!(u32::from_bytes(&read(status)?), &[0]);
            let actual_residual = if residual {
                Some(read(residual_out)?)
            } else {
                None
            };
            for first in (0..groups).step_by(MAX_GROUPS) {
                let rows = (groups - first).min(MAX_GROUPS);
                let lo = first * width;
                let hi = lo + rows * width;
                let staged_input = upload(&input[lo..hi]);
                let staged_prior = upload(&prior[lo..hi]);
                let staged_residual = client.empty((hi - lo) * 4);
                let staged_output = client.empty((hi - lo) * 4);
                let scratch: Vec<_> = scratch_words(width, rows)?
                    .iter()
                    .map(|n| client.empty(n * 4))
                    .collect();
                let status = client.create_from_slice(u32::as_bytes(&[0]));
                let binding = if residual {
                    RmsInput::Residual {
                        input: &staged_input,
                        prior: &staged_prior,
                        residual: &staged_residual,
                    }
                } else {
                    RmsInput::Direct(&staged_input)
                };
                launch_input(
                    &client,
                    &scratch,
                    binding,
                    Some(&gamma),
                    &staged_output,
                    &status,
                    &zero,
                    width,
                    rows,
                    mode,
                    false,
                )?;
                assert_eq!(u32::from_bytes(&read(status)?), &[0]);
                assert_eq!(
                    &actual[lo * 4..hi * 4],
                    &read(staged_output)?[..],
                    "width={width} groups={groups} first={first}"
                );
                if let Some(r) = &actual_residual {
                    assert_eq!(&r[lo * 4..hi * 4], &read(staged_residual)?[..]);
                }
            }
        }
        // An invalid widened BF16 input must still poison the shared status.
        let mut input = vec![1.0f32; 128];
        input[31] = f32::from_bits(0x3f800001);
        let input = upload(&input);
        let status = client.create_from_slice(u32::as_bytes(&[0]));
        launch(
            &client,
            &client.empty(4),
            RmsInput::Direct(&input),
            None,
            &client.empty(128 * 4),
            &status,
            &zero,
            128,
            1,
            Gamma::Unit,
        )?;
        assert_ne!(u32::from_bytes(&read(status)?), &[0]);
        client
            .memory_cleanup()
            .map_err(|e| Error::State(format!("RMS region cleanup: {e}")))?;
        Ok(())
    }
}
