//! Finite CPU signal/encoder primitives. Callers price plan, input, scratch
//! and output storage in their original resource domain.
use crate::{Error, Result, ops::encoder::check_cancel};
use std::sync::atomic::AtomicBool;

fn invalid(message: &'static str) -> Error {
    Error::InvalidInput(message.into())
}

/// Reusable radix-2 FFT plan for real signal magnitude. Twiddles and bit
/// reversal are prepared once; repeated frames reuse caller-owned scratch.
pub struct MagnitudeFft {
    size: usize,
    reversed: Vec<usize>,
    twiddles: Vec<[f64; 2]>,
}
impl MagnitudeFft {
    pub fn new(size: usize) -> Result<Self> {
        if !size.is_power_of_two() || !(2..=16384).contains(&size) {
            return Err(invalid("FFT power-of-two extent"));
        }
        let shift = usize::BITS - size.trailing_zeros();
        Ok(Self {
            size,
            reversed: (0..size).map(|n| n.reverse_bits() >> shift).collect(),
            twiddles: (0..size / 2)
                .map(|i| {
                    let phase = -std::f64::consts::TAU * i as f64 / size as f64;
                    let (sin, cos) = phase.sin_cos();
                    [cos, sin]
                })
                .collect(),
        })
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn magnitude(
        &self,
        signal: &[f32],
        scratch: &mut [[f64; 2]],
        output: &mut [f64],
    ) -> Result<()> {
        if signal.len() > self.size
            || scratch.len() != self.size
            || output.len() != self.size / 2 + 1
            || !signal.iter().all(|v| v.is_finite())
        {
            return Err(invalid("FFT input/scratch/output extent"));
        }
        for (&index, slot) in self.reversed.iter().zip(0..self.size) {
            scratch[index] = [signal.get(slot).copied().unwrap_or(0.) as f64, 0.];
        }
        let mut width = 2;
        while width <= self.size {
            let half = width / 2;
            let step = self.size / width;
            for block in scratch.chunks_exact_mut(width) {
                let (left, right) = block.split_at_mut(half);
                for (i, (a, b)) in left.iter_mut().zip(right).enumerate() {
                    let [wr, wi] = self.twiddles[i * step];
                    let [br, bi] = *b;
                    let [ar, ai] = *a;
                    let r = wr * br - wi * bi;
                    let im = wr * bi + wi * br;
                    *a = [ar + r, ai + im];
                    *b = [ar - r, ai - im];
                }
            }
            width *= 2;
        }
        for (value, &[r, i]) in output.iter_mut().zip(scratch.iter()) {
            *value = r.hypot(i);
        }
        Ok(())
    }
}

/// NHWC activation geometry, OIHW learned kernel, no bias. Reference 3x3
/// cross-correlation with stride 2 and one zero cell on every edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageRows {
    pub height: usize,
    pub width: usize,
    pub channels: usize,
}
impl ImageRows {
    pub fn elements(self) -> Option<usize> {
        self.height
            .checked_mul(self.width)?
            .checked_mul(self.channels)
    }
    pub fn subsampled(self, channels: usize) -> Self {
        Self {
            height: self.height.div_ceil(2),
            width: self.width.div_ceil(2),
            channels,
        }
    }
}
pub fn subsample_conv2d(
    input: &[f32],
    shape: ImageRows,
    weights: &[f32],
    channels: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<(Vec<f32>, ImageRows)> {
    if shape.height == 0
        || shape.width == 0
        || shape.channels == 0
        || channels == 0
        || shape.elements() != Some(input.len())
        || shape
            .channels
            .checked_mul(channels)
            .and_then(|n| n.checked_mul(9))
            != Some(weights.len())
    {
        return Err(invalid("subsample convolution geometry"));
    }
    let out_shape = shape.subsampled(channels);
    let extent = out_shape
        .elements()
        .ok_or_else(|| invalid("convolution output extent"))?;
    let mut output = Vec::with_capacity(extent);
    for t in 0..out_shape.height {
        check_cancel(cancelled)?;
        for f in 0..out_shape.width {
            for kernel in weights.chunks_exact(shape.channels * 9) {
                let mut sum = 0.0f64;
                for c in 0..shape.channels {
                    for y in 0..3 {
                        let Some(iy) = (t * 2 + y).checked_sub(1).filter(|&y| y < shape.height)
                        else {
                            continue;
                        };
                        for x in 0..3 {
                            let Some(ix) = (f * 2 + x).checked_sub(1).filter(|&x| x < shape.width)
                            else {
                                continue;
                            };
                            let i = (iy * shape.width + ix) * shape.channels + c;
                            sum += f64::from(input[i]) * f64::from(kernel[c * 9 + y * 3 + x]);
                        }
                    }
                }
                output.push(sum as f32);
            }
        }
    }
    Ok((output, out_shape))
}

/// Per-channel causal temporal cross-correlation over token-major rows.
/// Original weights are [channel,kernel], with zero left padding.
pub fn depthwise_causal(
    input: &[f32],
    channels: usize,
    weights: &[f32],
    kernel: usize,
    cancelled: Option<&AtomicBool>,
) -> Result<Vec<f32>> {
    if channels == 0
        || kernel == 0
        || input.is_empty()
        || !input.len().is_multiple_of(channels)
        || channels.checked_mul(kernel) != Some(weights.len())
    {
        return Err(invalid("depthwise causal convolution geometry"));
    }
    let mut output = Vec::with_capacity(input.len());
    for t in 0..input.len() / channels {
        check_cancel(cancelled)?;
        for c in 0..channels {
            let w = &weights[c * kernel..(c + 1) * kernel];
            let count = kernel.min(t + 1);
            let mut sum = 0.0f64;
            for (i, &value) in w[kernel - count..].iter().enumerate() {
                sum += f64::from(value) * f64::from(input[(t + 1 - count + i) * channels + c]);
            }
            output.push(sum as f32);
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fft_and_convolution_reference() {
        let fft = MagnitudeFft::new(8).unwrap();
        let mut scratch = [[0.; 2]; 8];
        let mut out = [0.; 5];
        fft.magnitude(&[1., 0., 0., 0., 0., 0., 0., 0.], &mut scratch, &mut out)
            .unwrap();
        assert_eq!(out, [1.; 5]);
        fft.magnitude(&[1., 1., 1., 1., 1., 1., 1., 1.], &mut scratch, &mut out)
            .unwrap();
        assert_eq!(out, [8., 0., 0., 0., 0.]);
        let input = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let (out, shape) = subsample_conv2d(
            &input,
            ImageRows {
                height: 3,
                width: 3,
                channels: 1,
            },
            &[1.; 9],
            1,
            None,
        )
        .unwrap();
        assert_eq!(
            shape,
            ImageRows {
                height: 2,
                width: 2,
                channels: 1
            }
        );
        assert_eq!(out, [12., 16., 24., 28.]);
        assert_eq!(
            depthwise_causal(&[1., 10., 2., 20., 3., 30.], 2, &[1., 2., 3., 4.], 2, None).unwrap(),
            [2., 40., 5., 110., 8., 180.]
        );
        let cancel = AtomicBool::new(true);
        assert!(depthwise_causal(&[1.], 1, &[1.], 1, Some(&cancel)).is_err());
    }
}
