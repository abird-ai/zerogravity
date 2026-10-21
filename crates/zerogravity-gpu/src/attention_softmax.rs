//! Original GGML padded softmax with physical FP32 FMA/reduction boundaries.
use crate::source_float::{correct_fma, round_source, widen_source};
use cubecl::prelude::*;
pub(crate) const SCRATCH: usize = 10;

#[cube(launch_unchecked)]
pub(crate) fn maxima(
    input: &[f32],
    maximum: &mut [f32],
    #[comptime] heads: usize,
    #[comptime] width: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let mut m = input[h * width];
        for i in 1..width {
            if input[h * width + i] > m {
                m = input[h * width + i];
            }
        }
        maximum[h] = m;
    }
}
#[cube(launch_unchecked)]
pub(crate) fn prepare(
    input: &[f32],
    maximum: &[f32],
    scratch: &mut [f32],
    #[comptime] count: usize,
    #[comptime] width: usize,
    #[comptime] stage: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let s = i * 10;
        if stage == 0usize {
            let delta = widen_source(input[i]) - widen_source(maximum[i / width]);
            let mut sign = 0u32;
            if delta < 0.0f64 {
                sign = 0x80000000u32;
            }
            // Keep the original rounded delta for the exact-zero branch.
            scratch[s + 9] = round_source(delta, sign);
        } else if stage == 1usize {
            let mut x = scratch[s + 9];
            if x < -128.0f32 {
                x = -128.0f32;
            }
            scratch[s] = x;
        } else if stage == 2usize {
            scratch[s + 2] = round_source(widen_source(scratch[s + 1]) - 12582912.0f64, 0u32);
        } else {
            let b = widen_source(scratch[s + 4]);
            scratch[s + 5] = round_source(b * b, 0u32);
            let t = widen_source(f32::reinterpret(0x3f7ffff6u32)) * b;
            let mut sign = 0u32;
            if t < 0.0f64 {
                sign = 0x80000000u32;
            }
            scratch[s + 8] = round_source(t, sign);
        }
    }
}
#[cube]
fn operands(scratch: &mut [f32], i: usize, #[comptime] step: usize) -> (f32, f32, f32, usize) {
    let s = i * 10;
    if step == 0usize {
        (
            scratch[s],
            f32::reinterpret(0x3fb8aa3bu32),
            f32::reinterpret(0x4b400000u32),
            s + 1,
        )
    } else if step == 1usize {
        (
            -scratch[s + 2],
            f32::reinterpret(0x3f317200u32),
            scratch[s],
            s + 3,
        )
    } else if step == 2usize {
        (
            -scratch[s + 2],
            f32::reinterpret(0x35bfbe8eu32),
            scratch[s + 3],
            s + 4,
        )
    } else if step == 3usize {
        (
            f32::reinterpret(0x3c072010u32),
            scratch[s + 4],
            f32::reinterpret(0x3d2b9f17u32),
            s + 6,
        )
    } else if step == 4usize {
        (
            f32::reinterpret(0x3e2aaf33u32),
            scratch[s + 4],
            f32::reinterpret(0x3efffedbu32),
            s + 7,
        )
    } else if step == 5usize {
        (scratch[s + 6], scratch[s + 5], scratch[s + 7], s + 6)
    } else if step == 6usize {
        (scratch[s + 6], scratch[s + 5], scratch[s + 8], s + 6)
    } else {
        let e = u32::reinterpret(scratch[s + 1]) << 23u32;
        let mut k = f32::reinterpret(e + 0x3f800000u32);
        if scratch[s + 2] < -126.0f32 {
            k = f32::reinterpret(e - 0x82000000u32);
        }
        (scratch[s + 6], k, k, s + 7)
    }
}
#[cube(launch_unchecked)]
pub(crate) fn polynomial(
    scratch: &mut [f32],
    candidate: &mut [f32],
    sums: &mut [f64],
    subtracted: &mut [f64],
    #[comptime] count: usize,
    #[comptime] step: usize,
    #[comptime] stage: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let (a, b, c, dest) = operands(scratch, i, step);
        let p = widen_source(a) * widen_source(b);
        let mut large = p;
        let mut small = widen_source(c);
        if small.abs() > p.abs() {
            large = small;
            small = p;
        }
        if stage == 0usize {
            let sum = p + widen_source(c);
            sums[i] = sum;
            let mut sign =
                ((u32::reinterpret(a) ^ u32::reinterpret(b)) & u32::reinterpret(c)) & 0x80000000u32;
            if sum < 0.0f64 {
                sign = 0x80000000u32;
            }
            candidate[i] = round_source(sum, sign);
        } else if stage == 1usize {
            subtracted[i] = sums[i] - large;
        } else {
            scratch[dest] = correct_fma(sums[i], small - subtracted[i], candidate[i]);
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn exponentials(scratch: &[f32], output: &mut [f32], #[comptime] count: usize) {
    let i = ABSOLUTE_POS;
    if i < count {
        let s = i * 10;
        let mut value = scratch[s + 7];
        if scratch[s + 2] < -126.0f32 {
            value = round_source(
                widen_source(value) * widen_source(f32::reinterpret(0x01000000u32)),
                0u32,
            );
        }
        if scratch[s + 9] < -128.0f32 {
            value = 0.0f32;
        }
        output[i] = value;
    }
}
#[cube(launch_unchecked)]
pub(crate) fn reduce(
    input: &[f32],
    output: &mut [f32],
    #[comptime] heads: usize,
    #[comptime] width: usize,
    #[comptime] level: usize,
) {
    let groups = width.div_ceil(8usize);
    let lanes = 4usize >> level;
    let at = ABSOLUTE_POS;
    if at < heads * groups * lanes {
        let group = at / lanes;
        let lane = at % lanes;
        let mut a = 0.0f32;
        let mut b = 0.0f32;
        if level == 0usize {
            let h = group / groups;
            let first = (group % groups) * 8 + lane;
            if first < width {
                a = input[h * width + first];
            }
            if first + 4usize < width {
                b = input[h * width + first + 4];
            }
        } else if level == 1usize {
            a = input[group * 4 + lane];
            b = input[group * 4 + lane + 2];
        } else {
            a = input[group * 2];
            b = input[group * 2 + 1];
        }
        output[at] = round_source(widen_source(a) + widen_source(b), 0u32);
    }
}
#[cube(launch_unchecked)]
pub(crate) fn totals(
    partial: &[f32],
    output: &mut [f64],
    #[comptime] heads: usize,
    #[comptime] groups: usize,
) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let mut total = 0.0f64;
        for i in 0..groups {
            total += widen_source(partial[h * groups + i]);
        }
        output[h] = total;
    }
}
/// Exact RN-even binary64 reciprocal for the attested softmax sum domain.
/// Division is implemented with integer quotient/remainder, not a device
/// approximate divide. The separately stored double is rounded again to f32.
#[allow(clippy::manual_range_contains)] // Scalar CubeCL IR does not use std ranges.
#[cube(launch_unchecked)]
pub(crate) fn reciprocal(input: &[f64], output: &mut [f64], #[comptime] heads: usize) {
    let h = ABSOLUTE_POS;
    if h < heads {
        let sum = input[h];
        if sum >= 1.0f64 && sum <= 4096.0f64 {
            let bits = u64::reinterpret(sum);
            let m = (bits & 0x000fffffffffffffu64) | 0x0010000000000000u64;
            let e = (bits >> 52u64) - 1023u64;
            let mut encoded = (1023u64 - e) << 52u64;
            if m != 0x0010000000000000u64 {
                let mut rem = 1u64;
                let mut q = 0u64;
                for _iteration in 0..105usize {
                    rem <<= 1u64;
                    q <<= 1u64;
                    if rem >= m {
                        rem -= m;
                        q += 1u64;
                    }
                }
                if rem * 2u64 > m || (rem * 2u64 == m && (q & 1u64) != 0u64) {
                    q += 1u64;
                }
                encoded = ((1022u64 - e) << 52u64) + (q - 0x0010000000000000u64);
            }
            output[h] = f64::reinterpret(encoded);
        } else {
            output[h] = 0.0f64;
        }
    }
}
#[cube(launch_unchecked)]
pub(crate) fn inverses(input: &[f64], output: &mut [f32], #[comptime] heads: usize) {
    let h = ABSOLUTE_POS;
    if h < heads {
        output[h] = round_source(input[h], 0u32);
    }
}
#[cube(launch_unchecked)]
pub(crate) fn probabilities(
    exps: &[f32],
    inverse: &[f32],
    output: &mut [f32],
    #[comptime] count: usize,
    #[comptime] width: usize,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        output[i] = round_source(
            widen_source(exps[i]) * widen_source(inverse[i / width]),
            0u32,
        );
    }
}

#[cfg(test)]
mod tests {
    fn exact_reciprocal(sum: f64) -> f64 {
        assert!((1.0..=4096.0).contains(&sum));
        let m = (sum.to_bits() & 0x000fffffffffffff) | 0x0010000000000000;
        let e = (sum.to_bits() >> 52) - 1023;
        if m == 0x0010000000000000 {
            return f64::from_bits((1023 - e) << 52);
        }
        let (mut rem, mut q) = (1u64, 0u64);
        for _ in 0..105 {
            rem <<= 1;
            q <<= 1;
            if rem >= m {
                rem -= m;
                q += 1;
            }
        }
        if rem * 2 > m || (rem * 2 == m && q & 1 != 0) {
            q += 1;
        }
        f64::from_bits(((1022 - e) << 52) + (q - 0x0010000000000000))
    }
    #[test]
    #[ignore = "million-input exact binary64 reciprocal host fingerprint; no GPU proof"]
    fn exact_reciprocal_preserves_source_double_then_single_rounding() {
        let s = f64::from_bits(0x3ff0000018000024);
        let q = exact_reciprocal(s);
        assert_eq!(q.to_bits(), 0x3fefffffd0000000);
        assert_eq!((q as f32).to_bits(), 0x3f7ffffe);
        let mut rng = 0x1234abcd9876fedcu64;
        for i in 0..1_000_000u64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let sum = f64::from_bits(((1023 + i % 12) << 52) | (rng & 0x000fffffffffffff));
            assert_eq!(
                exact_reciprocal(sum).to_bits(),
                (1.0 / sum).to_bits(),
                "S={sum:?}"
            );
        }
        for e in 0..=12u64 {
            let bits = (1023 + e) << 52;
            for offset in [-1i64, 0, 1] {
                let sum = f64::from_bits(bits.wrapping_add_signed(offset));
                if (1.0..=4096.0).contains(&sum) {
                    assert_eq!(exact_reciprocal(sum).to_bits(), (1.0 / sum).to_bits());
                }
            }
        }
    }
}
