//! CUDA-only ordered version of the original12-launch source SiLU.
//! No exp intrinsic, new polynomial or numerical profile.
use crate::source_accumulation::{SourceAccumulation, ensure_capability};
use crate::source_float::{correct_quotient, round_source, widen_source};
use cubecl::{client::Client, prelude::*, server::Handle};
use cubecl_common::device::ServiceId;
use std::ops::Range;
use zerogravity_core::{Error, Result};

const MAX_ROW_WORDS: usize = 65_536;
const LANES: u32 = 32;
pub(crate) const TRACE_WORDS: usize = 17;

/// Exact closed support; callers select the original route outside this geometry.
pub(crate) fn supports_geometry(rows: usize, width: usize) -> bool {
    matches!(rows, 1 | 2 | 4 | 8) && width > 0 && width <= MAX_ROW_WORDS
}

fn geometry(rows: usize, width: usize) -> Result<usize> {
    if !supports_geometry(rows, width) {
        return Err(Error::Unsupported(
            "ordered SiLU native row geometry".into(),
        ));
    }
    rows.checked_mul(width)
        .filter(|&n| u32::try_from(n).is_ok())
        .ok_or_else(|| Error::Limit("ordered SiLU aggregate domain".into()))
}

fn checked_span(size: u64, start: u64, trim: u64, words: usize) -> Result<Range<u64>> {
    let end = size
        .checked_sub(trim)
        .ok_or_else(|| Error::State("ordered SiLU end exceeds allocation".into()))?;
    let used = end
        .checked_sub(start)
        .ok_or_else(|| Error::State("ordered SiLU start exceeds end".into()))?;
    let bytes = u64::try_from(words)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| Error::Limit("ordered SiLU byte domain".into()))?;
    if bytes == 0 || !start.is_multiple_of(4) || !used.is_multiple_of(4) || used < bytes {
        return Err(Error::State("ordered SiLU buffer alignment/size".into()));
    }
    Ok(start..start + bytes)
}

fn binding(client: &Client, handle: &Handle, words: usize) -> Result<Range<u64>> {
    if handle.service != client.service_id() {
        return Err(Error::State("ordered SiLU foreign service".into()));
    }
    checked_span(
        handle.size(),
        handle.offset_start.unwrap_or(0),
        handle.offset_end.unwrap_or(0),
        words,
    )
}

pub(crate) struct Bindings<'a> {
    pub(crate) input: &'a Handle,
    pub(crate) multiplier: Option<&'a Handle>,
    pub(crate) output: &'a Handle,
    pub(crate) trace: Option<&'a Handle>,
}

/// Caller preflights ALL model nodes before its original flight; leases, source,
/// status, interval/finite checks, Ready/cancel/quarantine remain original owners.
/// This binding allocates/uploads no device storage, submits no physical work,
/// and never clears status.
pub(crate) struct PreparedOrderedSilu {
    service: ServiceId,
    input: Handle,
    multiplier: Handle,
    output: Handle,
    trace: Handle,
    count: usize,
    multiplied: bool,
    record: bool,
}

impl PreparedOrderedSilu {
    pub(crate) fn prepare(
        client: &Client,
        rows: usize,
        width: usize,
        bindings: Bindings<'_>,
    ) -> Result<Self> {
        let Bindings {
            input,
            multiplier,
            output,
            trace,
        } = bindings;
        if !crate::quantization::activation_fused::is_cuda_service(client) {
            return Err(Error::Unsupported(
                "ordered SiLU CUDA-only candidate".into(),
            ));
        }
        ensure_capability(client, SourceAccumulation::WideF64)?;
        let count = geometry(rows, width)?;
        let hardware = &client.properties().hardware;
        if hardware.max_bindings < 4
            || hardware.max_units_per_cube < LANES
            || hardware.max_cube_dim.0 < LANES
            || hardware.max_cube_dim.1 == 0
            || hardware.max_cube_dim.2 == 0
            || hardware.max_cube_count.0 < (count as u32).div_ceil(LANES)
            || hardware.max_cube_count.1 == 0
            || hardware.max_cube_count.2 == 0
        {
            return Err(Error::Unsupported(
                "ordered SiLU CUDA grid/binding limits".into(),
            ));
        }
        let input_range = binding(client, input, count)?;
        let output_range = binding(client, output, count)?;
        let reads = [
            Some((input, input_range)),
            if let Some(m) = multiplier {
                Some((m, binding(client, m, count)?))
            } else {
                None
            },
        ];
        let writes = [
            Some((output, output_range)),
            if let Some(t) = trace {
                Some((t, binding(client, t, count * TRACE_WORDS)?))
            } else {
                None
            },
        ];
        for (index, left) in writes.iter().enumerate() {
            if let Some((left, left_span)) = left {
                for (right, right_span) in reads.iter().chain(writes[..index].iter()).flatten() {
                    if (*left).clone().binding().memory.id()
                        == (*right).clone().binding().memory.id()
                        && left_span.start < right_span.end
                        && right_span.start < left_span.end
                    {
                        return Err(Error::State("ordered SiLU writable alias".into()));
                    }
                }
            }
        }
        Ok(Self {
            service: client.service_id(),
            input: input.clone(),
            // Inactive aliases have no device access under compile-time branches.
            multiplier: multiplier.unwrap_or(input).clone(),
            output: output.clone(),
            trace: trace.unwrap_or(output).clone(),
            count,
            multiplied: multiplier.is_some(),
            record: trace.is_some(),
        })
    }

    /// ONE actual source launch; no inferred driver count or timing claim.
    pub(crate) fn launch(&self, client: &Client) -> Result<usize> {
        if client.service_id() != self.service {
            return Err(Error::State("ordered SiLU service changed".into()));
        }
        // SAFETY: complete spans, exact original service and disjoint writes were
        // checked cold. Original flight must already retain all owning leases.
        unsafe {
            ordered::launch_unchecked(
                client,
                CubeCount::Static((self.count as u32).div_ceil(LANES), 1, 1),
                CubeDim::new_1d(LANES),
                BufferArg::from_raw_parts(self.input.clone(), self.count),
                BufferArg::from_raw_parts(self.multiplier.clone(), self.count),
                BufferArg::from_raw_parts(self.output.clone(), self.count),
                BufferArg::from_raw_parts(
                    self.trace.clone(),
                    if self.record {
                        self.count * TRACE_WORDS
                    } else {
                        self.count
                    },
                ),
                self.count,
                self.multiplied,
                self.record,
            );
        }
        Ok(1)
    }
}

// Explicit PTX .rn instructions, no .ftz. gpu_asm without `pure` lowers to
// volatile asm; f32 output constraints retain each rounded source result even
// when immediately widened by the next stage. No float-cast-only assumption.
// gpu_asm scalar inputs must be SSA values: its generic HasValue path also
// accepts pointers. Bit-only reinterpretation forces the frontend read_value
// load for prior mutable asm outputs without changing any floating-point bits.
#[cube]
fn rn32(value: f64) -> f32 {
    let value = f64::reinterpret(u64::reinterpret(value));
    let output: f32;
    gpu_asm!("cvt.rn.f32.f64 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
    output
}
#[cube]
fn wide(value: f32) -> f64 {
    let value = f32::reinterpret(u32::reinterpret(value));
    let output: f64;
    gpu_asm!("cvt.f64.f32 {out}, {value};",out=out(_) output,value=in(_) value,options(nomem));
    output
}
#[cube]
fn add64(left: f64, right: f64) -> f64 {
    let left = f64::reinterpret(u64::reinterpret(left));
    let right = f64::reinterpret(u64::reinterpret(right));
    let output: f64;
    gpu_asm!("add.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
    output
}
#[cube]
fn sub64(left: f64, right: f64) -> f64 {
    let left = f64::reinterpret(u64::reinterpret(left));
    let right = f64::reinterpret(u64::reinterpret(right));
    let output: f64;
    gpu_asm!("sub.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
    output
}
#[cube]
fn mul64(left: f64, right: f64) -> f64 {
    let left = f64::reinterpret(u64::reinterpret(left));
    let right = f64::reinterpret(u64::reinterpret(right));
    let output: f64;
    gpu_asm!("mul.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
    output
}
#[cube]
fn div64(left: f64, right: f64) -> f64 {
    let left = f64::reinterpret(u64::reinterpret(left));
    let right = f64::reinterpret(u64::reinterpret(right));
    let output: f64;
    gpu_asm!("div.rn.f64 {out}, {left}, {right};",out=out(_) output,left=in(_) left,right=in(_) right,options(nomem));
    output
}
#[cube]
fn fma64(a: f64, b: f64, c: f64) -> f64 {
    let a = f64::reinterpret(u64::reinterpret(a));
    let b = f64::reinterpret(u64::reinterpret(b));
    let c = f64::reinterpret(u64::reinterpret(c));
    let output: f64;
    gpu_asm!("fma.rn.f64 {out}, {a}, {b}, {c};",out=out(_) output,a=in(_) a,b=in(_) b,c=in(_) c,options(nomem));
    output
}
#[cube]
fn negative_input(input: f32) -> f32 {
    let input = f32::reinterpret(u32::reinterpret(input));
    let output: f32;
    gpu_asm!("sub.rn.f32 {out}, 0f00000000, {value};",out=out(_) output,value=in(_) input,options(nomem));
    output
}
#[cube]
fn stored_bits(input: f32) -> u32 {
    let input = f32::reinterpret(u32::reinterpret(input));
    let output: u32;
    gpu_asm!("mov.b32 {out}, {value};",out=out(_) output,value=in(_) input,options(nomem));
    output
}
#[cube]
fn negate(input: f32) -> f32 {
    f32::reinterpret(u32::reinterpret(input) ^ 0x80000000u32)
}
#[cube]
fn wide_fma(a: f32, b: f32, c: f32) -> f32 {
    // A finite f32 product is exact in f64. One explicit F64 FMA has the
    // same f64 result as the original potentially contracted wide expression.
    rn32(fma64(wide(a), wide(b), wide(c)))
}

#[cube(launch_unchecked)]
fn ordered(
    input: &[u32],
    multiplier: &[u32],
    output: &mut [u32],
    trace: &mut [u32],
    #[comptime] count: usize,
    #[comptime] multiplied: bool,
    #[comptime] record: bool,
) {
    let i = ABSOLUTE_POS;
    if i < count {
        let original = f32::reinterpret(input[i]);
        let x = negative_input(original);
        let z = wide_fma(
            x,
            f32::reinterpret(0x3fb8aa3bu32),
            f32::reinterpret(0x4b400000u32),
        );
        let n = rn32(sub64(wide(z), wide(f32::reinterpret(0x4b400000u32))));
        let b0 = wide_fma(negate(n), f32::reinterpret(0x3f317200u32), x);
        let b = wide_fma(negate(n), f32::reinterpret(0x35bfbe8eu32), b0);
        let u = rn32(mul64(wide(b), wide(b)));
        let j0 = wide_fma(
            f32::reinterpret(0x3c072010u32),
            b,
            f32::reinterpret(0x3d2b9f17u32),
        );
        let j1 = wide_fma(
            f32::reinterpret(0x3e2aaf33u32),
            b,
            f32::reinterpret(0x3efffedbu32),
        );
        let j2 = rn32(mul64(wide(f32::reinterpret(0x3f7ffff6u32)), wide(b)));
        let j3 = wide_fma(j0, u, j1);
        let j4 = wide_fma(j3, u, j2);
        let e = u32::reinterpret(z) << 23u32;
        let k = f32::reinterpret(e + 0x3f800000u32);
        let mut base = wide_fma(j4, k, k);
        let mut factor = 1.0f32;
        if f32::reinterpret(u32::reinterpret(n) & 0x7fffffffu32) > 126.0f32 {
            let mut g = 0u32;
            if n <= 0.0f32 {
                g = 0x82000000u32;
            }
            let s1 = f32::reinterpret(g + 0x7f000000u32);
            let s2 = f32::reinterpret(e - g);
            if f32::reinterpret(u32::reinterpret(n) & 0x7fffffffu32) > 192.0f32 {
                base = s1;
            } else {
                base = wide_fma(s2, j4, s2);
            }
            factor = s1;
        }
        let exponential = rn32(mul64(wide(base), wide(factor)));
        let denominator = rn32(add64(1.0f64, wide(exponential)));
        // Reuse exact original manual subnormal packing and neighboring-quotient
        // correction. Opaque F32 register bits separate these two stages.
        let estimate_bits = stored_bits(round_source(
            div64(widen_source(original), widen_source(denominator)),
            input[i] & 0x80000000u32,
        ));
        let quotient = correct_quotient(original, denominator, f32::reinterpret(estimate_bits));
        let quotient_bits = stored_bits(quotient);
        let mut result = quotient_bits;
        if multiplied {
            let m = f32::reinterpret(multiplier[i]);
            result = stored_bits(round_source(
                mul64(
                    widen_source(m),
                    widen_source(f32::reinterpret(quotient_bits)),
                ),
                (multiplier[i] ^ quotient_bits) & 0x80000000u32,
            ));
        }
        output[i] = result;
        if record {
            let t = i * 17;
            trace[t] = u32::reinterpret(z);
            trace[t + 1] = u32::reinterpret(n);
            trace[t + 2] = u32::reinterpret(b0);
            trace[t + 3] = u32::reinterpret(b);
            trace[t + 4] = u32::reinterpret(u);
            trace[t + 5] = u32::reinterpret(j0);
            trace[t + 6] = u32::reinterpret(j1);
            trace[t + 7] = u32::reinterpret(j2);
            trace[t + 8] = u32::reinterpret(j3);
            trace[t + 9] = u32::reinterpret(j4);
            trace[t + 10] = u32::reinterpret(base);
            trace[t + 11] = u32::reinterpret(factor);
            trace[t + 12] = u32::reinterpret(exponential);
            trace[t + 13] = u32::reinterpret(denominator);
            trace[t + 14] = estimate_bits;
            trace[t + 15] = quotient_bits;
            trace[t + 16] = result;
        }
    }
}

#[cfg(all(test, feature = "qualification", feature = "cuda"))]
mod qualification;
#[cfg(test)]
mod tests;
