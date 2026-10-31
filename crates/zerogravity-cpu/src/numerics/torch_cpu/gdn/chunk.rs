//! Original padded GDN arithmetic; no model, state-publication or native permit.
use crate::{Error, Result, accelerator::SourceIdentity};

const ORDER: usize = 64;
const WIDTH: usize = 128;
const MAX_SYSTEMS: usize = 48;

mod core;
pub use core::{ChunkCompleted, ChunkInputs, ChunkObserver, ChunkStage, core, core_observed};

/// Unit-lower source solve, logical C-order `[system,64,64]` and `[system,64,128]`.
/// Original SSE2 reduction: separate products, four interleaved accumulators,
/// `(s0+s2)+(s1+s3)`, scalar tail, then one RHS subtraction. Never FMA.
/// Diagonal/upper coefficients are ignored; all supplied values must be finite.
/// Caller admits output storage and retains the original source.
pub fn solve_unit_lower(
    source: SourceIdentity,
    coefficients: &[f32],
    rhs: &[f32],
) -> Result<Vec<f32>> {
    if !source.is_bound()
        || coefficients.is_empty()
        || !coefficients.len().is_multiple_of(ORDER * ORDER)
        || coefficients.len() > MAX_SYSTEMS * ORDER * ORDER
        || rhs.len() != coefficients.len() / ORDER * WIDTH
    {
        return Err(Error::Malformed(
            "source padded triangular system/RHS extent".into(),
        ));
    }
    if coefficients.iter().chain(rhs).any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite source padded triangular operand".into(),
        ));
    }
    let mut output = Vec::new();
    output
        .try_reserve_exact(rhs.len())
        .map_err(|_| Error::Limit("source padded triangular output".into()))?;
    if output.capacity() != rhs.len() {
        return Err(Error::Limit(
            "source padded triangular exact output capacity".into(),
        ));
    }
    output.extend_from_slice(rhs);
    for (a, x) in coefficients
        .chunks_exact(ORDER * ORDER)
        .zip(output.chunks_exact_mut(ORDER * WIDTH))
    {
        for row in 0..ORDER {
            let vector_end = row / 4 * 4;
            for column in 0..WIDTH {
                let mut sums = [0.0f32; 4];
                for start in (0..vector_end).step_by(4) {
                    for (lane, sum) in sums.iter_mut().enumerate() {
                        let k = start + lane;
                        let product = a[row * ORDER + k] * x[k * WIDTH + column];
                        *sum += product;
                    }
                }
                let mut sum = (sums[0] + sums[2]) + (sums[1] + sums[3]);
                for k in vector_end..row {
                    let product = a[row * ORDER + k] * x[k * WIDTH + column];
                    sum += product;
                }
                x[row * WIDTH + column] -= sum;
            }
        }
    }
    if output.iter().any(|v| !v.is_finite()) {
        return Err(Error::Malformed(
            "nonfinite source padded triangular result".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests;
