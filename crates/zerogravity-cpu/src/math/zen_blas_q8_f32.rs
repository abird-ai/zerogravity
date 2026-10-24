//! Closed original 2B Q8_0 `[2048,2048]` × F32 `[2048,64]` arithmetic.
//!
//! OpenBLAS 0.3.33 ZEN, observed configured16/OpenMP: each output visits
//! seven K panels320×5+224×2 through m128/n4 kernels. Every m8/n4 microtile
//! has separate even/odd K FMA chains, a rounded merge and alpha1 FMA write.
//! Scalar Rust preserves these rounding points; no BLAS/FFI/thread dependency.
//! This is an operator contract, not arbitrary SGEMM or a model capability.
//!
//! Derivation: OpenBLAS v0.3.33 kernel/x86_64/sgemm_kernel_8x4_haswell_2.c
//! lines21–26,109–125,273–284,342–365,409–413; driver/level3/level3_thread.c
//! lines321–329,371–394,417–431. Source: https://github.com/OpenMathLib/OpenBLAS/tree/v0.3.33

/* Source-derived arithmetic specialization; upstream notice retained.
Copyright (c) 2011-2014, The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in
   the documentation and/or other materials provided with the distribution.
3. Neither the name of the OpenBLAS project nor the names of
   its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

use super::haswell_bulk_f32::{BulkF32Admission, verify_environment};
use crate::accelerator::SourceIdentity;
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_artifacts::quant::{QuantKind, dequantize};
use zerogravity_core::resource_arbiter::{AllocationClass, AllocationRequest, GlobalResourceLease};

const WIDTH: usize = 2048;
const TEACHERS: usize = 64;
const WEIGHT_WORDS: usize = WIDTH * WIDTH;
const WEIGHT_BYTES: usize = WEIGHT_WORDS / 32 * 34;
const OUTPUT_WORDS: usize = WIDTH * TEACHERS;
const OUTPUT_BYTES: u64 = (OUTPUT_WORDS * std::mem::size_of::<f32>()) as u64;
const SCRATCH_BYTES: u64 = (WEIGHT_WORDS * std::mem::size_of::<f32>()) as u64 + 256;
const PANELS: [(usize, usize); 7] = [
    (0, 320),
    (320, 320),
    (640, 320),
    (960, 320),
    (1280, 320),
    (1600, 224),
    (1824, 224),
];

#[cfg(test)]
std::thread_local! {
    static AFTER_FIRST_OUTPUT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Exact original row-packed Q8_0 W and dense teacher-major F32 input.
/// Caller authenticates source bytes and retains their owner/admission for
/// this synchronous borrow; this descriptor does not verify model-file hashes.
#[derive(Clone, Copy)]
pub struct ZenQ8F32Request<'a> {
    pub source: SourceIdentity,
    pub weights: &'a [u8],
    pub input: &'a [f32],
}

/// Pure private result. Payload drops before its physical output lease;
/// dequantized weights are already freed. No Vec escape or state publication.
pub struct ZenQ8F32Completed {
    values: Vec<f32>,
    source: SourceIdentity,
    _lease: GlobalResourceLease,
}
impl ZenQ8F32Completed {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn shape(&self) -> [usize; 4] {
        [WIDTH, TEACHERS, 1, 1]
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("closed Zen Q8/F32 cancelled".into()));
    }
    Ok(())
}

fn validate(request: ZenQ8F32Request<'_>, cancelled: Option<&AtomicBool>) -> Result<()> {
    not_cancelled(cancelled)?;
    if !request.source.is_bound()
        || request.weights.len() != WEIGHT_BYTES
        || request.input.len() != OUTPUT_WORDS
    {
        return Err(Error::Malformed(
            "closed Zen Q8/F32 source or exact2048x2048x64 lengths".into(),
        ));
    }
    for input in request.input.chunks_exact(WIDTH) {
        not_cancelled(cancelled)?;
        if input.iter().any(|v| !v.is_finite()) {
            return Err(Error::Malformed("nonfinite Zen Q8/F32 input".into()));
        }
    }
    for row in request.weights.chunks_exact(WIDTH / 32 * 34) {
        not_cancelled(cancelled)?;
        // Only the binary16 exponent is inspected; actual source conversion
        // remains in the existing artifact codec after physical admission.
        if row
            .chunks_exact(34)
            .any(|block| u16::from_le_bytes([block[0], block[1]]) & 0x7c00 == 0x7c00)
        {
            return Err(Error::Malformed("nonfinite Zen Q8_0 scale".into()));
        }
    }
    Ok(())
}

fn cell(weights: &[f32], input: &[f32]) -> f32 {
    // Native beta0 initializes C to +0 before the first panel. Previous C
    // enters only the panel write, never either panel's K reduction chain.
    let mut result = 0.0f32;
    for (start, width) in PANELS {
        let mut even = 0.0f32;
        let mut odd = 0.0f32;
        for at in (start..start + width).step_by(2) {
            even = weights[at].mul_add(input[at], even);
            odd = weights[at + 1].mul_add(input[at + 1], odd);
        }
        // SAVE_m8n4: vaddps odd,even; FMA(alpha1,merged,previousC).
        let merged = odd + even;
        result = merged.mul_add(1.0, result);
    }
    result
}

/// Execute only the observed closed source schedule. Admission covers this
/// operation's full16MiB decoded scratch and512KiB result separately; borrowed
/// source/input reservations remain caller-owned. No worker or model permit.
pub fn execute(
    request: ZenQ8F32Request<'_>,
    admission: BulkF32Admission<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<ZenQ8F32Completed> {
    verify_environment()?;
    validate(request, cancelled)?;
    if admission.resource.backend != "host" {
        return Err(Error::Unsupported(
            "closed Zen Q8/F32 needs existing host admission".into(),
        ));
    }
    let scratch = admission.arbiter.admit(
        admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Scratch,
            bytes: SCRATCH_BYTES,
        }],
    )?;
    let output_lease = admission.arbiter.admit(
        admission.resource,
        admission.owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Output,
            bytes: OUTPUT_BYTES,
        }],
    )?;
    not_cancelled(cancelled)?;
    let weights = dequantize(QuantKind::Q8_0, request.weights, WEIGHT_WORDS)?;
    if weights.len() != WEIGHT_WORDS || weights.capacity() != WEIGHT_WORDS {
        return Err(Error::Limit("Zen Q8/F32 decoded capacity".into()));
    }
    not_cancelled(cancelled)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(OUTPUT_WORDS)
        .map_err(|_| Error::Limit("Zen Q8/F32 output allocation".into()))?;
    if values.capacity() != OUTPUT_WORDS {
        return Err(Error::Limit("Zen Q8/F32 output capacity".into()));
    }
    // Sixteen observed n4 teacher partitions. Distinct output cells do not
    // reduce into each other; no native OpenMP scheduling is reproduced.
    for partition in 0..16 {
        for teacher in partition * 4..(partition + 1) * 4 {
            let input = &request.input[teacher * WIDTH..(teacher + 1) * WIDTH];
            for row in weights.chunks_exact(WIDTH) {
                not_cancelled(cancelled)?;
                let value = cell(row, input);
                if !value.is_finite() {
                    return Err(Error::Limit("Zen Q8/F32 nonfinite result".into()));
                }
                values.push(value);
                #[cfg(test)]
                if values.len() == 1 {
                    AFTER_FIRST_OUTPUT.with(|hook| {
                        if let Some(hook) = hook.borrow_mut().take() {
                            hook();
                        }
                    });
                }
                // A caller signal may alter this thread's controls. Refuse
                // that private computation before using its next output.
                verify_environment()?;
            }
        }
    }
    drop(weights);
    drop(scratch);
    not_cancelled(cancelled)?;
    verify_environment()?;
    Ok(ZenQ8F32Completed {
        values,
        source: request.source,
        _lease: output_lease,
    })
}

#[cfg(test)]
mod tests;
