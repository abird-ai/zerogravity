//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
//!
//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.

// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

use crate::accelerator::SourceIdentity;
use crate::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, GlobalResourceLease, PhysicalDeviceKey, ResourceArbiter,
    ResourceOwner,
};

const MAX_SPAN_WORDS: usize = 8 * 1024 * 1024;
const MAX_OUTPUT_WORDS: usize = 1024 * 1024;
const MAX_FMAS: usize = 128 * 1024 * 1024;
const SCRATCH_BYTES: u64 = 256;

#[cfg(test)]
std::thread_local! {
    static AFTER_FIRST_OUTPUT: std::cell::RefCell<Option<Box<dyn FnOnce()>>> =
        const { std::cell::RefCell::new(None) };
}

/// Distinct closed arithmetic contract; no executable model/profile capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BulkF32Profile {
    GgmlD812350HaswellTinyBlas,
}

/// Original F32 axes `[inner,rows,head,sequence]`; strides/offset are in words.
/// Borrowed raw activations/weights retain their caller's owner and admission.
/// This descriptor validates a view, not its model-file SHA or source issuer.
#[derive(Clone, Copy)]
pub struct F32View<'a> {
    pub source: SourceIdentity,
    pub name: &'a str,
    pub values: &'a [f32],
    pub offset: usize,
    pub shape: [usize; 4],
    pub strides: [usize; 4],
}

/// Same source for both operands. B is the original contiguous RHS; A may
/// retain a strided V-cache view. All broadcast heads are independently read.
#[derive(Clone, Copy)]
pub struct BulkF32Request<'a> {
    pub profile: BulkF32Profile,
    pub source: SourceIdentity,
    pub left: F32View<'a>,
    pub right: F32View<'a>,
}

/// Existing host ledger, not a new pool or a claim about all host RAM.
pub struct BulkF32Admission<'a> {
    pub arbiter: &'a ResourceArbiter,
    pub resource: &'a PhysicalDeviceKey,
    pub owner: &'a ResourceOwner,
}

/// Payload drops before its original output/scratch ticket. No unleased Vec
/// escape or state publication; synchronous borrowed inputs outlive execution.
pub struct BulkF32Completed {
    values: Vec<f32>,
    source: SourceIdentity,
    shape: [usize; 4],
    _lease: GlobalResourceLease,
}
impl BulkF32Completed {
    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn source(&self) -> SourceIdentity {
        self.source
    }
    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }
    pub fn profile(&self) -> BulkF32Profile {
        BulkF32Profile::GgmlD812350HaswellTinyBlas
    }
}

fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
        return Err(Error::State("Haswell bulk F32 cancelled".into()));
    }
    Ok(())
}

/// Validate this thread's source arithmetic controls without changing them.
pub fn verify_environment() -> Result<()> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut controls = 0u32;
        // SAFETY: read-only control instruction writes one valid stack word.
        unsafe {
            std::arch::asm!("stmxcsr [{at}]", at = in(reg) &mut controls,
                options(nostack, preserves_flags));
        }
        // Ignore sticky exception flags; require masked exceptions, RNE and
        // gradual underflow. Never change the caller's floating-point policy.
        if controls & 0xffc0 == 0x1f80 {
            return Ok(());
        }
    }
    Err(Error::Unsupported(
        "Haswell bulk F32 requires x86_64 RNE/masked/noFTZ/noDAZ controls".into(),
    ))
}

fn checked_product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |acc, &next| {
        acc.checked_mul(next)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 geometry overflow".into()))
    })
}

impl F32View<'_> {
    fn validate(self, source: SourceIdentity, cancelled: Option<&AtomicBool>) -> Result<()> {
        not_cancelled(cancelled)?;
        if !source.is_bound()
            || self.source != source
            || self.name.is_empty()
            || self.name.len() > 256
            || self.name.contains('\0')
            || self.shape.contains(&0)
            || self.shape[0] > 8192
            || self.shape[1] > 4096
            || self.shape[2] > 64
            || self.shape[3] > 64
            || self.strides[0] != 1
        {
            return Err(Error::Malformed(
                "Haswell bulk F32 source/view geometry".into(),
            ));
        }
        // Original matrix rows do not overlap. Axis3's declared stride is
        // checked even when that dimension has size1, preserving raw geometry.
        let mut previous = self.shape[0];
        for (&stride, &size) in self.strides.iter().zip(&self.shape).skip(1) {
            if stride < previous {
                return Err(Error::Malformed(
                    "Haswell bulk F32 overlapping strides".into(),
                ));
            }
            previous = stride
                .checked_mul(size)
                .ok_or_else(|| Error::Limit("Haswell bulk F32 stride overflow".into()))?;
        }
        let end = self
            .shape
            .iter()
            .zip(self.strides)
            .try_fold(self.offset, |at, (&size, stride)| {
                (size - 1)
                    .checked_mul(stride)
                    .and_then(|n| at.checked_add(n))
                    .ok_or_else(|| Error::Limit("Haswell bulk F32 view span overflow".into()))
            })?
            .checked_add(1)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 view end overflow".into()))?;
        if end > self.values.len() || self.values.len() > MAX_SPAN_WORDS {
            return Err(Error::Limit(
                "Haswell bulk F32 view exceeds backing/bound".into(),
            ));
        }
        let words = checked_product(&self.shape)?;
        if words > MAX_OUTPUT_WORDS {
            return Err(Error::Limit(
                "Haswell bulk F32 logical operand bound".into(),
            ));
        }
        // Only actual logical words are read; storage gaps/canaries remain
        // outside the arithmetic and may contain arbitrary IEEE754 words.
        for sequence in 0..self.shape[3] {
            for head in 0..self.shape[2] {
                for row in 0..self.shape[1] {
                    not_cancelled(cancelled)?;
                    let start = self.at(0, row, head, sequence);
                    if self.values[start..start + self.shape[0]]
                        .iter()
                        .any(|x| !x.is_finite())
                    {
                        return Err(Error::Malformed(
                            "nonfinite Haswell bulk F32 operand".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
    fn at(self, inner: usize, row: usize, head: usize, sequence: usize) -> usize {
        // Whole view's final index and products were checked before access.
        self.offset
            + inner
            + row * self.strides[1]
            + head * self.strides[2]
            + sequence * self.strides[3]
    }
}

fn validate(
    request: BulkF32Request<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<([usize; 4], usize)> {
    not_cancelled(cancelled)?;
    let BulkF32Profile::GgmlD812350HaswellTinyBlas = request.profile;
    let a = request.left;
    let b = request.right;
    a.validate(request.source, cancelled)?;
    b.validate(request.source, cancelled)?;
    if a.shape[0] != b.shape[0]
        || !(2..=64).contains(&b.shape[1])
        || !a.shape[1].is_multiple_of(4)
        || !b.shape[2].is_multiple_of(a.shape[2])
        || !b.shape[3].is_multiple_of(a.shape[3])
    {
        // n<2 selects the original vec-dot path, never this bulk schedule.
        return Err(Error::Unsupported(
            "unsupported Haswell tinyBLAS F32 shape".into(),
        ));
    }
    let mut stride = 1usize;
    for (&actual, &size) in b.strides.iter().zip(&b.shape) {
        if actual != stride {
            return Err(Error::Malformed(
                "Haswell tinyBLAS requires original contiguous RHS".into(),
            ));
        }
        stride = stride
            .checked_mul(size)
            .ok_or_else(|| Error::Limit("Haswell bulk F32 contiguous span overflow".into()))?;
    }
    let shape = [a.shape[1], b.shape[1], b.shape[2], b.shape[3]];
    let count = checked_product(&shape)?;
    if count > MAX_OUTPUT_WORDS || count.checked_mul(a.shape[0]).is_none_or(|n| n > MAX_FMAS) {
        return Err(Error::Limit("Haswell bulk F32 output/work bound".into()));
    }
    not_cancelled(cancelled)?;
    Ok((shape, count))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut lanes = [0.0f32; 8];
    let full = a.len() & !7usize;
    for base in (0..full).step_by(8) {
        for (lane, sum) in lanes.iter_mut().enumerate() {
            *sum = a[base + lane].mul_add(b[base + lane], *sum);
        }
    }
    if full != a.len() {
        // tinyBLAS load_partial zero-fills BOTH remaining Vec8 operands.
        // Execute those +0*+0 FMAs too; skipping them changes zero signs.
        for (lane, sum) in lanes.iter_mut().enumerate() {
            let at = full + lane;
            let (x, y) = if at < a.len() {
                (a[at], b[at])
            } else {
                (0.0, 0.0)
            };
            *sum = x.mul_add(y, *sum);
        }
    }
    let low = [
        lanes[0] + lanes[4],
        lanes[1] + lanes[5],
        lanes[2] + lanes[6],
        lanes[3] + lanes[7],
    ];
    // Actual hsum(__m128): add(movehl), then add_ss(movehdup).
    // This is (0+2)+(1+3), not vec_dot's (0+1)+(2+3).
    (low[0] + low[2]) + (low[1] + low[3])
}

/// Pure source-derived primitive. Provider/model/session/checkpoint eligibility
/// is unchanged. Caller authenticates operands and retains input reservations;
/// this function admits only its own private result and bounded stack scratch.
pub fn execute(
    request: BulkF32Request<'_>,
    admission: BulkF32Admission<'_>,
    cancelled: Option<&AtomicBool>,
) -> Result<BulkF32Completed> {
    verify_environment()?;
    let (shape, count) = validate(request, cancelled)?;
    if admission.resource.backend != "host" {
        return Err(Error::Unsupported(
            "Haswell bulk F32 requires existing host admission".into(),
        ));
    }
    let output_bytes = count
        .checked_mul(4)
        .ok_or_else(|| Error::Limit("Haswell bulk F32 output bytes overflow".into()))?;
    let lease = admission.arbiter.admit(
        admission.resource,
        admission.owner.clone(),
        &[
            AllocationRequest {
                kind: AllocationClass::Output,
                bytes: output_bytes as u64,
            },
            AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: SCRATCH_BYTES,
            },
        ],
    )?;
    not_cancelled(cancelled)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| Error::Limit("Haswell bulk F32 output allocation".into()))?;
    if values.capacity() != count {
        return Err(Error::Limit(
            "Haswell bulk F32 unexpected output capacity".into(),
        ));
    }
    let a = request.left;
    let b = request.right;
    let ratio2 = b.shape[2] / a.shape[2];
    let ratio3 = b.shape[3] / a.shape[3];
    for sequence in 0..shape[3] {
        for head in 0..shape[2] {
            for column in 0..shape[1] {
                not_cancelled(cancelled)?;
                let br = b.at(0, column, head, sequence);
                for row in 0..shape[0] {
                    not_cancelled(cancelled)?;
                    let ar = a.at(0, row, head / ratio2, sequence / ratio3);
                    let value = dot(
                        &a.values[ar..ar + a.shape[0]],
                        &b.values[br..br + b.shape[0]],
                    );
                    if !value.is_finite() {
                        return Err(Error::Limit("Haswell bulk F32 nonfinite result".into()));
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
                }
            }
        }
    }
    not_cancelled(cancelled)?;
    verify_environment()?;
    Ok(BulkF32Completed {
        values,
        source: request.source,
        shape,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests;
