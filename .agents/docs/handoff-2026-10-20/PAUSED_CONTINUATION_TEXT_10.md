# Paused continuation text — volume 10

Preservation only. Revalidate source, ownership and status before use.
Never apply these snapshots blindly over a current tree.

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/prepare.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6030; SHA256 `2797547b5cc5f8df21f445d75de114b84587a81fe39dfe5650e8e5bfce779a47`.

````text
"""ROOT ONLY byte transport from immutable original AV/alpha captures/defaults."""
import argparse
from pathlib import Path
from contracts import HERE, AV, ALPHA, CASES, digest, frozen, parents, read, require, write

MODEL = {'av':'57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
         'alpha':'91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6'}

def transport_descriptor(case):
    root = AV if case == 'av' else ALPHA
    validation = read(root / 'root-validation.json')
    if case == 'av':
        require(validation['status'] == 'PUBLIC_AV_DEFAULT_REPLAY_EXACT_CONTROLS_DIAGNOSTIC'
                and validation['default_words'] == 262144, 'accepted original AV public default replay')
        public_invocation = root / 'root-replay.invocation.json'
    else:
        require(validation['status'] == 'TRANSPARENT_PUBLIC_ALPHA_EXPORTED_ENTRY_ROUTE_OBSERVED'
                and validation['all65_defaults_exact_to_actual_captures'] is True
                and validation['all130_plain_observed_outputs_exact'] is True,
                'accepted original alpha public default replay')
        public_invocation = root / 'root-plain.invocation.json'
    manifest = read(root / 'root-fixtures/manifest.json')
    original = manifest['cases'][0]
    require(original['case'] == 'bulk64' and original['mode'] == 'bulk64' and original['call'] == 0,
            'original complete default bulk64')
    def field(key):
        row = original['fields'][key]
        d = row if 'sha256' in row else row['digest']
        return {'file':case + '.' + key + '.f32le','name':row['captured_row']['native_name'],
                'shape':row['captured_row']['native_ne'],
                'strides':[n // 4 for n in row['captured_row']['native_nb']],
                'digest':{k:d[k] for k in ('bytes','sha256')}}, root / 'root-fixtures' / row['file']
    if case == 'av':
        left, left_path = field('v')
        right, right_path = field('p')
    else:
        weight = manifest['weight']
        left = {'file':'alpha.weight.f32le','name':weight['captured_row']['name'],
                'shape':weight['captured_row']['ne'],
                'strides':[n // 4 for n in weight['captured_row']['nb']], 'digest':weight['digest']}
        left_path = root / 'root-fixtures/weight.f32le'
        right,right_path = field('x')
    expected, expected_path = field('expected')
    public = root / ('root-replay' if case == 'av' else 'root-plain') / 'default-bulk64.actual.f32le'
    require(digest(public) == expected['digest'] == digest(expected_path),
            'unchanged public DEFAULT outputs equal authentic capture, before new math')
    value = {'case':case,'model_sha256':MODEL[case],'left':left,'right':right,'expected':expected,
             'source_provenance':{'original_fixture_manifest':str(root / 'root-fixtures/manifest.json'),
                                  'original_fixture_manifest_digest':digest(root / 'root-fixtures/manifest.json'),
                                  'original_public_validation':str(root / 'root-validation.json'),
                                  'original_public_validation_digest':digest(root / 'root-validation.json'),
                                  'original_public_invocation':str(public_invocation),
                                  'original_public_invocation_digest':digest(public_invocation),
                                  'public_default':str(public),'public_default_digest':digest(public)}}
    return value, (left_path,right_path,expected_path)

def verify_transport(folder):
    manifest = read(folder / 'manifest.json')
    require(manifest['status'] == 'AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY'
            and manifest['profile'] == 'GgmlD812350HaswellTinyBlas'
            and [row['case'] for row in manifest['cases']] == list(CASES), 'closed two-case fixture scope')
    for row in manifest['cases']:
        expected, paths = transport_descriptor(row['case'])
        require(row == expected, 'exact original source descriptor/provenance')
        for key,path in zip(('left','right','expected'), paths):
            require(digest(folder / row[key]['file']) == row[key]['digest'] == digest(path),
                    'full original input/output transport bytes')
    return manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-fixtures', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    parents()
    require(args.new_fixtures.is_absolute() and args.new_fixtures.parent.resolve() == HERE
            and not args.new_fixtures.exists(), 'fresh own fixture directory')
    args.new_fixtures.mkdir(mode=0o700)
    rows = []
    for case in CASES:
        row,paths = transport_descriptor(case)
        rows.append(row)
        for key,path in zip(('left','right','expected'), paths):
            require(digest(path, 8 * 1024 * 1024) == row[key]['digest'], 'actual bounded source copy')
            with path.open('rb') as src, (args.new_fixtures / row[key]['file']).open('xb') as dst:
                remaining = row[key]['digest']['bytes']
                while remaining:
                    raw = src.read(min(remaining,1024 * 1024))
                    require(raw, 'complete bounded source payload')
                    dst.write(raw)
                    remaining -= len(raw)
                require(not src.read(1), 'no source payload growth')
            (args.new_fixtures / row[key]['file']).chmod(0o600)
    write(args.new_fixtures / 'manifest.json', {'status':'AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY',
          'profile':'GgmlD812350HaswellTinyBlas','cases':rows,
          'arithmetic_executed':False,'expected_from_candidate':False,
          'gap_storage':'actual dense logical V restored at original strides; uncaptured gaps zero-filled'})
    verify_transport(args.new_fixtures)
    require(frozen() == pin, 'unchanged preparation/source')
    print('AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7696; SHA256 `dac2aed8e6f9e9f4a8c50a509811a4dffe44a35e961c50b87efa14884e50064a`.

````text
use super::*;
use std::sync::Arc;

fn budget(limit: u64) -> (ResourceArbiter, PhysicalDeviceKey, ResourceOwner) {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "bulk-f32-unit").unwrap();
    arbiter.register(key.clone(), limit).unwrap();
    (
        arbiter,
        key,
        ResourceOwner::new("source-free", "unit").unwrap(),
    )
}
fn request<'a>(source: SourceIdentity, a: &'a [f32], b: &'a [f32]) -> BulkF32Request<'a> {
    BulkF32Request {
        source,
        profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
        left: F32View {
            source,
            name: "left",
            values: a,
            offset: 0,
            shape: [8, 4, 1, 1],
            strides: [1, 8, 32, 32],
        },
        right: F32View {
            source,
            name: "right",
            values: b,
            offset: 0,
            shape: [8, 2, 1, 1],
            strides: [1, 8, 16, 16],
        },
    }
}

#[test]
fn source_one_chain_and_horizontal_order_have_independent_counterexamples() {
    let a = [1.0; 32];
    let mut b = [0.0; 32];
    b[0] = 16_777_216.0;
    b[8] = 1.0;
    b[16] = -16_777_216.0;
    b[24] = 1.0;
    // Four independent Vec8 partials yield2; the original one-chain yields1.
    assert_eq!(dot(&a, &b).to_bits(), 1.0f32.to_bits());
    let h = [67_108_864.0, 1.0, -67_108_864.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    // Even/odd horizontal pairing yields2; adjacent-pair horizontal gives0.
    assert_eq!(dot(&[1.0; 8], &h).to_bits(), 2.0f32.to_bits());
}

#[test]
fn real_zero_filled_tail_fmas_preserve_source_signed_zero_semantics() {
    let tiny = f32::from_bits(1);
    assert_eq!(dot(&[-tiny; 8], &[tiny; 8]).to_bits(), (-0.0f32).to_bits());
    // Original load_partial fills BOTH unused operands with +0 before FMA.
    assert_eq!(dot(&[-tiny; 9], &[tiny; 9]).to_bits(), 0.0f32.to_bits());
    assert_eq!(dot(&[2.0; 3], &[3.0; 3]).to_bits(), 18.0f32.to_bits());
}

#[test]
fn strided_offset_broadcast_canaries_and_output_lease_are_real() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut a = vec![f32::NAN; 128];
    for head in 0..2 {
        for row in 0..4 {
            a[3 + head * 64 + row * 12..3 + head * 64 + row * 12 + 8]
                .fill((head * 4 + row + 1) as f32);
        }
    }
    let mut b = vec![f32::NAN; 66];
    b[1..65].fill(1.0);
    let mut r = request(source, &a, &b);
    r.left.offset = 3;
    r.left.strides = [1, 12, 64, 128];
    r.left.shape = [8, 4, 2, 1];
    r.right.offset = 1;
    r.right.shape = [8, 2, 4, 1];
    r.right.strides = [1, 8, 16, 64];
    let (arbiter, key, owner) = budget(4096);
    let result = execute(
        r,
        BulkF32Admission {
            arbiter: &arbiter,
            resource: &key,
            owner: &owner,
        },
        None,
    )
    .unwrap();
    assert_eq!(result.shape(), [4, 2, 4, 1]);
    for head in 0..4 {
        for column in 0..2 {
            for row in 0..4 {
                assert_eq!(
                    result.values()[row + 4 * (column + 2 * head)],
                    (head / 2 * 4 + row + 1) as f32 * 8.0
                );
            }
        }
    }
    let live = arbiter.snapshot(&key).unwrap();
    assert_eq!(live.reservations, 1);
    assert_eq!(live.per_class[&AllocationClass::Output], 128);
    assert_eq!(live.per_class[&AllocationClass::Scratch], SCRATCH_BYTES);
    assert!(a[0].is_nan() && a[11].is_nan() && b[0].is_nan() && b[65].is_nan());
    assert_eq!(result.source(), source);
    drop(result);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
fn foreign_source_stride_span_overflow_and_ar1_are_closed() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(4096);
    let valid = request(source, &a, &b);
    let mut cases = Vec::new();
    let mut r = valid;
    r.right.source = SourceIdentity::for_new_allocation().unwrap();
    cases.push(r);
    let mut r = valid;
    r.left.strides[1] = 7;
    cases.push(r);
    let mut r = valid;
    r.left.offset = 1;
    cases.push(r);
    let mut r = valid;
    r.left.offset = usize::MAX;
    cases.push(r);
    let mut r = valid;
    r.left.strides[3] = usize::MAX;
    r.left.shape[3] = 2;
    cases.push(r);
    let mut r = valid;
    r.right.shape[1] = 1;
    r.right.strides = [1, 8, 8, 8];
    cases.push(r);
    let mut r = valid;
    r.right.strides[3] = 17;
    cases.push(r);
    for r in cases {
        assert!(
            execute(
                r,
                BulkF32Admission {
                    arbiter: &arbiter,
                    resource: &key,
                    owner: &owner
                },
                None
            )
            .is_err()
        );
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
}

#[test]
fn actual_capacity_pre_cancel_and_mid_output_cancel_release_only_private_storage() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(32 + SCRATCH_BYTES - 1);
    assert!(matches!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    let (arbiter, key, owner) = budget(4096);
    let flag = Arc::new(AtomicBool::new(true));
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            Some(&flag)
        )
        .is_err()
    );
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    flag.store(false, Ordering::Release);
    let observed_arbiter = arbiter.clone();
    let observed_key = key.clone();
    let observed_flag = flag.clone();
    AFTER_FIRST_OUTPUT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            assert_eq!(
                observed_arbiter.snapshot(&observed_key).unwrap().reserved,
                32 + SCRATCH_BYTES
            );
            observed_flag.store(true, Ordering::Release);
        }))
    });
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            Some(&flag)
        )
        .is_err()
    );
    let final_state = arbiter.snapshot(&key).unwrap();
    assert_eq!(final_state.reserved, 0);
    assert!(!final_state.fenced);
}

#[test]
fn nonfinite_used_words_and_finite_input_overflow_refuse_without_outputs() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let (arbiter, key, owner) = budget(4096);
    let mut a = [1.0; 32];
    let b = [2.0; 16];
    a[19] = f32::INFINITY;
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        )
        .is_err()
    );
    a.fill(f32::MAX);
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        )
        .is_err()
    );
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13182; SHA256 `1f17327c62f36e8d813aab902a907a01233d84967a92ed84033e6ebc4a03ac13`.

````text
//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
//!
//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.

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
    Ok(BulkF32Completed {
        values,
        source: request.source,
        shape,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/proposed/crates/zerogravity-cpu/src/math.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13304; SHA256 `16b2a904fe208bbdeecc8de2c072c885ff12839041cb42b53675a50bb4d71526`.

````text
//! Deterministic CPU reference operations used to validate compiled regions.
//!
//! These are mathematical building blocks, NOT full Qwen GDN or Gemma MoE layer
//! implementations. Model recipes will register exact higher-level semantics.

use crate::{Error, Result};

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Malformed(message.into()))
    }
}

/// Reference row-major matrix vector product, using diagnostic f64 accumulation.
pub fn matvec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    check(
        rows.checked_mul(cols) == Some(matrix.len()),
        "matvec matrix shape",
    )?;
    check(input.len() == cols, "matvec input size")?;
    check(rows <= 16_000_000, "matvec output too large")?;
    let mut out = Vec::new();
    out.try_reserve_exact(rows)
        .map_err(|_| Error::Limit("matvec output allocation failed".into()))?;
    for row in matrix.chunks_exact(cols.max(1)).take(rows) {
        let acc = row
            .iter()
            .zip(input)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        out.push(acc as f32);
    }
    if cols == 0 {
        out.resize(rows, 0.0);
    }
    Ok(out)
}

/// RMS norm reference, without residual addition or model-specific postscaling.
pub fn rms_norm(values: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    check(
        values.len() == weight.len() && !values.is_empty(),
        "rms norm shape",
    )?;
    check(epsilon.is_finite() && epsilon > 0.0, "rms norm epsilon")?;
    let sum: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    let inv = (sum / values.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    Ok(values
        .iter()
        .zip(weight)
        .map(|(x, w)| (*x as f64 * inv * f64::from(*w)) as f32)
        .collect())
}

/// Stable probability normalization. Reject NaN and infinities instead of masking
/// an unknown model's attention or router semantics.
pub fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    check(!values.is_empty(), "softmax empty")?;
    check(
        values.iter().all(|x| x.is_finite()),
        "softmax requires finite logits",
    )?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = values.iter().map(|v| f64::from(*v - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    check(sum > 0.0 && sum.is_finite(), "softmax unstable")?;
    Ok(weights.into_iter().map(|v| (v / sum) as f32).collect())
}

/// Router top-k selection is separated from model-specific scoring,
/// renormalization, and shared-expert accumulation rules.
pub fn top_k(logits: &[f32], k: usize) -> Result<Vec<usize>> {
    check(k > 0 && k <= logits.len(), "top-k size")?;
    check(
        logits.iter().all(|x| x.is_finite()),
        "top-k requires finite logits",
    )?;
    let mut rank: Vec<usize> = (0..logits.len()).collect();
    rank.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    rank.truncate(k);
    Ok(rank)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn gelu_tanh(x: f32) -> f32 {
    let a = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (a * (x + 0.044715 * x * x * x)).tanh())
}

/// One causal attention query over keys/values at positions 0..T.
/// `window=None` is full attention; `window=Some(n)` keeps the last n
/// positions inclusive of query position. This is diagnostic CPU math.
pub fn attention_one(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_position: usize,
    window: Option<usize>,
) -> Result<Vec<f32>> {
    let d = query.len();
    check(
        d > 0 && keys.len() == values.len() && keys.len().is_multiple_of(d),
        "attention head geometry",
    )?;
    let tokens = keys.len() / d;
    check(
        tokens > 0 && query_position < tokens,
        "attention query position",
    )?;
    if let Some(n) = window {
        check(n > 0, "sliding attention window")?;
    }
    let start = window.map_or(0, |n| (query_position + 1).saturating_sub(n));
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = Vec::new();
    for t in start..=query_position {
        let v = keys[t * d..(t + 1) * d]
            .iter()
            .zip(query)
            .map(|(&k, &q)| f64::from(k) * f64::from(q))
            .sum::<f64>();
        scores.push((v * scale) as f32);
    }
    let probs = softmax(&scores)?;
    let mut out = vec![0.0; d];
    for (i, p) in probs.iter().enumerate() {
        let v = &values[(start + i) * d..(start + i + 1) * d];
        for (dst, &x) in out.iter_mut().zip(v) {
            *dst += x * p;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_norm() {
        assert_eq!(
            matvec(&[1., 2., 3., 4.], 2, 2, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        assert!(matvec(&[1., 2.], 2, 2, &[1., 2.]).is_err());
        assert_eq!(matvec(&[], 3, 0, &[]).unwrap(), [0., 0., 0.]);
        let norm = rms_norm(&[3., 4.], &[1., 1.], 1e-6).unwrap();
        assert!((norm[0] - 0.848528).abs() < 1e-5);
        assert!((norm[1] - 1.131_37).abs() < 1e-5);
    }

    #[test]
    fn stable_softmax_and_tie_routing() {
        let probs = softmax(&[1000., 1000., 999.]).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert_eq!(top_k(&[3., 4., 4., 3.], 2).unwrap(), [1, 2]);
        assert!(top_k(&[f32::NAN, 1.], 1).is_err());
        assert!(softmax(&[f32::NEG_INFINITY]).is_err());
        assert!((silu(0.0)).abs() < 1e-9);
        assert!((gelu_tanh(0.0)).abs() < 1e-9);
    }

    #[test]
    fn sliding_vs_global_attention_and_future_mask() {
        let q = [1.];
        let k = [0., 0., 0., 0.];
        let v = [1., 2., 3., 100.];
        let global = attention_one(&q, &k, &v, 2, None).unwrap();
        let sliding = attention_one(&q, &k, &v, 2, Some(2)).unwrap();
        assert!((global[0] - 2.0).abs() < 1e-6);
        assert!((sliding[0] - 2.5).abs() < 1e-6);
        assert!(attention_one(&q, &k, &v, 1, Some(0)).is_err());
    }
}

/// Deterministic unfiltered logits/log-probability audit. These are produced
/// *before* temperature, penalties, truncation or grammar; a sampler's
/// post-filter probabilities are not an equivalent numerical oracle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogitRank {
    pub id: u32,
    pub delta: f32,
    pub logprob: f64,
}

pub fn top_logit_ranks(logits: &[f32], n: usize) -> Result<Vec<LogitRank>> {
    if logits.is_empty() || n == 0 || n > 128 || logits.len() > u32::MAX as usize {
        return Err(Error::Malformed("invalid logits audit width/rank".into()));
    }
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed("nonfinite logits in audit".into()));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|x| f64::from(*x - max).exp())
        .sum::<f64>();
    let ln_sum = sum.ln();
    if !ln_sum.is_finite() {
        return Err(Error::Malformed("nonfinite logits normalization".into()));
    }
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&i, &j| logits[j].total_cmp(&logits[i]).then_with(|| i.cmp(&j)));
    Ok(indices
        .into_iter()
        .take(n)
        .map(|id| {
            let delta = logits[id] - max;
            LogitRank {
                id: id as u32,
                delta,
                logprob: f64::from(delta) - ln_sum,
            }
        })
        .collect())
}

#[cfg(test)]
mod logit_audit_tests {
    use super::*;
    #[test]
    fn test_raw_log_probs_pre_sampler() {
        let ranks = top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 3).unwrap();
        assert_eq!(ranks.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 0]);
        assert_eq!(ranks[0].delta, 0.0);
        assert_eq!(ranks[2].delta, -2.0);
        assert!(
            (ranks.iter().map(|r| r.logprob.exp()).sum::<f64>()
                + top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 4).unwrap()[3]
                    .logprob
                    .exp()
                - 1.0)
                .abs()
                < 1e-10
        );
        assert!(top_logit_ranks(&[0.0, f32::NAN], 2).is_err());
    }
}

#[cfg(test)]
mod long_context_reference_tests {
    use super::*;

    #[test]
    fn thirty_two_thousand_causal_attention_positions_vs_independent_double_oracle() {
        const CONTEXT: usize = 32_768;
        const WIDTH: usize = 4;
        let query = [0.4, -0.7, 0.2, 0.1];
        let keys = (0..=CONTEXT)
            .flat_map(|t| (0..WIDTH).map(move |i| (((t * 7 + i * 37) % 113) as f32 - 56.0) / 97.0))
            .collect::<Vec<_>>();
        let mut values = (0..=CONTEXT)
            .flat_map(|t| {
                (0..WIDTH).map(move |i| (((t * 13 + i * 17) % 199) as f32 - 99.0) / 201.0)
            })
            .collect::<Vec<_>>();
        // Future token must never contribute; sentinel catches mask bugs.
        values[CONTEXT * WIDTH..].fill(1_000_000.0);
        let actual = attention_one(&query, &keys, &values, CONTEXT - 1, None).unwrap();
        let window = attention_one(&query, &keys, &values, CONTEXT - 1, Some(1024)).unwrap();
        let independently_evaluate = |start: usize| {
            let mut scores = Vec::new();
            for t in start..CONTEXT {
                let score = query
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| f64::from(q) * f64::from(keys[t * WIDTH + i]))
                    .sum::<f64>()
                    / (WIDTH as f64).sqrt();
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            std::array::from_fn::<_, WIDTH, _>(|i| {
                (weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(values[(start + j) * WIDTH + i]))
                    .sum::<f64>()
                    / sum) as f32
            })
        };
        let global_oracle = independently_evaluate(0);
        let sliding_oracle = independently_evaluate(CONTEXT - 1024);
        for i in 0..WIDTH {
            assert!(
                (actual[i] - global_oracle[i]).abs() < 2e-6,
                "full attention reference mismatch in channel {i}"
            );
            assert!(
                (window[i] - sliding_oracle[i]).abs() < 2e-6,
                "sliding attention reference mismatch in channel {i}"
            );
        }
    }
}

/// GGML Haswell/AVX2 scalar-equivalent approximate exponential used in the
/// source execution mode's SIMD SiLU implementation. This is *not* the
/// mathematical exp reference: its FMA polynomial and float-bit exponent
/// construction are part of the requested source-numerical specialization.
///
/// The common finite model gate range is handled by the exact polynomial
/// branch. For other finite ranges, follow the original SIMD scaling branches;
/// NaN/Inf values are rejected by the model state boundary before this stage.
pub fn ggml_source_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return x.exp();
    }
    let r = f32::from_bits(0x4b40_0000);
    let z = x.mul_add(f32::from_bits(0x3fb8_aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(
        f32::from_bits(0x35bf_be8e),
        (-n).mul_add(f32::from_bits(0x3f31_7200), x),
    );
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = f32::from_bits(0x3c07_2010)
        .mul_add(b, f32::from_bits(0x3d2b_9f17))
        .mul_add(
            u,
            f32::from_bits(0x3e2a_af33).mul_add(b, f32::from_bits(0x3eff_fedb)),
        )
        .mul_add(u, f32::from_bits(0x3f7f_fff6) * b);
    if n.abs() <= 126.0 {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// Opt-in, model-recipe-specific source-compatible SiLU on f32. This matches
/// GGML's vector approximation, rather than silently redefining generic SiLU.
pub fn ggml_source_silu(x: f32) -> f32 {
    x / (1.0 + ggml_source_exp(0.0 - x))
}

#[cfg(test)]
mod ggml_source_silu_tests {
    use super::*;
    #[test]
    fn source_approx_is_bounded_and_retains_general_silu_behavior() {
        for integer in -100..=100 {
            let x = integer as f32 / 10.0;
            let y = ggml_source_silu(x);
            assert!(y.is_finite());
            assert!(
                (y - silu(x)).abs() <= 1e-5,
                "GGML source approximation unexpectedly divergent at {x}"
            );
        }
        assert_eq!(ggml_source_silu(0.0), 0.0);
    }
}

/// Explicit pinned GGML bulk-F32 diagnostic; default source math is unchanged.
pub mod haswell_bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/run_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5557; SHA256 `e82614f3ecc80324e947fbeed431efde0f90dd6ba15616f3d5aed0c450b7f2ec`.

````text
"""ROOT ONLY: bounded owned Rust primitive replay; no provider/model invocation."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time
from contracts import HERE, digest, frozen, load, parents, read, require, write
from prepare import verify_transport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--build-attestation', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    build = read(args.build_attestation)
    binary = digest(args.binary, 64 * 1024 * 1024)
    require(args.binary.is_absolute() and binary['sha256'] == args.binary_sha256,
            'actual bounded fresh ROOT binary')
    require(build['status'] == 'ROOT_COMPILED_BULK_F32_PRIMITIVE_ONLY' and build['build_exit_code'] == 0
            and build['freeze_sha256'] == pin and build['binary'] == str(args.binary)
            and build['binary_digest'] == binary and len(build['effective_lock_sha256']) == 64
            and build['main_digest'] == digest(HERE / 'main.rs'), 'actual ROOT build/source binding')
    proposed = HERE / 'proposed'
    actual_files = []
    for path in sorted(proposed.rglob('*.rs')):
        actual = HERE.parents[2] / path.relative_to(proposed)
        expected = digest(path)
        require(digest(actual) == expected, 'ROOT actual merged primitive bytes')
        actual_files.append({'path':str(actual),'digest':expected})
    require(build['production_files'] == actual_files, 'actual ROOT compiled production source pins')
    require(args.fixtures.is_absolute() and args.fixtures.parent.resolve() == HERE,
            'own authentic fixture transport')
    parents()
    verify_transport(args.fixtures)
    fixture = digest(args.fixtures / 'manifest.json')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four available CPUs')
    require(args.new_run.is_absolute() and args.new_run.parent.resolve() == HERE
            and not args.new_run.exists(), 'fresh own run')
    args.new_run.mkdir(mode=0o700)
    output = args.new_run / 'output'
    argv = [str(args.binary),str(args.fixtures),str(output)]
    receipt = {'status':'UNQUALIFIED','freeze_sha256':pin,'argv':argv,'binary':binary,
               'build_attestation':build,'fixture_manifest':fixture,'actual_affinity_expected':cpus,
               'wall_seconds':300,'as_limit_bytes':1024**3,'math_scope':'captured primitive only',
               'provider_model_GPU_execution':False}
    write(args.new_run / 'prepared-invocation.json', receipt)
    h = load(HERE / 'owned_lifetime.py','bulk_f32_owned_lifetime')
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','RAYON_NUM_THREADS':'4',
           'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'-1',
           'NVIDIA_VISIBLE_DEVICES':'void','LD_PRELOAD':'','LD_AUDIT':''}
    def limits():
        os.sched_setaffinity(0,cpus)
        for kind,value in ((resource.RLIMIT_AS,1024**3),(resource.RLIMIT_CPU,1200),
                           (resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CORE,0),
                           (resource.RLIMIT_FSIZE,8*1024**2),(resource.RLIMIT_STACK,8*1024**2)):
            resource.setrlimit(kind,(value,value))
    child = log = failure = None
    start = time.monotonic()
    try:
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True,preexec_fn=limits)
        receipt['owned_cli_pid'] = child.pid
        log = h.Log(child.stdout,args.new_run / 'execution.log')
        receipt['exit_code'] = log.observe(child,300)
    except BaseException as error:
        failure = error
    finally:
        errors = receipt['cleanup_errors'] = []
        if child is not None:
            try:
                h.cleanup_group(child,receipt)
            except BaseException as error:
                errors.append('owned foreground:'+repr(error))
        if log is not None:
            try:
                log.finish()
            except BaseException as error:
                errors.append('log:'+repr(error))
            finally:
                receipt['log_cleanup'] = {'pipe_eof':log.eof,'all_closed':log.closed,
                    'total_bytes':log.total,'retained_bytes':log.retained}
        elif child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as error:
                errors.append('uninitialized pipe:'+repr(error))
        receipt['elapsed_seconds_not_benchmark'] = time.monotonic()-start
    if failure is not None:
        receipt['failure'] = repr(failure)
    good = failure is None and receipt.get('exit_code') == 0 and not receipt['cleanup_errors']
    if good:
        receipt['status'] = 'EXECUTED_PRIMITIVE_ONLY_REQUIRES_VALIDATION'
    write(args.new_run / 'invocation.json', receipt)
    require(good, 'failed source/control/cleanup attempt retained; no retry/fallback')
    require(digest(args.binary) == binary and digest(args.fixtures / 'manifest.json') == fixture,
            'unchanged binary/transport after known retirement')
    parents()
    verify_transport(args.fixtures)
    require(frozen() == pin, 'unchanged source after replay')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/test_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4834; SHA256 `a84e7590f87da81e40c339b87f6e0d71fd27a43c632083d327d884bf58430feb`.

````text
"""ROOT-only synthetic receipt regressions; no operator/data/model proof."""
import copy
import unittest
from validate import META, STAGING, WORDS, MODEL, report, invocation


def ledger(metadata=0,staging=0,output=0):
    return {'reserved':metadata+staging+output+(256 if output else 0),
            'reservations':int(metadata>0)+int(staging>0)+int(output>0),
            'fenced':False,'metadata':metadata,'staging':staging,'output':output,
            'scratch':256 if output else 0}


def synthetic_report():
    return {'status':'PASS_CAPTURED_DEFAULT_BULK_F32_ONLY','profile':'GgmlD812350HaswellTinyBlas',
            'output_words':132096,'default_production_changed':False,'historical_prefix_recovered':False,
            'model_or_state_qualified':False,'source_memory_admission_scope':
            'configured diagnostic host, actual logical payload reservations; no whole physical RAM claim',
            'actual_affinity':[0,1,2,3],'mxcsr_before':0x1f80,'mxcsr_after':0x1fa0,
            'before':ledger(),'final':ledger(),'cases':[
                {'case':key,'model_sha256':MODEL[key],'words':WORDS[key],'unequal_words':0,
                 'pre_cancel_refused':True,'capacity_refused':True,'expected_opened_after_math':True,
                 'physical_left_span_words':4317184 if key=='av' else 32768,
                 'live':ledger(META,STAGING,WORDS[key]*4),
                 'retained_output':ledger(META,0,WORDS[key]*4),'after_payload_drop':ledger(META)}
                for key in ('av','alpha')]}


class Contracts(unittest.TestCase):
    def reject(self,change):
        value=synthetic_report();change(value)
        with self.assertRaises((ValueError,KeyError,TypeError)):
            report(value)

    def test_valid_shape_is_explicitly_synthetic(self):
        report(synthetic_report())

    def test_all_cases_and_complete_output_counts_required(self):
        self.reject(lambda v:v['cases'].pop())
        self.reject(lambda v:v['cases'].append(copy.deepcopy(v['cases'][0])))
        self.reject(lambda v:v['cases'][0].__setitem__('words',131071))
        self.reject(lambda v:v['cases'][1].__setitem__('words',True))

    def test_source_and_stride_cannot_be_retagged(self):
        self.reject(lambda v:v['cases'][0].__setitem__('model_sha256',MODEL['alpha']))
        self.reject(lambda v:v['cases'][0].__setitem__('physical_left_span_words',131072))

    def test_signed_zero_or_any_raw_mismatch_fails(self):
        self.reject(lambda v:v['cases'][1].__setitem__('unequal_words',1))
        self.reject(lambda v:v['cases'][1].__setitem__('unequal_words',False))

    def test_aggregate_capacity_and_retained_ticket_required(self):
        self.reject(lambda v:v['cases'][0].__setitem__('live',ledger(META,STAGING)))
        self.reject(lambda v:v['cases'][1].__setitem__('retained_output',ledger(META)))
        self.reject(lambda v:v['cases'][1]['live'].__setitem__('scratch',0))
        self.reject(lambda v:v['cases'][1]['live'].__setitem__('reservations',True))
        self.reject(lambda v:v.__setitem__('final',ledger(META)))

    def test_real_negative_conditions_and_expected_isolation_required(self):
        for field in ('pre_cancel_refused','capacity_refused','expected_opened_after_math'):
            with self.subTest(field=field):
                self.reject(lambda v:v['cases'][0].__setitem__(field,False))

    def test_fp_policy_and_no_default_or_history_promotion(self):
        self.reject(lambda v:v.__setitem__('mxcsr_after',0x9f80))
        self.reject(lambda v:v.__setitem__('mxcsr_after',0x3f80))
        self.reject(lambda v:v.__setitem__('historical_prefix_recovered',True))
        self.reject(lambda v:v.__setitem__('default_production_changed',True))
        self.reject(lambda v:v.__setitem__('model_or_state_qualified',True))

    def test_unknown_lifecycle_refuses(self):
        v={'status':'EXECUTED_PRIMITIVE_ONLY_REQUIRES_VALIDATION','exit_code':0,
           'cleanup_errors':[],'provider_model_GPU_execution':False,'math_scope':'captured primitive only',
           'wall_seconds':300,'as_limit_bytes':1024**3,'owned_cli_pid':12,
           'foreground_cleanup':{'pgid':12,'leader_reaped':True,'group_absent':True,'stage_errors':[],
             'all_nonzero_signals_before_reap':True,'kill_attempts_before_reap':2,'leader_reap_attempts':1},
           'log_cleanup':{'pipe_eof':True,'all_closed':True,'total_bytes':0,'retained_bytes':0}}
        invocation(v)
        for field in ('leader_reaped','group_absent','all_nonzero_signals_before_reap'):
            bad=copy.deepcopy(v);bad['foreground_cleanup'][field]=False
            with self.assertRaises(ValueError):invocation(bad)
        bad=copy.deepcopy(v);bad['log_cleanup']['pipe_eof']=False
        with self.assertRaises(ValueError):invocation(bad)


if __name__=='__main__':
    unittest.main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/frozen/validate.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6074; SHA256 `abab362a92a7347f09d7f9ab8cf50da2ca4e6e9f120f0f56b7f88f7af2dcf5c3`.

````text
"""ROOT ONLY independent exact-byte/default-source/ledger/lifetime validator."""
import argparse
from pathlib import Path
from contracts import HERE, CASES, digest, frozen, parents, read, require, write
from prepare import verify_transport

META = 4 * 1024**2
STAGING = 32 * 1024**2
WORDS = {'av':131072,'alpha':1024}
MODEL = {'av':'57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
         'alpha':'91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6'}

def ledger(value, metadata=0, staging=0, output=0):
    scratch = 256 if output else 0
    require(all(type(value[key]) is int for key in
                ('reserved','reservations','output','scratch','staging','metadata'))
            and value['fenced'] is False, 'strict actual ledger value types')
    require(value == {'reserved':metadata+staging+output+scratch,
                      'reservations':int(metadata>0)+int(staging>0)+int(output>0),'fenced':False,
                      'output':output,'scratch':scratch,'staging':staging,'metadata':metadata},
            'exact original class/retained-output ledger')

def report(value):
    require(value['status'] == 'PASS_CAPTURED_DEFAULT_BULK_F32_ONLY'
            and value['profile'] == 'GgmlD812350HaswellTinyBlas'
            and value['output_words'] == 132096 and type(value['output_words']) is int,
            'complete explicit primitive scope')
    for field in ('default_production_changed','historical_prefix_recovered','model_or_state_qualified'):
        require(value[field] is False, 'no capability/default/history promotion')
    require(value['source_memory_admission_scope'] ==
            'configured diagnostic host, actual logical payload reservations; no whole physical RAM claim',
            'honest original admission scope')
    cpus = value['actual_affinity']
    require(len(cpus)==4 and len(set(cpus))==4 and all(type(n)is int and 0<=n<1024 for n in cpus),
            'actual four-CPU affinity')
    for field in ('mxcsr_before','mxcsr_after'):
        require(type(value[field]) is int and value[field]&0xffc0 == 0x1f80,
                'actual RNE/masked/noFTZ/noDAZ')
    ledger(value['before']); ledger(value['final'])
    require([x['case']for x in value['cases']] == list(CASES), 'complete distinct authentic case population')
    for row in value['cases']:
        key = row['case']; size = WORDS[key]*4
        require(row['model_sha256']==MODEL[key] and type(row['words'])is int and row['words']==WORDS[key]
                and type(row['unequal_words'])is int and row['unequal_words']==0,
                'actual complete raw trained default result')
        require(row['pre_cancel_refused'] is True and row['capacity_refused'] is True
                and row['expected_opened_after_math'] is True, 'real private-call refusal and expected isolation')
        require(row['physical_left_span_words'] == (4317184 if key=='av' else 32768),
                'original strided full view, no dense substitute')
        ledger(row['live'],META,STAGING,size)
        ledger(row['retained_output'],META,0,size)
        ledger(row['after_payload_drop'],META)

def invocation(value):
    require(value['status']=='EXECUTED_PRIMITIVE_ONLY_REQUIRES_VALIDATION'
            and value['exit_code']==0 and value['cleanup_errors']==[]
            and value['provider_model_GPU_execution'] is False
            and value['math_scope']=='captured primitive only'
            and value['wall_seconds']==300 and value['as_limit_bytes']==1024**3,
            'actual primitive-only invocation/finite scope')
    g=value['foreground_cleanup']; log=value['log_cleanup']
    require(g['pgid']==value['owned_cli_pid'] and g['leader_reaped'] is True
            and g['group_absent'] is True and g['stage_errors']==[]
            and g['all_nonzero_signals_before_reap'] is True
            and g['kill_attempts_before_reap']==2 and g['leader_reap_attempts']in(1,2),
            'actual owned unreaped-group retirement')
    require(log['pipe_eof'] is True and log['all_closed'] is True
            and log['total_bytes']==log['retained_bytes']<=4*1024**2,
            'complete bounded owned pipe EOF/close')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    args=p.parse_args()
    pin=frozen()
    require(args.run.is_absolute()and args.run.parent.resolve()==HERE
            and args.fixtures.is_absolute()and args.fixtures.parent.resolve()==HERE,'owned ROOT artifacts')
    parents()
    manifest=verify_transport(args.fixtures)
    called=read(args.run/'invocation.json'); invocation(called)
    require(called['freeze_sha256']==pin and called['fixture_manifest']==digest(args.fixtures/'manifest.json')
            and called['binary']==digest(called['argv'][0]) and called['argv'][1:]==
            [str(args.fixtures),str(args.run/'output')], 'actual source/input/binary/output binding')
    value=read(args.run/'output/report.json'); report(value)
    require(value['actual_affinity']==called['actual_affinity_expected'],'same observed CPU set')
    for row in manifest['cases']:
        key=row['case']; actual=args.run/'output'/f'{key}.actual.f32le'
        expected=args.fixtures/row['expected']['file']
        require(digest(actual,1024**2)==row['expected']['digest']==digest(expected,1024**2),
                'complete exact default-capture bytes including all zero signs')
        require(actual.read_bytes()==expected.read_bytes(), 'independent byte equality, no tolerances')
    require(frozen()==pin,'same exact reviewed source')
    write(args.run/'validation.json',{'status':'PASS_CAPTURED_DEFAULT_BULK_F32_ONLY','freeze_sha256':pin,
          'words':132096,'cases':list(CASES),'invocation':digest(args.run/'invocation.json'),
          'actual_report':digest(args.run/'output/report.json'),'model_state_or_history_qualified':False,
          'default_or_GPU_profile_changed':False,'whole_physical_RAM_admission_qualified':False})

if __name__=='__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/main.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 15344; SHA256 `c4e96a19bb2e44e8ca5991af31f0fcd66747730cfab62b60ccdf28c6255c1758`.

````text
//! ROOT-only captured primitive replay; no model/provider/session/token engine.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
use serde::Deserialize;
use serde_json::{Value, json};
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use zerogravity_core::accelerator::SourceIdentity;
use zerogravity_core::resource_arbiter::{
    AllocationClass, AllocationRequest, PhysicalDeviceKey, ResourceArbiter, ResourceOwner,
};
use zerogravity_cpu::math::haswell_bulk_f32::{
    BulkF32Admission, BulkF32Profile, BulkF32Request, F32View, execute,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const META: u64 = 4 * 1024 * 1024;
const STAGING: u64 = 32 * 1024 * 1024;
const LIMIT: u64 = 128 * 1024 * 1024;

fn need(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Digest {
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Part {
    file: String,
    name: String,
    shape: [usize; 4],
    strides: [usize; 4],
    digest: Digest,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    case: String,
    model_sha256: String,
    left: Part,
    right: Part,
    expected: Part,
    source_provenance: Value,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    status: String,
    profile: String,
    cases: Vec<Case>,
    arithmetic_executed: bool,
    expected_from_candidate: bool,
    gap_storage: String,
}
fn snapshot(arbiter: &ResourceArbiter, key: &PhysicalDeviceKey) -> Result<Value> {
    let s = arbiter.snapshot(key)?;
    Ok(
        json!({"reserved":s.reserved,"reservations":s.reservations,"fenced":s.fenced,
        "output":s.per_class.get(&AllocationClass::Output).copied().unwrap_or(0),
        "scratch":s.per_class.get(&AllocationClass::Scratch).copied().unwrap_or(0),
        "staging":s.per_class.get(&AllocationClass::TransferStaging).copied().unwrap_or(0),
        "metadata":s.per_class.get(&AllocationClass::Other).copied().unwrap_or(0)}),
    )
}
fn mxcsr() -> u32 {
    let mut value = 0u32;
    // SAFETY: read-only CPU-control instruction writes one valid local word.
    unsafe {
        std::arch::asm!("stmxcsr [{pointer}]", pointer = in(reg) &mut value,
        options(nostack, preserves_flags));
    }
    value
}
fn affinity() -> Result<Vec<usize>> {
    let mut mask = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    need(
        unsafe { libc::sched_getaffinity(0, std::mem::size_of_val(&mask), &mut mask) } == 0,
        "actual CPU mask",
    )?;
    let actual = (0..libc::CPU_SETSIZE as usize)
        .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
        .collect::<Vec<_>>();
    need(actual.len() == 4, "ROOT exact four available CPUs")?;
    Ok(actual)
}
fn read_words(folder: &Path, part: &Part) -> Result<Vec<f32>> {
    need(
        Path::new(&part.file).components().count() == 1 && !part.file.starts_with('.'),
        "closed file component",
    )?;
    need(
        part.digest.bytes > 0
            && part.digest.bytes <= 1024 * 1024
            && part.digest.bytes.is_multiple_of(4)
            && part.digest.sha256.len() == 64,
        "bounded complete raw F32 file",
    )?;
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(folder.join(&part.file))?;
    let before = file.metadata()?;
    need(
        before.is_file() && before.len() == part.digest.bytes as u64,
        "actual exact raw length",
    )?;
    let mut raw = Vec::new();
    raw.try_reserve_exact(part.digest.bytes)?;
    (&mut file)
        .take(part.digest.bytes as u64 + 1)
        .read_to_end(&mut raw)?;
    use std::os::unix::fs::MetadataExt;
    let after = file.metadata()?;
    need(
        raw.len() == part.digest.bytes
            && before.dev() == after.dev()
            && before.ino() == after.ino()
            && before.len() == after.len()
            && before.mtime_nsec() == after.mtime_nsec()
            && before.mtime() == after.mtime()
            && before.ctime_nsec() == after.ctime_nsec()
            && before.ctime() == after.ctime(),
        "same immutable input FD",
    )?;
    let mut words = Vec::new();
    words.try_reserve_exact(raw.len() / 4)?;
    for value in raw.chunks_exact(4) {
        let v = f32::from_le_bytes(value.try_into()?);
        need(
            v.is_finite(),
            "actual finite F32 words; no canonicalization",
        )?;
        words.push(v);
    }
    Ok(words)
}
fn expand(words: &[f32], part: &Part) -> Result<Vec<f32>> {
    let logical = part
        .shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v).ok_or("logical overflow"))?;
    need(words.len() == logical, "complete captured logical source")?;
    let span = part
        .shape
        .iter()
        .zip(part.strides)
        .try_fold(1usize, |n, (&v, stride)| {
            v.checked_sub(1)
                .and_then(|v| v.checked_mul(stride))
                .and_then(|v| n.checked_add(v))
                .ok_or("source span overflow")
        })?;
    need(
        span <= 8 * 1024 * 1024 && part.strides[0] == 1,
        "bounded original physical span",
    )?;
    let mut result = Vec::new();
    result.try_reserve_exact(span)?;
    result.resize(span, 0.0);
    let mut at = 0;
    for sequence in 0..part.shape[3] {
        for head in 0..part.shape[2] {
            for row in 0..part.shape[1] {
                let offset =
                    row * part.strides[1] + head * part.strides[2] + sequence * part.strides[3];
                result[offset..offset + part.shape[0]]
                    .copy_from_slice(&words[at..at + part.shape[0]]);
                at += part.shape[0];
            }
        }
    }
    Ok(result)
}
fn run(fixtures: &Path, output: &Path) -> Result<()> {
    let cpus = affinity()?;
    let fp_before = mxcsr();
    need(
        fp_before & 0xffc0 == 0x1f80,
        "RNE/masked/noFTZ/noDAZ source controls",
    )?;
    need(!output.exists(), "fresh ROOT output")?;
    std::fs::create_dir(output)?;
    let arbiter = ResourceArbiter::process_shared();
    let key = PhysicalDeviceKey::new("host", "bulk-f32-primitive-qualification")?;
    arbiter.register_matching(key.clone(), LIMIT)?;
    let owner = ResourceOwner::new("captured-default-GGUF-F32", "ROOT-only")?;
    let before = snapshot(&arbiter, &key)?;
    let metadata = arbiter.admit(
        &key,
        owner.clone(),
        &[AllocationRequest {
            kind: AllocationClass::Other,
            bytes: META,
        }],
    )?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(fixtures.join("manifest.json"))?;
    need(
        file.metadata()?.len() <= 512 * 1024,
        "bounded two-case metadata",
    )?;
    let manifest: Manifest = serde_json::from_reader(file.take(512 * 1024 + 1))?;
    need(
        manifest.status == "AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY"
            && manifest.profile == "GgmlD812350HaswellTinyBlas"
            && !manifest.arithmetic_executed
            && !manifest.expected_from_candidate
            && manifest.cases.len() == 2
            && manifest.gap_storage
                == "actual dense logical V restored at original strides; uncaptured gaps zero-filled",
        "authentic byte-only input scope",
    )?;
    let mut reports = Vec::new();
    for (index, case) in manifest.cases.iter().enumerate() {
        let (id, model, a_shape, a_stride, b_shape, b_stride, out_shape, a_name, b_name) =
            if index == 0 {
                (
                    "av",
                    "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
                    [256, 256, 2, 1],
                    [1, 8448, 2162688, 4325376],
                    [256, 64, 8, 1],
                    [1, 256, 16384, 131072],
                    [256, 64, 8, 1],
                    "cache_v_l3 (view) (permuted)",
                    "kq_soft_max-3",
                )
            } else {
                (
                    "alpha",
                    "91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6",
                    [2048, 16, 1, 1],
                    [1, 2048, 32768, 32768],
                    [2048, 64, 1, 1],
                    [1, 2048, 131072, 131072],
                    [16, 64, 1, 1],
                    "blk.0.ssm_alpha.weight",
                    "attn_norm-0",
                )
            };
        need(
            case.case == id
                && case.model_sha256 == model
                && case.left.shape == a_shape
                && case.left.strides == a_stride
                && case.right.shape == b_shape
                && case.right.strides == b_stride
                && case.expected.shape == out_shape
                && case.left.name == a_name
                && case.right.name == b_name
                && case.source_provenance.is_object(),
            "exact accepted original operand roles/geometry",
        )?;
        let staging = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::TransferStaging,
                bytes: STAGING,
            }],
        )?;
        let dense_left = read_words(fixtures, &case.left)?;
        let left = expand(&dense_left, &case.left)?;
        drop(dense_left);
        let right = read_words(fixtures, &case.right)?;
        let source = SourceIdentity::for_new_allocation()?;
        let request = BulkF32Request {
            source,
            profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
            left: F32View {
                source,
                name: &case.left.name,
                values: &left,
                offset: 0,
                shape: a_shape,
                strides: a_stride,
            },
            right: F32View {
                source,
                name: &case.right.name,
                values: &right,
                offset: 0,
                shape: b_shape,
                strides: b_stride,
            },
        };
        let admission = || BulkF32Admission {
            arbiter: &arbiter,
            resource: &key,
            owner: &owner,
        };
        let baseline = arbiter.snapshot(&key)?.reserved;
        let flag = AtomicBool::new(true);
        need(
            execute(request, admission(), Some(&flag)).is_err()
                && arbiter.snapshot(&key)?.reserved == baseline,
            "actual pre-cancel unchanged ledger",
        )?;
        let output_bytes = out_shape.iter().product::<usize>() * 4;
        let pressure = arbiter.admit(
            &key,
            owner.clone(),
            &[AllocationRequest {
                kind: AllocationClass::Scratch,
                bytes: LIMIT - baseline - (output_bytes as u64 + 256) + 1,
            }],
        )?;
        let pressure_live = snapshot(&arbiter, &key)?;
        let denied = execute(request, admission(), None);
        need(
            matches!(denied, Err(zerogravity_core::Error::Limit(ref reason))
            if reason == "physical device memory admission refused"),
            "actual original capacity refusal",
        )?;
        need(
            snapshot(&arbiter, &key)? == pressure_live,
            "no output charge after refusal",
        )?;
        drop(pressure);
        let result = execute(request, admission(), None)?;
        need(
            result.shape() == out_shape && result.source() == source,
            "same bound complete output",
        )?;
        let live = snapshot(&arbiter, &key)?;
        need(
            live["reserved"] == META + STAGING + output_bytes as u64 + 256
                && live["output"] == output_bytes as u64
                && live["scratch"] == 256,
            "actual aggregate payload+scratch reservations",
        )?;
        // Expected bytes are opened ONLY AFTER the public Rust math returns.
        let expected = read_words(fixtures, &case.expected)?;
        need(
            expected.len() == result.values().len(),
            "complete expected output cardinality",
        )?;
        let unequal = result
            .values()
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let mut actual = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join(format!("{id}.actual.f32le")))?;
        for word in result.values() {
            actual.write_all(&word.to_le_bytes())?;
        }
        actual.sync_all()?;
        drop(actual);
        drop(expected);
        drop(left);
        drop(right);
        drop(staging);
        let retained = snapshot(&arbiter, &key)?;
        need(
            retained["reserved"] == META + output_bytes as u64 + 256,
            "output retains original ticket after input release",
        )?;
        drop(result);
        need(
            arbiter.snapshot(&key)?.reserved == META,
            "case private payloads reclaimed",
        )?;
        reports.push(json!({"case":id,"model_sha256":model,"words":output_bytes/4,"unequal_words":unequal,
            "pre_cancel_refused":true,"capacity_refused":true,"live":live,"retained_output":retained,
            "after_payload_drop":snapshot(&arbiter,&key)?,"physical_left_span_words":
                a_shape.iter().zip(a_stride).map(|(&n,s)| (n-1)*s).sum::<usize>()+1,
            "expected_opened_after_math":true}));
    }
    drop(metadata);
    let final_state = snapshot(&arbiter, &key)?;
    need(
        final_state["reserved"] == 0 && final_state["fenced"] == false,
        "actual all leases final0",
    )?;
    need(mxcsr() & 0xffc0 == 0x1f80, "FP controls unchanged")?;
    let bit_exact = reports.iter().all(|row| row["unequal_words"] == 0);
    let report = json!({"status":if bit_exact {"PASS_CAPTURED_DEFAULT_BULK_F32_ONLY"} else {"FAIL_SOURCE_BITS"},
        "profile":"GgmlD812350HaswellTinyBlas","before":before,"final":final_state,"cases":reports,
        "actual_affinity":cpus,"mxcsr_before":fp_before,"mxcsr_after":mxcsr(),"output_words":132096,
        "default_production_changed":false,"historical_prefix_recovered":false,"model_or_state_qualified":false,
        "source_memory_admission_scope":"configured diagnostic host, actual logical payload reservations; no whole physical RAM claim"});
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("report.json"))?;
    serde_json::to_writer_pretty(&mut file, &report)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    need(
        bit_exact,
        "raw DEFAULT source output differences retained; no tolerance/fallback",
    )
}
fn main() {
    let args = std::env::args().map(PathBuf::from).collect::<Vec<_>>();
    let result = (|| -> Result<()> {
        need(args.len() == 3, "FIXTURES NEW_OUTPUT")?;
        run(&args[1], &args[2])
    })();
    if let Err(error) = result {
        eprintln!("BULK_F32_UNQUALIFIED:{error}");
        std::process::exit(2);
    }
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/owned_lifetime.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 11840; SHA256 `53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d`.

````text
"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.

Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
before leader reap: its reserved PID prevents accidental process-group reuse.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import time

PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
CAP = 4 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def command(argv, timeout=10):
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=timeout, check=False)
    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
    return result


def leader_status_unreaped(process):
    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if observed is None:
        return None
    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
            "owned leader exit observation")
    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status


class Log:
    """Main-thread bounded drain; no thread/future can survive cleanup."""
    def __init__(self, pipe, path):
        self.pipe, self.path = pipe, Path(path)
        self.selector = self.file = None
        self.retained = self.total = 0
        self.eof = self.closed = False
        self.errors = []
        try:
            self.file = self.path.open("xb")
            os.chmod(self.path, 0o600)
            os.set_blocking(pipe.fileno(), False)
            self.selector = selectors.DefaultSelector()
            self.selector.register(pipe, selectors.EVENT_READ)
        except BaseException:
            if self.selector is not None:
                self.selector.close()
            if self.file is not None:
                self.file.close()
            raise

    def drain(self, timeout):
        for key, _ in self.selector.select(timeout):
            # One bounded block per wake; the exit/deadline remains observable
            # even when an owned writer continuously produces output.
            try:
                raw = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not raw:
                self.selector.unregister(key.fileobj)
                self.eof = True
                continue
            self.total += len(raw)
            keep = raw[:max(0, CAP - self.retained)]
            if keep:
                try:
                    self.file.write(keep)
                    self.retained += len(keep)
                except OSError as error:
                    self.errors.append(repr(error))

    def observe(self, process, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.drain(0.05)
            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
            status = leader_status_unreaped(process)
            if status is not None:
                return status
        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")

    def finish(self):
        failures = []
        try:
            deadline = time.monotonic() + 3
            while not self.eof and time.monotonic() < deadline:
                self.drain(0.05)
            require(self.eof, "owned stdout pipe EOF deadline")
            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
            self.file.flush()
            os.fsync(self.file.fileno())
        except BaseException as error:
            failures.append("drain/evidence: " + repr(error))
        finally:
            # Close all three even when EOF, evidence I/O or one close fails.
            for name, close in (("selector", self.selector.close), ("log", self.file.close),
                                ("pipe", self.pipe.close)):
                try:
                    close()
                except BaseException as error:
                    failures.append(name + ": " + repr(error))
            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
        if failures:
            raise RuntimeError("owned log cleanup: " + "; ".join(failures))


def owned_container(name, nonce, cidfile):
    target, recorded_cid = name, None
    if cidfile.exists():
        cid = cidfile.read_text().strip()
        if cid:
            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
            target = recorded_cid = cid
    exists = command(PODMAN + ["container", "exists", target])
    if exists.returncode == 1:
        return None
    require(exists.returncode == 0, "owned container existence check failed")
    inspected = command(PODMAN + ["container", "inspect", target])
    require(inspected.returncode == 0, "owned container inspect failed")
    value, = json.loads(inspected.stdout)
    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
            "container name/nonce mismatch; no signal")
    cid = value["Id"]
    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
    if recorded_cid is not None:
        require(recorded_cid == cid, "owned CID changed; no signal")
    return cid


def cleanup_container(name, nonce, cidfile):
    errors = []
    try:
        cid = owned_container(name, nonce, cidfile)
    except BaseException as error:
        return ["container owner binding: " + repr(error)]
    if cid is None:
        return errors
    try:
        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
        if stopped.returncode != 0:
            errors.append("owned stop returned nonzero")
    except BaseException as error:
        errors.append("container stop: " + repr(error))
    try:
        # Fresh exact owner binding is required even after a failed/timedout
        # stop. A stop failure must not skip the bounded force-remove attempt.
        if owned_container(name, nonce, cidfile) is not None:
            removed = command(PODMAN + ["rm", "--force", cid], 15)
            require(removed.returncode == 0, "owned container force cleanup failed")
    except BaseException as error:
        errors.append("container force cleanup: " + repr(error))
    try:
        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
    except BaseException as error:
        errors.append("container final absence: " + repr(error))
    return errors


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group_unreaped(process, sig):
    # waitid authenticates that this is still our unreaped child. start_new_session
    # made it the session/group leader; no foreign group is inferred from a PID.
    leader_status_unreaped(process)
    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
            "owned leader session/PGID binding")
    os.killpg(process.pid, sig)


def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
    """Independent bounded stages; no nonzero signal after any reap attempt."""
    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
    errors = []
    try:
        signal_group_unreaped(process, signal.SIGTERM)
        state["term_sent"] = True
    except ProcessLookupError:
        state["term_group_absent_before_reap"] = True
    except BaseException as error:
        errors.append("owned TERM: " + repr(error))
    try:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            observe(process)
            pause(0.05)
    except BaseException as error:
        errors.append("owned grace: " + repr(error))
    # Both independently authenticated KILL attempts precede ANY wait/reap.
    for attempt in range(2):
        state["kill_attempts_before_reap"] += 1
        try:
            signal_group_unreaped(process, signal.SIGKILL)
            state["kill_sent_before_reap"] = True
        except ProcessLookupError:
            state["kill_group_absent_before_reap"] = True
        except BaseException as error:
            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
    state["all_nonzero_signals_before_reap"] = True
    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
    try:
        state["leader_reap_attempts"] += 1
        state["exit_code"] = process.wait(timeout=5)
        state["leader_reaped"] = True
    except BaseException as error:
        errors.append("owned first leader reap: " + repr(error))
    finally:
        if process.returncode is None:
            try:
                state["leader_reap_attempts"] += 1
                state["exit_code"] = process.wait(timeout=5)
                state["leader_reaped"] = True
            except BaseException as error:
                errors.append("owned final leader reap: " + repr(error))
        else:
            state["exit_code"] = process.returncode
            state["leader_reaped"] = True
    try:
        require(state["leader_reaped"], "owned leader was not reaped")
        deadline = time.monotonic() + 5
        while group_exists(process.pid) and time.monotonic() < deadline:
            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
        state["group_absent"] = not group_exists(process.pid)
        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
    except BaseException as error:
        errors.append("owned group absence: " + repr(error))
    state["stage_errors"] = list(errors)
    if errors:
        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))


def cleanup(process, name, nonce, cidfile, receipt):
    errors = []
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("initial owned container cleanup: " + repr(error))
    try:
        cleanup_group(process, receipt)
    except BaseException as error:
        errors.append("owned foreground cleanup: " + repr(error))
    # Rediscover startup races using only the original nonce/name/CID binding.
    try:
        errors.extend(cleanup_container(name, nonce, cidfile))
    except BaseException as error:
        errors.append("final owned container cleanup: " + repr(error))
    try:
        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
        receipt["owned_container_absent_after_reap"] = True
    except BaseException as error:
        errors.append("final owned absence: " + repr(error))
    return errors

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/prepare.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6030; SHA256 `2797547b5cc5f8df21f445d75de114b84587a81fe39dfe5650e8e5bfce779a47`.

````text
"""ROOT ONLY byte transport from immutable original AV/alpha captures/defaults."""
import argparse
from pathlib import Path
from contracts import HERE, AV, ALPHA, CASES, digest, frozen, parents, read, require, write

MODEL = {'av':'57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
         'alpha':'91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6'}

def transport_descriptor(case):
    root = AV if case == 'av' else ALPHA
    validation = read(root / 'root-validation.json')
    if case == 'av':
        require(validation['status'] == 'PUBLIC_AV_DEFAULT_REPLAY_EXACT_CONTROLS_DIAGNOSTIC'
                and validation['default_words'] == 262144, 'accepted original AV public default replay')
        public_invocation = root / 'root-replay.invocation.json'
    else:
        require(validation['status'] == 'TRANSPARENT_PUBLIC_ALPHA_EXPORTED_ENTRY_ROUTE_OBSERVED'
                and validation['all65_defaults_exact_to_actual_captures'] is True
                and validation['all130_plain_observed_outputs_exact'] is True,
                'accepted original alpha public default replay')
        public_invocation = root / 'root-plain.invocation.json'
    manifest = read(root / 'root-fixtures/manifest.json')
    original = manifest['cases'][0]
    require(original['case'] == 'bulk64' and original['mode'] == 'bulk64' and original['call'] == 0,
            'original complete default bulk64')
    def field(key):
        row = original['fields'][key]
        d = row if 'sha256' in row else row['digest']
        return {'file':case + '.' + key + '.f32le','name':row['captured_row']['native_name'],
                'shape':row['captured_row']['native_ne'],
                'strides':[n // 4 for n in row['captured_row']['native_nb']],
                'digest':{k:d[k] for k in ('bytes','sha256')}}, root / 'root-fixtures' / row['file']
    if case == 'av':
        left, left_path = field('v')
        right, right_path = field('p')
    else:
        weight = manifest['weight']
        left = {'file':'alpha.weight.f32le','name':weight['captured_row']['name'],
                'shape':weight['captured_row']['ne'],
                'strides':[n // 4 for n in weight['captured_row']['nb']], 'digest':weight['digest']}
        left_path = root / 'root-fixtures/weight.f32le'
        right,right_path = field('x')
    expected, expected_path = field('expected')
    public = root / ('root-replay' if case == 'av' else 'root-plain') / 'default-bulk64.actual.f32le'
    require(digest(public) == expected['digest'] == digest(expected_path),
            'unchanged public DEFAULT outputs equal authentic capture, before new math')
    value = {'case':case,'model_sha256':MODEL[case],'left':left,'right':right,'expected':expected,
             'source_provenance':{'original_fixture_manifest':str(root / 'root-fixtures/manifest.json'),
                                  'original_fixture_manifest_digest':digest(root / 'root-fixtures/manifest.json'),
                                  'original_public_validation':str(root / 'root-validation.json'),
                                  'original_public_validation_digest':digest(root / 'root-validation.json'),
                                  'original_public_invocation':str(public_invocation),
                                  'original_public_invocation_digest':digest(public_invocation),
                                  'public_default':str(public),'public_default_digest':digest(public)}}
    return value, (left_path,right_path,expected_path)

def verify_transport(folder):
    manifest = read(folder / 'manifest.json')
    require(manifest['status'] == 'AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY'
            and manifest['profile'] == 'GgmlD812350HaswellTinyBlas'
            and [row['case'] for row in manifest['cases']] == list(CASES), 'closed two-case fixture scope')
    for row in manifest['cases']:
        expected, paths = transport_descriptor(row['case'])
        require(row == expected, 'exact original source descriptor/provenance')
        for key,path in zip(('left','right','expected'), paths):
            require(digest(folder / row[key]['file']) == row[key]['digest'] == digest(path),
                    'full original input/output transport bytes')
    return manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new-fixtures', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    parents()
    require(args.new_fixtures.is_absolute() and args.new_fixtures.parent.resolve() == HERE
            and not args.new_fixtures.exists(), 'fresh own fixture directory')
    args.new_fixtures.mkdir(mode=0o700)
    rows = []
    for case in CASES:
        row,paths = transport_descriptor(case)
        rows.append(row)
        for key,path in zip(('left','right','expected'), paths):
            require(digest(path, 8 * 1024 * 1024) == row[key]['digest'], 'actual bounded source copy')
            with path.open('rb') as src, (args.new_fixtures / row[key]['file']).open('xb') as dst:
                remaining = row[key]['digest']['bytes']
                while remaining:
                    raw = src.read(min(remaining,1024 * 1024))
                    require(raw, 'complete bounded source payload')
                    dst.write(raw)
                    remaining -= len(raw)
                require(not src.read(1), 'no source payload growth')
            (args.new_fixtures / row[key]['file']).chmod(0o600)
    write(args.new_fixtures / 'manifest.json', {'status':'AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY',
          'profile':'GgmlD812350HaswellTinyBlas','cases':rows,
          'arithmetic_executed':False,'expected_from_candidate':False,
          'gap_storage':'actual dense logical V restored at original strides; uncaptured gaps zero-filled'})
    verify_transport(args.new_fixtures)
    require(frozen() == pin, 'unchanged preparation/source')
    print('AUTHENTIC_DEFAULT_BULK_F32_TRANSPORT_ONLY')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32/tests.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 7696; SHA256 `dac2aed8e6f9e9f4a8c50a509811a4dffe44a35e961c50b87efa14884e50064a`.

````text
use super::*;
use std::sync::Arc;

fn budget(limit: u64) -> (ResourceArbiter, PhysicalDeviceKey, ResourceOwner) {
    let arbiter = ResourceArbiter::new();
    let key = PhysicalDeviceKey::new("host", "bulk-f32-unit").unwrap();
    arbiter.register(key.clone(), limit).unwrap();
    (
        arbiter,
        key,
        ResourceOwner::new("source-free", "unit").unwrap(),
    )
}
fn request<'a>(source: SourceIdentity, a: &'a [f32], b: &'a [f32]) -> BulkF32Request<'a> {
    BulkF32Request {
        source,
        profile: BulkF32Profile::GgmlD812350HaswellTinyBlas,
        left: F32View {
            source,
            name: "left",
            values: a,
            offset: 0,
            shape: [8, 4, 1, 1],
            strides: [1, 8, 32, 32],
        },
        right: F32View {
            source,
            name: "right",
            values: b,
            offset: 0,
            shape: [8, 2, 1, 1],
            strides: [1, 8, 16, 16],
        },
    }
}

#[test]
fn source_one_chain_and_horizontal_order_have_independent_counterexamples() {
    let a = [1.0; 32];
    let mut b = [0.0; 32];
    b[0] = 16_777_216.0;
    b[8] = 1.0;
    b[16] = -16_777_216.0;
    b[24] = 1.0;
    // Four independent Vec8 partials yield2; the original one-chain yields1.
    assert_eq!(dot(&a, &b).to_bits(), 1.0f32.to_bits());
    let h = [67_108_864.0, 1.0, -67_108_864.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    // Even/odd horizontal pairing yields2; adjacent-pair horizontal gives0.
    assert_eq!(dot(&[1.0; 8], &h).to_bits(), 2.0f32.to_bits());
}

#[test]
fn real_zero_filled_tail_fmas_preserve_source_signed_zero_semantics() {
    let tiny = f32::from_bits(1);
    assert_eq!(dot(&[-tiny; 8], &[tiny; 8]).to_bits(), (-0.0f32).to_bits());
    // Original load_partial fills BOTH unused operands with +0 before FMA.
    assert_eq!(dot(&[-tiny; 9], &[tiny; 9]).to_bits(), 0.0f32.to_bits());
    assert_eq!(dot(&[2.0; 3], &[3.0; 3]).to_bits(), 18.0f32.to_bits());
}

#[test]
fn strided_offset_broadcast_canaries_and_output_lease_are_real() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let mut a = vec![f32::NAN; 128];
    for head in 0..2 {
        for row in 0..4 {
            a[3 + head * 64 + row * 12..3 + head * 64 + row * 12 + 8]
                .fill((head * 4 + row + 1) as f32);
        }
    }
    let mut b = vec![f32::NAN; 66];
    b[1..65].fill(1.0);
    let mut r = request(source, &a, &b);
    r.left.offset = 3;
    r.left.strides = [1, 12, 64, 128];
    r.left.shape = [8, 4, 2, 1];
    r.right.offset = 1;
    r.right.shape = [8, 2, 4, 1];
    r.right.strides = [1, 8, 16, 64];
    let (arbiter, key, owner) = budget(4096);
    let result = execute(
        r,
        BulkF32Admission {
            arbiter: &arbiter,
            resource: &key,
            owner: &owner,
        },
        None,
    )
    .unwrap();
    assert_eq!(result.shape(), [4, 2, 4, 1]);
    for head in 0..4 {
        for column in 0..2 {
            for row in 0..4 {
                assert_eq!(
                    result.values()[row + 4 * (column + 2 * head)],
                    (head / 2 * 4 + row + 1) as f32 * 8.0
                );
            }
        }
    }
    let live = arbiter.snapshot(&key).unwrap();
    assert_eq!(live.reservations, 1);
    assert_eq!(live.per_class[&AllocationClass::Output], 128);
    assert_eq!(live.per_class[&AllocationClass::Scratch], SCRATCH_BYTES);
    assert!(a[0].is_nan() && a[11].is_nan() && b[0].is_nan() && b[65].is_nan());
    assert_eq!(result.source(), source);
    drop(result);
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}

#[test]
fn foreign_source_stride_span_overflow_and_ar1_are_closed() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(4096);
    let valid = request(source, &a, &b);
    let mut cases = Vec::new();
    let mut r = valid;
    r.right.source = SourceIdentity::for_new_allocation().unwrap();
    cases.push(r);
    let mut r = valid;
    r.left.strides[1] = 7;
    cases.push(r);
    let mut r = valid;
    r.left.offset = 1;
    cases.push(r);
    let mut r = valid;
    r.left.offset = usize::MAX;
    cases.push(r);
    let mut r = valid;
    r.left.strides[3] = usize::MAX;
    r.left.shape[3] = 2;
    cases.push(r);
    let mut r = valid;
    r.right.shape[1] = 1;
    r.right.strides = [1, 8, 8, 8];
    cases.push(r);
    let mut r = valid;
    r.right.strides[3] = 17;
    cases.push(r);
    for r in cases {
        assert!(
            execute(
                r,
                BulkF32Admission {
                    arbiter: &arbiter,
                    resource: &key,
                    owner: &owner
                },
                None
            )
            .is_err()
        );
        assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    }
}

#[test]
fn actual_capacity_pre_cancel_and_mid_output_cancel_release_only_private_storage() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let a = [1.0; 32];
    let b = [1.0; 16];
    let (arbiter, key, owner) = budget(32 + SCRATCH_BYTES - 1);
    assert!(matches!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        ),
        Err(Error::Limit(_))
    ));
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    let (arbiter, key, owner) = budget(4096);
    let flag = Arc::new(AtomicBool::new(true));
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            Some(&flag)
        )
        .is_err()
    );
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
    flag.store(false, Ordering::Release);
    let observed_arbiter = arbiter.clone();
    let observed_key = key.clone();
    let observed_flag = flag.clone();
    AFTER_FIRST_OUTPUT.with(|hook| {
        *hook.borrow_mut() = Some(Box::new(move || {
            assert_eq!(
                observed_arbiter.snapshot(&observed_key).unwrap().reserved,
                32 + SCRATCH_BYTES
            );
            observed_flag.store(true, Ordering::Release);
        }))
    });
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            Some(&flag)
        )
        .is_err()
    );
    let final_state = arbiter.snapshot(&key).unwrap();
    assert_eq!(final_state.reserved, 0);
    assert!(!final_state.fenced);
}

#[test]
fn nonfinite_used_words_and_finite_input_overflow_refuse_without_outputs() {
    let source = SourceIdentity::for_new_allocation().unwrap();
    let (arbiter, key, owner) = budget(4096);
    let mut a = [1.0; 32];
    let b = [2.0; 16];
    a[19] = f32::INFINITY;
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        )
        .is_err()
    );
    a.fill(f32::MAX);
    assert!(
        execute(
            request(source, &a, &b),
            BulkF32Admission {
                arbiter: &arbiter,
                resource: &key,
                owner: &owner
            },
            None
        )
        .is_err()
    );
    assert_eq!(arbiter.snapshot(&key).unwrap().reserved, 0);
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/proposed/crates/zerogravity-cpu/src/math/haswell_bulk_f32.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13182; SHA256 `1f17327c62f36e8d813aab902a907a01233d84967a92ed84033e6ebc4a03ac13`.

````text
//! Closed GGML d812350 Haswell tinyBLAS F32 schedule, no model/plan selection.
//!
//! One Vec8 FMA chain per output and the original even/odd horizontal tree.
//! Default GGUF four-Vec8 projections/attention and source profiles are unchanged.
//! Source: ggml/src/ggml-cpu/llamafile/sgemm.cpp:252-270,559-730,3908-3934.

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
    Ok(BulkF32Completed {
        values,
        source: request.source,
        shape,
        _lease: lease,
    })
}

#[cfg(test)]
mod tests;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/proposed/crates/zerogravity-cpu/src/math.rs

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 13304; SHA256 `16b2a904fe208bbdeecc8de2c072c885ff12839041cb42b53675a50bb4d71526`.

````text
//! Deterministic CPU reference operations used to validate compiled regions.
//!
//! These are mathematical building blocks, NOT full Qwen GDN or Gemma MoE layer
//! implementations. Model recipes will register exact higher-level semantics.

use crate::{Error, Result};

fn check(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Malformed(message.into()))
    }
}

/// Reference row-major matrix vector product, using diagnostic f64 accumulation.
pub fn matvec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>> {
    check(
        rows.checked_mul(cols) == Some(matrix.len()),
        "matvec matrix shape",
    )?;
    check(input.len() == cols, "matvec input size")?;
    check(rows <= 16_000_000, "matvec output too large")?;
    let mut out = Vec::new();
    out.try_reserve_exact(rows)
        .map_err(|_| Error::Limit("matvec output allocation failed".into()))?;
    for row in matrix.chunks_exact(cols.max(1)).take(rows) {
        let acc = row
            .iter()
            .zip(input)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum::<f64>();
        out.push(acc as f32);
    }
    if cols == 0 {
        out.resize(rows, 0.0);
    }
    Ok(out)
}

/// RMS norm reference, without residual addition or model-specific postscaling.
pub fn rms_norm(values: &[f32], weight: &[f32], epsilon: f32) -> Result<Vec<f32>> {
    check(
        values.len() == weight.len() && !values.is_empty(),
        "rms norm shape",
    )?;
    check(epsilon.is_finite() && epsilon > 0.0, "rms norm epsilon")?;
    let sum: f64 = values.iter().map(|v| f64::from(*v).powi(2)).sum();
    let inv = (sum / values.len() as f64 + f64::from(epsilon))
        .sqrt()
        .recip();
    Ok(values
        .iter()
        .zip(weight)
        .map(|(x, w)| (*x as f64 * inv * f64::from(*w)) as f32)
        .collect())
}

/// Stable probability normalization. Reject NaN and infinities instead of masking
/// an unknown model's attention or router semantics.
pub fn softmax(values: &[f32]) -> Result<Vec<f32>> {
    check(!values.is_empty(), "softmax empty")?;
    check(
        values.iter().all(|x| x.is_finite()),
        "softmax requires finite logits",
    )?;
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f64> = values.iter().map(|v| f64::from(*v - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    check(sum > 0.0 && sum.is_finite(), "softmax unstable")?;
    Ok(weights.into_iter().map(|v| (v / sum) as f32).collect())
}

/// Router top-k selection is separated from model-specific scoring,
/// renormalization, and shared-expert accumulation rules.
pub fn top_k(logits: &[f32], k: usize) -> Result<Vec<usize>> {
    check(k > 0 && k <= logits.len(), "top-k size")?;
    check(
        logits.iter().all(|x| x.is_finite()),
        "top-k requires finite logits",
    )?;
    let mut rank: Vec<usize> = (0..logits.len()).collect();
    rank.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    rank.truncate(k);
    Ok(rank)
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn gelu_tanh(x: f32) -> f32 {
    let a = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (a * (x + 0.044715 * x * x * x)).tanh())
}

/// One causal attention query over keys/values at positions 0..T.
/// `window=None` is full attention; `window=Some(n)` keeps the last n
/// positions inclusive of query position. This is diagnostic CPU math.
pub fn attention_one(
    query: &[f32],
    keys: &[f32],
    values: &[f32],
    query_position: usize,
    window: Option<usize>,
) -> Result<Vec<f32>> {
    let d = query.len();
    check(
        d > 0 && keys.len() == values.len() && keys.len().is_multiple_of(d),
        "attention head geometry",
    )?;
    let tokens = keys.len() / d;
    check(
        tokens > 0 && query_position < tokens,
        "attention query position",
    )?;
    if let Some(n) = window {
        check(n > 0, "sliding attention window")?;
    }
    let start = window.map_or(0, |n| (query_position + 1).saturating_sub(n));
    let scale = 1.0 / (d as f64).sqrt();
    let mut scores = Vec::new();
    for t in start..=query_position {
        let v = keys[t * d..(t + 1) * d]
            .iter()
            .zip(query)
            .map(|(&k, &q)| f64::from(k) * f64::from(q))
            .sum::<f64>();
        scores.push((v * scale) as f32);
    }
    let probs = softmax(&scores)?;
    let mut out = vec![0.0; d];
    for (i, p) in probs.iter().enumerate() {
        let v = &values[(start + i) * d..(start + i + 1) * d];
        for (dst, &x) in out.iter_mut().zip(v) {
            *dst += x * p;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matvec_and_norm() {
        assert_eq!(
            matvec(&[1., 2., 3., 4.], 2, 2, &[5., 6.]).unwrap(),
            [17., 39.]
        );
        assert!(matvec(&[1., 2.], 2, 2, &[1., 2.]).is_err());
        assert_eq!(matvec(&[], 3, 0, &[]).unwrap(), [0., 0., 0.]);
        let norm = rms_norm(&[3., 4.], &[1., 1.], 1e-6).unwrap();
        assert!((norm[0] - 0.848528).abs() < 1e-5);
        assert!((norm[1] - 1.131_37).abs() < 1e-5);
    }

    #[test]
    fn stable_softmax_and_tie_routing() {
        let probs = softmax(&[1000., 1000., 999.]).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.).abs() < 1e-6);
        assert_eq!(top_k(&[3., 4., 4., 3.], 2).unwrap(), [1, 2]);
        assert!(top_k(&[f32::NAN, 1.], 1).is_err());
        assert!(softmax(&[f32::NEG_INFINITY]).is_err());
        assert!((silu(0.0)).abs() < 1e-9);
        assert!((gelu_tanh(0.0)).abs() < 1e-9);
    }

    #[test]
    fn sliding_vs_global_attention_and_future_mask() {
        let q = [1.];
        let k = [0., 0., 0., 0.];
        let v = [1., 2., 3., 100.];
        let global = attention_one(&q, &k, &v, 2, None).unwrap();
        let sliding = attention_one(&q, &k, &v, 2, Some(2)).unwrap();
        assert!((global[0] - 2.0).abs() < 1e-6);
        assert!((sliding[0] - 2.5).abs() < 1e-6);
        assert!(attention_one(&q, &k, &v, 1, Some(0)).is_err());
    }
}

/// Deterministic unfiltered logits/log-probability audit. These are produced
/// *before* temperature, penalties, truncation or grammar; a sampler's
/// post-filter probabilities are not an equivalent numerical oracle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LogitRank {
    pub id: u32,
    pub delta: f32,
    pub logprob: f64,
}

pub fn top_logit_ranks(logits: &[f32], n: usize) -> Result<Vec<LogitRank>> {
    if logits.is_empty() || n == 0 || n > 128 || logits.len() > u32::MAX as usize {
        return Err(Error::Malformed("invalid logits audit width/rank".into()));
    }
    if !logits.iter().all(|x| x.is_finite()) {
        return Err(Error::Malformed("nonfinite logits in audit".into()));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum = logits
        .iter()
        .map(|x| f64::from(*x - max).exp())
        .sum::<f64>();
    let ln_sum = sum.ln();
    if !ln_sum.is_finite() {
        return Err(Error::Malformed("nonfinite logits normalization".into()));
    }
    let mut indices: Vec<_> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&i, &j| logits[j].total_cmp(&logits[i]).then_with(|| i.cmp(&j)));
    Ok(indices
        .into_iter()
        .take(n)
        .map(|id| {
            let delta = logits[id] - max;
            LogitRank {
                id: id as u32,
                delta,
                logprob: f64::from(delta) - ln_sum,
            }
        })
        .collect())
}

#[cfg(test)]
mod logit_audit_tests {
    use super::*;
    #[test]
    fn test_raw_log_probs_pre_sampler() {
        let ranks = top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 3).unwrap();
        assert_eq!(ranks.iter().map(|r| r.id).collect::<Vec<_>>(), [1, 2, 0]);
        assert_eq!(ranks[0].delta, 0.0);
        assert_eq!(ranks[2].delta, -2.0);
        assert!(
            (ranks.iter().map(|r| r.logprob.exp()).sum::<f64>()
                + top_logit_ranks(&[2.0, 4.0, 4.0, 1.0], 4).unwrap()[3]
                    .logprob
                    .exp()
                - 1.0)
                .abs()
                < 1e-10
        );
        assert!(top_logit_ranks(&[0.0, f32::NAN], 2).is_err());
    }
}

#[cfg(test)]
mod long_context_reference_tests {
    use super::*;

    #[test]
    fn thirty_two_thousand_causal_attention_positions_vs_independent_double_oracle() {
        const CONTEXT: usize = 32_768;
        const WIDTH: usize = 4;
        let query = [0.4, -0.7, 0.2, 0.1];
        let keys = (0..=CONTEXT)
            .flat_map(|t| (0..WIDTH).map(move |i| (((t * 7 + i * 37) % 113) as f32 - 56.0) / 97.0))
            .collect::<Vec<_>>();
        let mut values = (0..=CONTEXT)
            .flat_map(|t| {
                (0..WIDTH).map(move |i| (((t * 13 + i * 17) % 199) as f32 - 99.0) / 201.0)
            })
            .collect::<Vec<_>>();
        // Future token must never contribute; sentinel catches mask bugs.
        values[CONTEXT * WIDTH..].fill(1_000_000.0);
        let actual = attention_one(&query, &keys, &values, CONTEXT - 1, None).unwrap();
        let window = attention_one(&query, &keys, &values, CONTEXT - 1, Some(1024)).unwrap();
        let independently_evaluate = |start: usize| {
            let mut scores = Vec::new();
            for t in start..CONTEXT {
                let score = query
                    .iter()
                    .enumerate()
                    .map(|(i, &q)| f64::from(q) * f64::from(keys[t * WIDTH + i]))
                    .sum::<f64>()
                    / (WIDTH as f64).sqrt();
                scores.push(score);
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights = scores.iter().map(|s| (s - max).exp()).collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            std::array::from_fn::<_, WIDTH, _>(|i| {
                (weights
                    .iter()
                    .enumerate()
                    .map(|(j, w)| w * f64::from(values[(start + j) * WIDTH + i]))
                    .sum::<f64>()
                    / sum) as f32
            })
        };
        let global_oracle = independently_evaluate(0);
        let sliding_oracle = independently_evaluate(CONTEXT - 1024);
        for i in 0..WIDTH {
            assert!(
                (actual[i] - global_oracle[i]).abs() < 2e-6,
                "full attention reference mismatch in channel {i}"
            );
            assert!(
                (window[i] - sliding_oracle[i]).abs() < 2e-6,
                "sliding attention reference mismatch in channel {i}"
            );
        }
    }
}

/// GGML Haswell/AVX2 scalar-equivalent approximate exponential used in the
/// source execution mode's SIMD SiLU implementation. This is *not* the
/// mathematical exp reference: its FMA polynomial and float-bit exponent
/// construction are part of the requested source-numerical specialization.
///
/// The common finite model gate range is handled by the exact polynomial
/// branch. For other finite ranges, follow the original SIMD scaling branches;
/// NaN/Inf values are rejected by the model state boundary before this stage.
pub fn ggml_source_exp(x: f32) -> f32 {
    if !x.is_finite() {
        return x.exp();
    }
    let r = f32::from_bits(0x4b40_0000);
    let z = x.mul_add(f32::from_bits(0x3fb8_aa3b), r);
    let n = z - r;
    let b = (-n).mul_add(
        f32::from_bits(0x35bf_be8e),
        (-n).mul_add(f32::from_bits(0x3f31_7200), x),
    );
    let e = z.to_bits().wrapping_shl(23);
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = f32::from_bits(0x3c07_2010)
        .mul_add(b, f32::from_bits(0x3d2b_9f17))
        .mul_add(
            u,
            f32::from_bits(0x3e2a_af33).mul_add(b, f32::from_bits(0x3eff_fedb)),
        )
        .mul_add(u, f32::from_bits(0x3f7f_fff6) * b);
    if n.abs() <= 126.0 {
        return j.mul_add(k, k);
    }
    let g = if n <= 0.0 { 0x8200_0000u32 } else { 0 };
    let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
    let s2 = f32::from_bits(e.wrapping_sub(g));
    if n.abs() > 192.0 {
        s1 * s1
    } else {
        s2.mul_add(j, s2) * s1
    }
}

/// Opt-in, model-recipe-specific source-compatible SiLU on f32. This matches
/// GGML's vector approximation, rather than silently redefining generic SiLU.
pub fn ggml_source_silu(x: f32) -> f32 {
    x / (1.0 + ggml_source_exp(0.0 - x))
}

#[cfg(test)]
mod ggml_source_silu_tests {
    use super::*;
    #[test]
    fn source_approx_is_bounded_and_retains_general_silu_behavior() {
        for integer in -100..=100 {
            let x = integer as f32 / 10.0;
            let y = ggml_source_silu(x);
            assert!(y.is_finite());
            assert!(
                (y - silu(x)).abs() <= 1e-5,
                "GGML source approximation unexpectedly divergent at {x}"
            );
        }
        assert_eq!(ggml_source_silu(0.0), 0.0);
    }
}

/// Explicit pinned GGML bulk-F32 diagnostic; default source math is unchanged.
pub mod haswell_bulk_f32;

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/run_replay.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 5557; SHA256 `e82614f3ecc80324e947fbeed431efde0f90dd6ba15616f3d5aed0c450b7f2ec`.

````text
"""ROOT ONLY: bounded owned Rust primitive replay; no provider/model invocation."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import time
from contracts import HERE, digest, frozen, load, parents, read, require, write
from prepare import verify_transport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--binary-sha256', required=True)
    p.add_argument('--build-attestation', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--new-run', type=Path, required=True)
    args = p.parse_args()
    pin = frozen()
    build = read(args.build_attestation)
    binary = digest(args.binary, 64 * 1024 * 1024)
    require(args.binary.is_absolute() and binary['sha256'] == args.binary_sha256,
            'actual bounded fresh ROOT binary')
    require(build['status'] == 'ROOT_COMPILED_BULK_F32_PRIMITIVE_ONLY' and build['build_exit_code'] == 0
            and build['freeze_sha256'] == pin and build['binary'] == str(args.binary)
            and build['binary_digest'] == binary and len(build['effective_lock_sha256']) == 64
            and build['main_digest'] == digest(HERE / 'main.rs'), 'actual ROOT build/source binding')
    proposed = HERE / 'proposed'
    actual_files = []
    for path in sorted(proposed.rglob('*.rs')):
        actual = HERE.parents[2] / path.relative_to(proposed)
        expected = digest(path)
        require(digest(actual) == expected, 'ROOT actual merged primitive bytes')
        actual_files.append({'path':str(actual),'digest':expected})
    require(build['production_files'] == actual_files, 'actual ROOT compiled production source pins')
    require(args.fixtures.is_absolute() and args.fixtures.parent.resolve() == HERE,
            'own authentic fixture transport')
    parents()
    verify_transport(args.fixtures)
    fixture = digest(args.fixtures / 'manifest.json')
    cpus = sorted(os.sched_getaffinity(0))[:4]
    require(len(cpus) == 4, 'four available CPUs')
    require(args.new_run.is_absolute() and args.new_run.parent.resolve() == HERE
            and not args.new_run.exists(), 'fresh own run')
    args.new_run.mkdir(mode=0o700)
    output = args.new_run / 'output'
    argv = [str(args.binary),str(args.fixtures),str(output)]
    receipt = {'status':'UNQUALIFIED','freeze_sha256':pin,'argv':argv,'binary':binary,
               'build_attestation':build,'fixture_manifest':fixture,'actual_affinity_expected':cpus,
               'wall_seconds':300,'as_limit_bytes':1024**3,'math_scope':'captured primitive only',
               'provider_model_GPU_execution':False}
    write(args.new_run / 'prepared-invocation.json', receipt)
    h = load(HERE / 'owned_lifetime.py','bulk_f32_owned_lifetime')
    env = {'PATH':'/usr/bin:/bin','LANG':'C','LC_ALL':'C','RAYON_NUM_THREADS':'4',
           'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','CUDA_VISIBLE_DEVICES':'-1',
           'NVIDIA_VISIBLE_DEVICES':'void','LD_PRELOAD':'','LD_AUDIT':''}
    def limits():
        os.sched_setaffinity(0,cpus)
        for kind,value in ((resource.RLIMIT_AS,1024**3),(resource.RLIMIT_CPU,1200),
                           (resource.RLIMIT_NOFILE,64),(resource.RLIMIT_CORE,0),
                           (resource.RLIMIT_FSIZE,8*1024**2),(resource.RLIMIT_STACK,8*1024**2)):
            resource.setrlimit(kind,(value,value))
    child = log = failure = None
    start = time.monotonic()
    try:
        child = subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 env=env,start_new_session=True,preexec_fn=limits)
        receipt['owned_cli_pid'] = child.pid
        log = h.Log(child.stdout,args.new_run / 'execution.log')
        receipt['exit_code'] = log.observe(child,300)
    except BaseException as error:
        failure = error
    finally:
        errors = receipt['cleanup_errors'] = []
        if child is not None:
            try:
                h.cleanup_group(child,receipt)
            except BaseException as error:
                errors.append('owned foreground:'+repr(error))
        if log is not None:
            try:
                log.finish()
            except BaseException as error:
                errors.append('log:'+repr(error))
            finally:
                receipt['log_cleanup'] = {'pipe_eof':log.eof,'all_closed':log.closed,
                    'total_bytes':log.total,'retained_bytes':log.retained}
        elif child is not None and child.stdout is not None:
            try:
                child.stdout.close()
            except BaseException as error:
                errors.append('uninitialized pipe:'+repr(error))
        receipt['elapsed_seconds_not_benchmark'] = time.monotonic()-start
    if failure is not None:
        receipt['failure'] = repr(failure)
    good = failure is None and receipt.get('exit_code') == 0 and not receipt['cleanup_errors']
    if good:
        receipt['status'] = 'EXECUTED_PRIMITIVE_ONLY_REQUIRES_VALIDATION'
    write(args.new_run / 'invocation.json', receipt)
    require(good, 'failed source/control/cleanup attempt retained; no retry/fallback')
    require(digest(args.binary) == binary and digest(args.fixtures / 'manifest.json') == fixture,
            'unchanged binary/transport after known retirement')
    parents()
    verify_transport(args.fixtures)
    require(frozen() == pin, 'unchanged source after replay')

if __name__ == '__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/test_contracts.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 4834; SHA256 `a84e7590f87da81e40c339b87f6e0d71fd27a43c632083d327d884bf58430feb`.

````text
"""ROOT-only synthetic receipt regressions; no operator/data/model proof."""
import copy
import unittest
from validate import META, STAGING, WORDS, MODEL, report, invocation


def ledger(metadata=0,staging=0,output=0):
    return {'reserved':metadata+staging+output+(256 if output else 0),
            'reservations':int(metadata>0)+int(staging>0)+int(output>0),
            'fenced':False,'metadata':metadata,'staging':staging,'output':output,
            'scratch':256 if output else 0}


def synthetic_report():
    return {'status':'PASS_CAPTURED_DEFAULT_BULK_F32_ONLY','profile':'GgmlD812350HaswellTinyBlas',
            'output_words':132096,'default_production_changed':False,'historical_prefix_recovered':False,
            'model_or_state_qualified':False,'source_memory_admission_scope':
            'configured diagnostic host, actual logical payload reservations; no whole physical RAM claim',
            'actual_affinity':[0,1,2,3],'mxcsr_before':0x1f80,'mxcsr_after':0x1fa0,
            'before':ledger(),'final':ledger(),'cases':[
                {'case':key,'model_sha256':MODEL[key],'words':WORDS[key],'unequal_words':0,
                 'pre_cancel_refused':True,'capacity_refused':True,'expected_opened_after_math':True,
                 'physical_left_span_words':4317184 if key=='av' else 32768,
                 'live':ledger(META,STAGING,WORDS[key]*4),
                 'retained_output':ledger(META,0,WORDS[key]*4),'after_payload_drop':ledger(META)}
                for key in ('av','alpha')]}


class Contracts(unittest.TestCase):
    def reject(self,change):
        value=synthetic_report();change(value)
        with self.assertRaises((ValueError,KeyError,TypeError)):
            report(value)

    def test_valid_shape_is_explicitly_synthetic(self):
        report(synthetic_report())

    def test_all_cases_and_complete_output_counts_required(self):
        self.reject(lambda v:v['cases'].pop())
        self.reject(lambda v:v['cases'].append(copy.deepcopy(v['cases'][0])))
        self.reject(lambda v:v['cases'][0].__setitem__('words',131071))
        self.reject(lambda v:v['cases'][1].__setitem__('words',True))

    def test_source_and_stride_cannot_be_retagged(self):
        self.reject(lambda v:v['cases'][0].__setitem__('model_sha256',MODEL['alpha']))
        self.reject(lambda v:v['cases'][0].__setitem__('physical_left_span_words',131072))

    def test_signed_zero_or_any_raw_mismatch_fails(self):
        self.reject(lambda v:v['cases'][1].__setitem__('unequal_words',1))
        self.reject(lambda v:v['cases'][1].__setitem__('unequal_words',False))

    def test_aggregate_capacity_and_retained_ticket_required(self):
        self.reject(lambda v:v['cases'][0].__setitem__('live',ledger(META,STAGING)))
        self.reject(lambda v:v['cases'][1].__setitem__('retained_output',ledger(META)))
        self.reject(lambda v:v['cases'][1]['live'].__setitem__('scratch',0))
        self.reject(lambda v:v['cases'][1]['live'].__setitem__('reservations',True))
        self.reject(lambda v:v.__setitem__('final',ledger(META)))

    def test_real_negative_conditions_and_expected_isolation_required(self):
        for field in ('pre_cancel_refused','capacity_refused','expected_opened_after_math'):
            with self.subTest(field=field):
                self.reject(lambda v:v['cases'][0].__setitem__(field,False))

    def test_fp_policy_and_no_default_or_history_promotion(self):
        self.reject(lambda v:v.__setitem__('mxcsr_after',0x9f80))
        self.reject(lambda v:v.__setitem__('mxcsr_after',0x3f80))
        self.reject(lambda v:v.__setitem__('historical_prefix_recovered',True))
        self.reject(lambda v:v.__setitem__('default_production_changed',True))
        self.reject(lambda v:v.__setitem__('model_or_state_qualified',True))

    def test_unknown_lifecycle_refuses(self):
        v={'status':'EXECUTED_PRIMITIVE_ONLY_REQUIRES_VALIDATION','exit_code':0,
           'cleanup_errors':[],'provider_model_GPU_execution':False,'math_scope':'captured primitive only',
           'wall_seconds':300,'as_limit_bytes':1024**3,'owned_cli_pid':12,
           'foreground_cleanup':{'pgid':12,'leader_reaped':True,'group_absent':True,'stage_errors':[],
             'all_nonzero_signals_before_reap':True,'kill_attempts_before_reap':2,'leader_reap_attempts':1},
           'log_cleanup':{'pipe_eof':True,'all_closed':True,'total_bytes':0,'retained_bytes':0}}
        invocation(v)
        for field in ('leader_reaped','group_absent','all_nonzero_signals_before_reap'):
            bad=copy.deepcopy(v);bad['foreground_cleanup'][field]=False
            with self.assertRaises(ValueError):invocation(bad)
        bad=copy.deepcopy(v);bad['log_cleanup']['pipe_eof']=False
        with self.assertRaises(ValueError):invocation(bad)


if __name__=='__main__':
    unittest.main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-primitive-20261020/validate.py

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 6074; SHA256 `abab362a92a7347f09d7f9ab8cf50da2ca4e6e9f120f0f56b7f88f7af2dcf5c3`.

````text
"""ROOT ONLY independent exact-byte/default-source/ledger/lifetime validator."""
import argparse
from pathlib import Path
from contracts import HERE, CASES, digest, frozen, parents, read, require, write
from prepare import verify_transport

META = 4 * 1024**2
STAGING = 32 * 1024**2
WORDS = {'av':131072,'alpha':1024}
MODEL = {'av':'57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf',
         'alpha':'91c102fc9a86de80e427057ee938e1e34fcaf3bba956b7296e252406e05f36f6'}

def ledger(value, metadata=0, staging=0, output=0):
    scratch = 256 if output else 0
    require(all(type(value[key]) is int for key in
                ('reserved','reservations','output','scratch','staging','metadata'))
            and value['fenced'] is False, 'strict actual ledger value types')
    require(value == {'reserved':metadata+staging+output+scratch,
                      'reservations':int(metadata>0)+int(staging>0)+int(output>0),'fenced':False,
                      'output':output,'scratch':scratch,'staging':staging,'metadata':metadata},
            'exact original class/retained-output ledger')

def report(value):
    require(value['status'] == 'PASS_CAPTURED_DEFAULT_BULK_F32_ONLY'
            and value['profile'] == 'GgmlD812350HaswellTinyBlas'
            and value['output_words'] == 132096 and type(value['output_words']) is int,
            'complete explicit primitive scope')
    for field in ('default_production_changed','historical_prefix_recovered','model_or_state_qualified'):
        require(value[field] is False, 'no capability/default/history promotion')
    require(value['source_memory_admission_scope'] ==
            'configured diagnostic host, actual logical payload reservations; no whole physical RAM claim',
            'honest original admission scope')
    cpus = value['actual_affinity']
    require(len(cpus)==4 and len(set(cpus))==4 and all(type(n)is int and 0<=n<1024 for n in cpus),
            'actual four-CPU affinity')
    for field in ('mxcsr_before','mxcsr_after'):
        require(type(value[field]) is int and value[field]&0xffc0 == 0x1f80,
                'actual RNE/masked/noFTZ/noDAZ')
    ledger(value['before']); ledger(value['final'])
    require([x['case']for x in value['cases']] == list(CASES), 'complete distinct authentic case population')
    for row in value['cases']:
        key = row['case']; size = WORDS[key]*4
        require(row['model_sha256']==MODEL[key] and type(row['words'])is int and row['words']==WORDS[key]
                and type(row['unequal_words'])is int and row['unequal_words']==0,
                'actual complete raw trained default result')
        require(row['pre_cancel_refused'] is True and row['capacity_refused'] is True
                and row['expected_opened_after_math'] is True, 'real private-call refusal and expected isolation')
        require(row['physical_left_span_words'] == (4317184 if key=='av' else 32768),
                'original strided full view, no dense substitute')
        ledger(row['live'],META,STAGING,size)
        ledger(row['retained_output'],META,0,size)
        ledger(row['after_payload_drop'],META)

def invocation(value):
    require(value['status']=='EXECUTED_PRIMITIVE_ONLY_REQUIRES_VALIDATION'
            and value['exit_code']==0 and value['cleanup_errors']==[]
            and value['provider_model_GPU_execution'] is False
            and value['math_scope']=='captured primitive only'
            and value['wall_seconds']==300 and value['as_limit_bytes']==1024**3,
            'actual primitive-only invocation/finite scope')
    g=value['foreground_cleanup']; log=value['log_cleanup']
    require(g['pgid']==value['owned_cli_pid'] and g['leader_reaped'] is True
            and g['group_absent'] is True and g['stage_errors']==[]
            and g['all_nonzero_signals_before_reap'] is True
            and g['kill_attempts_before_reap']==2 and g['leader_reap_attempts']in(1,2),
            'actual owned unreaped-group retirement')
    require(log['pipe_eof'] is True and log['all_closed'] is True
            and log['total_bytes']==log['retained_bytes']<=4*1024**2,
            'complete bounded owned pipe EOF/close')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True)
    args=p.parse_args()
    pin=frozen()
    require(args.run.is_absolute()and args.run.parent.resolve()==HERE
            and args.fixtures.is_absolute()and args.fixtures.parent.resolve()==HERE,'owned ROOT artifacts')
    parents()
    manifest=verify_transport(args.fixtures)
    called=read(args.run/'invocation.json'); invocation(called)
    require(called['freeze_sha256']==pin and called['fixture_manifest']==digest(args.fixtures/'manifest.json')
            and called['binary']==digest(called['argv'][0]) and called['argv'][1:]==
            [str(args.fixtures),str(args.run/'output')], 'actual source/input/binary/output binding')
    value=read(args.run/'output/report.json'); report(value)
    require(value['actual_affinity']==called['actual_affinity_expected'],'same observed CPU set')
    for row in manifest['cases']:
        key=row['case']; actual=args.run/'output'/f'{key}.actual.f32le'
        expected=args.fixtures/row['expected']['file']
        require(digest(actual,1024**2)==row['expected']['digest']==digest(expected,1024**2),
                'complete exact default-capture bytes including all zero signs')
        require(actual.read_bytes()==expected.read_bytes(), 'independent byte equality, no tolerances')
    require(frozen()==pin,'same exact reviewed source')
    write(args.run/'validation.json',{'status':'PASS_CAPTURED_DEFAULT_BULK_F32_ONLY','freeze_sha256':pin,
          'words':132096,'cases':list(CASES),'invocation':digest(args.run/'invocation.json'),
          'actual_report':digest(args.run/'output/report.json'),'model_state_or_history_qualified':False,
          'default_or_GPU_profile_changed':False,'whole_physical_RAM_admission_qualified':False})

if __name__=='__main__':
    main()

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/BASELINE.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1275; SHA256 `3cafb3ac7c423461a5d0b14ccc85d7a87fc4e0a9e30d0dc3abef4e152c3060e4`.

````text
{
  "scope": "Original current source baseline only; no MAIN writes. Primitive689221fe plus attributiona7e8cc3b separate dependency, unchanged.",
  "files": [
    {
      "path": "crates/zerogravity-core/src/accelerator/program.rs",
      "bytes": 76201,
      "sha256": "2ee7db7698fe0c65330cdf2e074c8335fd2cef36033c291b9aa553764695452f"
    },
    {
      "path": "crates/zerogravity-model-qwen35/src/host_math.rs",
      "bytes": 39194,
      "sha256": "8937d8d8efb7269ef15e5a9bb8338dacc60ae8483c29050c8f2f018818d36355"
    },
    {
      "path": "crates/zerogravity-model-qwen35/src/operator_slice.rs",
      "bytes": 8579,
      "sha256": "3f0d399bf936c6db3ee02315558fd626e09e956ad5d97d28b01a827fdb0d4686"
    },
    {
      "path": "crates/zerogravity-model-qwen35/src/forward.rs",
      "bytes": 23682,
      "sha256": "a96a993f73df31ff56ec20b3704480f48c4dc03c8e62d016fa83956d88aa5820"
    },
    {
      "path": "crates/zerogravity-gpu/src/resident_program.rs",
      "bytes": 106452,
      "sha256": "32ae5012e22a62570127b93f336a00c4be740e9bdc29c751dc8c5e9c8fa501a6"
    },
    {
      "path": "crates/zerogravity-gpu/src/resident_program/batch.rs",
      "bytes": 73755,
      "sha256": "e119a80bb17b2a95750e4d684baa72f3d96d2e3c7993ebc3bd1a7ebf80264deb"
    }
  ]
}

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/Cargo.toml

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 506; SHA256 `1764ad9286e46dcd24fb2c92d303a1e093672fb06a916f2c2ab0e02517e52e80`.

````text
[package]
name = "zg-same-engine-bulk-f32-replay"
version = "0.0.0"
edition = "2024"
publish = false

[workspace]

[[bin]]
name = "bulk-f32-typed-replay"
path = "main.rs"

[dependencies]
serde = { version = "=1.0.229", features = ["derive"] }
serde_json = "=1.0.151"
zerogravity-core = { path = "../../../../crates/zerogravity-core" }
zerogravity-model-qwen35 = { path = "../../../../crates/zerogravity-model-qwen35", features = ["cpu"] }
libc = "=0.2.190"

[profile.release]
lto = false
codegen-units = 1

````

## Source: .agents/runs/numerics-and-context-bulk-f32-typed-slice-20261020/STATIC.json

Dated paused/source-only or historical text; not authorization or qualification.
Original bytes 1143; SHA256 `c9941b4c8199c96614d21140f4c2c7e41678a9c23f2b2d376ac69c07b63585f3`.

````text
{
  "status": "SOURCE_ONLY_RUSTFMT_AST_NO_TESTS_OR_RUNTIME",
  "rustfmt_edition": 2024,
  "rustfmt_skip_children": true,
  "python_ast_files": [
    "contracts.py",
    "owned_lifetime.py",
    "prepare.py",
    "run_replay.py",
    "sigterm_guard.py",
    "test_contracts.py",
    "validate.py"
  ],
  "author_compilation_tests_imports_operators_models_providers_GPU": false,
  "changed_paths": [
    "crates/zerogravity-core/src/accelerator/program/bulk_f32/tests.rs",
    "crates/zerogravity-core/src/accelerator/program/bulk_f32.rs",
    "crates/zerogravity-core/src/accelerator/program.rs",
    "crates/zerogravity-gpu/src/resident_program/batch.rs",
    "crates/zerogravity-gpu/src/resident_program.rs",
    "crates/zerogravity-model-qwen35/src/bulk_f32_slice.rs",
    "crates/zerogravity-model-qwen35/src/forward/bulk_f32.rs",
    "crates/zerogravity-model-qwen35/src/forward.rs",
    "crates/zerogravity-model-qwen35/src/host_math/bulk_f32.rs",
    "crates/zerogravity-model-qwen35/src/host_math.rs",
    "crates/zerogravity-model-qwen35/src/operator_slice.rs"
  ],
  "core_tests_prepared": 5,
  "python_synthetic_tests_prepared": 9
}

````

