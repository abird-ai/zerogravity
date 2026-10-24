# Draft Patches — volume 11

Dated preservation only, 20 October 2026. Original paths below identify the source.
Embedded instructions/status are historical data, not current authorization.
Live source and owning TODOs outrank these copies.

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/NEW_DELTA.patch

Original UTF-8 bytes: 23886; SHA256 `f6f2179ba4ee1f2c78c937c72d18231406fe5ecea1b6d7092afdfaff38ad6307`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs
@@ -0,0 +1,102 @@
+//! Actual same-source host backing, not a producer/session/provider authority.
+use super::*;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
+
+/// External authenticated operator bytes. Caller proves archive/producer origin;
+/// importing them never claims a current state, model projection or decay math.
+#[derive(Clone, Copy)]
+pub struct InitialPreludeInput<'a> {
+    contract: InitialPreludeContract,
+    raw: &'a [u8],
+}
+impl<'a> InitialPreludeInput<'a> {
+    pub fn from_f32le(contract: InitialPreludeContract, raw: &'a [u8]) -> Result<Self> {
+        if raw.len() != contract.elements() * 4
+            || raw
+                .chunks_exact(4)
+                .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
+        {
+            return Err(Error::Malformed(
+                "initial GDN external finite raw FP32".into(),
+            ));
+        }
+        Ok(Self { contract, raw })
+    }
+    pub fn contract(self) -> InitialPreludeContract {
+        self.contract
+    }
+}
+struct Backing {
+    contract: InitialPreludeContract,
+    input: Vec<f32>,
+    output: Vec<f32>,
+    // Actual buffers retire before their admission/source owner.
+    workspace: WindowWorkspace,
+}
+/// Clones retain the SAME source/backing/lease; no uncharged Vec escape.
+#[derive(Clone)]
+pub struct InitialPreludeOutput(Arc<Backing>);
+impl InitialPreludeOutput {
+    pub fn contract(&self) -> InitialPreludeContract {
+        self.0.contract
+    }
+    pub fn imported_input(&self) -> &[f32] {
+        &self.0.input
+    }
+    pub fn output(&self) -> &[f32] {
+        &self.0.output
+    }
+    pub fn charged_bytes(&self) -> usize {
+        self.0.workspace.bytes()
+    }
+    pub fn same_issuer(&self, source: &VerifiedTensorSource) -> bool {
+        self.0.workspace.source().same_issuer(source)
+    }
+    pub fn required_bytes(contract: InitialPreludeContract) -> usize {
+        contract.elements() * 8 + 256
+    }
+}
+
+/// One pure synchronous leaf. All payload allocation follows aggregate admission.
+/// No GPU/BLAS/provider path and no fallback; cancelled output is discarded.
+pub fn execute_initial_prelude(
+    source: &VerifiedTensorSource,
+    input: InitialPreludeInput<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    let contract = input.contract;
+    if contract.source_identity() != source.source_identity() {
+        return Err(Error::State("initial GDN foreign raw source issuer".into()));
+    }
+    let bytes = contract.elements() * 4;
+    let workspace = source.reserve_operator_buffers(bytes, bytes + 256)?;
+    not_cancelled(cancelled)?;
+    let mut imported = Vec::with_capacity(contract.elements());
+    imported.extend(
+        input
+            .raw
+            .chunks_exact(4)
+            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
+    );
+    let mut output = vec![0.0f32; contract.elements()];
+    if imported.capacity() != contract.elements() || output.capacity() != contract.elements() {
+        return Err(Error::Limit("initial GDN exact payload capacities".into()));
+    }
+    initial_prelude_into(contract, &imported, &mut output, cancelled)?;
+    let result = InitialPreludeOutput(Arc::new(Backing {
+        contract,
+        input: imported,
+        output,
+        workspace,
+    }));
+    completed(result, cancelled)
+}
+pub(super) fn completed(
+    result: InitialPreludeOutput,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    Ok(result)
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs
@@ -0,0 +1,257 @@
+use super::*;
+fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
+    InitialPreludeContract::new(
+        SourceIdentity::for_new_allocation().unwrap(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        GdnPreludePhase::InitialChunk64,
+        operation,
+        heads,
+    )
+    .unwrap()
+}
+#[test]
+fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    input[0] = 1.0;
+    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
+    input[2] = -1.0;
+    input[64] = -0.0;
+    input[128] = -1.0;
+    let mut output = vec![9.0; input.len()];
+    initial_prelude_into(c, &input, &mut output, None).unwrap();
+    assert_eq!(output[0].to_bits(), 0x3f80_0000);
+    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
+    assert_eq!(output[2].to_bits(), 0x3380_0000);
+    assert_eq!(output[63].to_bits(), 0x3380_0000);
+    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
+    assert_eq!(output[127].to_bits(), 0);
+    assert_eq!(output[128].to_bits(), 0xbf80_0000);
+    assert_eq!(output[191].to_bits(), 0xbf80_0000);
+}
+#[test]
+fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
+    let c = contract(InitialPreludeOperation::QueryScale128, 48);
+    let mut input = vec![1.0; c.elements()];
+    input[..6].copy_from_slice(&[
+        1.0,
+        -1.0,
+        0.0,
+        -0.0,
+        f32::from_bits(1),
+        f32::from_bits(0x8000_0001),
+    ]);
+    let mut out = vec![0.0; input.len()];
+    initial_prelude_into(c, &input, &mut out, None).unwrap();
+    assert_eq!(
+        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
+        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
+    );
+    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
+}
+#[test]
+fn cached_profile_source_geometry_and_complete_span_refuse() {
+    for (source, profile, phase, heads) in [
+        (
+            SourceIdentity::unbound(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::GgufSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::Cached,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            32,
+        ),
+    ] {
+        assert!(
+            InitialPreludeContract::new(
+                source,
+                profile,
+                phase,
+                InitialPreludeOperation::DecayPrefix64,
+                heads
+            )
+            .is_err()
+        );
+    }
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    let mut out = vec![7.0; c.elements()];
+    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
+    input[0] = f32::NAN;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    input[0] = f32::INFINITY;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = 0.0;
+    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = f32::MAX;
+    input[1] = f32::MAX;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod physical {
+    use super::*;
+    use std::fs::{self, File};
+    use std::path::{Path, PathBuf};
+    use std::sync::atomic::AtomicU64;
+    use zerogravity_artifacts::artifact::Artifact;
+    use zerogravity_artifacts::cache::TransferAdmission;
+    use zerogravity_artifacts::source_weights::{
+        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
+    };
+    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
+    const META: u64 = 16 * 1024 * 1024;
+    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
+    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
+    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
+    struct Fixture(PathBuf);
+    impl Fixture {
+        fn new() -> Self {
+            static NEXT: AtomicU64 = AtomicU64::new(1);
+            let p = std::env::temp_dir().join(format!(
+                "zg-initial-prefix-{}-{}",
+                std::process::id(),
+                NEXT.fetch_add(1, Ordering::Relaxed)
+            ));
+            fs::create_dir(&p).unwrap();
+            fs::write(p.join("model.safetensors"), BYTES).unwrap();
+            Self(p)
+        }
+        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
+            SourceWeights::prepare_verified_windows(
+                INDEX,
+                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
+                vec![
+                    ExpectedSourceShard::new(
+                        "model.safetensors".into(),
+                        Artifact::from_file(
+                            File::open(self.0.join("model.safetensors")).unwrap(),
+                            Path::new("model.safetensors"),
+                        )
+                        .unwrap(),
+                        BYTES.len() as u64,
+                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
+                    )
+                    .unwrap(),
+                ],
+                WindowOpenLimits {
+                    max_source_bytes: 1024 * 1024,
+                    max_metadata_bytes: META,
+                    max_shards: 1,
+                    max_tensors: 6,
+                    scan_chunk_bytes: 1024,
+                },
+                h,
+            )
+            .unwrap()
+            .open()
+            .unwrap()
+        }
+    }
+    impl Drop for Fixture {
+        fn drop(&mut self) {
+            let _ = fs::remove_dir_all(&self.0);
+        }
+    }
+    fn host() -> TransferAdmission {
+        let arbiter = ResourceArbiter::new();
+        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
+        arbiter
+            .register(resource.clone(), 64 * 1024 * 1024)
+            .unwrap();
+        TransferAdmission {
+            arbiter,
+            resource,
+            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
+        }
+    }
+    fn reserved(h: &TransferAdmission) -> u64 {
+        h.arbiter.snapshot(&h.resource).unwrap().reserved
+    }
+    fn raw(c: InitialPreludeContract) -> Vec<u8> {
+        1.0f32.to_le_bytes().repeat(c.elements())
+    }
+    #[test]
+    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let foreign = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::QueryScale128,
+            48,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let before = reserved(&h);
+        assert!(execute_initial_prelude(&foreign, input, None).is_err());
+        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), before);
+        let need = InitialPreludeOutput::required_bytes(c) as u64;
+        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
+        let pressure = source.reserve_workspace(hold as usize).unwrap();
+        let pressured = reserved(&h);
+        assert!(execute_initial_prelude(&source, input, None).is_err());
+        assert_eq!(reserved(&h), pressured);
+        drop(pressure);
+        assert_eq!(reserved(&h), before);
+        let output = execute_initial_prelude(&source, input, None).unwrap();
+        let clone = output.clone();
+        assert!(output.same_issuer(&source));
+        assert!(!output.same_issuer(&foreign));
+        assert_eq!(reserved(&h), before + need);
+        assert_eq!(output.charged_bytes() as u64, need);
+        drop(foreign);
+        drop(source);
+        assert_eq!(reserved(&h), META + need);
+        assert_eq!(clone.imported_input().len(), 48 * 128);
+        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
+        drop(output);
+        assert_eq!(reserved(&h), META + need);
+        drop(clone);
+        assert_eq!(reserved(&h), 0);
+    }
+    #[test]
+    fn completed_result_cancel_drops_real_admitted_backing() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::DecayPrefix64,
+            16,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let value = execute_initial_prelude(&source, input, None).unwrap();
+        assert!(reserved(&h) > META);
+        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), META);
+        drop(source);
+        assert_eq!(reserved(&h), 0);
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE
@@ -0,0 +1,84 @@
+From PyTorch:
+
+Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
+Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
+Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
+Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
+Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
+Copyright (c) 2011-2013 NYU                      (Clement Farabet)
+Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
+Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
+Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
+
+From Caffe2:
+
+Copyright (c) 2016-present, Facebook Inc. All rights reserved.
+
+All contributions by Facebook:
+Copyright (c) 2016 Facebook Inc.
+
+All contributions by Google:
+Copyright (c) 2015 Google Inc.
+All rights reserved.
+
+All contributions by Yangqing Jia:
+Copyright (c) 2015 Yangqing Jia
+All rights reserved.
+
+All contributions by Kakao Brain:
+Copyright 2019-2020 Kakao Brain
+
+All contributions by Cruise LLC:
+Copyright (c) 2022 Cruise LLC.
+All rights reserved.
+
+All contributions by Tri Dao:
+Copyright (c) 2024 Tri Dao.
+All rights reserved.
+
+All contributions by Arm:
+Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
+
+All contributions from Caffe:
+Copyright(c) 2013, 2014, 2015, the respective contributors
+All rights reserved.
+
+All other contributions:
+Copyright(c) 2015, 2016 the respective contributors
+All rights reserved.
+
+Caffe2 uses a copyright model similar to Caffe: each contributor holds
+copyright over their contributions to Caffe2. The project versioning records
+all such contribution and copyright details. If a contributor wants to further
+mark their specific copyright on a particular contribution, they should
+indicate their copyright solely in the commit message of the change when it is
+committed.
+
+All rights reserved.
+
+Redistribution and use in source and binary forms, with or without
+modification, are permitted provided that the following conditions are met:
+
+1. Redistributions of source code must retain the above copyright
+   notice, this list of conditions and the following disclaimer.
+
+2. Redistributions in binary form must reproduce the above copyright
+   notice, this list of conditions and the following disclaimer in the
+   documentation and/or other materials provided with the distribution.
+
+3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
+   and IDIAP Research Institute nor the names of its contributors may be
+   used to endorse or promote products derived from this software without
+   specific prior written permission.
+
+THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
+AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
+IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
+ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
+LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
+CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
+SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
+INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
+CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
+ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
+POSSIBILITY OF SUCH DAMAGE.
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs
@@ -0,0 +1,155 @@
+//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
+//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
+//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
+use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
+use crate::{Error, Result};
+use std::sync::atomic::{AtomicBool, Ordering};
+
+/// Phase is explicit; cached query division is deliberately unsupported here.
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum GdnPreludePhase {
+    InitialChunk64,
+    Cached,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum InitialPreludeOperation {
+    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
+    QueryScale128,
+    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
+    DecayPrefix64,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub struct InitialPreludeContract {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    operation: InitialPreludeOperation,
+    heads: usize,
+}
+impl InitialPreludeContract {
+    pub fn new(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: GdnPreludePhase,
+        operation: InitialPreludeOperation,
+        heads: usize,
+    ) -> Result<Self> {
+        if phase != GdnPreludePhase::InitialChunk64
+            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
+        }
+        if !source.is_bound() || !matches!(heads, 16 | 48) {
+            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
+        }
+        Ok(Self {
+            source,
+            profile,
+            operation,
+            heads,
+        })
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn phase(self) -> GdnPreludePhase {
+        GdnPreludePhase::InitialChunk64
+    }
+    pub fn operation(self) -> InitialPreludeOperation {
+        self.operation
+    }
+    pub fn heads(self) -> usize {
+        self.heads
+    }
+    pub fn elements(self) -> usize {
+        self.heads
+            * match self.operation {
+                InitialPreludeOperation::QueryScale128 => 128,
+                InitialPreludeOperation::DecayPrefix64 => 64,
+            }
+    }
+}
+fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("initial GDN leaf cancelled".into()));
+    }
+    Ok(())
+}
+fn environment() -> Result<()> {
+    #[cfg(target_arch = "x86_64")]
+    {
+        let mut mxcsr = 0u32;
+        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
+        // Passive read only; exception status bits may change during arithmetic.
+        unsafe {
+            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
+                options(nostack, preserves_flags));
+        }
+        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
+            return Err(Error::Unsupported(
+                "initial GDN pinned CPU FP controls".into(),
+            ));
+        }
+        Ok(())
+    }
+    #[cfg(not(target_arch = "x86_64"))]
+    Err(Error::Unsupported(
+        "initial GDN pinned CPU architecture".into(),
+    ))
+}
+
+/// No allocation/provider callback. Caller owns both complete, disjoint slices.
+/// Output remains private on error; successful completion grants no state effect.
+pub fn initial_prelude_into(
+    contract: InitialPreludeContract,
+    input: &[f32],
+    output: &mut [f32],
+    cancelled: Option<&AtomicBool>,
+) -> Result<()> {
+    not_cancelled(cancelled)?;
+    environment()?;
+    if input.len() != contract.elements()
+        || output.len() != input.len()
+        || input.iter().any(|v| !v.is_finite())
+    {
+        return Err(Error::Malformed(
+            "initial GDN complete finite FP32 span".into(),
+        ));
+    }
+    match contract.operation {
+        InitialPreludeOperation::QueryScale128 => {
+            // Source query * (128 ** -0.5), scalar converted once to FP32.
+            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
+            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
+            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
+                not_cancelled(cancelled)?;
+                for (&value, result) in row.iter().zip(target) {
+                    *result = value * SCALE;
+                }
+            }
+        }
+        InitialPreludeOperation::DecayPrefix64 => {
+            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
+                not_cancelled(cancelled)?;
+                let mut carry = 0.0f64;
+                for (&value, result) in row.iter().zip(target) {
+                    carry += f64::from(value);
+                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
+                    if !result.is_finite() {
+                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
+                    }
+                }
+            }
+        }
+    }
+    not_cancelled(cancelled)
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod admitted;
+#[cfg(all(feature = "cache", target_os = "linux"))]
+pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
+#[cfg(test)]
+mod tests;
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -4,6 +4,7 @@
 //! This does not enable complete trained Qwen3.8 inference.
 
 pub mod coefficient_preparation;
+pub mod initial_prelude;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/ROOT_WIRING.patch

Original UTF-8 bytes: 286; SHA256 `971e77af71031eefcb479f2524f63f417257ea237d83bfa859ecfa661175fa64`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -4,6 +4,7 @@
 //! This does not enable complete trained Qwen3.8 inference.
 
 pub mod coefficient_preparation;
+pub mod initial_prelude;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/NEW_DELTA.patch

Original UTF-8 bytes: 23886; SHA256 `f6f2179ba4ee1f2c78c937c72d18231406fe5ecea1b6d7092afdfaff38ad6307`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/admitted.rs
@@ -0,0 +1,102 @@
+//! Actual same-source host backing, not a producer/session/provider authority.
+use super::*;
+use std::sync::Arc;
+use zerogravity_artifacts::source_weights::{VerifiedTensorSource, WindowWorkspace};
+
+/// External authenticated operator bytes. Caller proves archive/producer origin;
+/// importing them never claims a current state, model projection or decay math.
+#[derive(Clone, Copy)]
+pub struct InitialPreludeInput<'a> {
+    contract: InitialPreludeContract,
+    raw: &'a [u8],
+}
+impl<'a> InitialPreludeInput<'a> {
+    pub fn from_f32le(contract: InitialPreludeContract, raw: &'a [u8]) -> Result<Self> {
+        if raw.len() != contract.elements() * 4
+            || raw
+                .chunks_exact(4)
+                .any(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) & 0x7f80_0000 == 0x7f80_0000)
+        {
+            return Err(Error::Malformed(
+                "initial GDN external finite raw FP32".into(),
+            ));
+        }
+        Ok(Self { contract, raw })
+    }
+    pub fn contract(self) -> InitialPreludeContract {
+        self.contract
+    }
+}
+struct Backing {
+    contract: InitialPreludeContract,
+    input: Vec<f32>,
+    output: Vec<f32>,
+    // Actual buffers retire before their admission/source owner.
+    workspace: WindowWorkspace,
+}
+/// Clones retain the SAME source/backing/lease; no uncharged Vec escape.
+#[derive(Clone)]
+pub struct InitialPreludeOutput(Arc<Backing>);
+impl InitialPreludeOutput {
+    pub fn contract(&self) -> InitialPreludeContract {
+        self.0.contract
+    }
+    pub fn imported_input(&self) -> &[f32] {
+        &self.0.input
+    }
+    pub fn output(&self) -> &[f32] {
+        &self.0.output
+    }
+    pub fn charged_bytes(&self) -> usize {
+        self.0.workspace.bytes()
+    }
+    pub fn same_issuer(&self, source: &VerifiedTensorSource) -> bool {
+        self.0.workspace.source().same_issuer(source)
+    }
+    pub fn required_bytes(contract: InitialPreludeContract) -> usize {
+        contract.elements() * 8 + 256
+    }
+}
+
+/// One pure synchronous leaf. All payload allocation follows aggregate admission.
+/// No GPU/BLAS/provider path and no fallback; cancelled output is discarded.
+pub fn execute_initial_prelude(
+    source: &VerifiedTensorSource,
+    input: InitialPreludeInput<'_>,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    let contract = input.contract;
+    if contract.source_identity() != source.source_identity() {
+        return Err(Error::State("initial GDN foreign raw source issuer".into()));
+    }
+    let bytes = contract.elements() * 4;
+    let workspace = source.reserve_operator_buffers(bytes, bytes + 256)?;
+    not_cancelled(cancelled)?;
+    let mut imported = Vec::with_capacity(contract.elements());
+    imported.extend(
+        input
+            .raw
+            .chunks_exact(4)
+            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
+    );
+    let mut output = vec![0.0f32; contract.elements()];
+    if imported.capacity() != contract.elements() || output.capacity() != contract.elements() {
+        return Err(Error::Limit("initial GDN exact payload capacities".into()));
+    }
+    initial_prelude_into(contract, &imported, &mut output, cancelled)?;
+    let result = InitialPreludeOutput(Arc::new(Backing {
+        contract,
+        input: imported,
+        output,
+        workspace,
+    }));
+    completed(result, cancelled)
+}
+pub(super) fn completed(
+    result: InitialPreludeOutput,
+    cancelled: Option<&AtomicBool>,
+) -> Result<InitialPreludeOutput> {
+    not_cancelled(cancelled)?;
+    Ok(result)
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude/tests.rs
@@ -0,0 +1,257 @@
+use super::*;
+fn contract(operation: InitialPreludeOperation, heads: usize) -> InitialPreludeContract {
+    InitialPreludeContract::new(
+        SourceIdentity::for_new_allocation().unwrap(),
+        ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+        GdnPreludePhase::InitialChunk64,
+        operation,
+        heads,
+    )
+    .unwrap()
+}
+#[test]
+fn prefix_retains_f64_carry_and_stores_each_f32_prefix() {
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    input[0] = 1.0;
+    input[1] = f32::from_bits(0x3380_0000); // 2^-24, exact half ULP at1.
+    input[2] = -1.0;
+    input[64] = -0.0;
+    input[128] = -1.0;
+    let mut output = vec![9.0; input.len()];
+    initial_prelude_into(c, &input, &mut output, None).unwrap();
+    assert_eq!(output[0].to_bits(), 0x3f80_0000);
+    assert_eq!(output[1].to_bits(), 0x3f80_0000); // Store rounded, carry not rounded.
+    assert_eq!(output[2].to_bits(), 0x3380_0000);
+    assert_eq!(output[63].to_bits(), 0x3380_0000);
+    assert_eq!(output[64].to_bits(), 0); // +0 accumulator + -0 is +0.
+    assert_eq!(output[127].to_bits(), 0);
+    assert_eq!(output[128].to_bits(), 0xbf80_0000);
+    assert_eq!(output[191].to_bits(), 0xbf80_0000);
+}
+#[test]
+fn query_multiply_preserves_signed_zero_and_subnormal_rounding() {
+    let c = contract(InitialPreludeOperation::QueryScale128, 48);
+    let mut input = vec![1.0; c.elements()];
+    input[..6].copy_from_slice(&[
+        1.0,
+        -1.0,
+        0.0,
+        -0.0,
+        f32::from_bits(1),
+        f32::from_bits(0x8000_0001),
+    ]);
+    let mut out = vec![0.0; input.len()];
+    initial_prelude_into(c, &input, &mut out, None).unwrap();
+    assert_eq!(
+        out[..6].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
+        [0x3db5_04f3, 0xbdb5_04f3, 0, 0x8000_0000, 0, 0x8000_0000]
+    );
+    assert!(out[6..].iter().all(|v| v.to_bits() == 0x3db5_04f3));
+}
+#[test]
+fn cached_profile_source_geometry_and_complete_span_refuse() {
+    for (source, profile, phase, heads) in [
+        (
+            SourceIdentity::unbound(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::GgufSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::Cached,
+            16,
+        ),
+        (
+            SourceIdentity::for_new_allocation().unwrap(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            32,
+        ),
+    ] {
+        assert!(
+            InitialPreludeContract::new(
+                source,
+                profile,
+                phase,
+                InitialPreludeOperation::DecayPrefix64,
+                heads
+            )
+            .is_err()
+        );
+    }
+    let c = contract(InitialPreludeOperation::DecayPrefix64, 16);
+    let mut input = vec![0.0; c.elements()];
+    let mut out = vec![7.0; c.elements()];
+    assert!(initial_prelude_into(c, &input[..input.len() - 1], &mut out, None).is_err());
+    input[0] = f32::NAN;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    input[0] = f32::INFINITY;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = 0.0;
+    assert!(initial_prelude_into(c, &input, &mut out, Some(&AtomicBool::new(true))).is_err());
+    assert!(out.iter().all(|v| *v == 7.0));
+    input[0] = f32::MAX;
+    input[1] = f32::MAX;
+    assert!(initial_prelude_into(c, &input, &mut out, None).is_err());
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod physical {
+    use super::*;
+    use std::fs::{self, File};
+    use std::path::{Path, PathBuf};
+    use std::sync::atomic::AtomicU64;
+    use zerogravity_artifacts::artifact::Artifact;
+    use zerogravity_artifacts::cache::TransferAdmission;
+    use zerogravity_artifacts::source_weights::{
+        ExpectedSourceShard, SourceWeights, VerifiedTensorSource, WindowOpenLimits,
+    };
+    use zerogravity_core::resource_arbiter::{PhysicalDeviceKey, ResourceArbiter, ResourceOwner};
+    const META: u64 = 16 * 1024 * 1024;
+    // Existing accepted SYNTHETIC source; no trained fixture or new header alias.
+    const BYTES: &[u8] = include_bytes!("../window_projection/fixture.safetensors");
+    const INDEX: &[u8] = include_bytes!("../window_projection/fixture.index.json");
+    struct Fixture(PathBuf);
+    impl Fixture {
+        fn new() -> Self {
+            static NEXT: AtomicU64 = AtomicU64::new(1);
+            let p = std::env::temp_dir().join(format!(
+                "zg-initial-prefix-{}-{}",
+                std::process::id(),
+                NEXT.fetch_add(1, Ordering::Relaxed)
+            ));
+            fs::create_dir(&p).unwrap();
+            fs::write(p.join("model.safetensors"), BYTES).unwrap();
+            Self(p)
+        }
+        fn source(&self, h: &TransferAdmission) -> VerifiedTensorSource {
+            SourceWeights::prepare_verified_windows(
+                INDEX,
+                "42e2f4619abb5431bccce39b473b23eaa90679c9947d26c9d439308c78a86149",
+                vec![
+                    ExpectedSourceShard::new(
+                        "model.safetensors".into(),
+                        Artifact::from_file(
+                            File::open(self.0.join("model.safetensors")).unwrap(),
+                            Path::new("model.safetensors"),
+                        )
+                        .unwrap(),
+                        BYTES.len() as u64,
+                        "973fca86c0585f6b11d0b686c051bad06b896f13b2b3d10ef0299eea4510f033".into(),
+                    )
+                    .unwrap(),
+                ],
+                WindowOpenLimits {
+                    max_source_bytes: 1024 * 1024,
+                    max_metadata_bytes: META,
+                    max_shards: 1,
+                    max_tensors: 6,
+                    scan_chunk_bytes: 1024,
+                },
+                h,
+            )
+            .unwrap()
+            .open()
+            .unwrap()
+        }
+    }
+    impl Drop for Fixture {
+        fn drop(&mut self) {
+            let _ = fs::remove_dir_all(&self.0);
+        }
+    }
+    fn host() -> TransferAdmission {
+        let arbiter = ResourceArbiter::new();
+        let resource = PhysicalDeviceKey::new("host", "synthetic-initial-prefix").unwrap();
+        arbiter
+            .register(resource.clone(), 64 * 1024 * 1024)
+            .unwrap();
+        TransferAdmission {
+            arbiter,
+            resource,
+            owner: ResourceOwner::new("synthetic", "initial-prefix").unwrap(),
+        }
+    }
+    fn reserved(h: &TransferAdmission) -> u64 {
+        h.arbiter.snapshot(&h.resource).unwrap().reserved
+    }
+    fn raw(c: InitialPreludeContract) -> Vec<u8> {
+        1.0f32.to_le_bytes().repeat(c.elements())
+    }
+    #[test]
+    fn admission_foreign_cancel_pressure_clone_and_lastdrop() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let foreign = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::QueryScale128,
+            48,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let before = reserved(&h);
+        assert!(execute_initial_prelude(&foreign, input, None).is_err());
+        assert!(execute_initial_prelude(&source, input, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), before);
+        let need = InitialPreludeOutput::required_bytes(c) as u64;
+        let hold = h.arbiter.snapshot(&h.resource).unwrap().remaining - (need - 1);
+        let pressure = source.reserve_workspace(hold as usize).unwrap();
+        let pressured = reserved(&h);
+        assert!(execute_initial_prelude(&source, input, None).is_err());
+        assert_eq!(reserved(&h), pressured);
+        drop(pressure);
+        assert_eq!(reserved(&h), before);
+        let output = execute_initial_prelude(&source, input, None).unwrap();
+        let clone = output.clone();
+        assert!(output.same_issuer(&source));
+        assert!(!output.same_issuer(&foreign));
+        assert_eq!(reserved(&h), before + need);
+        assert_eq!(output.charged_bytes() as u64, need);
+        drop(foreign);
+        drop(source);
+        assert_eq!(reserved(&h), META + need);
+        assert_eq!(clone.imported_input().len(), 48 * 128);
+        assert!(clone.output().iter().all(|v| v.to_bits() == 0x3db5_04f3));
+        drop(output);
+        assert_eq!(reserved(&h), META + need);
+        drop(clone);
+        assert_eq!(reserved(&h), 0);
+    }
+    #[test]
+    fn completed_result_cancel_drops_real_admitted_backing() {
+        let f = Fixture::new();
+        let h = host();
+        let source = f.source(&h);
+        let c = InitialPreludeContract::new(
+            source.source_identity(),
+            ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1,
+            GdnPreludePhase::InitialChunk64,
+            InitialPreludeOperation::DecayPrefix64,
+            16,
+        )
+        .unwrap();
+        let raw = raw(c);
+        let input = InitialPreludeInput::from_f32le(c, &raw).unwrap();
+        let value = execute_initial_prelude(&source, input, None).unwrap();
+        assert!(reserved(&h) > META);
+        assert!(super::super::admitted::completed(value, Some(&AtomicBool::new(true))).is_err());
+        assert_eq!(reserved(&h), META);
+        drop(source);
+        assert_eq!(reserved(&h), 0);
+    }
+}
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.PYTORCH-LICENSE
@@ -0,0 +1,84 @@
+From PyTorch:
+
+Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
+Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
+Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
+Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
+Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
+Copyright (c) 2011-2013 NYU                      (Clement Farabet)
+Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
+Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
+Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
+
+From Caffe2:
+
+Copyright (c) 2016-present, Facebook Inc. All rights reserved.
+
+All contributions by Facebook:
+Copyright (c) 2016 Facebook Inc.
+
+All contributions by Google:
+Copyright (c) 2015 Google Inc.
+All rights reserved.
+
+All contributions by Yangqing Jia:
+Copyright (c) 2015 Yangqing Jia
+All rights reserved.
+
+All contributions by Kakao Brain:
+Copyright 2019-2020 Kakao Brain
+
+All contributions by Cruise LLC:
+Copyright (c) 2022 Cruise LLC.
+All rights reserved.
+
+All contributions by Tri Dao:
+Copyright (c) 2024 Tri Dao.
+All rights reserved.
+
+All contributions by Arm:
+Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates
+
+All contributions from Caffe:
+Copyright(c) 2013, 2014, 2015, the respective contributors
+All rights reserved.
+
+All other contributions:
+Copyright(c) 2015, 2016 the respective contributors
+All rights reserved.
+
+Caffe2 uses a copyright model similar to Caffe: each contributor holds
+copyright over their contributions to Caffe2. The project versioning records
+all such contribution and copyright details. If a contributor wants to further
+mark their specific copyright on a particular contribution, they should
+indicate their copyright solely in the commit message of the change when it is
+committed.
+
+All rights reserved.
+
+Redistribution and use in source and binary forms, with or without
+modification, are permitted provided that the following conditions are met:
+
+1. Redistributions of source code must retain the above copyright
+   notice, this list of conditions and the following disclaimer.
+
+2. Redistributions in binary form must reproduce the above copyright
+   notice, this list of conditions and the following disclaimer in the
+   documentation and/or other materials provided with the distribution.
+
+3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
+   and IDIAP Research Institute nor the names of its contributors may be
+   used to endorse or promote products derived from this software without
+   specific prior written permission.
+
+THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
+AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
+IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
+ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
+LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
+CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
+SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
+INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
+CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
+ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
+POSSIBILITY OF SUCH DAMAGE.
--- /dev/null
+++ b/crates/zerogravity-cpu/src/original_hf/initial_prelude.rs
@@ -0,0 +1,155 @@
+//! Closed initial-GDN leaves; no L2/exp/chunk solver, state or model permit.
+//! Cumsum schedule: Torch cf30153 ReduceOpsKernel.cpp79–95,
+//! AccumulateType.h64–66/155. Attribution: initial_prelude.PYTORCH-LICENSE.
+use crate::accelerator::{SourceIdentity, program::ProgramNumericalProfile};
+use crate::{Error, Result};
+use std::sync::atomic::{AtomicBool, Ordering};
+
+/// Phase is explicit; cached query division is deliberately unsupported here.
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum GdnPreludePhase {
+    InitialChunk64,
+    Cached,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub enum InitialPreludeOperation {
+    /// One already-normalized contiguous `[heads,1,128]` FP32 query.
+    QueryScale128,
+    /// One already-padded contiguous `[heads,1,64]` FP32 decay chunk.
+    DecayPrefix64,
+}
+#[derive(Clone, Copy, Debug, Eq, PartialEq)]
+pub struct InitialPreludeContract {
+    source: SourceIdentity,
+    profile: ProgramNumericalProfile,
+    operation: InitialPreludeOperation,
+    heads: usize,
+}
+impl InitialPreludeContract {
+    pub fn new(
+        source: SourceIdentity,
+        profile: ProgramNumericalProfile,
+        phase: GdnPreludePhase,
+        operation: InitialPreludeOperation,
+        heads: usize,
+    ) -> Result<Self> {
+        if phase != GdnPreludePhase::InitialChunk64
+            || profile != ProgramNumericalProfile::TorchCpuBf16Avx2MixedSourceV1
+        {
+            return Err(Error::Unsupported("initial GDN leaf phase/profile".into()));
+        }
+        if !source.is_bound() || !matches!(heads, 16 | 48) {
+            return Err(Error::Malformed("initial GDN leaf source/heads".into()));
+        }
+        Ok(Self {
+            source,
+            profile,
+            operation,
+            heads,
+        })
+    }
+    pub fn source_identity(self) -> SourceIdentity {
+        self.source
+    }
+    pub fn numerical_profile(self) -> ProgramNumericalProfile {
+        self.profile
+    }
+    pub fn phase(self) -> GdnPreludePhase {
+        GdnPreludePhase::InitialChunk64
+    }
+    pub fn operation(self) -> InitialPreludeOperation {
+        self.operation
+    }
+    pub fn heads(self) -> usize {
+        self.heads
+    }
+    pub fn elements(self) -> usize {
+        self.heads
+            * match self.operation {
+                InitialPreludeOperation::QueryScale128 => 128,
+                InitialPreludeOperation::DecayPrefix64 => 64,
+            }
+    }
+}
+fn not_cancelled(cancelled: Option<&AtomicBool>) -> Result<()> {
+    if cancelled.is_some_and(|flag| flag.load(Ordering::Acquire)) {
+        return Err(Error::State("initial GDN leaf cancelled".into()));
+    }
+    Ok(())
+}
+fn environment() -> Result<()> {
+    #[cfg(target_arch = "x86_64")]
+    {
+        let mut mxcsr = 0u32;
+        // SAFETY: stmxcsr writes exactly four bytes to this valid local word.
+        // Passive read only; exception status bits may change during arithmetic.
+        unsafe {
+            std::arch::asm!("stmxcsr [{p}]", p = in(reg) (&mut mxcsr as *mut u32),
+                options(nostack, preserves_flags));
+        }
+        if !std::arch::is_x86_feature_detected!("avx2") || mxcsr & 0xffc0 != 0x1f80 {
+            return Err(Error::Unsupported(
+                "initial GDN pinned CPU FP controls".into(),
+            ));
+        }
+        Ok(())
+    }
+    #[cfg(not(target_arch = "x86_64"))]
+    Err(Error::Unsupported(
+        "initial GDN pinned CPU architecture".into(),
+    ))
+}
+
+/// No allocation/provider callback. Caller owns both complete, disjoint slices.
+/// Output remains private on error; successful completion grants no state effect.
+pub fn initial_prelude_into(
+    contract: InitialPreludeContract,
+    input: &[f32],
+    output: &mut [f32],
+    cancelled: Option<&AtomicBool>,
+) -> Result<()> {
+    not_cancelled(cancelled)?;
+    environment()?;
+    if input.len() != contract.elements()
+        || output.len() != input.len()
+        || input.iter().any(|v| !v.is_finite())
+    {
+        return Err(Error::Malformed(
+            "initial GDN complete finite FP32 span".into(),
+        ));
+    }
+    match contract.operation {
+        InitialPreludeOperation::QueryScale128 => {
+            // Source query * (128 ** -0.5), scalar converted once to FP32.
+            // No reciprocal/division, FMA, BF16 cast or normalization rewrite.
+            const SCALE: f32 = f32::from_bits(0x3db5_04f3);
+            for (row, target) in input.chunks_exact(128).zip(output.chunks_exact_mut(128)) {
+                not_cancelled(cancelled)?;
+                for (&value, result) in row.iter().zip(target) {
+                    *result = value * SCALE;
+                }
+            }
+        }
+        InitialPreludeOperation::DecayPrefix64 => {
+            for (row, target) in input.chunks_exact(64).zip(output.chunks_exact_mut(64)) {
+                not_cancelled(cancelled)?;
+                let mut carry = 0.0f64;
+                for (&value, result) in row.iter().zip(target) {
+                    carry += f64::from(value);
+                    *result = carry as f32; // Store each prefix; retain unrounded F64 carry.
+                    if !result.is_finite() {
+                        return Err(Error::Malformed("initial GDN prefix FP32 overflow".into()));
+                    }
+                }
+            }
+        }
+    }
+    not_cancelled(cancelled)
+}
+
+#[cfg(all(feature = "cache", target_os = "linux"))]
+mod admitted;
+#[cfg(all(feature = "cache", target_os = "linux"))]
+pub use admitted::{InitialPreludeInput, InitialPreludeOutput, execute_initial_prelude};
+#[cfg(test)]
+mod tests;
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -4,6 +4,7 @@
 //! This does not enable complete trained Qwen3.8 inference.
 
 pub mod coefficient_preparation;
+pub mod initial_prelude;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;

````

## Source: .agents/runs/model-execution-hf-initial-prefix-20261020/frozen-prepared/ROOT_WIRING.patch

Original UTF-8 bytes: 286; SHA256 `971e77af71031eefcb479f2524f63f417257ea237d83bfa859ecfa661175fa64`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- a/crates/zerogravity-cpu/src/original_hf.rs
+++ b/crates/zerogravity-cpu/src/original_hf.rs
@@ -4,6 +4,7 @@
 //! This does not enable complete trained Qwen3.8 inference.
 
 pub mod coefficient_preparation;
+pub mod initial_prelude;
 pub mod conv;
 pub mod norm;
 pub mod recurrent;

````

## Source: .agents/runs/model-execution-hf2b-attention-paired-20261020/NEW_PREPARATION.patch

Original UTF-8 bytes: 174133; SHA256 `e8d6897bf2fb73ccb4741a8921f53568393fa6b3ddfcc6e8bbe55c79c6c360d6`.
Resolve embedded relative paths against the original path, not this volume.

````text
--- /dev/null
+++ CAPTURE_INTEGRATION.review.diff
@@ -0,0 +1,288 @@
+--- accepted-capture
++++ paired-capture
+@@ -14,6 +14,13 @@
+ import resource
+ import sys
+ import time
++
++# Pin the four allowed CPUs before importing the original Torch runtime.
++allowed = [int(value) for value in os.environ["ZG_PROBE_ALLOWED_CPUS"].split(",")]
++assert len(allowed) == len(set(allowed)) == 4 and set(allowed) <= os.sched_getaffinity(0)
++os.sched_setaffinity(0, allowed)
++MODE = os.environ["ZG_ATTENTION_MODE"]
++assert MODE in ("plain", "observed")
+ 
+ import torch
+ from safetensors import safe_open
+@@ -250,115 +257,147 @@
+     handles.append(module.register_forward_pre_hook(lambda _module, args: remember(name, args[0])))
+ 
+ 
+-language = model.model.language_model
+-layer0 = language.layers[0]
+-output_hook(language.embed_tokens, "embedding")
+-output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
+-output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
+-output_hook(layer0.linear_attn, "layer-00.attention_output")
+-input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
+-output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
+-output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
+-output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
+-input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
+-output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
+-output_hook(layer0, "layer-00.post_ffn_residual")
+-output_hook(language.norm, "final_norm")
+-
+-manifest = {
+-    "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
+-    "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
+-    "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
+-    "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
+-    "runtime_source": runtime_source, "package_records": PINS["package_records"],
+-    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
+-    "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
+-    "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
+-    "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
+-    "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
+-    "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
+-    "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
+-    "profiles": {},
+-    "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
+-}
+-model_source = pathlib.Path(inspect.getsourcefile(official))
+-manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
+-manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
+-
+-for profile in PROFILES:
+-    if profile == "source-36-f32-restored":
+-        with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+-            for name in f32_names:
+-                parameter = model.get_parameter(name)
+-                parameter.data = source_handle.get_tensor(name).clone()
+-                assert parameter.dtype == torch.float32
+-        print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
+-    profile_dir = OUTPUT / profile
+-    profile_dir.mkdir()
+-    audit = parameter_audit(model, source_file)
+-    write_json(profile_dir / "parameter-audit.json", audit)
+-    assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
+-    dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
+-    print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
+-    profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
+-    cache = None
+-    with torch.inference_mode():
+-        for step, token in enumerate(TOKENS):
+-            intermediates = {}
+-            calls = collections.Counter()
+-            signatures = {}
+-
+-            def observe_calls(frame, event, arg):
+-                if frame.f_code.co_filename != str(model_source):
+-                    return
+-                name = frame.f_code.co_name
+-                if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+-                    caller = frame.f_back
+-                    while caller is not None:
+-                        owner = caller.f_locals.get("self")
+-                        if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+-                            if owner.layer_idx == 0:
+-                                remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
+-                            break
+-                        caller = caller.f_back
+-                    return
+-                if event != "call":
+-                    return
+-                if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
+-                    key = name + ":" + str(frame.f_code.co_firstlineno)
+-                    calls[key] += 1
+-                    signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
+-
+-            step_start = time.time()
+-            sys.setprofile(observe_calls)
+-            try:
+-                result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
+-            finally:
+-                sys.setprofile(None)
+-            cache = result.past_key_values
+-            expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
+-            observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
+-            assert observed_by_name == expected, (expected, observed_by_name)
+-            step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
+-            remember("final_logits", result.logits)
+-            remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
+-            assert len(intermediates) == 15
+-            intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
+-            save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
+-            step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
+-            step_result["observed_official_operator_calls"] = dict(calls)
+-            step_result["last_observed_operator_tensor_arguments"] = signatures
+-            step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
+-            write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
+-            profile_result["steps"].append(step_result)
+-            artifact_bound()
+-            print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
+-    manifest["profiles"][profile] = profile_result
+-    write_json(OUTPUT / "manifest.json", manifest)
+-    del result, cache
+-
+-for handle in handles:
+-    handle.remove()
++from attention_observation_hook import AttentionProjectionObserver
++attention = None
++try:
++    language = model.model.language_model
++    layer0 = language.layers[0]
++    output_hook(language.embed_tokens, "embedding")
++    output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
++    output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
++    output_hook(layer0.linear_attn, "layer-00.attention_output")
++    input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
++    output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
++    output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
++    output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
++    input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
++    output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
++    output_hook(layer0, "layer-00.post_ffn_residual")
++    output_hook(language.norm, "final_norm")
++
++    manifest = {
++        "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
++        "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
++        "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
++        "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
++        "runtime_source": runtime_source, "package_records": PINS["package_records"],
++        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
++        "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
++        "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
++        "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
++        "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
++        "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
++        "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
++        "profiles": {},
++        "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
++    }
++    model_source = pathlib.Path(inspect.getsourcefile(official))
++    manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
++    manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
++
++    if MODE == "observed":
++        attention = AttentionProjectionObserver(model, official)
++    for profile in PROFILES:
++        if profile == "source-36-f32-restored":
++            with safe_open(source_file, framework="pt", device="cpu") as source_handle:
++                for name in f32_names:
++                    parameter = model.get_parameter(name)
++                    parameter.data = source_handle.get_tensor(name).clone()
++                    assert parameter.dtype == torch.float32
++            print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
++        profile_dir = OUTPUT / profile
++        profile_dir.mkdir()
++        audit = parameter_audit(model, source_file)
++        write_json(profile_dir / "parameter-audit.json", audit)
++        assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
++        dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
++        print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
++        profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
++        cache = None
++        with torch.inference_mode():
++            for step, token in enumerate(TOKENS):
++                intermediates = {}
++                calls = collections.Counter()
++                signatures = {}
++
++                def observe_calls(frame, event, arg):
++                    if attention is not None:
++                        attention.observe_source_return(frame, event, arg)
++                    if frame.f_code.co_filename != str(model_source):
++                        return
++                    name = frame.f_code.co_name
++                    if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
++                        caller = frame.f_back
++                        while caller is not None:
++                            owner = caller.f_locals.get("self")
++                            if isinstance(owner, official.Qwen3_5GatedDeltaNet):
++                                if owner.layer_idx == 0:
++                                    remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
++                                break
++                            caller = caller.f_back
++                        return
++                    if event != "call":
++                        return
++                    if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
++                        key = name + ":" + str(frame.f_code.co_firstlineno)
++                        calls[key] += 1
++                        signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
++
++                if attention is not None:
++                    attention.begin_step(profile, step, token)
++                step_start = time.time()
++                sys.setprofile(observe_calls)
++                try:
++                    result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
++                finally:
++                    sys.setprofile(None)
++                cache = result.past_key_values
++                expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
++                observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
++                assert observed_by_name == expected, (expected, observed_by_name)
++                step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
++                if attention is not None:
++                    attention.finish_step(profile_dir / f"step-{step}" / "attention-observation")
++                remember("final_logits", result.logits)
++                remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
++                assert len(intermediates) == 15
++                intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
++                save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
++                step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
++                step_result["observed_official_operator_calls"] = dict(calls)
++                step_result["last_observed_operator_tensor_arguments"] = signatures
++                step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
++                write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
++                profile_result["steps"].append(step_result)
++                artifact_bound()
++                print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
++        manifest["profiles"][profile] = profile_result
++        write_json(OUTPUT / "manifest.json", manifest)
++        del result, cache
++
++finally:
++    # Every installed hook is removed even after forward/capture/receipt failure.
++    primary = sys.exception()
++    removal = []
++    try:
++        sys.setprofile(None)
++    except BaseException as error:
++        removal.append(error)
++    if attention is not None:
++        try:
++            attention.close()
++        except BaseException as error:
++            removal.append(error)
++    attention = None  # Release observed module owners before archive validation.
++    for handle in reversed(handles):
++        try:
++            handle.remove()
++        except BaseException as error:
++            removal.append(error)
++    handles.clear()
++    if removal:
++        raise BaseExceptionGroup("capture and all hook removal failures",
++                                 ([primary] if primary is not None else []) + removal) from None
+ del model, embedding, language, layer0, intermediates
+ gc.collect()
+ assert verify_source() == source_receipt
+@@ -368,6 +407,14 @@
+ manifest["source_reverified_after_capture"] = True
+ manifest["elapsed_seconds_not_benchmark"] = time.time() - started
+ manifest["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
++write_json(pathlib.Path("/output/attention-mode.json"), {
++    "mode": MODE, "owned_hooks_removed": True, "profiler_retired": sys.getprofile() is None,
++    "cpu_affinity": sorted(os.sched_getaffinity(0)), "teacher_tokens": TOKENS,
++    "profiles": PROFILES, "observer_hook_sha256": sha(pathlib.Path("/probe/attention_observation_hook.py")),
++    "accepted_capture_sha256": "6ee140185766dee237ea4df802504ef7b7be80110f5566876a644414cf66049d",
++    "expected_attention_tensor_count": 216 if MODE == "observed" else 0,
++    "full_model_or_Rust_or_GPU_parity_qualified": False,
++})
+ write_json(OUTPUT / "manifest.json", manifest)
+ artifact_bound()
+ print("REFERENCE_CAPTURE_COMPLETE", json.dumps({"profiles": list(manifest["profiles"]), "steps_each": 2, "full_vocab": 248320, "semantic_domains_each": 42, "max_rss_kib": manifest["max_rss_kib"]}), flush=True)
--- /dev/null
+++ README.md
@@ -0,0 +1,90 @@
+# Original HF2B paired attention observation — ROOT executes
+
+- Owner: baseline_review. Parent: existing model-execution plan; ignored preparation only.
+- Source-only, unexecuted. No model/container/operator/test/build execution by author.
+- Exact original Qwen/Qwen3.5-2B revision15852e8c16360a2fea060d615a32b45270f8a8fc,
+  snapshot88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52.
+- CPU Transformers eager/slow GDN in the already acquired pinned image. This is
+  not vLLM inference or GPU evidence despite that image's installed packages.
+- No production source, shared plans, existing run bytes, weights or math changed.
+
+## Existing accepted reference
+
+`../model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference`
+is immutable. `parent-pins.json` pins its actual manifest/validation and original
+invocation, accepted capture/validator/source metadata, Batch's frozen attention
+hook and the reviewed GEMM `owned_container_v4.py` (SHA53ee4270…). Original raw
+archive hashes come from that pinned accepted manifest; author did not scan or
+load raw states, model payloads or large DSOs.
+
+Both source profiles remain separate: `official-bf16-loader` then
+`source-36-f32-restored`, each teacher11 then353 with its own fresh cache. The
+original loader/casts, source/RECORD audits, parameter audit, module hooks,
+token calls, snapshot codecs and unchanged archive validator remain in use.
+`CAPTURE_INTEGRATION.review.diff` exposes every insertion. Original named
+functions are AST-identical except delegation at the beginning of the existing
+`observe_calls` callback. No replacement profiler, tensor, operator or model.
+
+## New observation and transparency gates
+
+- Fresh serialized plain then observed containers, each running the accepted
+  two-profile/two-step procedure. No historical failed attempt is reused.
+- Copied exact `AttentionProjectionObserver` observes six FullAttention layers
+  3/7/11/15/19/23, nine BF16 tensors each, four profile/steps:216 tensors total.
+  Stages: hidden, query/gate projection, K/V projections, raw query/key,
+  normalized query/key, gate. Actual original module hooks and source-return
+  gate copy only; original profiler call counts remain authoritative.
+- Expected genuine2B shapes: hidden/gate2048, query/gate projection4096,
+  query8×256, K/V2×256. Raw query stride512 preserves interleaved query/gate.
+  These are stage transport descriptors, not reconstructed expected arithmetic.
+- Every fresh mode must retain all196 raw tensors, four full248320 BF16 logits,
+ 168 canonical domains and60 old intermediates byte-identical to the accepted
+  reference. Each mode also passes the unchanged complete archive/layout
+  validator;632 source/617 loaded parameter audit bytes match exactly.
+- Stdlib comparator reads bounded Safetensors descriptors and exact payloads,
+  including signed zeros; no Torch import, rounding, tolerance or tensor math.
+  Source/profiles/teachers/all-layer stage namespace, finite raw BF16,
+  dtype/shape/strides/cumulative byte cap and all cleanup evidence are required.
+- All base and new hooks are owned inside one finally boundary, every removal
+  attempted, profiler retired, primary and cleanup failures retained. Observer
+  module references are released before model disposal/archive validation.
+
+## Bounds and lifetime
+
+- Existing imageIDf57d77f5…/codec manifest eec72f47…; exact local inspect, no pull.
+- No NVIDIA CDI/device, CUDA_VISIBLE_DEVICES=-1, NVIDIA_VISIBLE_DEVICES=void,
+  no preload/audit, networknone, read-only source/probe/rootfs, cap-dropall.
+- Memory+swap12GiB, CPUquota4, original4 Torch threads/interops1, AVX2. Affinity
+  sets exactly four allowed CPUs before Torch import and validates afterward;
+  no unavailable OCI cpuset controller is requested.
+- Each capture300s outer/290s original alarm; original aggregate capture256MiB,
+  per-file256MiB, helper log4MiB. Additional attention raw payload172032B/step,
+  688128B total; its independent cumulative cap64MiB remains unchanged.
+- Original4.548GB checkpoint/model memory estimate remains an estimate, not
+  physical admission. Actual cgroup peak/RSS remain in each reference manifest.
+- Reviewed helper is copied unchanged: WNOWAIT keeps owned PID reserved;
+  independent TERM/grace/two KILL stages precede reap, exact nonce/name/CID
+  discovery before and after, group/container absence and pipe EOF/closure.
+  Postspawn receipt errors cannot bypass cleanup. Failed evidence prevents the
+  next mode and final PASS. Existing processes/containers are never targeted.
+- Final bounded stdlib comparison120s, block≤1MiB, tensor/archive geometry and
+  complete EOF enforced. Separate fresh output directly beneath this run.
+
+## ROOT-only commands after source review
+
+```sh
+cd /home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-attention-paired-20261020
+PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_validate_pair
+PYTHONDONTWRITEBYTECODE=1 python3 run_pair.py --output "$PWD/root-pair-1"
+```
+
+Ten fixture methods are prepared, not run by author. Synthetic bytes test strict
+raw comparison, missing stages, teacher/profile binding, query stride and finite
+guards; they are never genuine trained expectations. ROOT should independently
+authenticate all prepared source and parent pins before either command.
+
+Final possible scope: `PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY`. This
+qualifies observation transparency and actual reference-stage availability for
+the named two CPU profiles/two teacher tokens. It does not qualify Rust private
+slice output, full HF execution/SessionAuthority/StateRoot, GPU/FLA, attention
+RoPE/BMM/softmax integration, long context, trained27B or matched performance.
--- /dev/null
+++ TODO.md
@@ -0,0 +1,13 @@
+# Own run — original HF2B paired attention observation
+
+- Parent: `.agents/plans/model-execution/TODO.md`; owner baseline_review.
+- Owned scope: this ignored preparation directory only; no shared tracker.
+- [x] Read accepted two-profile CPU capture/validator/root-probe-2 metadata and Batch frozen hook.
+- [x] Preserve exact original source/parameter/semantic/raw math path; finally-owned observation integration.
+- [x] Prepare exact reviewed owned-container lifecycle, serialized fresh plain/observed launcher and strict raw comparator.
+- [x] Prepare216 actual-stage contract and10 source-free fixtures; retain explicit CPU-only/reference-only labels.
+- [x] AST/static source checks and frozen source/parent pins only; no executed tests or model/runtime claims.
+- [ ] Independent source review and ROOT fixture execution.
+- [ ] ROOT paired capture; each mode all196 raw tensors/full4 logits/all168 domains/60 intermediates byte-exact to accepted reference.
+- [ ] ROOT actual216 attention stages, source audit, cgroup/affinity and full owned cleanup validation.
+- [ ] Later separate Rust operator-slice replay against these authentic stages; full HF/session/GPU guards remain closed.
--- /dev/null
+++ attention_observation_hook.py
@@ -0,0 +1,189 @@
+"""ROOT ONLY. Transparent hooks; outer accepted loader/source/lifetime stay owned.
+
+No model factory or token loop. Integrate observe_source_return into the existing
+accepted profiler callback; never replace it or set a new global profiler here.
+Full logits/cache transparency remains an independent outer acceptance gate.
+"""
+import hashlib
+import json
+from pathlib import Path
+
+MODELING_SHA256 = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+TORCH_SOURCE = "cf30153c4c131c8164ee7798e5022d810682e2cb"
+PROFILES = {"official-bf16-loader", "source-36-f32-restored"}
+MAX_PAYLOAD = 64 * 1024 * 1024
+STAGES = ("hidden", "query_gate_projection", "key_projection", "value_projection",
+          "query", "key", "normalized_query", "normalized_key", "gate")
+
+
+def require(value, message):
+    if not value:
+        raise RuntimeError(message)
+
+
+class AttentionProjectionObserver:
+    """Observe genuine CPU public modules at all actual FullAttention layers.
+
+    Call only after the unchanged accepted loader and parameter/source audit.
+    No hook returns a replacement. The copy precedes possible downstream reuse.
+    """
+
+    def __init__(self, model, official):
+        import torch
+        self.torch = torch
+        self.official = official
+        self.source_path = str(Path(official.__file__).resolve())
+        require(hashlib.sha256(Path(self.source_path).read_bytes()).hexdigest()
+                == MODELING_SHA256, "exact accepted original modeling source")
+        require(torch.version.git_version == TORCH_SOURCE,
+                "exact accepted Torch source")
+        require(torch.backends.cpu.get_cpu_capability() == "AVX2",
+                "closed original CPU runtime capability")
+        self.language = model.model.language_model
+        cfg = model.config.text_config
+        require(cfg.hidden_size in (1024, 2048) and cfg.head_dim == 256,
+                "approved .8B/2B attention observation geometry")
+        self.hidden = cfg.hidden_size
+        self.query_heads = cfg.num_attention_heads
+        self.kv_heads = cfg.num_key_value_heads
+        self.layers = {}
+        for index, layer in enumerate(self.language.layers):
+            if cfg.layer_types[index] != "full_attention":
+                continue
+            attention = layer.self_attn
+            require(isinstance(attention, official.Qwen3_5Attention)
+                    and attention.layer_idx == index, "actual original attention module")
+            require(tuple(attention.q_proj.weight.shape)
+                    == (2*self.query_heads*256, self.hidden)
+                    and tuple(attention.k_proj.weight.shape)
+                    == tuple(attention.v_proj.weight.shape)
+                    == (self.kv_heads*256, self.hidden), "unchanged full source matrices")
+            require(all(parameter.device.type == "cpu" for parameter in attention.parameters()),
+                    "CPU-only actual source module")
+            self.layers[index] = attention
+        require(len(self.layers) == 6, "all six original FullAttention modules")
+        self.handles = []
+        self.active = None
+        self.seen = set()
+        self.payload = 0
+        self.tensors = {}
+        self.metadata = {}
+        try:
+            for index, attention in self.layers.items():
+                self.handles.append(attention.register_forward_pre_hook(
+                    self._input_hook(index, "hidden"), with_kwargs=True))
+                for member, name in (("q_proj", "query_gate_projection"),
+                                     ("k_proj", "key_projection"),
+                                     ("v_proj", "value_projection"),
+                                     ("q_norm", "normalized_query"),
+                                     ("k_norm", "normalized_key")):
+                    self.handles.append(getattr(attention, member).register_forward_hook(
+                        self._output_hook(index, name)))
+                for member, name in (("q_norm", "query"), ("k_norm", "key")):
+                    self.handles.append(getattr(attention, member).register_forward_pre_hook(
+                        self._input_hook(index, name), with_kwargs=True))
+        except BaseException as primary:
+            try:
+                self.close()
+            except BaseException as cleanup:
+                raise BaseExceptionGroup("observer install and removal failed", [primary, cleanup]) from None
+            raise
+
+    def _remember(self, layer, stage, value):
+        torch = self.torch
+        require(self.active is not None, "unexpected attention call outside fixed teacher step")
+        key = f"layer-{layer:02d}.{stage}"
+        require(key not in self.tensors, "duplicate genuine attention stage: " + key)
+        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
+                and value.dtype == torch.bfloat16 and value.numel() <= 16384,
+                "closed actual BF16 attention tensor: " + key)
+        require(torch.is_inference_mode_enabled(), "unchanged accepted inference mode")
+        expected = {"hidden": self.hidden,
+                    "query_gate_projection": 2*self.query_heads*256,
+                    "key_projection": self.kv_heads*256,
+                    "value_projection": self.kv_heads*256,
+                    "query": self.query_heads*256, "key": self.kv_heads*256,
+                    "normalized_query": self.query_heads*256,
+                    "normalized_key": self.kv_heads*256, "gate": self.query_heads*256}[stage]
+        require(value.numel() == expected and value.shape[0] == value.shape[1] == 1,
+                "single teacher original attention shape: " + key)
+        count = value.numel()*value.element_size()
+        require(self.payload+count <= MAX_PAYLOAD, "bounded attention-only tensor payload")
+        copied = value.detach().cpu().contiguous().clone()
+        require(torch.equal(value.view(torch.uint16), copied.view(torch.uint16)),
+                "bit-preserving observational copy, including signed zero")
+        self.payload += count
+        self.tensors[key] = copied
+        self.metadata[key] = {"dtype": str(value.dtype), "shape": list(value.shape),
+                              "original_stride": list(value.stride()),
+                              "saved_stride": list(copied.stride()), "bytes": count,
+                              "source_module_layer": layer, "stage": stage}
+        # Implicit None: never return a replacement to a public module hook.
+
+    def _input_hook(self, layer, stage):
+        def observe(_module, args, kwargs):
+            require(bool(args) != ("hidden_states" in kwargs),
+                    "one genuine public activation argument")
+            self._remember(layer, stage, args[0] if args else kwargs["hidden_states"])
+        return observe
+
+    def _output_hook(self, layer, stage):
+        def observe(_module, _args, output):
+            self._remember(layer, stage, output)
+        return observe
+
+    def begin_step(self, profile, step, token):
+        require(self.active is None and not self.tensors, "previous attention step retired")
+        require(profile in PROFILES and step in (0, 1) and token == (11, 353)[step],
+                "same accepted two-token teacher protocol")
+        require((profile, step) not in self.seen, "no repeated source-profile step")
+        self.seen.add((profile, step))
+        self.active = {"profile": profile, "step": step, "teacher_token": token}
+
+    def observe_source_return(self, frame, event, _arg):
+        # Delegate from the existing accepted callback, without changing its
+        # original GDN/conv/attention call-count observations or source math.
+        if event != "return" or frame.f_code.co_filename != self.source_path:
+            return
+        owner = frame.f_locals.get("self")
+        if frame.f_code.co_name != "forward" or not isinstance(owner, self.official.Qwen3_5Attention):
+            return
+        layer = owner.layer_idx
+        require(self.layers.get(layer) is owner, "genuine observed original layer owner")
+        require("gate" in frame.f_locals, "actual untouched forward gate local")
+        self._remember(layer, "gate", frame.f_locals["gate"])
+
+    def finish_step(self, directory):
+        from safetensors.torch import save_file
+        expected = {f"layer-{layer:02d}.{stage}" for layer in self.layers for stage in STAGES}
+        require(self.active is not None and set(self.tensors) == expected,
+                "all actual layers/stages exactly once")
+        directory = Path(directory)
+        directory.mkdir(exist_ok=False)
+        archive = directory/"attention.safetensors"
+        save_file(self.tensors, str(archive), metadata={
+            "scope": "actual public source observation only; full transparency unqualified",
+            **{key: str(value) for key, value in self.active.items()}})
+        report = {"status": "OBSERVED_OPERATORS_ONLY_FULL_TRANSPARENCY_UNQUALIFIED",
+                  **self.active, "modeling_sha256": MODELING_SHA256,
+                  "torch_source": TORCH_SOURCE, "tensor_count": len(self.tensors),
+                  "aggregate_payload_bytes": self.payload, "tensors": self.metadata,
+                  "archive": {"file": archive.name, "bytes": archive.stat().st_size,
+                              "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()},
+                  "no_model_or_state_or_provider_eligibility": True}
+        (directory/"observation.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
+        self.tensors = {}
+        self.metadata = {}
+        self.active = None
+        return report
+
+    def close(self):
+        errors = []
+        for handle in reversed(self.handles):
+            try:
+                handle.remove()
+            except BaseException as error:
+                errors.append(repr(error))
+        self.handles = []
+        if errors:
+            raise RuntimeError("all owned hook removals attempted: " + repr(errors))
--- /dev/null
+++ baseline/capture_reference.py
@@ -0,0 +1,373 @@
+"""Pinned, offline CPU reference; operators remain the installed official source."""
+
+import collections
+import gc
+import signal
+import stat
+import hashlib
+import importlib.metadata
+import inspect
+import json
+import os
+import pathlib
+import platform
+import resource
+import sys
+import time
+
+import torch
+from safetensors import safe_open
+from safetensors.torch import save_file
+from transformers import AutoTokenizer
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+
+SOURCE = pathlib.Path("/source")
+PINS = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert PINS["repo"] == "Qwen/Qwen3.5-2B" and PINS["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
+AUDIT_ELEMENTS = 262144  # ≤1MiB per F32 audit operand; model operators untouched.
+ARTIFACT_CAP = 256 * 1024 * 1024
+signal.alarm(290)  # Outer owner enforces300s and always cleans its bound CID.
+resource.setrlimit(resource.RLIMIT_FSIZE, (ARTIFACT_CAP, ARTIFACT_CAP))
+assert pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip() == str(12 * 1024**3)
+assert pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip() == "400000 100000"
+assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
+OUTPUT = pathlib.Path("/output/captured-reference")
+TOKENS = [11, 353]
+PROFILES = ["official-bf16-loader", "source-36-f32-restored"]
+torch.set_num_threads(4)
+torch.set_num_interop_threads(1)
+assert not torch.cuda.is_available()
+assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
+assert not OUTPUT.exists(), "refuse to overwrite an earlier reference"
+OUTPUT.mkdir()
+
+
+def sha(path):
+    digest = hashlib.sha256()
+    with open(path, "rb") as stream:
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(chunk)
+    return digest.hexdigest()
+
+
+def write_json(path, data):
+    def official_json_types(value):
+        if isinstance(value, set):
+            return sorted(value)
+        raise TypeError(f"unhandled JSON type {type(value).__name__}")
+
+    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=official_json_types) + "\n")
+
+
+def verify_source():
+    for item in PINS["source_files"]:
+        path = SOURCE / item["path"]
+        assert not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"]
+        assert sha(path) == item["sha256"], item["path"]
+    return {"repo": PINS["repo"], "revision": PINS["revision"], "snapshot": PINS["snapshot"],
+            "files": PINS["source_files"], "independent_verification": PINS["root_source_validation"]}
+
+
+def audit_chunks(value):
+    assert value.device.type == "cpu" and value.is_contiguous()
+    flat = value.detach().view(-1)
+    for start in range(0, flat.numel(), AUDIT_ELEMENTS):
+        yield flat[start:start + AUDIT_ELEMENTS]
+
+
+def finite(value):
+    return all(bool(torch.isfinite(part).all()) for part in audit_chunks(value.detach().contiguous()))
+
+
+def artifact_bound():
+    total = 0
+    for path in OUTPUT.rglob("*"):
+        assert not path.is_symlink()
+        if path.is_file():
+            total += path.stat().st_size
+    assert total <= ARTIFACT_CAP, ("aggregate capture cap", total)
+    return total
+
+
+def verify_runtime():
+    expected = PINS["runtime_source"]
+    for part in expected["files"].values():
+        path = pathlib.Path(part["runtime_path"])
+        assert sha(path) == part["sha256"], str(path)
+    for name, pin in PINS["package_records"].items():
+        distribution = importlib.metadata.distribution(name)
+        assert distribution.version == pin["version"], name
+        record = distribution.read_text("RECORD")
+        assert record is not None and hashlib.sha256(record.encode()).hexdigest() == pin["record_sha256"], name
+    assert torch.__version__ == "2.13.0+cu129" and importlib.metadata.version("transformers") == "5.17.0"
+    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
+    return expected
+
+
+def tensor_info(value):
+    return {
+        "dtype": str(value.dtype), "shape": list(value.shape),
+        "stride": list(value.stride()), "numel": value.numel(),
+        "bytes": value.numel() * value.element_size(),
+        "device": str(value.device), "contiguous": value.is_contiguous(),
+        "finite": finite(value),
+    }
+
+
+def f32_file(directory, name, value):
+    path = directory / (name + ".f32le.bin")
+    wide = value.detach().cpu().float().contiguous()
+    assert finite(wide)
+    # numpy explicitly requests little-endian; widening BF16 is exact.
+    path.write_bytes(wide.numpy().astype("<f4", copy=False).tobytes())
+    return {"file": str(path.relative_to(OUTPUT)), "sha256": sha(path), **tensor_info(wide)}
+
+
+def parameter_audit(model, source_file):
+    rows = []
+    with safe_open(source_file, framework="pt", device="cpu") as handle:
+        for name in sorted(handle.keys()):
+            original = handle.get_tensor(name)
+            try:
+                actual = model.get_parameter(name)
+            except AttributeError:
+                assert name.startswith("mtp."), name
+                rows.append({"name": name, "source": tensor_info(original), "loaded": None, "exact_after_declared_dtype_cast": None, "exclusion": "official Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected=['^mtp.*']; auxiliary MTP is absent from the ordinary conditional-generation class"})
+                continue
+            assert actual.device.type == "cpu"
+            assert actual.shape == original.shape
+            assert original.is_contiguous() and actual.is_contiguous()
+            cast_changed, cast_max = 0, 0.0
+            for left, right in zip(audit_chunks(original), audit_chunks(actual)):
+                cast = left.to(right.dtype)
+                assert torch.equal(cast.view(torch.uint8), right.view(torch.uint8)), name
+                if original.dtype != actual.dtype:
+                    delta = left.float() - right.float()
+                    cast_changed += int((delta != 0).sum())
+                    cast_max = max(cast_max, float(delta.abs().max()))
+            row = {"name": name, "source": tensor_info(original), "loaded": tensor_info(actual)}
+            row["exact_after_declared_dtype_cast"] = True
+            if original.dtype != actual.dtype:
+                row["cast_changed_elements"] = cast_changed
+                row["cast_max_abs_delta"] = cast_max
+            rows.append(row)
+    return rows
+
+
+def snapshot(cache, logits, directory, profile, step, token):
+    directory.mkdir()
+    raw = {"logits": logits.detach().cpu().contiguous().clone()}
+    raw_meta = {"logits": tensor_info(logits)}
+    domains = []
+    assert len(cache.layers) == 24
+    assert logits.shape == (1, 1, 248320)
+    for index, layer in enumerate(cache.layers):
+        prefix = f"layer-{index:02d}"
+        if hasattr(layer, "conv_states"):
+            assert set(layer.conv_states) == {0} and set(layer.recurrent_states) == {0}
+            conv, recurrent = layer.conv_states[0], layer.recurrent_states[0]
+            assert conv.shape == (1, 6144, 4)
+            assert recurrent.shape == (1, 16, 128, 128)
+            assert recurrent.dtype == torch.float32 and conv.dtype == torch.bfloat16
+            for kind, value in [("conv", conv), ("recurrent", recurrent)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            # Official cache retains four raw projection positions. The next
+            # four-tap convolution needs the newest three as prior history.
+            domains.append({"layer": index, "kind": "gdn-convolution-history", "layout": "batch,channel,oldest-to-newest-3", "derivation": "official conv_states[0][..., -3:]", **f32_file(directory, prefix + ".conv-history", conv[..., -3:])})
+            # Source state is key-major; the Rust state reader is value-major.
+            domains.append({"layer": index, "kind": "gdn-recurrent", "layout": "batch,head,value,key", "derivation": "official recurrent_states[0].permute(0,1,3,2)", **f32_file(directory, prefix + ".recurrent-value-major", recurrent.permute(0, 1, 3, 2))})
+        else:
+            keys, values = layer.keys, layer.values
+            assert keys.shape == values.shape == (1, 2, step + 1, 256)
+            assert keys.dtype == values.dtype == torch.bfloat16
+            for kind, value in [("keys", keys), ("values", values)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            keys = keys.permute(0, 2, 1, 3).contiguous()
+            values = values.permute(0, 2, 1, 3).contiguous()
+            pair = torch.cat([keys.reshape(-1), values.reshape(-1)])
+            domains.append({"layer": index, "kind": "attention-kv-pair", "layout": "all keys(batch,position,kv-head,width), then all values(same)", "valid_positions": step + 1, "derivation": "official keys/values permute(0,2,1,3), flatten and concatenate", **f32_file(directory, prefix + ".kv-pair", pair)})
+    assert len(domains) == 42
+    assert len(raw) == 49  # logits + 36 GDN tensors + 12 K/V tensors
+    assert all(info["finite"] for info in raw_meta.values())
+    raw_path = directory / "raw-official.safetensors"
+    save_file(raw, str(raw_path), metadata={"profile": profile, "step": str(step), "input_token": str(token), "source_revision": PINS["revision"]})
+    wide_logits = f32_file(directory, "logits", logits)
+    flat = logits.float().reshape(-1)
+    top_values, top_ids = torch.topk(flat, 10, largest=True, sorted=True)
+    result = {"profile": profile, "step": step, "input_token": token, "consumed_tokens": TOKENS[:step + 1], "cache_class": type(cache).__name__, "cache_seq_length": cache.get_seq_length(), "raw_archive": {"file": str(raw_path.relative_to(OUTPUT)), "sha256": sha(raw_path), "bytes": raw_path.stat().st_size}, "raw_tensor_layouts": raw_meta, "logits_f32": wide_logits, "argmax_token_id": int(flat.argmax()), "top10": [{"token_id": int(i), "logit": float(v)} for i, v in zip(top_ids, top_values)], "semantic_domains": domains, "semantic_domain_count": len(domains), "physical_ready_or_StateRoot_claim": False}
+    write_json(directory / "manifest.json", result)
+    return result
+
+
+started = time.time()
+runtime_source = verify_runtime()
+source_receipt = verify_source()
+write_json(OUTPUT / "source-receipt.json", source_receipt)
+source_file = SOURCE / "model.safetensors-00001-of-00001.safetensors"
+with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+    names = sorted(source_handle.keys())
+    f32_names = [name for name in names if source_handle.get_slice(name).get_dtype() == "F32"]
+assert len(names) == 632 and len(f32_names) == 36
+assert f32_names == PINS["raw_f32_names"]
+assert all(".linear_attn.A_log" in name or ".linear_attn.norm.weight" in name for name in f32_names)
+assert "lm_head.weight" not in names
+print("SOURCE_VERIFIED tensors=632 original_f32=36 no_lm_head=1", flush=True)
+
+model, loading_info = official.Qwen3_5ForConditionalGeneration.from_pretrained(
+    str(SOURCE), dtype=torch.bfloat16, local_files_only=True,
+    attn_implementation="eager", use_kernels=False, output_loading_info=True,
+)
+model.eval()
+assert not loading_info.get("missing_keys") and not loading_info.get("unexpected_keys") and not loading_info.get("mismatched_keys") and not loading_info.get("error_msgs"), loading_info
+assert all(parameter.device.type == "cpu" for parameter in model.parameters())
+embedding = model.model.language_model.embed_tokens.weight
+assert model.lm_head.weight is embedding
+assert embedding.shape == (248320, 2048)
+assert model.config.text_config.hidden_size == 2048 and model.config.text_config.num_hidden_layers == 24
+tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
+print("OFFICIAL_MODEL_LOADED tied_head=1 cuda=0", flush=True)
+
+# Observation hooks copy actual outputs, return None and never replace an
+# operator/input/output. The down-projection input exposes the actual fused
+# SiLU(gate)*up tensor without reconstructing its arithmetic.
+intermediates = {}
+handles = []
+
+
+def remember(name, value):
+    assert name not in intermediates, name
+    assert isinstance(value, torch.Tensor)
+    intermediates[name] = value.detach().cpu().contiguous().clone()
+
+
+def output_hook(module, name):
+    handles.append(module.register_forward_hook(lambda _module, _args, value: remember(name, value)))
+
+
+def input_hook(module, name):
+    handles.append(module.register_forward_pre_hook(lambda _module, args: remember(name, args[0])))
+
+
+language = model.model.language_model
+layer0 = language.layers[0]
+output_hook(language.embed_tokens, "embedding")
+output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
+output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
+output_hook(layer0.linear_attn, "layer-00.attention_output")
+input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
+output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
+output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
+output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
+input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
+output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
+output_hook(layer0, "layer-00.post_ffn_residual")
+output_hook(language.norm, "final_norm")
+
+manifest = {
+    "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
+    "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
+    "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
+    "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
+    "runtime_source": runtime_source, "package_records": PINS["package_records"],
+    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
+    "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
+    "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
+    "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
+    "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
+    "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
+    "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
+    "profiles": {},
+    "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
+}
+model_source = pathlib.Path(inspect.getsourcefile(official))
+manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
+manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
+
+for profile in PROFILES:
+    if profile == "source-36-f32-restored":
+        with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+            for name in f32_names:
+                parameter = model.get_parameter(name)
+                parameter.data = source_handle.get_tensor(name).clone()
+                assert parameter.dtype == torch.float32
+        print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
+    profile_dir = OUTPUT / profile
+    profile_dir.mkdir()
+    audit = parameter_audit(model, source_file)
+    write_json(profile_dir / "parameter-audit.json", audit)
+    assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
+    dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
+    print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
+    profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
+    cache = None
+    with torch.inference_mode():
+        for step, token in enumerate(TOKENS):
+            intermediates = {}
+            calls = collections.Counter()
+            signatures = {}
+
+            def observe_calls(frame, event, arg):
+                if frame.f_code.co_filename != str(model_source):
+                    return
+                name = frame.f_code.co_name
+                if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                    caller = frame.f_back
+                    while caller is not None:
+                        owner = caller.f_locals.get("self")
+                        if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                            if owner.layer_idx == 0:
+                                remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
+                            break
+                        caller = caller.f_back
+                    return
+                if event != "call":
+                    return
+                if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
+                    key = name + ":" + str(frame.f_code.co_firstlineno)
+                    calls[key] += 1
+                    signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
+
+            step_start = time.time()
+            sys.setprofile(observe_calls)
+            try:
+                result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
+            finally:
+                sys.setprofile(None)
+            cache = result.past_key_values
+            expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
+            observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
+            assert observed_by_name == expected, (expected, observed_by_name)
+            step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
+            remember("final_logits", result.logits)
+            remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
+            assert len(intermediates) == 15
+            intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
+            save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
+            step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
+            step_result["observed_official_operator_calls"] = dict(calls)
+            step_result["last_observed_operator_tensor_arguments"] = signatures
+            step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
+            write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
+            profile_result["steps"].append(step_result)
+            artifact_bound()
+            print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
+    manifest["profiles"][profile] = profile_result
+    write_json(OUTPUT / "manifest.json", manifest)
+    del result, cache
+
+for handle in handles:
+    handle.remove()
+del model, embedding, language, layer0, intermediates
+gc.collect()
+assert verify_source() == source_receipt
+assert verify_runtime() == runtime_source
+manifest["artifact_bytes_before_final_manifest"] = artifact_bound()
+manifest["cgroup_memory_peak_bytes"] = pathlib.Path("/sys/fs/cgroup/memory.peak").read_text().strip()
+manifest["source_reverified_after_capture"] = True
+manifest["elapsed_seconds_not_benchmark"] = time.time() - started
+manifest["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+write_json(OUTPUT / "manifest.json", manifest)
+artifact_bound()
+print("REFERENCE_CAPTURE_COMPLETE", json.dumps({"profiles": list(manifest["profiles"]), "steps_each": 2, "full_vocab": 248320, "semantic_domains_each": 42, "max_rss_kib": manifest["max_rss_kib"]}), flush=True)
--- /dev/null
+++ baseline/source-pins.json
@@ -0,0 +1,226 @@
+{
+  "schema": "zerogravity.hf2b-reference-pins.v1",
+  "repo": "Qwen/Qwen3.5-2B",
+  "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
+  "snapshot": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/files",
+  "source_files": [
+    {
+      "path": "LICENSE",
+      "bytes": 11544,
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "README.md",
+      "bytes": 62814,
+      "sha256": "c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "chat_template.jinja",
+      "bytes": 7755,
+      "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "config.json",
+      "bytes": 2908,
+      "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "merges.txt",
+      "bytes": 3353259,
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "model.safetensors-00001-of-00001.safetensors",
+      "bytes": 4548221488,
+      "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "model.safetensors.index.json",
+      "bytes": 64460,
+      "sha256": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "preprocessor_config.json",
+      "bytes": 390,
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "tokenizer.json",
+      "bytes": 12807982,
+      "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "tokenizer_config.json",
+      "bytes": 16709,
+      "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "video_preprocessor_config.json",
+      "bytes": 385,
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "vocab.json",
+      "bytes": 6722759,
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    }
+  ],
+  "raw_f32_names": [
+    "model.language_model.layers.0.linear_attn.A_log",
+    "model.language_model.layers.0.linear_attn.norm.weight",
+    "model.language_model.layers.1.linear_attn.A_log",
+    "model.language_model.layers.1.linear_attn.norm.weight",
+    "model.language_model.layers.10.linear_attn.A_log",
+    "model.language_model.layers.10.linear_attn.norm.weight",
+    "model.language_model.layers.12.linear_attn.A_log",
+    "model.language_model.layers.12.linear_attn.norm.weight",
+    "model.language_model.layers.13.linear_attn.A_log",
+    "model.language_model.layers.13.linear_attn.norm.weight",
+    "model.language_model.layers.14.linear_attn.A_log",
+    "model.language_model.layers.14.linear_attn.norm.weight",
+    "model.language_model.layers.16.linear_attn.A_log",
+    "model.language_model.layers.16.linear_attn.norm.weight",
+    "model.language_model.layers.17.linear_attn.A_log",
+    "model.language_model.layers.17.linear_attn.norm.weight",
+    "model.language_model.layers.18.linear_attn.A_log",
+    "model.language_model.layers.18.linear_attn.norm.weight",
+    "model.language_model.layers.2.linear_attn.A_log",
+    "model.language_model.layers.2.linear_attn.norm.weight",
+    "model.language_model.layers.20.linear_attn.A_log",
+    "model.language_model.layers.20.linear_attn.norm.weight",
+    "model.language_model.layers.21.linear_attn.A_log",
+    "model.language_model.layers.21.linear_attn.norm.weight",
+    "model.language_model.layers.22.linear_attn.A_log",
+    "model.language_model.layers.22.linear_attn.norm.weight",
+    "model.language_model.layers.4.linear_attn.A_log",
+    "model.language_model.layers.4.linear_attn.norm.weight",
+    "model.language_model.layers.5.linear_attn.A_log",
+    "model.language_model.layers.5.linear_attn.norm.weight",
+    "model.language_model.layers.6.linear_attn.A_log",
+    "model.language_model.layers.6.linear_attn.norm.weight",
+    "model.language_model.layers.8.linear_attn.A_log",
+    "model.language_model.layers.8.linear_attn.norm.weight",
+    "model.language_model.layers.9.linear_attn.A_log",
+    "model.language_model.layers.9.linear_attn.norm.weight"
+  ],
+  "tensor_counts": {
+    "F32": 36,
+    "BF16": 596
+  },
+  "dtype_elements": {
+    "F32": 2592,
+    "BF16": 2274067232
+  },
+  "root_source_validation": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-source-plan-20261020/root-post-acquisition-validation.json",
+    "bytes": 5496,
+    "sha256": "57929dd60414c1f2b4df536c05f39bcbef9957221ae45ae42f7982d6d1666ccb"
+  },
+  "cache_manifest": {
+    "path": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/manifest.json",
+    "bytes": 2246,
+    "sha256": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52"
+  },
+  "base_capture": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/capture_reference.py",
+    "bytes": 18717,
+    "sha256": "225d8637f927f8bc6ce99af9d9dda6a6a5cd33fc9d7d24042dda7b98297f335d"
+  },
+  "base_validator": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/validate_reference.py",
+    "bytes": 5573,
+    "sha256": "5ef1077ed6abdb37093701cb2846fc044a35f62eb9d883891876c5e98f0ced09"
+  },
+  "runtime_source_manifest": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/runtime-source-manifest.json",
+    "bytes": 1942,
+    "sha256": "24eabe68b10c2ac6a17e2dcdcd53805abe47556a98b8b6a4f2f1c27c60e6c673"
+  },
+  "runtime_source": {
+    "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
+    "versions": {
+      "transformers": "5.17.0",
+      "torch": "2.13.0+cu129",
+      "safetensors": "0.8.0",
+      "vllm": "0.31.0+cu129",
+      "flash-linear-attention": null,
+      "fla-core": null,
+      "triton": "3.7.1"
+    },
+    "files": {
+      "transformers/models/qwen3_5/modeling_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
+        "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
+        "bytes": 95685,
+        "artifact": "pinned-runtime-source/modeling_qwen3_5.py"
+      },
+      "transformers/models/qwen3_5/configuration_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/configuration_qwen3_5.py",
+        "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da",
+        "bytes": 8395,
+        "artifact": "pinned-runtime-source/configuration_qwen3_5.py"
+      },
+      "transformers/cache_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py",
+        "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c",
+        "bytes": 104677,
+        "artifact": "pinned-runtime-source/cache_utils.py"
+      },
+      "transformers/modeling_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py",
+        "sha256": "38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d",
+        "bytes": 268806,
+        "artifact": "pinned-runtime-source/modeling_utils.py"
+      },
+      "transformers/integrations/hub_kernels.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/integrations/hub_kernels.py",
+        "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95",
+        "bytes": 44974,
+        "artifact": "pinned-runtime-source/hub_kernels.py"
+      }
+    }
+  },
+  "derived_image_receipt": {
+    "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/torchcodec-cpu/derived-image-receipt.json",
+    "bytes": 7655,
+    "sha256": "c53c8421af4c832b901c00f4aabb65c7a1901b92c0655af188fa886fdaac63c0"
+  },
+  "derived_image_id": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "base_image": "docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f",
+  "package_records": {
+    "torch": {
+      "record_sha256": "4ae8e32005981a22c51a85130862a0ca173a01441a9de439b405a380c8dba243",
+      "version": "2.13.0+cu129"
+    },
+    "transformers": {
+      "record_sha256": "02b3d38575d5dc45992f859592fe143a86b04046a7c8da1b6ebc82390a90ed26",
+      "version": "5.17.0"
+    },
+    "safetensors": {
+      "record_sha256": "c5110881c0c39a4f163a23b2ccc88cad19ce1823b6dce1ceef353d33f16c004b",
+      "version": "0.8.0"
+    },
+    "vllm": {
+      "record_sha256": "bc16d6292493309ae02ca009dd57d2862e63880357a64a40be9dc6f0d6513895",
+      "version": "0.31.0+cu129"
+    },
+    "torchcodec": {
+      "record_sha256": "506253df6850b5d0af73d56c64c1e2d34c6f92cd3794a1ad1330050674855285",
+      "version": "0.17.0+cpu"
+    }
+  },
+  "source_execution_qualified": false
+}
--- /dev/null
+++ baseline/validate_reference.py
@@ -0,0 +1,135 @@
+"""Validate archive closure and lossless layout transforms; no model execution."""
+
+import collections
+import hashlib
+import json
+import pathlib
+
+import numpy as np
+import torch
+from safetensors.torch import load_file
+
+ROOT = pathlib.Path("/output/captured-reference")
+manifest = json.loads((ROOT / "manifest.json").read_text())
+runtime_source = manifest["runtime_source"]
+pins = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert manifest["source_pins_sha256"] == hashlib.sha256(pathlib.Path("/probe/source-pins.json").read_bytes()).hexdigest()
+for pin in runtime_source["files"].values():
+    for path in [pathlib.Path(pin["runtime_path"])]:
+        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], path
+assert manifest["official_modeling_source"]["sha256"] == runtime_source["files"]["transformers/models/qwen3_5/modeling_qwen3_5.py"]["sha256"]
+assert manifest["source_reverified_after_capture"]
+assert manifest["limits"]["container_memory_max"] == str(12 * 1024**3)
+assert manifest["limits"]["container_cpu_max"] == "400000 100000"
+assert not manifest["runtime"]["cuda_available"]
+assert manifest["runtime"]["torch_num_threads"] == 4
+profiles = ["official-bf16-loader", "source-36-f32-restored"]
+counts = {"profiles": 0, "step_archives": 0, "canonical_domains": 0, "full_vocab_logits": 0, "observed_intermediate_tensors": 0}
+
+
+def verified_file(row):
+    path = ROOT / row["file"]
+    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink() and path.is_file()
+    assert path.stat().st_size <= 32 * 1024**2
+    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
+    return path
+
+
+def verify_f32(row, expected):
+    actual = np.fromfile(verified_file(row), dtype="<f4")
+    expected = expected.detach().cpu().float().contiguous().numpy().reshape(-1)
+    # Compare bits, including zero sign; array_equal would hide that distinction.
+    assert np.array_equal(actual.view("<u4"), expected.view("<u4"))
+    assert actual.size == row["numel"] and np.isfinite(actual).all()
+
+
+for profile in profiles:
+    data = manifest["profiles"][profile]
+    audit_path = ROOT / profile / "parameter-audit.json"
+    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == data["parameter_audit_sha256"]
+    audit = json.loads(audit_path.read_text())
+    assert len(audit) == 632
+    assert all(row["exact_after_declared_dtype_cast"] for row in audit if row["loaded"] is not None)
+    excluded = [row for row in audit if row["loaded"] is None]
+    assert len(excluded) == 15 and all(row["name"].startswith("mtp.") for row in excluded)
+    original_f32 = [row for row in audit if row["source"]["dtype"] == "torch.float32"]
+    assert len(original_f32) == 36
+    assert [row["name"] for row in original_f32] == pins["raw_f32_names"]
+    assert dict(collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)) == data["actual_checkpoint_parameter_dtype_counts"]
+    assert len(data["steps"]) == 2
+    if profile == "source-36-f32-restored":
+        assert all(row["loaded"]["dtype"] == "torch.float32" for row in original_f32)
+    counts["profiles"] += 1
+    for step, saved in enumerate(data["steps"]):
+        raw = load_file(str(verified_file(saved["raw_archive"])), device="cpu")
+        assert len(raw) == 49
+        assert raw["logits"].shape == (1, 1, 248320)
+        assert raw["logits"].dtype == torch.bfloat16
+        assert saved["consumed_tokens"] == [11, 353][:step+1] and saved["step"] == step
+        assert saved["semantic_domain_count"] == 42 and len(saved["raw_tensor_layouts"]) == 49
+        for layer in range(24):
+            prefix = f"layer-{layer:02d}"
+            if layer % 4 == 3:
+                for key in ["keys", "values"]:
+                    assert raw[prefix + "." + key].shape == (1, 2, step+1, 256)
+                    assert raw[prefix + "." + key].dtype == torch.bfloat16
+            else:
+                assert raw[prefix + ".conv"].shape == (1, 6144, 4)
+                assert raw[prefix + ".conv"].dtype == torch.bfloat16
+                assert raw[prefix + ".recurrent"].shape == (1, 16, 128, 128)
+                assert raw[prefix + ".recurrent"].dtype == torch.float32
+        assert len(saved["semantic_domains"]) == 42
+        observed = load_file(str(verified_file(saved["observed_intermediates"])), device="cpu")
+        assert len(observed) == 15
+        assert torch.equal(observed["final_logits"].contiguous().view(torch.uint8), raw["logits"].contiguous().view(torch.uint8))
+        assert torch.equal(observed["layer-00.gdn_recurrent_fp32"].contiguous().view(torch.uint8), raw["layer-00.recurrent"].permute(0, 1, 3, 2).contiguous().view(torch.uint8))
+        assert all(bool(torch.isfinite(value).all()) for value in observed.values())
+        counts["observed_intermediate_tensors"] += len(observed)
+        verify_f32(saved["logits_f32"], raw["logits"])
+        counts["full_vocab_logits"] += 1
+        expected_domains = {(layer, kind) for layer in range(24) for kind in
+                            (["attention-kv-pair"] if layer % 4 == 3 else
+                             ["gdn-convolution-history", "gdn-recurrent"])}
+        seen = set()
+        for domain in saved["semantic_domains"]:
+            key = (domain["layer"], domain["kind"])
+            assert key in expected_domains and key not in seen
+            seen.add(key)
+            prefix = f"layer-{domain['layer']:02d}"
+            if domain["kind"] == "gdn-convolution-history":
+                expected = raw[prefix + ".conv"][..., -3:]
+            elif domain["kind"] == "gdn-recurrent":
+                expected = raw[prefix + ".recurrent"].permute(0, 1, 3, 2)
+            else:
+                key = raw[prefix + ".keys"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                val = raw[prefix + ".values"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                expected = torch.cat([key, val])
+            verify_f32(domain, expected)
+            counts["canonical_domains"] += 1
+        assert seen == expected_domains
+        assert all(bool(torch.isfinite(value).all()) for value in raw.values())
+        counts["step_archives"] += 1
+
+differences = []
+for step in range(2):
+    left_data = manifest["profiles"][profiles[0]]["steps"][step]
+    right_data = manifest["profiles"][profiles[1]]["steps"][step]
+    left = load_file(str(ROOT / left_data["raw_archive"]["file"]))
+    right = load_file(str(ROOT / right_data["raw_archive"]["file"]))
+    rows = []
+    for name in sorted(left):
+        a, b = left[name].float().contiguous().numpy(), right[name].float().contiguous().numpy()
+        changed = int(np.count_nonzero(a.view("<u4") != b.view("<u4")))
+        rows.append({"tensor": name, "elements": int(a.size), "bit_different_elements": changed, "max_abs_delta": float(np.abs(a - b).max())})
+    differences.append({"step": step, "tensors": rows})
+result = {"archive_validation": "PASS", "counts": counts, "runtime_source_files_verified": len(runtime_source["files"]), "model_or_math_parity_qualified": False, "source_revision": pins["revision"], "cross_profile_differences": differences, "scope": "archive closure, full finite tensors and exact layout derivations only; numerical profile differences are reported without a tolerance or cross-engine acceptance claim"}
+total_artifacts = 0
+for path in ROOT.rglob("*"):
+    assert not path.is_symlink()
+    if path.is_file():
+        total_artifacts += path.stat().st_size
+assert total_artifacts <= 256 * 1024**2
+assert 0 < int(manifest["cgroup_memory_peak_bytes"]) <= 12 * 1024**3
+assert counts == {"profiles": 2, "step_archives": 4, "canonical_domains": 168, "full_vocab_logits": 4, "observed_intermediate_tensors": 60}
+(ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
+print(json.dumps({"archive_validation": "PASS", "counts": counts}, sort_keys=True))
--- /dev/null
+++ capture_and_validate.py
@@ -0,0 +1,7 @@
+"""ROOT ONLY: exact accepted capture integration then unchanged archive checks."""
+import gc
+import runpy
+
+runpy.run_path("/probe/capture_reference.py", run_name="__main__")
+gc.collect()
+runpy.run_path("/probe/validate_reference.py", run_name="__main__")
--- /dev/null
+++ capture_reference.py
@@ -0,0 +1,420 @@
+"""Pinned, offline CPU reference; operators remain the installed official source."""
+
+import collections
+import gc
+import signal
+import stat
+import hashlib
+import importlib.metadata
+import inspect
+import json
+import os
+import pathlib
+import platform
+import resource
+import sys
+import time
+
+# Pin the four allowed CPUs before importing the original Torch runtime.
+allowed = [int(value) for value in os.environ["ZG_PROBE_ALLOWED_CPUS"].split(",")]
+assert len(allowed) == len(set(allowed)) == 4 and set(allowed) <= os.sched_getaffinity(0)
+os.sched_setaffinity(0, allowed)
+MODE = os.environ["ZG_ATTENTION_MODE"]
+assert MODE in ("plain", "observed")
+
+import torch
+from safetensors import safe_open
+from safetensors.torch import save_file
+from transformers import AutoTokenizer
+from transformers.models.qwen3_5 import modeling_qwen3_5 as official
+
+SOURCE = pathlib.Path("/source")
+PINS = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert PINS["repo"] == "Qwen/Qwen3.5-2B" and PINS["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc"
+AUDIT_ELEMENTS = 262144  # ≤1MiB per F32 audit operand; model operators untouched.
+ARTIFACT_CAP = 256 * 1024 * 1024
+signal.alarm(290)  # Outer owner enforces300s and always cleans its bound CID.
+resource.setrlimit(resource.RLIMIT_FSIZE, (ARTIFACT_CAP, ARTIFACT_CAP))
+assert pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip() == str(12 * 1024**3)
+assert pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip() == "400000 100000"
+assert os.environ["ATEN_CPU_CAPABILITY"] == "avx2"
+OUTPUT = pathlib.Path("/output/captured-reference")
+TOKENS = [11, 353]
+PROFILES = ["official-bf16-loader", "source-36-f32-restored"]
+torch.set_num_threads(4)
+torch.set_num_interop_threads(1)
+assert not torch.cuda.is_available()
+assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
+assert not OUTPUT.exists(), "refuse to overwrite an earlier reference"
+OUTPUT.mkdir()
+
+
+def sha(path):
+    digest = hashlib.sha256()
+    with open(path, "rb") as stream:
+        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(chunk)
+    return digest.hexdigest()
+
+
+def write_json(path, data):
+    def official_json_types(value):
+        if isinstance(value, set):
+            return sorted(value)
+        raise TypeError(f"unhandled JSON type {type(value).__name__}")
+
+    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False, default=official_json_types) + "\n")
+
+
+def verify_source():
+    for item in PINS["source_files"]:
+        path = SOURCE / item["path"]
+        assert not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"]
+        assert sha(path) == item["sha256"], item["path"]
+    return {"repo": PINS["repo"], "revision": PINS["revision"], "snapshot": PINS["snapshot"],
+            "files": PINS["source_files"], "independent_verification": PINS["root_source_validation"]}
+
+
+def audit_chunks(value):
+    assert value.device.type == "cpu" and value.is_contiguous()
+    flat = value.detach().view(-1)
+    for start in range(0, flat.numel(), AUDIT_ELEMENTS):
+        yield flat[start:start + AUDIT_ELEMENTS]
+
+
+def finite(value):
+    return all(bool(torch.isfinite(part).all()) for part in audit_chunks(value.detach().contiguous()))
+
+
+def artifact_bound():
+    total = 0
+    for path in OUTPUT.rglob("*"):
+        assert not path.is_symlink()
+        if path.is_file():
+            total += path.stat().st_size
+    assert total <= ARTIFACT_CAP, ("aggregate capture cap", total)
+    return total
+
+
+def verify_runtime():
+    expected = PINS["runtime_source"]
+    for part in expected["files"].values():
+        path = pathlib.Path(part["runtime_path"])
+        assert sha(path) == part["sha256"], str(path)
+    for name, pin in PINS["package_records"].items():
+        distribution = importlib.metadata.distribution(name)
+        assert distribution.version == pin["version"], name
+        record = distribution.read_text("RECORD")
+        assert record is not None and hashlib.sha256(record.encode()).hexdigest() == pin["record_sha256"], name
+    assert torch.__version__ == "2.13.0+cu129" and importlib.metadata.version("transformers") == "5.17.0"
+    assert torch.backends.cpu.get_cpu_capability() == "AVX2"
+    return expected
+
+
+def tensor_info(value):
+    return {
+        "dtype": str(value.dtype), "shape": list(value.shape),
+        "stride": list(value.stride()), "numel": value.numel(),
+        "bytes": value.numel() * value.element_size(),
+        "device": str(value.device), "contiguous": value.is_contiguous(),
+        "finite": finite(value),
+    }
+
+
+def f32_file(directory, name, value):
+    path = directory / (name + ".f32le.bin")
+    wide = value.detach().cpu().float().contiguous()
+    assert finite(wide)
+    # numpy explicitly requests little-endian; widening BF16 is exact.
+    path.write_bytes(wide.numpy().astype("<f4", copy=False).tobytes())
+    return {"file": str(path.relative_to(OUTPUT)), "sha256": sha(path), **tensor_info(wide)}
+
+
+def parameter_audit(model, source_file):
+    rows = []
+    with safe_open(source_file, framework="pt", device="cpu") as handle:
+        for name in sorted(handle.keys()):
+            original = handle.get_tensor(name)
+            try:
+                actual = model.get_parameter(name)
+            except AttributeError:
+                assert name.startswith("mtp."), name
+                rows.append({"name": name, "source": tensor_info(original), "loaded": None, "exact_after_declared_dtype_cast": None, "exclusion": "official Qwen3_5PreTrainedModel._keys_to_ignore_on_load_unexpected=['^mtp.*']; auxiliary MTP is absent from the ordinary conditional-generation class"})
+                continue
+            assert actual.device.type == "cpu"
+            assert actual.shape == original.shape
+            assert original.is_contiguous() and actual.is_contiguous()
+            cast_changed, cast_max = 0, 0.0
+            for left, right in zip(audit_chunks(original), audit_chunks(actual)):
+                cast = left.to(right.dtype)
+                assert torch.equal(cast.view(torch.uint8), right.view(torch.uint8)), name
+                if original.dtype != actual.dtype:
+                    delta = left.float() - right.float()
+                    cast_changed += int((delta != 0).sum())
+                    cast_max = max(cast_max, float(delta.abs().max()))
+            row = {"name": name, "source": tensor_info(original), "loaded": tensor_info(actual)}
+            row["exact_after_declared_dtype_cast"] = True
+            if original.dtype != actual.dtype:
+                row["cast_changed_elements"] = cast_changed
+                row["cast_max_abs_delta"] = cast_max
+            rows.append(row)
+    return rows
+
+
+def snapshot(cache, logits, directory, profile, step, token):
+    directory.mkdir()
+    raw = {"logits": logits.detach().cpu().contiguous().clone()}
+    raw_meta = {"logits": tensor_info(logits)}
+    domains = []
+    assert len(cache.layers) == 24
+    assert logits.shape == (1, 1, 248320)
+    for index, layer in enumerate(cache.layers):
+        prefix = f"layer-{index:02d}"
+        if hasattr(layer, "conv_states"):
+            assert set(layer.conv_states) == {0} and set(layer.recurrent_states) == {0}
+            conv, recurrent = layer.conv_states[0], layer.recurrent_states[0]
+            assert conv.shape == (1, 6144, 4)
+            assert recurrent.shape == (1, 16, 128, 128)
+            assert recurrent.dtype == torch.float32 and conv.dtype == torch.bfloat16
+            for kind, value in [("conv", conv), ("recurrent", recurrent)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            # Official cache retains four raw projection positions. The next
+            # four-tap convolution needs the newest three as prior history.
+            domains.append({"layer": index, "kind": "gdn-convolution-history", "layout": "batch,channel,oldest-to-newest-3", "derivation": "official conv_states[0][..., -3:]", **f32_file(directory, prefix + ".conv-history", conv[..., -3:])})
+            # Source state is key-major; the Rust state reader is value-major.
+            domains.append({"layer": index, "kind": "gdn-recurrent", "layout": "batch,head,value,key", "derivation": "official recurrent_states[0].permute(0,1,3,2)", **f32_file(directory, prefix + ".recurrent-value-major", recurrent.permute(0, 1, 3, 2))})
+        else:
+            keys, values = layer.keys, layer.values
+            assert keys.shape == values.shape == (1, 2, step + 1, 256)
+            assert keys.dtype == values.dtype == torch.bfloat16
+            for kind, value in [("keys", keys), ("values", values)]:
+                raw[prefix + "." + kind] = value.detach().cpu().contiguous().clone()
+                raw_meta[prefix + "." + kind] = tensor_info(value)
+            keys = keys.permute(0, 2, 1, 3).contiguous()
+            values = values.permute(0, 2, 1, 3).contiguous()
+            pair = torch.cat([keys.reshape(-1), values.reshape(-1)])
+            domains.append({"layer": index, "kind": "attention-kv-pair", "layout": "all keys(batch,position,kv-head,width), then all values(same)", "valid_positions": step + 1, "derivation": "official keys/values permute(0,2,1,3), flatten and concatenate", **f32_file(directory, prefix + ".kv-pair", pair)})
+    assert len(domains) == 42
+    assert len(raw) == 49  # logits + 36 GDN tensors + 12 K/V tensors
+    assert all(info["finite"] for info in raw_meta.values())
+    raw_path = directory / "raw-official.safetensors"
+    save_file(raw, str(raw_path), metadata={"profile": profile, "step": str(step), "input_token": str(token), "source_revision": PINS["revision"]})
+    wide_logits = f32_file(directory, "logits", logits)
+    flat = logits.float().reshape(-1)
+    top_values, top_ids = torch.topk(flat, 10, largest=True, sorted=True)
+    result = {"profile": profile, "step": step, "input_token": token, "consumed_tokens": TOKENS[:step + 1], "cache_class": type(cache).__name__, "cache_seq_length": cache.get_seq_length(), "raw_archive": {"file": str(raw_path.relative_to(OUTPUT)), "sha256": sha(raw_path), "bytes": raw_path.stat().st_size}, "raw_tensor_layouts": raw_meta, "logits_f32": wide_logits, "argmax_token_id": int(flat.argmax()), "top10": [{"token_id": int(i), "logit": float(v)} for i, v in zip(top_ids, top_values)], "semantic_domains": domains, "semantic_domain_count": len(domains), "physical_ready_or_StateRoot_claim": False}
+    write_json(directory / "manifest.json", result)
+    return result
+
+
+started = time.time()
+runtime_source = verify_runtime()
+source_receipt = verify_source()
+write_json(OUTPUT / "source-receipt.json", source_receipt)
+source_file = SOURCE / "model.safetensors-00001-of-00001.safetensors"
+with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+    names = sorted(source_handle.keys())
+    f32_names = [name for name in names if source_handle.get_slice(name).get_dtype() == "F32"]
+assert len(names) == 632 and len(f32_names) == 36
+assert f32_names == PINS["raw_f32_names"]
+assert all(".linear_attn.A_log" in name or ".linear_attn.norm.weight" in name for name in f32_names)
+assert "lm_head.weight" not in names
+print("SOURCE_VERIFIED tensors=632 original_f32=36 no_lm_head=1", flush=True)
+
+model, loading_info = official.Qwen3_5ForConditionalGeneration.from_pretrained(
+    str(SOURCE), dtype=torch.bfloat16, local_files_only=True,
+    attn_implementation="eager", use_kernels=False, output_loading_info=True,
+)
+model.eval()
+assert not loading_info.get("missing_keys") and not loading_info.get("unexpected_keys") and not loading_info.get("mismatched_keys") and not loading_info.get("error_msgs"), loading_info
+assert all(parameter.device.type == "cpu" for parameter in model.parameters())
+embedding = model.model.language_model.embed_tokens.weight
+assert model.lm_head.weight is embedding
+assert embedding.shape == (248320, 2048)
+assert model.config.text_config.hidden_size == 2048 and model.config.text_config.num_hidden_layers == 24
+tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
+print("OFFICIAL_MODEL_LOADED tied_head=1 cuda=0", flush=True)
+
+# Observation hooks copy actual outputs, return None and never replace an
+# operator/input/output. The down-projection input exposes the actual fused
+# SiLU(gate)*up tensor without reconstructing its arithmetic.
+intermediates = {}
+handles = []
+
+
+def remember(name, value):
+    assert name not in intermediates, name
+    assert isinstance(value, torch.Tensor)
+    intermediates[name] = value.detach().cpu().contiguous().clone()
+
+
+def output_hook(module, name):
+    handles.append(module.register_forward_hook(lambda _module, _args, value: remember(name, value)))
+
+
+def input_hook(module, name):
+    handles.append(module.register_forward_pre_hook(lambda _module, args: remember(name, args[0])))
+
+
+from attention_observation_hook import AttentionProjectionObserver
+attention = None
+try:
+    language = model.model.language_model
+    layer0 = language.layers[0]
+    output_hook(language.embed_tokens, "embedding")
+    output_hook(layer0.input_layernorm, "layer-00.attention_input_norm")
+    output_hook(layer0.linear_attn.in_proj_qkv, "layer-00.gdn_qkv")
+    output_hook(layer0.linear_attn, "layer-00.attention_output")
+    input_hook(layer0.post_attention_layernorm, "layer-00.post_attention_residual")
+    output_hook(layer0.post_attention_layernorm, "layer-00.ffn_input_norm")
+    output_hook(layer0.mlp.gate_proj, "layer-00.ffn_gate")
+    output_hook(layer0.mlp.up_proj, "layer-00.ffn_up")
+    input_hook(layer0.mlp.down_proj, "layer-00.ffn_fused")
+    output_hook(layer0.mlp.down_proj, "layer-00.ffn_output")
+    output_hook(layer0, "layer-00.post_ffn_residual")
+    output_hook(language.norm, "final_norm")
+
+    manifest = {
+        "schema": "zerogravity.official-cpu-qwen35-reference.v2-hf2b",
+        "source_pins_sha256": sha(pathlib.Path("/probe/source-pins.json")),
+        "source_repo": source_receipt["repo"], "source_revision": source_receipt["revision"],
+        "image": PINS["derived_image_id"], "base_image": PINS["base_image"],
+        "runtime_source": runtime_source, "package_records": PINS["package_records"],
+        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": importlib.metadata.version("transformers"), "safetensors": importlib.metadata.version("safetensors"), "vllm": importlib.metadata.version("vllm"), "torch_num_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(), "cuda_available": torch.cuda.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled, "default_dtype": str(torch.get_default_dtype()), "cpu_capability": torch.backends.cpu.get_cpu_capability(), "machine": platform.machine(), "processor": platform.processor()},
+        "loader": {"model_class": type(model).__name__, "dtype": "torch.bfloat16", "attn_implementation": "eager", "use_kernels": False, "local_files_only": True, "loading_info": loading_info},
+        "config": model.config.to_dict(), "tokenizer": {"class": type(tokenizer).__name__, "vocab_size": len(tokenizer), "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id, "tokens": TOKENS, "token_strings": tokenizer.convert_ids_to_tokens(TOKENS), "decoded": tokenizer.decode(TOKENS, skip_special_tokens=False)},
+        "head": {"tied_parameter_object": True, "source_has_lm_head_tensor": False, "embedding_name": "model.language_model.embed_tokens.weight", "output_dtype_rule": "official lm_head(hidden); no loss, no automatic FP32 upcast"},
+        "protocol": "one cached official forward per fixed input token, no sampling or generation; first token uses chunk reference, next token recurrent reference",
+        "limits": {"container_memory_max": pathlib.Path("/sys/fs/cgroup/memory.max").read_text().strip(), "container_cpu_max": pathlib.Path("/sys/fs/cgroup/cpu.max").read_text().strip(), "network": "none", "source_mount": "read-only", "gpu_exposed": False},
+        "scope": "Official Transformers eager CPU attention and slow PyTorch GDN operators only. No GPU FLA, vLLM inference, throughput, quantization, Rust bit parity, or trained27B qualification.",
+        "profiles": {},
+        "source_geometry": {"hidden": 2048, "ffn": 6144, "layers": 24, "gdn_layers": 18, "full_attention_layers": 6, "gdn_heads": 16, "gdn_head_width": 128, "kv_heads": 2, "kv_head_width": 256},
+    }
+    model_source = pathlib.Path(inspect.getsourcefile(official))
+    manifest["official_modeling_source"] = {"path": str(model_source), "sha256": sha(model_source)}
+    manifest["capture_script_sha256"] = sha(pathlib.Path(__file__))
+
+    if MODE == "observed":
+        attention = AttentionProjectionObserver(model, official)
+    for profile in PROFILES:
+        if profile == "source-36-f32-restored":
+            with safe_open(source_file, framework="pt", device="cpu") as source_handle:
+                for name in f32_names:
+                    parameter = model.get_parameter(name)
+                    parameter.data = source_handle.get_tensor(name).clone()
+                    assert parameter.dtype == torch.float32
+            print("RESTORED_ORIGINAL_F32_PARAMETERS count=36", flush=True)
+        profile_dir = OUTPUT / profile
+        profile_dir.mkdir()
+        audit = parameter_audit(model, source_file)
+        write_json(profile_dir / "parameter-audit.json", audit)
+        assert len(audit) == 632 and sum(row["loaded"] is None for row in audit) == 15
+        dtype_counts = collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)
+        print("PROFILE_AUDIT", profile, dict(dtype_counts), flush=True)
+        profile_result = {"profile": profile, "parameter_audit_sha256": sha(profile_dir / "parameter-audit.json"), "source_parameter_count": len(audit), "actual_checkpoint_parameter_dtype_counts": dict(dtype_counts), "source_f32_parameter_names": f32_names, "restoration": "none; unmodified official requested-BF16 loader" if profile == PROFILES[0] else "only the 36 originally F32 checkpoint parameters restored to their exact source dtype and bytes; official operators unchanged", "steps": []}
+        cache = None
+        with torch.inference_mode():
+            for step, token in enumerate(TOKENS):
+                intermediates = {}
+                calls = collections.Counter()
+                signatures = {}
+
+                def observe_calls(frame, event, arg):
+                    if attention is not None:
+                        attention.observe_source_return(frame, event, arg)
+                    if frame.f_code.co_filename != str(model_source):
+                        return
+                    name = frame.f_code.co_name
+                    if event == "return" and name in {"causal_conv1d_fn", "causal_conv1d_update"}:
+                        caller = frame.f_back
+                        while caller is not None:
+                            owner = caller.f_locals.get("self")
+                            if isinstance(owner, official.Qwen3_5GatedDeltaNet):
+                                if owner.layer_idx == 0:
+                                    remember("layer-00.gdn_conv_silu", arg.transpose(1, 2))
+                                break
+                            caller = caller.f_back
+                        return
+                    if event != "call":
+                        return
+                    if name in {"torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update", "eager_attention_forward"}:
+                        key = name + ":" + str(frame.f_code.co_firstlineno)
+                        calls[key] += 1
+                        signatures[key] = {key: tensor_info(value) for key, value in frame.f_locals.items() if isinstance(value, torch.Tensor)}
+
+                if attention is not None:
+                    attention.begin_step(profile, step, token)
+                step_start = time.time()
+                sys.setprofile(observe_calls)
+                try:
+                    result = model(input_ids=torch.tensor([[token]], dtype=torch.int64), past_key_values=cache, use_cache=True, return_dict=True)
+                finally:
+                    sys.setprofile(None)
+                cache = result.past_key_values
+                expected = {("torch_chunk_gated_delta_rule" if step == 0 else "torch_recurrent_gated_delta_rule"): 18, ("causal_conv1d_fn" if step == 0 else "causal_conv1d_update"): 18, "eager_attention_forward": 6}
+                observed_by_name = {key.split(":")[0]: value for key, value in calls.items()}
+                assert observed_by_name == expected, (expected, observed_by_name)
+                step_result = snapshot(cache, result.logits, profile_dir / f"step-{step}", profile, step, token)
+                if attention is not None:
+                    attention.finish_step(profile_dir / f"step-{step}" / "attention-observation")
+                remember("final_logits", result.logits)
+                remember("layer-00.gdn_recurrent_fp32", cache.layers[0].recurrent_states[0].permute(0, 1, 3, 2))
+                assert len(intermediates) == 15
+                intermediate_path = profile_dir / f"step-{step}" / "observed-intermediates.safetensors"
+                save_file(intermediates, str(intermediate_path), metadata={"profile": profile, "step": str(step), "scope": "actual layer0/module input/output copies; no reconstructed arithmetic"})
+                step_result["observed_intermediates"] = {"file": str(intermediate_path.relative_to(OUTPUT)), "sha256": sha(intermediate_path), "bytes": intermediate_path.stat().st_size, "tensors": {name: tensor_info(value) for name, value in intermediates.items()}, "hook_scope": "official module pre/post hooks, CPU tensor clone only; slow causal-conv source return observed with sys.setprofile; no replacement return"}
+                step_result["observed_official_operator_calls"] = dict(calls)
+                step_result["last_observed_operator_tensor_arguments"] = signatures
+                step_result["diagnostic_elapsed_seconds_not_benchmark"] = time.time() - step_start
+                write_json(profile_dir / f"step-{step}" / "manifest.json", step_result)
+                profile_result["steps"].append(step_result)
+                artifact_bound()
+                print("REFERENCE_STEP_COMPLETE", profile, "step", step, "argmax", step_result["argmax_token_id"], "domains", len(step_result["semantic_domains"]), "seconds", step_result["diagnostic_elapsed_seconds_not_benchmark"], flush=True)
+        manifest["profiles"][profile] = profile_result
+        write_json(OUTPUT / "manifest.json", manifest)
+        del result, cache
+
+finally:
+    # Every installed hook is removed even after forward/capture/receipt failure.
+    primary = sys.exception()
+    removal = []
+    try:
+        sys.setprofile(None)
+    except BaseException as error:
+        removal.append(error)
+    if attention is not None:
+        try:
+            attention.close()
+        except BaseException as error:
+            removal.append(error)
+    attention = None  # Release observed module owners before archive validation.
+    for handle in reversed(handles):
+        try:
+            handle.remove()
+        except BaseException as error:
+            removal.append(error)
+    handles.clear()
+    if removal:
+        raise BaseExceptionGroup("capture and all hook removal failures",
+                                 ([primary] if primary is not None else []) + removal) from None
+del model, embedding, language, layer0, intermediates
+gc.collect()
+assert verify_source() == source_receipt
+assert verify_runtime() == runtime_source
+manifest["artifact_bytes_before_final_manifest"] = artifact_bound()
+manifest["cgroup_memory_peak_bytes"] = pathlib.Path("/sys/fs/cgroup/memory.peak").read_text().strip()
+manifest["source_reverified_after_capture"] = True
+manifest["elapsed_seconds_not_benchmark"] = time.time() - started
+manifest["max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
+write_json(pathlib.Path("/output/attention-mode.json"), {
+    "mode": MODE, "owned_hooks_removed": True, "profiler_retired": sys.getprofile() is None,
+    "cpu_affinity": sorted(os.sched_getaffinity(0)), "teacher_tokens": TOKENS,
+    "profiles": PROFILES, "observer_hook_sha256": sha(pathlib.Path("/probe/attention_observation_hook.py")),
+    "accepted_capture_sha256": "6ee140185766dee237ea4df802504ef7b7be80110f5566876a644414cf66049d",
+    "expected_attention_tensor_count": 216 if MODE == "observed" else 0,
+    "full_model_or_Rust_or_GPU_parity_qualified": False,
+})
+write_json(OUTPUT / "manifest.json", manifest)
+artifact_bound()
+print("REFERENCE_CAPTURE_COMPLETE", json.dumps({"profiles": list(manifest["profiles"]), "steps_each": 2, "full_vocab": 248320, "semantic_domains_each": 42, "max_rss_kib": manifest["max_rss_kib"]}), flush=True)
--- /dev/null
+++ owned_container_v4.py
@@ -0,0 +1,280 @@
+"""ROOT ONLY finite owned lifetime. No execution occurs by importing this file.
+
+Additive v4: original nonce/CID binding and commands preserved. WNOWAIT and
+selector cleanup follow the reviewed AV runner, with whole-group KILL strictly
+before leader reap: its reserved PID prevents accidental process-group reuse.
+"""
+import hashlib
+import json
+import os
+from pathlib import Path
+import re
+import selectors
+import signal
+import subprocess
+import time
+
+PODMAN = ["podman", "--root", "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/containers", "--runroot", "/run/user/1000/abird/zerogravity/vllm-0.31.0-cu129/containers"]
+LABEL = "abird.zerogravity.vllm-gpu-oracle-owner"
+CAP = 4 * 1024 * 1024
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    with Path(path).open("rb") as file:
+        return hashlib.file_digest(file, "sha256").hexdigest()
+
+
+def command(argv, timeout=10):
+    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                            stderr=subprocess.PIPE, timeout=timeout, check=False)
+    require(len(result.stdout) + len(result.stderr) <= 1024 * 1024, "bounded command receipt")
+    return result
+
+
+def leader_status_unreaped(process):
+    require(process.returncode is None, "owned leader already reaped; never signal a recycled PGID")
+    observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
+    if observed is None:
+        return None
+    require(observed.si_pid == process.pid and observed.si_code in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED),
+            "owned leader exit observation")
+    return observed.si_status if observed.si_code == os.CLD_EXITED else -observed.si_status
+
+
+class Log:
+    """Main-thread bounded drain; no thread/future can survive cleanup."""
+    def __init__(self, pipe, path):
+        self.pipe, self.path = pipe, Path(path)
+        self.selector = self.file = None
+        self.retained = self.total = 0
+        self.eof = self.closed = False
+        self.errors = []
+        try:
+            self.file = self.path.open("xb")
+            os.chmod(self.path, 0o600)
+            os.set_blocking(pipe.fileno(), False)
+            self.selector = selectors.DefaultSelector()
+            self.selector.register(pipe, selectors.EVENT_READ)
+        except BaseException:
+            if self.selector is not None:
+                self.selector.close()
+            if self.file is not None:
+                self.file.close()
+            raise
+
+    def drain(self, timeout):
+        for key, _ in self.selector.select(timeout):
+            # One bounded block per wake; the exit/deadline remains observable
+            # even when an owned writer continuously produces output.
+            try:
+                raw = os.read(key.fd, 65536)
+            except BlockingIOError:
+                continue
+            if not raw:
+                self.selector.unregister(key.fileobj)
+                self.eof = True
+                continue
+            self.total += len(raw)
+            keep = raw[:max(0, CAP - self.retained)]
+            if keep:
+                try:
+                    self.file.write(keep)
+                    self.retained += len(keep)
+                except OSError as error:
+                    self.errors.append(repr(error))
+
+    def observe(self, process, seconds):
+        deadline = time.monotonic() + seconds
+        while time.monotonic() < deadline:
+            self.drain(0.05)
+            require(self.total <= CAP and not self.errors, "owned log failed or exceeded4MiB")
+            status = leader_status_unreaped(process)
+            if status is not None:
+                return status
+        raise TimeoutError("owned foreground exit deadline (leader remains unreaped)")
+
+    def finish(self):
+        failures = []
+        try:
+            deadline = time.monotonic() + 3
+            while not self.eof and time.monotonic() < deadline:
+                self.drain(0.05)
+            require(self.eof, "owned stdout pipe EOF deadline")
+            require(self.total <= CAP and not self.errors, "owned log evidence failed/truncated")
+            self.file.flush()
+            os.fsync(self.file.fileno())
+        except BaseException as error:
+            failures.append("drain/evidence: " + repr(error))
+        finally:
+            # Close all three even when EOF, evidence I/O or one close fails.
+            for name, close in (("selector", self.selector.close), ("log", self.file.close),
+                                ("pipe", self.pipe.close)):
+                try:
+                    close()
+                except BaseException as error:
+                    failures.append(name + ": " + repr(error))
+            self.closed = self.pipe.closed and self.file.closed and not self.selector.get_map()
+        if failures:
+            raise RuntimeError("owned log cleanup: " + "; ".join(failures))
+
+
+def owned_container(name, nonce, cidfile):
+    target, recorded_cid = name, None
+    if cidfile.exists():
+        cid = cidfile.read_text().strip()
+        if cid:
+            require(re.fullmatch(r"[a-f0-9]{64}", cid), "invalid owned CID; no container signal")
+            target = recorded_cid = cid
+    exists = command(PODMAN + ["container", "exists", target])
+    if exists.returncode == 1:
+        return None
+    require(exists.returncode == 0, "owned container existence check failed")
+    inspected = command(PODMAN + ["container", "inspect", target])
+    require(inspected.returncode == 0, "owned container inspect failed")
+    value, = json.loads(inspected.stdout)
+    require(value["Name"] == name and value["Config"].get("Labels", {}).get(LABEL) == nonce,
+            "container name/nonce mismatch; no signal")
+    cid = value["Id"]
+    require(re.fullmatch(r"[a-f0-9]{64}", cid), "inspected owned CID invalid")
+    if recorded_cid is not None:
+        require(recorded_cid == cid, "owned CID changed; no signal")
+    return cid
+
+
+def cleanup_container(name, nonce, cidfile):
+    errors = []
+    try:
+        cid = owned_container(name, nonce, cidfile)
+    except BaseException as error:
+        return ["container owner binding: " + repr(error)]
+    if cid is None:
+        return errors
+    try:
+        stopped = command(PODMAN + ["stop", "--time", "5", cid], 15)
+        if stopped.returncode != 0:
+            errors.append("owned stop returned nonzero")
+    except BaseException as error:
+        errors.append("container stop: " + repr(error))
+    try:
+        # Fresh exact owner binding is required even after a failed/timedout
+        # stop. A stop failure must not skip the bounded force-remove attempt.
+        if owned_container(name, nonce, cidfile) is not None:
+            removed = command(PODMAN + ["rm", "--force", cid], 15)
+            require(removed.returncode == 0, "owned container force cleanup failed")
+    except BaseException as error:
+        errors.append("container force cleanup: " + repr(error))
+    try:
+        require(owned_container(name, nonce, cidfile) is None, "owned container survived cleanup")
+    except BaseException as error:
+        errors.append("container final absence: " + repr(error))
+    return errors
+
+
+def group_exists(pgid):
+    try:
+        os.killpg(pgid, 0)
+        return True
+    except ProcessLookupError:
+        return False
+
+
+def signal_group_unreaped(process, sig):
+    # waitid authenticates that this is still our unreaped child. start_new_session
+    # made it the session/group leader; no foreign group is inferred from a PID.
+    leader_status_unreaped(process)
+    require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
+            "owned leader session/PGID binding")
+    os.killpg(process.pid, sig)
+
+
+def cleanup_group(process, receipt, *, observe=leader_status_unreaped, pause=time.sleep):
+    """Independent bounded stages; no nonzero signal after any reap attempt."""
+    state = receipt["foreground_cleanup"] = {"pgid": process.pid, "term_sent": False,
+        "kill_sent_before_reap": False, "leader_reaped": False, "group_absent": False,
+        "kill_attempts_before_reap": 0, "leader_reap_attempts": 0}
+    errors = []
+    try:
+        signal_group_unreaped(process, signal.SIGTERM)
+        state["term_sent"] = True
+    except ProcessLookupError:
+        state["term_group_absent_before_reap"] = True
+    except BaseException as error:
+        errors.append("owned TERM: " + repr(error))
+    try:
+        deadline = time.monotonic() + 1
+        while time.monotonic() < deadline:
+            observe(process)
+            pause(0.05)
+    except BaseException as error:
+        errors.append("owned grace: " + repr(error))
+    # Both independently authenticated KILL attempts precede ANY wait/reap.
+    for attempt in range(2):
+        state["kill_attempts_before_reap"] += 1
+        try:
+            signal_group_unreaped(process, signal.SIGKILL)
+            state["kill_sent_before_reap"] = True
+        except ProcessLookupError:
+            state["kill_group_absent_before_reap"] = True
+        except BaseException as error:
+            errors.append(f"owned KILL attempt{attempt + 1}: " + repr(error))
+    state["all_nonzero_signals_before_reap"] = True
+    # Popen.wait targets only this child. A failed first wait cannot skip final wait.
+    try:
+        state["leader_reap_attempts"] += 1
+        state["exit_code"] = process.wait(timeout=5)
+        state["leader_reaped"] = True
+    except BaseException as error:
+        errors.append("owned first leader reap: " + repr(error))
+    finally:
+        if process.returncode is None:
+            try:
+                state["leader_reap_attempts"] += 1
+                state["exit_code"] = process.wait(timeout=5)
+                state["leader_reaped"] = True
+            except BaseException as error:
+                errors.append("owned final leader reap: " + repr(error))
+        else:
+            state["exit_code"] = process.returncode
+            state["leader_reaped"] = True
+    try:
+        require(state["leader_reaped"], "owned leader was not reaped")
+        deadline = time.monotonic() + 5
+        while group_exists(process.pid) and time.monotonic() < deadline:
+            pause(0.05)  # group_exists uses signal0 only; never signal after reap.
+        state["group_absent"] = not group_exists(process.pid)
+        require(state["group_absent"], "owned foreground group remains after final KILL/reap")
+    except BaseException as error:
+        errors.append("owned group absence: " + repr(error))
+    state["stage_errors"] = list(errors)
+    if errors:
+        raise RuntimeError("independent owned foreground cleanup: " + "; ".join(errors))
+
+
+def cleanup(process, name, nonce, cidfile, receipt):
+    errors = []
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("initial owned container cleanup: " + repr(error))
+    try:
+        cleanup_group(process, receipt)
+    except BaseException as error:
+        errors.append("owned foreground cleanup: " + repr(error))
+    # Rediscover startup races using only the original nonce/name/CID binding.
+    try:
+        errors.extend(cleanup_container(name, nonce, cidfile))
+    except BaseException as error:
+        errors.append("final owned container cleanup: " + repr(error))
+    try:
+        require(receipt.get("foreground_cleanup", {}).get("leader_reaped"), "owned leader not reaped")
+        require(receipt["foreground_cleanup"]["group_absent"], "owned group absence not established")
+        require(owned_container(name, nonce, cidfile) is None, "owned container appeared after foreground reap")
+        receipt["owned_container_absent_after_reap"] = True
+    except BaseException as error:
+        errors.append("final owned absence: " + repr(error))
+    return errors
--- /dev/null
+++ parent-pins.json
@@ -0,0 +1,49 @@
+{
+  "accepted_capture": {
+    "bytes": 22266,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/capture_reference.py",
+    "sha256": "6ee140185766dee237ea4df802504ef7b7be80110f5566876a644414cf66049d"
+  },
+  "accepted_source_pins": {
+    "bytes": 10079,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/source-pins.json",
+    "sha256": "9bd17bb4d58582373034fd7b0097e36d3e39bf91d22bce837cb7fe109ee9fee4"
+  },
+  "accepted_validator": {
+    "bytes": 7906,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/validate_reference.py",
+    "sha256": "e29ea2a21c850cb4e3210f4206412282fb857c40344c6338c7604bf967e8fd72"
+  },
+  "attention_hook": {
+    "bytes": 9970,
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-original-attention-slice-20261020/observation-prep/attention_observation_hook.py",
+    "sha256": "12407465ee6795c2c94d191f1ff3d5550b7bef29e8d916f8f824b6b736dd9495"
+  },
+  "codec_manifest": "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634",
+  "image_id": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "legacy_metadata": [
+    {
+      "bytes": 3829,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/invocation.json",
+      "sha256": "c601fa7e495807332e57c80a177276f57f35c90430d8c82a5691f453781c8b72"
+    },
+    {
+      "bytes": 313368,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference/manifest.json",
+      "sha256": "650a09dd392206231e21c2ea92064fc9e71497b1334c2db362fd3e097a4d1e2d"
+    },
+    {
+      "bytes": 17355,
+      "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference/validation.json",
+      "sha256": "6d7d6ba1a16555c4930c922c3bcd6a6c8d957e6c231b3d9b5b6924dd54a3ee92"
+    }
+  ],
+  "legacy_reference_root": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-cpu-reference-20261020/root-probe-2/captured-reference",
+  "owned_lifetime_helper": {
+    "bytes": 11840,
+    "path": "/home/pvl/spaces/abird/src/zerogravity-worktrees/p12-hf-ordinary-rms-20261020/.agents/runs/model-execution-hf-vllm-gemm-observation-20261020/owned_container_v4.py",
+    "sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+  },
+  "raw_payloads_hashed_or_loaded_by_author": false,
+  "scope": "PREPARED_ONLY_NO_RUNTIME_BY_AUTHOR"
+}
--- /dev/null
+++ run_pair.py
@@ -0,0 +1,139 @@
+"""ROOT ONLY: serialized plain/observed original CPU capture; no GPU exposure."""
+import argparse
+import importlib.util
+import json
+import os
+from pathlib import Path
+import subprocess
+import time
+import uuid
+
+from support import HERE, read_json, require, sha, write
+
+IMAGE = "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe"
+CODEC_MANIFEST = "sha256:eec72f473eb14854e7dcc1682257a531c1cea15f56030eaab2abc5dba5fbf634"
+HELPER_SHA = "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+ENTRY_FILES = ("support.py", "source-pins.json", "parent-pins.json", "capture_reference.py",
+               "validate_reference.py", "capture_and_validate.py", "attention_observation_hook.py",
+               "owned_container_v4.py", "run_pair.py", "validate_pair.py")
+
+
+def reviewed_helper():
+    path = HERE / "owned_container_v4.py"
+    require(sha(path) == HELPER_SHA, "exact reviewed GEMM owned_container_v4 lifetime")
+    spec = importlib.util.spec_from_file_location("owned_cpu_attention", path)
+    helper = importlib.util.module_from_spec(spec)
+    spec.loader.exec_module(helper)
+    return helper
+
+
+def run_mode(helper, output, source, mode, cpus):
+    directory = output / mode
+    directory.mkdir(mode=0o700)
+    nonce = uuid.uuid4().hex
+    name = "zg-hf2b-attention-" + mode + "-" + nonce
+    cidfile = directory / "owned-container.cid"
+    environment = {
+        "ZG_PROBE_ALLOWED_CPUS": ",".join(map(str, cpus)), "ZG_ATTENTION_MODE": mode,
+        "PYTHONPATH": "/probe", "PYTHONDONTWRITEBYTECODE": "1",
+        "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
+        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_PLUGINS": "",
+        "TOKENIZERS_PARALLELISM": "false", "ATEN_CPU_CAPABILITY": "avx2",
+        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
+        "LD_PRELOAD": "", "LD_AUDIT": "",
+    }
+    argv = helper.PODMAN + ["run", "--name", name, "--label", helper.LABEL + "=" + nonce,
+             "--cidfile", str(cidfile), "--rm", "--pull=never", "--network=none", "--read-only",
+             "--memory=12g", "--memory-swap=12g", "--cpus=4", "--pids-limit=256",
+             "--cap-drop=all", "--security-opt=no-new-privileges",
+             "--ulimit", "fsize=268435456:268435456", "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g"]
+    for key, value in environment.items():
+        argv += ["-e", key + "=" + value]
+    for mount, target in ((HERE, "/probe:ro"), (source, "/source:ro"), (directory, "/output:rw")):
+        argv += ["-v", str(mount) + ":" + target]
+    argv += ["--entrypoint", "python3", IMAGE, "/probe/capture_and_validate.py"]
+    receipt = {"status": "UNQUALIFIED", "mode": mode, "argv": argv, "owned_name": name,
+               "owned_nonce": nonce, "image_id": IMAGE, "expected_cpus": cpus,
+               "entry_files": {name: sha(HERE / name) for name in ENTRY_FILES},
+               "owned_lifetime_helper_sha256": HELPER_SHA,
+               "scope": "original CPU paired observation only; no Rust/model/profile eligibility"}
+    write(directory / "invocation.json", receipt)
+    child = log = failure = None
+    started = time.monotonic()
+    try:
+        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
+                                 stderr=subprocess.STDOUT, start_new_session=True)
+        log = helper.Log(child.stdout, directory / "execution.log")
+        receipt["owned_cli_pid"] = child.pid
+        write(directory / "invocation.json", receipt)
+        receipt["exit_code"] = log.observe(child, 300)
+    except BaseException as caught:
+        failure = caught
+    finally:
+        receipt["cleanup_errors"] = helper.cleanup(child, name, nonce, cidfile, receipt) if child is not None else []
+        try:
+            if log is not None:
+                log.finish()
+        except BaseException as caught:
+            receipt["cleanup_errors"].append(repr(caught))
+        finally:
+            if log is not None:
+                receipt["log_cleanup"] = {"pipe_eof": log.eof, "all_closed": log.closed,
+                                          "total_bytes": log.total, "retained_bytes": log.retained}
+        if log is None and child is not None and child.stdout is not None:
+            try:
+                child.stdout.close()
+            except BaseException as caught:
+                receipt["cleanup_errors"].append("uninitialized log pipe close: " + repr(caught))
+        receipt["elapsed_seconds_not_benchmark"] = time.monotonic() - started
+    if failure is not None:
+        receipt["failure"] = repr(failure)
+    if failure is None and receipt.get("exit_code") == 0 and not receipt["cleanup_errors"]:
+        receipt["status"] = "CAPTURE_COMPLETED_REQUIRES_VALIDATION"
+    try:
+        write(directory / "invocation.json", receipt)
+    except BaseException as caught:
+        raise RuntimeError("final receipt I/O failed: " + repr(caught) + "; primary=" + repr(failure)
+                           + "; cleanup=" + repr(receipt["cleanup_errors"])) from caught
+    require(receipt["status"] == "CAPTURE_COMPLETED_REQUIRES_VALIDATION",
+            "owned CPU capture/cleanup failed; inspect " + str(directory))
+
+
+def main():
+    parser = argparse.ArgumentParser()
+    parser.add_argument("--output", required=True, type=Path, help="fresh directory beneath this prepared run")
+    options = parser.parse_args()
+    require(options.output.is_absolute() and options.output.parent.resolve() == HERE,
+            "fresh absolute output directly beneath this owned run")
+    frozen = read_json(HERE / "freeze-manifest.json")
+    for pin in frozen["files"]:
+        path = HERE / pin["file"]
+        require(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], "frozen preparation")
+    parents = read_json(HERE / "parent-pins.json")
+    for pin in parents["legacy_metadata"] + [parents[key] for key in
+            ("accepted_capture", "accepted_validator", "accepted_source_pins", "attention_hook", "owned_lifetime_helper")]:
+        path = Path(pin["path"])
+        require(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], "original parent evidence unchanged")
+    pins = read_json(HERE / "source-pins.json")
+    require(pins["repo"] == "Qwen/Qwen3.5-2B" and pins["revision"] == "15852e8c16360a2fea060d615a32b45270f8a8fc",
+            "exact original source")
+    source = Path(pins["source_directory"])
+    require(source.is_dir(), "already acquired original source")
+    helper = reviewed_helper()
+    inspected = helper.command(helper.PODMAN + ["image", "inspect", IMAGE], 15)
+    require(inspected.returncode == 0, "existing exact image; never pull")
+    image, = json.loads(inspected.stdout)
+    require(image["Id"] == IMAGE.removeprefix("sha256:") and image["Digest"] == CODEC_MANIFEST,
+            "exact approved codec-only image")
+    cpus = sorted(os.sched_getaffinity(0))[:4]
+    require(len(cpus) == 4, "four actually allowed CPUs")
+    options.output.mkdir(mode=0o700)
+    write(options.output / "image-inspect.json", image)
+    for mode in ("plain", "observed"):
+        run_mode(helper, options.output, source, mode, cpus)
+    from validate_pair import bounded_validate
+    print(json.dumps(bounded_validate(options.output), indent=2, sort_keys=True))
+
+
+if __name__ == "__main__":
+    main()
--- /dev/null
+++ source-pins.json
@@ -0,0 +1,226 @@
+{
+  "schema": "zerogravity.hf2b-reference-pins.v1",
+  "repo": "Qwen/Qwen3.5-2B",
+  "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
+  "snapshot": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52",
+  "source_directory": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/files",
+  "source_files": [
+    {
+      "path": "LICENSE",
+      "bytes": 11544,
+      "sha256": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "README.md",
+      "bytes": 62814,
+      "sha256": "c0e83a849c776e6fa843d011f023132d942a0f0140d903205bf1c363adad2275",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "chat_template.jinja",
+      "bytes": 7755,
+      "sha256": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "config.json",
+      "bytes": 2908,
+      "sha256": "ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "merges.txt",
+      "bytes": 3353259,
+      "sha256": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "model.safetensors-00001-of-00001.safetensors",
+      "bytes": 4548221488,
+      "sha256": "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "model.safetensors.index.json",
+      "bytes": 64460,
+      "sha256": "aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "preprocessor_config.json",
+      "bytes": 390,
+      "sha256": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "tokenizer.json",
+      "bytes": 12807982,
+      "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
+      "provenance": "independent official LFS SHA256"
+    },
+    {
+      "path": "tokenizer_config.json",
+      "bytes": 16709,
+      "sha256": "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "video_preprocessor_config.json",
+      "bytes": 385,
+      "sha256": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    },
+    {
+      "path": "vocab.json",
+      "bytes": 6722759,
+      "sha256": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
+      "provenance": "independent official Git blob SHA1 plus cache SHA256"
+    }
+  ],
+  "raw_f32_names": [
+    "model.language_model.layers.0.linear_attn.A_log",
+    "model.language_model.layers.0.linear_attn.norm.weight",
+    "model.language_model.layers.1.linear_attn.A_log",
+    "model.language_model.layers.1.linear_attn.norm.weight",
+    "model.language_model.layers.10.linear_attn.A_log",
+    "model.language_model.layers.10.linear_attn.norm.weight",
+    "model.language_model.layers.12.linear_attn.A_log",
+    "model.language_model.layers.12.linear_attn.norm.weight",
+    "model.language_model.layers.13.linear_attn.A_log",
+    "model.language_model.layers.13.linear_attn.norm.weight",
+    "model.language_model.layers.14.linear_attn.A_log",
+    "model.language_model.layers.14.linear_attn.norm.weight",
+    "model.language_model.layers.16.linear_attn.A_log",
+    "model.language_model.layers.16.linear_attn.norm.weight",
+    "model.language_model.layers.17.linear_attn.A_log",
+    "model.language_model.layers.17.linear_attn.norm.weight",
+    "model.language_model.layers.18.linear_attn.A_log",
+    "model.language_model.layers.18.linear_attn.norm.weight",
+    "model.language_model.layers.2.linear_attn.A_log",
+    "model.language_model.layers.2.linear_attn.norm.weight",
+    "model.language_model.layers.20.linear_attn.A_log",
+    "model.language_model.layers.20.linear_attn.norm.weight",
+    "model.language_model.layers.21.linear_attn.A_log",
+    "model.language_model.layers.21.linear_attn.norm.weight",
+    "model.language_model.layers.22.linear_attn.A_log",
+    "model.language_model.layers.22.linear_attn.norm.weight",
+    "model.language_model.layers.4.linear_attn.A_log",
+    "model.language_model.layers.4.linear_attn.norm.weight",
+    "model.language_model.layers.5.linear_attn.A_log",
+    "model.language_model.layers.5.linear_attn.norm.weight",
+    "model.language_model.layers.6.linear_attn.A_log",
+    "model.language_model.layers.6.linear_attn.norm.weight",
+    "model.language_model.layers.8.linear_attn.A_log",
+    "model.language_model.layers.8.linear_attn.norm.weight",
+    "model.language_model.layers.9.linear_attn.A_log",
+    "model.language_model.layers.9.linear_attn.norm.weight"
+  ],
+  "tensor_counts": {
+    "F32": 36,
+    "BF16": 596
+  },
+  "dtype_elements": {
+    "F32": 2592,
+    "BF16": 2274067232
+  },
+  "root_source_validation": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/model-execution-hf2b-source-plan-20261020/root-post-acquisition-validation.json",
+    "bytes": 5496,
+    "sha256": "57929dd60414c1f2b4df536c05f39bcbef9957221ae45ae42f7982d6d1666ccb"
+  },
+  "cache_manifest": {
+    "path": "/home/pvl/.cache/abird/zerogravity/models/snapshots/88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52/manifest.json",
+    "bytes": 2246,
+    "sha256": "88606494611dfa37253320ef82c88ca6ae6f665ac873fec28859fafd216dbf52"
+  },
+  "base_capture": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/capture_reference.py",
+    "bytes": 18717,
+    "sha256": "225d8637f927f8bc6ce99af9d9dda6a6a5cd33fc9d7d24042dda7b98297f335d"
+  },
+  "base_validator": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/validate_reference.py",
+    "bytes": 5573,
+    "sha256": "5ef1077ed6abdb37093701cb2846fc044a35f62eb9d883891876c5e98f0ced09"
+  },
+  "runtime_source_manifest": {
+    "path": "/home/pvl/spaces/abird/src/zerogravity/.agents/runs/serving-and-speculation-pinned-reference-2026-10-19/runtime-source-manifest.json",
+    "bytes": 1942,
+    "sha256": "24eabe68b10c2ac6a17e2dcdcd53805abe47556a98b8b6a4f2f1c27c60e6c673"
+  },
+  "runtime_source": {
+    "python": "3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]",
+    "versions": {
+      "transformers": "5.17.0",
+      "torch": "2.13.0+cu129",
+      "safetensors": "0.8.0",
+      "vllm": "0.31.0+cu129",
+      "flash-linear-attention": null,
+      "fla-core": null,
+      "triton": "3.7.1"
+    },
+    "files": {
+      "transformers/models/qwen3_5/modeling_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/modeling_qwen3_5.py",
+        "sha256": "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b",
+        "bytes": 95685,
+        "artifact": "pinned-runtime-source/modeling_qwen3_5.py"
+      },
+      "transformers/models/qwen3_5/configuration_qwen3_5.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/models/qwen3_5/configuration_qwen3_5.py",
+        "sha256": "19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da",
+        "bytes": 8395,
+        "artifact": "pinned-runtime-source/configuration_qwen3_5.py"
+      },
+      "transformers/cache_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/cache_utils.py",
+        "sha256": "702144bb44553f6339ea1bf23c8205a708bb5f8c7c09cb3a2db484182646743c",
+        "bytes": 104677,
+        "artifact": "pinned-runtime-source/cache_utils.py"
+      },
+      "transformers/modeling_utils.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py",
+        "sha256": "38c2bd02ed7af229f54e2f02dae65663be7af29ed2c9498bd1c460cf60fbb62d",
+        "bytes": 268806,
+        "artifact": "pinned-runtime-source/modeling_utils.py"
+      },
+      "transformers/integrations/hub_kernels.py": {
+        "runtime_path": "/usr/local/lib/python3.12/dist-packages/transformers/integrations/hub_kernels.py",
+        "sha256": "c1eafefe0cbdf7f0deaa21b1ef6cb6d7c6082b3f486651d069ff38fcf3e53c95",
+        "bytes": 44974,
+        "artifact": "pinned-runtime-source/hub_kernels.py"
+      }
+    }
+  },
+  "derived_image_receipt": {
+    "path": "/home/pvl/.cache/abird/zerogravity/external-baselines/vllm-0.31.0-cu129/torchcodec-cpu/derived-image-receipt.json",
+    "bytes": 7655,
+    "sha256": "c53c8421af4c832b901c00f4aabb65c7a1901b92c0655af188fa886fdaac63c0"
+  },
+  "derived_image_id": "sha256:f57d77f5ae7228fa5c7be8a82958d145d9382296591b73e507ed7d37fb4b0dbe",
+  "base_image": "docker.io/vllm/vllm-openai@sha256:b18abb2df97b8f798e81862bd93f872ea18613372e2c3adc0cc2ac21e66ac12f",
+  "package_records": {
+    "torch": {
+      "record_sha256": "4ae8e32005981a22c51a85130862a0ca173a01441a9de439b405a380c8dba243",
+      "version": "2.13.0+cu129"
+    },
+    "transformers": {
+      "record_sha256": "02b3d38575d5dc45992f859592fe143a86b04046a7c8da1b6ebc82390a90ed26",
+      "version": "5.17.0"
+    },
+    "safetensors": {
+      "record_sha256": "c5110881c0c39a4f163a23b2ccc88cad19ce1823b6dce1ceef353d33f16c004b",
+      "version": "0.8.0"
+    },
+    "vllm": {
+      "record_sha256": "bc16d6292493309ae02ca009dd57d2862e63880357a64a40be9dc6f0d6513895",
+      "version": "0.31.0+cu129"
+    },
+    "torchcodec": {
+      "record_sha256": "506253df6850b5d0af73d56c64c1e2d34c6f92cd3794a1ad1330050674855285",
+      "version": "0.17.0+cpu"
+    }
+  },
+  "source_execution_qualified": false
+}
--- /dev/null
+++ static-checks.json
@@ -0,0 +1,39 @@
+{
+  "accepted_named_function_changed_only_by_observer_delegation": [
+    "observe_calls"
+  ],
+  "accepted_named_functions_unchanged": [
+    "sha",
+    "write_json",
+    "verify_source",
+    "audit_chunks",
+    "finite",
+    "artifact_bound",
+    "verify_runtime",
+    "tensor_info",
+    "f32_file",
+    "parameter_audit",
+    "snapshot",
+    "remember",
+    "output_hook",
+    "input_hook",
+    "official_json_types"
+  ],
+  "author_executed_models_operators_containers_tests": false,
+  "original_model_call_AST_unchanged": true,
+  "owned_helper_sha256": "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d",
+  "python_AST_files": [
+    "attention_observation_hook.py",
+    "capture_and_validate.py",
+    "capture_reference.py",
+    "owned_container_v4.py",
+    "run_pair.py",
+    "support.py",
+    "test_validate_pair.py",
+    "validate_pair.py",
+    "validate_reference.py"
+  ],
+  "scope": "AST_AND_SOURCE_BYTES_ONLY",
+  "tests_prepared_not_executed": 10,
+  "unchanged_original_validator_and_source_pins": true
+}
--- /dev/null
+++ support.py
@@ -0,0 +1,34 @@
+"""Bounded preparation support; no model/container side effects on import."""
+import hashlib
+import json
+import os
+from pathlib import Path
+
+HERE = Path(__file__).resolve().parent
+
+
+def require(value, message):
+    if not value:
+        raise ValueError(message)
+
+
+def sha(path):
+    digest = hashlib.sha256()
+    with Path(path).open("rb") as stream:
+        for block in iter(lambda: stream.read(1024 * 1024), b""):
+            digest.update(block)
+    return digest.hexdigest()
+
+
+def write(path, value):
+    with Path(path).open("w", encoding="utf-8") as stream:
+        os.fchmod(stream.fileno(), 0o600)
+        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
+        stream.write("\n")
+
+
+def read_json(path, cap=1024 * 1024):
+    path = Path(path)
+    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= cap,
+            "bounded regular JSON file: " + str(path))
+    return json.loads(path.read_text())
--- /dev/null
+++ test_validate_pair.py
@@ -0,0 +1,136 @@
+"""SOURCE-FREE fixtures, PREPARED ONLY; ROOT may execute after review."""
+import hashlib
+import json
+from pathlib import Path
+import struct
+import tempfile
+import unittest
+
+from validate_pair import LAYERS, MODELING_SHA, STAGES, TORCH_SOURCE, archive, equal_archives, integer, observation
+
+
+def make_archive(path, values, *, gap=False):
+    header, payload = {}, b"\0\0" if gap else b""
+    for name, (dtype, shape, raw) in sorted(values.items()):
+        lo = len(payload)
+        payload += raw
+        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [lo, len(payload)]}
+    encoded = json.dumps(header, separators=(",", ":")).encode()
+    encoded += b" " * ((-len(encoded)) % 8)
+    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
+
+
+def fixture(directory, *, word=0):
+    values, metadata = {}, {}
+    total = 0
+    for layer in LAYERS:
+        for stage, shape in STAGES.items():
+            name = f"layer-{layer:02d}.{stage}"
+            elements, stride, strides = 1, 1, []
+            for width in reversed(shape):
+                strides.insert(0, stride)
+                stride *= width
+                elements *= width
+            raw = struct.pack("<H", word) * elements
+            values[name] = ("BF16", shape, raw)
+            original = list(strides)
+            if stage == "query":
+                original[-2] = 512
+            metadata[name] = {"dtype": "torch.bfloat16", "shape": shape,
+                              "original_stride": original, "saved_stride": strides,
+                              "bytes": len(raw), "source_module_layer": layer, "stage": stage}
+            total += len(raw)
+    path = directory / "attention.safetensors"
+    make_archive(path, values)
+    report = {"status": "OBSERVED_OPERATORS_ONLY_FULL_TRANSPARENCY_UNQUALIFIED",
+              "profile": "official-bf16-loader", "step": 0, "teacher_token": 11,
+              "modeling_sha256": MODELING_SHA, "torch_source": TORCH_SOURCE,
+              "no_model_or_state_or_provider_eligibility": True,
+              "tensor_count": 54, "tensors": metadata, "aggregate_payload_bytes": total,
+              "archive": {"file": path.name, "bytes": path.stat().st_size,
+                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}
+    (directory / "observation.json").write_text(json.dumps(report))
+    return report, total
+
+
+class ComparatorFixtures(unittest.TestCase):
+    def test_identical_raw_bits(self):
+        with tempfile.TemporaryDirectory() as folder:
+            a, b = Path(folder) / "a", Path(folder) / "b"
+            for path in (a, b):
+                make_archive(path, {"tensor": ("BF16", [2], b"\0\0\0\x80")})
+            self.assertEqual(equal_archives(a, b, 1), 4)
+
+    def test_signed_zero_difference_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            a, b = Path(folder) / "a", Path(folder) / "b"
+            make_archive(a, {"tensor": ("BF16", [1], b"\0\0")})
+            make_archive(b, {"tensor": ("BF16", [1], b"\0\x80")})
+            with self.assertRaisesRegex(ValueError, "exact source raw bytes"):
+                equal_archives(a, b, 1)
+
+    def test_payload_gap_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder) / "a"
+            make_archive(path, {"tensor": ("BF16", [1], b"\0\0")}, gap=True)
+            with self.assertRaisesRegex(ValueError, "without gaps"):
+                archive(path)
+
+    def test_namespace_omission_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            a, b = Path(folder) / "a", Path(folder) / "b"
+            make_archive(a, {"one": ("BF16", [1], b"\0\0")})
+            make_archive(b, {"another": ("BF16", [1], b"\0\0")})
+            with self.assertRaisesRegex(ValueError, "namespace"):
+                equal_archives(a, b, 1)
+
+    def test_complete_synthetic_stage_contract(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            _, total = fixture(path)
+            self.assertEqual(total, 172032)
+            self.assertEqual(observation(path, "official-bf16-loader", 0, 0), total)
+
+    def test_missing_stage_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            row, _ = fixture(path)
+            del row["tensors"]["layer-23.gate"]
+            (path / "observation.json").write_text(json.dumps(row))
+            with self.assertRaisesRegex(ValueError, "all six layers"):
+                observation(path, "official-bf16-loader", 0, 0)
+
+    def test_profile_and_teacher_misbinding_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            row, _ = fixture(path)
+            row["teacher_token"] = 353
+            (path / "observation.json").write_text(json.dumps(row))
+            with self.assertRaisesRegex(ValueError, "source/protocol"):
+                observation(path, "official-bf16-loader", 0, 0)
+            with self.assertRaises(ValueError):
+                observation(path, "source-36-f32-restored", 0, 0)
+
+    def test_nonfinite_raw_observation_is_refused(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            fixture(path, word=0x7fc1)
+            with self.assertRaisesRegex(ValueError, "finite actual BF16"):
+                observation(path, "official-bf16-loader", 0, 0)
+
+    def test_strided_query_role_is_required(self):
+        with tempfile.TemporaryDirectory() as folder:
+            path = Path(folder)
+            row, _ = fixture(path)
+            row["tensors"]["layer-03.query"]["original_stride"][-2] = 256
+            (path / "observation.json").write_text(json.dumps(row))
+            with self.assertRaisesRegex(ValueError, "interleaved query"):
+                observation(path, "official-bf16-loader", 0, 0)
+
+    def test_boolean_step_is_not_an_integer(self):
+        with self.assertRaises(ValueError):
+            integer(False, 0, 1, "strict teacher")
+
+
+if __name__ == "__main__":
+    unittest.main()
--- /dev/null
+++ validate_pair.py
@@ -0,0 +1,301 @@
+"""ROOT ONLY stdlib byte comparator; never imports Torch or performs model math."""
+import json
+from pathlib import Path
+import signal
+import struct
+
+from support import HERE, read_json, require, sha, write
+
+PROFILES = ("official-bf16-loader", "source-36-f32-restored")
+LAYERS = (3, 7, 11, 15, 19, 23)
+TOKENS = (11, 353)
+MODELING_SHA = "762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b"
+TORCH_SOURCE = "cf30153c4c131c8164ee7798e5022d810682e2cb"
+COUNTS = {"profiles": 2, "step_archives": 4, "canonical_domains": 168,
+          "full_vocab_logits": 4, "observed_intermediate_tensors": 60}
+STAGES = {
+    "hidden": [1, 1, 2048], "query_gate_projection": [1, 1, 4096],
+    "key_projection": [1, 1, 512], "value_projection": [1, 1, 512],
+    "query": [1, 1, 8, 256], "key": [1, 1, 2, 256],
+    "normalized_query": [1, 1, 8, 256], "normalized_key": [1, 1, 2, 256],
+    "gate": [1, 1, 2048],
+}
+
+
+def integer(value, lo, hi, description):
+    require(type(value) is int and lo <= value <= hi, description)
+    return value
+
+
+def joined(root, relative):
+    require(type(relative) is str and relative and not Path(relative).is_absolute(), "relative artifact path")
+    path = Path(root) / relative
+    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(), "contained artifact")
+    return path
+
+
+def verified(root, row, cap=32 * 1024 * 1024):
+    path = joined(root, row["file"])
+    require(path.is_file() and 0 < path.stat().st_size <= cap and sha(path) == row["sha256"],
+            "complete original artifact SHA: " + str(path))
+    if "bytes" in row:
+        require(path.stat().st_size == row["bytes"], "recorded artifact length")
+    return path
+
+
+def archive(path):
+    path = Path(path)
+    require(path.is_file() and not path.is_symlink() and 8 < path.stat().st_size <= 32 * 1024 * 1024,
+            "bounded complete Safetensors")
+    with path.open("rb") as stream:
+        header_bytes, = struct.unpack("<Q", stream.read(8))
+        require(1 < header_bytes <= 1024 * 1024 and header_bytes + 8 <= path.stat().st_size,
+                "bounded actual Safetensors header")
+        header = json.loads(stream.read(header_bytes))
+    require(type(header) is dict, "actual tensor namespace")
+    base = header_bytes + 8
+    cursor = 0
+    spans, tensors = [], {}
+    for name, row in header.items():
+        if name == "__metadata__":
+            require(type(row) is dict and all(type(k) is str and type(v) is str for k, v in row.items()),
+                    "bounded string archive metadata")
+            continue
+        require(type(name) is str and 0 < len(name) <= 256 and type(row) is dict,
+                "actual named tensor descriptor")
+        kind, shape, offsets = row["dtype"], row["shape"], row["data_offsets"]
+        require(kind in ("BF16", "F32") and type(shape) is list and 0 < len(shape) <= 8
+                and type(offsets) is list and len(offsets) == 2, "closed raw tensor dtype/shape")
+        elements = 1
+        for width in shape:
+            elements *= integer(width, 1, 248320, "positive raw shape")
+            require(elements <= 8 * 1024 * 1024, "bounded raw tensor words")
+        lo, hi = [integer(value, 0, 32 * 1024 * 1024, "bounded tensor offset") for value in offsets]
+        require(hi - lo == elements * (2 if kind == "BF16" else 4), "exact dtype/shape byte geometry")
+        spans.append((lo, hi)); tensors[name] = {"dtype": kind, "shape": shape, "offset": base + lo, "bytes": hi - lo}
+    for lo, hi in sorted(spans):
+        require(lo == cursor, "complete nonoverlapping raw payload without gaps")
+        cursor = hi
+    require(base + cursor == path.stat().st_size and 0 < len(tensors) <= 256, "complete archive EOF/namespace")
+    return tensors
+
+
+def equal_range(left, lo, right, ro, count):
+    require(type(count) is int and count > 0, "nonvacuous byte comparison")
+    with Path(left).open("rb") as a, Path(right).open("rb") as b:
+        a.seek(lo); b.seek(ro)
+        while count:
+            width = min(count, 1024 * 1024)
+            av, bv = a.read(width), b.read(width)
+            require(len(av) == len(bv) == width and av == bv,
+                    "exact source raw bytes, including signed zero: " + str(left))
+            count -= width
+
+
+def equal_archives(left, right, count):
+    a, b = archive(left), archive(right)
+    require(set(a) == set(b) and len(a) == count, "complete equal raw tensor namespace")
+    for name in sorted(a):
+        av, bv = a[name], b[name]
+        require({key: av[key] for key in ("dtype", "shape", "bytes")}
+                == {key: bv[key] for key in ("dtype", "shape", "bytes")}, "same original tensor geometry")
+        equal_range(left, av["offset"], right, bv["offset"], av["bytes"])
+    return sum(value["bytes"] for value in a.values())
+
+
+def lifecycle(row):
+    group = row["foreground_cleanup"]
+    require(row["status"] == "CAPTURE_COMPLETED_REQUIRES_VALIDATION" and row["exit_code"] == 0
+            and row["cleanup_errors"] == [] and row["owned_container_absent_after_reap"] is True
+            and row["owned_lifetime_helper_sha256"] == "53ee4270b6ba663b1acd4d64476fa7903937533510b45147a3b8c1038b16344d"
+            and group["all_nonzero_signals_before_reap"] is True
+            and group["kill_attempts_before_reap"] == 2 and group["stage_errors"] == []
+            and group["leader_reaped"] is True and group["group_absent"] is True
+            and group["exit_code"] == 0 and group["pgid"] == row["owned_cli_pid"]
+            and group["term_sent"] is True and group["kill_sent_before_reap"] is True
+            and row["log_cleanup"]["pipe_eof"] is True and row["log_cleanup"]["all_closed"] is True,
+            "complete exact reviewed owned lifetime")
+    for name, digest in row["entry_files"].items():
+        require(Path(name).name == name and sha(HERE / name) == digest, "original invocation source unchanged")
+
+
+def protocol(manifest, source_pins):
+    require(manifest["source_repo"] == source_pins["repo"]
+            and manifest["source_revision"] == source_pins["revision"]
+            and manifest["source_pins_sha256"] == sha(HERE / "source-pins.json")
+            and manifest["source_reverified_after_capture"] is True
+            and set(manifest["profiles"]) == set(PROFILES), "same complete source/profiles")
+    require(manifest["runtime"]["torch_num_threads"] == 4
+            and manifest["runtime"]["torch_interop_threads"] == 1
+            and manifest["runtime"]["cpu_capability"] == "AVX2"
+            and manifest["runtime"]["cuda_available"] is False, "exact accepted CPU runtime mode")
+    require(manifest["loader"]["attn_implementation"] == "eager"
+            and manifest["loader"]["use_kernels"] is False
+            and manifest["loader"]["dtype"] == "torch.bfloat16", "unchanged original official loader")
+    cfg = manifest["config"]["text_config"]
+    require(cfg["hidden_size"] == 2048 and cfg["num_attention_heads"] == 8
+            and cfg["num_key_value_heads"] == 2 and cfg["head_dim"] == 256
+            and [index for index, kind in enumerate(cfg["layer_types"]) if kind == "full_attention"] == list(LAYERS),
+            "all exact original full-attention layers")
+
+
+def observation(directory, profile, step, cumulative):
+    report = read_json(directory / "observation.json")
+    require(report["status"] == "OBSERVED_OPERATORS_ONLY_FULL_TRANSPARENCY_UNQUALIFIED"
+            and report["profile"] == profile
+            and integer(report["step"], 0, 1, "teacher step") == step
+            and integer(report["teacher_token"], 0, 248319, "teacher token") == TOKENS[step]
+            and report["modeling_sha256"] == MODELING_SHA and report["torch_source"] == TORCH_SOURCE
+            and report["no_model_or_state_or_provider_eligibility"] is True, "genuine original hook source/protocol")
+    require(report["archive"]["file"] == "attention.safetensors", "one bounded original archive")
+    path = verified(directory, report["archive"], 1024 * 1024)
+    actual = archive(path)
+    expected = {f"layer-{layer:02d}.{stage}" for layer in LAYERS for stage in STAGES}
+    require(set(actual) == set(report["tensors"]) == expected and report["tensor_count"] == 54,
+            "all six layers/nine stages exactly once")
+    total = 0
+    with path.open("rb") as stream:
+        for layer in LAYERS:
+            for stage, shape in STAGES.items():
+                name = f"layer-{layer:02d}.{stage}"
+                descriptor, meta = actual[name], report["tensors"][name]
+                require(descriptor["dtype"] == "BF16" and descriptor["shape"] == shape
+                        and meta["dtype"] == "torch.bfloat16" and meta["shape"] == shape
+                        and meta["bytes"] == descriptor["bytes"]
+                        and meta["source_module_layer"] == layer and meta["stage"] == stage,
+                        "exact source layer/stage/dtype/full geometry")
+                require(len(meta["original_stride"]) == len(meta["saved_stride"]) == len(shape)
+                        and all(type(value) is int and value > 0 for value in meta["original_stride"]),
+                        "actual nonnegative source tensor strides")
+                stride = 1
+                for width, saved in zip(reversed(shape), reversed(meta["saved_stride"])):
+                    require(type(saved) is int and saved > 0 and (width == 1 or saved == stride),
+                            "contiguous saved raw transport; singleton strides carry no payload")
+                    stride *= width
+                if stage == "query":
+                    require(meta["original_stride"][-2:] == [512, 1], "actual interleaved query/gate source view")
+                stream.seek(descriptor["offset"])
+                payload = stream.read(descriptor["bytes"])
+                require(len(payload) == descriptor["bytes"] and all((word & 0x7f80) != 0x7f80
+                        for word, in struct.iter_unpack("<H", payload)), "complete finite actual BF16 bits")
+                total += descriptor["bytes"]
+    require(report["aggregate_payload_bytes"] == cumulative + total <= 64 * 1024 * 1024,
+            "exact cumulative hook payload cap")
+    return total
+
+
+def validate(output):
+    output = Path(output)
+    parents = read_json(HERE / "parent-pins.json")
+    for pin in parents["legacy_metadata"]:
+        path = Path(pin["path"])
+        require(path.stat().st_size == pin["bytes"] and sha(path) == pin["sha256"], "immutable accepted parent evidence")
+    legacy = Path(parents["legacy_reference_root"])
+    source_pins = read_json(HERE / "source-pins.json")
+    baseline = read_json(legacy / "manifest.json")
+    accepted = read_json(legacy / "validation.json")
+    require(accepted["archive_validation"] == "PASS" and accepted["counts"] == COUNTS,
+            "actual independent accepted 2B two-profile reference")
+    protocol(baseline, source_pins)
+    totals = {}
+    for mode in ("plain", "observed"):
+        directory = output / mode
+        lifetime = read_json(directory / "invocation.json")
+        lifecycle(lifetime)
+        require(lifetime["mode"] == mode, "actual paired mode")
+        mode_report = read_json(directory / "attention-mode.json")
+        require(mode_report["mode"] == mode and mode_report["owned_hooks_removed"] is True
+                and mode_report["profiler_retired"] is True
+                and mode_report["teacher_tokens"] == list(TOKENS) and mode_report["profiles"] == list(PROFILES)
+                and mode_report["cpu_affinity"] == lifetime["expected_cpus"]
+                and len(set(mode_report["cpu_affinity"])) == 4
+                and mode_report["observer_hook_sha256"] == sha(HERE / "attention_observation_hook.py")
+                and mode_report["accepted_capture_sha256"] == parents["accepted_capture"]["sha256"]
+                and mode_report["expected_attention_tensor_count"] == (216 if mode == "observed" else 0),
+                "all actual hook/profiler owners retired and exact CPU affinity")
+        root = directory / "captured-reference"
+        validation = read_json(root / "validation.json")
+        require(validation["archive_validation"] == "PASS" and validation["counts"] == COUNTS,
+                "unchanged complete archive/layout validator passed")
+        manifest = read_json(root / "manifest.json")
+        protocol(manifest, source_pins)
+        require(manifest["capture_script_sha256"] == sha(HERE / "capture_reference.py"), "exact actual integration source")
+        for key in ("runtime_source", "package_records", "loader", "source_geometry", "tokenizer", "head", "config"):
+            require(manifest[key] == baseline[key], "unchanged original source/runtime/loader metadata: " + key)
+        raw_count = domains = intermediate_count = full_logits = attention_count = attention_bytes = compared_bytes = 0
+        for profile in PROFILES:
+            row, old = manifest["profiles"][profile], baseline["profiles"][profile]
+            require(len(row["steps"]) == len(old["steps"]) == 2, "both original teacher steps")
+            for key in ("actual_checkpoint_parameter_dtype_counts", "source_parameter_count", "source_f32_parameter_names", "restoration"):
+                require(row[key] == old[key], "unchanged profile-specific source dtype/audit")
+            audit = root / profile / "parameter-audit.json"
+            old_audit = legacy / profile / "parameter-audit.json"
+            require(sha(audit) == row["parameter_audit_sha256"] == old["parameter_audit_sha256"] == sha(old_audit),
+                    "all 632 source/617 loaded parameter audits exactly unchanged")
+            for step, (current, prior) in enumerate(zip(row["steps"], old["steps"])):
+                require(current["profile"] == prior["profile"] == profile
+                        and integer(current["step"], 0, 1, "strict actual teacher step") == prior["step"] == step
+                        and current["input_token"] == prior["input_token"] == TOKENS[step]
+                        and current["consumed_tokens"] == prior["consumed_tokens"] == list(TOKENS[:step + 1])
+                        and current["observed_official_operator_calls"] == prior["observed_official_operator_calls"],
+                        "same actual teacher and slow-GDN/conv/eager-attention call scope")
+                compared_bytes += equal_archives(verified(root, current["raw_archive"]),
+                                                  verified(legacy, prior["raw_archive"]), 49)
+                raw_count += 49; full_logits += 1
+                compared_bytes += equal_archives(verified(root, current["observed_intermediates"]),
+                                                  verified(legacy, prior["observed_intermediates"]), 15)
+                intermediate_count += 15
+                require(len(current["semantic_domains"]) == len(prior["semantic_domains"]) == 42,
+                        "all canonical domains retained")
+                actual_domains = {(value["layer"], value["kind"]): value for value in current["semantic_domains"]}
+                old_domains = {(value["layer"], value["kind"]): value for value in prior["semantic_domains"]}
+                require(len(actual_domains) == len(old_domains) == 42 and set(actual_domains) == set(old_domains),
+                        "every canonical layer/domain identity unique and complete")
+                for key, value in actual_domains.items():
+                    old_value = old_domains[key]
+                    a, b = verified(root, value), verified(legacy, old_value)
+                    require(a.stat().st_size == b.stat().st_size, "canonical domain byte geometry")
+                    equal_range(a, 0, b, 0, a.stat().st_size)
+                    domains += 1
+                attention_dir = root / profile / f"step-{step}" / "attention-observation"
+                if mode == "observed":
+                    attention_bytes += observation(attention_dir, profile, step, attention_bytes)
+                    attention_count += 54
+                else:
+                    require(not attention_dir.exists(), "plain run contains no new observation")
+        require((raw_count, domains, intermediate_count, full_logits, attention_count)
+                == (196, 168, 60, 4, 216 if mode == "observed" else 0), "nonvacuous complete paired evidence")
+        totals[mode] = {"raw_tensors_bit_exact_to_accepted": raw_count, "canonical_domains_bit_exact": domains,
+                        "prior_intermediate_tensors_bit_exact": intermediate_count, "full_vocab_logits_bit_exact": full_logits,
+                        "new_attention_tensors": attention_count, "attention_payload_bytes": attention_bytes,
+                        "raw_and_prior_intermediate_compared_bytes": compared_bytes,
+                        "manifest_sha256": sha(root / "manifest.json")}
+    result = {"status": "PASS_CPU_HF2B_PAIRED_ATTENTION_OBSERVATION_ONLY", "counts": totals,
+              "source_repo": source_pins["repo"], "source_revision": source_pins["revision"],
+              "profiles": list(PROFILES), "teacher_tokens": list(TOKENS), "stage_scope": list(STAGES),
+              "Rust_or_GPU_or_full_HF_profile_qualified": False, "matched_performance_qualified": False,
+              "scope": "fresh original CPU observer transparency to prior raw tensors plus genuine source-stage capture only"}
+    write(output / "paired-validation.json", result)
+    return result
+
+
+def bounded_validate(output):
+    def expired(_number, _frame):
+        raise TimeoutError("bounded paired byte validation exceeded120s")
+
+    require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0),
+            "standalone validator requires no inherited alarm")
+    previous = signal.signal(signal.SIGALRM, expired)
+    signal.alarm(120)
+    try:
+        return validate(output)
+    finally:
+        signal.alarm(0)
+        signal.signal(signal.SIGALRM, previous)
+
+
+if __name__ == "__main__":
+    import argparse
+    parser = argparse.ArgumentParser()
+    parser.add_argument("directory", type=Path)
+    print(json.dumps(bounded_validate(parser.parse_args().directory), indent=2, sort_keys=True))
--- /dev/null
+++ validate_reference.py
@@ -0,0 +1,135 @@
+"""Validate archive closure and lossless layout transforms; no model execution."""
+
+import collections
+import hashlib
+import json
+import pathlib
+
+import numpy as np
+import torch
+from safetensors.torch import load_file
+
+ROOT = pathlib.Path("/output/captured-reference")
+manifest = json.loads((ROOT / "manifest.json").read_text())
+runtime_source = manifest["runtime_source"]
+pins = json.loads(pathlib.Path("/probe/source-pins.json").read_text())
+assert manifest["source_pins_sha256"] == hashlib.sha256(pathlib.Path("/probe/source-pins.json").read_bytes()).hexdigest()
+for pin in runtime_source["files"].values():
+    for path in [pathlib.Path(pin["runtime_path"])]:
+        assert hashlib.sha256(path.read_bytes()).hexdigest() == pin["sha256"], path
+assert manifest["official_modeling_source"]["sha256"] == runtime_source["files"]["transformers/models/qwen3_5/modeling_qwen3_5.py"]["sha256"]
+assert manifest["source_reverified_after_capture"]
+assert manifest["limits"]["container_memory_max"] == str(12 * 1024**3)
+assert manifest["limits"]["container_cpu_max"] == "400000 100000"
+assert not manifest["runtime"]["cuda_available"]
+assert manifest["runtime"]["torch_num_threads"] == 4
+profiles = ["official-bf16-loader", "source-36-f32-restored"]
+counts = {"profiles": 0, "step_archives": 0, "canonical_domains": 0, "full_vocab_logits": 0, "observed_intermediate_tensors": 0}
+
+
+def verified_file(row):
+    path = ROOT / row["file"]
+    assert path.resolve().is_relative_to(ROOT) and not path.is_symlink() and path.is_file()
+    assert path.stat().st_size <= 32 * 1024**2
+    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], path
+    return path
+
+
+def verify_f32(row, expected):
+    actual = np.fromfile(verified_file(row), dtype="<f4")
+    expected = expected.detach().cpu().float().contiguous().numpy().reshape(-1)
+    # Compare bits, including zero sign; array_equal would hide that distinction.
+    assert np.array_equal(actual.view("<u4"), expected.view("<u4"))
+    assert actual.size == row["numel"] and np.isfinite(actual).all()
+
+
+for profile in profiles:
+    data = manifest["profiles"][profile]
+    audit_path = ROOT / profile / "parameter-audit.json"
+    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == data["parameter_audit_sha256"]
+    audit = json.loads(audit_path.read_text())
+    assert len(audit) == 632
+    assert all(row["exact_after_declared_dtype_cast"] for row in audit if row["loaded"] is not None)
+    excluded = [row for row in audit if row["loaded"] is None]
+    assert len(excluded) == 15 and all(row["name"].startswith("mtp.") for row in excluded)
+    original_f32 = [row for row in audit if row["source"]["dtype"] == "torch.float32"]
+    assert len(original_f32) == 36
+    assert [row["name"] for row in original_f32] == pins["raw_f32_names"]
+    assert dict(collections.Counter(row["loaded"]["dtype"] for row in audit if row["loaded"] is not None)) == data["actual_checkpoint_parameter_dtype_counts"]
+    assert len(data["steps"]) == 2
+    if profile == "source-36-f32-restored":
+        assert all(row["loaded"]["dtype"] == "torch.float32" for row in original_f32)
+    counts["profiles"] += 1
+    for step, saved in enumerate(data["steps"]):
+        raw = load_file(str(verified_file(saved["raw_archive"])), device="cpu")
+        assert len(raw) == 49
+        assert raw["logits"].shape == (1, 1, 248320)
+        assert raw["logits"].dtype == torch.bfloat16
+        assert saved["consumed_tokens"] == [11, 353][:step+1] and saved["step"] == step
+        assert saved["semantic_domain_count"] == 42 and len(saved["raw_tensor_layouts"]) == 49
+        for layer in range(24):
+            prefix = f"layer-{layer:02d}"
+            if layer % 4 == 3:
+                for key in ["keys", "values"]:
+                    assert raw[prefix + "." + key].shape == (1, 2, step+1, 256)
+                    assert raw[prefix + "." + key].dtype == torch.bfloat16
+            else:
+                assert raw[prefix + ".conv"].shape == (1, 6144, 4)
+                assert raw[prefix + ".conv"].dtype == torch.bfloat16
+                assert raw[prefix + ".recurrent"].shape == (1, 16, 128, 128)
+                assert raw[prefix + ".recurrent"].dtype == torch.float32
+        assert len(saved["semantic_domains"]) == 42
+        observed = load_file(str(verified_file(saved["observed_intermediates"])), device="cpu")
+        assert len(observed) == 15
+        assert torch.equal(observed["final_logits"].contiguous().view(torch.uint8), raw["logits"].contiguous().view(torch.uint8))
+        assert torch.equal(observed["layer-00.gdn_recurrent_fp32"].contiguous().view(torch.uint8), raw["layer-00.recurrent"].permute(0, 1, 3, 2).contiguous().view(torch.uint8))
+        assert all(bool(torch.isfinite(value).all()) for value in observed.values())
+        counts["observed_intermediate_tensors"] += len(observed)
+        verify_f32(saved["logits_f32"], raw["logits"])
+        counts["full_vocab_logits"] += 1
+        expected_domains = {(layer, kind) for layer in range(24) for kind in
+                            (["attention-kv-pair"] if layer % 4 == 3 else
+                             ["gdn-convolution-history", "gdn-recurrent"])}
+        seen = set()
+        for domain in saved["semantic_domains"]:
+            key = (domain["layer"], domain["kind"])
+            assert key in expected_domains and key not in seen
+            seen.add(key)
+            prefix = f"layer-{domain['layer']:02d}"
+            if domain["kind"] == "gdn-convolution-history":
+                expected = raw[prefix + ".conv"][..., -3:]
+            elif domain["kind"] == "gdn-recurrent":
+                expected = raw[prefix + ".recurrent"].permute(0, 1, 3, 2)
+            else:
+                key = raw[prefix + ".keys"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                val = raw[prefix + ".values"].permute(0, 2, 1, 3).contiguous().reshape(-1)
+                expected = torch.cat([key, val])
+            verify_f32(domain, expected)
+            counts["canonical_domains"] += 1
+        assert seen == expected_domains
+        assert all(bool(torch.isfinite(value).all()) for value in raw.values())
+        counts["step_archives"] += 1
+
+differences = []
+for step in range(2):
+    left_data = manifest["profiles"][profiles[0]]["steps"][step]
+    right_data = manifest["profiles"][profiles[1]]["steps"][step]
+    left = load_file(str(ROOT / left_data["raw_archive"]["file"]))
+    right = load_file(str(ROOT / right_data["raw_archive"]["file"]))
+    rows = []
+    for name in sorted(left):
+        a, b = left[name].float().contiguous().numpy(), right[name].float().contiguous().numpy()
+        changed = int(np.count_nonzero(a.view("<u4") != b.view("<u4")))
+        rows.append({"tensor": name, "elements": int(a.size), "bit_different_elements": changed, "max_abs_delta": float(np.abs(a - b).max())})
+    differences.append({"step": step, "tensors": rows})
+result = {"archive_validation": "PASS", "counts": counts, "runtime_source_files_verified": len(runtime_source["files"]), "model_or_math_parity_qualified": False, "source_revision": pins["revision"], "cross_profile_differences": differences, "scope": "archive closure, full finite tensors and exact layout derivations only; numerical profile differences are reported without a tolerance or cross-engine acceptance claim"}
+total_artifacts = 0
+for path in ROOT.rglob("*"):
+    assert not path.is_symlink()
+    if path.is_file():
+        total_artifacts += path.stat().st_size
+assert total_artifacts <= 256 * 1024**2
+assert 0 < int(manifest["cgroup_memory_peak_bytes"]) <= 12 * 1024**3
+assert counts == {"profiles": 2, "step_archives": 4, "canonical_domains": 168, "full_vocab_logits": 4, "observed_intermediate_tensors": 60}
+(ROOT / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
+print(json.dumps({"archive_validation": "PASS", "counts": counts}, sort_keys=True))

````

